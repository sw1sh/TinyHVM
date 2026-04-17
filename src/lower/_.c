// lower/_.c — private lowering IC arena for structural KERNEL dispatch

// Forward declarations (defined in debug/graph.c, included after schedule/_.c)
static int thvm_step_graph_is_active(void);
static u32 thvm_step_graph_current_index(void);
static const char *thvm_step_graph_dir(void);
static const char *thvm_step_graph_last_name(void);
static const char *thvm_step_graph_lower_anchor_name(void);
static u32 thvm_step_graph_lower_anchor_index(void);
#ifdef __OBJC__
static char *thvm_metal_render_uop_kernel_source(const UOpKernel *k);
#endif

typedef enum {
    LOP_SINK = 1,
    LOP_GID,
    LOP_CONST_U,
    LOP_CONST_F,
    LOP_RANGE,
    LOP_ENDRANGE,
    LOP_LOAD,
    LOP_STORE,
    LOP_ALU,
    LOP_ACC_INIT,
    LOP_ACC,
    LOP_INDEX,
    LOP_MOD,
    LOP_DIV,
    LOP_MASK,
} LowerOp;

typedef struct {
    Term index;
    Term mask;
} LowerLeafAccess;

typedef struct {
    UOpKernel *uk;
    u32       *memo;
    u8        *memo_valid;
    int        failed;
} LowerEmitCtx;

static inline const char *lower_op_name(u32 lop) {
    switch (lop) {
        case LOP_SINK: return "LSINK";
        case LOP_GID: return "LGID";
        case LOP_CONST_U: return "LCONST_U";
        case LOP_CONST_F: return "LCONST_F";
        case LOP_RANGE: return "LRANGE";
        case LOP_ENDRANGE: return "LENDRANGE";
        case LOP_LOAD: return "LLOAD";
        case LOP_STORE: return "LSTORE";
        case LOP_ALU: return "LALU";
        case LOP_ACC_INIT: return "LACC_INIT";
        case LOP_ACC: return "LACC";
        case LOP_INDEX: return "LINDEX";
        case LOP_MOD: return "LMOD";
        case LOP_DIV: return "LDIV";
        case LOP_MASK: return "LMASK";
        default: return "LOP?";
    }
}

static inline void lower_ctx_init_heap(TinyHVM *ctx) {
    LowerCtx *lc = &ctx->lower_ctx;
    if (lc->heap) return;
    lc->heap_cap = 4096;
    lc->heap = (Term *)calloc((size_t)lc->heap_cap, sizeof(Term));
    lc->heap_pos = 1;
    lc->heap[0] = term_era();
    lc->root = term_era();
}

static inline void lower_ctx_reset(TinyHVM *ctx) {
    lower_ctx_init_heap(ctx);
    ctx->lower_ctx.heap_pos = 1;
    ctx->lower_ctx.heap[0] = term_era();
    ctx->lower_ctx.root = term_era();
    ctx->lower_ctx.rewrite_count = 0;
    ctx->lower_ctx.normalized_sig = 0;
}

static inline u64 lower_heap_alloc(TinyHVM *ctx, u32 n) {
    LowerCtx *lc = &ctx->lower_ctx;
    if (n == 0) return 0;
    lower_ctx_init_heap(ctx);
    if (lc->heap_pos + n >= lc->heap_cap) {
        u64 need = lc->heap_pos + n + 1;
        u64 cap = lc->heap_cap ? lc->heap_cap : 4096;
        while (cap < need) cap <<= 1;
        lc->heap = (Term *)realloc(lc->heap, (size_t)cap * sizeof(Term));
        memset(lc->heap + lc->heap_cap, 0, (size_t)(cap - lc->heap_cap) * sizeof(Term));
        lc->heap_cap = cap;
    }
    u64 loc = lc->heap_pos;
    lc->heap_pos += n;
    return loc;
}

static inline Term lower_heap_read(TinyHVM *ctx, u64 loc) {
    return ctx->lower_ctx.heap[loc];
}

static inline void lower_heap_set(TinyHVM *ctx, u64 loc, Term t) {
    ctx->lower_ctx.heap[loc] = t;
}

static inline u32 lower_term_arity(Term t) {
    if (term_tag(t) != TAG_TOP) return 0;
    switch (term_ext(t)) {
        case LOP_SINK: return 1;
        case LOP_GID: return 0;
        case LOP_CONST_U: return 0;
        case LOP_CONST_F: return 0;
        case LOP_RANGE: return 1;
        case LOP_ENDRANGE: return 2;
        case LOP_LOAD: return 2;
        case LOP_STORE: return 2;
        case LOP_ALU: return 4;
        case LOP_ACC_INIT: return 1;
        case LOP_ACC: return 3;
        case LOP_INDEX: return 3;
        case LOP_MOD: return 2;
        case LOP_DIV: return 2;
        case LOP_MASK: return 2;
        default: return 0;
    }
}

static inline Term lower_make1(TinyHVM *ctx, u32 lop, Term a) {
    u64 loc = lower_heap_alloc(ctx, 1);
    lower_heap_set(ctx, loc + 0, a);
    return term_new(TAG_TOP, lop, loc);
}

static inline Term lower_make2(TinyHVM *ctx, u32 lop, Term a, Term b) {
    u64 loc = lower_heap_alloc(ctx, 2);
    lower_heap_set(ctx, loc + 0, a);
    lower_heap_set(ctx, loc + 1, b);
    return term_new(TAG_TOP, lop, loc);
}

static inline Term lower_make3(TinyHVM *ctx, u32 lop, Term a, Term b, Term c) {
    u64 loc = lower_heap_alloc(ctx, 3);
    lower_heap_set(ctx, loc + 0, a);
    lower_heap_set(ctx, loc + 1, b);
    lower_heap_set(ctx, loc + 2, c);
    return term_new(TAG_TOP, lop, loc);
}

static inline Term lower_make4(TinyHVM *ctx, u32 lop, Term a, Term b, Term c, Term d) {
    u64 loc = lower_heap_alloc(ctx, 4);
    lower_heap_set(ctx, loc + 0, a);
    lower_heap_set(ctx, loc + 1, b);
    lower_heap_set(ctx, loc + 2, c);
    lower_heap_set(ctx, loc + 3, d);
    return term_new(TAG_TOP, lop, loc);
}

static inline Term lower_const_u(u32 v) {
    return term_new(TAG_TOP, LOP_CONST_U, (u64)v);
}

static inline Term lower_const_f(f32 v) {
    u32 bits = 0;
    memcpy(&bits, &v, sizeof(bits));
    return term_new(TAG_TOP, LOP_CONST_F, (u64)bits);
}

static inline Term lower_gid(u32 axis) {
    return term_new(TAG_TOP, LOP_GID, (u64)axis);
}

static inline int lower_is_const_u(Term t, u32 *out) {
    if (term_tag(t) != TAG_TOP || term_ext(t) != LOP_CONST_U) return 0;
    if (out) *out = (u32)term_val(t);
    return 1;
}

static inline int lower_is_const_f(Term t, f32 *out) {
    if (term_tag(t) != TAG_TOP || term_ext(t) != LOP_CONST_F) return 0;
    u32 bits = (u32)term_val(t);
    if (out) memcpy(out, &bits, sizeof(bits));
    return 1;
}

static inline int lower_is_const_num(Term t, f32 *out) {
    u32 u = 0;
    f32 f = 0.0f;
    if (lower_is_const_u(t, &u)) {
        if (out) *out = (f32)u;
        return 1;
    }
    if (lower_is_const_f(t, &f)) {
        if (out) *out = f;
        return 1;
    }
    return 0;
}

static inline int lower_is_zero(Term t) {
    u32 u = 0;
    f32 f = 0.0f;
    return (lower_is_const_u(t, &u) && u == 0) ||
           (lower_is_const_f(t, &f) && f == 0.0f);
}

static inline int lower_is_one(Term t) {
    u32 u = 0;
    f32 f = 0.0f;
    return (lower_is_const_u(t, &u) && u == 1) ||
           (lower_is_const_f(t, &f) && f == 1.0f);
}

static inline f32 lower_eval_alu(u32 uop, u32 aux, f32 a, f32 b) {
    switch (uop) {
        case UOP_ADD:  return a + b;
        case UOP_SUB:  return a - b;
        case UOP_MUL:  return a * b;
        case UOP_DIV:  return b != 0.0f ? a / b : 0.0f;
        case UOP_MAX:  return a > b ? a : b;
        case UOP_CMP:  return a > b ? 1.0f : 0.0f;
        case UOP_NEG:  return -a;
        case UOP_RELU: return a > 0.0f ? a : 0.0f;
        case UOP_EXP:  return expf(a);
        case UOP_LOG:  return logf(a);
        case UOP_SQRT: return sqrtf(a);
        case UOP_CAST: return dtype_cast_from_f32(a, aux);
        default:       return a;
    }
}

static Term lower_normalize_term(TinyHVM *ctx, Term t);

static Term lower_index(TinyHVM *ctx, Term a, Term b, u32 scale) {
    return lower_make3(ctx, LOP_INDEX, a, b, lower_const_u(scale));
}

static Term lower_mod(TinyHVM *ctx, Term a, u32 mod) {
    return lower_make2(ctx, LOP_MOD, a, lower_const_u(mod));
}

static Term lower_div_term(TinyHVM *ctx, Term a, u32 div) {
    return lower_make2(ctx, LOP_DIV, a, lower_const_u(div));
}

static Term lower_load(TinyHVM *ctx, Term index, u32 leaf_buf_idx) {
    return lower_make2(ctx, LOP_LOAD, index, lower_const_u(leaf_buf_idx + 1));
}

static Term lower_store(TinyHVM *ctx, Term index, Term value) {
    return lower_make2(ctx, LOP_STORE, index, value);
}

static Term lower_alu(TinyHVM *ctx, Term a, Term b, u32 uop, u32 aux) {
    return lower_make4(ctx, LOP_ALU, a, b, lower_const_u(uop), lower_const_u(aux));
}

static Term lower_acc_init(TinyHVM *ctx, f32 init) {
    return lower_make1(ctx, LOP_ACC_INIT, lower_const_f(init));
}

static Term lower_acc(TinyHVM *ctx, Term acc, Term value, u32 reduce_uop) {
    return lower_make3(ctx, LOP_ACC, acc, value, lower_const_u(reduce_uop));
}

static Term lower_range(TinyHVM *ctx, u32 trip_count) {
    return lower_make1(ctx, LOP_RANGE, lower_const_u(trip_count));
}

static Term lower_endrange(TinyHVM *ctx, Term range, Term body) {
    return lower_make2(ctx, LOP_ENDRANGE, range, body);
}

static Term lower_mask(TinyHVM *ctx, Term cond, Term value) {
    return lower_make2(ctx, LOP_MASK, cond, value);
}

static Term lower_sink(TinyHVM *ctx, Term value) {
    return lower_make1(ctx, LOP_SINK, value);
}

static Term lower_cmp_gt(TinyHVM *ctx, Term a, Term b) {
    return lower_alu(ctx, a, b, UOP_CMP, 0);
}

static Term lower_add(TinyHVM *ctx, Term a, Term b) {
    return lower_alu(ctx, a, b, UOP_ADD, 0);
}

static Term lower_mul(TinyHVM *ctx, Term a, Term b) {
    return lower_alu(ctx, a, b, UOP_MUL, 0);
}

static Term lower_bool_and(TinyHVM *ctx, Term a, Term b) {
    if (lower_is_one(a)) return b;
    if (lower_is_one(b)) return a;
    if (lower_is_zero(a) || lower_is_zero(b)) return lower_const_f(0.0f);
    return lower_mul(ctx, a, b);
}

static Term lower_masked_load(TinyHVM *ctx, Term index, Term mask, u32 leaf_idx) {
    Term val = lower_load(ctx, index, leaf_idx);
    if (lower_is_one(mask)) return val;
    return lower_mask(ctx, mask, val);
}

static void lower_strip_view_mask(View *dst, const View *src) {
    *dst = *src;
    dst->has_mask = 0;
    dst->n_compound_masks = 0;
    memset(dst->mask_begin, 0, sizeof(dst->mask_begin));
    memset(dst->mask_end, 0, sizeof(dst->mask_end));
    memset(dst->compound_masks, 0, sizeof(dst->compound_masks));
}

static Term lower_normalize_term(TinyHVM *ctx, Term t) {
    if (term_tag(t) != TAG_TOP) return t;
    u32 lop = term_ext(t);
    u32 arity = lower_term_arity(t);
    u64 loc = term_val(t);
    for (u32 i = 0; i < arity; i++) {
        Term child = lower_heap_read(ctx, loc + i);
        Term norm = lower_normalize_term(ctx, child);
        if (norm != child) {
            lower_heap_set(ctx, loc + i, norm);
            ctx->lower_ctx.rewrite_count++;
        }
    }

    switch (lop) {
        case LOP_INDEX: {
            Term a = lower_heap_read(ctx, loc + 0);
            Term b = lower_heap_read(ctx, loc + 1);
            u32 scale = 1, av = 0, bv = 0;
            lower_is_const_u(lower_heap_read(ctx, loc + 2), &scale);
            if (scale == 0) return b;
            if (scale == 1 && lower_is_zero(b)) return a;
            if (scale == 1 && lower_is_zero(a)) return b;
            if (lower_is_const_u(a, &av) && lower_is_const_u(b, &bv))
                return lower_const_u(av * scale + bv);
            break;
        }
        case LOP_MOD: {
            Term a = lower_heap_read(ctx, loc + 0);
            u32 mod = 1, av = 0;
            lower_is_const_u(lower_heap_read(ctx, loc + 1), &mod);
            if (mod <= 1) return lower_const_u(0);
            if (lower_is_const_u(a, &av)) return lower_const_u(av % mod);
            break;
        }
        case LOP_DIV: {
            Term a = lower_heap_read(ctx, loc + 0);
            u32 div = 1, av = 0;
            lower_is_const_u(lower_heap_read(ctx, loc + 1), &div);
            if (div <= 1) return a;
            if (lower_is_const_u(a, &av)) return lower_const_u(av / div);
            break;
        }
        case LOP_ALU: {
            Term a = lower_heap_read(ctx, loc + 0);
            Term b = lower_heap_read(ctx, loc + 1);
            u32 uop = 0, aux = 0;
            lower_is_const_u(lower_heap_read(ctx, loc + 2), &uop);
            lower_is_const_u(lower_heap_read(ctx, loc + 3), &aux);
            f32 fa = 0.0f, fb = 0.0f;
            if (lower_is_const_num(a, &fa) && lower_is_const_num(b, &fb))
                return lower_const_f(lower_eval_alu(uop, aux, fa, fb));
            if ((uop == UOP_ADD || uop == UOP_SUB) && lower_is_zero(b)) return a;
            if (uop == UOP_MUL && lower_is_one(b)) return a;
            if (uop == UOP_MUL && lower_is_zero(b)) return lower_const_f(0.0f);
            if ((uop == UOP_NEG || uop == UOP_RELU || uop == UOP_EXP ||
                 uop == UOP_LOG || uop == UOP_SQRT || uop == UOP_CAST) &&
                lower_is_const_num(a, &fa))
                return lower_const_f(lower_eval_alu(uop, aux, fa, 0.0f));
            break;
        }
        case LOP_MASK: {
            Term cond = lower_heap_read(ctx, loc + 0);
            Term value = lower_heap_read(ctx, loc + 1);
            f32 fv = 0.0f;
            if (lower_is_const_num(cond, &fv))
                return fv != 0.0f ? value : lower_const_f(0.0f);
            break;
        }
        case LOP_SINK: {
            Term child = lower_heap_read(ctx, loc + 0);
            if (term_tag(child) == TAG_TOP && term_ext(child) == LOP_SINK)
                return child;
            break;
        }
        default:
            break;
    }
    return t;
}

static Term lower_normalize_root(TinyHVM *ctx, Term root) {
    for (u32 i = 0; i < 4; i++) {
        u32 before = ctx->lower_ctx.rewrite_count;
        root = lower_normalize_term(ctx, root);
        if (ctx->lower_ctx.rewrite_count == before) break;
    }
    return root;
}

static Term lower_build_simple_mask(TinyHVM *ctx, const View *v, Term *coords, u32 rank) {
    if (!v || !v->has_mask) return lower_const_u(1);
    Term mask = lower_const_u(1);
    for (u32 d = 0; d < rank && d < v->shape.rank; d++) {
        if (v->mask_begin[d] > 0)
            mask = lower_bool_and(ctx, mask,
                                  lower_cmp_gt(ctx, coords[d], lower_const_u(v->mask_begin[d] - 1)));
        if (v->mask_end[d] < v->shape.dims[d])
            mask = lower_bool_and(ctx, mask,
                                  lower_cmp_gt(ctx, lower_const_u(v->mask_end[d]), coords[d]));
    }
    for (u32 cm = 0; cm < v->n_compound_masks; cm++) {
        u32 da = v->compound_masks[cm].dim_a;
        u32 db = v->compound_masks[cm].dim_b;
        i32 sa = v->compound_masks[cm].stride_a;
        Term expr = coords[db];
        if (sa != 0) {
            Term scaled = lower_mul(ctx, coords[da], lower_const_f((f32)sa));
            expr = lower_add(ctx, scaled, coords[db]);
        }
        if (v->compound_masks[cm].begin > 0)
            mask = lower_bool_and(ctx, mask,
                                  lower_cmp_gt(ctx, expr, lower_const_f((f32)(v->compound_masks[cm].begin - 1))));
        if (v->compound_masks[cm].end > 0)
            mask = lower_bool_and(ctx, mask,
                                  lower_cmp_gt(ctx, lower_const_f((f32)v->compound_masks[cm].end), expr));
    }
    return mask;
}

static int lower_build_view_index(TinyHVM *ctx, const View *v, Term *coords,
                                  u32 rank, const Shape *full_shape,
                                  LowerLeafAccess *out) {
    if (!v || !out || !full_shape) return 0;
    Term zero = lower_const_u(0);
    Term idx = lower_const_u((u32)(i32)v->offset);
    int coord_ok = (v->shape.rank == rank);
    if (coord_ok) {
        for (u32 d = 0; d < rank; d++) {
            if (v->shape.dims[d] != full_shape->dims[d] &&
                v->strides[d] != 0 && v->shape.dims[d] != 1) {
                coord_ok = 0;
                break;
            }
            if (v->strides[d] < 0) {
                coord_ok = 0;
                break;
            }
        }
    }
    if (!coord_ok) {
        u32 flat_strides[MAX_DIM] = {0};
        if (rank > 0) {
            flat_strides[rank - 1] = 1;
            for (int d = (int)rank - 2; d >= 0; d--)
                flat_strides[d] = flat_strides[d + 1] * full_shape->dims[d + 1];
        }
        Term fi = lower_const_u(0);
        for (u32 d = 0; d < rank; d++) {
            Term term = coords[d];
            if (flat_strides[d] != 1)
                term = lower_index(ctx, coords[d], zero, flat_strides[d]);
            fi = lower_index(ctx, fi, term, 1);
        }
        Term leaf_idx = lower_const_u((u32)(i32)v->offset);
        u32 leaf_div = 1;
        for (int d = (int)v->shape.rank - 1; d >= 0; d--) {
            if (v->strides[d] < 0) return 0;
            if (v->strides[d] == 0) {
                leaf_div *= v->shape.dims[d];
                continue;
            }
            u32 mod = (v->mod_size[d] > 0 && v->mod_size[d] < v->shape.dims[d]) ? v->mod_size[d] : v->shape.dims[d];
            Term cd = (leaf_div == 1) ? lower_mod(ctx, fi, mod)
                                      : lower_mod(ctx, lower_div_term(ctx, fi, leaf_div), mod);
            Term term = (v->strides[d] == 1) ? cd : lower_index(ctx, cd, zero, (u32)v->strides[d]);
            leaf_idx = lower_index(ctx, leaf_idx, term, 1);
            leaf_div *= v->shape.dims[d];
        }
        out->index = leaf_idx;
    } else {
        for (u32 d = 0; d < rank && d < v->shape.rank; d++) {
            if (v->strides[d] < 0) return 0;
            if (v->strides[d] == 0 || v->shape.dims[d] == 1) continue;
            Term coord = coords[d];
            if (v->mod_size[d] > 0 && v->mod_size[d] < v->shape.dims[d])
                coord = lower_mod(ctx, coord, v->mod_size[d]);
            Term term = (v->strides[d] == 1) ? coord : lower_index(ctx, coord, zero, (u32)v->strides[d]);
            idx = lower_index(ctx, idx, term, 1);
        }
        out->index = idx;
    }
    out->mask = lower_build_simple_mask(ctx, v, coords, rank);
    return 1;
}

static int lower_build_tracker_index(TinyHVM *ctx, const ShapeTracker *st, Term *coords,
                                     u32 rank, const Shape *full_shape,
                                     LowerLeafAccess *out) {
    if (!st || st->n_views < 2 || !out || !full_shape) return 0;
    Term zero = lower_const_u(0);
    u32 full_strides[MAX_DIM] = {0};
    if (rank > 0) {
        full_strides[rank - 1] = 1;
        for (int d = (int)rank - 2; d >= 0; d--)
            full_strides[d] = full_strides[d + 1] * full_shape->dims[d + 1];
    }
    Term si = lower_const_u(0);
    for (u32 d = 0; d < rank; d++) {
        Term term = coords[d];
        if (full_strides[d] != 1)
            term = lower_index(ctx, coords[d], zero, full_strides[d]);
        si = lower_index(ctx, si, term, 1);
    }

    Term mask = lower_const_u(1);
    for (int vi = (int)st->n_views - 2; vi >= 0; vi--) {
        const View *vw = &st->views[vi];
        const View *outer = &st->views[vi + 1];
        Term new_si = lower_const_u((u32)(i32)vw->offset);
        Term unr_coords[MAX_DIM] = {0};
        u32 udiv = 1;
        for (int d = (int)outer->shape.rank - 1; d >= 0; d--) {
            u32 dim = outer->shape.dims[d];
            if (dim <= 1) {
                udiv *= dim;
                continue;
            }
            Term cd = (udiv == 1) ? lower_mod(ctx, si, dim)
                                  : lower_mod(ctx, lower_div_term(ctx, si, udiv), dim);
            unr_coords[d] = cd;
            if ((u32)d < vw->shape.rank) {
                if (vw->strides[d] < 0) return 0;
                if (vw->strides[d] != 0) {
                    Term use_cd = cd;
                    if (vw->mod_size[d] > 0 && vw->mod_size[d] < vw->shape.dims[d])
                        use_cd = lower_mod(ctx, cd, vw->mod_size[d]);
                    Term term = (vw->strides[d] == 1) ? use_cd : lower_index(ctx, use_cd, zero, (u32)vw->strides[d]);
                    new_si = lower_index(ctx, new_si, term, 1);
                }
                if (vw->has_mask) {
                    if (vw->mask_begin[d] > 0)
                        mask = lower_bool_and(ctx, mask,
                                              lower_cmp_gt(ctx, cd, lower_const_u(vw->mask_begin[d] - 1)));
                    if (vw->mask_end[d] < vw->shape.dims[d])
                        mask = lower_bool_and(ctx, mask,
                                              lower_cmp_gt(ctx, lower_const_u(vw->mask_end[d]), cd));
                }
            }
            udiv *= dim;
        }
        for (u32 cm = 0; cm < vw->n_compound_masks; cm++) {
            u32 da = vw->compound_masks[cm].dim_a;
            u32 db = vw->compound_masks[cm].dim_b;
            i32 sa = vw->compound_masks[cm].stride_a;
            Term expr = unr_coords[db] ? unr_coords[db] : zero;
            if (sa != 0) {
                Term scaled = lower_mul(ctx, unr_coords[da] ? unr_coords[da] : zero, lower_const_f((f32)sa));
                expr = lower_add(ctx, scaled, expr);
            }
            if (vw->compound_masks[cm].begin > 0)
                mask = lower_bool_and(ctx, mask,
                                      lower_cmp_gt(ctx, expr, lower_const_f((f32)(vw->compound_masks[cm].begin - 1))));
            if (vw->compound_masks[cm].end > 0)
                mask = lower_bool_and(ctx, mask,
                                      lower_cmp_gt(ctx, lower_const_f((f32)vw->compound_masks[cm].end), expr));
        }
        si = new_si;
    }
    out->index = si;
    out->mask = mask;
    return 1;
}

static int lower_build_leaf_access(TinyHVM *ctx, const KernelEntry *ke, u32 leaf_idx,
                                   Term *coords, u32 rank, LowerLeafAccess *out) {
    if (!ke || !out || leaf_idx >= ke->n_leaves) return 0;
    const ShapeTracker *st = &ke->leaf_sts[leaf_idx];
    if (st && st->n_views >= 2)
        return lower_build_tracker_index(ctx, st, coords, rank, &ke->full_shape, out);
    return lower_build_view_index(ctx, &ke->leaf_views[leaf_idx], coords, rank, &ke->full_shape, out);
}

static inline u32 lower_emit_op(UOpKernel *uk, KOpType type, u32 a0, u32 a1, u32 a2, u32 imm_u) {
    if (uk->n_ops >= KOP_MAX) return 0xFFFFFFFFu;
    u32 id = uk->n_ops++;
    uk->ops[id].type = type;
    uk->ops[id].arg[0] = a0;
    uk->ops[id].arg[1] = a1;
    uk->ops[id].arg[2] = a2;
    uk->ops[id].imm.u = imm_u;
    return id;
}

static inline u32 lower_emit_op_f(UOpKernel *uk, KOpType type, u32 a0, u32 a1, u32 a2, f32 imm_f) {
    if (uk->n_ops >= KOP_MAX) return 0xFFFFFFFFu;
    u32 id = uk->n_ops++;
    uk->ops[id].type = type;
    uk->ops[id].arg[0] = a0;
    uk->ops[id].arg[1] = a1;
    uk->ops[id].arg[2] = a2;
    uk->ops[id].imm.f = imm_f;
    return id;
}

static int lower_is_pure_op(u32 lop) {
    return lop == LOP_GID || lop == LOP_CONST_U || lop == LOP_CONST_F ||
           lop == LOP_RANGE || lop == LOP_LOAD || lop == LOP_ALU ||
           lop == LOP_ACC_INIT || lop == LOP_INDEX || lop == LOP_MOD ||
           lop == LOP_DIV || lop == LOP_MASK;
}

static u32 lower_emit_term(TinyHVM *ctx, Term t, LowerEmitCtx *ec) {
    if (ec->failed) return 0xFFFFFFFFu;
    if (term_tag(t) != TAG_TOP) {
        ec->failed = 1;
        return 0xFFFFFFFFu;
    }
    u32 lop = term_ext(t);
    u64 loc = term_val(t);

    if (lower_term_arity(t) > 0 &&
        loc > 0 && loc < ctx->lower_ctx.heap_pos && lower_is_pure_op(lop) &&
        ec->memo_valid[loc]) {
        return ec->memo[loc];
    }

    u32 out = 0xFFFFFFFFu;
    switch (lop) {
        case LOP_CONST_U:
            out = lower_emit_op(ec->uk, KOP_CONST_U, 0, 0, 0, (u32)term_val(t));
            break;
        case LOP_CONST_F: {
            u32 bits = (u32)term_val(t);
            f32 f = 0.0f;
            memcpy(&f, &bits, sizeof(bits));
            out = lower_emit_op_f(ec->uk, KOP_CONST_F, 0, 0, 0, f);
            break;
        }
        case LOP_GID:
            out = lower_emit_op(ec->uk, KOP_GID, 0, 0, 0, (u32)term_val(t));
            break;
        case LOP_INDEX: {
            u32 a = lower_emit_term(ctx, lower_heap_read(ctx, loc + 0), ec);
            u32 b = lower_emit_term(ctx, lower_heap_read(ctx, loc + 1), ec);
            u32 scale = 1;
            lower_is_const_u(lower_heap_read(ctx, loc + 2), &scale);
            out = lower_emit_op(ec->uk, KOP_IDX, a, b, 0, scale);
            break;
        }
        case LOP_MOD: {
            u32 a = lower_emit_term(ctx, lower_heap_read(ctx, loc + 0), ec);
            u32 mod = 1;
            lower_is_const_u(lower_heap_read(ctx, loc + 1), &mod);
            out = lower_emit_op(ec->uk, KOP_MOD, a, 0, 0, mod);
            break;
        }
        case LOP_DIV: {
            u32 a = lower_emit_term(ctx, lower_heap_read(ctx, loc + 0), ec);
            u32 div = 1;
            lower_is_const_u(lower_heap_read(ctx, loc + 1), &div);
            out = lower_emit_op(ec->uk, KOP_DIV, a, 0, 0, div);
            break;
        }
        case LOP_LOAD: {
            u32 idx = lower_emit_term(ctx, lower_heap_read(ctx, loc + 0), ec);
            u32 buf = 1;
            lower_is_const_u(lower_heap_read(ctx, loc + 1), &buf);
            out = lower_emit_op(ec->uk, KOP_LOAD, idx, 0, 0, buf);
            break;
        }
        case LOP_MASK: {
            u32 cond = lower_emit_term(ctx, lower_heap_read(ctx, loc + 0), ec);
            u32 val = lower_emit_term(ctx, lower_heap_read(ctx, loc + 1), ec);
            out = lower_emit_op(ec->uk, KOP_MASK, cond, val, 0, 0);
            break;
        }
        case LOP_ALU: {
            u32 a = lower_emit_term(ctx, lower_heap_read(ctx, loc + 0), ec);
            u32 b = lower_emit_term(ctx, lower_heap_read(ctx, loc + 1), ec);
            u32 uop = 0, aux = 0;
            lower_is_const_u(lower_heap_read(ctx, loc + 2), &uop);
            lower_is_const_u(lower_heap_read(ctx, loc + 3), &aux);
            out = lower_emit_op(ec->uk, KOP_ALU, a, b, aux, uop);
            break;
        }
        case LOP_ACC_INIT: {
            f32 init = 0.0f;
            lower_is_const_f(lower_heap_read(ctx, loc + 0), &init);
            out = lower_emit_op_f(ec->uk, KOP_ACC_INIT, 0, 0, 0, init);
            break;
        }
        case LOP_RANGE: {
            u32 trip = 0;
            lower_is_const_u(lower_heap_read(ctx, loc + 0), &trip);
            out = lower_emit_op(ec->uk, KOP_RANGE, 0, 0, 0, trip);
            break;
        }
        case LOP_ACC: {
            u32 acc = lower_emit_term(ctx, lower_heap_read(ctx, loc + 0), ec);
            u32 val = lower_emit_term(ctx, lower_heap_read(ctx, loc + 1), ec);
            u32 red = 0;
            lower_is_const_u(lower_heap_read(ctx, loc + 2), &red);
            if (lower_emit_op(ec->uk, KOP_ACC, acc, val, 0, red) == 0xFFFFFFFFu)
                ec->failed = 1;
            out = acc;
            break;
        }
        case LOP_ENDRANGE: {
            Term body_term = lower_heap_read(ctx, loc + 1);
            u32 body = 0xFFFFFFFFu;
            if (term_tag(body_term) == TAG_TOP && term_ext(body_term) == LOP_ACC) {
                u64 bloc = term_val(body_term);
                u32 acc = lower_emit_term(ctx, lower_heap_read(ctx, bloc + 0), ec);
                u32 range = lower_emit_term(ctx, lower_heap_read(ctx, loc + 0), ec);
                u32 val = lower_emit_term(ctx, lower_heap_read(ctx, bloc + 1), ec);
                u32 red = 0;
                lower_is_const_u(lower_heap_read(ctx, bloc + 2), &red);
                if (lower_emit_op(ec->uk, KOP_ACC, acc, val, 0, red) == 0xFFFFFFFFu)
                    ec->failed = 1;
                if (lower_emit_op(ec->uk, KOP_ENDRANGE, range, 0, 0, 0) == 0xFFFFFFFFu)
                    ec->failed = 1;
                out = acc;
                break;
            }
            u32 range = lower_emit_term(ctx, lower_heap_read(ctx, loc + 0), ec);
            body = lower_emit_term(ctx, body_term, ec);
            if (lower_emit_op(ec->uk, KOP_ENDRANGE, range, 0, 0, 0) == 0xFFFFFFFFu)
                ec->failed = 1;
            out = body;
            break;
        }
        case LOP_STORE: {
            u32 idx = lower_emit_term(ctx, lower_heap_read(ctx, loc + 0), ec);
            u32 val = lower_emit_term(ctx, lower_heap_read(ctx, loc + 1), ec);
            if (lower_emit_op(ec->uk, KOP_STORE, idx, val, 0, 0) == 0xFFFFFFFFu)
                ec->failed = 1;
            out = val;
            break;
        }
        case LOP_SINK:
            out = lower_emit_term(ctx, lower_heap_read(ctx, loc + 0), ec);
            break;
        default:
            ec->failed = 1;
            return 0xFFFFFFFFu;
    }

    if (out == 0xFFFFFFFFu) ec->failed = 1;
    if (!ec->failed && lower_term_arity(t) > 0 &&
        loc > 0 && loc < ctx->lower_ctx.heap_pos && lower_is_pure_op(lop)) {
        ec->memo_valid[loc] = 1;
        ec->memo[loc] = out;
    }
    return out;
}

static int thvm_lower_kernel_uop(TinyHVM *ctx, const KernelEntry *ke, UOpKernel *out, u64 *out_lower_sig) {
    if (!ctx || !ke || !out) return 0;
    lower_ctx_reset(ctx);
    memset(out, 0, sizeof(*out));

    u32 rank = ke->full_shape.rank;
    if (rank == 0) rank = 1;
    out->rank = rank;
    out->full_shape = ke->full_shape;
    out->n_leaves = ke->n_leaves;
    out->n_bufs = 1 + ke->n_leaves;
    out->out_dtype = 0;
    u32 out_tid = ke->output_tid ? ke->output_tid : ke->raw_output_tid;
    if (out_tid < ctx->tensor_count) out->out_dtype = ctx->tensors[out_tid].dtype;
    for (u32 i = 0; i < ke->n_leaves && i < FUSE_MAX_LEAVES; i++) {
        lower_strip_view_mask(&out->leaf_views[i], &ke->leaf_views[i]);
        out->leaf_dtypes[i] = ke->leaf_dtypes[i];
    }

    int has_reduce = ke->has_reduce != 0;
    u32 out_dims[MAX_DIM] = {0}, n_out = 0;
    u32 red_dims[MAX_DIM] = {0}, n_red = 0;
    u32 out_numel = 1, reduce_numel = 1;
    for (u32 d = 0; d < rank; d++) {
        if (has_reduce && ke->reduce.is_reduce[d]) {
            red_dims[n_red++] = d;
            reduce_numel *= ke->full_shape.dims[d];
        } else {
            out_dims[n_out++] = d;
            out_numel *= ke->full_shape.dims[d];
        }
    }
    u32 inner = 1, mid = 1, outer = 1;
    u32 inner_start = n_out;
    u32 out_shape[MAX_DIM] = {0};
    for (u32 i = 0; i < n_out; i++) out_shape[i] = ke->full_shape.dims[out_dims[i]];
    for (int i = (int)n_out - 1; i >= 0; i--) {
        if (inner * out_shape[i] <= 1024) { inner *= out_shape[i]; inner_start = (u32)i; }
        else break;
    }
    u32 mid_start = inner_start;
    for (int i = (int)inner_start - 1; i >= 0; i--) {
        if (mid * out_shape[i] <= 65535) { mid *= out_shape[i]; mid_start = (u32)i; }
        else break;
    }
    for (u32 i = 0; i < mid_start; i++) outer *= out_shape[i];
    out->grid[0] = inner;
    out->grid[1] = mid;
    out->grid[2] = outer;
    out->tg[2] = 1;
    if (has_reduce) {
        u32 local_size = 256;
        int use_gr = (reduce_numel >= 256) && (ke->reduce.post_reduce_start == 0) &&
                     (ke->reduce.reduce2_type == 0);
        if (use_gr) {
            out->local_size = local_size;
            out->reduce_numel = reduce_numel;
            out->grid[0] = out_numel;
            out->grid[1] = 1;
            out->grid[2] = 1;
        }
    }

    Term coords[MAX_DIM];
    for (u32 d = 0; d < MAX_DIM; d++) coords[d] = lower_const_u(0);
    Term gid = lower_gid(0);
    Term zero = lower_const_u(0);
    if (n_out > 0) {
        Term rem = gid;
        for (int oi = (int)n_out - 1; oi >= 0; oi--) {
            u32 d = out_dims[oi];
            u32 dim = ke->full_shape.dims[d];
            if (dim == 1) {
                coords[d] = zero;
                continue;
            }
            coords[d] = lower_mod(ctx, rem, dim);
            if (oi > 0) rem = lower_div_term(ctx, rem, dim);
        }
    }

    Term vals[FUSE_MAX_LEAVES + FUSE_MAX_OPS];
    memset(vals, 0, sizeof(vals));
    Term root = term_era();

    if (!has_reduce) {
        for (u32 li = 0; li < ke->n_leaves; li++) {
            LowerLeafAccess access = {0};
            if (!lower_build_leaf_access(ctx, ke, li, coords, rank, &access)) return 0;
            vals[li] = lower_masked_load(ctx, access.index, access.mask, li);
        }
        for (u32 i = 0; i < ke->n_ops; i++) {
            u32 aa = ke->ops[i].arg_a;
            u32 bb = ke->ops[i].arg_b;
            Term a = vals[aa];
            Term b = vals[bb];
            vals[ke->n_leaves + i] = lower_alu(ctx, a, b, ke->ops[i].uop, ke->ops[i].aux);
        }
        Term result = (ke->n_ops > 0) ? vals[ke->n_leaves + ke->n_ops - 1] : vals[0];
        Term out_idx = lower_index(ctx, gid, zero, 1);
        root = lower_sink(ctx, lower_store(ctx, out_idx, result));
    } else {
        f32 init = (ke->reduce.reduce_type == UOP_RMAX) ? -1e30f : 0.0f;
        Term acc = lower_acc_init(ctx, init);
        Term range = lower_range(ctx, reduce_numel);
        Term range_rem = range;
        for (int ri = (int)n_red - 1; ri >= 0; ri--) {
            u32 d = red_dims[ri];
            u32 dim = ke->full_shape.dims[d];
            coords[d] = lower_mod(ctx, range_rem, dim);
            if (ri > 0) range_rem = lower_div_term(ctx, range_rem, dim);
        }
        for (u32 li = 0; li < ke->n_leaves; li++) {
            LowerLeafAccess access = {0};
            if (!lower_build_leaf_access(ctx, ke, li, coords, rank, &access)) return 0;
            vals[li] = lower_masked_load(ctx, access.index, access.mask, li);
        }
        u32 pre_ops = (ke->reduce.post_reduce_start > 0) ? ke->reduce.post_reduce_start : ke->n_ops;
        for (u32 i = 0; i < pre_ops; i++) {
            u32 aa = ke->ops[i].arg_a;
            u32 bb = ke->ops[i].arg_b;
            vals[ke->n_leaves + i] = lower_alu(ctx, vals[aa], vals[bb], ke->ops[i].uop, ke->ops[i].aux);
        }
        Term pre_result = (pre_ops > 0) ? vals[ke->n_leaves + pre_ops - 1] : vals[0];
        Term result = lower_endrange(ctx, range, lower_acc(ctx, acc, pre_result, ke->reduce.reduce_type));

        if (ke->reduce.post_reduce_start > 0 && ke->reduce.post_reduce_start < ke->n_ops) {
            Term post_coords[MAX_DIM];
            memcpy(post_coords, coords, sizeof(post_coords));
            for (u32 ri = 0; ri < n_red; ri++) post_coords[red_dims[ri]] = zero;
            u32 prs = ke->reduce.post_reduce_start;
            u32 post_leaf_base = ke->n_leaves - ke->reduce.n_post_leaves;
            for (u32 pli = post_leaf_base; pli < ke->n_leaves; pli++) {
                LowerLeafAccess access = {0};
                if (!lower_build_leaf_access(ctx, ke, pli, post_coords, rank, &access)) return 0;
                vals[pli] = lower_masked_load(ctx, access.index, access.mask, pli);
            }
            Term prev = result;
            for (u32 i = prs; i < ke->n_ops; i++) {
                Term b = result;
                if (is_binary(ke->ops[i].uop)) {
                    u32 ob = ke->ops[i].arg_b;
                    if (ob >= post_leaf_base && ob < ke->n_leaves)
                        b = vals[ob];
                }
                prev = lower_alu(ctx, prev, b, ke->ops[i].uop, ke->ops[i].aux);
            }
            result = prev;
        }

        if (ke->reduce.reduce2_type && ke->reduce.reduce2_start < ke->n_ops) {
            u32 reduce2_numel = 1;
            u32 reduce2_dims[MAX_DIM] = {0};
            u32 n_red2 = 0;
            for (u32 d = 0; d < rank; d++) {
                if (ke->reduce.is_reduce2[d]) {
                    reduce2_dims[n_red2++] = d;
                    reduce2_numel *= ke->full_shape.dims[d];
                }
            }
            if (n_red2 == 0) {
                n_red2 = n_red;
                reduce2_numel = reduce_numel;
                memcpy(reduce2_dims, red_dims, sizeof(red_dims));
            }
            f32 init2 = (ke->reduce.reduce2_type == UOP_RMAX) ? -1e30f : 0.0f;
            Term acc2 = lower_acc_init(ctx, init2);
            Term range2 = lower_range(ctx, reduce2_numel);
            Term range2_rem = range2;
            for (int ri = (int)n_red2 - 1; ri >= 0; ri--) {
                u32 d = reduce2_dims[ri];
                u32 dim = ke->full_shape.dims[d];
                coords[d] = lower_mod(ctx, range2_rem, dim);
                if (ri > 0) range2_rem = lower_div_term(ctx, range2_rem, dim);
            }
            for (u32 li = 0; li < ke->n_leaves; li++) {
                LowerLeafAccess access = {0};
                if (!lower_build_leaf_access(ctx, ke, li, coords, rank, &access)) return 0;
                vals[li] = lower_masked_load(ctx, access.index, access.mask, li);
            }
            for (u32 i = ke->reduce.reduce2_start; i < ke->n_ops; i++) {
                u32 aa = ke->ops[i].arg_a;
                u32 bb = ke->ops[i].arg_b;
                Term va = (aa < ke->n_leaves) ? vals[aa] : (aa == ke->n_leaves) ? result : vals[aa];
                Term vb = (bb < ke->n_leaves) ? vals[bb] : (bb == ke->n_leaves) ? result : vals[bb];
                vals[ke->n_leaves + i] = lower_alu(ctx, va, vb, ke->ops[i].uop, ke->ops[i].aux);
            }
            Term acc_body = lower_acc(ctx, acc2, vals[ke->n_leaves + ke->n_ops - 1], ke->reduce.reduce2_type);
            result = lower_endrange(ctx, range2, acc_body);
        }

        Term out_idx = lower_index(ctx, gid, zero, 1);
        root = lower_sink(ctx, lower_store(ctx, out_idx, result));
    }

    ctx->lower_ctx.root = lower_normalize_root(ctx, root);

    u32 *memo = (u32 *)calloc((size_t)ctx->lower_ctx.heap_pos, sizeof(u32));
    u8  *memo_valid = (u8 *)calloc((size_t)ctx->lower_ctx.heap_pos, 1);
    LowerEmitCtx ec = { .uk = out, .memo = memo, .memo_valid = memo_valid, .failed = 0 };
    (void)lower_emit_term(ctx, ctx->lower_ctx.root, &ec);
    free(memo_valid);
    free(memo);
    if (ec.failed) return 0;

    ctx->lower_ctx.normalized_sig = thvm_uop_kernel_signature(out);
    if (out_lower_sig) *out_lower_sig = ctx->lower_ctx.normalized_sig;
    return 1;
}

static void thvm_lower_dump_uop_kernel(const UOpKernel *uk) {
    if (!uk) return;
    for (u32 i = 0; i < uk->n_ops; i++) {
        const KOp *op = &uk->ops[i];
        switch (op->type) {
            case KOP_CONST_F:
            case KOP_ACC_INIT:
                fprintf(stderr, "  UOP[%u] type=%u a=%u b=%u c=%u immf=%g\n",
                        i, op->type, op->arg[0], op->arg[1], op->arg[2], op->imm.f);
                break;
            default:
                fprintf(stderr, "  UOP[%u] type=%u a=%u b=%u c=%u imm=%u\n",
                        i, op->type, op->arg[0], op->arg[1], op->arg[2], op->imm.u);
                break;
        }
    }
}

static int lower_env_enabled(const char *name) {
    const char *env = getenv(name);
    return env && env[0] && strcmp(env, "0") != 0;
}

static const char *lower_backend_short(TinyHVM *ctx, Backend *backend) {
    if (!backend && ctx) backend = ctx_default_backend(ctx);
    if (!backend) return "?";
    if (ctx && ctx->backends[THVM_DEV_CPU] == backend) return "cpu";
    if (ctx && ctx->backends[THVM_DEV_METAL] == backend) return "mtl";
    return "dev";
}

static void lower_shape_str(const Shape *shape, char *buf, size_t nbuf) {
    if (!buf || nbuf == 0) return;
    if (!shape) {
        snprintf(buf, nbuf, "[]");
        return;
    }
    size_t pos = 0;
    pos += snprintf(buf + pos, nbuf - pos, "[");
    if (shape->rank == 0) {
        snprintf(buf + pos, nbuf - pos, "1]");
        return;
    }
    for (u32 d = 0; d < shape->rank && pos + 8 < nbuf; d++)
        pos += snprintf(buf + pos, nbuf - pos, "%s%u", d ? "," : "", shape->dims[d]);
    snprintf(buf + pos, nbuf - pos, "]");
}

static void lower_axes_str(const ReduceSpec *reduce, u32 rank, char *buf, size_t nbuf) {
    if (!buf || nbuf == 0) return;
    size_t pos = 0;
    pos += snprintf(buf + pos, nbuf - pos, "[");
    int first = 1;
    if (reduce) {
        for (u32 d = 0; d < rank && pos + 8 < nbuf; d++) {
            if (!reduce->is_reduce[d]) continue;
            pos += snprintf(buf + pos, nbuf - pos, "%s%u", first ? "" : ",", d);
            first = 0;
        }
    }
    snprintf(buf + pos, nbuf - pos, "]");
}

static void lower_view_str(const View *view, char *buf, size_t nbuf) {
    if (!buf || nbuf == 0) return;
    if (!view) {
        snprintf(buf, nbuf, "view=none");
        return;
    }
    char shape[96];
    size_t pos = 0;
    lower_shape_str(&view->shape, shape, sizeof(shape));
    pos += snprintf(buf + pos, nbuf - pos, "shape=%s strides=[", shape);
    for (u32 d = 0; d < view->shape.rank && pos + 16 < nbuf; d++)
        pos += snprintf(buf + pos, nbuf - pos, "%s%d", d ? "," : "", view->strides[d]);
    pos += snprintf(buf + pos, nbuf - pos, "] off=%d", view->offset);
    if (view->has_mask && pos + 64 < nbuf) {
        pos += snprintf(buf + pos, nbuf - pos, " mask=[");
        for (u32 d = 0; d < view->shape.rank && pos + 16 < nbuf; d++)
            pos += snprintf(buf + pos, nbuf - pos, "%s%u:%u",
                            d ? "," : "", view->mask_begin[d], view->mask_end[d]);
        snprintf(buf + pos, nbuf - pos, "]");
    }
}

static const char *lower_lop_port_name(u32 lop, u32 idx) {
    switch (lop) {
        case LOP_SINK: return "in";
        case LOP_RANGE: return "trip";
        case LOP_ENDRANGE: return idx == 0 ? "range" : "body";
        case LOP_LOAD: return idx == 0 ? "idx" : "buf";
        case LOP_STORE: return idx == 0 ? "idx" : "val";
        case LOP_ALU: return idx == 0 ? "a" : idx == 1 ? "b" : idx == 2 ? "uop" : "aux";
        case LOP_ACC_INIT: return "init";
        case LOP_ACC: return idx == 0 ? "acc" : idx == 1 ? "val" : "reduce";
        case LOP_INDEX: return idx == 0 ? "base" : idx == 1 ? "add" : "scale";
        case LOP_MOD: return idx == 0 ? "a" : "mod";
        case LOP_DIV: return idx == 0 ? "a" : "div";
        case LOP_MASK: return idx == 0 ? "cond" : "val";
        default: return idx == 0 ? "a" : "b";
    }
}

static const char *lower_kop_name(KOpType type) {
    switch (type) {
        case KOP_GID: return "KOP_GID";
        case KOP_CONST_F: return "KOP_CONST_F";
        case KOP_CONST_U: return "KOP_CONST_U";
        case KOP_RANGE: return "KOP_RANGE";
        case KOP_ENDRANGE: return "KOP_ENDRANGE";
        case KOP_LOAD: return "KOP_LOAD";
        case KOP_STORE: return "KOP_STORE";
        case KOP_ALU: return "KOP_ALU";
        case KOP_ACC_INIT: return "KOP_ACC_INIT";
        case KOP_ACC: return "KOP_ACC";
        case KOP_IDX: return "KOP_IDX";
        case KOP_MOD: return "KOP_MOD";
        case KOP_DIV: return "KOP_DIV";
        case KOP_MASK: return "KOP_MASK";
        default: return "KOP_NOOP";
    }
}

static const char *lower_kop_port_name(const KOp *op, u32 idx) {
    if (!op) return "";
    switch (op->type) {
        case KOP_MOD:
        case KOP_DIV:
            return idx == 0 ? "a" : "";
        case KOP_IDX:
            return idx == 0 ? "base" : idx == 1 ? "add" : "";
        case KOP_LOAD:
            return idx == 0 ? "idx" : "";
        case KOP_STORE:
            return idx == 0 ? "idx" : idx == 1 ? "val" : "";
        case KOP_ALU:
            return idx == 0 ? "a" : idx == 1 ? "b" : "";
        case KOP_ACC:
            return idx == 0 ? "acc" : idx == 1 ? "val" : "";
        case KOP_ENDRANGE:
            return idx == 0 ? "range" : "";
        case KOP_MASK:
            return idx == 0 ? "cond" : idx == 1 ? "val" : "";
        default:
            return "";
    }
}

static void lower_term_id(Term t, char *buf, size_t nbuf) {
    if (!buf || nbuf == 0) return;
    if (term_tag(t) == TAG_TOP && lower_term_arity(t) > 0) {
        snprintf(buf, nbuf, "l%llu", (unsigned long long)term_val(t));
        return;
    }
    snprintf(buf, nbuf, "li_%u_%llu",
             (u32)term_ext(t), (unsigned long long)term_val(t));
}

static const char *lower_dot_fill(Term t) {
    if (term_tag(t) != TAG_TOP) return "#f0f0f0";
    switch (term_ext(t)) {
        case LOP_SINK: return "#ffd9d9";
        case LOP_RANGE:
        case LOP_ENDRANGE: return "#ffe7cc";
        case LOP_LOAD:
        case LOP_STORE: return "#d9f2e6";
        case LOP_ALU:
        case LOP_ACC:
        case LOP_ACC_INIT:
        case LOP_MASK: return "#dbe8ff";
        case LOP_INDEX:
        case LOP_MOD:
        case LOP_DIV: return "#efe2ff";
        case LOP_GID:
        case LOP_CONST_U:
        case LOP_CONST_F: return "#fff2cc";
        default: return "#f0f0f0";
    }
}

static void lower_term_label(TinyHVM *ctx, Term t, char *buf, size_t nbuf) {
    if (!buf || nbuf == 0) return;
    if (term_tag(t) != TAG_TOP) {
        snprintf(buf, nbuf, "TERM");
        return;
    }
    u32 lop = term_ext(t);
    u64 loc = term_val(t);
    switch (lop) {
        case LOP_GID:
            snprintf(buf, nbuf, "LGID\\naxis=%llu", (unsigned long long)term_val(t));
            return;
        case LOP_CONST_U:
            snprintf(buf, nbuf, "LCONST_U\\n%u", (u32)term_val(t));
            return;
        case LOP_CONST_F: {
            u32 bits = (u32)term_val(t);
            f32 f = 0.0f;
            memcpy(&f, &bits, sizeof(bits));
            snprintf(buf, nbuf, "LCONST_F\\n%.6g", (double)f);
            return;
        }
        case LOP_RANGE: {
            u32 trip = 0;
            lower_is_const_u(lower_heap_read(ctx, loc + 0), &trip);
            snprintf(buf, nbuf, "LRANGE\\ntrip=%u\\n@%llu",
                     trip, (unsigned long long)loc);
            return;
        }
        case LOP_ENDRANGE:
            snprintf(buf, nbuf, "LENDRANGE\\n@%llu", (unsigned long long)loc);
            return;
        case LOP_LOAD: {
            u32 buf_idx = 0;
            lower_is_const_u(lower_heap_read(ctx, loc + 1), &buf_idx);
            snprintf(buf, nbuf, "LLOAD\\nleaf=%u\\n@%llu",
                     buf_idx > 0 ? buf_idx - 1 : 0, (unsigned long long)loc);
            return;
        }
        case LOP_STORE:
            snprintf(buf, nbuf, "LSTORE\\n@%llu", (unsigned long long)loc);
            return;
        case LOP_ALU: {
            u32 uop = 0, aux = 0;
            lower_is_const_u(lower_heap_read(ctx, loc + 2), &uop);
            lower_is_const_u(lower_heap_read(ctx, loc + 3), &aux);
            snprintf(buf, nbuf, "LALU\\n%s aux=%u\\n@%llu",
                     uop < UOP_COUNT ? uop_names[uop] : "?", aux,
                     (unsigned long long)loc);
            return;
        }
        case LOP_ACC_INIT: {
            f32 init = 0.0f;
            lower_is_const_f(lower_heap_read(ctx, loc + 0), &init);
            snprintf(buf, nbuf, "LACC_INIT\\n%.6g\\n@%llu",
                     (double)init, (unsigned long long)loc);
            return;
        }
        case LOP_ACC: {
            u32 red = 0;
            lower_is_const_u(lower_heap_read(ctx, loc + 2), &red);
            snprintf(buf, nbuf, "LACC\\n%s\\n@%llu",
                     red < UOP_COUNT ? uop_names[red] : "?",
                     (unsigned long long)loc);
            return;
        }
        case LOP_INDEX: {
            u32 scale = 1;
            lower_is_const_u(lower_heap_read(ctx, loc + 2), &scale);
            snprintf(buf, nbuf, "LINDEX\\nscale=%u\\n@%llu",
                     scale, (unsigned long long)loc);
            return;
        }
        case LOP_MOD: {
            u32 mod = 1;
            lower_is_const_u(lower_heap_read(ctx, loc + 1), &mod);
            snprintf(buf, nbuf, "LMOD\\n%u\\n@%llu",
                     mod, (unsigned long long)loc);
            return;
        }
        case LOP_DIV: {
            u32 div = 1;
            lower_is_const_u(lower_heap_read(ctx, loc + 1), &div);
            snprintf(buf, nbuf, "LDIV\\n%u\\n@%llu",
                     div, (unsigned long long)loc);
            return;
        }
        case LOP_MASK:
            snprintf(buf, nbuf, "LMASK\\n@%llu", (unsigned long long)loc);
            return;
        case LOP_SINK:
            snprintf(buf, nbuf, "LSINK\\n@%llu", (unsigned long long)loc);
            return;
        default:
            snprintf(buf, nbuf, "%s\\n@%llu",
                     lower_op_name(lop), (unsigned long long)loc);
            return;
    }
}

static const char *lower_kop_fill(const KOp *op) {
    if (!op) return "#f0f0f0";
    switch (op->type) {
        case KOP_RANGE:
        case KOP_ENDRANGE: return "#ffe7cc";
        case KOP_LOAD:
        case KOP_STORE: return "#d9f2e6";
        case KOP_ALU:
        case KOP_ACC:
        case KOP_ACC_INIT:
        case KOP_MASK: return "#dbe8ff";
        case KOP_IDX:
        case KOP_MOD:
        case KOP_DIV: return "#efe2ff";
        case KOP_GID:
        case KOP_CONST_U:
        case KOP_CONST_F: return "#fff2cc";
        default: return "#f0f0f0";
    }
}

static void lower_kop_label(const UOpKernel *uk, u32 idx, char *buf, size_t nbuf) {
    if (!buf || nbuf == 0 || !uk || idx >= uk->n_ops) return;
    const KOp *op = &uk->ops[idx];
    switch (op->type) {
        case KOP_GID:
            snprintf(buf, nbuf, "%u\\nKOP_GID\\naxis=%u", idx, op->imm.u);
            return;
        case KOP_CONST_U:
            snprintf(buf, nbuf, "%u\\nKOP_CONST_U\\n%u", idx, op->imm.u);
            return;
        case KOP_CONST_F:
            snprintf(buf, nbuf, "%u\\nKOP_CONST_F\\n%.6g", idx, (double)op->imm.f);
            return;
        case KOP_RANGE:
            snprintf(buf, nbuf, "%u\\nKOP_RANGE\\ntrip=%u", idx, op->imm.u);
            return;
        case KOP_ENDRANGE:
            snprintf(buf, nbuf, "%u\\nKOP_ENDRANGE\\nrange=#%u", idx, op->arg[0]);
            return;
        case KOP_LOAD:
            snprintf(buf, nbuf, "%u\\nKOP_LOAD\\nbuf=%u idx=#%u",
                     idx, op->imm.u, op->arg[0]);
            return;
        case KOP_STORE:
            snprintf(buf, nbuf, "%u\\nKOP_STORE\\nout idx=#%u\\nval=#%u",
                     idx, op->arg[0], op->arg[1]);
            return;
        case KOP_ALU:
            snprintf(buf, nbuf, "%u\\nKOP_ALU\\n%s aux=%u\\na=#%u b=#%u",
                     idx, op->imm.u < UOP_COUNT ? uop_names[op->imm.u] : "?",
                     op->arg[2], op->arg[0], op->arg[1]);
            return;
        case KOP_ACC_INIT:
            snprintf(buf, nbuf, "%u\\nKOP_ACC_INIT\\n%.6g", idx, (double)op->imm.f);
            return;
        case KOP_ACC:
            snprintf(buf, nbuf, "%u\\nKOP_ACC\\n%s\\nacc=#%u val=#%u",
                     idx, op->imm.u < UOP_COUNT ? uop_names[op->imm.u] : "?",
                     op->arg[0], op->arg[1]);
            return;
        case KOP_IDX:
            snprintf(buf, nbuf, "%u\\nKOP_IDX\\nbase=#%u add=#%u\\nscale=%u",
                     idx, op->arg[0], op->arg[1], op->imm.u);
            return;
        case KOP_MOD:
            snprintf(buf, nbuf, "%u\\nKOP_MOD\\na=#%u mod=%u",
                     idx, op->arg[0], op->imm.u);
            return;
        case KOP_DIV:
            snprintf(buf, nbuf, "%u\\nKOP_DIV\\na=#%u div=%u",
                     idx, op->arg[0], op->imm.u);
            return;
        case KOP_MASK:
            snprintf(buf, nbuf, "%u\\nKOP_MASK\\ncond=#%u val=#%u",
                     idx, op->arg[0], op->arg[1]);
            return;
        default:
            snprintf(buf, nbuf, "%u\\n%s", idx, lower_kop_name(op->type));
            return;
    }
}

static int lower_term_in_list(const Term *items, u32 count, Term t) {
    for (u32 i = 0; i < count; i++) {
        if (items[i] == t) return 1;
    }
    return 0;
}

static u32 lower_collect_terms(TinyHVM *ctx, Term root, Term *out_terms, u32 cap) {
    if (!ctx || !out_terms || cap == 0 || term_tag(root) != TAG_TOP) return 0;
    Term *stack = (Term *)calloc(cap, sizeof(Term));
    u8 *seen_loc = (u8 *)calloc((size_t)(ctx->lower_ctx.heap_pos ? ctx->lower_ctx.heap_pos : 1), 1);
    Term *seen_imm = (Term *)calloc(cap, sizeof(Term));
    u32 sp = 0, count = 0, n_imm = 0;
    stack[sp++] = root;
    while (sp > 0 && count < cap) {
        Term t = stack[--sp];
        if (term_tag(t) != TAG_TOP) continue;
        u32 arity = lower_term_arity(t);
        if (arity > 0) {
            u64 loc = term_val(t);
            if (loc == 0 || loc >= ctx->lower_ctx.heap_pos || seen_loc[loc]) continue;
            seen_loc[loc] = 1;
        } else {
            if (lower_term_in_list(seen_imm, n_imm, t)) continue;
            seen_imm[n_imm++] = t;
        }
        out_terms[count++] = t;
        if (arity == 0) continue;
        u64 loc = term_val(t);
        for (u32 i = 0; i < arity && sp < cap; i++)
            stack[sp++] = lower_heap_read(ctx, loc + i);
    }
    free(seen_imm);
    free(seen_loc);
    free(stack);
    return count;
}

static void lower_dump_manifest(TinyHVM *ctx, const char *path,
                                const KernelEntry *ke, const UOpKernel *uk,
                                u32 kid, u64 kernel_loc, u64 cache_sig, u64 lower_sig,
                                Backend *backend, u32 out_buf,
                                const u32 *leaf_bufs, u32 n_leaf_bufs,
                                u32 step_index) {
    if (!ctx || !path || !ke || !uk) return;
    FILE *f = fopen(path, "w");
    if (!f) return;
    char full_shape[96], raw_shape[96], out_shape[96], reduce_axes[96], result_view[256];
    const char *interaction = thvm_step_graph_is_active() ? thvm_step_graph_last_name() : "";
    lower_shape_str(&ke->full_shape, full_shape, sizeof(full_shape));
    lower_shape_str(&ke->raw_out_shape, raw_shape, sizeof(raw_shape));
    lower_shape_str(&ke->out_shape, out_shape, sizeof(out_shape));
    lower_axes_str(ke->has_reduce ? &ke->reduce : NULL, uk->rank, reduce_axes, sizeof(reduce_axes));
    if (ke->has_result_view) lower_view_str(&ke->result_view, result_view, sizeof(result_view));
    else snprintf(result_view, sizeof(result_view), "none");
    fprintf(f,
            "step_index=%u\ninteraction=%s\nkid=%u\nkernel_loc=%llu\nbackend=%s\ncache_sig=0x%016llx\n"
            "lower_sig=0x%016llx\nlower_heap=%llu\nrewrites=%u\n"
            "full_shape=%s\nraw_out_shape=%s\nout_shape=%s\nresult_view=%s\n"
            "output_slot=%u\nraw_output_tid=%u\noutput_tid=%u\nout_buf=%u\n"
            "grid=[%u,%u,%u]\ntg=[%u,%u,%u]\nlocal_size=%u\nreduce_numel=%u\n"
            "n_leaves=%u\nn_ops=%u\nreduce_type=%s\nreduce_axes=%s\n"
            "dispatch_mode=%s/uop\n",
            step_index, interaction && interaction[0] ? interaction : "none",
            kid, (unsigned long long)kernel_loc,
            lower_backend_short(ctx, backend),
            (unsigned long long)cache_sig,
            (unsigned long long)lower_sig,
            (unsigned long long)ctx->lower_ctx.heap_pos,
            ctx->lower_ctx.rewrite_count,
            full_shape, raw_shape, out_shape, result_view,
            ke->output_slot, ke->raw_output_tid, ke->output_tid, out_buf,
            uk->grid[0], uk->grid[1], uk->grid[2],
            uk->tg[0], uk->tg[1], uk->tg[2],
            uk->local_size, uk->reduce_numel,
            ke->n_leaves, ke->n_ops,
            ke->has_reduce ? uop_names[ke->reduce.reduce_type] : "NONE",
            reduce_axes,
            lower_backend_short(ctx, backend));
    fprintf(f, "\nfused_ops:\n");
    for (u32 i = 0; i < ke->n_ops; i++) {
        fprintf(f, "  %u: %s a=%u b=%u aux=%u\n",
                i,
                ke->ops[i].uop < UOP_COUNT ? uop_names[ke->ops[i].uop] : "?",
                ke->ops[i].arg_a, ke->ops[i].arg_b, ke->ops[i].aux);
    }
    fprintf(f, "\nleaves:\n");
    for (u32 i = 0; i < ke->n_leaves; i++) {
        char view[256];
        lower_view_str(&ke->leaf_views[i], view, sizeof(view));
        if (ke->leaf_kinds[i] == KERNEL_LEAF_TENSOR && ke->leaf_ids[i] < ctx->tensor_count) {
            TensorMeta *m = &ctx->tensors[ke->leaf_ids[i]];
            fprintf(f,
                    "  %u: tensor t%u dispatch_buf=%u tensor_buf=%u slot=%u dtype=%s st_views=%u %s\n",
                    i, ke->leaf_ids[i],
                    i < n_leaf_bufs ? leaf_bufs[i] : 0,
                    m->buf_id, m->planned_slot, dtype_name(ke->leaf_dtypes[i]),
                    ke->leaf_sts[i].n_views, view);
        } else if (ke->leaf_kinds[i] == KERNEL_LEAF_NUM) {
            fprintf(f,
                    "  %u: const %.6g dispatch_buf=%u dtype=%s %s\n",
                    i, (double)ke->leaf_nums[i],
                    i < n_leaf_bufs ? leaf_bufs[i] : 0,
                    dtype_name(ke->leaf_dtypes[i]), view);
        } else {
            fprintf(f, "  %u: leaf dispatch_buf=%u dtype=%s %s\n",
                    i, i < n_leaf_bufs ? leaf_bufs[i] : 0,
                    dtype_name(ke->leaf_dtypes[i]), view);
        }
    }
    fprintf(f, "\nuop_ops:\n");
    for (u32 i = 0; i < uk->n_ops; i++) {
        char label[256];
        lower_kop_label(uk, i, label, sizeof(label));
        fprintf(f, "  %s\n", label);
    }
    fclose(f);
}

static int lower_file_exists(const char *path) {
    if (!path || !path[0]) return 0;
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static char *lower_read_file(const char *path, size_t *out_len) {
    if (out_len) *out_len = 0;
    if (!path || !path[0]) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long len = ftell(f);
    if (len < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[got] = '\0';
    if (out_len) *out_len = got;
    return buf;
}

static size_t lower_trim_graph_footer_len(const char *buf, size_t len) {
    if (!buf) return 0;
    while (len > 0) {
        char c = buf[len - 1];
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t')
            len--;
        else
            break;
    }
    if (len > 0 && buf[len - 1] == '}')
        len--;
    while (len > 0) {
        char c = buf[len - 1];
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t')
            len--;
        else
            break;
    }
    return len;
}

static void lower_prefixed_name(char *dst, size_t nbuf, const char *prefix, const char *base) {
    if (!dst || nbuf == 0) return;
    if (prefix && prefix[0])
        snprintf(dst, nbuf, "%s_%s", prefix, base);
    else
        snprintf(dst, nbuf, "%s", base);
}

static void lower_write_dot_header(FILE *f) {
    if (!f) return;
    fprintf(f, "digraph Lowered {\n");
    fprintf(f, "  graph [rankdir=LR, nodesep=0.35, ranksep=0.55];\n");
    fprintf(f, "  node [shape=box, style=filled, fontname=\"Courier\", fontsize=10, margin=\"0.08,0.05\"];\n");
    fprintf(f, "  edge [fontname=\"Courier\", fontsize=8];\n\n");
}

static void lower_write_dot_footer(FILE *f) {
    if (!f) return;
    fprintf(f, "}\n");
}

static void lower_write_dot_section(FILE *f, const char *prefix,
                                    TinyHVM *ctx, const KernelEntry *ke, const UOpKernel *uk,
                                    u32 kid, u64 kernel_loc, u64 cache_sig, u64 lower_sig,
                                    Backend *backend, u32 out_buf,
                                    const u32 *leaf_bufs, u32 n_leaf_bufs,
                                    u32 step_index) {
    if (!f || !ctx || !ke || !uk) return;
    char full_shape[96], raw_shape[96], out_shape[96], reduce_axes[96], result_view[256];
    const char *interaction = thvm_step_graph_is_active() ? thvm_step_graph_last_name() : "";
    u32 root_uop = (term_tag(ke->original_term) == TAG_TOP) ? term_ext(ke->original_term) : UOP_COUNT;
    const char *root_name = (root_uop < UOP_COUNT) ? uop_names[root_uop] : "KERNEL";
    char section_label[96];
    snprintf(section_label, sizeof(section_label), "kid=%u %s", kid, root_name);
    lower_shape_str(&ke->full_shape, full_shape, sizeof(full_shape));
    lower_shape_str(&ke->raw_out_shape, raw_shape, sizeof(raw_shape));
    lower_shape_str(&ke->out_shape, out_shape, sizeof(out_shape));
    lower_axes_str(ke->has_reduce ? &ke->reduce : NULL, uk->rank, reduce_axes, sizeof(reduce_axes));
    if (ke->has_result_view) lower_view_str(&ke->result_view, result_view, sizeof(result_view));
    else snprintf(result_view, sizeof(result_view), "none");

    char cluster_meta[64], cluster_fused[64], cluster_mem[64], cluster_lower[64], cluster_uop[64];
    char meta0_id[64], meta1_id[64], meta2_id[64], mem_out_id[64];
    lower_prefixed_name(cluster_meta, sizeof(cluster_meta), prefix, "cluster_meta");
    lower_prefixed_name(cluster_fused, sizeof(cluster_fused), prefix, "cluster_fused");
    lower_prefixed_name(cluster_mem, sizeof(cluster_mem), prefix, "cluster_mem");
    lower_prefixed_name(cluster_lower, sizeof(cluster_lower), prefix, "cluster_lower");
    lower_prefixed_name(cluster_uop, sizeof(cluster_uop), prefix, "cluster_uop");
    lower_prefixed_name(meta0_id, sizeof(meta0_id), prefix, "meta0");
    lower_prefixed_name(meta1_id, sizeof(meta1_id), prefix, "meta1");
    lower_prefixed_name(meta2_id, sizeof(meta2_id), prefix, "meta2");
    lower_prefixed_name(mem_out_id, sizeof(mem_out_id), prefix, "mem_out");

    fprintf(f, "  subgraph %s {\n", cluster_meta);
    fprintf(f, "    label=\"Kernel Trace %s\";\n", section_label);
    fprintf(f, "    color=\"#bbbbbb\";\n");
    fprintf(f, "    %s [shape=note, fillcolor=\"#f8f8f8\", label=\"step=%03u\\ninteraction=%s\\nkid=%u\\nkernel_loc=@%llu\\nbackend=%s\\ncache_sig=0x%016llx\\nlower_sig=0x%016llx\\nlower_heap=%llu\\nrewrites=%u\"];\n",
            meta0_id,
            step_index, interaction && interaction[0] ? interaction : "none",
            kid, (unsigned long long)kernel_loc, lower_backend_short(ctx, backend),
            (unsigned long long)cache_sig, (unsigned long long)lower_sig,
            (unsigned long long)ctx->lower_ctx.heap_pos, ctx->lower_ctx.rewrite_count);
    fprintf(f, "    %s [shape=note, fillcolor=\"#f8f8f8\", label=\"full=%s\\nraw_out=%s\\nout=%s\\nresult_view=%s\"];\n",
            meta1_id, full_shape, raw_shape, out_shape, result_view);
    fprintf(f, "    %s [shape=note, fillcolor=\"#f8f8f8\", label=\"reduce=%s axes=%s\\ngrid=[%u,%u,%u]\\ntg=[%u,%u,%u]\\nlocal=%u reduce_numel=%u\\nout_buf=%u slot=%u\"];\n",
            meta2_id,
            ke->has_reduce ? uop_names[ke->reduce.reduce_type] : "NONE",
            reduce_axes,
            uk->grid[0], uk->grid[1], uk->grid[2],
            uk->tg[0], uk->tg[1], uk->tg[2],
            uk->local_size, uk->reduce_numel, out_buf, ke->output_slot);
    fprintf(f, "  }\n\n");

    fprintf(f, "  subgraph %s {\n", cluster_fused);
    fprintf(f, "    label=\"Fused Ops %s\";\n", section_label);
    fprintf(f, "    color=\"#bbbbbb\";\n");
    for (u32 i = 0; i < ke->n_ops; i++) {
        char fop_id[64], prev_fop_id[64];
        char base_id[32];
        snprintf(base_id, sizeof(base_id), "fop%u", i);
        lower_prefixed_name(fop_id, sizeof(fop_id), prefix, base_id);
        fprintf(f, "    %s [fillcolor=\"#dbe8ff\", label=\"%u\\n%s\\na=%u b=%u aux=%u\"];\n",
                fop_id, i,
                ke->ops[i].uop < UOP_COUNT ? uop_names[ke->ops[i].uop] : "?",
                ke->ops[i].arg_a, ke->ops[i].arg_b, ke->ops[i].aux);
        if (i > 0) {
            snprintf(base_id, sizeof(base_id), "fop%u", i - 1);
            lower_prefixed_name(prev_fop_id, sizeof(prev_fop_id), prefix, base_id);
            fprintf(f, "    %s -> %s [style=dotted, arrowhead=none, color=\"#888888\"];\n",
                    prev_fop_id, fop_id);
        }
    }
    fprintf(f, "  }\n\n");

    fprintf(f, "  subgraph %s {\n", cluster_mem);
    fprintf(f, "    label=\"Memory Plan %s\";\n", section_label);
    fprintf(f, "    color=\"#bbbbbb\";\n");
    fprintf(f, "    %s [shape=box3d, fillcolor=\"#d9f2e6\", label=\"out\\nbuf=%u\\nslot=%u\\nraw_tid=%u\\nout_tid=%u\\ndtype=%s\"];\n",
            mem_out_id, out_buf, ke->output_slot, ke->raw_output_tid, ke->output_tid, dtype_name(uk->out_dtype));
    for (u32 i = 0; i < ke->n_leaves; i++) {
        char view[256], mem_leaf_id[64], base_id[32];
        snprintf(base_id, sizeof(base_id), "mem_leaf%u", i);
        lower_prefixed_name(mem_leaf_id, sizeof(mem_leaf_id), prefix, base_id);
        lower_view_str(&ke->leaf_views[i], view, sizeof(view));
        if (ke->leaf_kinds[i] == KERNEL_LEAF_TENSOR && ke->leaf_ids[i] < ctx->tensor_count) {
            TensorMeta *m = &ctx->tensors[ke->leaf_ids[i]];
            fprintf(f, "    %s [fillcolor=\"#eef5ff\", label=\"leaf%u t%u\\ndispatch_buf=%u\\ntensor_buf=%u\\nslot=%u\\ndtype=%s\\nst_views=%u\\n%s\"];\n",
                    mem_leaf_id, i, ke->leaf_ids[i],
                    i < n_leaf_bufs ? leaf_bufs[i] : 0,
                    m->buf_id, m->planned_slot, dtype_name(ke->leaf_dtypes[i]),
                    ke->leaf_sts[i].n_views, view);
        } else if (ke->leaf_kinds[i] == KERNEL_LEAF_NUM) {
            fprintf(f, "    %s [fillcolor=\"#fff2cc\", label=\"leaf%u const\\n%.6g\\ndispatch_buf=%u\\ndtype=%s\\n%s\"];\n",
                    mem_leaf_id, i, (double)ke->leaf_nums[i],
                    i < n_leaf_bufs ? leaf_bufs[i] : 0,
                    dtype_name(ke->leaf_dtypes[i]), view);
        } else {
            fprintf(f, "    %s [fillcolor=\"#f0f0f0\", label=\"leaf%u\\ndispatch_buf=%u\\ndtype=%s\\n%s\"];\n",
                    mem_leaf_id, i, i < n_leaf_bufs ? leaf_bufs[i] : 0,
                    dtype_name(ke->leaf_dtypes[i]), view);
        }
    }
    fprintf(f, "  }\n\n");

    fprintf(f, "  subgraph %s {\n", cluster_lower);
    fprintf(f, "    label=\"LowerCtx IC %s\";\n", section_label);
    fprintf(f, "    color=\"#bbbbbb\";\n");
    u32 cap = (u32)(ctx->lower_ctx.heap_pos + 64);
    Term *terms = (Term *)calloc(cap ? cap : 1, sizeof(Term));
    u32 n_terms = lower_collect_terms(ctx, ctx->lower_ctx.root, terms, cap ? cap : 1);
    for (u32 i = 0; i < n_terms; i++) {
        char base_id[64], id[96], label[256];
        lower_term_id(terms[i], base_id, sizeof(base_id));
        lower_prefixed_name(id, sizeof(id), prefix, base_id);
        lower_term_label(ctx, terms[i], label, sizeof(label));
        fprintf(f, "    %s [fillcolor=\"%s\", label=\"%s\"%s];\n",
                id, lower_dot_fill(terms[i]), label,
                terms[i] == ctx->lower_ctx.root ? ",color=\"#cc0000\",penwidth=2.0" : "");
    }
    for (u32 i = 0; i < n_terms; i++) {
        Term t = terms[i];
        u32 arity = lower_term_arity(t);
        if (arity == 0) continue;
        u64 loc = term_val(t);
        char base_dst[64], dst[96];
        lower_term_id(t, base_dst, sizeof(base_dst));
        lower_prefixed_name(dst, sizeof(dst), prefix, base_dst);
        for (u32 ai = 0; ai < arity; ai++) {
            Term child = lower_heap_read(ctx, loc + ai);
            if (term_tag(child) != TAG_TOP) continue;
            char base_src[64], src[96];
            lower_term_id(child, base_src, sizeof(base_src));
            lower_prefixed_name(src, sizeof(src), prefix, base_src);
            fprintf(f, "    %s -> %s [label=\"%s\"];\n",
                    src, dst, lower_lop_port_name(term_ext(t), ai));
        }
        if (term_ext(t) == LOP_LOAD) {
            u32 buf_idx = 0;
            lower_is_const_u(lower_heap_read(ctx, loc + 1), &buf_idx);
            if (buf_idx > 0) {
                char mem_leaf_id[64], base_id[32];
                snprintf(base_id, sizeof(base_id), "mem_leaf%u", buf_idx - 1);
                lower_prefixed_name(mem_leaf_id, sizeof(mem_leaf_id), prefix, base_id);
                fprintf(f, "    %s -> %s [style=dashed, color=\"#888888\", label=\"buf\"];\n",
                        mem_leaf_id, dst);
            }
        } else if (term_ext(t) == LOP_STORE) {
            fprintf(f, "    %s -> %s [style=dashed, color=\"#888888\", label=\"store\"];\n",
                    dst, mem_out_id);
        }
    }
    fprintf(f, "  }\n\n");
    free(terms);

    fprintf(f, "  subgraph %s {\n", cluster_uop);
    fprintf(f, "    label=\"Emitted UOpKernel / KOP %s\";\n", section_label);
    fprintf(f, "    color=\"#bbbbbb\";\n");
    for (u32 i = 0; i < uk->n_ops; i++) {
        char label[256], k_id[64], prev_k_id[64], base_id[32];
        snprintf(base_id, sizeof(base_id), "k%u", i);
        lower_prefixed_name(k_id, sizeof(k_id), prefix, base_id);
        lower_kop_label(uk, i, label, sizeof(label));
        fprintf(f, "    %s [fillcolor=\"%s\", label=\"%s\"];\n",
                k_id, lower_kop_fill(&uk->ops[i]), label);
        if (i > 0) {
            snprintf(base_id, sizeof(base_id), "k%u", i - 1);
            lower_prefixed_name(prev_k_id, sizeof(prev_k_id), prefix, base_id);
            fprintf(f, "    %s -> %s [style=dotted, arrowhead=none, color=\"#888888\"];\n",
                    prev_k_id, k_id);
        }
    }
    for (u32 i = 0; i < uk->n_ops; i++) {
        const KOp *op = &uk->ops[i];
        char k_id[64], base_id[32];
        snprintf(base_id, sizeof(base_id), "k%u", i);
        lower_prefixed_name(k_id, sizeof(k_id), prefix, base_id);
        for (u32 ai = 0; ai < 3; ai++) {
            const char *port = lower_kop_port_name(op, ai);
            if (!port[0]) continue;
            if (op->arg[ai] >= uk->n_ops) continue;
            char src_id[64];
            snprintf(base_id, sizeof(base_id), "k%u", op->arg[ai]);
            lower_prefixed_name(src_id, sizeof(src_id), prefix, base_id);
            fprintf(f, "    %s -> %s [label=\"%s\"];\n", src_id, k_id, port);
        }
        if (op->type == KOP_LOAD && op->imm.u > 0 && op->imm.u - 1 < ke->n_leaves) {
            char mem_leaf_id[64];
            snprintf(base_id, sizeof(base_id), "mem_leaf%u", op->imm.u - 1);
            lower_prefixed_name(mem_leaf_id, sizeof(mem_leaf_id), prefix, base_id);
            fprintf(f, "    %s -> %s [style=dashed, color=\"#888888\", label=\"buf\"];\n",
                    mem_leaf_id, k_id);
        } else if (op->type == KOP_STORE) {
            fprintf(f, "    %s -> %s [style=dashed, color=\"#888888\", label=\"store\"];\n",
                    k_id, mem_out_id);
        }
    }
    fprintf(f, "  }\n\n");
}

static void lower_dump_dot(TinyHVM *ctx, const char *path,
                           const KernelEntry *ke, const UOpKernel *uk,
                           u32 kid, u64 kernel_loc, u64 cache_sig, u64 lower_sig,
                           Backend *backend, u32 out_buf,
                           const u32 *leaf_bufs, u32 n_leaf_bufs,
                           u32 step_index) {
    if (!ctx || !path || !ke || !uk) return;
    FILE *f = fopen(path, "w");
    if (!f) return;
    lower_write_dot_header(f);
    lower_write_dot_section(f, "", ctx, ke, uk, kid, kernel_loc, cache_sig, lower_sig,
                            backend, out_buf, leaf_bufs, n_leaf_bufs, step_index);
    lower_write_dot_footer(f);
    fclose(f);
}

static void lower_append_shared_dot(TinyHVM *ctx, const char *path,
                                    const KernelEntry *ke, const UOpKernel *uk,
                                    u32 kid, u64 kernel_loc, u64 cache_sig, u64 lower_sig,
                                    Backend *backend, u32 out_buf,
                                    const u32 *leaf_bufs, u32 n_leaf_bufs,
                                    u32 step_index) {
    if (!ctx || !path || !ke || !uk) return;
    size_t existing_len = 0;
    char *existing = lower_read_file(path, &existing_len);
    FILE *f = fopen(path, "w");
    if (!f) {
        free(existing);
        return;
    }
    if (!existing || existing_len == 0) {
        lower_write_dot_header(f);
    } else {
        size_t body_len = lower_trim_graph_footer_len(existing, existing_len);
        if (body_len > 0)
            fwrite(existing, 1, body_len, f);
        fprintf(f, "\n\n");
    }
    char prefix[32];
    snprintf(prefix, sizeof(prefix), "kid%u", kid);
    lower_write_dot_section(f, prefix, ctx, ke, uk, kid, kernel_loc, cache_sig, lower_sig,
                            backend, out_buf, leaf_bufs, n_leaf_bufs, step_index);
    lower_write_dot_footer(f);
    fclose(f);
    free(existing);
}

static void thvm_lower_dump_graphs(TinyHVM *ctx, const KernelEntry *ke, const UOpKernel *uk,
                                   u32 kid, u64 kernel_loc, u64 cache_sig, u64 lower_sig,
                                   Backend *backend, u32 out_buf,
                                   const u32 *leaf_bufs, u32 n_leaf_bufs) {
    if (!ctx || !ke || !uk || !lower_env_enabled("THVM_LOWER_GRAPH")) return;
    const char *anchor_name = thvm_step_graph_lower_anchor_name();
    u32 anchor_index = thvm_step_graph_lower_anchor_index();
    char base_dir[512];
    char dir[512];
    const char *dir_env = getenv("THVM_LOWER_GRAPH_DIR");
    if (dir_env && dir_env[0]) {
        snprintf(base_dir, sizeof(base_dir), "%s", dir_env);
    } else if (thvm_step_graph_is_active() ||
               (getenv("THVM_STEP_GRAPH") && getenv("THVM_STEP_GRAPH_DIR"))) {
        snprintf(base_dir, sizeof(base_dir), "%s_lower", thvm_step_graph_dir());
    } else {
        snprintf(base_dir, sizeof(base_dir), "graphs/lower");
    }
    if (ctx->step_graph_settled_replay) {
        snprintf(dir, sizeof(dir), "%s/settled", base_dir);
    } else {
        snprintf(dir, sizeof(dir), "%s", base_dir);
    }
    {
        static char announced_dir[1024];
        if (strcmp(announced_dir, base_dir) != 0) {
            snprintf(announced_dir, sizeof(announced_dir), "%s", base_dir);
            fprintf(stderr, "THVM lower graph dump -> %s/\n", base_dir);
        }
    }
    char mkdir_cmd[1024];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p \"%s\"", dir);
    system(mkdir_cmd);

    u32 step_index = thvm_step_graph_is_active() ? thvm_step_graph_current_index()
                    : (anchor_name && anchor_name[0]) ? anchor_index
                    : kid;
    u32 root_uop = (term_tag(ke->original_term) == TAG_TOP) ? term_ext(ke->original_term) : UOP_COUNT;
    const char *root_name = (root_uop < UOP_COUNT) ? uop_names[root_uop] : "KERNEL";
    const char *interaction = thvm_step_graph_is_active() ? thvm_step_graph_last_name()
                             : (anchor_name && anchor_name[0]) ? anchor_name
                             : "";
    int share_settled_graph = ctx->step_graph_settled_replay ? 1 : 0;
    char bundle_stem[256];
    char graph_stem[256];
    if (interaction && interaction[0]) {
        snprintf(bundle_stem, sizeof(bundle_stem), "step_%03u_%s_kid%u",
                 step_index, interaction, kid);
        snprintf(graph_stem, sizeof(graph_stem), "step_%03u_%s",
                 step_index, interaction);
    } else if (thvm_step_graph_is_active() ||
               (getenv("THVM_STEP_GRAPH") && getenv("THVM_STEP_GRAPH_DIR"))) {
        snprintf(bundle_stem, sizeof(bundle_stem), "step_%03u_KERNEL_h%llu_%s_kid%u",
                 step_index, (unsigned long long)kernel_loc, root_name, kid);
        snprintf(graph_stem, sizeof(graph_stem), "step_%03u_KERNEL_h%llu_%s",
                 step_index, (unsigned long long)kernel_loc, root_name);
    } else {
        snprintf(bundle_stem, sizeof(bundle_stem), "kernel_%03u_h%llu_%s_kid%u",
                 step_index, (unsigned long long)kernel_loc, root_name, kid);
        snprintf(graph_stem, sizeof(graph_stem), "kernel_%03u_h%llu_%s",
                 step_index, (unsigned long long)kernel_loc, root_name);
    }

    char dot_path[768], txt_path[768], msl_path[768], png_path[768];
    snprintf(dot_path, sizeof(dot_path), "%s/%s.dot", dir,
             share_settled_graph ? graph_stem : bundle_stem);
    snprintf(txt_path, sizeof(txt_path), "%s/%s.txt", dir, bundle_stem);
    snprintf(msl_path, sizeof(msl_path), "%s/%s.msl", dir, bundle_stem);
    snprintf(png_path, sizeof(png_path), "%s/%s.png", dir,
             share_settled_graph ? graph_stem : bundle_stem);
    if (share_settled_graph) {
        lower_append_shared_dot(ctx, dot_path, ke, uk, kid, kernel_loc, cache_sig, lower_sig,
                                backend, out_buf, leaf_bufs, n_leaf_bufs, step_index);
    } else {
        lower_dump_dot(ctx, dot_path, ke, uk, kid, kernel_loc, cache_sig, lower_sig,
                       backend, out_buf, leaf_bufs, n_leaf_bufs, step_index);
    }
    lower_dump_manifest(ctx, txt_path, ke, uk, kid, kernel_loc, cache_sig, lower_sig,
                        backend, out_buf, leaf_bufs, n_leaf_bufs, step_index);
#ifdef __OBJC__
    if (backend && backend == ctx->backends[THVM_DEV_METAL]) {
        char *src = thvm_metal_render_uop_kernel_source(uk);
        if (src) {
            FILE *mf = fopen(msl_path, "w");
            if (mf) {
                fputs(src, mf);
                fclose(mf);
            }
            free(src);
        }
    }
#endif
    if (!lower_env_enabled("THVM_LOWER_NO_PNG") && !getenv("THVM_STEP_GRAPH_NO_PNG")) {
        char png_cmd[1600];
        snprintf(png_cmd, sizeof(png_cmd),
                 "dot -Tpng -Gdpi=150 \"%s\" -o \"%s\" 2>/dev/null",
                 dot_path, png_path);
        system(png_cmd);
    }
}
