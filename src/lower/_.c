// lower/_.c — private lowering IC arena for structural KERNEL dispatch

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
