// fuse/_.c — Elementwise fusion: walk lazy TAG_TOP tree, fuse into one kernel
//
// The fuser collects elementwise ops and their leaf tensors into a single
// codegen kernel. View ops (RESHAPE, PERMUTE, EXPAND) are transparent —
// they compose into leaf index expressions. Non-elementwise TAG_TOPs
// (SUM, MM) become lazy leaf boundaries with shapes from the shape table.

static int is_view_op(u32 uop) {
    return uop == UOP_RESHAPE || uop == UOP_PERMUTE || uop == UOP_EXPAND ||
           uop == UOP_SHRINK || uop == UOP_PAD;
}
static int is_elementwise(u32 uop) {
    return uop==UOP_ADD||uop==UOP_SUB||uop==UOP_MUL||uop==UOP_DIV||
           uop==UOP_MAX||uop==UOP_CMP||uop==UOP_NEG||uop==UOP_RELU||
           uop==UOP_EXP||uop==UOP_LOG||uop==UOP_SQRT||uop==UOP_CAST;
}
static int is_binary(u32 uop) {
    return uop==UOP_ADD||uop==UOP_SUB||uop==UOP_MUL||uop==UOP_DIV||
           uop==UOP_MAX||uop==UOP_CMP;
}

// FUSE_MAX_OPS and FUSE_MAX_LEAVES defined in tinyhvm.h

// Per-walk storage
#define WALK_LEAF_BASE 10000
static View fuse_composed_views[FUSE_MAX_LEAVES];
static ShapeTracker fuse_leaf_sts[FUSE_MAX_LEAVES];
static u8 fuse_leaf_kinds[FUSE_MAX_LEAVES];
static f32 fuse_leaf_nums[FUSE_MAX_LEAVES];
static u32 fuse_leaf_dtypes[FUSE_MAX_LEAVES];

// Current scheduler-selected realization boundaries.
static const u64 *fuse_boundary_locs = NULL;
static const u32 *fuse_boundary_output_tids = NULL;
static const u32 *fuse_boundary_kids = NULL;
static u32        fuse_boundary_count = 0;
static u64        fuse_root_compute_loc = 0;
static u32        fuse_dep_kids[FUSE_MAX_LEAVES];
static u32        fuse_n_dep_kids = 0;

static void fuse_set_schedule_boundaries(const u64 *locs,
                                         const u32 *output_tids,
                                         const u32 *kids,
                                         u32 n_boundaries,
                                         u64 root_compute_loc) {
    fuse_boundary_locs = locs;
    fuse_boundary_output_tids = output_tids;
    fuse_boundary_kids = kids;
    fuse_boundary_count = n_boundaries;
    fuse_root_compute_loc = root_compute_loc;
    fuse_n_dep_kids = 0;
}

static void fuse_clear_schedule_boundaries(void) {
    fuse_boundary_locs = NULL;
    fuse_boundary_output_tids = NULL;
    fuse_boundary_kids = NULL;
    fuse_boundary_count = 0;
    fuse_root_compute_loc = 0;
    fuse_n_dep_kids = 0;
}

static int fuse_boundary_index_for_loc(u64 loc) {
    if (!fuse_boundary_locs || loc == 0) return -1;
    for (u32 i = 0; i < fuse_boundary_count; i++)
        if (fuse_boundary_locs[i] == loc) return (int)i;
    return -1;
}

static void fuse_dep_add(u32 kid) {
    if (kid >= SCHED_MAX_KERNELS) return;
    for (u32 i = 0; i < fuse_n_dep_kids; i++)
        if (fuse_dep_kids[i] == kid) return;
    if (fuse_n_dep_kids < FUSE_MAX_LEAVES)
        fuse_dep_kids[fuse_n_dep_kids++] = kid;
}

static u32 fuse_cast_aux_from_term(TinyHVM *ctx, u64 loc) {
    Term dt = heap_read(ctx, loc + 1);
    if (term_tag(dt) != TAG_TEN) return DTYPE_F32;
    u32 tid = (u32)term_val(dt);
    u32 raw[MAX_DIM];
    if (tensor_meta_read_u32(ctx, tid, raw, MAX_DIM) == 1 && raw[0] < DTYPE_COUNT)
        return raw[0];
    return DTYPE_F32;
}

static int fuse_append_leaf_tensor(TinyHVM *ctx, u32 tid,
                                   const View *view,
                                   const ShapeTracker *st,
                                   u32 *leaf_ids, const View **leaf_views, u32 *n_leaves) {
    if (!view || *n_leaves >= FUSE_MAX_LEAVES) return -1;
    u32 idx = (*n_leaves)++;
    leaf_ids[idx] = tid;
    fuse_leaf_kinds[idx] = KERNEL_LEAF_TENSOR;
    fuse_leaf_nums[idx] = 0.0f;
    fuse_leaf_dtypes[idx] = ctx->tensors[tid].dtype;
    fuse_composed_views[idx] = *view;
    leaf_views[idx] = &fuse_composed_views[idx];
    fuse_leaf_sts[idx] = st ? *st : st_from_view(*view);
    return (int)(WALK_LEAF_BASE + idx);
}

static int fuse_append_leaf_num(f32 val, u32 dtype,
                                const View *view,
                                const ShapeTracker *st,
                                u32 *leaf_ids, const View **leaf_views, u32 *n_leaves) {
    View scalar = view ? *view : view_create((Shape){.dims={1}, .rank=1});
    if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
    u32 idx = (*n_leaves)++;
    leaf_ids[idx] = 0;
    fuse_leaf_kinds[idx] = KERNEL_LEAF_NUM;
    fuse_leaf_nums[idx] = val;
    fuse_leaf_dtypes[idx] = dtype;
    fuse_composed_views[idx] = scalar;
    leaf_views[idx] = &fuse_composed_views[idx];
    fuse_leaf_sts[idx] = st ? *st : st_from_view(scalar);
    return (int)(WALK_LEAF_BASE + idx);
}

static Term fuse_deref_links(TinyHVM *ctx, Term t);

static int fuse_make_boundary_leaf(TinyHVM *ctx, Term t,
                                   const View *view,
                                   const ShapeTracker *st,
                                   u32 *leaf_ids, const View **leaf_views, u32 *n_leaves) {
    Term cur = t;
    for (int i = 0; i < 16; i++) {
        cur = fuse_deref_links(ctx, cur);
        if (term_tag(cur) == TAG_TOP && is_view_op(term_ext(cur))) {
            cur = heap_read(ctx, term_val(cur));
            continue;
        }
        break;
    }
    if (term_tag(cur) == TAG_TEN)
        return fuse_append_leaf_tensor(ctx, (u32)term_val(cur), view, st,
                                       leaf_ids, leaf_views, n_leaves);
    if (term_tag(cur) == TAG_NUM) {
        f32 num_val = term_ext(cur) == NUM_U32 ? (f32)term_as_u32(cur) : term_as_f32(cur);
        u32 num_dtype = term_ext(cur) == NUM_U32 ? DTYPE_U32 : DTYPE_F32;
        return fuse_append_leaf_num(num_val, num_dtype, view, st,
                                    leaf_ids, leaf_views, n_leaves);
    }
    if (term_tag(cur) == TAG_TOP) {
        int bidx = fuse_boundary_index_for_loc(term_val(cur));
        if (bidx >= 0) {
            fuse_dep_add(fuse_boundary_kids[bidx]);
            return fuse_append_leaf_tensor(ctx, fuse_boundary_output_tids[bidx], view, st,
                                           leaf_ids, leaf_views, n_leaves);
        }
    }
    if (getenv("THVM_SCHED_DIAG2")) {
        fprintf(stderr, "  boundary_leaf_fail: start_tag=%u start_ext=%u start_val=%llu cur_tag=%u cur_ext=%u cur_val=%llu\n",
                (u32)term_tag(t), (u32)term_ext(t), (unsigned long long)term_val(t),
                (u32)term_tag(cur), (u32)term_ext(cur), (unsigned long long)term_val(cur));
        if (term_tag(cur) == TAG_VAR) {
            Term sub = heap_read(ctx, term_val(cur));
            fprintf(stderr, "  boundary_leaf_var: loc=%llu sub_tag=%u sub_ext=%u sub_val=%llu sub_is_sub=%d\n",
                    (unsigned long long)term_val(cur),
                    (u32)term_tag(sub), (u32)term_ext(sub), (unsigned long long)term_val(sub),
                    term_is_sub(sub));
        }
    }
    return -1;
}

// Per-fuse state: absorbed TAG_TOP terms (for marking as FUSING after scheduling)
#define FUSE_MAX_ABSORBED 256
static Term fuse_absorbed[FUSE_MAX_ABSORBED];
static u32  fuse_n_absorbed = 0;
static void fuse_mark_absorbed(Term t) {
    if (fuse_n_absorbed < FUSE_MAX_ABSORBED) fuse_absorbed[fuse_n_absorbed++] = t;
}

// Shared absorbed terms registry (populated by sched_one, checked by fuse_walk_inner)
#define SCHED_ABSORBED_MAX 4096
static Term _sched_absorbed[SCHED_ABSORBED_MAX];
static u32  _sched_n_absorbed = 0;
static int _sched_is_absorbed(Term t) {
    for (u32 i = 0; i < _sched_n_absorbed; i++)
        if (_sched_absorbed[i] == t) return 1;
    return 0;
}

// Per-fuse state: PERMUTE between ew and reduce (set by fuse_walk_inner)
static f32 _fuse_perm[MAX_DIM];
static u32 _fuse_perm_rank = 0;
static int _fuse_has_perm = 0;

// Per-fuse: allow absorbing one unscheduled SUM/RMAX during walk
static int _fuse_can_absorb_reduce = 0;
static Term _fuse_absorbed_reduce;

// Per-fuse: DUP shared locations traversed during walk (for heap replacement)
#define FUSE_MAX_DUP_LOCS 64
static u64 _fuse_dup_locs[FUSE_MAX_DUP_LOCS];
static u32 _fuse_n_dup_locs = 0;

static Term fuse_deref_links(TinyHVM *ctx, Term t) {
    for (int i = 0; i < 16; i++) {
        if (term_tag(t) == TAG_DP0 || term_tag(t) == TAG_DP1) {
            t = heap_read(ctx, term_val(t));
            continue;
        }
        if (term_tag(t) == TAG_VAR) {
            Term sub = heap_read(ctx, term_val(t));
            if (!term_is_sub(sub)) {
                t = sub;
                continue;
            }
        }
        break;
    }
    return t;
}

// Compute a broadcast leaf's target shape in a reshaped iteration space.
// For each OP dim, consume RESHAPE dims until product matches, then:
//   leaf_dim==1 → all consumed dims become 1 (broadcast)
//   leaf_dim==op_dim → copy consumed dims from rs_shape
static int fuse_leaf_reshape_target(Shape op_shape, Shape rs_shape,
                                     Shape leaf_shape, Shape *out_shape) {
    *out_shape = (Shape){.rank = rs_shape.rank};
    u32 ri = 0;
    for (u32 oi = 0; oi < op_shape.rank && ri < rs_shape.rank; oi++) {
        u32 op_dim = op_shape.dims[oi];
        u32 leaf_dim = (oi < leaf_shape.rank) ? leaf_shape.dims[oi] : 1;
        u32 prod = 1, ri_start = ri;
        while (ri < rs_shape.rank && prod < op_dim) { prod *= rs_shape.dims[ri]; ri++; }
        if (prod != op_dim) return 0;
        if (leaf_dim == 1)
            for (u32 j = ri_start; j < ri; j++) out_shape->dims[j] = 1;
        else if (leaf_dim == op_dim)
            for (u32 j = ri_start; j < ri; j++) out_shape->dims[j] = rs_shape.dims[j];
        else return 0;
    }
    while (ri < rs_shape.rank) { out_shape->dims[ri] = 1; ri++; }
    // Verify output numel matches leaf numel
    u32 out_numel = 1;
    for (u32 d = 0; d < out_shape->rank; d++) out_numel *= out_shape->dims[d];
    u32 leaf_numel = 1;
    for (u32 d = 0; d < leaf_shape.rank; d++) leaf_numel *= leaf_shape.dims[d];
    if (out_numel != leaf_numel) return 0;
    return 1;
}

// ============================================================
// fuse_walk_inner: recursive walk of lazy TAG_TOP tree
// ============================================================
// Returns WALK_LEAF_BASE + leaf_idx for leaves,
//         WALK_LEAF_BASE*2 + op_idx for ops, or -1 on failure.

static int fuse_walk_inner(TinyHVM *ctx, Term t,
                           FusedOp *ops, u32 *n_ops,
                           u32 *leaf_ids, const View **leaf_views, u32 *n_leaves) {
    if (term_tag(t) == TAG_TOP) {
        int bidx = fuse_boundary_index_for_loc(term_val(t));
        if (bidx >= 0 && term_val(t) != fuse_root_compute_loc) {
            u32 out_tid = fuse_boundary_output_tids[bidx];
            if (out_tid == 0) return -1;
            ENSURE(ctx, out_tid);
            fuse_dep_add(fuse_boundary_kids[bidx]);
            return fuse_append_leaf_tensor(ctx, out_tid,
                                           tensor_view_get(&ctx->tensors[out_tid]),
                                           tensor_st_get(&ctx->tensors[out_tid]),
                                           leaf_ids, leaf_views, n_leaves);
        }
    }

    // TAG_TEN: concrete tensor leaf (or deferred ew to walk through)
    if (term_tag(t) == TAG_TEN) {
        u32 tid = (u32)term_val(t);
        TensorMeta *m = &ctx->tensors[tid];
        // Deferred ew tensor (buf_id=0): walk through as a fuse op.
        if (m->buf_id == 0 && m->creator_op && is_elementwise(m->creator_op) &&
            m->src_ids[0]) {
            if (*n_ops >= FUSE_MAX_OPS) return -1;
            int a_res = fuse_walk_inner(ctx, term_ten(m->src_ids[0], m->dtype),
                                        ops, n_ops, leaf_ids, leaf_views, n_leaves);
            if (a_res < 0) return -1;
            int b_res = a_res;
            if (is_binary(m->creator_op) && m->src_ids[1]) {
                b_res = fuse_walk_inner(ctx, term_ten(m->src_ids[1], m->dtype),
                                        ops, n_ops, leaf_ids, leaf_views, n_leaves);
                if (b_res < 0) return -1;
            }
            u32 oi = (*n_ops)++;
            ops[oi] = (FusedOp){
                .uop = m->creator_op,
                .arg_a = (u32)a_res,
                .arg_b = (u32)b_res,
                .aux = m->creator_op == UOP_CAST ? m->dtype : 0,
            };
            return (int)(WALK_LEAF_BASE * 2 + oi);
        }
        // Deferred view tensor (buf_id=0): walk through and compose view
        // onto the leaf (same as TAG_TOP view handling).
        if (m->buf_id == 0 && m->creator_op && is_view_op(m->creator_op) &&
            m->src_ids[0]) {
            int inner = fuse_walk_inner(ctx, term_ten(m->src_ids[0], m->dtype),
                                        ops, n_ops, leaf_ids, leaf_views, n_leaves);
            if (inner < 0) return -1;
            if (inner >= (int)(WALK_LEAF_BASE * 2)) {
                // Inner returned OP. Store PERMUTE perm for axes transform.
                if (m->creator_op == UOP_PERMUTE && m->src_ids[1] && !_fuse_has_perm) {
                    TensorMeta *ppm = &ctx->tensors[m->src_ids[1]];
                    _fuse_perm_rank = ppm->view.numel;
                    u32 perm[MAX_DIM];
                    tensor_meta_read_u32(ctx, m->src_ids[1], perm, MAX_DIM);
                    for (u32 j = 0; j < _fuse_perm_rank; j++) _fuse_perm[j] = (f32)perm[j];
                    _fuse_has_perm = 1;
                }
                return inner;
            }
            u32 leaf_idx = inner - WALK_LEAF_BASE;
            const View *base = leaf_views[leaf_idx];
            if (!base) return -1;
            // Read view params and compose
            if (m->creator_op == UOP_PERMUTE && m->src_ids[1]) {
                TensorMeta *mp = &ctx->tensors[m->src_ids[1]];
                u32 rank = mp->view.numel;
                u32 pf[MAX_DIM];
                tensor_meta_read_u32(ctx, m->src_ids[1], pf, MAX_DIM);
                View nv = {0}; nv.offset = base->offset; nv.shape.rank = rank; nv.numel = base->numel;
                for (u32 j = 0; j < rank; j++) {
                    nv.shape.dims[j] = base->shape.dims[pf[j]];
                    nv.strides[j] = base->strides[pf[j]];
                }
                nv.contiguous = 0;
                fuse_composed_views[leaf_idx] = nv;
                leaf_views[leaf_idx] = &fuse_composed_views[leaf_idx];
            } else if (m->creator_op == UOP_RESHAPE && m->src_ids[1]) {
                TensorMeta *mp = &ctx->tensors[m->src_ids[1]];
                u32 rank = mp->view.numel;
                u32 rf[MAX_DIM];
                tensor_meta_read_u32(ctx, m->src_ids[1], rf, MAX_DIM);
                Shape ns = {.rank = rank};
                for (u32 j = 0; j < rank; j++) ns.dims[j] = rf[j];
                View nv = view_reshape(*base, ns);
                if (nv.numel == 0) return -1; // can't merge — fail this walk
                fuse_composed_views[leaf_idx] = nv;
                leaf_views[leaf_idx] = &fuse_composed_views[leaf_idx];
            }
            // For EXPAND, SHRINK, PAD: similar composition (not implemented yet)
            return inner; // return the composed leaf
        }
        // Deferred non-ew leaf: materialize before use.
        if (m->buf_id == 0 && m->creator_op) {
            tensor_materialize(ctx, tid);
            m = &ctx->tensors[tid];
        }
        // Normal leaf (materialized buffer)
        for (u32 i = 0; i < *n_leaves; i++)
            if (fuse_leaf_kinds[i] == KERNEL_LEAF_TENSOR && leaf_ids[i] == tid)
                return (int)(WALK_LEAF_BASE + i);
        if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
        u32 idx = (*n_leaves)++;
        leaf_ids[idx] = tid;
        fuse_leaf_kinds[idx] = KERNEL_LEAF_TENSOR;
        fuse_leaf_nums[idx] = 0.0f;
        fuse_leaf_dtypes[idx] = ctx->tensors[tid].dtype;
        leaf_views[idx] = tensor_view_get(&ctx->tensors[tid]);
        { const ShapeTracker *tst = tensor_st_get(&ctx->tensors[tid]);
          fuse_leaf_sts[idx] = tst ? *tst : st_from_view(ctx->tensors[tid].view); }
        return (int)(WALK_LEAF_BASE + idx);
    }
    // DP0/DP1: look through to the shared value in the DUP node.
    // DUP means 2+ consumers → don't absorb reduces through DUP
    // (they should be standalone kernels shared by both consumers).
    if (term_tag(t) == TAG_DP0 || term_tag(t) == TAG_DP1) {
        Term shared = heap_read(ctx, term_val(t));
        int saved_absorb = _fuse_can_absorb_reduce;
        _fuse_can_absorb_reduce = 0;
        int r = fuse_walk_inner(ctx, shared, ops, n_ops, leaf_ids, leaf_views, n_leaves);
        _fuse_can_absorb_reduce = saved_absorb;
        if (r < 0 && getenv("THVM_SCHED_DIAG"))
            fprintf(stderr, "  dp_walk_fail: dp%u@%llu → tag%u/%s\n",
                    (term_tag(t)==TAG_DP1)?1:0, term_val(t),
                    term_tag(shared),
                    term_tag(shared)==TAG_TOP ? uop_names[term_ext(shared)] : "");
        return r;
    }
    // TAG_VAR: if substituted, walk through the bound value.
    if (term_tag(t) == TAG_VAR) {
        Term sub = heap_read(ctx, term_val(t));
        if (!term_is_sub(sub))
            return fuse_walk_inner(ctx, sub, ops, n_ops, leaf_ids, leaf_views, n_leaves);
        if (getenv("THVM_SCHED_DIAG"))
            fprintf(stderr, "  walk_var_unbound: loc=%llu\n", (unsigned long long)term_val(t));
        return -1;
    }
    // TAG_NUM: scalar constant leaf. Keep it explicit in the kernel IR.
    if (term_tag(t) == TAG_NUM) {
        for (u32 i = 0; i < *n_leaves; i++)
            if (fuse_leaf_kinds[i] == KERNEL_LEAF_NUM &&
                fuse_leaf_dtypes[i] == (term_ext(t) == NUM_U32 ? DTYPE_U32 : DTYPE_F32) &&
                fuse_leaf_nums[i] == (term_ext(t) == NUM_U32 ? (f32)term_as_u32(t) : term_as_f32(t)))
                return (int)(WALK_LEAF_BASE + i);
        if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
        u32 idx = (*n_leaves)++;
        leaf_ids[idx] = 0;
        fuse_leaf_kinds[idx] = KERNEL_LEAF_NUM;
        fuse_leaf_nums[idx] = term_ext(t) == NUM_U32 ? (f32)term_as_u32(t) : term_as_f32(t);
        fuse_leaf_dtypes[idx] = term_ext(t) == NUM_U32 ? DTYPE_U32 : DTYPE_F32;
        fuse_composed_views[idx] = view_create((Shape){.dims={1}, .rank=1});
        leaf_views[idx] = &fuse_composed_views[idx];
        fuse_leaf_sts[idx] = st_from_view(fuse_composed_views[idx]);
        return (int)(WALK_LEAF_BASE + idx);
    }
    if (term_tag(t) != TAG_TOP) {
        if (getenv("THVM_SCHED_DIAG")) fprintf(stderr, "  walk_nottop: tag%u\n", term_tag(t));
        return -1;
    }
    u32 uop = term_ext(t);
    {
        int bidx = fuse_boundary_index_for_loc(term_val(t));
        if (bidx >= 0 && term_val(t) != fuse_root_compute_loc) {
            u32 out_tid = fuse_boundary_output_tids[bidx];
            if (out_tid == 0) return -1;
            for (u32 i = 0; i < *n_leaves; i++) {
                if (fuse_leaf_kinds[i] == KERNEL_LEAF_TENSOR && leaf_ids[i] == out_tid) {
                    fuse_dep_add(fuse_boundary_kids[bidx]);
                    return (int)(WALK_LEAF_BASE + i);
                }
            }
            if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
            u32 idx = (*n_leaves)++;
            leaf_ids[idx] = out_tid;
            fuse_leaf_kinds[idx] = KERNEL_LEAF_TENSOR;
            fuse_leaf_nums[idx] = 0.0f;
            fuse_leaf_dtypes[idx] = ctx->tensors[out_tid].dtype;
            leaf_views[idx] = tensor_view_get(&ctx->tensors[out_tid]);
            { const ShapeTracker *tst = tensor_st_get(&ctx->tensors[out_tid]);
              fuse_leaf_sts[idx] = tst ? *tst : st_from_view(ctx->tensors[out_tid].view); }
            fuse_dep_add(fuse_boundary_kids[bidx]);
            return (int)(WALK_LEAF_BASE + idx);
        }
    }

    // View ops: walk through, compose view onto leaf.
    // TAG_TEN input: walk through and compose view.
    // TAG_TOP ew/view input: walk through (ew chain is fusable).
    // TAG_TOP non-ew: lazy leaf boundary.
    if (is_view_op(uop)) {
        u64 loc = term_val(t);
        Term view_input = heap_read(ctx, loc);
        Term boundary_inner = view_input;
        for (int bd = 0; bd < 16; bd++) {
            boundary_inner = fuse_deref_links(ctx, boundary_inner);
            if (term_tag(boundary_inner) != TAG_TOP || !is_view_op(term_ext(boundary_inner))) break;
            boundary_inner = heap_read(ctx, term_val(boundary_inner));
        }
        if (term_tag(boundary_inner) == TAG_TOP) {
            int bidx = fuse_boundary_index_for_loc(term_val(boundary_inner));
            if (bidx >= 0 && term_val(boundary_inner) != fuse_root_compute_loc) {
                const View *sv = st_get(loc);
                if (!sv) return -1;
                return fuse_make_boundary_leaf(ctx, t, sv, NULL, leaf_ids, leaf_views, n_leaves);
            }
        }
        // Try to walk through: TAG_TEN, ew TAG_TOP, view TAG_TOP, or FUSING (scheduled kernel)
        Term vi0 = fuse_deref_links(ctx, view_input);
        int can_walk = (term_tag(vi0) == TAG_TEN || term_tag(vi0) == TAG_NUM);
        if (!can_walk && term_tag(vi0) == TAG_TOP) {
            u32 vi_uop = term_ext(vi0);
            can_walk = is_elementwise(vi_uop) || is_view_op(vi_uop) || vi_uop == UOP_KERNEL;
        }
        // DP0/DP1: deref to check what they point to.
        // can_walk when deref → TAG_TEN, FUSING, or ew/view TAG_TOP.
        // All view ops now handle the OP-inner case (EXPAND/SHRINK/PAD compose onto leaves).
        if (!can_walk && (term_tag(view_input) == TAG_DP0 || term_tag(view_input) == TAG_DP1 ||
                          term_tag(view_input) == TAG_VAR)) {
            Term dp_inner = fuse_deref_links(ctx, view_input);
            if (term_tag(dp_inner) == TAG_TEN) can_walk = 1;
            else if (term_tag(dp_inner) == TAG_TOP) {
                u32 di_uop = term_ext(dp_inner);
                can_walk = (di_uop == UOP_KERNEL || is_elementwise(di_uop) || is_view_op(di_uop));
            }
        }
        // Rank-changing RESHAPE: boundary when inner op produces different rank.
        // Walking through would create mixed-rank leaves → broadcast failure.
        if (can_walk && uop == UOP_RESHAPE) {
            Term vi_check = fuse_deref_links(ctx, view_input);
            if (term_tag(vi_check) == TAG_TOP) {
                Term rarg = fuse_deref_links(ctx, heap_read(ctx, loc + 1));
                if (term_tag(rarg) == TAG_TEN) {
                    u32 target_rank = ctx->tensors[(u32)term_val(rarg)].view.numel;
                    const View *inner_v = st_get(term_val(vi_check));
                    if (inner_v && target_rank != inner_v->shape.rank)
                        can_walk = 0;
                }
            }
        }
        if (!can_walk) {
            // Check if the inner (possibly through view chain) is an absorbable reduce
            if (_fuse_can_absorb_reduce) {
                Term vi = fuse_deref_links(ctx, view_input);
                // Unwrap through view ops to find SUM/RMAX
                Term reduce_found = term_era();
                Term vi_walk = vi;
                for (int _vd = 0; _vd < 10; _vd++) {
                    if (term_tag(vi_walk) != TAG_TOP) break;
                    u32 vi_uop = term_ext(vi_walk);
                    if (vi_uop == UOP_SUM || vi_uop == UOP_RMAX) {
                        reduce_found = vi_walk; break;
                    }
                    if (!is_view_op(vi_uop)) break;
                    Term nx = fuse_deref_links(ctx, heap_read(ctx, term_val(vi_walk)));
                    vi_walk = nx;
                }
                if (getenv("THVM_SCHED_DIAG2"))
                    fprintf(stderr, "  absorb_check: uop=%s vi_tag=%u vi_ext=%u found=%d\n",
                            uop_names[uop], term_tag(vi), term_tag(vi)==TAG_TOP?term_ext(vi):0,
                            term_tag(reduce_found)==TAG_TOP);
                if (term_tag(reduce_found) == TAG_TOP && !_sched_is_absorbed(reduce_found)) {
                    _fuse_absorbed_reduce = reduce_found;
                    _fuse_can_absorb_reduce = 0;
                    Term ri = fuse_deref_links(ctx, heap_read(ctx, term_val(reduce_found)));
                    return fuse_walk_inner(ctx, ri, ops, n_ops, leaf_ids, leaf_views, n_leaves);
                }
            }
            // Boundary leaf: realized child kernel or concrete leaf.
            const View *sv = st_get(loc);
            if (!sv) return -1;
            if (getenv("THVM_SCHED_DIAG2"))
                fprintf(stderr, "  lazy_leaf: outer=%s vi_tag=%u vi_ext=%u vi_uop=%s@%llu\n",
                        uop_names[uop],
                        (u32)term_tag(view_input),
                        (u32)term_ext(view_input),
                        term_tag(view_input)==TAG_TOP?uop_names[term_ext(view_input)]:"?",
                        (unsigned long long)term_val(view_input));
            return fuse_make_boundary_leaf(ctx, t, sv, NULL, leaf_ids, leaf_views, n_leaves);
        }
        u32 leaves_before = *n_leaves;
        u32 ops_before = *n_ops;
        u32 absorbed_before = fuse_n_absorbed;
        int inner = fuse_walk_inner(ctx, view_input, ops, n_ops, leaf_ids, leaf_views, n_leaves);
        if (inner < 0) {
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "  view_inner_fail: %s vi=tag%u/%s\n",
                        uop_names[uop], term_tag(view_input),
                        term_tag(view_input)==TAG_TOP?uop_names[term_ext(view_input)]:"");
            return -1;
        }
        if (inner >= (int)(WALK_LEAF_BASE * 2)) {
            // Inner returned OP (ew chain result).
            // PERMUTE: store perm for reduce axis transform.
            if (uop == UOP_PERMUTE) {
                Term parg = heap_read(ctx, loc + 1);
                parg = fuse_deref_links(ctx, parg);
                if (term_tag(parg) == TAG_TEN) {
                    TensorMeta *pp2 = &ctx->tensors[(u32)term_val(parg)];
                    _fuse_perm_rank = pp2->view.numel;
                    u32 perm[MAX_DIM];
                    tensor_meta_read_u32(ctx, (u32)term_val(parg), perm, MAX_DIM);
                    for (u32 j = 0; j < _fuse_perm_rank; j++) _fuse_perm[j] = (f32)perm[j];
                    _fuse_has_perm = 1;
                    return inner;
                }
                if (getenv("THVM_SCHED_DIAG"))
                    fprintf(stderr, "  perm_op_noten: parg=tag%u/%s\n", term_tag(parg),
                            term_tag(parg)==TAG_TOP?uop_names[term_ext(parg)]:"");
            }
            if (uop == UOP_RESHAPE) {
                // Compose RESHAPE onto each leaf's ShapeTracker.
                // On failure, fall back to boundary (undo walk, create lazy leaf).
                Term rarg2 = heap_read(ctx, loc + 1);
                rarg2 = fuse_deref_links(ctx, rarg2);
                int compose_ok = 0;
                if (term_tag(rarg2) == TAG_TEN) {
                    TensorMeta *rmp2 = &ctx->tensors[(u32)term_val(rarg2)];
                    u32 rrank2 = rmp2->view.numel;
                    if (rrank2 <= MAX_DIM) {
                        u32 rf2[MAX_DIM];
                        tensor_meta_read_u32(ctx, (u32)term_val(rarg2), rf2, MAX_DIM);
                        Shape rs_shape = {.rank = rrank2};
                        for (u32 j = 0; j < rrank2; j++) rs_shape.dims[j] = rf2[j];
                        Term vi_rs = fuse_deref_links(ctx, view_input);
                        const View *op_v = (term_tag(vi_rs) == TAG_TOP)
                            ? st_get(term_val(vi_rs)) : NULL;
                        u32 rs_numel = 1;
                        for (u32 j = 0; j < rrank2; j++) rs_numel *= rs_shape.dims[j];
                        if (op_v && rs_numel == op_v->numel) {
                            compose_ok = 1;
                            for (u32 li = leaves_before; li < *n_leaves; li++) {
                                const View *lv = leaf_views[li];
                                if (!lv) { compose_ok = 0; break; }
                                Shape leaf_target;
                                if (lv->numel == op_v->numel)
                                    leaf_target = rs_shape;
                                else if (!fuse_leaf_reshape_target(op_v->shape, rs_shape,
                                            lv->shape, &leaf_target))
                                    { compose_ok = 0; break; }
                                // Verify numel
                                u32 tn = 1;
                                for (u32 d = 0; d < leaf_target.rank; d++) tn *= leaf_target.dims[d];
                                if (tn != lv->numel) { compose_ok = 0; break; }
                                fuse_leaf_sts[li] = st_reshape(fuse_leaf_sts[li], leaf_target);
                                fuse_composed_views[li] = fuse_leaf_sts[li].views[fuse_leaf_sts[li].n_views - 1];
                                leaf_views[li] = &fuse_composed_views[li];
                            }
                        }
                    }
                }
                if (compose_ok) return inner;
                // Composition failed — fall back to boundary.
                // Undo inner walk: restore leaves, ops, and absorbed.
                *n_leaves = leaves_before;
                *n_ops = ops_before;
                fuse_n_absorbed = absorbed_before;
                { const View *sv = st_get(loc);
                  if (!sv) return -1;
                  return fuse_make_boundary_leaf(ctx, t, sv, NULL, leaf_ids, leaf_views, n_leaves);
                }
            }
            // EXPAND/SHRINK/PAD: compose onto each leaf's ShapeTracker.
            // On failure, fall back to boundary (undo walk, create lazy leaf).
            if (uop == UOP_EXPAND || uop == UOP_SHRINK || uop == UOP_PAD) {
                Term parg = heap_read(ctx, loc + 1);
                parg = fuse_deref_links(ctx, parg);
                if (term_tag(parg) != TAG_TEN) goto view_op_boundary;
                TensorMeta *mp2 = &ctx->tensors[(u32)term_val(parg)];
                u32 rank2 = mp2->view.numel;
                if (rank2 > MAX_DIM) goto view_op_boundary;
                u32 pf2[MAX_DIM];
                tensor_meta_read_u32(ctx, (u32)term_val(parg), pf2, MAX_DIM);
                int compose_ok2 = 1;
                for (u32 li = leaves_before; li < *n_leaves; li++) {
                    const View *base2 = leaf_views[li];
                    if (!base2) { compose_ok2 = 0; break; }
                    if (uop == UOP_EXPAND) {
                        // Rank mismatch check: if leaf rank != expand rank,
                        // the dim mapping is ambiguous. Fall back.
                        if (base2->shape.rank != rank2) { compose_ok2 = 0; break; }
                        fuse_leaf_sts[li] = st_expand(fuse_leaf_sts[li], pf2, rank2);
                    } else if (uop == UOP_SHRINK) {
                        u32 ndim2 = rank2 / 2;
                        if (ndim2 != base2->shape.rank) { compose_ok2 = 0; break; }
                        u32 s2[MAX_DIM], e2[MAX_DIM];
                        for (u32 j = 0; j < ndim2; j++) { s2[j]=(u32)pf2[j*2]; e2[j]=(u32)pf2[j*2+1]; }
                        fuse_leaf_sts[li] = st_shrink(fuse_leaf_sts[li], s2, e2, ndim2);
                    } else { // PAD
                        u32 ndim2 = rank2 / 2;
                        u32 pb2[MAX_DIM], pa2[MAX_DIM];
                        for (u32 j = 0; j < ndim2; j++) { pb2[j]=(u32)pf2[j*2]; pa2[j]=(u32)pf2[j*2+1]; }
                        fuse_leaf_sts[li] = st_pad(fuse_leaf_sts[li], pb2, pa2);
                    }
                    // Update leaf view to outermost for broadcast checking
                    fuse_composed_views[li] = fuse_leaf_sts[li].views[fuse_leaf_sts[li].n_views - 1];
                    leaf_views[li] = &fuse_composed_views[li];
                }
                if (compose_ok2) return inner;
                // Fall back to boundary
                view_op_boundary:
                *n_leaves = leaves_before;
                *n_ops = ops_before;
                fuse_n_absorbed = absorbed_before;
                { const View *sv = st_get(loc);
                  if (!sv) return -1;
                  return fuse_make_boundary_leaf(ctx, t, sv, NULL, leaf_ids, leaf_views, n_leaves);
                }
            }
            return -1;
        }
        u32 leaf_idx = inner - WALK_LEAF_BASE;
        const View *base = leaf_views[leaf_idx];
        if (!base) return -1;
        // Fast path: use pre-built ShapeTracker from shape table.
        // Composed at creation time (ctx/init.c) — handles all view ops
        // including rank-changing RESHAPEs, masks, compound views.
        { const ShapeTracker *pre_st = st_get_tracker(loc);
          if (pre_st && pre_st->n_views > 0) {
            if (fuse_leaf_kinds[leaf_idx] == KERNEL_LEAF_TENSOR)
                return fuse_append_leaf_tensor(ctx, leaf_ids[leaf_idx],
                                               &pre_st->views[pre_st->n_views - 1],
                                               pre_st, leaf_ids, leaf_views, n_leaves);
            return fuse_append_leaf_num(fuse_leaf_nums[leaf_idx], fuse_leaf_dtypes[leaf_idx],
                                        &pre_st->views[pre_st->n_views - 1],
                                        pre_st, leaf_ids, leaf_views, n_leaves);
          }
        }
        // Fallback: manual composition (when ST not in shape table)
        Term arg2 = heap_read(ctx, loc + 1);
        arg2 = fuse_deref_links(ctx, arg2);
        if (term_tag(arg2) != TAG_TEN) {
            // No shape params (e.g. pool stride view with ERA arg1).
            // Fall back to non-ew leaf using st_get for the view.
            const View *sv = st_get(term_val(t));
            if (!sv) {
                if (getenv("THVM_SCHED_DIAG")) fprintf(stderr, "  era_noview: %s@%llu arg2=tag%u\n", uop_names[uop], term_val(t), term_tag(arg2));
                return -1;
            }
            if (fuse_leaf_kinds[leaf_idx] == KERNEL_LEAF_TENSOR)
                return fuse_append_leaf_tensor(ctx, leaf_ids[leaf_idx], sv, NULL,
                                               leaf_ids, leaf_views, n_leaves);
            return fuse_append_leaf_num(fuse_leaf_nums[leaf_idx], fuse_leaf_dtypes[leaf_idx], sv, NULL,
                                        leaf_ids, leaf_views, n_leaves);
        }
        TensorMeta *mp = &ctx->tensors[(u32)term_val(arg2)];
        u32 rank = mp->view.numel;
        if (rank > MAX_DIM) return -1;
        u32 pf[MAX_DIM];
        tensor_meta_read_u32(ctx, (u32)term_val(arg2), pf, MAX_DIM);

        View nv = {0};
        if (uop == UOP_PERMUTE) {
            nv.offset = base->offset; nv.shape.rank = rank; nv.numel = base->numel;
            for (u32 j = 0; j < rank; j++) {
                nv.shape.dims[j] = base->shape.dims[pf[j]];
                nv.strides[j] = base->strides[pf[j]];
            }
            nv.contiguous = 0;
        } else if (uop == UOP_EXPAND) {
            nv = *base; nv.shape.rank = rank; nv.numel = 1;
            for (u32 j = 0; j < rank; j++) {
                u32 nd = pf[j];
                if (j < base->shape.rank && base->shape.dims[j] == 1 && nd > 1)
                    nv.strides[j] = 0;
                nv.shape.dims[j] = nd; nv.numel *= nd;
            }
            nv.contiguous = 0;
        } else if (uop == UOP_RESHAPE) {
            Shape ns = {.rank = rank};
            for (u32 j = 0; j < rank; j++) ns.dims[j] = pf[j];
            u32 new_numel = 1;
            for (u32 j = 0; j < rank; j++) new_numel *= ns.dims[j];
            if (new_numel != base->numel) {
                // Broadcast leaf through rank-changing reshape: try fuse_leaf_reshape_target
                // to compute the leaf's target shape in the new coordinate space.
                Term vi_rs = fuse_deref_links(ctx, view_input);
                const View *op_v = (term_tag(vi_rs) == TAG_TOP)
                    ? st_get(term_val(vi_rs)) : NULL;
                if (!op_v && term_tag(vi_rs) == TAG_TEN)
                    op_v = &ctx->tensors[(u32)term_val(vi_rs)].view;
                if (op_v && new_numel == op_v->numel) {
                    Shape leaf_target;
                    if (fuse_leaf_reshape_target(op_v->shape, ns, base->shape, &leaf_target)) {
                        u32 tn = 1;
                        for (u32 d = 0; d < leaf_target.rank; d++) tn *= leaf_target.dims[d];
                        if (tn == base->numel) {
                            nv = view_reshape(*base, leaf_target);
                            if (nv.numel != 0) goto reshape_leaf_done;
                        }
                    }
                }
                // Fall back to boundary: pool window or unmappable reshape.
                const View *sv = st_get(term_val(t));
                if (!sv) {
                    if (getenv("THVM_SCHED_DIAG"))
                        fprintf(stderr, "  reshape_numel_mismatch_noview: base=%u new=%u\n", base->numel, new_numel);
                    return -1;
                }
                if (fuse_leaf_kinds[leaf_idx] == KERNEL_LEAF_TENSOR)
                    return fuse_append_leaf_tensor(ctx, leaf_ids[leaf_idx], sv, NULL,
                                                   leaf_ids, leaf_views, n_leaves);
                return fuse_append_leaf_num(fuse_leaf_nums[leaf_idx], fuse_leaf_dtypes[leaf_idx], sv, NULL,
                                            leaf_ids, leaf_views, n_leaves);
            }
            reshape_leaf_done:
            nv = view_reshape(*base, ns);
            if (nv.numel == 0) {
                // Can't merge — use st_get view (ShapeTracker already composed correctly)
                const View *sv = st_get(term_val(t));
                if (!sv) return -1;
                nv = *sv;
            }
        } else if (uop == UOP_SHRINK) {
            u32 ndim = rank / 2;
            u32 starts[MAX_DIM], ends[MAX_DIM];
            for (u32 j = 0; j < ndim && j < MAX_DIM; j++) {
                starts[j] = pf[j * 2];
                ends[j]   = pf[j * 2 + 1];
            }
            if (ndim != base->shape.rank) return -1; // rank mismatch
            nv = view_shrink(*base, starts, ends);
            if (nv.numel == 0) return -1; // degenerate
        } else if (uop == UOP_PAD) {
            u32 ndim = rank / 2;
            u32 pad_before[MAX_DIM], pad_after[MAX_DIM];
            for (u32 j = 0; j < ndim && j < MAX_DIM; j++) {
                pad_before[j] = pf[j * 2];
                pad_after[j]  = pf[j * 2 + 1];
            }
            nv = view_pad(*base, pad_before, pad_after);
        } else {
            return -1; // unknown view op
        }
        // Propagate mask from base to composed view.
        // PERMUTE: rearrange mask_begin/mask_end by axes
        // EXPAND: keep mask on non-expanded dims
        // RESHAPE: already handled by view_reshape (for contiguous strides)
        if (base->has_mask && !nv.has_mask) {
            if (uop == UOP_PERMUTE) {
                nv.has_mask = 1;
                for (u32 j = 0; j < rank; j++) {
                    nv.mask_begin[j] = base->mask_begin[pf[j]];
                    nv.mask_end[j] = base->mask_end[pf[j]];
                }
            } else if (uop == UOP_EXPAND) {
                nv.has_mask = 1;
                for (u32 j = 0; j < rank; j++) {
                    if (j < base->shape.rank) {
                        nv.mask_begin[j] = base->mask_begin[j];
                        nv.mask_end[j] = (base->shape.dims[j] == 1 && pf[j] > 1) ?
                            pf[j] : base->mask_end[j]; // expanded dims: full range
                    } else {
                        nv.mask_begin[j] = 0;
                        nv.mask_end[j] = pf[j];
                    }
                }
            }
            // RESHAPE: view_reshape already propagated mask for contiguous case.
            // For non-contiguous + mask: the fuser shouldn't reach here
            // (view_reshape returns fallback → needs_materialize → contiguify).
        }
        // Always create a NEW leaf for composed views (no dedup —
        // the composed view differs from the base leaf's view)
        if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
        u32 new_idx = (*n_leaves)++;
        leaf_ids[new_idx] = leaf_ids[leaf_idx];
        fuse_leaf_kinds[new_idx] = fuse_leaf_kinds[leaf_idx];
        fuse_leaf_nums[new_idx] = fuse_leaf_nums[leaf_idx];
        fuse_leaf_dtypes[new_idx] = fuse_leaf_dtypes[leaf_idx];
        // Compose ST: apply the view op to the leaf's ShapeTracker
        { ShapeTracker st_composed = fuse_leaf_sts[leaf_idx];
          if (uop == UOP_RESHAPE) {
            Shape ns2 = {.rank = rank};
            for (u32 j = 0; j < rank; j++) ns2.dims[j] = pf[j];
            st_composed = st_reshape(st_composed, ns2);
          } else if (uop == UOP_PERMUTE) {
            u32 axes[MAX_DIM];
            for (u32 j = 0; j < rank; j++) axes[j] = pf[j];
            st_composed = st_permute(st_composed, axes, rank);
          } else if (uop == UOP_EXPAND) {
            st_composed = st_expand(st_composed, pf, rank);
          } else if (uop == UOP_SHRINK) {
            u32 ndim = rank / 2;
            u32 s3[MAX_DIM], e3[MAX_DIM];
            for (u32 j = 0; j < ndim; j++) { s3[j]=pf[j*2]; e3[j]=pf[j*2+1]; }
            st_composed = st_shrink(st_composed, s3, e3, ndim);
          } else if (uop == UOP_PAD) {
            u32 ndim = rank / 2;
            u32 pb3[MAX_DIM], pa3[MAX_DIM];
            for (u32 j = 0; j < ndim; j++) { pb3[j]=pf[j*2]; pa3[j]=pf[j*2+1]; }
            st_composed = st_pad(st_composed, pb3, pa3);
          }
          fuse_leaf_sts[new_idx] = st_composed;
        }
        fuse_composed_views[new_idx] = nv;
        leaf_views[new_idx] = &fuse_composed_views[new_idx];
        return (int)(WALK_LEAF_BASE + new_idx);
    }

    // Absorb unscheduled reduce: walk INTO SUM/RMAX instead of boundary.
    // Handles direct SUM/RMAX or view-wrapped: RESHAPE(SUM), PERMUTE(SUM), etc.
    // Check sched_absorbed to prevent double-absorption through DUP.
    if (!is_elementwise(uop) && _fuse_can_absorb_reduce) {
        if (getenv("THVM_SCHED_DIAG2"))
            fprintf(stderr, "  absorb_direct: uop=%s\n", uop < UOP_COUNT ? uop_names[uop] : "?");
        int is_absorbable = (uop == UOP_SUM || uop == UOP_RMAX);
        Term rt = t;
        // Unwrap through view ops to find SUM/RMAX
        if (!is_absorbable && is_view_op(uop)) {
            Term _cur = t;
            for (int _vd = 0; _vd < 10; _vd++) {
                Term _ri = fuse_deref_links(ctx, heap_read(ctx, term_val(_cur)));
                if (term_tag(_ri)==TAG_TOP && (term_ext(_ri)==UOP_SUM||term_ext(_ri)==UOP_RMAX)) {
                    is_absorbable = 1;
                    rt = _ri;
                    break;
                }
                if (term_tag(_ri) != TAG_TOP || !is_view_op(term_ext(_ri))) break;
                _cur = _ri;
            }
        }
        if (is_absorbable && !_sched_is_absorbed(rt)) {
            _fuse_absorbed_reduce = rt;
            _fuse_can_absorb_reduce = 0;
            Term ri = fuse_deref_links(ctx, heap_read(ctx, term_val(rt)));
            return fuse_walk_inner(ctx, ri, ops, n_ops, leaf_ids, leaf_views, n_leaves);
        }
    }

    // Non-elementwise TAG_TOP (SUM, pool RESHAPE, UOP_KERNEL, etc.): leaf boundary.
    if (!is_elementwise(uop)) {
        const View *sv = st_get(term_val(t));
        if (!sv) {
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "  walk_noview: %s@%llu\n", uop < UOP_COUNT ? uop_names[uop] : "?", term_val(t));
            return -1;
        }
        return fuse_make_boundary_leaf(ctx, t, sv, NULL, leaf_ids, leaf_views, n_leaves);
    }

    // Elementwise ops: recurse into children, record op
    if (*n_ops >= FUSE_MAX_OPS) return -1;
    fuse_mark_absorbed(t); // track this op for post-schedule marking
    u64 loc = term_val(t);
    int arg_a = fuse_walk_inner(ctx, heap_read(ctx, loc), ops, n_ops, leaf_ids, leaf_views, n_leaves);
    if (arg_a < 0) {
        if (getenv("THVM_SCHED_DIAG")) { Term _ta = heap_read(ctx, loc); fprintf(stderr, "  ew_arga_fail: %s arg0=tag%u/%s\n", uop_names[uop], term_tag(_ta), term_tag(_ta)==TAG_TOP?uop_names[term_ext(_ta)]:""); }
        return -1;
    }
    int arg_b = 0;
    if (is_binary(uop)) {
        arg_b = fuse_walk_inner(ctx, heap_read(ctx, loc + 1), ops, n_ops, leaf_ids, leaf_views, n_leaves);
        if (arg_b < 0) {
            if (getenv("THVM_SCHED_DIAG")) { Term _tb = heap_read(ctx, loc+1); fprintf(stderr, "  ew_argb_fail: %s arg1=tag%u/%s\n", uop_names[uop], term_tag(_tb), term_tag(_tb)==TAG_TOP?uop_names[term_ext(_tb)]:""); }
            return -1;
        }
    }
    u32 op_idx = (*n_ops)++;
    ops[op_idx] = (FusedOp){
        .uop = uop,
        .arg_a = (u32)arg_a,
        .arg_b = (u32)arg_b,
        .aux = uop == UOP_CAST ? fuse_cast_aux_from_term(ctx, loc) : 0,
    };
    return (int)(WALK_LEAF_BASE * 2 + op_idx);
}

// Remap walk references to final indices
static void fuse_remap(FusedOp *ops, u32 n_ops, u32 n_leaves) {
    for (u32 i = 0; i < n_ops; i++) {
        if (ops[i].arg_a >= WALK_LEAF_BASE * 2) ops[i].arg_a = n_leaves + (ops[i].arg_a - WALK_LEAF_BASE * 2);
        else if (ops[i].arg_a >= WALK_LEAF_BASE) ops[i].arg_a -= WALK_LEAF_BASE;
        if (ops[i].arg_b >= WALK_LEAF_BASE * 2) ops[i].arg_b = n_leaves + (ops[i].arg_b - WALK_LEAF_BASE * 2);
        else if (ops[i].arg_b >= WALK_LEAF_BASE) ops[i].arg_b -= WALK_LEAF_BASE;
    }
}

static u32 fuse_unfused_count = 0, fuse_fused_count = 0;

static void kernel_capture_result_view(TinyHVM *ctx, Term root, Shape raw_out_shape, KernelEntry *ke) {
    ke->raw_out_shape = raw_out_shape;
    ke->out_shape = raw_out_shape;
    ke->has_result_view = 0;
    ke->result_view = view_create(raw_out_shape);
    ke->result_st = st_from_view(ke->result_view);
    if (term_tag(root) != TAG_TOP || !is_view_op(term_ext(root))) return;
    const ShapeTracker *rst = st_get_tracker(term_val(root));
    if (!rst || rst->n_views == 0) return;
    const View *rv = st_last_view(rst);
    if (!rv) return;
    ke->result_st = *rst;
    ke->result_view = *rv;
    ke->out_shape = rv->shape;
    ke->has_result_view = 1;
}

// fuse_build_kernel: walk from any compute root, collect ops + leaves + reduce.
// No pattern matching — accepts ew, SUM, RMAX, RESHAPE(SUM/RMAX), or ew chains.
// Returns 1 on success (ke filled), 0 on failure.
static int fuse_build_kernel(TinyHVM *ctx, Term t, KernelEntry *ke) {
    _fuse_has_perm = 0;
    if (term_tag(t) != TAG_TOP) return 0;
    if (!ctx_default_backend(ctx)) return 0;
    u32 has_reduce = 0;
    Term ew_root = t, sum_term = term_era();

    // Unwrap any outer movement chain that sits on top of a real compute root.
    // The chain becomes result-side kernel metadata, not a standalone phase-2 net.
    Term cur = t;
    while (term_tag(cur) == TAG_TOP && is_view_op(term_ext(cur))) {
        Term inner = fuse_deref_links(ctx, heap_read(ctx, term_val(cur)));
        if (term_tag(inner) != TAG_TOP) return 0;
        cur = inner;
    }

    // Check for reduce
    u32 cur_uop = term_ext(cur);
    if (cur_uop == UOP_SUM || cur_uop == UOP_RMAX) {
        has_reduce = cur_uop;
        sum_term = cur;
        Term ri = fuse_deref_links(ctx, heap_read(ctx, term_val(cur)));
        ew_root = ri; // walk from reduce input
    } else if (is_elementwise(cur_uop)) {
        ew_root = cur; // walk from ew root
    } else {
        return 0; // MM, ASSIGN, etc. — not handled by fuser
    }

    // Walk the tree (reset absorbed + dup tracking)
    fuse_n_absorbed = 0;
    _fuse_n_dup_locs = 0;
    _fuse_can_absorb_reduce = (!has_reduce && is_elementwise(cur_uop)) ? 1 : 0;
    _fuse_absorbed_reduce = term_era();
    FusedOp ops[FUSE_MAX_OPS]; u32 n_ops = 0;
    u32 leaf_ids[FUSE_MAX_LEAVES]; const View *leaf_views_p[FUSE_MAX_LEAVES]; u32 n_leaves = 0;
    int walk_result = fuse_walk_inner(ctx, ew_root, ops, &n_ops, leaf_ids, leaf_views_p, &n_leaves);
    if (walk_result < 0) { _fuse_can_absorb_reduce = 0; ke->fail_code = 1; return 0; }
    fuse_remap(ops, n_ops, n_leaves);
    // If a reduce was absorbed during the walk, record it + mark absorbed
    if (term_tag(_fuse_absorbed_reduce) == TAG_TOP) {
        has_reduce = term_ext(_fuse_absorbed_reduce);
        sum_term = _fuse_absorbed_reduce;
        fuse_mark_absorbed(_fuse_absorbed_reduce);
    }
    _fuse_can_absorb_reduce = 0;

    // Leaves are concrete at schedule time. Shared deferred subgraphs become
    // explicit boundary outputs tracked in dep_kids.

    // Accept any kernel with at least 1 op or a reduce (no min_ops gate)

    // Leaf used + ew_view
    // n_ops==0: walk returned a single leaf (possibly after view compositions that created
    // intermediate leaves).  Only the returned leaf is live; mark only it as used.
    u8 leaf_used[FUSE_MAX_LEAVES] = {0};
    if (n_ops == 0) {
        if (walk_result >= (int)WALK_LEAF_BASE && walk_result < (int)(WALK_LEAF_BASE * 2))
            leaf_used[walk_result - WALK_LEAF_BASE] = 1;
        else
            for (u32 i = 0; i < n_leaves; i++) leaf_used[i] = 1;
    }
    for (u32 i = 0; i < n_ops; i++) {
        if (ops[i].arg_a < n_leaves) leaf_used[ops[i].arg_a] = 1;
        if (ops[i].arg_b < n_leaves) leaf_used[ops[i].arg_b] = 1;
    }
    View ew_view = {0}; int ew_init = 0;
    for (u32 i = 0; i < n_leaves; i++) {
        if (!leaf_used[i]) continue;
        if (!ew_init) { ew_view = *leaf_views_p[i]; ew_init = 1; continue; }
        View av_bc, bv_bc; u32 bc_shape[MAX_DIM], bc_ndim;
        if (!view_broadcast(&ew_view, leaf_views_p[i], &av_bc, &bv_bc, bc_shape, &bc_ndim)) {
            // Rank mismatch: try expanding lower-rank leaf with singleton dims.
            // E.g. [4,32,10,10] in an 8D iteration space [4,1,32,10,10,1,1,1]
            // → insert 1s to match: [4,1,32,10,10,1,1,1] with stride 0 for new dims.
            const View *lo = leaf_views_p[i], *hi = &ew_view;
            int lo_is_leaf = 1;
            if (lo->shape.rank > hi->shape.rank) { lo = &ew_view; hi = leaf_views_p[i]; lo_is_leaf = 0; }
            if (lo->shape.rank < hi->shape.rank && lo->numel <= hi->numel) {
                // Try to map lo's dims into hi's dims by matching products
                View expanded = {0};
                expanded.shape.rank = hi->shape.rank;
                expanded.offset = lo->offset;
                u32 li_d = 0; int ok = 1;
                for (u32 hd = 0; hd < hi->shape.rank; hd++) {
                    if (li_d < lo->shape.rank && lo->shape.dims[li_d] == hi->shape.dims[hd]) {
                        expanded.shape.dims[hd] = lo->shape.dims[li_d];
                        expanded.strides[hd] = lo->strides[li_d];
                        li_d++;
                    } else if (hi->shape.dims[hd] == 1) {
                        expanded.shape.dims[hd] = 1;
                        expanded.strides[hd] = 0;
                    } else if (li_d < lo->shape.rank && lo->shape.dims[li_d] == 1) {
                        // lo has a 1 that doesn't align — skip it
                        expanded.shape.dims[hd] = hi->shape.dims[hd];
                        expanded.strides[hd] = 0; // broadcast
                        // Don't advance li_d — try matching at next hd
                    } else {
                        ok = 0; break;
                    }
                }
                if (ok && li_d == lo->shape.rank) {
                    expanded.numel = hi->numel;
                    if (lo_is_leaf) {
                        fuse_composed_views[i] = expanded;
                        leaf_views_p[i] = &fuse_composed_views[i];
                    } else {
                        ew_view = expanded;
                    }
                    // Retry broadcast
                    if (view_broadcast(&ew_view, leaf_views_p[i], &av_bc, &bv_bc, bc_shape, &bc_ndim)) {
                        ew_view = view_create(shape_of(bc_shape, bc_ndim));
                        continue;
                    }
                }
            }
            ke->fail_code = 6;
            if (getenv("THVM_SCHED_DIAG")) {
                fprintf(stderr, "  bc_fail: n_leaves=%u i=%u lid%u=%u lid%u=%u ew_shape=[", n_leaves, i, i-1, leaf_ids[i-1], i, leaf_ids[i]);
                for (u32 _d = 0; _d < ew_view.shape.rank; _d++) fprintf(stderr, "%u,", ew_view.shape.dims[_d]);
                fprintf(stderr, "] leaf_shape=[");
                for (u32 _d = 0; _d < leaf_views_p[i]->shape.rank; _d++) fprintf(stderr, "%u,", leaf_views_p[i]->shape.dims[_d]);
                fprintf(stderr, "]\n");
                for (u32 _li = 0; _li < n_leaves; _li++) {
                    fprintf(stderr, "    leaf[%u]: id=%u shape=[", _li, leaf_ids[_li]);
                    for (u32 _d = 0; _d < leaf_views_p[_li]->shape.rank; _d++) fprintf(stderr, "%u,", leaf_views_p[_li]->shape.dims[_d]);
                    fprintf(stderr, "]\n");
                }
            }
            return 0;
        }
        ew_view = view_create(shape_of(bc_shape, bc_ndim));
    }
    if (!ew_init) { ke->fail_code = 7; return 0; }

    // Compact: remove dead leaves (view-composition intermediates).
    // Remap leaf indices in ops, shift op-reference indices accordingly.
    {
        u32 leaf_remap[FUSE_MAX_LEAVES];
        u32 new_n_leaves = 0;
        for (u32 i = 0; i < n_leaves; i++) {
            if (leaf_used[i]) {
                leaf_remap[i] = new_n_leaves;
                leaf_ids[new_n_leaves]     = leaf_ids[i];
                leaf_views_p[new_n_leaves] = leaf_views_p[i];
                fuse_leaf_kinds[new_n_leaves] = fuse_leaf_kinds[i];
                fuse_leaf_nums[new_n_leaves] = fuse_leaf_nums[i];
                fuse_leaf_dtypes[new_n_leaves] = fuse_leaf_dtypes[i];
                fuse_leaf_sts[new_n_leaves] = fuse_leaf_sts[i];
                new_n_leaves++;
            } else {
                leaf_remap[i] = 0xFFFFFFFFu;
            }
        }
        for (u32 i = 0; i < n_ops; i++) {
            if (ops[i].arg_a < n_leaves)
                ops[i].arg_a = leaf_remap[ops[i].arg_a];
            else
                ops[i].arg_a = ops[i].arg_a - n_leaves + new_n_leaves;
            if (ops[i].arg_b < n_leaves)
                ops[i].arg_b = leaf_remap[ops[i].arg_b];
            else
                ops[i].arg_b = ops[i].arg_b - n_leaves + new_n_leaves;
        }
        n_leaves = new_n_leaves;
    }

    // Reduce spec
    View out_view = ew_view;
    ReduceSpec rs = {0};
    if (has_reduce) {
        rs.reduce_type = has_reduce;
        u64 sum_loc = term_val(sum_term);
        Term sum_axes = fuse_deref_links(ctx, heap_read(ctx, sum_loc + 1));
        int found_axes = 0;
        if (term_tag(sum_axes) == TAG_TEN) {
            u32 ax_id = (u32)term_val(sum_axes);
            TensorMeta *axt = &ctx->tensors[ax_id];
            u32 n_axes = axt->view.numel;
            u32 axes_f[MAX_DIM];
            tensor_meta_read_u32(ctx, ax_id, axes_f, MAX_DIM);
            for (u32 i = 0; i < n_axes; i++) {
                int ax = (int)axes_f[i];
                // Don't transform axes through perm — SUM axes are in the ew chain's
                // output space (post-permute). The leaf views already have composed strides.
                if (ax >= 0 && ax < (int)ew_view.shape.rank) {
                    rs.is_reduce[ax] = 1; out_view.shape.dims[ax] = 1; found_axes = 1;
                }
            }
            if (found_axes) {
                out_view.numel = 1;
                for (u32 d = 0; d < out_view.shape.rank; d++) out_view.numel *= out_view.shape.dims[d];
            }
        }
        if (!found_axes) {
            for (int d = (int)ew_view.shape.rank - 1; d >= 0; d--) {
                if (ew_view.shape.dims[d] > 1) {
                    rs.is_reduce[d] = 1; out_view.shape.dims[d] = 1;
                    out_view.numel = ew_view.numel / ew_view.shape.dims[d];
                    found_axes = 1; break;
                }
            }
        }
        if (!found_axes) { ke->fail_code = 8; return 0; }
    }

    // Fill KernelEntry
    memcpy(ke->ops, ops, n_ops * sizeof(FusedOp));
    ke->n_ops = n_ops;
    for (u32 i = 0; i < n_leaves; i++) {
        ke->leaf_ids[i] = leaf_ids[i];
        ke->leaf_kinds[i] = fuse_leaf_kinds[i];
        ke->leaf_nums[i] = fuse_leaf_nums[i];
        ke->leaf_dtypes[i] = fuse_leaf_dtypes[i];
        ke->leaf_views[i] = *leaf_views_p[i];
        ke->leaf_sts[i] = fuse_leaf_sts[i];
    }
    ke->n_leaves = n_leaves;
    ke->n_deps = fuse_n_dep_kids;
    for (u32 i = 0; i < fuse_n_dep_kids; i++) ke->dep_kids[i] = fuse_dep_kids[i];
    ke->full_shape = ew_view.shape;
    ke->reduce = rs;
    ke->has_reduce = has_reduce;
    kernel_capture_result_view(ctx, t, out_view.shape, ke);
    ke->sum_term = sum_term;
    if (getenv("THVM_KERN_DIAG") && has_reduce) {
        fprintf(stderr, "FK ops=%u lv=%u perm=%d full=[", n_ops, n_leaves, _fuse_has_perm);
        for(u32 d=0;d<ew_view.shape.rank;d++) fprintf(stderr,"%u,",ew_view.shape.dims[d]);
        fprintf(stderr,"] out=[");
        for(u32 d=0;d<out_view.shape.rank;d++) fprintf(stderr,"%u,",out_view.shape.dims[d]);
        fprintf(stderr,"] ax=[");
        for(u32 d=0;d<MAX_DIM;d++) if(rs.is_reduce[d]) fprintf(stderr,"%u,",d);
        fprintf(stderr,"]\n");
    }
    return 1;
}
