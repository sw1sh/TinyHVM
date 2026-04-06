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
           uop==UOP_EXP||uop==UOP_LOG||uop==UOP_SQRT;
}
static int is_binary(u32 uop) {
    return uop==UOP_ADD||uop==UOP_SUB||uop==UOP_MUL||uop==UOP_DIV||
           uop==UOP_MAX||uop==UOP_CMP;
}

// FUSE_MAX_OPS and FUSE_MAX_LEAVES defined in tinyhvm.h

// Per-walk storage
#define WALK_LEAF_BASE 10000
static View fuse_composed_views[FUSE_MAX_LEAVES];
static Term fuse_leaf_terms[FUSE_MAX_LEAVES];

// Per-fuse state: PERMUTE between ew and reduce (set by fuse_walk_inner)
static f32 _fuse_perm[MAX_DIM];
static u32 _fuse_perm_rank = 0;
static int _fuse_has_perm = 0;

// ============================================================
// fuse_walk_inner: recursive walk of lazy TAG_TOP tree
// ============================================================
// Returns WALK_LEAF_BASE + leaf_idx for leaves,
//         WALK_LEAF_BASE*2 + op_idx for ops, or -1 on failure.

static int fuse_walk_inner(TinyHVM *ctx, Term t,
                           FusedOp *ops, u32 *n_ops,
                           u32 *leaf_ids, const View **leaf_views, u32 *n_leaves) {
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
            ops[oi] = (FusedOp){.uop = m->creator_op, .arg_a = (u32)a_res, .arg_b = (u32)b_res};
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
                    META_READ(ppm->backend, ppm->buf_id, _fuse_perm, _fuse_perm_rank * sizeof(f32));
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
                f32 pf[MAX_DIM];
                META_READ(mp->backend, mp->buf_id, pf, rank * sizeof(f32));
                View nv = {0}; nv.offset = base->offset; nv.shape.rank = rank; nv.numel = base->numel;
                for (u32 j = 0; j < rank; j++) {
                    nv.shape.dims[j] = base->shape.dims[(u32)pf[j]];
                    nv.strides[j] = base->strides[(u32)pf[j]];
                }
                nv.contiguous = 0;
                fuse_composed_views[leaf_idx] = nv;
                leaf_views[leaf_idx] = &fuse_composed_views[leaf_idx];
            } else if (m->creator_op == UOP_RESHAPE && m->src_ids[1]) {
                TensorMeta *mp = &ctx->tensors[m->src_ids[1]];
                u32 rank = mp->view.numel;
                f32 rf[MAX_DIM];
                META_READ(mp->backend, mp->buf_id, rf, rank * sizeof(f32));
                Shape ns = {.rank = rank};
                for (u32 j = 0; j < rank; j++) ns.dims[j] = (u32)rf[j];
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
            if (leaf_ids[i] == tid) return (int)(WALK_LEAF_BASE + i);
        if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
        u32 idx = (*n_leaves)++;
        leaf_ids[idx] = tid;
        leaf_views[idx] = &ctx->tensors[tid].view;
        fuse_leaf_terms[idx] = t;
        return (int)(WALK_LEAF_BASE + idx);
    }
    // DP0/DP1: look through to the shared value in the DUP node
    if (term_tag(t) == TAG_DP0 || term_tag(t) == TAG_DP1) {
        Term shared = heap_read(ctx, term_val(t));
        int r = fuse_walk_inner(ctx, shared, ops, n_ops, leaf_ids, leaf_views, n_leaves);
        if (r < 0 && getenv("THVM_SCHED_DIAG"))
            fprintf(stderr, "  dp_walk_fail: dp%u@%llu → tag%u/%s\n",
                    (term_tag(t)==TAG_DP1)?1:0, term_val(t),
                    term_tag(shared),
                    term_tag(shared)==TAG_TOP ? uop_names[term_ext(shared)] : "");
        return r;
    }
    // TAG_NUM: scalar constant (e.g. epsilon in BN). Convert to 1-element tensor.
    if (term_tag(t) == TAG_NUM) {
        f32 val = term_as_f32(t);
        // Create a scalar tensor on the fly
        Term scalar = thvm_tensor(ctx, &val, (Shape){.dims={1},.rank=1});
        return fuse_walk_inner(ctx, scalar, ops, n_ops, leaf_ids, leaf_views, n_leaves);
    }
    if (term_tag(t) != TAG_TOP) {
        if (getenv("THVM_SCHED_DIAG")) fprintf(stderr, "  walk_nottop: tag%u\n", term_tag(t));
        return -1;
    }
    u32 uop = term_ext(t);

    // View ops: walk through, compose view onto leaf.
    // TAG_TEN input: walk through and compose view.
    // TAG_TOP ew/view input: walk through (ew chain is fusable).
    // TAG_TOP non-ew: lazy leaf boundary.
    if (is_view_op(uop)) {
        u64 loc = term_val(t);
        Term view_input = heap_read(ctx, loc);
        // Try to walk through: TAG_TEN, ew TAG_TOP, view TAG_TOP, or FUSING (scheduled kernel)
        int can_walk = (term_tag(view_input) == TAG_TEN);
        if (!can_walk && term_tag(view_input) == TAG_TOP) {
            u32 vi_uop = term_ext(view_input);
            can_walk = is_elementwise(vi_uop) || is_view_op(vi_uop) || vi_uop == UOP_FUSING;
        }
        // DP0/DP1: deref to check what they point to.
        // can_walk when deref → TAG_TEN, FUSING, or ew/view TAG_TOP.
        // All view ops now handle the OP-inner case (EXPAND/SHRINK/PAD compose onto leaves).
        if (!can_walk && (term_tag(view_input) == TAG_DP0 || term_tag(view_input) == TAG_DP1)) {
            Term dp_inner = heap_read(ctx, term_val(view_input));
            if (term_tag(dp_inner) == TAG_TEN) can_walk = 1;
            else if (term_tag(dp_inner) == TAG_TOP) {
                u32 di_uop = term_ext(dp_inner);
                can_walk = (di_uop == UOP_FUSING || is_elementwise(di_uop) || is_view_op(di_uop));
            }
        }
        if (!can_walk) {
            // Non-ew, non-view TAG_TOP (e.g. SUM): lazy leaf boundary
            const View *sv = st_get(loc);
            if (!sv) return -1;
            if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
            u32 idx = (*n_leaves)++;
            if (getenv("THVM_SCHED_DIAG2")) fprintf(stderr, "  lazy_leaf: outer=%s vi_uop=%s@%llu\n", uop_names[uop], term_tag(view_input)==TAG_TOP?uop_names[term_ext(view_input)]:"?", term_val(view_input));
            leaf_ids[idx] = (u32)(term_val(view_input) | 0x80000000u);
            fuse_composed_views[idx] = *sv;
            leaf_views[idx] = &fuse_composed_views[idx];
            fuse_leaf_terms[idx] = t;
            return (int)(WALK_LEAF_BASE + idx);
        }
        u32 leaves_before = *n_leaves;
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
                if (term_tag(parg) == TAG_DP0 || term_tag(parg) == TAG_DP1)
                    parg = heap_read(ctx, term_val(parg));
                if (term_tag(parg) == TAG_TEN) {
                    TensorMeta *pp2 = &ctx->tensors[(u32)term_val(parg)];
                    _fuse_perm_rank = pp2->view.numel;
                    META_READ(pp2->backend, pp2->buf_id, _fuse_perm, _fuse_perm_rank * sizeof(f32));
                    _fuse_has_perm = 1;
                    return inner;
                }
                if (getenv("THVM_SCHED_DIAG"))
                    fprintf(stderr, "  perm_op_noten: parg=tag%u/%s\n", term_tag(parg),
                            term_tag(parg)==TAG_TOP?uop_names[term_ext(parg)]:"");
            }
            if (uop == UOP_RESHAPE) return inner; // transparent for computation
            // EXPAND/SHRINK/PAD: compose view onto all leaves added by inner walk.
            // EXPAND(ew_chain) = ew_chain with each leaf broadcast to expanded coords.
            if (uop == UOP_EXPAND || uop == UOP_SHRINK || uop == UOP_PAD) {
                Term parg = heap_read(ctx, loc + 1);
                if (term_tag(parg) == TAG_DP0 || term_tag(parg) == TAG_DP1)
                    parg = heap_read(ctx, term_val(parg));
                if (term_tag(parg) != TAG_TEN) {
                    if (getenv("THVM_SCHED_DIAG")) fprintf(stderr, "  view_op_noten2: %s parg=tag%u\n", uop_names[uop], term_tag(parg));
                    return -1;
                }
                TensorMeta *mp2 = &ctx->tensors[(u32)term_val(parg)];
                u32 rank2 = mp2->view.numel;
                if (rank2 > MAX_DIM) return -1;
                f32 pf2[MAX_DIM];
                const f32 *cached2 = (const f32 *)mp2->host_ptr;
                if (cached2) memcpy(pf2, cached2, rank2 * sizeof(f32));
                else META_READ(mp2->backend, mp2->buf_id, pf2, rank2 * sizeof(f32));
                for (u32 li = leaves_before; li < *n_leaves; li++) {
                    const View *base2 = leaf_views[li];
                    if (!base2) return -1;
                    View nv2 = {0};
                    if (uop == UOP_EXPAND) {
                        nv2 = *base2; nv2.shape.rank = rank2; nv2.numel = 1;
                        for (u32 j = 0; j < rank2; j++) {
                            u32 nd = (u32)pf2[j];
                            if (j < base2->shape.rank && base2->shape.dims[j] == 1 && nd > 1)
                                nv2.strides[j] = 0;
                            nv2.shape.dims[j] = nd; nv2.numel *= nd;
                        }
                        nv2.contiguous = 0;
                    } else if (uop == UOP_SHRINK) {
                        u32 ndim2 = rank2 / 2;
                        u32 s2[MAX_DIM], e2[MAX_DIM];
                        for (u32 j = 0; j < ndim2 && j < MAX_DIM; j++) { s2[j]=(u32)pf2[j*2]; e2[j]=(u32)pf2[j*2+1]; }
                        if (ndim2 != base2->shape.rank) return -1;
                        nv2 = view_shrink(*base2, s2, e2);
                        if (nv2.numel == 0) return -1;
                    } else { // PAD
                        u32 ndim2 = rank2 / 2;
                        u32 pb2[MAX_DIM], pa2[MAX_DIM];
                        for (u32 j = 0; j < ndim2 && j < MAX_DIM; j++) { pb2[j]=(u32)pf2[j*2]; pa2[j]=(u32)pf2[j*2+1]; }
                        nv2 = view_pad(*base2, pb2, pa2);
                    }
                    fuse_composed_views[li] = nv2;
                    leaf_views[li] = &fuse_composed_views[li];
                }
                return inner; // pass OP through with composed leaf views
            }
            return -1;
        }
        u32 leaf_idx = inner - WALK_LEAF_BASE;
        const View *base = leaf_views[leaf_idx];
        if (!base) return -1;
        Term arg2 = heap_read(ctx, loc + 1);
        // Look through DUP references for shared shape params
        if (term_tag(arg2) == TAG_DP0 || term_tag(arg2) == TAG_DP1)
            arg2 = heap_read(ctx, term_val(arg2));
        if (term_tag(arg2) != TAG_TEN) {
            // No shape params (e.g. pool stride view with ERA arg1).
            // Fall back to non-ew leaf using st_get for the view.
            const View *sv = st_get(term_val(t));
            if (!sv) {
                if (getenv("THVM_SCHED_DIAG")) fprintf(stderr, "  era_noview: %s@%llu arg2=tag%u\n", uop_names[uop], term_val(t), term_tag(arg2));
                return -1;
            }
            if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
            u32 fallback_idx = (*n_leaves)++;
            leaf_ids[fallback_idx] = leaf_ids[leaf_idx]; // inherit inner leaf's tensor ID
            fuse_composed_views[fallback_idx] = *sv;
            leaf_views[fallback_idx] = &fuse_composed_views[fallback_idx];
            fuse_leaf_terms[fallback_idx] = fuse_leaf_terms[leaf_idx];
            return (int)(WALK_LEAF_BASE + fallback_idx);
        }
        TensorMeta *mp = &ctx->tensors[(u32)term_val(arg2)];
        u32 rank = mp->view.numel;
        if (rank > MAX_DIM) return -1;
        f32 pf[MAX_DIM];
        const f32 *cached = (const f32 *)mp->host_ptr;
        if (cached) memcpy(pf, cached, rank * sizeof(f32));
        else META_READ(mp->backend, mp->buf_id, pf, rank * sizeof(f32));

        View nv = {0};
        if (uop == UOP_PERMUTE) {
            nv.offset = base->offset; nv.shape.rank = rank; nv.numel = base->numel;
            for (u32 j = 0; j < rank; j++) {
                nv.shape.dims[j] = base->shape.dims[(u32)pf[j]];
                nv.strides[j] = base->strides[(u32)pf[j]];
            }
            nv.contiguous = 0;
        } else if (uop == UOP_EXPAND) {
            nv = *base; nv.shape.rank = rank; nv.numel = 1;
            for (u32 j = 0; j < rank; j++) {
                u32 nd = (u32)pf[j];
                if (j < base->shape.rank && base->shape.dims[j] == 1 && nd > 1)
                    nv.strides[j] = 0;
                nv.shape.dims[j] = nd; nv.numel *= nd;
            }
            nv.contiguous = 0;
        } else if (uop == UOP_RESHAPE) {
            Shape ns = {.rank = rank};
            for (u32 j = 0; j < rank; j++) ns.dims[j] = (u32)pf[j];
            u32 new_numel = 1;
            for (u32 j = 0; j < rank; j++) new_numel *= ns.dims[j];
            if (new_numel != base->numel) {
                // Pool window view: overlapping windows expand the apparent element count.
                // The st_set view (with correct pool strides) IS the right view to use.
                // Fall back to st_get(loc) as the leaf view, inheriting the inner leaf's ID.
                const View *sv = st_get(term_val(t));
                if (!sv) {
                    if (getenv("THVM_SCHED_DIAG"))
                        fprintf(stderr, "  reshape_numel_mismatch_noview: base=%u new=%u\n", base->numel, new_numel);
                    return -1;
                }
                if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
                u32 new_idx2 = (*n_leaves)++;
                leaf_ids[new_idx2] = leaf_ids[leaf_idx]; // inherit (0 = FUSING placeholder)
                fuse_composed_views[new_idx2] = *sv;
                leaf_views[new_idx2] = &fuse_composed_views[new_idx2];
                fuse_leaf_terms[new_idx2] = fuse_leaf_terms[leaf_idx];
                return (int)(WALK_LEAF_BASE + new_idx2);
            }
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
                starts[j] = (u32)pf[j * 2];
                ends[j]   = (u32)pf[j * 2 + 1];
            }
            if (ndim != base->shape.rank) return -1; // rank mismatch
            nv = view_shrink(*base, starts, ends);
            if (nv.numel == 0) return -1; // degenerate
        } else if (uop == UOP_PAD) {
            u32 ndim = rank / 2;
            u32 pad_before[MAX_DIM], pad_after[MAX_DIM];
            for (u32 j = 0; j < ndim && j < MAX_DIM; j++) {
                pad_before[j] = (u32)pf[j * 2];
                pad_after[j]  = (u32)pf[j * 2 + 1];
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
                    nv.mask_begin[j] = base->mask_begin[(u32)pf[j]];
                    nv.mask_end[j] = base->mask_end[(u32)pf[j]];
                }
            } else if (uop == UOP_EXPAND) {
                nv.has_mask = 1;
                for (u32 j = 0; j < rank; j++) {
                    if (j < base->shape.rank) {
                        nv.mask_begin[j] = base->mask_begin[j];
                        nv.mask_end[j] = (base->shape.dims[j] == 1 && (u32)pf[j] > 1) ?
                            (u32)pf[j] : base->mask_end[j]; // expanded dims: full range
                    } else {
                        nv.mask_begin[j] = 0;
                        nv.mask_end[j] = (u32)pf[j];
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
        fuse_leaf_terms[new_idx] = fuse_leaf_terms[leaf_idx];
        fuse_composed_views[new_idx] = nv;
        leaf_views[new_idx] = &fuse_composed_views[new_idx];
        return (int)(WALK_LEAF_BASE + new_idx);
    }

    // Non-elementwise TAG_TOP (SUM, pool RESHAPE, UOP_FUSING, etc.): leaf boundary.
    // TAG_TEN input or UOP_FUSING (scheduled) → non-lazy leaf.
    // Other TAG_TOP → lazy leaf (needs scheduling in a later pass).
    if (!is_elementwise(uop)) {
        const View *sv = st_get(term_val(t));
        if (!sv) {
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "  walk_noview: %s@%llu\n", uop < UOP_COUNT ? uop_names[uop] : "?", term_val(t));
            return -1;
        }
        if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
        u32 idx = (*n_leaves)++;
        Term leaf_input = heap_read(ctx, term_val(t));
        if (term_tag(leaf_input) == TAG_TEN) {
            leaf_ids[idx] = (u32)term_val(leaf_input); // real tensor ID
        } else if (uop == UOP_FUSING) {
            // Scheduled kernel — will become TAG_TEN during dispatch.
            // Use a placeholder; dispatch handler resolves via thvm_reduce.
            leaf_ids[idx] = 0; // placeholder (resolved at dispatch time)
        } else {
            leaf_ids[idx] = (u32)(term_val(t) | 0x80000000u); // lazy marker
        }
        fuse_composed_views[idx] = *sv;
        leaf_views[idx] = &fuse_composed_views[idx];
        fuse_leaf_terms[idx] = (term_tag(leaf_input) == TAG_TEN) ? leaf_input : t;
        return (int)(WALK_LEAF_BASE + idx);
    }

    // Elementwise ops: recurse into children, record op
    if (*n_ops >= FUSE_MAX_OPS) return -1;
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
    ops[op_idx] = (FusedOp){ .uop = uop, .arg_a = (u32)arg_a, .arg_b = (u32)arg_b };
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

#define LEAF_IS_LAZY(id) ((id) & 0x80000000u)

static u32 fuse_unfused_count = 0, fuse_fused_count = 0;

// fuse_build_kernel: pure walk + spec building. No dispatch, no TensorMeta.
// Returns 1 on success (ke filled), 0 on failure (lazy leaves, pattern mismatch).
static int fuse_build_kernel(TinyHVM *ctx, Term t, KernelEntry *ke) {
    _fuse_has_perm = 0;
    if (term_tag(t) != TAG_TOP) return 0;
    if (!ctx_default_backend(ctx)) return 0;
    u32 top_uop = term_ext(t);
    u32 has_reduce = 0;
    Term ew_root = t, sum_term = term_era(), reshape_term = term_era();

    // Pattern match
    if (top_uop == UOP_RESHAPE) {
        u64 rs_loc = term_val(t);
        Term inner = heap_read(ctx, rs_loc);
        // Look through DUP
        if (term_tag(inner) == TAG_DP0 || term_tag(inner) == TAG_DP1)
            inner = heap_read(ctx, term_val(inner));
        if (term_tag(inner) == TAG_TOP && (term_ext(inner) == UOP_SUM || term_ext(inner) == UOP_RMAX)) {
            u64 sum_loc = term_val(inner);
            Term sum_input = heap_read(ctx, sum_loc);
            if (term_tag(sum_input) == TAG_DP0 || term_tag(sum_input) == TAG_DP1)
                sum_input = heap_read(ctx, term_val(sum_input));
            if (term_tag(sum_input) == TAG_TOP && is_elementwise(term_ext(sum_input))) {
                has_reduce = term_ext(inner); sum_term = inner; reshape_term = t; ew_root = sum_input;
            }
        }
        if (!has_reduce) return 0;
    } else if (top_uop == UOP_SUM || top_uop == UOP_RMAX) {
        u64 sum_loc = term_val(t);
        Term sum_input = heap_read(ctx, sum_loc);
        if (term_tag(sum_input) == TAG_DP0 || term_tag(sum_input) == TAG_DP1)
            sum_input = heap_read(ctx, term_val(sum_input));
        if (term_tag(sum_input) == TAG_TOP && is_elementwise(term_ext(sum_input))) {
            has_reduce = top_uop; sum_term = t; ew_root = sum_input;
        } else if (term_tag(sum_input) == TAG_TEN) {
            has_reduce = top_uop; sum_term = t; ew_root = sum_input;
        } else if (term_tag(sum_input) == TAG_TOP && term_ext(sum_input) == UOP_FUSING) {
            // Scheduled kernel input — will resolve to TAG_TEN. Accept as reduce input.
            has_reduce = top_uop; sum_term = t; ew_root = sum_input;
        } else if (term_tag(sum_input) == TAG_TOP && is_view_op(term_ext(sum_input))) {
            Term probe = sum_input; int found = 0;
            for (int d = 0; d < 5; d++) {
                if (term_tag(probe) == TAG_TEN) { found = 1; break; }
                if (term_tag(probe) == TAG_DP0 || term_tag(probe) == TAG_DP1)
                    probe = heap_read(ctx, term_val(probe)); // deref DUP
                if (term_tag(probe) == TAG_TEN) { found = 1; break; }
                if (term_tag(probe) != TAG_TOP) break;
                u32 pu = term_ext(probe);
                if (is_elementwise(pu)) { found = 1; break; }
                if (pu == UOP_FUSING) { found = 1; break; }
                if (!is_view_op(pu)) break;
                probe = heap_read(ctx, term_val(probe));
            }
            if (found) { has_reduce = top_uop; sum_term = t; ew_root = sum_input; }
            else { ke->fail_code = 4; return 0; }
        } else { ke->fail_code = 5; return 0; }
    } else if (!is_elementwise(top_uop)) {
        return 0;
    }

    // Post-reduce detection for ew chains above reduces
    if (!has_reduce && is_elementwise(top_uop)) {
        Term probe = t;
        for (u32 d = 0; d < 10; d++) {
            if (term_tag(probe) != TAG_TOP) break;
            u32 pu = term_ext(probe);
            if (is_elementwise(pu)) { probe = heap_read(ctx, term_val(probe)); }
            else if (is_view_op(pu)) { probe = heap_read(ctx, term_val(probe)); }
            else if (pu == UOP_SUM || pu == UOP_RMAX) {
                Term si2 = heap_read(ctx, term_val(probe));
                if (term_tag(si2) == TAG_TOP && is_elementwise(term_ext(si2))) {
                    has_reduce = pu; sum_term = probe; ew_root = si2;
                }
                break;
            } else break;
        }
    }

    // Walk the tree
    FusedOp ops[FUSE_MAX_OPS]; u32 n_ops = 0;
    u32 leaf_ids[FUSE_MAX_LEAVES]; const View *leaf_views_p[FUSE_MAX_LEAVES]; u32 n_leaves = 0;
    int walk_result = fuse_walk_inner(ctx, ew_root, ops, &n_ops, leaf_ids, leaf_views_p, &n_leaves);
    if (walk_result < 0) { ke->fail_code = 1; return 0; }
    fuse_remap(ops, n_ops, n_leaves);

    // Lazy leaves (TAG_TOP non-ew inputs): allowed if they have known shapes
    // from st_get. The dispatch handler resolves them via thvm_reduce(leaf_term).
    // leaf_id has bit 31 set as a lazy marker; cleared at dispatch time.

    // min_ops check
    extern int fuse_no_lazy_resolve;
    u32 min_ops = has_reduce ? (fuse_no_lazy_resolve ? 0 : 1) :
                               (fuse_no_lazy_resolve ? 1 : 2);
    if (n_ops < min_ops) { ke->fail_code = 3; return 0; }

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
                fuse_leaf_terms[new_n_leaves] = fuse_leaf_terms[i];
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
        Term sum_axes = heap_read(ctx, sum_loc + 1);
        if (term_tag(sum_axes) == TAG_DP0 || term_tag(sum_axes) == TAG_DP1)
            sum_axes = heap_read(ctx, term_val(sum_axes));
        int found_axes = 0;
        if (term_tag(sum_axes) == TAG_TEN) {
            u32 ax_id = (u32)term_val(sum_axes);
            TensorMeta *axt = &ctx->tensors[ax_id];
            u32 n_axes = axt->view.numel;
            f32 axes_f[MAX_DIM];
            META_READ(axt->backend, axt->buf_id, axes_f, n_axes * sizeof(f32));
            for (u32 i = 0; i < n_axes; i++) {
                int ax = (int)axes_f[i];
                if (_fuse_has_perm && ax >= 0 && (u32)ax < _fuse_perm_rank)
                    ax = (int)_fuse_perm[ax];
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
        ke->leaf_views[i] = *leaf_views_p[i];
        ke->leaf_terms[i] = fuse_leaf_terms[i];
    }
    ke->n_leaves = n_leaves;
    ke->full_shape = ew_view.shape;
    ke->out_shape = out_view.shape;
    ke->reduce = rs;
    ke->has_reduce = has_reduce;
    ke->reshape_term = reshape_term;
    ke->sum_term = sum_term;
    return 1;
}

