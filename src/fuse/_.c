// fuse/_.c — Elementwise fusion: walk lazy TAG_TOP tree, fuse into one kernel
//
// The fuser collects elementwise ops and their leaf tensors into a single
// codegen kernel. View ops (RESHAPE, PERMUTE, EXPAND) are transparent —
// they compose into leaf index expressions. Non-elementwise TAG_TOPs
// (SUM, MM) become lazy leaf boundaries with shapes from the shape table.

static int is_elementwise(u32 uop) {
    return uop==UOP_ADD||uop==UOP_SUB||uop==UOP_MUL||uop==UOP_DIV||
           uop==UOP_MAX||uop==UOP_CMP||uop==UOP_NEG||uop==UOP_RELU||
           uop==UOP_EXP||uop==UOP_LOG||uop==UOP_SQRT;
}
static int is_binary(u32 uop) {
    return uop==UOP_ADD||uop==UOP_SUB||uop==UOP_MUL||uop==UOP_DIV||
           uop==UOP_MAX||uop==UOP_CMP;
}

#define FUSE_MAX_OPS 32
#define FUSE_MAX_LEAVES 16

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
        return fuse_walk_inner(ctx, shared, ops, n_ops, leaf_ids, leaf_views, n_leaves);
    }
    if (term_tag(t) != TAG_TOP) return -1;
    u32 uop = term_ext(t);

    // View ops: walk through, compose view onto leaf.
    // TAG_TEN input: walk through and compose view.
    // TAG_TOP ew/view input: walk through (ew chain is fusable).
    // TAG_TOP non-ew: lazy leaf boundary.
    if (uop == UOP_EXPAND || uop == UOP_PERMUTE || uop == UOP_RESHAPE) {
        u64 loc = term_val(t);
        Term view_input = heap_read(ctx, loc);
        // Try to walk through: TAG_TEN, ew TAG_TOP, or view TAG_TOP
        int can_walk = (term_tag(view_input) == TAG_TEN);
        if (!can_walk && term_tag(view_input) == TAG_TOP) {
            u32 vi_uop = term_ext(view_input);
            can_walk = is_elementwise(vi_uop) || is_view_op(vi_uop);
        }
        if (!can_walk) {
            // Non-ew, non-view TAG_TOP (e.g. SUM): lazy leaf boundary
            const View *sv = st_get(loc);
            if (!sv) return -1;
            if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
            u32 idx = (*n_leaves)++;
            leaf_ids[idx] = (u32)(term_val(view_input) | 0x80000000u);
            fuse_composed_views[idx] = *sv;
            leaf_views[idx] = &fuse_composed_views[idx];
            fuse_leaf_terms[idx] = t;
            return (int)(WALK_LEAF_BASE + idx);
        }
        int inner = fuse_walk_inner(ctx, view_input, ops, n_ops, leaf_ids, leaf_views, n_leaves);
        if (inner < 0) return -1;
        if (inner >= (int)(WALK_LEAF_BASE * 2)) {
            // Inner returned OP (e.g., deferred MUL).
            // PERMUTE: store perm for reduce axis transform. RESHAPE: transparent.
            if (uop == UOP_PERMUTE) {
                Term parg = heap_read(ctx, loc + 1);
                if (term_tag(parg) == TAG_TEN) {
                    TensorMeta *pp2 = &ctx->tensors[(u32)term_val(parg)];
                    _fuse_perm_rank = pp2->view.numel;
                    META_READ(pp2->backend, pp2->buf_id, _fuse_perm, _fuse_perm_rank * sizeof(f32));
                    _fuse_has_perm = 1;
                    return inner;
                }
            }
            if (uop == UOP_RESHAPE) return inner; // transparent for computation
            return -1;
        }
        u32 leaf_idx = inner - WALK_LEAF_BASE;
        const View *base = leaf_views[leaf_idx];
        if (!base) return -1;
        Term arg2 = heap_read(ctx, loc + 1);
        // Look through DUP references for shared shape params
        if (term_tag(arg2) == TAG_DP0 || term_tag(arg2) == TAG_DP1)
            arg2 = heap_read(ctx, term_val(arg2));
        if (term_tag(arg2) != TAG_TEN) return -1;
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
        } else { // RESHAPE
            Shape ns = {.rank = rank};
            for (u32 j = 0; j < rank; j++) ns.dims[j] = (u32)pf[j];
            nv = view_reshape(*base, ns);
            if (!nv.contiguous && !base->contiguous) {
                i32 exp = 1; int dense = 1;
                for (int d = (int)rank - 1; d >= 0; d--) {
                    if (ns.dims[d] > 1 && nv.strides[d] != exp) { dense = 0; break; }
                    exp *= (i32)ns.dims[d];
                }
                if (dense) nv = *base; // non-aliasable: keep physical view
            }
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

    // Non-elementwise TAG_TOP (SUM, MM, etc.): lazy leaf boundary.
    // Shape from the shape table (set at node creation in thvm_op).
    if (!is_elementwise(uop)) {
        const View *sv = st_get(term_val(t));
        if (!sv) return -1;
        if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
        u32 idx = (*n_leaves)++;
        // Use heap loc as unique ID (not ~0u) to prevent false dedup
        leaf_ids[idx] = (u32)(term_val(t) | 0x80000000u); // high bit = lazy marker
        fuse_composed_views[idx] = *sv;
        leaf_views[idx] = &fuse_composed_views[idx];
        fuse_leaf_terms[idx] = t;
        return (int)(WALK_LEAF_BASE + idx);
    }

    // Elementwise ops: recurse into children, record op
    if (*n_ops >= FUSE_MAX_OPS) return -1;
    u64 loc = term_val(t);
    int arg_a = fuse_walk_inner(ctx, heap_read(ctx, loc), ops, n_ops, leaf_ids, leaf_views, n_leaves);
    if (arg_a < 0) return -1;
    int arg_b = 0;
    if (is_binary(uop)) {
        arg_b = fuse_walk_inner(ctx, heap_read(ctx, loc + 1), ops, n_ops, leaf_ids, leaf_views, n_leaves);
        if (arg_b < 0) return -1;
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

// ============================================================
// fuse_or_reduce: try to fuse, or fall back to normal reduction
// ============================================================
static u32 fuse_unfused_count = 0, fuse_fused_count = 0;

static u32 fuse_or_reduce(TinyHVM *ctx, Term t) {
    // return reduce_id(ctx, t); // fusion enabled
    _fuse_has_perm = 0; // reset per-fuse state
    if (term_tag(t) != TAG_TOP) return reduce_id(ctx, t);
    // Skip fusion when backend has no codegen (CPU) — avoids stack overflow in fuse_walk_inner
    if (!ctx_default_backend(ctx) || !ctx_default_backend(ctx)->dispatch_kernel_rs)
        return reduce_id(ctx, t);
    u32 top_uop = term_ext(t);

    // Pattern match: elementwise, SUM/RMAX(ew), RESHAPE(SUM/RMAX(ew))
    u32 has_reduce = 0; // 0=none, UOP_SUM, UOP_RMAX
    Term ew_root = t, sum_term = term_era(), reshape_term = term_era();

    if (top_uop == UOP_RESHAPE) {
        u64 rs_loc = term_val(t);
        Term inner = heap_read(ctx, rs_loc);
        if (term_tag(inner) == TAG_TOP && (term_ext(inner) == UOP_SUM || term_ext(inner) == UOP_RMAX)) {
            u64 sum_loc = term_val(inner);
            Term sum_input = heap_read(ctx, sum_loc);
            if (term_tag(sum_input) == TAG_TOP && is_elementwise(term_ext(sum_input))) {
                has_reduce = term_ext(inner); sum_term = inner; reshape_term = t; ew_root = sum_input;
            }
        }
        if (!has_reduce) return reduce_id(ctx, t);
    } else if (top_uop == UOP_SUM || top_uop == UOP_RMAX) {
        u64 sum_loc = term_val(t);
        Term sum_input = heap_read(ctx, sum_loc);
        // Look through DUP references
        if (term_tag(sum_input) == TAG_DP0 || term_tag(sum_input) == TAG_DP1)
            sum_input = heap_read(ctx, term_val(sum_input));
        if (term_tag(sum_input) == TAG_TOP && is_elementwise(term_ext(sum_input))) {
            has_reduce = top_uop; sum_term = t; ew_root = sum_input;
        } else if (term_tag(sum_input) == TAG_TEN) {
            // SUM(TAG_TEN): reduce-only kernel (no ew ops, just reduce the leaf)
            has_reduce = top_uop; sum_term = t; ew_root = sum_input;
        } else if (term_tag(sum_input) == TAG_TOP && is_view_op(term_ext(sum_input))) {
            // SUM(view_chain(ew_or_ten)) — walk through view ops.
            // fuse_walk_inner handles view ops transparently now.
            Term probe = sum_input;
            int found = 0;
            for (int d = 0; d < 5; d++) {
                if (term_tag(probe) == TAG_TEN) {
                    u32 _pid = (u32)term_val(probe);
                    TensorMeta *_pm = &ctx->tensors[_pid];
                    if (_pm->buf_id == 0 && _pm->creator_op) {
                        if (is_elementwise(_pm->creator_op)) { found = 1; break; }
                        if (is_view_op(_pm->creator_op) && _pm->src_ids[0]) {
                            probe = term_ten(_pm->src_ids[0], _pm->dtype);
                            continue;
                        }
                    }
                    found = 1; break; // materialized TAG_TEN: valid reduce leaf
                }
                if (term_tag(probe) != TAG_TOP) break;
                u32 pu = term_ext(probe);
                if (is_elementwise(pu)) { found = 1; break; }
                if (!is_view_op(pu)) break;
                probe = heap_read(ctx, term_val(probe));
            }
            if (found) {
                has_reduce = top_uop; sum_term = t; ew_root = sum_input;
            } else return reduce_id(ctx, t);
        } else return reduce_id(ctx, t);
    } else if (!is_elementwise(top_uop)) {
        return reduce_id(ctx, t);
    }

    // Post-reduce detection: if the ew chain's src[0] path leads to
    // a deferred RESHAPE(SUM(ew_chain)), absorb the reduce into this fusion.
    // This handles relu(mm+b) where MM is decomposed to expand+MUL+SUM.
    if (!has_reduce && is_elementwise(top_uop)) {
        Term probe = t;
        for (u32 d = 0; d < 10; d++) {
            if (term_tag(probe) != TAG_TOP) break;
            u32 pu = term_ext(probe);
            if (is_elementwise(pu)) {
                probe = heap_read(ctx, term_val(probe)); // follow src[0]
            } else if (is_view_op(pu)) {
                probe = heap_read(ctx, term_val(probe));
            } else if (pu == UOP_SUM || pu == UOP_RMAX) {
                // Found reduce under the ew chain!
                Term sum_input2 = heap_read(ctx, term_val(probe));
                if (term_tag(sum_input2) == TAG_TOP && is_elementwise(term_ext(sum_input2))) {
                    // Pattern: ew_chain → view? → SUM(ew_chain)
                    // Treat as reshape_reduce_fuse with post-reduce ops
                    has_reduce = pu;
                    sum_term = probe;
                    ew_root = sum_input2;
                    // The post-reduce ops are the ew chain from t down to the reduce
                    // They'll be handled by the ReduceSpec post_reduce_start
                }
                break;
            } else break;
        }
    }

    // Walk the tree
    FusedOp ops[FUSE_MAX_OPS]; u32 n_ops = 0;
    u32 leaf_ids[FUSE_MAX_LEAVES]; const View *leaf_views[FUSE_MAX_LEAVES]; u32 n_leaves = 0;

    int walk_result = fuse_walk_inner(ctx, ew_root, ops, &n_ops, leaf_ids, leaf_views, &n_leaves);
    if (walk_result < 0) {
        return reduce_id(ctx, t);
    }
    fuse_remap(ops, n_ops, n_leaves);

    int has_lazy = 0;
    for (u32 i = 0; i < n_leaves; i++)
        if (LEAF_IS_LAZY(leaf_ids[i])) { has_lazy = 1; break; }

    // Lazy leaves: resolve FIRST, then re-walk to get correct ops/leaves.
    // In scheduler mode (no_lazy_resolve): skip — return failure so caller
    // can dispatch dependencies first and retry.
    if (has_lazy) {
        extern int fuse_no_lazy_resolve;
        if (fuse_no_lazy_resolve) return ~0u; // scheduler: skip, retry later
        for (u32 i = 0; i < n_leaves; i++) {
            if (!LEAF_IS_LAZY(leaf_ids[i])) continue;
            Term lt = fuse_leaf_terms[i];
            thvm_reduce(ctx, lt);
        }
        n_ops = 0; n_leaves = 0;
        int walk2 = fuse_walk_inner(ctx, ew_root, ops, &n_ops, leaf_ids, leaf_views, &n_leaves);
        if (walk2 < 0) return reduce_id(ctx, t);
        fuse_remap(ops, n_ops, n_leaves);
        for (u32 i = 0; i < n_leaves; i++)
            if (LEAF_IS_LAZY(leaf_ids[i])) return reduce_id(ctx, t);
        has_lazy = 0; // resolved
    }

    // In scheduler mode: allow 0 ops for reduces (SUM(TAG_TEN) = reduce-only),
    // and 1 op for standalone ew (single RELU etc. must still dispatch).
    extern int fuse_no_lazy_resolve;
    u32 min_ops = (has_reduce) ? (fuse_no_lazy_resolve ? 0 : 1) :
                                 (fuse_no_lazy_resolve ? 1 : 2);
    if (n_ops < min_ops) return reduce_id(ctx, t);

    // Fused reduce backward safety: most backward ops now use y (output) not at (input).
    // Only DIV and MAX backward still need at from intermediates. Reject those in grad chains.
    if (has_reduce && n_ops > 1) {
        for (u32 i = 0; i < n_ops; i++) {
            u32 u = ops[i].uop;
            if (ops[i].arg_a >= n_leaves && (u == UOP_DIV || u == UOP_MAX || u == UOP_LOG))
                return reduce_id(ctx, t);
        }
    }

    // Virtual intermediate tensors handle backward provenance —
    // each op in the chain gets a virtual tensor with correct creator_op
    // and src_ids, so standard GRAD rules trace through the fused chain.

    // Output shape: broadcast only USED leaves (not unused base entries)
    u8 leaf_used[FUSE_MAX_LEAVES] = {0};
    if (n_ops == 0) {
        // Reduce-only kernel (SUM(TAG_TEN)): leaf is the direct input
        for (u32 i = 0; i < n_leaves; i++) leaf_used[i] = 1;
    }
    for (u32 i = 0; i < n_ops; i++) {
        if (ops[i].arg_a < n_leaves) leaf_used[ops[i].arg_a] = 1;
        if (ops[i].arg_b < n_leaves) leaf_used[ops[i].arg_b] = 1;
    }
    View ew_view = {0}; int ew_init = 0;
    for (u32 i = 0; i < n_leaves; i++) {
        if (!leaf_used[i]) continue;
        if (!ew_init) { ew_view = *leaf_views[i]; ew_init = 1; continue; }
        View av_bc, bv_bc; u32 bc_shape[MAX_DIM], bc_ndim;
        if (!view_broadcast(&ew_view, leaf_views[i], &av_bc, &bv_bc, bc_shape, &bc_ndim))
            return reduce_id(ctx, t);
        ew_view = view_create(shape_of(bc_shape, bc_ndim));
    }
    if (!ew_init) {
        return reduce_id(ctx, t);
    }

    // Reduce axes: read all axes, build ReduceSpec for general codegen.
    // No trailing-axis restriction — codegen handles any axis config.
    View out_view = ew_view;
    ReduceSpec rs = {0};
    if (has_reduce) {
        rs.reduce_type = has_reduce;
        u64 sum_loc = term_val(sum_term);
        Term sum_axes = heap_read(ctx, sum_loc + 1);
        // Look through DUP references for shared axes tensors
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
                // Transform axis through PERMUTE if present.
                // SUM axes are in permuted space; ew_view is in unpermuted (MUL) space.
                // PERMUTE: out[j] = in[perm[j]], so reducing out dim j
                // means reducing in dim perm[j].
                int ax = (int)axes_f[i];
                if (_fuse_has_perm && ax >= 0 && (u32)ax < _fuse_perm_rank)
                    ax = (int)_fuse_perm[ax];
                if (ax >= 0 && ax < (int)ew_view.shape.rank) {
                    rs.is_reduce[ax] = 1;
                    out_view.shape.dims[ax] = 1;
                    found_axes = 1;
                }
            }
            if (found_axes) {
                out_view.numel = 1;
                for (u32 d = 0; d < out_view.shape.rank; d++)
                    out_view.numel *= out_view.shape.dims[d];
            }
        }
        if (!found_axes) {
            // No explicit axes: reduce last non-1 dim
            for (int d = (int)ew_view.shape.rank - 1; d >= 0; d--) {
                if (ew_view.shape.dims[d] > 1) {
                    rs.is_reduce[d] = 1;
                    out_view.shape.dims[d] = 1;
                    out_view.numel = ew_view.numel / ew_view.shape.dims[d];
                    found_axes = 1;
                    break;
                }
            }
        }
        if (!found_axes) return reduce_id(ctx, t);
    }
    u32 out_numel = out_view.numel;

    // (Lazy leaf resolution moved above — has_lazy is always 0 here)
    if (0 && has_lazy) {
        if (has_reduce) {
            rs.reduce_type = has_reduce;
            u64 sloc = term_val(sum_term);
            Term saxes = heap_read(ctx, sloc + 1);
            if (term_tag(saxes) == TAG_TEN) {
                u32 axid = (u32)term_val(saxes);
                TensorMeta *axt = &ctx->tensors[axid];
                u32 nax = axt->view.numel;
                f32 axf[MAX_DIM];
                META_READ(axt->backend, axt->buf_id, axf, nax * sizeof(f32));
                for (u32 i2 = 0; i2 < nax; i2++) {
                    int ax = (int)axf[i2];
                    if (ax>=0 && ax<(int)ew_view.shape.rank) {
                        rs.is_reduce[ax] = 1;
                        out_view.shape.dims[ax] = 1;
                    }
                }
                out_view.numel = 1;
                for (u32 d = 0; d < out_view.shape.rank; d++)
                    out_view.numel *= out_view.shape.dims[d];
            } else {
                for (int d=(int)ew_view.shape.rank-1;d>=0;d--) {
                    if (ew_view.shape.dims[d]>1) {
                        rs.is_reduce[d] = 1;
                        out_view.shape.dims[d] = 1;
                        out_view.numel = ew_view.numel / ew_view.shape.dims[d];
                        break;
                    }
                }
            }
        }
        out_numel = out_view.numel;
    }

    // ── Provenance + dispatch ─────────────────────────────────────
    // For fused reduces: create virtual intermediates FIRST (lower IDs),
    // then dst (higher ID). Virtual intermediates must have LOWER IDs
    // than dst so backward tape walks (if any) visit them in correct order.
    u32 dst_id;
    u32 fuse_side_bufs[8], fuse_side_ops[8]; u32 fuse_n_sides = 0;
    {
        u32 var_tid[FUSE_MAX_LEAVES + FUSE_MAX_OPS];
        // For leaves where the composed view differs from the original tensor's view,
        // create a view-alias tensor. This ensures backward sees the correct shape
        // (e.g., 8D composed view instead of 4D original weight).
        for (u32 i = 0; i < n_leaves; i++) {
            u32 lid = leaf_ids[i];
            const View *composed = leaf_views[i];
            const View *original = &ctx->tensors[lid].view;
            // Check if shapes match
            int same = (composed->shape.rank == original->shape.rank);
            if (same)
                for (u32 d = 0; d < composed->shape.rank; d++)
                    if (composed->shape.dims[d] != original->shape.dims[d]) { same = 0; break; }
            if (!same && ctx->tensors[lid].requires_grad) {
                // Create view-alias with composed shape for correct backward broadcasting
                u32 alias_id = tensor_view_of(ctx, lid, *composed);
                ctx->tensors[alias_id].creator_op = UOP_RESHAPE;
                ctx->tensors[alias_id].src_ids[0] = lid;
                ctx->tensors[alias_id].requires_grad = 1;
                var_tid[i] = alias_id;
            } else {
                var_tid[i] = lid;
            }
        }
        if (has_reduce) {
            // Create virtual intermediates FIRST (lower IDs)
            u32 ew_last_id = 0;
            for (u32 i = 0; i < n_ops; i++) {
                u32 tid = ctx->tensor_count++;
                u32 a_var = ops[i].arg_a, b_var = ops[i].arg_b;
                View iv;
                View va_v = ctx->tensors[var_tid[a_var]].view;
                if (is_binary(ops[i].uop)) {
                    View vb_v = ctx->tensors[var_tid[b_var]].view;
                    View av_bc, bv_bc; u32 bc_s[MAX_DIM], bc_n;
                    if (view_broadcast(&va_v, &vb_v, &av_bc, &bv_bc, bc_s, &bc_n))
                        iv = view_create(shape_of(bc_s, bc_n));
                    else iv = va_v;
                } else iv = va_v;
                ctx->tensors[tid] = (TensorMeta){
                    .buf_id = 1, // placeholder — updated after dst_id created
                    .dtype = DTYPE_F32, .view = (i == n_ops - 1) ? ew_view : iv,
                    .backend = ctx->tensors[var_tid[a_var]].backend,
                };
                ctx->tensors[tid].creator_op = ops[i].uop;
                ctx->tensors[tid].src_ids[0] = var_tid[a_var];
                ctx->tensors[tid].src_ids[1] = is_binary(ops[i].uop) ? var_tid[b_var] : 0;
                if (!ctx->no_grad_alloc) {
                    if (ctx->tensors[var_tid[a_var]].requires_grad)
                        ctx->tensors[tid].requires_grad = 1;
                    if (is_binary(ops[i].uop) && ctx->tensors[var_tid[b_var]].requires_grad)
                        ctx->tensors[tid].requires_grad = 1;
                }
                var_tid[n_leaves + i] = tid;
                ew_last_id = tid;
            }
            // Create dst AFTER virtual intermediates (higher ID)
            dst_id = tensor_create(ctx, out_view.shape, DTYPE_F32);
            // Fix buf_id on virtual intermediates to point to dst's buffer.
            // EXCEPT: intermediates with requires_grad need their own buffers
            // so the GRAD handler can read their individual values for backward.
            for (u32 i = 0; i < n_ops; i++) {
                u32 itid = var_tid[n_leaves + i];
                // During backward (no_fuse=1): skip separate buffer allocation
                // for requires_grad intermediates — backward-of-backward doesn't
                // need these values. Without this, conv backward's MUL virtual
                // allocates 5.2GB (1.3B elements) for requires_grad provenance.
                if (ctx->tensors[itid].requires_grad && !ctx->no_grad_alloc) {
                    ctx->tensors[itid].buf_id = ctx->tensors[itid].backend->buf_alloc(
                        ctx->tensors[itid].view.numel * sizeof(f32));
                } else {
                    ctx->tensors[itid].buf_id = ctx->tensors[dst_id].buf_id;
                    if (ctx->tensors[dst_id].backend && ctx->tensors[dst_id].backend->buf_incref)
                        ctx->tensors[dst_id].backend->buf_incref(ctx->tensors[dst_id].buf_id);
                }
            }
            // dst = SUM/RMAX(ew_last_id, axes)
            ctx->tensors[dst_id].creator_op = has_reduce;
            ctx->tensors[dst_id].src_ids[0] = ew_last_id;
            u64 sl = term_val(sum_term);
            Term sa = heap_read(ctx, sl + 1);
            if (term_tag(sa) == TAG_TEN)
                ctx->tensors[dst_id].src_ids[1] = (u32)term_val(sa);
            if (ctx->tensors[ew_last_id].requires_grad)
                ctx->tensors[dst_id].requires_grad = 1;
        } else {
            dst_id = tensor_create(ctx, out_view.shape, DTYPE_F32);
            // Non-reduce: virtual intermediates with correct per-op provenance
            for (u32 i = 0; i < n_ops; i++) {
                u32 tid = (i == n_ops - 1) ? dst_id : ctx->tensor_count++;
                View iv;
                u32 a_var = ops[i].arg_a, b_var = ops[i].arg_b;
                View va_v = ctx->tensors[var_tid[a_var]].view;
                if (is_binary(ops[i].uop)) {
                    View vb_v = ctx->tensors[var_tid[b_var]].view;
                    View av_bc, bv_bc; u32 bc_s[MAX_DIM], bc_n;
                    if (view_broadcast(&va_v, &vb_v, &av_bc, &bv_bc, bc_s, &bc_n))
                        iv = view_create(shape_of(bc_s, bc_n));
                    else
                        iv = va_v;
                } else {
                    iv = va_v;
                }
                if (tid != dst_id) {
                    u32 ibuf = ctx->tensors[dst_id].buf_id;
                    // Intermediates with requires_grad need own buffer for backward
                    int needs_rg = 0;
                    if (!ctx->no_grad_alloc) {
                        needs_rg = ctx->tensors[var_tid[a_var]].requires_grad;
                        if (is_binary(ops[i].uop) && ctx->tensors[var_tid[b_var]].requires_grad)
                            needs_rg = 1;
                    }
                    if (needs_rg)
                        ibuf = ctx->tensors[dst_id].backend->buf_alloc(iv.numel * sizeof(f32));
                    else if (ctx->tensors[dst_id].backend && ctx->tensors[dst_id].backend->buf_incref)
                        ctx->tensors[dst_id].backend->buf_incref(ibuf);
                    ctx->tensors[tid] = (TensorMeta){
                        .buf_id = ibuf,
                        .dtype = DTYPE_F32,
                        .view = iv,
                        .backend = ctx->tensors[dst_id].backend,
                    };
                }
                ctx->tensors[tid].creator_op = ops[i].uop;
                ctx->tensors[tid].src_ids[0] = var_tid[a_var];
                ctx->tensors[tid].src_ids[1] = is_binary(ops[i].uop) ? var_tid[b_var] : 0;
                if (!ctx->no_grad_alloc) {
                    if (ctx->tensors[var_tid[a_var]].requires_grad)
                        ctx->tensors[tid].requires_grad = 1;
                    if (is_binary(ops[i].uop) && ctx->tensors[var_tid[b_var]].requires_grad)
                        ctx->tensors[tid].requires_grad = 1;
                }
                var_tid[n_leaves + i] = tid;
            }
        }
        // Collect side outputs: intermediates with requires_grad and own buffer
        for (u32 i = 0; i < n_ops && fuse_n_sides < 8; i++) {
            u32 itid = var_tid[n_leaves + i];
            if (itid != dst_id && ctx->tensors[itid].requires_grad &&
                ctx->tensors[itid].buf_id != ctx->tensors[dst_id].buf_id) {
                fuse_side_bufs[fuse_n_sides] = ctx->tensors[itid].buf_id;
                fuse_side_ops[fuse_n_sides] = n_leaves + i;
                fuse_n_sides++;
            }
        }
    }

    fuse_fused_count++;
    if (ctx->tensors[dst_id].backend->dispatch_kernel_rs) {
        u32 bufs[FUSE_MAX_LEAVES];
        for (u32 i = 0; i < n_leaves; i++) {
            ENSURE(ctx, leaf_ids[i]);
            bufs[i] = ctx->tensors[leaf_ids[i]].buf_id;
        }
        ctx->tensors[dst_id].backend->dispatch_kernel_rs(
            ctx->tensors[dst_id].buf_id,
            bufs, leaf_views, n_leaves, ops, n_ops,
            &ew_view.shape, has_reduce ? &rs : NULL,
            fuse_n_sides ? fuse_side_bufs : NULL,
            fuse_n_sides ? fuse_side_ops : NULL, fuse_n_sides);
    } else { return reduce_id(ctx, t); }

    if (has_reduce && term_tag(reshape_term) != TAG_ERA) {
        u64 rs_loc = term_val(reshape_term);
        Term shape_t = heap_read(ctx, rs_loc + 1);
        if (term_tag(shape_t) == TAG_TEN) {
            TensorMeta *ms = &ctx->tensors[(u32)term_val(shape_t)];
            u32 rank = ms->view.numel;
            f32 dims_f[MAX_DIM];
            META_READ(ms->backend, ms->buf_id, dims_f, rank * sizeof(f32));
            Shape ns = {.rank = rank};
            for (u32 i = 0; i < rank; i++) ns.dims[i] = (u32)dims_f[i];
            u32 rs_id = tensor_view_of(ctx, dst_id, view_create(ns));
            // Propagate provenance + requires_grad for backward
            ctx->tensors[rs_id].creator_op = UOP_RESHAPE;
            ctx->tensors[rs_id].src_ids[0] = dst_id;
            ctx->tensors[rs_id].requires_grad = ctx->tensors[dst_id].requires_grad;
            return rs_id;
        }
    }
    return dst_id;
}
