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

// ============================================================
// fuse_walk_inner: recursive walk of lazy TAG_TOP tree
// ============================================================
// Returns WALK_LEAF_BASE + leaf_idx for leaves,
//         WALK_LEAF_BASE*2 + op_idx for ops, or -1 on failure.

static int fuse_walk_inner(TinyHVM *ctx, Term t,
                           FusedOp *ops, u32 *n_ops,
                           u32 *leaf_ids, const View **leaf_views, u32 *n_leaves) {
    // TAG_TEN: concrete tensor leaf
    if (term_tag(t) == TAG_TEN) {
        u32 tid = (u32)term_val(t);
        // Dedup: same tensor → reuse leaf index
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
    // Works for both TAG_TEN inputs (compose view) and TAG_TOP inputs (treat as leaf).
    if (uop == UOP_EXPAND || uop == UOP_PERMUTE || uop == UOP_RESHAPE) {
        u64 loc = term_val(t);
        Term view_input = heap_read(ctx, loc);
        // If input is TAG_TOP (lazy), treat the VIEW(TAG_TOP) as a lazy leaf.
        // The TAG_TOP will be reduced before the fused kernel runs, producing
        // a TAG_TEN that the view's strides/shape correctly reference.
        if (term_tag(view_input) != TAG_TEN) {
            // Use shape tracking to get the VIEW output shape for the leaf
            const View *sv = st_get(loc);
            if (!sv) return -1;
            if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
            u32 idx = (*n_leaves)++;
            leaf_ids[idx] = ~0u; // sentinel: lazy leaf (will be resolved during reduce)
            leaf_views[idx] = sv;
            fuse_leaf_terms[idx] = t; // store the VIEW(TAG_TOP) term
            return (int)(WALK_LEAF_BASE + idx);
        }
        int inner = fuse_walk_inner(ctx, view_input, ops, n_ops, leaf_ids, leaf_views, n_leaves);
        if (inner < 0) return -1;
        if (inner >= (int)(WALK_LEAF_BASE * 2)) return -1;
        u32 leaf_idx = inner - WALK_LEAF_BASE;
        const View *base = leaf_views[leaf_idx];
        if (!base) return -1;
        Term arg2 = heap_read(ctx, loc + 1);
        if (term_tag(arg2) != TAG_TEN) return -1;
        TensorMeta *mp = &ctx->tensors[(u32)term_val(arg2)];
        u32 rank = mp->view.numel;
        if (rank > MAX_DIM) return -1;
        f32 pf[MAX_DIM];
        const f32 *cached = (const f32 *)mp->host_ptr;
        if (cached) memcpy(pf, cached, rank * sizeof(f32));
        else META_READ(ctx, mp->buf_id, pf, rank * sizeof(f32));

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
    if (term_tag(t) != TAG_TOP) return reduce_id(ctx, t);
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
        if (term_tag(sum_input) == TAG_TOP && is_elementwise(term_ext(sum_input))) {
            has_reduce = top_uop; sum_term = t; ew_root = sum_input;
        } else return reduce_id(ctx, t);
    } else if (!is_elementwise(top_uop)) {
        return reduce_id(ctx, t);
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

    u32 min_ops = (has_reduce || has_lazy) ? 1 : 2;
    if (n_ops < min_ops) return reduce_id(ctx, t);

    // Virtual intermediate tensors handle backward provenance —
    // each op in the chain gets a virtual tensor with correct creator_op
    // and src_ids, so standard GRAD rules trace through the fused chain.

    // Output shape: broadcast only USED leaves (not unused base entries)
    u8 leaf_used[FUSE_MAX_LEAVES] = {0};
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
    if (!ew_init) return reduce_id(ctx, t);

    // Reduce axes: read all axes, compute combined reduce dim.
    // The codegen flattens all reduce dims into one reduce loop.
    View out_view = ew_view; u32 reduce_dim = 1;
    if (has_reduce) {
        u64 sum_loc = term_val(sum_term);
        Term sum_axes = heap_read(ctx, sum_loc + 1);
        int found_axes = 0;
        if (term_tag(sum_axes) == TAG_TEN) {
            u32 ax_id = (u32)term_val(sum_axes);
            TensorMeta *axt = &ctx->tensors[ax_id];
            u32 n_axes = axt->view.numel;
            f32 axes_f[MAX_DIM];
            META_READ(ctx, axt->buf_id, axes_f, n_axes * sizeof(f32));
            // The codegen treats reduce dims as one flat inner loop.
            // This only works if reduce axes are TRAILING (contiguous at the end).
            // Non-trailing reduce axes require permute — fall back to non-fused.
            u8 is_reduce_ax[MAX_DIM] = {0};
            for (u32 i = 0; i < n_axes; i++) {
                int ax = (int)axes_f[i];
                if (ax >= 0 && ax < (int)ew_view.shape.rank) is_reduce_ax[ax] = 1;
            }
            // Check trailing: all reduce axes must be at the end
            int trailing = 1;
            { int seen_nonreduce_after = 0;
              for (int d = (int)ew_view.shape.rank - 1; d >= 0; d--) {
                  if (is_reduce_ax[d] && seen_nonreduce_after) { trailing = 0; break; }
                  if (!is_reduce_ax[d] && ew_view.shape.dims[d] > 1) seen_nonreduce_after = 1;
              }
            }
            if (!trailing) return reduce_id(ctx, t); // non-trailing → can't fuse
            for (u32 i = 0; i < n_axes; i++) {
                int ax = (int)axes_f[i];
                if (ax >= 0 && ax < (int)ew_view.shape.rank) {
                    reduce_dim *= ew_view.shape.dims[ax];
                    out_view.shape.dims[ax] = 1;
                    found_axes = 1;
                }
            }
            if (found_axes)
                out_view.numel = ew_view.numel / reduce_dim;
        }
        if (!found_axes) {
            // No explicit axes: reduce last non-1 dim
            for (int d = (int)ew_view.shape.rank - 1; d >= 0; d--) {
                if (ew_view.shape.dims[d] > 1) {
                    reduce_dim = ew_view.shape.dims[d];
                    out_view.shape.dims[d] = 1;
                    out_view.numel = ew_view.numel / reduce_dim;
                    found_axes = 1;
                    break;
                }
            }
        }
        if (!found_axes) return reduce_id(ctx, t);
    }
    u32 out_numel = out_view.numel;

    // Lazy leaves: fall back to normal reduction.
    // The trampoline reduces lazy leaves depth-first, then the rewrite
    // rules catch the chain again with all-TAG_TEN leaves.
    if (has_lazy) return reduce_id(ctx, t);

    // ── Immediate dispatch (all leaves TAG_TEN) ─────────────────
    u32 dst_id = tensor_create(ctx, out_view.shape, DTYPE_F32);

    // Provenance: create virtual intermediate tensors for backward.
    // Each intermediate gets the shape from the original lazy term's
    // shape table entry (computed at graph construction time).
    // Backward reads shapes (for sum_to_shape) not data from intermediates.
    {
        u32 var_tid[FUSE_MAX_LEAVES + FUSE_MAX_OPS];
        for (u32 i = 0; i < n_leaves; i++) var_tid[i] = leaf_ids[i];

        // Reconstruct the original term for each op to get its shape
        // from the shape table. The terms are on the heap in the original
        // lazy structure. We walk `ew_root` to find them.
        // For simplicity: compute intermediate shapes from leaf shapes + ops.
        for (u32 i = 0; i < n_ops; i++) {
            u32 tid = (i == n_ops - 1) ? dst_id : ctx->tensor_count++;
            // Compute intermediate shape from operand shapes
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
                iv = va_v; // unary: output shape = input shape
            }
            if (tid != dst_id) {
                ctx->tensors[tid] = (TensorMeta){
                    .buf_id = ctx->tensors[dst_id].buf_id,
                    .dtype = DTYPE_F32,
                    .view = iv,
                };
            } else {
                // dst already has correct shape from tensor_create
            }
            ctx->tensors[tid].creator_op = ops[i].uop;
            ctx->tensors[tid].src_ids[0] = var_tid[a_var];
            ctx->tensors[tid].src_ids[1] = is_binary(ops[i].uop) ? var_tid[b_var] : 0;
            if (ctx->tensors[var_tid[a_var]].requires_grad)
                ctx->tensors[tid].requires_grad = 1;
            if (is_binary(ops[i].uop) && ctx->tensors[var_tid[b_var]].requires_grad)
                ctx->tensors[tid].requires_grad = 1;
            var_tid[n_leaves + i] = tid;
        }
    }

    fuse_fused_count++;
    #ifdef __APPLE__
    if (ctx->backend == &metal_backend) {
        u32 bufs[FUSE_MAX_LEAVES];
        for (u32 i = 0; i < n_leaves; i++) {
            ENSURE(ctx, leaf_ids[i]);
            bufs[i] = ctx->tensors[leaf_ids[i]].buf_id;
        }
        metal_dispatch_fused_v2(ctx->tensors[dst_id].buf_id, out_numel,
                                  bufs, leaf_views, n_leaves, ops, n_ops,
                                  has_reduce, reduce_dim, &ew_view.shape);
    } else
    #endif
    { return reduce_id(ctx, t); }

    if (has_reduce && term_tag(reshape_term) != TAG_ERA) {
        u64 rs_loc = term_val(reshape_term);
        Term shape_t = heap_read(ctx, rs_loc + 1);
        if (term_tag(shape_t) == TAG_TEN) {
            TensorMeta *ms = &ctx->tensors[(u32)term_val(shape_t)];
            u32 rank = ms->view.numel;
            f32 dims_f[MAX_DIM];
            META_READ(ctx, ms->buf_id, dims_f, rank * sizeof(f32));
            Shape ns = {.rank = rank};
            for (u32 i = 0; i < rank; i++) ns.dims[i] = (u32)dims_f[i];
            u32 rs_id = tensor_view_of(ctx, dst_id, view_create(ns));
            return rs_id;
        }
    }
    return dst_id;
}
