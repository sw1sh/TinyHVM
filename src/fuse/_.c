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
        if (term_tag(sum_input) == TAG_TOP && is_elementwise(term_ext(sum_input))) {
            has_reduce = top_uop; sum_term = t; ew_root = sum_input;
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

    u32 min_ops = (has_reduce || has_lazy) ? 1 : 2;
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

    // Reduce axes: read all axes, build ReduceSpec for general codegen.
    // No trailing-axis restriction — codegen handles any axis config.
    View out_view = ew_view;
    ReduceSpec rs = {0};
    if (has_reduce) {
        rs.reduce_type = has_reduce;
        u64 sum_loc = term_val(sum_term);
        Term sum_axes = heap_read(ctx, sum_loc + 1);
        int found_axes = 0;
        if (term_tag(sum_axes) == TAG_TEN) {
            u32 ax_id = (u32)term_val(sum_axes);
            TensorMeta *axt = &ctx->tensors[ax_id];
            u32 n_axes = axt->view.numel;
            f32 axes_f[MAX_DIM];
            META_READ(axt->backend, axt->buf_id, axes_f, n_axes * sizeof(f32));
            for (u32 i = 0; i < n_axes; i++) {
                int ax = (int)axes_f[i];
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

    // Lazy leaves: reduce them FIRST, then re-walk the graph.
    // We can't resolve lazy leaves in-place because thvm_reduce modifies
    // the heap, invalidating the fuse's collected ops/leaf data.
    // Two-phase: reduce → re-walk (graph is now all-TAG_TEN) → dispatch.
    if (has_lazy) {
        // Phase 1: reduce all lazy leaves
        for (u32 i = 0; i < n_leaves; i++) {
            if (!LEAF_IS_LAZY(leaf_ids[i])) continue;
            Term lt = fuse_leaf_terms[i];
            thvm_reduce(ctx, lt); // reduces in-place, updates heap
        }
        // Phase 2: re-walk the graph (now all leaves should be TAG_TEN)
        n_ops = 0; n_leaves = 0;
        int walk2 = fuse_walk_inner(ctx, ew_root, ops, &n_ops, leaf_ids, leaf_views, &n_leaves);
        if (walk2 < 0) return reduce_id(ctx, t);
        fuse_remap(ops, n_ops, n_leaves);
        // Check for any remaining lazy leaves (shouldn't happen)
        for (u32 i = 0; i < n_leaves; i++)
            if (LEAF_IS_LAZY(leaf_ids[i])) return reduce_id(ctx, t);
        // Recompute output shape from fresh leaf data
        leaf_used[0] = 0; // reset
        memset(leaf_used, 0, sizeof(leaf_used));
        for (u32 i = 0; i < n_ops; i++) {
            if (ops[i].arg_a < n_leaves) leaf_used[ops[i].arg_a] = 1;
            if (ops[i].arg_b < n_leaves) leaf_used[ops[i].arg_b] = 1;
        }
        ew_view = (View){0}; ew_init = 0;
        for (u32 i = 0; i < n_leaves; i++) {
            if (!leaf_used[i]) continue;
            if (!ew_init) { ew_view = *leaf_views[i]; ew_init = 1; continue; }
            View av_bc, bv_bc; u32 bc_shape[MAX_DIM], bc_ndim;
            if (!view_broadcast(&ew_view, leaf_views[i], &av_bc, &bv_bc, bc_shape, &bc_ndim))
                return reduce_id(ctx, t);
            ew_view = view_create(shape_of(bc_shape, bc_ndim));
        }
        if (!ew_init) return reduce_id(ctx, t);
        // Recompute reduce (rebuild ReduceSpec, no trailing restriction)
        out_view = ew_view;
        memset(&rs, 0, sizeof(rs));
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
        for (u32 i = 0; i < n_leaves; i++) var_tid[i] = leaf_ids[i];
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
                if (ctx->tensors[var_tid[a_var]].requires_grad)
                    ctx->tensors[tid].requires_grad = 1;
                if (is_binary(ops[i].uop) && ctx->tensors[var_tid[b_var]].requires_grad)
                    ctx->tensors[tid].requires_grad = 1;
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
                if (ctx->tensors[itid].requires_grad) {
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
                    int needs_rg = ctx->tensors[var_tid[a_var]].requires_grad;
                    if (is_binary(ops[i].uop) && ctx->tensors[var_tid[b_var]].requires_grad)
                        needs_rg = 1;
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
                if (ctx->tensors[var_tid[a_var]].requires_grad)
                    ctx->tensors[tid].requires_grad = 1;
                if (is_binary(ops[i].uop) && ctx->tensors[var_tid[b_var]].requires_grad)
                    ctx->tensors[tid].requires_grad = 1;
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
