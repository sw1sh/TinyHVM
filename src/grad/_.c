Term thvm_grad(TinyHVM *ctx, Term y, Term x) {
    f32 one = 1.0f;
    Term seed = thvm_tensor(ctx, &one, SHAPE(1));
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     y);
    heap_set(ctx, loc + 1, seed);
    heap_set(ctx, loc + 2, x);
    return term_new(TAG_TOP, UOP_GRAD, loc);
}

// ============================================================
// Single-pass backward (reverse tape walk)
// ============================================================
//
// Tensors are allocated monotonically: id(input) < id(output).
// Walking from loss_id down to 0 = reverse topological order.
// Each tensor is visited ONCE. Gradients accumulate via ADD.
// No REACHES DFS, no per-parameter traversal.

static u32 reduce_id(TinyHVM *ctx, Term t);// ============================================================
// Elementwise fusion: walk lazy TAG_TOP tree, fuse into one kernel
// ============================================================

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

// Walk a lazy TAG_TOP tree, collecting ops and leaves.
// Returns number of ops collected, or 0 if not fusable.
// Two-pass walk: first discover all leaves, then assign op indices.
// Pass 1: walk tree, collect leaves and ops with PLACEHOLDER indices.
// Pass 2: remap op indices to n_leaves + op_idx.

// Internal: walk tree, record ops. Leaf indices are NEGATIVE (-1 - leaf_idx).
// Op return values are NEGATIVE too during walk, fixed up after.
#define WALK_LEAF_BASE 10000  // offset to distinguish leaf refs from op refs during walk

static int fuse_walk_inner(TinyHVM *ctx, Term t,
                           FusedOp *ops, u32 *n_ops,
                           u32 *leaf_ids, const View **leaf_views, u32 *n_leaves) {
    if (term_tag(t) == TAG_TEN) {
        u32 tid = (u32)term_val(t);
        for (u32 i = 0; i < *n_leaves; i++)
            if (leaf_ids[i] == tid) return (int)(WALK_LEAF_BASE + i);
        if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
        u32 idx = (*n_leaves)++;
        leaf_ids[idx] = tid;
        leaf_views[idx] = &ctx->tensors[tid].view;
        return (int)(WALK_LEAF_BASE + idx);
    }
    if (term_tag(t) != TAG_TOP) return -1;
    u32 uop = term_ext(t);
    if (!is_elementwise(uop)) return -1;
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
    // Use a temporary marker: ops start at WALK_LEAF_BASE * 2
    ops[op_idx] = (FusedOp){ .uop = uop, .arg_a = (u32)arg_a, .arg_b = (u32)arg_b };
    return (int)(WALK_LEAF_BASE * 2 + op_idx);
}

// After walk: remap all references.
// WALK_LEAF_BASE + i → i (leaf var index)
// WALK_LEAF_BASE*2 + i → n_leaves + i (op var index)
static void fuse_remap(FusedOp *ops, u32 n_ops, u32 n_leaves) {
    for (u32 i = 0; i < n_ops; i++) {
        if (ops[i].arg_a >= WALK_LEAF_BASE * 2) ops[i].arg_a = n_leaves + (ops[i].arg_a - WALK_LEAF_BASE * 2);
        else if (ops[i].arg_a >= WALK_LEAF_BASE) ops[i].arg_a -= WALK_LEAF_BASE;
        if (ops[i].arg_b >= WALK_LEAF_BASE * 2) ops[i].arg_b = n_leaves + (ops[i].arg_b - WALK_LEAF_BASE * 2);
        else if (ops[i].arg_b >= WALK_LEAF_BASE) ops[i].arg_b -= WALK_LEAF_BASE;
    }
}

// Try to fuse a lazy term into a single kernel. Returns tensor id, or ~0u if not fusable.
// Handles: elementwise chains AND SUM(elementwise_chain).
static u32 fuse_unfused_count = 0, fuse_fused_count = 0;
static u32 fuse_or_reduce(TinyHVM *ctx, Term t) {
    if (term_tag(t) != TAG_TOP) return reduce_id(ctx, t);
    u32 top_uop = term_ext(t);

    // Check for fusable patterns:
    // 1. elementwise_chain → fuse into one kernel
    // 2. SUM(elementwise_chain) → fuse reduce+elementwise
    // 3. RESHAPE(SUM(elementwise_chain)) → same as #2, reshape output shape
    int has_reduce = 0;
    Term ew_root = t;
    Term sum_term = term_era();  // the SUM term if has_reduce
    Term reshape_term = term_era();  // outer RESHAPE if present

    if (top_uop == UOP_RESHAPE) {
        // Look through RESHAPE to find SUM(ew_chain) pattern
        u64 rs_loc = term_val(t);
        Term inner = heap_read(ctx, rs_loc);
        if (term_tag(inner) == TAG_TOP && term_ext(inner) == UOP_SUM) {
            u64 sum_loc = term_val(inner);
            Term sum_input = heap_read(ctx, sum_loc);
            if (term_tag(sum_input) == TAG_TOP && is_elementwise(term_ext(sum_input))) {
                has_reduce = 1;
                sum_term = inner;
                reshape_term = t;
                ew_root = sum_input;
            }
        }
        if (!has_reduce) { fuse_unfused_count++; return reduce_id(ctx, t); }
    } else if (top_uop == UOP_SUM) {
        u64 sum_loc = term_val(t);
        Term sum_input = heap_read(ctx, sum_loc);
        if (term_tag(sum_input) == TAG_TOP && is_elementwise(term_ext(sum_input))) {
            has_reduce = 1;
            sum_term = t;
            ew_root = sum_input;
        } else {
            fuse_unfused_count++;
            return reduce_id(ctx, t);
        }
    } else if (!is_elementwise(term_ext(ew_root))) {
        fuse_unfused_count++;
        return reduce_id(ctx, t);
    }

    FusedOp ops[FUSE_MAX_OPS];
    u32 n_ops = 0;
    u32 leaf_ids[FUSE_MAX_LEAVES];
    const View *leaf_views[FUSE_MAX_LEAVES];
    u32 n_leaves = 0;

    int walk_result = fuse_walk_inner(ctx, ew_root, ops, &n_ops, leaf_ids, leaf_views, &n_leaves);
    if (walk_result < 0) return reduce_id(ctx, t);
    fuse_remap(ops, n_ops, n_leaves);
    u32 min_ops = has_reduce ? 1 : 2;  // reduce+1op is worth fusing
    if (n_ops < min_ops) return reduce_id(ctx, t);

    // Determine elementwise output shape (broadcast of all leaves)
    View ew_view = *leaf_views[0];
    for (u32 i = 1; i < n_leaves; i++) {
        View av_bc, bv_bc;
        u32 bc_shape[MAX_DIM], bc_ndim;
        if (!view_broadcast(&ew_view, leaf_views[i], &av_bc, &bv_bc, bc_shape, &bc_ndim)) {
            return reduce_id(ctx, t);
        }
        ew_view = view_create(shape_of(bc_shape, bc_ndim));
    }

    // For reduce: output shape has reduce dims = 1
    View out_view = ew_view;
    u32 reduce_dim = 1;
    if (has_reduce) {
        // Get reduce axis from SUM's second arg, or default to last dim
        u64 sum_loc = term_val(sum_term);
        Term sum_axes = heap_read(ctx, sum_loc + 1);
        int reduce_axis = -1;
        if (term_tag(sum_axes) == TAG_TEN) {
            // Explicit axis — read it (nosync since it's CPU metadata)
            u32 ax_id = (u32)term_val(sum_axes);
            TensorMeta *axt = &ctx->tensors[ax_id];
            if (axt->view.numel == 1) {
                f32 ax_val;
                META_READ(ctx, axt->buf_id, &ax_val, sizeof(f32));
                reduce_axis = (int)ax_val;
            }
        }
        if (reduce_axis < 0) {
            // No explicit axis or multi-axis: reduce last non-1 dim
            for (int d = (int)ew_view.shape.rank - 1; d >= 0; d--)
                if (ew_view.shape.dims[d] > 1) { reduce_axis = d; break; }
        }
        if (reduce_axis >= 0 && reduce_axis < (int)ew_view.shape.rank) {
            reduce_dim = ew_view.shape.dims[reduce_axis];
            out_view.shape.dims[reduce_axis] = 1;
            out_view.numel = ew_view.numel / reduce_dim;
        } else {
            // Can't determine reduce axis — fall back
            return reduce_id(ctx, t);
        }
    }
    u32 out_numel = out_view.numel;

    // Allocate output tensor
    u32 dst_id = tensor_create(ctx, out_view.shape, DTYPE_F32);

    fuse_fused_count++;
    // Dispatch fused kernel
    #ifdef __APPLE__
    if (ctx->backend == &metal_backend) {
        u32 bufs[FUSE_MAX_LEAVES];
        for (u32 i = 0; i < n_leaves; i++) bufs[i] = ctx->tensors[leaf_ids[i]].buf_id;
        metal_dispatch_fused_v2(ctx->tensors[dst_id].buf_id, out_numel,
                                  bufs, leaf_views, n_leaves, ops, n_ops,
                                  has_reduce, reduce_dim);
    } else
    #endif
    {
        // CPU fallback: just reduce normally
        return reduce_id(ctx, t);
    }

    // If there was an outer RESHAPE, apply it as a view alias
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
            // Create view alias with new shape (same buffer, same numel)
            u32 rs_id = tensor_view_of(ctx, dst_id, view_create(ns));
            return rs_id;
        }
    }

    return dst_id;
}

// Helper: reduce a lazy Term to TAG_TEN, return tensor id. Returns ~0u on failure.
static u32 reduce_id(TinyHVM *ctx, Term t) {
    t = thvm_reduce(ctx, t);
    return (term_tag(t) == TAG_TEN) ? (u32)term_val(t) : ~0u;
}

// Helper: accumulate gradient. If slot is empty (0), set it. Otherwise ADD.
static void grad_accum(TinyHVM *ctx, u32 *ga, u32 target, u32 grad_id) {
    if (!grad_id || grad_id == ~0u) return;
    if (!ga[target]) {
        ga[target] = grad_id;
    } else {
        Term sum = thvm_op(ctx, UOP_ADD,
            term_ten(ga[target], DTYPE_F32),
            term_ten(grad_id, DTYPE_F32));
        u32 result = fuse_or_reduce(ctx, sum);
        if (result != ~0u) ga[target] = result;
    }
}

// Compute local gradients for tensor y_id given upstream gradient gy_id.
// Writes gradient tensor IDs into ga[src_ids[0]] and ga[src_ids[1]].
static void backward_local(TinyHVM *ctx, u32 y_id, u32 gy_id, u32 *ga) {
    TensorMeta *my = &ctx->tensors[y_id];
    u32 cop = my->creator_op;
    if (!cop) return;  // leaf — no backward

    u32 aid = my->src_ids[0], bid = my->src_ids[1];
    TensorMeta *ma = &ctx->tensors[aid];
    Term gy = term_ten(gy_id, DTYPE_F32);
    Term at = term_ten(aid, ma->dtype);
    Term y  = term_ten(y_id, my->dtype);

    int is_bin = (cop==UOP_ADD||cop==UOP_SUB||cop==UOP_MUL||
                  cop==UOP_DIV||cop==UOP_MAX||cop==UOP_MM||cop==UOP_CMP);
    TensorMeta *mb = is_bin ? &ctx->tensors[bid] : NULL;
    Term bt = is_bin ? term_ten(bid, mb->dtype) : term_era();

    switch (cop) {
        case UOP_ADD: {
            Term da = sum_to_shape(ctx, gy, my->view.shape, ma->view.shape);
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            if (mb) {
                Term db = sum_to_shape(ctx, gy, my->view.shape, mb->view.shape);
                grad_accum(ctx, ga, bid, fuse_or_reduce(ctx, db));
            }
            break;
        }
        case UOP_SUB: {
            Term da = sum_to_shape(ctx, gy, my->view.shape, ma->view.shape);
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            if (mb) {
                Term neg = thvm_op(ctx, UOP_NEG,
                    sum_to_shape(ctx, gy, my->view.shape, mb->view.shape), term_era());
                grad_accum(ctx, ga, bid, fuse_or_reduce(ctx, neg));
            }
            break;
        }
        case UOP_MUL: {
            // d_a = sum_to_shape(gy * b, y_shape, a_shape)
            // d_b = sum_to_shape(gy * a, y_shape, b_shape)
            View av_bc, bv_bc;
            u32 bc_shape[MAX_DIM], bc_ndim;
            int bc_ok = view_broadcast(&ctx->tensors[gy_id].view, &mb->view,
                                       &av_bc, &bv_bc, bc_shape, &bc_ndim);
            if (bc_ok && bc_ndim == my->view.shape.rank) { // fused MUL backward
                // Compute reduce axes for d_a: dims where a was broadcast
                u32 da_rdims[MAX_DIM], da_rstrides_gy[MAX_DIM], da_rstrides_b[MAX_DIM];
                u32 da_n = 0;
                View gy_v = ctx->tensors[gy_id].view;
                // Broadcast gy vs b to get the MUL output shape
                // Then reduce axes = dims where a_shape < out_shape
                for (u32 d = 0; d < my->view.shape.rank; d++) {
                    if (my->view.shape.dims[d] != ma->view.shape.dims[d]) {
                        da_rdims[da_n] = my->view.shape.dims[d];
                        da_rstrides_gy[da_n] = (u32)(d < gy_v.shape.rank ? (gy_v.strides[d] > 0 ? gy_v.strides[d] : 0) : 0);
                        da_rstrides_b[da_n] = (u32)(d < mb->view.shape.rank ? (mb->view.strides[d] > 0 ? mb->view.strides[d] : 0) : 0);
                        da_n++;
                    }
                }
                if (da_n > 0) {
                    u32 da_id = tensor_create(ctx, ma->view.shape, ma->dtype);
                    View da_ov = view_create(ma->view.shape);
                    #ifdef __APPLE__
                    if (ctx->backend == &metal_backend) {
                        metal_mul_reduce_sum(
                            ctx->tensors[da_id].buf_id, ma->view.numel,
                            ctx->tensors[gy_id].buf_id, &gy_v,
                            mb->buf_id, &mb->view,
                            &da_ov, da_n, da_rdims, da_rstrides_gy, da_rstrides_b);
                    } else
                    #endif
                    {
                        Term da = sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, bt),
                                               my->view.shape, ma->view.shape);
                        da_id = fuse_or_reduce(ctx, da);
                    }
                    grad_accum(ctx, ga, aid, da_id);
                } else {
                    Term da = thvm_op(ctx, UOP_MUL, gy, bt);
                    grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
                }
                // d_b
                if (mb) {
                    u32 db_rdims[MAX_DIM], db_rstrides_gy[MAX_DIM], db_rstrides_a[MAX_DIM];
                    u32 db_n = 0;
                    for (u32 d = 0; d < my->view.shape.rank; d++) {
                        if (my->view.shape.dims[d] != mb->view.shape.dims[d]) {
                            db_rdims[db_n] = my->view.shape.dims[d];
                            db_rstrides_gy[db_n] = (u32)(d < gy_v.shape.rank ? (gy_v.strides[d] > 0 ? gy_v.strides[d] : 0) : 0);
                            db_rstrides_a[db_n] = (u32)(d < ma->view.shape.rank ? (ma->view.strides[d] > 0 ? ma->view.strides[d] : 0) : 0);
                            db_n++;
                        }
                    }
                    if (db_n > 0) {
                        u32 db_id = tensor_create(ctx, mb->view.shape, mb->dtype);
                        View db_ov = view_create(mb->view.shape);
                        #ifdef __APPLE__
                        if (ctx->backend == &metal_backend) {
                            metal_mul_reduce_sum(
                                ctx->tensors[db_id].buf_id, mb->view.numel,
                                ctx->tensors[gy_id].buf_id, &gy_v,
                                ma->buf_id, &ma->view,
                                &db_ov, db_n, db_rdims, db_rstrides_gy, db_rstrides_a);
                        } else
                        #endif
                        {
                            Term db = sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, at),
                                                   my->view.shape, mb->view.shape);
                            db_id = fuse_or_reduce(ctx, db);
                        }
                        grad_accum(ctx, ga, bid, db_id);
                    } else {
                        Term db = thvm_op(ctx, UOP_MUL, gy, at);
                        grad_accum(ctx, ga, bid, fuse_or_reduce(ctx, db));
                    }
                }
            } else {
                // Fallback: non-fusable
                Term da = sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, bt),
                                       my->view.shape, ma->view.shape);
                grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
                if (mb) {
                    Term db = sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, at),
                                           my->view.shape, mb->view.shape);
                    grad_accum(ctx, ga, bid, fuse_or_reduce(ctx, db));
                }
            }
            break;
        }
        case UOP_MM: {
            u32 bt_id = tensor_transpose_2d(ctx, bid);
            Term da = thvm_op(ctx, UOP_MM, gy, term_ten(bt_id, mb->dtype));
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            u32 at_id = tensor_transpose_2d(ctx, aid);
            Term db = thvm_op(ctx, UOP_MM, term_ten(at_id, ma->dtype), gy);
            grad_accum(ctx, ga, bid, fuse_or_reduce(ctx, db));
            break;
        }
        case UOP_RELU: {
            f32 z = 0.0f;
            Term mask = thvm_op(ctx, UOP_CMP, at, thvm_tensor(ctx, &z, SHAPE(1)));
            Term da = thvm_op(ctx, UOP_MUL, gy, mask);
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            break;
        }
        case UOP_NEG: {
            Term da = thvm_op(ctx, UOP_NEG, gy, term_era());
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            break;
        }
        case UOP_EXP: {
            Term da = thvm_op(ctx, UOP_MUL, gy, y);
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            break;
        }
        case UOP_LOG: {
            Term da = thvm_op(ctx, UOP_DIV, gy, at);
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            break;
        }
        case UOP_SQRT: {
            f32 two = 2.0f;
            Term denom = thvm_op(ctx, UOP_MUL, thvm_tensor(ctx, &two, SHAPE(1)), y);
            Term da = thvm_op(ctx, UOP_DIV, gy, denom);
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            break;
        }
        case UOP_DIV: {
            Term da = thvm_op(ctx, UOP_DIV, gy, bt);
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            Term ng = thvm_op(ctx, UOP_NEG, gy, term_era());
            Term db = thvm_op(ctx, UOP_DIV,
                thvm_op(ctx, UOP_MUL, ng, at),
                thvm_op(ctx, UOP_MUL, bt, bt));
            grad_accum(ctx, ga, bid, fuse_or_reduce(ctx, db));
            break;
        }
        case UOP_MAX: {
            Term mask = thvm_op(ctx, UOP_CMP, at, bt);
            Term da = sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, mask),
                                   my->view.shape, ma->view.shape);
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            f32 one = 1.0f;
            Term inv = thvm_op(ctx, UOP_SUB, thvm_tensor(ctx, &one, SHAPE(1)), mask);
            Term db = sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, inv),
                                   my->view.shape, mb->view.shape);
            grad_accum(ctx, ga, bid, fuse_or_reduce(ctx, db));
            break;
        }
        case UOP_CMP:
            break;  // CMP has zero gradient (step function)
        case UOP_SUM: {
            Term g = thvm_expand(ctx, term_ten(gy_id, DTYPE_F32), ma->view.shape);
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, g));
            break;
        }
        case UOP_RMAX: {
            Term max_bc = thvm_expand(ctx,
                thvm_reshape(ctx, y, my->view.shape), ma->view.shape);
            f32 one = 1.0f;
            Term mask = thvm_op(ctx, UOP_SUB,
                thvm_tensor(ctx, &one, SHAPE(1)),
                thvm_op(ctx, UOP_CMP, max_bc, at));
            Term gbc = thvm_expand(ctx, term_ten(gy_id, DTYPE_F32), ma->view.shape);
            Term da = thvm_op(ctx, UOP_MUL, gbc, mask);
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            break;
        }
        case UOP_RESHAPE: {
            Term da = thvm_reshape(ctx, gy, ma->view.shape);
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            break;
        }
        case UOP_EXPAND: {
            // EXPAND backward = sum over broadcast axes.
            // Use mul_reduce_sum(gy, ones=1, axes) on GPU to avoid CPU multi-axis SUM.
            // If gy has a mask, fall back to sum_to_shape (mask-aware CPU path)
            if (ctx->tensors[gy_id].view.has_mask) {
                Term da = sum_to_shape(ctx, gy, my->view.shape, ma->view.shape);
                grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
                break;
            }
            u32 rn_e = 0;
            u32 rdims_e[MAX_DIM], rstrides_gy_e[MAX_DIM], rstrides_ones_e[MAX_DIM];
            View gy_v_e = ctx->tensors[gy_id].view;
            for (u32 d = 0; d < my->view.shape.rank; d++) {
                if (d < ma->view.shape.rank && my->view.shape.dims[d] != ma->view.shape.dims[d]) {
                    rdims_e[rn_e] = my->view.shape.dims[d];
                    rstrides_gy_e[rn_e] = (u32)(d < gy_v_e.shape.rank && gy_v_e.strides[d] > 0 ? gy_v_e.strides[d] : 0);
                    rstrides_ones_e[rn_e] = 0;
                    rn_e++;
                }
            }
            if (rn_e > 0) {
                #ifdef __APPLE__
                if (ctx->backend == &metal_backend) {
                    f32 one_val = 1.0f;
                    u32 ones_buf = ctx->backend->buf_alloc(sizeof(f32));
                    ctx->backend->buf_write(ones_buf, &one_val, sizeof(f32));
                    View ones_v = view_create(gy_v_e.shape);  // same shape as gy, no mask
                    for (u32 d2=0; d2<ones_v.shape.rank; d2++) ones_v.strides[d2] = 0;
                    ones_v.offset = 0; ones_v.contiguous = 0; ones_v.has_mask = 0;
                    u32 da_id = tensor_create(ctx, ma->view.shape, DTYPE_F32);
                    View da_ov = view_create(ma->view.shape);
                    metal_mul_reduce_sum(
                        ctx->tensors[da_id].buf_id, ma->view.numel,
                        ctx->tensors[gy_id].buf_id, &gy_v_e,
                        ones_buf, &ones_v,
                        &da_ov, rn_e, rdims_e, rstrides_gy_e, rstrides_ones_e);
                    grad_accum(ctx, ga, aid, da_id);
                } else
                #endif
                {
                    Term da = sum_to_shape(ctx, gy, my->view.shape, ma->view.shape);
                    grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
                }
            } else {
                grad_accum(ctx, ga, aid, gy_id);
            }
            break;
        }
        case UOP_PERMUTE: {
            u32 rank = ctx->tensors[bid].view.numel;
            f32 *af = malloc(rank * sizeof(f32));
            META_READ(ctx, ctx->tensors[bid].buf_id, af, rank*sizeof(f32));
            u32 inv[MAX_DIM];
            for (u32 j = 0; j < rank; j++) inv[(u32)af[j]] = j;
            free(af);
            Term da = thvm_permute(ctx, gy, inv, rank);
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            break;
        }
        case UOP_PAD: {
            TensorMeta *mp = &ctx->tensors[bid];
            u32 nd = mp->view.numel / 2;
            f32 *pf = malloc(mp->view.numel * sizeof(f32));
            META_READ(ctx, mp->buf_id, pf, mp->view.numel*sizeof(f32));
            u32 sp[MAX_DIM*2];
            for (u32 j=0;j<nd;j++){sp[j*2]=(u32)pf[j*2];sp[j*2+1]=(u32)pf[j*2]+ma->view.shape.dims[j];}
            free(pf);
            Term da = thvm_shrink(ctx, gy, sp, nd);
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            break;
        }
        case UOP_SHRINK: {
            TensorMeta *ms2 = &ctx->tensors[bid];
            u32 nd = ms2->view.numel / 2;
            f32 *sf = malloc(ms2->view.numel * sizeof(f32));
            META_READ(ctx, ms2->buf_id, sf, ms2->view.numel*sizeof(f32));
            u32 pp[MAX_DIM*2];
            for (u32 j=0;j<nd;j++){pp[j*2]=(u32)sf[j*2];pp[j*2+1]=ma->view.shape.dims[j]-(u32)sf[j*2+1];}
            free(sf);
            Term da = thvm_pad(ctx, gy, pp, nd);
            grad_accum(ctx, ga, aid, fuse_or_reduce(ctx, da));
            break;
        }
        case UOP_FUSING: {
            // Fused SUM(MUL(a, b)) backward using mul_reduce_sum directly.
            // d_a = mul_reduce_sum(expand(gy), b, axes_for_a)
            // d_b = mul_reduce_sum(expand(gy), a, axes_for_b)
            // This avoids materializing the huge MUL(a,b) intermediate.
            u32 ma_id = my->src_ids[0], mb_id = my->src_ids[1];
            TensorMeta *fa = &ctx->tensors[ma_id];
            TensorMeta *fb = &ctx->tensors[mb_id];

            // Reconstruct broadcast views (same as forward)
            View av_bc, bv_bc;
            u32 bc_shape[MAX_DIM], bc_ndim;
            int ok = view_broadcast(&fa->view, &fb->view, &av_bc, &bv_bc,
                                    bc_shape, &bc_ndim);
            if (!ok) break;

            // Read original reduce axes from fusing_loc
            u64 sum_loc = my->fusing_loc;
            u32 n_reduce = 0, reduce_axes[MAX_DIM];
            Term sum_arg = heap_read(ctx, sum_loc + 1);
            if (term_tag(sum_arg) == TAG_TOP || term_tag(sum_arg) == TAG_TEN) {
                Term axes_t = thvm_reduce(ctx, sum_arg);
                if (term_tag(axes_t) == TAG_TEN) {
                    u32 ax_id = (u32)term_val(axes_t);
                    TensorMeta *axt = &ctx->tensors[ax_id];
                    n_reduce = axt->view.numel;
                    f32 *axes_f = malloc(n_reduce * sizeof(f32));
                    META_READ(ctx, axt->buf_id, axes_f, n_reduce * sizeof(f32));
                    for (u32 i = 0; i < n_reduce; i++) reduce_axes[i] = (u32)axes_f[i];
                    free(axes_f);
                }
            }
            if (!n_reduce) {
                // Fallback: last non-1 dim
                for (int i = (int)bc_ndim - 1; i >= 0; i--) {
                    if (bc_shape[i] > 1) { reduce_axes[0] = (u32)i; n_reduce = 1; break; }
                }
            }

            // Build gy view expanded to broadcast shape (stride=0 on reduce dims)
            u32 gy_exp_id = ctx->tensor_count++;
            ctx->tensors[gy_exp_id] = ctx->tensors[gy_id];
            ctx->tensors[gy_exp_id].host_ptr = NULL;
            ctx->tensors[gy_exp_id].creator_op = 0;
            View *gyv = &ctx->tensors[gy_exp_id].view;
            // gy has shape with reduce dims = 1. Expand to bc_shape.
            for (u32 d = 0; d < bc_ndim; d++) {
                if (gyv->shape.dims[d] == 1 && bc_shape[d] > 1)
                    gyv->strides[d] = 0;
                gyv->shape.dims[d] = bc_shape[d];
            }
            gyv->numel = 1;
            for (u32 d = 0; d < bc_ndim; d++) gyv->numel *= bc_shape[d];
            gyv->contiguous = 0;

            // For d_a: reduce over {original reduce axes} ∪ {axes where a was broadcast}
            // For d_b: reduce over {original reduce axes} ∪ {axes where b was broadcast}
            u8 is_reduce[MAX_DIM] = {0};
            for (u32 i = 0; i < n_reduce; i++) is_reduce[reduce_axes[i]] = 1;

            // d_a = mul_reduce_sum(gy_expanded, b, da_axes)
            if (fa->requires_grad || ga[ma_id]) {
                u32 da_axes[MAX_DIM], da_n = 0;
                u32 da_dims[MAX_DIM], da_strides_gy[MAX_DIM], da_strides_b[MAX_DIM];
                for (u32 d = 0; d < bc_ndim; d++) {
                    if (is_reduce[d] || av_bc.strides[d] == 0) {
                        da_axes[da_n] = d;
                        da_dims[da_n] = bc_shape[d];
                        da_strides_gy[da_n] = (u32)(gyv->strides[d] > 0 ? gyv->strides[d] : 0);
                        da_strides_b[da_n] = (u32)(bv_bc.strides[d] > 0 ? bv_bc.strides[d] : 0);
                        da_n++;
                    }
                }
                // Output shape: a's original shape
                u32 da_id = tensor_create(ctx, fa->view.shape, fa->dtype);
                View da_ov = view_create(fa->view.shape);
                #ifdef __APPLE__
                if (ctx->backend == &metal_backend) {
                    metal_mul_reduce_sum(
                        ctx->tensors[da_id].buf_id, fa->view.numel,
                        ctx->tensors[gy_exp_id].buf_id, gyv,
                        fb->buf_id, &bv_bc,
                        &da_ov, da_n, da_dims, da_strides_gy, da_strides_b);
                } else
                #endif
                {
                    // CPU fallback: sum(gy_expanded * b, axes_for_a)
                    Shape bc_s = {.rank = bc_ndim};
                    for (u32 d2 = 0; d2 < bc_ndim; d2++) bc_s.dims[d2] = bc_shape[d2];
                    Term da_lazy = sum_to_shape(ctx,
                        thvm_op(ctx, UOP_MUL, term_ten(gy_exp_id, DTYPE_F32), term_ten(mb_id, DTYPE_F32)),
                        bc_s, fa->view.shape);
                    da_id = fuse_or_reduce(ctx, da_lazy);
                }
                grad_accum(ctx, ga, ma_id, da_id);
            }

            // d_b = mul_reduce_sum(gy_expanded, a, db_axes)
            if (fb->requires_grad || ga[mb_id]) {
                u32 db_axes[MAX_DIM], db_n = 0;
                u32 db_dims[MAX_DIM], db_strides_gy[MAX_DIM], db_strides_a[MAX_DIM];
                for (u32 d = 0; d < bc_ndim; d++) {
                    if (is_reduce[d] || bv_bc.strides[d] == 0) {
                        db_axes[db_n] = d;
                        db_dims[db_n] = bc_shape[d];
                        db_strides_gy[db_n] = (u32)(gyv->strides[d] > 0 ? gyv->strides[d] : 0);
                        db_strides_a[db_n] = (u32)(av_bc.strides[d] > 0 ? av_bc.strides[d] : 0);
                        db_n++;
                    }
                }
                u32 db_id = tensor_create(ctx, fb->view.shape, fb->dtype);
                View db_ov = view_create(fb->view.shape);
                #ifdef __APPLE__
                if (ctx->backend == &metal_backend) {
                    metal_mul_reduce_sum(
                        ctx->tensors[db_id].buf_id, fb->view.numel,
                        ctx->tensors[gy_exp_id].buf_id, gyv,
                        fa->buf_id, &av_bc,
                        &db_ov, db_n, db_dims, db_strides_gy, db_strides_a);
                } else
                #endif
                {
                    // CPU fallback: sum(gy_expanded * a, axes_for_b)
                    Shape bc_s = {.rank = bc_ndim};
                    for (u32 d2 = 0; d2 < bc_ndim; d2++) bc_s.dims[d2] = bc_shape[d2];
                    Term db_lazy = sum_to_shape(ctx,
                        thvm_op(ctx, UOP_MUL, term_ten(gy_exp_id, DTYPE_F32), term_ten(ma_id, DTYPE_F32)),
                        bc_s, fb->view.shape);
                    db_id = fuse_or_reduce(ctx, db_lazy);
                }
                grad_accum(ctx, ga, mb_id, db_id);
            }
            break;
        }
        default:
            break;
    }
}

void thvm_backward(TinyHVM *ctx, Term loss, Term *params, Term *grads, u32 n_params) {
    // Reduce loss to get tensor id
    Term loss_r = thvm_reduce(ctx, loss);
    if (term_tag(loss_r) != TAG_TEN) {
        for (u32 i = 0; i < n_params; i++) grads[i] = term_era();
        return;
    }
    u32 loss_id = (u32)term_val(loss_r);

    // Allocate gradient accumulator: ga[tensor_id] = gradient tensor id (0 = none)
    u32 ga_size = loss_id + 1;
    u32 *ga = calloc(ga_size, sizeof(u32));

    // Seed: grad of loss w.r.t. itself = 1
    f32 one = 1.0f;
    ga[loss_id] = (u32)term_val(thvm_tensor(ctx, &one, SHAPE(1)));

    // Single reverse pass
    for (u32 i = loss_id; i > 0; i--) {
        if (!ga[i]) continue;
        if (!ctx->tensors[i].creator_op) continue;
        backward_local(ctx, i, ga[i], ga);
    }

    // Extract gradients for requested parameters
    for (u32 p = 0; p < n_params; p++) {
        if (term_tag(params[p]) != TAG_TEN) { grads[p] = term_era(); continue; }
        u32 pid = (u32)term_val(params[p]);
        if (pid < ga_size && ga[pid])
            grads[p] = term_ten(ga[pid], DTYPE_F32);
        else
            grads[p] = term_era();
    }

    free(ga);
}

// ============================================================
// Profiling — dispatch to backend
// ============================================================

void thvm_profile_report(TinyHVM *ctx) {
    if (ctx->backend && ctx->backend->profile_report)
        ctx->backend->profile_report();
}

void thvm_profile_reset(TinyHVM *ctx) {
    if (ctx->backend && ctx->backend->profile_reset)
        ctx->backend->profile_reset();
}


// ============================================================
// Eval helpers — argmax + accuracy
// ============================================================

Term thvm_argmax(TinyHVM *ctx, Term x, u32 rows, u32 cols) {
    x = thvm_reduce(ctx, x);
    f32 *data = thvm_to_host(ctx, x);

    u32 *preds = malloc(rows * sizeof(u32));
    for (u32 i = 0; i < rows; i++) {
        u32 best = 0;
        f32 mv = data[i * cols];
        for (u32 j = 1; j < cols; j++) {
            if (data[i * cols + j] > mv) {
                mv = data[i * cols + j];
                best = j;
            }
        }
        preds[i] = best;
    }

    u32 id = ctx->tensor_count++;
    u32 buf = ctx->backend->buf_alloc(rows * sizeof(u32));
    ctx->tensors[id] = (TensorMeta){
        .buf_id = buf, .dtype = DTYPE_U32,
        .view = view_create(SHAPE(rows)),
    };
    ctx->backend->buf_write(buf, preds, rows * sizeof(u32));
    free(preds);
    return term_ten(id, DTYPE_U32);
}

f32 thvm_eval_accuracy(TinyHVM *ctx, Term logits, const u8 *labels,
                       u32 n_samples, u32 n_classes) {
    logits = thvm_reduce(ctx, logits);
    f32 *data = thvm_to_host(ctx, logits);

    u32 correct = 0;
    for (u32 i = 0; i < n_samples; i++) {
        u32 best = 0;
        f32 mv = data[i * n_classes];
        for (u32 j = 1; j < n_classes; j++) {
            if (data[i * n_classes + j] > mv) {
                mv = data[i * n_classes + j];
                best = j;
            }
        }
        if (best == labels[i]) correct++;
    }
    return 100.0f * (f32)correct / (f32)n_samples;
}


// ============================================================
// inet.c — Interaction combinator API
// ============================================================

Term thvm_lam(TinyHVM *ctx, Term *var_out, Term body) {
    u64 loc = heap_alloc(ctx, 2);
    Term var = term_new(TAG_VAR, 0, loc);
    heap_set(ctx, loc,     term_set_sub(var));
    heap_set(ctx, loc + 1, body);
    if (var_out) *var_out = var;
    return term_new(TAG_LAM, 0, loc);
}

Term thvm_app(TinyHVM *ctx, Term fun, Term arg) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     fun);
    heap_set(ctx, loc + 1, arg);
    return term_new(TAG_APP, 0, loc);
}

Term thvm_sup(TinyHVM *ctx, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     a);
    heap_set(ctx, loc + 1, b);
    return term_new(TAG_SUP, 0, loc);
}

u32 thvm_define(TinyHVM *ctx, Term body) {
    assert(ctx->def_count < 256);
    u32 name = ctx->def_count++;
    ctx->defs[name] = body;
    return name;
}

Term thvm_ref(TinyHVM *ctx, u32 name) {
    return term_new(TAG_REF, name, 0);
}

Term thvm_where(TinyHVM *ctx, Term cond, Term then_t, Term else_t) {
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     cond);
    heap_set(ctx, loc + 1, then_t);
    heap_set(ctx, loc + 2, else_t);
    return term_new(TAG_TOP, UOP_WHERE, loc);
}

Term thvm_assign(TinyHVM *ctx, Term dst, Term src) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     dst);
    heap_set(ctx, loc + 1, src);
    return term_new(TAG_TOP, UOP_ASSIGN, loc);
}

Term thvm_ifz(TinyHVM *ctx, Term counter, Term zero_case, Term succ_lam) {
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     counter);
    heap_set(ctx, loc + 1, zero_case);
    heap_set(ctx, loc + 2, succ_lam);
    return term_new(TAG_TOP, UOP_IFZ, loc);
}

Term thvm_log_print(TinyHVM *ctx, Term tensor) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     tensor);
    heap_set(ctx, loc + 1, term_era());
    return term_new(TAG_TOP, UOP_LOG_PRINT, loc);
}
