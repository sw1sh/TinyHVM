// grad/_.c — Autograd: thvm_grad, backward_local, thvm_backward
//
// Pure gradient computation. Fusion lives in fuse/_.c.

Term thvm_grad(TinyHVM *ctx, Term y, Term x) {
    f32 one = 1.0f;
    Term seed = thvm_tensor(ctx, &one, SHAPE(1));
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     y);
    heap_set(ctx, loc + 1, seed);
    heap_set(ctx, loc + 2, x);
    return term_new(TAG_TOP, UOP_GRAD, loc);
}

// Multi-target GRAD: compute gradients for ALL params in ONE backward walk.
// Reduces dispatch count from O(N×D) to O(D) for N params.
void thvm_grad_all(TinyHVM *ctx, Term loss, Term *params, Term *grads_out, u32 n_params) {
    // Initialize results to ERA (no gradient yet)
    for (u32 i = 0; i < n_params; i++) grads_out[i] = term_era();
    // Set multi-target state on context
    ctx->grad_params = params;
    ctx->grad_results = grads_out;
    ctx->grad_n_params = n_params;
    // Single GRAD walk with x=ERA (multi-target mode)
    f32 one = 1.0f;
    Term seed = thvm_tensor(ctx, &one, SHAPE(1));
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     loss);
    heap_set(ctx, loc + 1, seed);
    heap_set(ctx, loc + 2, term_era());  // x=ERA → multi-target
    thvm_reduce(ctx, term_new(TAG_TOP, UOP_GRAD, loc));
    // Clear multi-target state
    ctx->grad_params = NULL;
    ctx->grad_results = NULL;
    ctx->grad_n_params = 0;
}

// ============================================================
// Single-pass backward (reverse tape walk)
// ============================================================

static u32 reduce_id(TinyHVM *ctx, Term t) {
    t = thvm_reduce(ctx, t);
    return (term_tag(t) == TAG_TEN) ? (u32)term_val(t) : ~0u;
}

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

static void backward_local(TinyHVM *ctx, u32 y_id, u32 gy_id, u32 *ga) {
    TensorMeta *my = &ctx->tensors[y_id];
    u32 cop = my->creator_op;
    if (!cop) return;  // leaf — no backward

    u32 aid = my->src_ids[0], bid = my->src_ids[1];
    ENSURE(ctx, gy_id); ENSURE(ctx, aid); if (bid) ENSURE(ctx, bid);
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
