// fuse/materialize.c — Materialize lazy tensor chains at boundaries
//
// Walk provenance chain, collect ops + leaf buffers, compile fused kernel, dispatch.
// Shared intermediates (defer_consumers > 0) get side-output buffers written by
// the same fused kernel — no separate dispatch needed.

// Walk provenance chain. op_tids[i] = tensor ID for op i.
static int materialize_walk(TinyHVM *ctx, u32 tid,
                             FusedOp *ops, u32 *n_ops, u32 *op_tids,
                             u32 *leaf_ids, const View **leaf_views, u32 *n_leaves) {
    TensorMeta *m = &ctx->tensors[tid];

    if (m->buf_id != 0) {
        for (u32 i = 0; i < *n_leaves; i++)
            if (leaf_ids[i] == tid) return (int)i;
        if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
        u32 idx = (*n_leaves)++;
        leaf_ids[idx] = tid;
        leaf_views[idx] = &m->view;
        return (int)idx;
    }

    u32 uop = m->creator_op;

    if (is_view_op(uop)) {
        u32 base = m->src_ids[0];
        if (base && ctx->tensors[base].buf_id == 0 && ctx->tensors[base].creator_op)
            tensor_materialize(ctx, base);
        if (base && ctx->tensors[base].buf_id != 0) {
            m->buf_id = ctx->tensors[base].buf_id;
            return -2;
        }
        return -1;
    }

    if (!is_elementwise(uop)) return -1;

    int arg_a = materialize_walk(ctx, m->src_ids[0], ops, n_ops, op_tids, leaf_ids, leaf_views, n_leaves);
    if (arg_a == -2)
        arg_a = materialize_walk(ctx, m->src_ids[0], ops, n_ops, op_tids, leaf_ids, leaf_views, n_leaves);
    if (arg_a < 0) return -1;

    int arg_b = -1;
    if (is_binary(uop)) {
        arg_b = materialize_walk(ctx, m->src_ids[1], ops, n_ops, op_tids, leaf_ids, leaf_views, n_leaves);
        if (arg_b == -2)
            arg_b = materialize_walk(ctx, m->src_ids[1], ops, n_ops, op_tids, leaf_ids, leaf_views, n_leaves);
        if (arg_b < 0) return -1;
    }

    if (*n_ops >= FUSE_MAX_OPS) return -1;
    u32 op_idx = (*n_ops)++;
    ops[op_idx] = (FusedOp){ .uop = uop, .arg_a = (u32)arg_a, .arg_b = is_binary(uop) ? (u32)arg_b : 0 };
    op_tids[op_idx] = tid;

    // Shared intermediate: allocate side output buffer
    if (m->defer_consumers > 0 && m->buf_id == 0)
        m->buf_id = m->backend->buf_alloc(m->view.numel * sizeof(f32));

    return (int)(FUSE_MAX_LEAVES + op_idx);
}

static void tensor_materialize(TinyHVM *ctx, u32 tid) {
    TensorMeta *m = &ctx->tensors[tid];
    if (m->buf_id != 0) return;

    // Deferred reduce (SUM/RMAX): fuse reduce + elementwise input chain
    u32 reduce_type = 0, reduce_axes_id = 0, walk_tid = tid;
    if ((m->creator_op == UOP_SUM || m->creator_op == UOP_RMAX) && m->src_ids[0] &&
        ctx->tensors[m->src_ids[0]].buf_id == 0 &&
        is_elementwise(ctx->tensors[m->src_ids[0]].creator_op)) {
        reduce_type = m->creator_op;
        reduce_axes_id = m->src_ids[1];
        walk_tid = m->src_ids[0];
    }

    FusedOp ops[FUSE_MAX_OPS]; u32 n_ops = 0; u32 op_tids[FUSE_MAX_OPS];
    u32 leaf_ids[FUSE_MAX_LEAVES]; const View *leaf_views[FUSE_MAX_LEAVES]; u32 n_leaves = 0;

    int result = materialize_walk(ctx, walk_tid, ops, &n_ops, op_tids, leaf_ids, leaf_views, &n_leaves);
    if (m->buf_id != 0) return;
    if (result < 0 || n_ops == 0) {
        if (reduce_type) tensor_materialize(ctx, m->src_ids[0]);
        m->buf_id = m->backend->buf_alloc(m->view.numel * sizeof(f32));
        return;
    }

    // Remap op indices
    for (u32 i = 0; i < n_ops; i++) {
        if (ops[i].arg_a >= FUSE_MAX_LEAVES) ops[i].arg_a = n_leaves + (ops[i].arg_a - FUSE_MAX_LEAVES);
        if (ops[i].arg_b >= FUSE_MAX_LEAVES) ops[i].arg_b = n_leaves + (ops[i].arg_b - FUSE_MAX_LEAVES);
    }

    m->buf_id = m->backend->buf_alloc(m->view.numel * sizeof(f32));

    // Collect side outputs (shared intermediates with allocated buffers)
    u32 side_bufs[8]; u32 side_ops[8]; u32 n_sides = 0;
    for (u32 i = 0; i < n_ops && n_sides < 8; i++) {
        TensorMeta *sm = &ctx->tensors[op_tids[i]];
        if (sm->buf_id != 0 && sm->defer_consumers > 0 && op_tids[i] != walk_tid) {
            side_bufs[n_sides] = sm->buf_id;
            side_ops[n_sides] = n_leaves + i; // remapped op index
            n_sides++;
        }
    }

    if (m->backend->dispatch_kernel_rs) {
        u32 bufs[FUSE_MAX_LEAVES];
        for (u32 i = 0; i < n_leaves; i++) bufs[i] = ctx->tensors[leaf_ids[i]].buf_id;

        if (reduce_type) {
            ReduceSpec rs = {0};
            rs.reduce_type = reduce_type;
            Shape full_shape = ctx->tensors[walk_tid].view.shape;
            if (reduce_axes_id) {
                TensorMeta *axt = &ctx->tensors[reduce_axes_id];
                u32 n_ax = axt->view.numel;
                f32 af[MAX_DIM]; META_READ(axt->backend, axt->buf_id, af, n_ax * 4);
                for (u32 i = 0; i < n_ax; i++) {
                    u32 ax = (u32)af[i];
                    if (ax < full_shape.rank) rs.is_reduce[ax] = 1;
                }
            } else {
                for (int d = (int)full_shape.rank - 1; d >= 0; d--)
                    if (full_shape.dims[d] > 1) { rs.is_reduce[d] = 1; break; }
            }
            m->backend->dispatch_kernel_rs(m->buf_id, bufs, leaf_views, n_leaves,
                                            ops, n_ops, &full_shape, &rs,
                                            n_sides ? side_bufs : NULL,
                                            n_sides ? side_ops : NULL, n_sides);
        } else {
            m->backend->dispatch_kernel_rs(m->buf_id, bufs, leaf_views, n_leaves,
                                            ops, n_ops, &m->view.shape, NULL,
                                            side_bufs, side_ops, n_sides);
        }
        return;
    }

    // CPU fallback: execute ops sequentially (no fusing)
    // temp_bufs[0..n_leaves-1] = leaf buf_ids, [n_leaves..] = intermediate results
    u32 temp_bufs[FUSE_MAX_LEAVES + FUSE_MAX_OPS];
    const View *temp_views[FUSE_MAX_LEAVES + FUSE_MAX_OPS];
    for (u32 i = 0; i < n_leaves; i++) {
        temp_bufs[i] = ctx->tensors[leaf_ids[i]].buf_id;
        temp_views[i] = &ctx->tensors[leaf_ids[i]].view;
    }
    for (u32 i = 0; i < n_ops; i++) {
        u32 op_tid = op_tids[i];
        TensorMeta *om = &ctx->tensors[op_tid];
        // Allocate intermediate buffer if not the final output and not already allocated
        u32 dst_buf;
        if (op_tid == walk_tid) {
            dst_buf = m->buf_id;
        } else if (om->buf_id != 0) {
            dst_buf = om->buf_id;
        } else {
            dst_buf = m->backend->buf_alloc(om->view.numel * sizeof(f32));
            om->buf_id = dst_buf;
        }
        View dst_view = om->view;
        u32 a_buf = temp_bufs[ops[i].arg_a];
        const View *a_view = temp_views[ops[i].arg_a];
        if (is_binary(ops[i].uop)) {
            u32 b_buf = temp_bufs[ops[i].arg_b];
            const View *b_view = temp_views[ops[i].arg_b];
            // Broadcast views to match output shape for CPU strided indexing
            View av_bc, bv_bc; u32 bc_shape[MAX_DIM], bc_ndim;
            if (view_broadcast(a_view, b_view, &av_bc, &bv_bc, bc_shape, &bc_ndim))
                m->backend->op_binary(ops[i].uop, dst_buf, &dst_view,
                                      a_buf, &av_bc, b_buf, &bv_bc);
            else
                m->backend->op_binary(ops[i].uop, dst_buf, &dst_view,
                                      a_buf, a_view, b_buf, b_view);
        } else {
            // For unary ops, input view might not match output shape.
            // Use output shape with input strides for correct indexing.
            View uv = *a_view;
            uv.shape = dst_view.shape;
            uv.numel = dst_view.numel;
            m->backend->op_unary(ops[i].uop, dst_buf, &dst_view,
                                 a_buf, &uv);
        }
        temp_bufs[n_leaves + i] = dst_buf;
        temp_views[n_leaves + i] = &ctx->tensors[op_tid].view;
    }
}

// Eager reduce fusion: walk deferred ew chain at `input_tid`, dispatch fused reduce+ew
// into `out_buf`. Returns 1 if dispatched, 0 if fusion not possible.
static int tensor_materialize_reduce(TinyHVM *ctx, u32 input_tid, u32 out_buf,
                                      const ReduceSpec *rs) {
    TensorMeta *im = &ctx->tensors[input_tid];
    if (im->buf_id != 0 || !is_elementwise(im->creator_op)) return 0;

    FusedOp ops[FUSE_MAX_OPS]; u32 n_ops = 0, op_tids[FUSE_MAX_OPS];
    u32 leaf_ids[FUSE_MAX_LEAVES]; const View *leaf_views[FUSE_MAX_LEAVES]; u32 n_leaves = 0;

    int result = materialize_walk(ctx, input_tid, ops, &n_ops, op_tids,
                                   leaf_ids, leaf_views, &n_leaves);
    if (result < 0 || n_ops == 0) return 0;

    // Remap op indices
    for (u32 i = 0; i < n_ops; i++) {
        if (ops[i].arg_a >= FUSE_MAX_LEAVES) ops[i].arg_a = n_leaves + (ops[i].arg_a - FUSE_MAX_LEAVES);
        if (ops[i].arg_b >= FUSE_MAX_LEAVES) ops[i].arg_b = n_leaves + (ops[i].arg_b - FUSE_MAX_LEAVES);
    }

    // Collect side outputs (shared intermediates)
    u32 side_bufs[8], side_ops[8]; u32 n_sides = 0;
    for (u32 i = 0; i < n_ops && n_sides < 8; i++) {
        TensorMeta *sm = &ctx->tensors[op_tids[i]];
        if (sm->buf_id != 0 && sm->defer_consumers > 0) {
            side_bufs[n_sides] = sm->buf_id;
            side_ops[n_sides] = n_leaves + i;
            n_sides++;
        }
    }

    if (im->backend->dispatch_kernel_rs) {
        u32 bufs[FUSE_MAX_LEAVES];
        for (u32 i = 0; i < n_leaves; i++) bufs[i] = ctx->tensors[leaf_ids[i]].buf_id;
        Shape full_shape = im->view.shape;
        im->backend->dispatch_kernel_rs(out_buf, bufs, leaf_views, n_leaves,
                                         ops, n_ops, &full_shape, rs,
                                         n_sides ? side_bufs : NULL,
                                         n_sides ? side_ops : NULL, n_sides);
        return 1;
    }
    return 0;
}
