// fuse/materialize.c — Materialize lazy tensor chains at boundaries
//
// When a boundary op (SUM, MM, thvm_to_host) needs data from a lazy
// tensor (buf_id == 0), walk the provenance chain (creator_op/src_ids),
// collect ops + leaf buffers, compile a fused kernel, dispatch.

// Walk provenance chain, collect ops and leaves into arrays.
// Returns number of ops, or -1 on failure.
static int materialize_walk(TinyHVM *ctx, u32 tid,
                             FusedOp *ops, u32 *n_ops,
                             u32 *leaf_ids, const View **leaf_views, u32 *n_leaves) {
    TensorMeta *m = &ctx->tensors[tid];

    // Real tensor (has buffer): it's a leaf
    if (m->buf_id != 0) {
        for (u32 i = 0; i < *n_leaves; i++)
            if (leaf_ids[i] == tid) return (int)i; // dedup
        if (*n_leaves >= FUSE_MAX_LEAVES) return -1;
        u32 idx = (*n_leaves)++;
        leaf_ids[idx] = tid;
        leaf_views[idx] = &m->view;
        return (int)idx;
    }

    // Lazy tensor (buf_id == 0): it's an op in the chain
    u32 uop = m->creator_op;
    if (!is_elementwise(uop)) return -1; // boundary — shouldn't be lazy

    // Recurse into inputs
    int arg_a = materialize_walk(ctx, m->src_ids[0], ops, n_ops, leaf_ids, leaf_views, n_leaves);
    if (arg_a < 0) return -1;

    int arg_b = -1;
    if (is_binary(uop) && m->src_ids[1]) {
        arg_b = materialize_walk(ctx, m->src_ids[1], ops, n_ops, leaf_ids, leaf_views, n_leaves);
        if (arg_b < 0) return -1;
    }

    if (*n_ops >= FUSE_MAX_OPS) return -1;
    u32 op_idx = (*n_ops)++;
    // Leaf references use raw index, op references use n_leaves + op_idx
    ops[op_idx] = (FusedOp){
        .uop = uop,
        .arg_a = (u32)arg_a,
        .arg_b = is_binary(uop) ? (u32)arg_b : 0
    };
    return (int)(*n_leaves + op_idx); // op var index
}

// Materialize a lazy tensor: compile + dispatch its provenance chain.
// Allocates buffer, fills with computed data.
static void tensor_materialize(TinyHVM *ctx, u32 tid) {
    TensorMeta *m = &ctx->tensors[tid];
    if (m->buf_id != 0) return; // already materialized

    FusedOp ops[FUSE_MAX_OPS]; u32 n_ops = 0;
    u32 leaf_ids[FUSE_MAX_LEAVES]; const View *leaf_views[FUSE_MAX_LEAVES]; u32 n_leaves = 0;

    int result = materialize_walk(ctx, tid, ops, &n_ops, leaf_ids, leaf_views, &n_leaves);
    if (result < 0 || n_ops == 0) {
        // Can't fuse — allocate buffer and zero-fill
        m->buf_id = ctx->backend->buf_alloc(m->view.numel * sizeof(f32));
        return;
    }

    // Remap: materialize_walk returns leaf indices directly (0..n_leaves-1)
    // and op indices as n_leaves + op_idx. The codegen expects this format.

    // Allocate output buffer
    m->buf_id = ctx->backend->buf_alloc(m->view.numel * sizeof(f32));

    // Also allocate buffers for any lazy intermediate tensors in the chain
    // (virtual intermediates created by previous fuse_or_reduce calls)
    // Actually, intermediates in this chain have buf_id=0 too. They share
    // the walk — only the final result needs a buffer. Intermediates are
    // virtual (backward reads src_ids, not data).

    #ifdef __APPLE__
    if (ctx->backend == &metal_backend) {
        u32 bufs[FUSE_MAX_LEAVES];
        for (u32 i = 0; i < n_leaves; i++) bufs[i] = ctx->tensors[leaf_ids[i]].buf_id;
        metal_dispatch_fused_v2(m->buf_id, m->view.numel,
                                  bufs, leaf_views, n_leaves, ops, n_ops,
                                  0, 0, &m->view.shape);
    }
    #endif
}
