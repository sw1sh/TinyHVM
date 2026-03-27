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

    // View ops: materialize base, share its buffer
    if (is_view_op(uop)) {
        u32 base = m->src_ids[0];
        if (base && ctx->tensors[base].buf_id == 0 && ctx->tensors[base].creator_op)
            tensor_materialize(ctx, base);
        if (base && ctx->tensors[base].buf_id != 0) {
            m->buf_id = ctx->tensors[base].buf_id;
            return -2; // special: buf_id set but not an op
        }
        return -1;
    }

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
    // Use FUSE_MAX_LEAVES as fixed offset for op indices so they don't shift
    // as more leaves are added. Remapped to n_leaves-based indices after walk.
    ops[op_idx] = (FusedOp){
        .uop = uop,
        .arg_a = (u32)arg_a,
        .arg_b = is_binary(uop) ? (u32)arg_b : 0
    };
    return (int)(FUSE_MAX_LEAVES + op_idx);
}

// Materialize a lazy tensor: compile + dispatch its provenance chain.
// Allocates buffer, fills with computed data.
static void tensor_materialize(TinyHVM *ctx, u32 tid) {
    TensorMeta *m = &ctx->tensors[tid];
    if (m->buf_id != 0) return;
    static int _mc = 0; _mc++;

    FusedOp ops[FUSE_MAX_OPS]; u32 n_ops = 0;
    u32 leaf_ids[FUSE_MAX_LEAVES]; const View *leaf_views[FUSE_MAX_LEAVES]; u32 n_leaves = 0;

    int result = materialize_walk(ctx, tid, ops, &n_ops, leaf_ids, leaf_views, &n_leaves);
    if (m->buf_id != 0) return; // view op handler set buf_id directly
    if (result < 0 || n_ops == 0) {
        m->buf_id = ctx->backend->buf_alloc(m->view.numel * sizeof(f32));
        return;
    }

    // Remap op indices: walk used FUSE_MAX_LEAVES as fixed offset to avoid
    // shifting as leaves were added. Codegen expects n_leaves-based offsets.
    for (u32 i = 0; i < n_ops; i++) {
        if (ops[i].arg_a >= FUSE_MAX_LEAVES)
            ops[i].arg_a = n_leaves + (ops[i].arg_a - FUSE_MAX_LEAVES);
        if (ops[i].arg_b >= FUSE_MAX_LEAVES)
            ops[i].arg_b = n_leaves + (ops[i].arg_b - FUSE_MAX_LEAVES);
    }

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
