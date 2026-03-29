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
            if (m->backend && m->backend->buf_incref)
                m->backend->buf_incref(m->buf_id);
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

    // NOTE: side output buffer allocation moved to tensor_materialize
    // (after walk succeeds) to avoid zombie buffers from failed walks.

    return (int)(FUSE_MAX_LEAVES + op_idx);
}

// Forward declaration for graph-level materialize
static void tensor_materialize_graph(TinyHVM *ctx, u32 tid);

// Chain-level materialize (original implementation)
static void tensor_materialize_chain(TinyHVM *ctx, u32 tid) {
    TensorMeta *m = &ctx->tensors[tid];
    if (m->buf_id != 0) return;

    // Post-reduce fusion: if this deferred ew chain contains a deferred reduce
    // (reachable via views), fuse pre-reduce ew → reduce → post-reduce ew in one kernel.
    if (is_elementwise(m->creator_op) && m->backend->dispatch_kernel_rs) {
        // Scan chain for a deferred reduce (follow src[0] through ew + views)
        u32 reduce_tid = 0, reduce_input = 0;
        u32 cur = tid;
        for (u32 depth = 0; depth < 20; depth++) {
            TensorMeta *cm = &ctx->tensors[cur];
            if (cm->buf_id != 0) break;
            if (cm->creator_op == UOP_SUM || cm->creator_op == UOP_RMAX) {
                reduce_tid = cur; reduce_input = cm->src_ids[0]; break;
            }
            if (is_elementwise(cm->creator_op) || is_view_op(cm->creator_op))
                cur = cm->src_ids[0];
            else break;
        }
        if (reduce_tid && reduce_input &&
            ctx->tensors[reduce_input].buf_id == 0 &&
            is_elementwise(ctx->tensors[reduce_input].creator_op)) {
            // Walk pre-reduce ew chain
            FusedOp ops[FUSE_MAX_OPS]; u32 n_ops = 0, op_tids[FUSE_MAX_OPS];
            u32 leaf_ids[FUSE_MAX_LEAVES]; const View *leaf_views[FUSE_MAX_LEAVES]; u32 n_leaves = 0;
            int pre_res = materialize_walk(ctx, reduce_input, ops, &n_ops, op_tids,
                                            leaf_ids, leaf_views, &n_leaves);
            if (pre_res >= 0 && n_ops > 0) {
                u32 pre_ops = n_ops, pre_leaves = n_leaves;
                // Collect post-reduce ew ops (from reduce output to tid)
                // Walk backward from tid, collecting ops above the reduce
                u32 post_chain[16]; u32 pcn = 0;
                cur = tid;
                while (cur != reduce_tid && pcn < 16) {
                    TensorMeta *cm = &ctx->tensors[cur];
                    if (is_elementwise(cm->creator_op)) {
                        post_chain[pcn++] = cur;
                        cur = cm->src_ids[0]; // follow toward reduce
                    } else if (is_view_op(cm->creator_op)) {
                        cur = cm->src_ids[0]; // skip views
                    } else break;
                }
                // Add post-reduce ops + leaves (reverse order: bottom-up)
                u32 n_post_leaves = 0;
                for (int pi = (int)pcn - 1; pi >= 0; pi--) {
                    TensorMeta *pm = &ctx->tensors[post_chain[pi]];
                    if (is_binary(pm->creator_op) && pm->src_ids[1]) {
                        u32 other = pm->src_ids[1]; // non-chain input (e.g., bias)
                        ENSURE(ctx, other);
                        if (ctx->tensors[other].buf_id != 0 && n_leaves < FUSE_MAX_LEAVES) {
                            leaf_ids[n_leaves] = other;
                            leaf_views[n_leaves] = &ctx->tensors[other].view;
                            ops[n_ops] = (FusedOp){ .uop = pm->creator_op, .arg_a = 0, .arg_b = n_leaves };
                            op_tids[n_ops] = post_chain[pi];
                            n_leaves++; n_post_leaves++; n_ops++;
                        }
                    } else {
                        ops[n_ops] = (FusedOp){ .uop = pm->creator_op, .arg_a = 0, .arg_b = 0 };
                        op_tids[n_ops] = post_chain[pi];
                        n_ops++;
                    }
                }
                if (n_ops > pre_ops) {
                    // Remap pre-reduce op indices
                    for (u32 i = 0; i < pre_ops; i++) {
                        if (ops[i].arg_a >= FUSE_MAX_LEAVES) ops[i].arg_a = n_leaves + (ops[i].arg_a - FUSE_MAX_LEAVES);
                        if (ops[i].arg_b >= FUSE_MAX_LEAVES) ops[i].arg_b = n_leaves + (ops[i].arg_b - FUSE_MAX_LEAVES);
                    }
                    // Side outputs for pre-reduce shared intermediates
                    u32 side_bufs[8], side_ops[8]; u32 n_sides = 0;
                    for (u32 i = 0; i < pre_ops && n_sides < 8; i++) {
                        TensorMeta *sm = &ctx->tensors[op_tids[i]];
                        if (sm->buf_id != 0 && sm->defer_consumers > 0) {
                            side_bufs[n_sides] = sm->buf_id; side_ops[n_sides] = n_leaves + i; n_sides++;
                        }
                    }
                    // Build ReduceSpec with post-reduce boundary
                    TensorMeta *rm = &ctx->tensors[reduce_tid];
                    ReduceSpec rs = {0};
                    rs.reduce_type = rm->creator_op;
                    rs.post_reduce_start = pre_ops;
                    rs.n_post_leaves = n_post_leaves;
                    u32 axes_id = rm->src_ids[1];
                    Shape fs = ctx->tensors[reduce_input].view.shape;
                    if (axes_id) {
                        ENSURE(ctx, axes_id);
                        TensorMeta *axt = &ctx->tensors[axes_id];
                        f32 af[MAX_DIM]; META_READ(axt->backend, axt->buf_id, af, axt->view.numel*4);
                        for (u32 i=0;i<axt->view.numel;i++) { u32 ax=(u32)af[i]; if(ax<fs.rank) rs.is_reduce[ax]=1; }
                    } else {
                        for (int d=(int)fs.rank-1;d>=0;d--) if(fs.dims[d]>1){rs.is_reduce[d]=1;break;}
                    }
                    m->buf_id = m->backend->buf_alloc(m->view.numel * sizeof(f32));
                    u32 bufs[FUSE_MAX_LEAVES];
                    for (u32 i = 0; i < n_leaves; i++) bufs[i] = ctx->tensors[leaf_ids[i]].buf_id;
                    m->backend->dispatch_kernel_rs(m->buf_id, bufs, leaf_views, n_leaves,
                        ops, n_ops, &fs, &rs, n_sides?side_bufs:NULL, n_sides?side_ops:NULL, n_sides);
                    return;
                }
            }
        }
    }

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
        // Post-reduce fusion: if the walk failed because it hit a deferred reduce,
        // try to fuse: pre-reduce ew → reduce → post-reduce ew in one kernel.
        if (!reduce_type && is_elementwise(m->creator_op) && m->backend->dispatch_kernel_rs) {
            // Scan for deferred reduce in the ew chain inputs
            u32 post_tid = tid;  // the ew chain root
            u32 reduce_tid = 0;
            // Walk the deferred ew chain to find the reduce boundary
            u32 cur = tid;
            while (ctx->tensors[cur].buf_id == 0 && is_elementwise(ctx->tensors[cur].creator_op)) {
                u32 s0 = ctx->tensors[cur].src_ids[0];
                u32 s1 = ctx->tensors[cur].src_ids[1];
                if (s0 && ctx->tensors[s0].buf_id == 0 &&
                    (ctx->tensors[s0].creator_op == UOP_SUM || ctx->tensors[s0].creator_op == UOP_RMAX)) {
                    reduce_tid = s0; break;
                }
                if (s1 && ctx->tensors[s1].buf_id == 0 &&
                    (ctx->tensors[s1].creator_op == UOP_SUM || ctx->tensors[s1].creator_op == UOP_RMAX)) {
                    reduce_tid = s1; break;
                }
                // Follow the deferred chain (first deferred input)
                if (s0 && ctx->tensors[s0].buf_id == 0 && is_elementwise(ctx->tensors[s0].creator_op))
                    cur = s0;
                else break;
            }

            if (reduce_tid && ctx->tensors[reduce_tid].src_ids[0]) {
                TensorMeta *rm = &ctx->tensors[reduce_tid];
                u32 reduce_input = rm->src_ids[0];
                // Walk the pre-reduce ew chain
                if (ctx->tensors[reduce_input].buf_id == 0 &&
                    is_elementwise(ctx->tensors[reduce_input].creator_op)) {
                    n_ops = 0; n_leaves = 0;
                    int pre_result = materialize_walk(ctx, reduce_input, ops, &n_ops, op_tids,
                                                       leaf_ids, leaf_views, &n_leaves);
                    if (pre_result >= 0 && n_ops > 0) {
                        u32 pre_reduce_ops = n_ops;
                        u32 pre_reduce_leaves = n_leaves;

                        // Now add post-reduce ops (the ew chain from reduce output to tid)
                        // Walk from tid backward, collecting ew ops until we hit reduce_tid
                        // For simplicity: walk post-reduce chain iteratively
                        u32 post_chain[16]; u32 pcn = 0;
                        cur = tid;
                        while (cur != reduce_tid && pcn < 16) {
                            if (!is_elementwise(ctx->tensors[cur].creator_op)) break;
                            post_chain[pcn++] = cur;
                            // Follow toward reduce
                            u32 s0 = ctx->tensors[cur].src_ids[0];
                            if (s0 == reduce_tid || (s0 && ctx->tensors[s0].buf_id == 0 &&
                                is_elementwise(ctx->tensors[s0].creator_op))) {
                                cur = s0;
                            } else break;
                        }

                        // Add post-reduce leaves (non-reduce inputs to post ops)
                        u32 n_post_leaves = 0;
                        for (int pi = (int)pcn - 1; pi >= 0; pi--) {
                            u32 ptid = post_chain[pi];
                            TensorMeta *pm = &ctx->tensors[ptid];
                            // For binary post-ops: the non-reduce input is a leaf
                            if (is_binary(pm->creator_op)) {
                                u32 other = (pm->src_ids[0] == reduce_tid ||
                                    (pm->src_ids[0] && ctx->tensors[pm->src_ids[0]].buf_id == 0 &&
                                     is_elementwise(ctx->tensors[pm->src_ids[0]].creator_op)))
                                    ? pm->src_ids[1] : pm->src_ids[0];
                                if (other && ctx->tensors[other].buf_id == 0)
                                    ENSURE(ctx, other);
                                if (other && ctx->tensors[other].buf_id != 0 && n_leaves < FUSE_MAX_LEAVES) {
                                    leaf_ids[n_leaves] = other;
                                    leaf_views[n_leaves] = &ctx->tensors[other].view;
                                    n_post_leaves++;
                                    // Add post-reduce op
                                    u32 acc_ref = n_leaves + pre_reduce_ops - 1; // last pre-reduce op = acc
                                    if (pi == (int)pcn - 1) acc_ref = pre_reduce_leaves + pre_reduce_ops - 1;
                                    ops[n_ops] = (FusedOp){
                                        .uop = pm->creator_op,
                                        .arg_a = acc_ref, // previous result (acc or last post op)
                                        .arg_b = n_leaves
                                    };
                                    n_leaves++;
                                    op_tids[n_ops] = ptid;
                                    n_ops++;
                                }
                            } else {
                                // Unary post-op
                                u32 prev = (n_ops > pre_reduce_ops) ?
                                    n_leaves + n_ops - 1 : pre_reduce_leaves + pre_reduce_ops - 1;
                                ops[n_ops] = (FusedOp){ .uop = pm->creator_op, .arg_a = prev, .arg_b = 0 };
                                op_tids[n_ops] = ptid;
                                n_ops++;
                            }
                        }

                        if (n_ops > pre_reduce_ops) {
                            // Remap pre-reduce ops
                            for (u32 i = 0; i < pre_reduce_ops; i++) {
                                if (ops[i].arg_a >= FUSE_MAX_LEAVES) ops[i].arg_a = pre_reduce_leaves + (ops[i].arg_a - FUSE_MAX_LEAVES);
                                if (ops[i].arg_b >= FUSE_MAX_LEAVES) ops[i].arg_b = pre_reduce_leaves + (ops[i].arg_b - FUSE_MAX_LEAVES);
                            }
                            // Remap post-reduce ops
                            for (u32 i = pre_reduce_ops; i < n_ops; i++) {
                                if (ops[i].arg_a >= FUSE_MAX_LEAVES) ops[i].arg_a = n_leaves + (ops[i].arg_a - FUSE_MAX_LEAVES);
                                if (ops[i].arg_b >= FUSE_MAX_LEAVES) ops[i].arg_b = n_leaves + (ops[i].arg_b - FUSE_MAX_LEAVES);
                            }

                            // Build ReduceSpec with post-reduce
                            ReduceSpec rs = {0};
                            rs.reduce_type = rm->creator_op;
                            rs.post_reduce_start = pre_reduce_ops;
                            rs.n_post_leaves = n_post_leaves;
                            u32 axes_id = rm->src_ids[1];
                            Shape fs = ctx->tensors[reduce_input].view.shape;
                            if (axes_id) {
                                ENSURE(ctx, axes_id);
                                TensorMeta *axt = &ctx->tensors[axes_id];
                                f32 af[MAX_DIM]; META_READ(axt->backend, axt->buf_id, af, axt->view.numel*4);
                                for (u32 i=0;i<axt->view.numel;i++) { u32 ax=(u32)af[i]; if(ax<fs.rank) rs.is_reduce[ax]=1; }
                            } else {
                                for (int d=(int)fs.rank-1;d>=0;d--) if(fs.dims[d]>1){rs.is_reduce[d]=1;break;}
                            }

                            // Collect side outputs
                            u32 side_bufs2[8], side_ops2[8]; u32 n_sides2 = 0;
                            for (u32 i = 0; i < pre_reduce_ops && n_sides2 < 8; i++) {
                                TensorMeta *sm = &ctx->tensors[op_tids[i]];
                                if (sm->buf_id != 0 && sm->defer_consumers > 0) {
                                    side_bufs2[n_sides2] = sm->buf_id; side_ops2[n_sides2] = pre_reduce_leaves + i; n_sides2++;
                                }
                            }

                            m->buf_id = m->backend->buf_alloc(m->view.numel * sizeof(f32));
                            u32 bufs[FUSE_MAX_LEAVES];
                            for (u32 i = 0; i < n_leaves; i++) bufs[i] = ctx->tensors[leaf_ids[i]].buf_id;
                            m->backend->dispatch_kernel_rs(m->buf_id, bufs, leaf_views, n_leaves,
                                ops, n_ops, &fs, &rs,
                                n_sides2 ? side_bufs2 : NULL, n_sides2 ? side_ops2 : NULL, n_sides2);
                            return;
                        }
                    }
                }
            }
        }

        if (reduce_type) {
            // Fusion failed (input pre-materialized). Dispatch standalone reduce.
            ENSURE(ctx, m->src_ids[0]);
            TensorMeta *ms = &ctx->tensors[m->src_ids[0]];
            m->buf_id = m->backend->buf_alloc(m->view.numel * sizeof(f32));
            if (m->backend->dispatch_kernel_rs) {
                ReduceSpec rs = {0}; rs.reduce_type = reduce_type;
                Shape fs = ms->view.shape;
                if (reduce_axes_id) {
                    ENSURE(ctx, reduce_axes_id);
                    TensorMeta *axt = &ctx->tensors[reduce_axes_id];
                    f32 af[MAX_DIM]; META_READ(axt->backend, axt->buf_id, af, axt->view.numel*4);
                    for (u32 i=0;i<axt->view.numel;i++) { u32 ax=(u32)af[i]; if(ax<fs.rank) rs.is_reduce[ax]=1; }
                } else {
                    for (int d=(int)fs.rank-1;d>=0;d--) if(fs.dims[d]>1){rs.is_reduce[d]=1;break;}
                }
                u32 bufs[]={ms->buf_id}; const View *views[]={&ms->view};
                m->backend->dispatch_kernel_rs(m->buf_id, bufs, views, 1, NULL, 0, &fs, &rs, NULL, NULL, 0);
            }
            return;
        }
        // Walk failed: scan for deferred non-ew ops blocking the chain.
        // Materialize them (deepest first), then retry the walk.
        {
            u32 to_mat[16]; u32 n_mat = 0;
            u32 stk[32]; u32 sn = 0;
            if (m->src_ids[0]) stk[sn++] = m->src_ids[0];
            if (m->src_ids[1]) stk[sn++] = m->src_ids[1];
            while (sn > 0 && n_mat < 16) {
                u32 s = stk[--sn];
                TensorMeta *sm = &ctx->tensors[s];
                if (sm->buf_id != 0 || !sm->creator_op) continue;
                if (is_elementwise(sm->creator_op) || is_view_op(sm->creator_op)) {
                    if (sm->src_ids[0] && sn < 30) stk[sn++] = sm->src_ids[0];
                    if (sm->src_ids[1] && sn < 30) stk[sn++] = sm->src_ids[1];
                } else {
                    to_mat[n_mat++] = s;
                    if (sm->src_ids[0] && sn < 30) stk[sn++] = sm->src_ids[0];
                    if (sm->src_ids[1] && sn < 30) stk[sn++] = sm->src_ids[1];
                }
            }
            for (int i = (int)n_mat - 1; i >= 0; i--)
                if (ctx->tensors[to_mat[i]].buf_id == 0)
                    tensor_materialize(ctx, to_mat[i]);
            if (n_mat > 0) {
                n_ops = 0; n_leaves = 0;
                result = materialize_walk(ctx, walk_tid, ops, &n_ops, op_tids,
                                           leaf_ids, leaf_views, &n_leaves);
                if (m->buf_id != 0) return;
                if (result >= 0 && n_ops > 0) goto dispatch_chain;
            }
        }
        m->buf_id = m->backend->buf_alloc(m->view.numel * sizeof(f32));
        return;
    }

dispatch_chain:
    // Remap op indices
    for (u32 i = 0; i < n_ops; i++) {
        if (ops[i].arg_a >= FUSE_MAX_LEAVES) ops[i].arg_a = n_leaves + (ops[i].arg_a - FUSE_MAX_LEAVES);
        if (ops[i].arg_b >= FUSE_MAX_LEAVES) ops[i].arg_b = n_leaves + (ops[i].arg_b - FUSE_MAX_LEAVES);
    }

    m->buf_id = m->backend->buf_alloc(m->view.numel * sizeof(f32));

    // Allocate side output buffers for shared/grad-needed intermediates (not walk_tid).
    // Moved here from materialize_walk to avoid zombie buffers from failed walks.
    for (u32 i = 0; i < n_ops; i++) {
        TensorMeta *sm = &ctx->tensors[op_tids[i]];
        if (sm->defer_consumers > 0 && sm->buf_id == 0 && op_tids[i] != walk_tid)
            sm->buf_id = sm->backend->buf_alloc(sm->view.numel * sizeof(f32));
    }

    // Collect side outputs (shared intermediates + grad-needed with own buffers)
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

    // CPU: if this was a deferred reduce, dispatch the reduce on the now-materialized input
    if (reduce_type && n_ops > 0 && m->backend->op_reduce) {
        // The ew chain was materialized above. walk_tid's buf_id should now be set.
        TensorMeta *ms = &ctx->tensors[walk_tid];
        if (ms->buf_id == 0 && n_leaves + n_ops > 0)
            ms->buf_id = temp_bufs[n_leaves + n_ops - 1];
        if (ms->buf_id != 0) {
            u32 reduce_dim = 1;
            for (int d = (int)ms->view.shape.rank - 1; d >= 0; d--)
                if (ms->view.shape.dims[d] > 1) { reduce_dim = ms->view.shape.dims[d]; break; }
            m->backend->op_reduce(reduce_type, m->buf_id, m->view.numel,
                                   ms->buf_id, ms->view.numel, reduce_dim);
        }
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

// ════════════════════════════════════════════════════════════════════
// Graph-level materialize: walk entire deferred subgraph, partition
// into optimal fusion groups, dispatch all at once.
// ════════════════════════════════════════════════════════════════════

#define GMAX 512  // max deferred nodes in subgraph

// Discover all deferred tensors reachable from root via src_ids.
// Returns count of discovered deferred nodes (not including leaves).
static u32 graph_discover(TinyHVM *ctx, u32 root, u32 *nodes, u32 *n_nodes) {
    u32 stk[GMAX], sp = 0;
    stk[sp++] = root;
    *n_nodes = 0;
    // Simple visited check via temporary flag (reuse creator_loc as mark)
    // We'll clear it after discovery.
    u32 visited[GMAX]; u32 nv = 0;

    while (sp > 0 && *n_nodes < GMAX) {
        u32 tid = stk[--sp];
        if (!tid) continue;
        TensorMeta *m = &ctx->tensors[tid];
        if (m->buf_id != 0) continue;  // already materialized = leaf
        if (!m->creator_op) continue;   // raw tensor

        // Check if already visited
        int found = 0;
        for (u32 i = 0; i < nv; i++) if (visited[i] == tid) { found = 1; break; }
        if (found) continue;
        visited[nv++] = tid;

        // View ops: transparent — follow through without adding as a node
        if (is_view_op(m->creator_op)) {
            if (m->src_ids[0] && sp < GMAX) stk[sp++] = m->src_ids[0];
            continue;
        }

        nodes[(*n_nodes)++] = tid;

        // Follow deferred inputs
        if (m->src_ids[0] && sp < GMAX) stk[sp++] = m->src_ids[0];
        if (m->src_ids[1] && sp < GMAX) stk[sp++] = m->src_ids[1];
    }
    return *n_nodes;
}

// Partition nodes into fusion groups. Returns number of groups.
// group_of[i] = group ID for nodes[i]. group_root[g] = output tid of group g.
static u32 graph_partition(TinyHVM *ctx, const u32 *nodes, u32 n_nodes,
                           u32 *group_of, u32 *group_root, u8 *group_has_reduce) {
    u32 n_groups = 0;

    // Process in reverse order (deepest first = inputs before outputs).
    // Each node joins the group of its first deferred consumer, or starts a new group.
    // Reduce nodes always start a new group boundary.

    // First pass: assign groups bottom-up
    for (u32 i = 0; i < n_nodes; i++) group_of[i] = ~0u;

    // Build a node-id → index lookup (for finding consumers)
    // Walk from last to first (inputs before outputs in discovery order)
    for (int i = (int)n_nodes - 1; i >= 0; i--) {
        u32 tid = nodes[i];
        TensorMeta *m = &ctx->tensors[tid];

        // MM always its own group
        if (m->creator_op == UOP_MM) {
            group_of[i] = n_groups;
            group_root[n_groups] = tid;
            group_has_reduce[n_groups] = 0;
            n_groups++;
            continue;
        }

        int is_red = (m->creator_op == UOP_SUM || m->creator_op == UOP_RMAX);

        // Find which group our consumer belongs to
        u32 consumer_group = ~0u;
        for (u32 j = 0; j < n_nodes; j++) {
            if (j == (u32)i) continue;
            TensorMeta *cm = &ctx->tensors[nodes[j]];
            // Follow through view ops to find the actual consumer
            u32 check = nodes[j];
            while (is_view_op(ctx->tensors[check].creator_op))
                check = ctx->tensors[check].src_ids[0];
            if (ctx->tensors[check].src_ids[0] == tid ||
                ctx->tensors[check].src_ids[1] == tid) {
                if (group_of[j] != ~0u) { consumer_group = group_of[j]; break; }
            }
            (void)cm;
        }

        if (consumer_group != ~0u && !is_red && !group_has_reduce[consumer_group]) {
            // Can join consumer's group (no reduce conflict)
            group_of[i] = consumer_group;
        } else if (consumer_group != ~0u && is_red && !group_has_reduce[consumer_group]) {
            // Reduce can join consumer's group (as the group's reduce)
            group_of[i] = consumer_group;
            group_has_reduce[consumer_group] = m->creator_op;
        } else {
            // Start new group
            group_of[i] = n_groups;
            group_root[n_groups] = tid;
            group_has_reduce[n_groups] = is_red ? m->creator_op : 0;
            n_groups++;
        }
    }

    // Update group roots: the root should be the node closest to the output
    for (u32 g = 0; g < n_groups; g++) group_root[g] = 0;
    for (u32 i = 0; i < n_nodes; i++) {
        u32 g = group_of[i];
        if (g < n_groups) group_root[g] = nodes[i]; // last (closest to output) wins
    }

    return n_groups;
}

static void tensor_materialize_graph(TinyHVM *ctx, u32 root_tid) {
    // Delegates to chain-level. The graph discovery + partitioning infrastructure
    // is in place but not yet beneficial because the codegen only supports
    // one reduce per kernel. Multi-reduce codegen is the prerequisite for
    // graph-level dispatch to actually reduce the dispatch count.
    tensor_materialize_chain(ctx, root_tid);
}

// Entry point — tries graph-level, falls back to chain-level
static void tensor_materialize(TinyHVM *ctx, u32 tid) {
    TensorMeta *m = &ctx->tensors[tid];
    if (m->buf_id != 0) return;
    tensor_materialize_graph(ctx, tid);
}
