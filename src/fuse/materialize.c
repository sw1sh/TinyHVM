// fuse/materialize.c — Materialize lazy tensor chains at boundaries
//
// Walk provenance chain, collect ops + leaf buffers, compile fused kernel, dispatch.
// Shared intermediates (defer_consumers > 0) get side-output buffers written by
// the same fused kernel — no separate dispatch needed.

// Pre-computed reference counts: how many tensors reference each tid as src.
// Populated once by ref_prescan, then used by materialize_walk for O(1) checks.
static u16 tensor_ref_count[MAX_TENSORS];
static u32 ref_prescan_done = 0;

static void ref_prescan(TinyHVM *ctx) {
    if (ref_prescan_done >= ctx->tensor_count) return;
    memset(tensor_ref_count, 0, ctx->tensor_count * sizeof(u16));
    for (u32 t = 0; t < ctx->tensor_count; t++) {
        u32 s0 = ctx->tensors[t].src_ids[0];
        u32 s1 = ctx->tensors[t].src_ids[1];
        if (s0 && s0 < ctx->tensor_count) tensor_ref_count[s0]++;
        if (s1 && s1 < ctx->tensor_count) tensor_ref_count[s1]++;
    }
    ref_prescan_done = ctx->tensor_count;
}

// When set, materialize_walk won't walk through reshape (used in reduce contexts
// where reshape changes the coordinate space that reduce axes reference).
static u8 walk_no_reshape_through = 0;
// When set, materialize_walk won't trigger tensor_materialize as side effect
// (used by scheduler during planning — pure read-only walk).
static u8 walk_no_materialize = 0;

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
        // Walk THROUGH reshape: use base's coordinate space, not reshape's.
        // The output buffer is flat — reshape just reinterprets it.
        // Only for contiguous reshapes where all leaves have matching numel.
        // Walk through reshape into base ew chain.
        // Uses pre-computed ref counts to check if consumed ops are safe.
        if (uop == UOP_RESHAPE && !walk_no_reshape_through &&
            base && ctx->tensors[base].buf_id == 0 && ctx->tensors[base].creator_op &&
            is_elementwise(ctx->tensors[base].creator_op) &&
            !ctx->tensors[base].requires_grad &&
            tensor_ref_count[base] <= 1) {
            // Pre-check: scan base chain for any op with ref_count > 1
            int safe = 1;
            {
                u32 stk[32]; u32 sn = 0;
                stk[sn++] = base;
                while (sn > 0 && safe) {
                    u32 s = stk[--sn];
                    TensorMeta *sm = &ctx->tensors[s];
                    if (sm->buf_id != 0) continue; // leaf — OK
                    if (tensor_ref_count[s] > 1) { safe = 0; break; }
                    if (!sm->creator_op) continue;
                    if (is_elementwise(sm->creator_op)) {
                        if (sm->src_ids[0] && sn < 30) stk[sn++] = sm->src_ids[0];
                        if (sm->src_ids[1] && is_binary(sm->creator_op) && sn < 30)
                            stk[sn++] = sm->src_ids[1];
                    }
                    // Stop at non-ew (will be materialized as leaf)
                }
            }
            if (safe) {
                int sub = materialize_walk(ctx, base, ops, n_ops, op_tids, leaf_ids, leaf_views, n_leaves);
                if (sub >= 0) return sub;
            }
        }
        // Fallback: materialize base, share buffer (skip during pure planning walk)
        if (base && ctx->tensors[base].buf_id == 0 && ctx->tensors[base].creator_op &&
            !walk_no_materialize)
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

    if (*n_ops >= FUSE_MAX_OPS) {
        if (getenv("THVM_DIAG")) fprintf(stderr, "FUSE_LIMIT: ops=%u leaves=%u tid=%u\n", *n_ops, *n_leaves, tid);
        return -1;
    }
    u32 op_idx = (*n_ops)++;
    ops[op_idx] = (FusedOp){ .uop = uop, .arg_a = (u32)arg_a, .arg_b = is_binary(uop) ? (u32)arg_b : 0 };
    op_tids[op_idx] = tid;

    // NOTE: side output buffer allocation moved to tensor_materialize
    // (after walk succeeds) to avoid zombie buffers from failed walks.

    return (int)(FUSE_MAX_LEAVES + op_idx);
}

// Forward declaration for graph-level materialize
static void tensor_materialize_graph(TinyHVM *ctx, u32 tid);

// Pre-scan: walk all deferred tensors to count how many future dispatches
// will read each materialized buffer. This enables safe mid-step buffer
// stealing: a buffer is stealable when buf_remaining_uses reaches 0.
// Runs once per step (before first dispatch).
static u32 prescan_from = 0; // start of next scan range
static void memory_prescan(TinyHVM *ctx) {
    // Walk NEW ephemeral tensors (since last prescan). For each deferred
    // tensor (buf_id==0), find which materialized buffers it references.
    u32 start = prescan_from;
    prescan_from = ctx->tensor_count;
    for (u32 tid = start; tid < ctx->tensor_count; tid++) {
        TensorMeta *m = &ctx->tensors[tid];
        if (m->buf_id != 0) continue;     // already materialized
        if (!m->creator_op) continue;      // no provenance
        if (!is_elementwise(m->creator_op) &&
            m->creator_op != UOP_SUM && m->creator_op != UOP_RMAX) continue;
        // This deferred tensor will be materialized → its leaf bufs will be read.
        // Walk src_ids to find leaf buf_ids (follow through ew + view ops).
        u32 stk[32]; u32 sn = 0;
        if (m->src_ids[0]) stk[sn++] = m->src_ids[0];
        if (m->src_ids[1] && is_binary(m->creator_op)) stk[sn++] = m->src_ids[1];
        while (sn > 0 && sn < 30) {
            u32 s = stk[--sn];
            TensorMeta *sm = &ctx->tensors[s];
            if (sm->buf_id != 0) {
                // Leaf: materialized buffer → mark as future use
                if (sm->backend && sm->backend->buf_mark_use)
                    sm->backend->buf_mark_use(sm->buf_id);
                continue;
            }
            if (!sm->creator_op) continue;
            if (sm->src_ids[0]) stk[sn++] = sm->src_ids[0];
            if (sm->src_ids[1] && is_binary(sm->creator_op) && sn < 30)
                stk[sn++] = sm->src_ids[1];
        }
    }
}

// Chain-level materialize (original implementation)
static void tensor_materialize_chain(TinyHVM *ctx, u32 tid) {
    TensorMeta *m = &ctx->tensors[tid];
    if (m->buf_id != 0) return;
    ref_prescan(ctx); // ensure reference counts are up to date
    if (getenv("THVM_TRACE")) fprintf(stderr, "MAT_CHAIN tid=%u op=%u buf=%u\n", tid, m->creator_op, m->buf_id);

    // Deferred MM: materialize inputs, then dispatch via MPS.
    if (m->creator_op == UOP_MM && m->src_ids[0] && m->src_ids[1]) {
        ENSURE(ctx, m->src_ids[0]);
        ENSURE(ctx, m->src_ids[1]);
        TensorMeta *ma = &ctx->tensors[m->src_ids[0]];
        TensorMeta *mb = &ctx->tensors[m->src_ids[1]];
        if (ma->buf_id && mb->buf_id) {
            // Use ASSIGN elision target if available
            if (m->assign_target) m->buf_id = m->assign_target;
            else m->buf_id = m->backend->buf_alloc(m->view.numel * sizeof(f32));
            u32 M = ma->view.shape.dims[0], K = ma->view.shape.dims[1], N = mb->view.shape.dims[1];
            m->backend->op_mm(m->buf_id, ma->buf_id, &ma->view,
                              mb->buf_id, &mb->view, M, K, N);
            return;
        }
    }

    // Post-reduce fusion: if this deferred ew chain contains a deferred reduce
    // (reachable via views), fuse pre-reduce ew → reduce → post-reduce ew in one kernel.
    if (is_elementwise(m->creator_op) && m->backend->dispatch_kernel_rs) {
        // Scan chain for a deferred reduce (follow src[0] through ew + views)
        u32 reduce_tid = 0, reduce_input = 0;
        u32 cur = tid;
        for (u32 depth = 0; depth < 20; depth++) {
            TensorMeta *cm = &ctx->tensors[cur];
            if (getenv("THVM_TRACE")) fprintf(stderr, "  SCAN d=%u tid=%u op=%u buf=%u\n", depth, cur, cm->creator_op, cm->buf_id);
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
            if(getenv("THVM_TRACE")) fprintf(stderr, "POST_REDUCE_FUSION: tid=%u reduce=%u input=%u\n", tid, reduce_tid, reduce_input);
            FusedOp ops[FUSE_MAX_OPS]; u32 n_ops = 0, op_tids[FUSE_MAX_OPS];
            u32 leaf_ids[FUSE_MAX_LEAVES]; const View *leaf_views[FUSE_MAX_LEAVES]; u32 n_leaves = 0;
            walk_no_reshape_through = 1;
            int pre_res = materialize_walk(ctx, reduce_input, ops, &n_ops, op_tids,
                                            leaf_ids, leaf_views, &n_leaves);
            walk_no_reshape_through = 0;
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
                    m->backend->dispatch_kernel_rs(m->buf_id, bufs, leaf_views, NULL, n_leaves,
                        ops, n_ops, &fs, &rs, n_sides?side_bufs:NULL, n_sides?side_ops:NULL, n_sides);
                    return;
                }
            }
        }
    }

    // Deferred reduce (SUM/RMAX): fuse reduce + elementwise input chain
    u32 reduce_type = 0, reduce_axes_id = 0, walk_tid = tid;
    if ((m->creator_op == UOP_SUM || m->creator_op == UOP_RMAX) && m->src_ids[0]) {
        TensorMeta *ri = &ctx->tensors[m->src_ids[0]];
        if (ri->buf_id == 0 && is_elementwise(ri->creator_op)) {
            reduce_type = m->creator_op;
            reduce_axes_id = m->src_ids[1];
            walk_tid = m->src_ids[0];
        }
    }

    FusedOp ops[FUSE_MAX_OPS]; u32 n_ops = 0; u32 op_tids[FUSE_MAX_OPS];
    u32 leaf_ids[FUSE_MAX_LEAVES]; const View *leaf_views[FUSE_MAX_LEAVES]; u32 n_leaves = 0;

    if (reduce_type) walk_no_reshape_through = 1;
    int result = materialize_walk(ctx, walk_tid, ops, &n_ops, op_tids, leaf_ids, leaf_views, &n_leaves);
    walk_no_reshape_through = 0;
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
                    walk_no_reshape_through = 1;
                    int pre_result = materialize_walk(ctx, reduce_input, ops, &n_ops, op_tids,
                                                       leaf_ids, leaf_views, &n_leaves);
                    walk_no_reshape_through = 0;
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
                            m->backend->dispatch_kernel_rs(m->buf_id, bufs, leaf_views, NULL, n_leaves,
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
                m->backend->dispatch_kernel_rs(m->buf_id, bufs, views, NULL, 1, NULL, 0, &fs, &rs, NULL, NULL, 0);
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

    // ASSIGN elision: use pre-assigned target buffer if available
    if (m->assign_target) {
        m->buf_id = m->assign_target;
    } else {
        m->buf_id = m->backend->buf_alloc(m->view.numel * sizeof(f32));
    }

    // Allocate side output buffers for shared/grad-needed intermediates.
    // Exclude tid (the tensor being materialized — it gets the main output buffer).
    // For reduce fusion: walk_tid != tid, so walk_tid CAN get a side output.
    // But walk_tid's defer_consumers includes the reduce itself (+1), so only
    // allocate when defer_consumers > 1 (meaning someone ELSE needs the value).
    for (u32 i = 0; i < n_ops; i++) {
        TensorMeta *sm = &ctx->tensors[op_tids[i]];
        u32 min_dc = (reduce_type && op_tids[i] == walk_tid) ? 1 : 0;
        if (sm->defer_consumers > min_dc && sm->buf_id == 0 && op_tids[i] != tid)
            sm->buf_id = sm->backend->buf_alloc(sm->view.numel * sizeof(f32));
    }

    // Propagate side-output buf_ids to views that reference them
    // (view-through walk marked bases with defer_consumers++, views need their buf_id)
    for (u32 i = 0; i < n_ops; i++) {
        u32 oid = op_tids[i];
        TensorMeta *sm = &ctx->tensors[oid];
        if (sm->buf_id != 0) {
            // Check if any view references this tensor as base
            for (u32 vt = oid + 1; vt < ctx->tensor_count && vt < oid + 200; vt++) {
                TensorMeta *vm = &ctx->tensors[vt];
                if (vm->buf_id == 0 && is_view_op(vm->creator_op) && vm->src_ids[0] == oid) {
                    vm->buf_id = sm->buf_id;
                    if (vm->backend && vm->backend->buf_incref) vm->backend->buf_incref(sm->buf_id);
                }
            }
        }
    }

    // Collect side outputs (shared intermediates + grad-needed with own buffers)
    u32 side_bufs[8]; u32 side_ops[8]; u32 n_sides = 0;
    for (u32 i = 0; i < n_ops && n_sides < 8; i++) {
        TensorMeta *sm = &ctx->tensors[op_tids[i]];
        u32 min_dc2 = (reduce_type && op_tids[i] == walk_tid) ? 1 : 0;
        if (sm->buf_id != 0 && sm->defer_consumers > min_dc2 && op_tids[i] != tid) {
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
            m->backend->dispatch_kernel_rs(m->buf_id, bufs, leaf_views, NULL, n_leaves,
                                            ops, n_ops, &full_shape, &rs,
                                            n_sides ? side_bufs : NULL,
                                            n_sides ? side_ops : NULL, n_sides);
        } else {
            m->backend->dispatch_kernel_rs(m->buf_id, bufs, leaf_views, NULL, n_leaves,
                                            ops, n_ops, &m->view.shape, NULL,
                                            side_bufs, side_ops, n_sides);
        }
        // No post-dispatch decref here — view-op increfs are permanent.
        // Buffer steal uses buf_last_use timeline instead of refcount.
        // Memory checkpoint: mark leaf bufs and flush if memory pressure high.
        if (m->backend->mem_checkpoint) {
            u32 lbufs[FUSE_MAX_LEAVES];
            for (u32 i = 0; i < n_leaves; i++) lbufs[i] = ctx->tensors[leaf_ids[i]].buf_id;
            m->backend->mem_checkpoint(lbufs, n_leaves);
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

    walk_no_reshape_through = 1;
    int result = materialize_walk(ctx, input_tid, ops, &n_ops, op_tids,
                                   leaf_ids, leaf_views, &n_leaves);
    walk_no_reshape_through = 0;
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
        im->backend->dispatch_kernel_rs(out_buf, bufs, leaf_views, NULL, n_leaves,
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

// Detect and dispatch multi-reduce pattern:
//   ew_chain1 → REDUCE1 → ew_chain2 → REDUCE2 → [ew_chain3]
// Dispatches as one kernel with two reduce phases.
// Returns 1 if dispatched, 0 if pattern not found.
static int try_multi_reduce(TinyHVM *ctx, u32 tid) {
    TensorMeta *m = &ctx->tensors[tid];
    if (m->buf_id != 0) return 0;
    if (!m->backend || !m->backend->dispatch_kernel_rs) return 0;

    // Pattern: tid is the output. Walk backward looking for:
    //   tid (ew/reduce) → ... → reduce2 → ew_chain → reduce1 → ew_chain → leaves
    // OR: tid IS reduce2, and its input chain contains reduce1

    // Scan from tid downward for two deferred reduces
    u32 reduce2_tid = 0, reduce1_tid = 0;
    u32 cur = tid;

    // Find reduce2 (outermost reduce)
    for (u32 d = 0; d < 20; d++) {
        TensorMeta *cm = &ctx->tensors[cur];
        if (cm->buf_id != 0) break;
        if (!cm->creator_op) break;
        if (cm->creator_op == UOP_SUM || cm->creator_op == UOP_RMAX) {
            reduce2_tid = cur; break;
        }
        if (is_elementwise(cm->creator_op) || is_view_op(cm->creator_op))
            cur = cm->src_ids[0];
        else break;
    }
    if (!reduce2_tid) return 0;

    // Find reduce1 (below reduce2) — scan both branches of binary ops
    TensorMeta *r2m = &ctx->tensors[reduce2_tid];
    {
        u32 stk[32]; u32 sn = 0;
        if (r2m->src_ids[0]) stk[sn++] = r2m->src_ids[0];
        while (sn > 0 && !reduce1_tid) {
            cur = stk[--sn];
            if (!cur) continue;
            TensorMeta *cm = &ctx->tensors[cur];
            if (cm->buf_id != 0) continue;
            if (!cm->creator_op) continue;
            if (cm->creator_op == UOP_SUM || cm->creator_op == UOP_RMAX) {
                reduce1_tid = cur; break;
            }
            if (is_elementwise(cm->creator_op) || is_view_op(cm->creator_op)) {
                if (cm->src_ids[0] && sn < 30) stk[sn++] = cm->src_ids[0];
                if (cm->src_ids[1] && sn < 30) stk[sn++] = cm->src_ids[1];
            }
        }
    }
    if (!reduce1_tid) return 0;

    // Found two reduces! Verify axes match BEFORE any side effects.
    TensorMeta *r1m = &ctx->tensors[reduce1_tid];
    u32 r1_input = r1m->src_ids[0];
    if (!r1_input) return 0;

    // Check axes compatibility via host_ptr (no ENSURE needed for small axes tensors)
    {
        u32 r1a = r1m->src_ids[1], r2a = r2m->src_ids[1];
        if (r1a && r2a) {
            const f32 *r1f = tensor_host_f32(ctx, r1a);
            const f32 *r2f = tensor_host_f32(ctx, r2a);
            if (!r1f || !r2f) return 0; // can't read axes without ENSURE
            u32 n1 = ctx->tensors[r1a].view.numel, n2 = ctx->tensors[r2a].view.numel;
            if (n1 != n2) return 0;
            for (u32 i = 0; i < n1; i++)
                if ((u32)r1f[i] != (u32)r2f[i]) return 0;
        } else if (r1a != r2a) return 0; // one has axes, other doesn't
    }

    // Axes match! Now safe to materialize.
    ENSURE(ctx, r1_input);
    if (ctx->tensors[r1_input].buf_id == 0) return 0;

    // Walk pre-reduce1 ew chain
    FusedOp ops[FUSE_MAX_OPS]; u32 n_ops = 0, op_tids[FUSE_MAX_OPS];
    u32 leaf_ids[FUSE_MAX_LEAVES]; const View *leaf_views[FUSE_MAX_LEAVES]; u32 n_leaves = 0;

    // Reduce1 input is materialized — use as sole leaf
    leaf_ids[0] = r1_input;
    leaf_views[0] = &ctx->tensors[r1_input].view;
    n_leaves = 1;

    // Build ReduceSpec for reduce1
    ReduceSpec rs = {0};
    rs.reduce_type = r1m->creator_op;
    u32 r1_axes = r1m->src_ids[1];
    Shape fs = ctx->tensors[r1_input].view.shape;
    if (r1_axes) {
        ENSURE(ctx, r1_axes);
        TensorMeta *axt = &ctx->tensors[r1_axes];
        f32 af[MAX_DIM]; META_READ(axt->backend, axt->buf_id, af, axt->view.numel*4);
        for (u32 i = 0; i < axt->view.numel; i++) {
            u32 ax = (u32)af[i]; if (ax < fs.rank) rs.is_reduce[ax] = 1;
        }
    } else {
        for (int d = (int)fs.rank-1; d >= 0; d--)
            if (fs.dims[d] > 1) { rs.is_reduce[d] = 1; break; }
    }

    // Collect ops between reduce1 and reduce2 (the "pre-reduce2" chain).
    // These ops reference BOTH acc (reduce1 result) AND reduce-varying leaves,
    // so they go INSIDE the reduce2 loop, not between the loops.
    u32 between_chain[16]; u32 bn = 0;
    cur = r2m->src_ids[0];
    while (cur != reduce1_tid && bn < 16) {
        TensorMeta *cm = &ctx->tensors[cur];
        if (is_elementwise(cm->creator_op)) {
            between_chain[bn++] = cur;
            cur = cm->src_ids[0];
        } else if (is_view_op(cm->creator_op)) {
            cur = cm->src_ids[0];
        } else break;
    }

    // No between-loop ops (reduce1 feeds directly into reduce2's inner ops).
    // post_reduce_start = 0 means reduce1 has no pre-reduce ew ops.
    rs.post_reduce_start = 0;
    rs.n_post_leaves = 0;

    // Build reduce2 phase ops from between_chain (goes inside reduce2 loop)
    rs.reduce2_type = r2m->creator_op;
    rs.reduce2_start = 0; // all ops are inside reduce2 loop

    // Add between-chain ops as reduce2 ops (reverse order: bottom→top)
    for (int bi = (int)bn - 1; bi >= 0; bi--) {
        TensorMeta *bm = &ctx->tensors[between_chain[bi]];
        if (n_ops >= FUSE_MAX_OPS) return 0;
        u32 a_ref, b_ref = 0;
        // arg_a: follow src_ids[0] — either a leaf or a previous op
        u32 a_src = bm->src_ids[0];
        if (a_src == reduce1_tid) {
            // References reduce1 result — codegen maps this to "acc"
            a_ref = n_leaves; // special: codegen treats >= n_leaves as "acc" ref
        } else {
            // Check if it's a leaf
            int found_leaf = -1;
            for (u32 li = 0; li < n_leaves; li++)
                if (leaf_ids[li] == a_src) { found_leaf = (int)li; break; }
            if (found_leaf >= 0) a_ref = (u32)found_leaf;
            else a_ref = n_leaves + n_ops - 1; // previous op
        }
        // arg_b: second input (for binary ops)
        if (is_binary(bm->creator_op)) {
            u32 b_src = bm->src_ids[1];
            if (b_src == between_chain[bi]) { b_ref = a_ref; } // self-ref (x*x)
            else {
                // Check if b_src is a leaf
                int found_leaf = -1;
                for (u32 li = 0; li < n_leaves; li++)
                    if (leaf_ids[li] == b_src) { found_leaf = (int)li; break; }
                if (found_leaf >= 0) b_ref = (u32)found_leaf;
                else if (b_src == reduce1_tid) b_ref = n_leaves;
                else {
                    // b_src is another between-chain op — find it
                    ENSURE(ctx, b_src);
                    if (ctx->tensors[b_src].buf_id != 0 && n_leaves < FUSE_MAX_LEAVES) {
                        leaf_ids[n_leaves] = b_src;
                        leaf_views[n_leaves] = &ctx->tensors[b_src].view;
                        b_ref = n_leaves++;
                    } else b_ref = a_ref; // fallback
                }
            }
        }
        ops[n_ops] = (FusedOp){ .uop = bm->creator_op, .arg_a = a_ref, .arg_b = b_ref };
        op_tids[n_ops] = between_chain[bi];
        n_ops++;
    }
    // Reduce2 axes
    u32 r2_axes = r2m->src_ids[1];
    if (r2_axes) {
        ENSURE(ctx, r2_axes);
        TensorMeta *axt = &ctx->tensors[r2_axes];
        f32 af[MAX_DIM]; META_READ(axt->backend, axt->buf_id, af, axt->view.numel*4);
        for (u32 i = 0; i < axt->view.numel; i++) {
            u32 ax = (u32)af[i]; if (ax < fs.rank) rs.is_reduce2[ax] = 1;
        }
    } else {
        memcpy(rs.is_reduce2, rs.is_reduce, sizeof(rs.is_reduce)); // same axes
    }

    // Verify reduce2 axes match reduce1 (required for shared loop structure)
    if (memcmp(rs.is_reduce, rs.is_reduce2, sizeof(rs.is_reduce)) != 0)
        return 0; // different axes — can't fuse into one loop

    // Allocate output and dispatch
    m->buf_id = m->backend->buf_alloc(m->view.numel * sizeof(f32));
    u32 bufs[FUSE_MAX_LEAVES];
    for (u32 i = 0; i < n_leaves; i++) bufs[i] = ctx->tensors[leaf_ids[i]].buf_id;
    m->backend->dispatch_kernel_rs(m->buf_id, bufs, leaf_views, NULL, n_leaves,
        ops, n_ops, &fs, &rs, NULL, NULL, 0);

    // Mark intermediate deferred tensors as "materialized" by allocating
    // minimal buffers. They won't actually be read (the fused kernel wrote
    // the final output directly), but this prevents re-materialization.
    for (u32 i = 0; i < bn; i++) {
        TensorMeta *bm = &ctx->tensors[between_chain[i]];
        if (bm->buf_id == 0 && bm->backend)
            bm->buf_id = bm->backend->buf_alloc(4); // minimal alloc
    }
    if (reduce1_tid != tid && ctx->tensors[reduce1_tid].buf_id == 0)
        ctx->tensors[reduce1_tid].buf_id = ctx->tensors[reduce1_tid].backend->buf_alloc(
            ctx->tensors[reduce1_tid].view.numel * sizeof(f32));
    if (reduce2_tid != tid && ctx->tensors[reduce2_tid].buf_id == 0)
        ctx->tensors[reduce2_tid].buf_id = ctx->tensors[reduce2_tid].backend->buf_alloc(
            ctx->tensors[reduce2_tid].view.numel * sizeof(f32));

    return 1;
}

static void tensor_materialize_graph(TinyHVM *ctx, u32 root_tid) {
    TensorMeta *rm = &ctx->tensors[root_tid];
    if (rm->buf_id != 0) return;

    // Try multi-reduce pattern first
    if (rm->backend && rm->backend->dispatch_kernel_rs) {
        if (try_multi_reduce(ctx, root_tid)) return;
    }

    // Fall back to chain-level
    tensor_materialize_chain(ctx, root_tid);
}

// Entry point — tries graph-level, falls back to chain-level
static u32 prescan_tc = 0; // tensor_count at last prescan

static void tensor_materialize(TinyHVM *ctx, u32 tid) {
    TensorMeta *m = &ctx->tensors[tid];
    if (m->buf_id != 0) return;
    // Run memory prescan periodically to find newly deferred tensors.
    // Reset scan state when tensor_count drops (pool_reset happened).
    if (ctx->tensor_count < prescan_tc) { prescan_from = 0; prescan_tc = 0; ref_prescan_done = 0; }
    if (ctx->tensor_count > prescan_tc + 20) {
        memory_prescan(ctx);
        prescan_tc = ctx->tensor_count;
    }
    tensor_materialize_graph(ctx, tid);
}
