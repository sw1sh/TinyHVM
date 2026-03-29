// grad/_.c — Autograd: thvm_grad, thvm_grad_multi
//
// Pure IC term constructors. No computation here — everything
// is driven by thvm_reduce through the UOP_GRAD interaction handler.

Term thvm_grad(TinyHVM *ctx, Term y, Term x) {
    // Seed = ones matching y's shape (not just scalar 1.0)
    // MM backward needs gy to be rank-2, etc.
    Shape seed_shape = SHAPE(1);
    if (term_tag(y) == TAG_TEN) {
        u32 y_id = (u32)term_val(y);
        seed_shape = ctx->tensors[y_id].view.shape;
    } else if (term_tag(y) == TAG_TOP) {
        const View *yv = st_get(term_val(y));
        if (yv) seed_shape = yv->shape;
    }
    u32 numel = 1;
    for (u32 i = 0; i < seed_shape.rank; i++) numel *= seed_shape.dims[i];
    f32 *ones = malloc(numel * sizeof(f32));
    for (u32 i = 0; i < numel; i++) ones[i] = 1.0f;
    Term seed = thvm_tensor(ctx, ones, seed_shape);
    free(ones);

    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     y);
    heap_set(ctx, loc + 1, seed);
    heap_set(ctx, loc + 2, x);
    return term_new(TAG_TOP, UOP_GRAD, loc);
}

// Multi-target GRAD: returns a lazy term that, when reduced, walks the
// provenance chain ONCE and deposits all param gradients via ASSIGN.
// Pure IC — no mutable state. Driven by ONE thvm_reduce.
//
// x = TAG_CTR encoding N (param, grad_slot) pairs:
//   heap[loc] = N (TAG_NUM)
//   heap[loc+1..2N] = param_0, slot_0, param_1, slot_1, ...
//
// At each leaf: if y matches any param_i, fire ASSIGN(slot_i, gy)
// and return ERA. BIN_GRAD's ADD absorbs ERA correctly.
// Pre-scan: walk loss provenance DAG, count backward refs for each tensor.
// A tensor with N consumers on the loss path gets N GRAD visits.
// When N > 1, the GRAD handler parks gradients and walks once with combined gy.
static void grad_prescan(TinyHVM *ctx, Term loss) {
    // When called from GRAD handler, loss is already TAG_TEN (reduced by trampoline).
    // When called eagerly (legacy), it may be TAG_TOP → reduce it first.
    if (term_tag(loss) != TAG_TEN) loss = thvm_reduce(ctx, loss);
    if (term_tag(loss) != TAG_TEN) return;
    u32 loss_id = (u32)term_val(loss);

    // BFS through provenance
    u32 *queue = malloc(ctx->tensor_count * sizeof(u32));
    u8  *visited = calloc(ctx->tensor_count, 1);
    u32 head = 0, tail = 0;
    queue[tail++] = loss_id;
    visited[loss_id] = 1;

    while (head < tail) {
        u32 tid = queue[head++];
        TensorMeta *tm = &ctx->tensors[tid];
        if (!tm->creator_op) continue;
        u32 src0 = tm->src_ids[0], src1 = tm->src_ids[1];
        // Each source gets one GRAD visit from this consumer
        if (src0) {
            ctx->tensors[src0].grad_refs++;
            if (!visited[src0]) { visited[src0] = 1; queue[tail++] = src0; }
        }
        if (src1 && src1 != src0) {
            // Only count binary ops with valid second source
            u32 cop = tm->creator_op;
            int is_bin = (cop==UOP_ADD||cop==UOP_SUB||cop==UOP_MUL||
                          cop==UOP_DIV||cop==UOP_MAX||cop==UOP_MM||cop==UOP_CMP);
            if (is_bin) {
                ctx->tensors[src1].grad_refs++;
                if (!visited[src1]) { visited[src1] = 1; queue[tail++] = src1; }
            }
        }
    }
    free(queue);
    free(visited);
}

Term thvm_grad_multi(TinyHVM *ctx, Term loss, Term *params, Term *grad_slots, u32 n_params) {
    // grad_prescan is deferred to the GRAD handler at reduction time.
    // This ensures the forward pass (required to establish provenance)
    // runs inside the JIT capture window.

    // Encode targets as CTR
    u64 tgt_loc = heap_alloc(ctx, 1 + 2 * n_params);
    heap_set(ctx, tgt_loc, term_new(TAG_NUM, NUM_U32, n_params));
    for (u32 i = 0; i < n_params; i++) {
        heap_set(ctx, tgt_loc + 1 + 2*i, params[i]);
        heap_set(ctx, tgt_loc + 1 + 2*i + 1, grad_slots[i]);
    }
    Term x_multi = term_new(TAG_CTR, 0, tgt_loc);
    // Seed = ones matching loss shape (MM backward needs rank-2 gy, etc.)
    Shape seed_shape = SHAPE(1);
    if (term_tag(loss) == TAG_TEN) {
        u32 loss_id = (u32)term_val(loss);
        seed_shape = ctx->tensors[loss_id].view.shape;
    } else if (term_tag(loss) == TAG_TOP) {
        const View *lv = st_get(term_val(loss));
        if (lv) seed_shape = lv->shape;
    }
    u32 numel = 1;
    for (u32 i = 0; i < seed_shape.rank; i++) numel *= seed_shape.dims[i];
    f32 *ones = malloc(numel * sizeof(f32));
    for (u32 i = 0; i < numel; i++) ones[i] = 1.0f;
    Term seed = thvm_tensor(ctx, ones, seed_shape);
    free(ones);

    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc, loss);
    heap_set(ctx, loc + 1, seed);
    heap_set(ctx, loc + 2, x_multi);
    return term_new(TAG_TOP, UOP_GRAD, loc);
}
