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
Term thvm_grad_multi(TinyHVM *ctx, Term loss, Term *params, Term *grad_slots, u32 n_params) {
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
