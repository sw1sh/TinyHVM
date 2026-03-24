// nn/loss.c — Cross-entropy loss (pure UOp composition)

static Term cross_entropy_loss(TinyHVM *ctx, Term logits, u8 *labels, u32 B, u32 C) {
    Term probs = softmax(ctx, logits, B, C);

    f32 eps = 1e-7f;
    Term eps_t = thvm_expand(ctx, thvm_tensor(ctx, &eps, SHAPE(1, 1)), SHAPE(B, C));
    Term clamped = thvm_op(ctx, UOP_MAX, probs, eps_t);
    Term log_probs = thvm_op(ctx, UOP_LOG, clamped, term_era());

    f32 *oh = calloc(B * C, sizeof(f32));
    for (u32 i = 0; i < B; i++) oh[i * C + labels[i]] = 1.0f;
    Term one_hot = thvm_tensor(ctx, oh, SHAPE(B, C));
    free(oh);

    Term masked = thvm_op(ctx, UOP_MUL, one_hot, log_probs);
    Term sum_c = thvm_op(ctx, UOP_SUM, masked, term_era());
    Term sum_b = thvm_op(ctx, UOP_SUM, sum_c, term_era());
    Term neg = thvm_op(ctx, UOP_NEG, sum_b, term_era());

    f32 inv_B = 1.0f / (f32)B;
    Term scale = thvm_tensor(ctx, &inv_B, SHAPE(1, 1));
    return thvm_op(ctx, UOP_MUL, neg, scale);
}
