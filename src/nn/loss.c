// nn/loss.c — Cross-entropy loss (pure UOp composition)

// Logsumexp form: CE(logits, label) = logsumexp(logits) - logits[label]
// This avoids the DIV(e, expand(sum(e))) pattern in softmax (which has a
// self-referential backward dependency: e appears in both numerator and
// denominator via SUM). The logsumexp form has NO DIV — only RMAX, SUB,
// EXP, SUM, LOG, ADD, MUL — so backward composes cleanly.
static Term cross_entropy_loss(TinyHVM *ctx, Term logits, u8 *labels, u32 B, u32 C) {
    // logits reused in RMAX, SUB, and MUL(onehot, logits). DUP for linearity.
    Term l0, l1, l2;
    {
        Term tmp;
        thvm_dup(ctx, thvm_fresh_label(ctx), logits, &l0, &tmp);
        thvm_dup(ctx, thvm_fresh_label(ctx), tmp, &l1, &l2);
    }

    // max_per_row: [B,1]
    i32 ax1[] = {1};
    Term x_max = thvm_op(ctx, UOP_RMAX, l0, thvm_tensor_i32(ctx, ax1, SHAPE(1)));
    // DUP x_max: used once in SUB (shifted) and once in ADD (lse)
    Term xm0, xm1;
    thvm_dup(ctx, thvm_fresh_label(ctx), x_max, &xm0, &xm1);
    Term xm0_bc = thvm_expand(ctx, xm0, SHAPE(B, C));
    Term shifted = thvm_op(ctx, UOP_SUB, l1, xm0_bc);
    Term e = thvm_op(ctx, UOP_EXP, shifted, term_era());
    Term e_sum = thvm_sum_axes(ctx, e, (u32[]){1}, 1);  // [B,1]
    Term log_sum_exp = thvm_op(ctx, UOP_LOG, e_sum, term_era());
    Term lse = thvm_op(ctx, UOP_ADD, xm1, log_sum_exp);  // [B,1]

    // correct_logit = sum_axis1(onehot * logits) — [B,1]
    f32 *oh = calloc(B * C, sizeof(f32));
    for (u32 i = 0; i < B; i++) oh[i * C + labels[i]] = 1.0f;
    Term one_hot = thvm_tensor(ctx, oh, SHAPE(B, C));
    free(oh);
    Term masked = thvm_op(ctx, UOP_MUL, one_hot, l2);
    Term correct = thvm_sum_axes(ctx, masked, (u32[]){1}, 1);  // [B,1]

    // per-row loss = lse - correct; total = sum / B.
    Term per_row = thvm_op(ctx, UOP_SUB, lse, correct);
    Term sum_all = thvm_sum_axes(ctx, per_row, (u32[]){0, 1}, 2);  // [1,1]
    f32 inv_B = 1.0f / (f32)B;
    Term scale = thvm_tensor(ctx, &inv_B, SHAPE(1, 1));
    return thvm_op(ctx, UOP_MUL, sum_all, scale);
}
