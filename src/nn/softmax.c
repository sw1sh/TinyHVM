// nn/softmax.c — Softmax: max → sub → exp → sum → div

static Term softmax(TinyHVM *ctx, Term logits, u32 B, u32 C) {
    // logits used twice (RMAX + SUB). Explicit DUP for linearity so
    // backward chain captures both grad contributions.
    Term l0, l1;
    thvm_dup(ctx, thvm_fresh_label(ctx), logits, &l0, &l1);
    // RMAX along class axis (dim 1). ERA as axes evaluates to ERA — pass
    // an explicit axes tensor.
    i32 ax1[] = {1};
    Term x_max = thvm_op(ctx, UOP_RMAX, l0, thvm_tensor_i32(ctx, ax1, SHAPE(1)));
    Term x_max_bc = thvm_expand(ctx, x_max, SHAPE(B, C));
    Term shifted = thvm_op(ctx, UOP_SUB, l1, x_max_bc);
    Term e = thvm_op(ctx, UOP_EXP, shifted, term_era());
    // e used in both SUM (denominator path) and DIV (numerator). Without
    // explicit DUP, backward chain loses one of the two gradient
    // contributions into e (bundle CTR ends up with fewer slots filled).
    Term e0, e1;
    thvm_dup(ctx, thvm_fresh_label(ctx), e, &e0, &e1);
    Term e_sum = thvm_sum_axes(ctx, e0, (u32[]){1}, 1);  // [B,C] → [B,1]
    Term e_sum_bc = thvm_expand(ctx, e_sum, SHAPE(B, C));
    return thvm_op(ctx, UOP_DIV, e1, e_sum_bc);
}
