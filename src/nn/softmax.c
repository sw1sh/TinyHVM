// nn/softmax.c — Softmax: max → sub → exp → sum → div

static Term softmax(TinyHVM *ctx, Term logits, u32 B, u32 C) {
    Term x_max = thvm_op(ctx, UOP_RMAX, logits, term_era());
    Term x_max_bc = thvm_expand(ctx, x_max, SHAPE(B, C));
    Term shifted = thvm_op(ctx, UOP_SUB, logits, x_max_bc);
    Term e = thvm_op(ctx, UOP_EXP, shifted, term_era());
    Term e_sum = thvm_op(ctx, UOP_SUM, e, term_era());
    Term e_sum_bc = thvm_expand(ctx, e_sum, SHAPE(B, C));
    return thvm_op(ctx, UOP_DIV, e, e_sum_bc);
}
