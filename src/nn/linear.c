// nn/linear.c — Fully connected layer: x @ W + b

static Term linear(TinyHVM *ctx, Term x, Term w, Term b, u32 B, u32 in_f, u32 out_f) {
    Term y = thvm_op(ctx, UOP_MM, x, w);
    Term b_reshaped = thvm_reshape(ctx, b, SHAPE(1, out_f));
    Term b_bc = thvm_expand(ctx, b_reshaped, SHAPE(B, out_f));
    return thvm_op(ctx, UOP_ADD, y, b_bc);
    (void)in_f;
}
