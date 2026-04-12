// nn/linear.c — Fully connected layer: x @ W + b (tinygrad Tensor.linear layout)

static Term linear(TinyHVM *ctx, Term x, Term w, Term b, u32 B, u32 in_f, u32 out_f) {
    (void)B;
    (void)in_f;
    (void)out_f;
    return thvm_tg_linear(ctx, x, w, &b);
}
