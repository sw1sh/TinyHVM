// nn/sequential.c — Sequential model composition
// No thvm_reduce calls — layers compose lazily where possible.
// BN is inherently eager (training stats + stored tensor IDs for backward).

Term thvm_sequential(TinyHVM *ctx, Term x, Layer *layers, u32 n,
                     u32 BS, int training) {
    for (u32 i = 0; i < n; i++) {
        Layer *l = &layers[i];
        switch (l->type) {
            case LAYER_CONV2D: {
                u32 padding[] = {0, 0, 0, 0};
                u32 stride[] = {1, 1};
                x = thvm_conv2d(ctx, x, l->conv.w, l->conv.b,
                                1, stride, padding);
                break;
            }
            case LAYER_BN: {
                // BN needs shapes from layer struct, not from tensor reduction.
                // BN forward is eager internally (running stats + backward tensor IDs).
                x = batchnorm_term(ctx, x,
                    l->bn.gamma, l->bn.beta, l->bn.rmean, l->bn.rvar,
                    BS, l->bn.c, l->bn.h, l->bn.w, training);
                break;
            }
            case LAYER_MAXPOOL: {
                u32 kernel[] = {l->pool.ks, l->pool.ks};
                u32 stride[] = {l->pool.ks, l->pool.ks};
                x = thvm_maxpool2d(ctx, x, kernel, stride);
                break;
            }
            case LAYER_FLATTEN: {
                // Use layer-provided feature count — no thvm_reduce needed.
                x = thvm_reshape(ctx, x, SHAPE(BS, l->flat.flat_features));
                break;
            }
            case LAYER_LINEAR: {
                x = linear(ctx, x, l->lin.w, l->lin.b,
                           BS, l->lin.in_f, l->lin.out_f);
                break;
            }
            case LAYER_FN:
                x = l->fn(ctx, x);
                break;
        }
    }
    return x;
}
