// nn/sequential.c — Sequential model composition

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
                Term xr = thvm_reduce(ctx, x);
                TensorMeta *mx = &ctx->tensors[(u32)term_val(xr)];
                u32 C = mx->view.shape.dims[1];
                u32 H = mx->view.shape.dims[2];
                u32 W = mx->view.shape.dims[3];
                x = batchnorm_term(ctx, xr,
                    l->bn.gamma, l->bn.beta, l->bn.rmean, l->bn.rvar,
                    BS, C, H, W, training);
                break;
            }
            case LAYER_MAXPOOL: {
                u32 kernel[] = {l->pool.ks, l->pool.ks};
                u32 stride[] = {l->pool.ks, l->pool.ks};
                x = thvm_maxpool2d(ctx, x, kernel, stride);
                break;
            }
            case LAYER_FLATTEN: {
                Term xr = thvm_reduce(ctx, x);
                TensorMeta *mx = &ctx->tensors[(u32)term_val(xr)];
                u32 numel = mx->view.numel / BS;
                x = thvm_reshape(ctx, xr, SHAPE(BS, numel));
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
