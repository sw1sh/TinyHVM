// Minimal conv2d grad test: BS=1, Cin=1, Cout=1, 4x4 input, 3x3 kernel
// Expected: d_w[i,j] = sum over (oh,ow) of gy[oh,ow] * x[oh+i, ow+j]

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>

int main(void) {
    TinyHVM *ctx = thvm_init("metal");

    // 4x4 input, sequential values
    f32 x_data[16]; for (int i=0;i<16;i++) x_data[i] = (f32)(i+1);
    Term x = thvm_tensor(ctx, x_data, (Shape){.dims={1,1,4,4},.rank=4});
    thvm_set_requires_grad(ctx, x);

    // 3x3 kernel, all ones
    f32 w_data[9]; for (int i=0;i<9;i++) w_data[i] = 1.0f;
    Term w = thvm_tensor(ctx, w_data, (Shape){.dims={1,1,3,3},.rank=4});
    thvm_set_requires_grad(ctx, w);

    // No bias
    u32 padding[] = {0,0,0,0}, stride[] = {1,1};
    Term out = thvm_conv2d(ctx, x, w, term_era(), 1, stride, padding);
    // out shape: [1,1,2,2] (valid padding)

    // Loss = sum of output (so gy = ones)
    Term loss = thvm_op(ctx, UOP_SUM, thvm_reshape(ctx, out, SHAPE(4)), term_era());
    thvm_reduce(ctx, loss);

    printf("loss = %.1f\n", thvm_to_host(ctx, loss)[0]);
    // Expected: conv with all-ones kernel = sum of 3x3 patches
    // patch(0,0) = sum(1..9)=45, patch(0,1)=sum(2..10)=54, patch(1,0)=sum(5..13)=81, patch(1,1)=sum(6..14)=90
    // total = 45+54+81+90 = 270

    f32 *out_data = thvm_to_host(ctx, thvm_reduce(ctx, out));
    printf("out = [%.0f, %.0f, %.0f, %.0f]\n", out_data[0], out_data[1], out_data[2], out_data[3]);

    // Dump x_perm shape/strides for debugging
    // x_perm is the a-side input to the MUL in conv2d
    // Find it: it's the MUL result's src_ids[0]
    {
        // out is ADD result or RESHAPE result. Walk provenance to find MUL.
        // out = RESHAPE(SUM(MUL(x_perm, w_rs), axes), shape)
        // Actually, since we have out as a term, let's trace. The "out" from conv2d
        // is already reduced. Find the MUL result by looking for creator_op==UOP_MUL
        // among recent tensors.
        for (u32 i = ctx->tensor_count; i-- > 0; ) {
            if (ctx->tensors[i].creator_op == UOP_MUL && ctx->tensors[i].view.shape.rank == 8) {
                u32 aid = ctx->tensors[i].src_ids[0];
                TensorMeta *ma = &ctx->tensors[aid];
                printf("x_perm (tid=%u) shape=[", aid);
                for (u32 d=0;d<ma->view.shape.rank;d++) printf("%u%s", ma->view.shape.dims[d], d+1<ma->view.shape.rank?",":"");
                printf("] strides=[");
                for (u32 d=0;d<ma->view.shape.rank;d++) printf("%d%s", ma->view.strides[d], d+1<ma->view.shape.rank?",":"");
                printf("] offset=%d numel=%u\n", ma->view.offset, ma->view.numel);
                // Dump first few values via strided access
                f32 *raw = thvm_to_host(ctx, term_ten(aid, ma->dtype));
                printf("x_perm values (flat order from to_host):\n  ");
                u32 n = ma->view.numel < 36 ? ma->view.numel : 36;
                for (u32 j=0;j<n;j++) printf("%.0f ", raw[j]);
                printf("\n");
                break;
            }
        }
    }

    // Grad w.r.t. w (kernel)
    Term gw = thvm_reduce(ctx, thvm_grad(ctx, loss, w));
    f32 *gw_data = thvm_to_host(ctx, gw);
    printf("gw (TinyHVM) = [\n");
    for (int i=0;i<3;i++) {
        printf("  ");
        for (int j=0;j<3;j++) printf("%.1f ", gw_data[i*3+j]);
        printf("\n");
    }
    printf("]\n");

    // Expected: d_w[i,j] = sum_{oh,ow} gy[oh,ow] * x[oh+i, ow+j]
    // gy = ones(2,2), x = 1..16 in 4x4
    // d_w[0,0] = x[0,0]+x[0,1]+x[1,0]+x[1,1] = 1+2+5+6 = 14
    // d_w[0,1] = x[0,1]+x[0,2]+x[1,1]+x[1,2] = 2+3+6+7 = 18
    // d_w[0,2] = x[0,2]+x[0,3]+x[1,2]+x[1,3] = 3+4+7+8 = 22
    // d_w[1,0] = x[1,0]+x[1,1]+x[2,0]+x[2,1] = 5+6+9+10 = 30
    // d_w[1,1] = x[1,1]+x[1,2]+x[2,1]+x[2,2] = 6+7+10+11 = 34
    // d_w[1,2] = x[1,2]+x[1,3]+x[2,2]+x[2,3] = 7+8+11+12 = 38
    // d_w[2,0] = x[2,0]+x[2,1]+x[3,0]+x[3,1] = 9+10+13+14 = 46
    // d_w[2,1] = x[2,1]+x[2,2]+x[3,1]+x[3,2] = 10+11+14+15 = 50
    // d_w[2,2] = x[2,2]+x[2,3]+x[3,2]+x[3,3] = 11+12+15+16 = 54
    printf("expected = [\n  14 18 22\n  30 34 38\n  46 50 54\n]\n");

    thvm_free(ctx);
    return 0;
}
