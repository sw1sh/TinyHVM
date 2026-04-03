// test_conv_bwd_chain.m — Minimal test: is conv backward's da_ (input gradient) correct on Metal?
// Conv creates: pool → reshape → expand → permute → MUL → SUM → reshape
// Backward walks: reshape → SUM → MUL → permute → expand → reshape → pool inverse
// We test the MUL backward's "a" branch specifically.
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void test_device(const char *dev) {
    srand(42);
    TinyHVM *ctx = thvm_init(dev);
    u32 BS=2, Cin=1, H=6, W=6;
    u32 Cout=2, KH=3, KW=3;

    // Input
    u32 n = BS*Cin*H*W;
    f32 *xd = malloc(n*4);
    for (u32 i=0;i<n;i++) xd[i] = (f32)rand()/(f32)RAND_MAX * 2 - 1;
    Term x = thvm_tensor(ctx, xd, (Shape){.dims={BS,Cin,H,W},.rank=4});
    free(xd);
    thvm_set_requires_grad(ctx, x);

    // Weight
    u32 wn = Cout*Cin*KH*KW;
    f32 *wd = malloc(wn*4);
    for (u32 i=0;i<wn;i++) wd[i] = (f32)rand()/(f32)RAND_MAX * 2 - 1;
    Term w = thvm_tensor(ctx, wd, (Shape){.dims={Cout,Cin,KH,KW},.rank=4});
    free(wd);
    thvm_set_requires_grad(ctx, w);

    // Conv forward (no bias, no padding, stride 1)
    Term h = thvm_conv2d(ctx, x, w, term_era(), 1, (u32[]){1,1}, (u32[]){0,0,0,0});

    // Loss = sum (conv output: [BS, Cout, OY, OX] = [2,2,4,4])
    extern Term thvm_sum_axes(TinyHVM*, Term, const u32*, u32);
    Term flat = thvm_reshape(ctx, h, SHAPE(BS*Cout*4*4));
    Term loss = thvm_sum_axes(ctx, flat, (u32[]){0}, 1);
    loss = thvm_reshape(ctx, loss, SHAPE(1));
    printf("  conv out shape should be [%u,%u,%u,%u]\n", BS, Cout, H-KH+1, W-KW+1);

    // Backward for x
    f32 *gxd = calloc(n, 4);
    Term gx = thvm_tensor(ctx, gxd, (Shape){.dims={BS,Cin,H,W},.rank=4});
    free(gxd);
    Term grad = thvm_grad_multi(ctx, loss, (Term[]){x}, (Term[]){gx}, 1);
    thvm_reduce(ctx, grad);

    f32 *gx_data = thvm_to_host(ctx, gx);
    double norm = 0;
    for (u32 i=0;i<n;i++) norm += gx_data[i]*gx_data[i];
    printf("%-6s x_grad norm=%.6f first=[%.4f,%.4f,%.4f,%.4f]\n",
        dev, sqrt(norm), gx_data[0], gx_data[1], gx_data[2], gx_data[3]);

    thvm_free(ctx);
}

int main(void) {
    test_device("metal");
    test_device("cpu");
    return 0;
}
