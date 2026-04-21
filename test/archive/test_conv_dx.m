// test_conv_dx.m — Does a single conv backward produce correct dx on Metal?
// Tests dx = d(sum(conv(x, w)))/dx via autograd on CPU vs Metal
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void test(const char *dev) {
    srand(42);
    TinyHVM *ctx = thvm_init(dev);

    // x=[2,1,6,6], w=[2,1,3,3], conv → [2,2,4,4] → sum → scalar
    f32 xd[72]; for(int i=0;i<72;i++) xd[i]=(f32)rand()/(f32)RAND_MAX;
    f32 wd[18]; for(int i=0;i<18;i++) wd[i]=(f32)rand()/(f32)RAND_MAX;

    Term x = thvm_tensor(ctx, xd, (Shape){.dims={2,1,6,6},.rank=4});
    Term w = thvm_tensor(ctx, wd, (Shape){.dims={2,1,3,3},.rank=4});
    thvm_set_requires_grad(ctx, x);
    thvm_set_requires_grad(ctx, w);

    // conv(x, w, no_bias, groups=1, stride=1, no_pad)
    Term h = thvm_conv2d(ctx, x, w, term_era(), 1, (u32[]){1,1}, (u32[]){0,0,0,0});

    // loss = sum of all
    extern Term thvm_sum_axes(TinyHVM*, Term, const u32*, u32);
    Term flat = thvm_reshape(ctx, h, SHAPE(2*2*4*4));
    Term loss = thvm_sum_axes(ctx, flat, (u32[]){0}, 1);
    loss = thvm_reshape(ctx, loss, SHAPE(1));

    // Backward for both x and w
    f32 gxd[72]={0}, gwd[18]={0};
    Term gx = thvm_tensor(ctx, gxd, (Shape){.dims={2,1,6,6},.rank=4});
    Term gw = thvm_tensor(ctx, gwd, (Shape){.dims={2,1,3,3},.rank=4});
    Term grad = thvm_grad_multi(ctx, loss, (Term[]){x,w}, (Term[]){gx,gw}, 2);
    thvm_reduce(ctx, grad);

    f32 *dx = thvm_to_host(ctx, gx);
    f32 *dw = thvm_to_host(ctx, gw);
    double dx_norm=0, dw_norm=0;
    for(int i=0;i<72;i++) dx_norm+=dx[i]*dx[i];
    for(int i=0;i<18;i++) dw_norm+=dw[i]*dw[i];
    printf("%-6s: dx_norm=%.4f dw_norm=%.4f dx[0..3]=[%.4f,%.4f,%.4f,%.4f]\n",
        dev, sqrt(dx_norm), sqrt(dw_norm), dx[0], dx[1], dx[2], dx[3]);

    thvm_free(ctx);
}

int main(void) {
    test("cpu");
    test("metal");
    return 0;
}
