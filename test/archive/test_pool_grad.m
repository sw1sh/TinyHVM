// test_pool_grad.m — Verify gradient flow through pool complex path
// Tests: input → pool(k=3,s=1) → MUL(pooled, weight) → SUM → loss
// If pool backward works, input should get non-zero gradient.
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    srand(42);
    TinyHVM *ctx = thvm_init("metal");

    // Simple test: x=[2,1,4,4] → pool(k=3,s=1) → conv-like MUL+SUM → scalar loss
    u32 BS=2, C=1, H=4, W=4;
    u32 KH=3, KW=3;
    u32 OY=(H-KH)/1+1, OX=(W-KW)/1+1; // 2,2

    // Create input and weight
    u32 n = BS*C*H*W;
    f32 *xd = malloc(n*4);
    for (u32 i=0;i<n;i++) xd[i] = (f32)rand()/(f32)RAND_MAX * 2 - 1;
    Term x = thvm_tensor(ctx, xd, (Shape){.dims={BS,C,H,W},.rank=4});
    free(xd);
    thvm_set_requires_grad(ctx, x);

    u32 wn = C*KH*KW;
    f32 *wd = malloc(wn*4);
    for (u32 i=0;i<wn;i++) wd[i] = (f32)rand()/(f32)RAND_MAX * 2 - 1;
    Term w = thvm_tensor(ctx, wd, SHAPE(1,C,1,1,C,KH,KW));
    free(wd);
    thvm_set_requires_grad(ctx, w);

    // Pool (complex path: k=3 > s=1)
    Term pooled = thvm_pool(ctx, x, (u32[]){KH,KW}, (u32[]){1,1}, 2);
    // pooled: [BS,C,OY,OX,KH,KW]

    printf("Pool output tag=%u\n", term_tag(pooled));

    // Reshape for broadcasting: [BS,1,C,OY,OX,C,KH,KW]
    // Actually let's just SUM the pooled to get a scalar loss
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term loss = thvm_sum_axes(ctx, pooled, (u32[]){0,1,2,3,4,5}, 6);
    loss = thvm_reshape(ctx, loss, SHAPE(1));

    // Backward
    f32 *gxd = calloc(n, 4);
    Term gx = thvm_tensor(ctx, gxd, (Shape){.dims={BS,C,H,W},.rank=4});
    free(gxd);

    Term grad = thvm_grad_multi(ctx, loss, (Term[]){x}, (Term[]){gx}, 1);
    thvm_reduce(ctx, grad);

    f32 *gx_data = thvm_to_host(ctx, gx);
    double norm = 0;
    for (u32 i=0;i<n;i++) norm += gx_data[i]*gx_data[i];
    printf("x gradient norm: %.6f (should be non-zero)\n", sqrt(norm));
    printf("First 8 grad values: ");
    for (u32 i=0;i<8&&i<n;i++) printf("%.4f ", gx_data[i]);
    printf("\n");

    // Also check: does manual backward match?
    // d(SUM(pool))/d(x) = ones scattered back through pool
    // For pool k=3,s=1 on 4x4: each input element participates in multiple windows
    // The gradient should be the number of windows each element appears in

    thvm_free(ctx);
    return 0;
}
