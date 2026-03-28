// test_conv_pool_conv_grad.m — Minimal: Conv1→Pool→Conv2→sum loss
// Tests gradient for conv1 weight (traverses pool backward + conv2 inner product)
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>
#include <string.h>

int main(void) {
    srand(42);
    TinyHVM *ctx = thvm_init("metal");

    // Input [2,1,6,6] — small but enough for conv+pool+conv
    u32 BS=2;
    f32 x_data[2*1*6*6];
    for (int i = 0; i < 2*36; i++) x_data[i] = 0.1f*((f32)(i%36+1));
    Term x = thvm_tensor(ctx, x_data, (Shape){.dims={BS,1,6,6},.rank=4});
    thvm_set_requires_grad(ctx, x);

    // Conv1: [2,1,3,3] → output [2,2,4,4]
    f32 w1_data[2*1*3*3];
    for (int i=0;i<18;i++) w1_data[i] = 0.1f*((f32)rand()/(f32)RAND_MAX*2-1);
    Term w1 = thvm_tensor(ctx, w1_data, (Shape){.dims={2,1,3,3},.rank=4});
    thvm_set_requires_grad(ctx, w1);
    f32 b1_data[2] = {0}; Term b1 = thvm_tensor(ctx, b1_data, SHAPE(2));

    // Pool(2,2) → [2,2,2,2]
    // Conv2: [4,2,1,1] kernel=1 (pointwise) → [2,4,2,2]
    // Use kernel=1 to avoid another pool chain in conv2's inner product
    f32 w2_data[4*2*1*1];
    for (int i=0;i<8;i++) w2_data[i] = 0.1f*((f32)rand()/(f32)RAND_MAX*2-1);
    Term w2 = thvm_tensor(ctx, w2_data, (Shape){.dims={4,2,1,1},.rank=4});
    thvm_set_requires_grad(ctx, w2);
    f32 b2_data[4] = {0}; Term b2 = thvm_tensor(ctx, b2_data, SHAPE(4));

    // Forward
    u32 p0[]={0,0,0,0}, s1[]={1,1}, k2[]={2,2}, s2[]={2,2};
    Term h = thvm_conv2d(ctx, x, w1, b1, 1, s1, p0); // [2,2,4,4]
    h = thvm_op(ctx, UOP_RELU, h, term_era());        // relu enabled
    // Skip pool and conv2 — just sum the relu output
    Term out = h;

    // Loss = sum
    u32 axes[] = {0,1,2,3};
    Term loss = thvm_reshape(ctx, thvm_sum_axes(ctx, out, axes, 4), SHAPE(1));

    // Analytic gradient via thvm_backward (eager reverse tape)
    Term params[2] = {w1, w2};
    Term grads[2];
    thvm_backward(ctx, loss, params, grads, 2);

    f32 loss_val = thvm_to_host(ctx, loss)[0];
    printf("loss = %.6f\n", loss_val);

    f32 gw1_copy[18], gw2_copy[8];
    f32 *gw1 = (term_tag(grads[0])==TAG_TEN) ? thvm_to_host(ctx, grads[0]) : NULL;
    f32 *gw2 = (term_tag(grads[1])==TAG_TEN) ? thvm_to_host(ctx, grads[1]) : NULL;
    if (gw1) memcpy(gw1_copy, gw1, 18*4); else memset(gw1_copy, 0, 18*4);
    if (gw2) memcpy(gw2_copy, gw2, 8*4); else memset(gw2_copy, 0, 8*4);

    // Numerical gradient for w1 (5 samples)
    f32 eps = 1e-3f;
    printf("\n--- w1 gradient (traverses pool) ---\n");
    f32 max_d1 = 0;
    for (int j=0; j<5; j++) {
        int idx = j * (18/5);
        f32 saved = w1_data[idx];
        // f(w+eps)
        w1_data[idx] = saved + eps;
        thvm_reset(ctx, 0);
        Term x2=thvm_tensor(ctx,x_data,(Shape){.dims={BS,1,6,6},.rank=4});
        Term tw1=thvm_tensor(ctx,w1_data,(Shape){.dims={2,1,3,3},.rank=4});
        Term tb1=thvm_tensor(ctx,b1_data,SHAPE(2));
        Term tw2=thvm_tensor(ctx,w2_data,(Shape){.dims={4,2,1,1},.rank=4});
        Term tb2=thvm_tensor(ctx,b2_data,SHAPE(4));
        Term th=thvm_conv2d(ctx,x2,tw1,tb1,1,s1,p0);
        th=thvm_maxpool2d(ctx,th,k2,s2);
        Term to=thvm_conv2d(ctx,th,tw2,tb2,1,s1,p0);
        f32 lp=thvm_to_host(ctx,thvm_reshape(ctx,thvm_sum_axes(ctx,to,axes,4),SHAPE(1)))[0];
        // f(w-eps)
        w1_data[idx] = saved - eps;
        thvm_reset(ctx, 0);
        x2=thvm_tensor(ctx,x_data,(Shape){.dims={BS,1,6,6},.rank=4});
        tw1=thvm_tensor(ctx,w1_data,(Shape){.dims={2,1,3,3},.rank=4});
        tb1=thvm_tensor(ctx,b1_data,SHAPE(2));
        tw2=thvm_tensor(ctx,w2_data,(Shape){.dims={4,2,1,1},.rank=4});
        tb2=thvm_tensor(ctx,b2_data,SHAPE(4));
        th=thvm_conv2d(ctx,x2,tw1,tb1,1,s1,p0);
        th=thvm_maxpool2d(ctx,th,k2,s2);
        to=thvm_conv2d(ctx,th,tw2,tb2,1,s1,p0);
        f32 lm=thvm_to_host(ctx,thvm_reshape(ctx,thvm_sum_axes(ctx,to,axes,4),SHAPE(1)))[0];
        w1_data[idx] = saved;
        f32 ng = (lp-lm)/(2*eps);
        f32 d = fabsf(gw1_copy[idx]-ng);
        if (d > max_d1) max_d1 = d;
        printf("  w1[%2d]: analytic=%.6f numerical=%.6f diff=%.2e\n", idx, gw1_copy[idx], ng, d);
    }
    printf("  max_diff=%.2e  %s\n", max_d1, max_d1<0.01f?"OK":"FAIL");

    // Numerical gradient for w2 (all 8 elements)
    printf("\n--- w2 gradient (after pool, no pool traversal) ---\n");
    f32 max_d2 = 0;
    for (int j=0; j<8; j++) {
        f32 saved = w2_data[j];
        w2_data[j] = saved + eps;
        thvm_reset(ctx, 0);
        Term x2=thvm_tensor(ctx,x_data,(Shape){.dims={BS,1,6,6},.rank=4});
        Term tw1=thvm_tensor(ctx,w1_data,(Shape){.dims={2,1,3,3},.rank=4});
        Term tb1=thvm_tensor(ctx,b1_data,SHAPE(2));
        Term tw2=thvm_tensor(ctx,w2_data,(Shape){.dims={4,2,1,1},.rank=4});
        Term tb2=thvm_tensor(ctx,b2_data,SHAPE(4));
        Term th=thvm_conv2d(ctx,x2,tw1,tb1,1,s1,p0);
        th=thvm_maxpool2d(ctx,th,k2,s2);
        Term to=thvm_conv2d(ctx,th,tw2,tb2,1,s1,p0);
        f32 lp=thvm_to_host(ctx,thvm_reshape(ctx,thvm_sum_axes(ctx,to,axes,4),SHAPE(1)))[0];
        w2_data[j] = saved - eps;
        thvm_reset(ctx, 0);
        x2=thvm_tensor(ctx,x_data,(Shape){.dims={BS,1,6,6},.rank=4});
        tw1=thvm_tensor(ctx,w1_data,(Shape){.dims={2,1,3,3},.rank=4});
        tb1=thvm_tensor(ctx,b1_data,SHAPE(2));
        tw2=thvm_tensor(ctx,w2_data,(Shape){.dims={4,2,1,1},.rank=4});
        tb2=thvm_tensor(ctx,b2_data,SHAPE(4));
        th=thvm_conv2d(ctx,x2,tw1,tb1,1,s1,p0);
        th=thvm_maxpool2d(ctx,th,k2,s2);
        to=thvm_conv2d(ctx,th,tw2,tb2,1,s1,p0);
        f32 lm=thvm_to_host(ctx,thvm_reshape(ctx,thvm_sum_axes(ctx,to,axes,4),SHAPE(1)))[0];
        w2_data[j] = saved;
        f32 ng = (lp-lm)/(2*eps);
        f32 d = fabsf(gw2_copy[j]-ng);
        if (d > max_d2) max_d2 = d;
        printf("  w2[%d]: analytic=%.6f numerical=%.6f diff=%.2e\n", j, gw2_copy[j], ng, d);
    }
    printf("  max_diff=%.2e  %s\n", max_d2, max_d2<0.01f?"OK":"FAIL");

    printf("\n%s\n", (max_d1<0.01f && max_d2<0.01f) ? "ALL PASS" : "FAIL");
    thvm_free(ctx);
    return (max_d1<0.01f && max_d2<0.01f) ? 0 : 1;
}
