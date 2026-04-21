// test_ic_grad_conv.m — Verify IC-native GRAD path for conv+relu+sum
// Uses thvm_grad_multi (IC path), NOT thvm_backward (eager tape)
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
    u32 BS = 2;

    f32 x_data[2*1*6*6];
    for (int i = 0; i < 72; i++) x_data[i] = 0.1f * (f32)(i % 36 + 1);
    Term x = thvm_tensor(ctx, x_data, (Shape){.dims={BS,1,6,6},.rank=4});
    thvm_set_requires_grad(ctx, x);

    f32 w1_data[2*1*3*3];
    for (int i = 0; i < 18; i++) w1_data[i] = 0.1f * ((f32)rand()/(f32)RAND_MAX*2-1);
    Term w1 = thvm_tensor(ctx, w1_data, (Shape){.dims={2,1,3,3},.rank=4});
    thvm_set_requires_grad(ctx, w1);
    f32 b1_data[2] = {0};
    Term b1 = thvm_tensor(ctx, b1_data, SHAPE(2));

    u32 p0[] = {0,0,0,0}, s1[] = {1,1};
    // Conv2: pointwise [4,2,1,1]
    f32 w2_data[4*2*1*1];
    for (int i = 0; i < 8; i++) w2_data[i] = 0.1f * ((f32)rand()/(f32)RAND_MAX*2-1);
    Term w2 = thvm_tensor(ctx, w2_data, (Shape){.dims={4,2,1,1},.rank=4});
    thvm_set_requires_grad(ctx, w2);
    f32 b2_data[4] = {0};
    Term b2 = thvm_tensor(ctx, b2_data, SHAPE(4));

    u32 k2[] = {2,2}, s2[] = {2,2};
    Term h = thvm_conv2d(ctx, x, w1, b1, 1, s1, p0); // [2,2,4,4]
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    h = thvm_maxpool2d(ctx, h, k2, s2);               // [2,2,2,2]
    Term out = thvm_conv2d(ctx, h, w2, b2, 1, s1, p0); // [2,4,2,2]

    // Simple SUM loss
    u32 axes[] = {0,1,2,3};
    Term loss = thvm_reshape(ctx, thvm_sum_axes(ctx, out, axes, 4), SHAPE(1));

    // IC-native gradient via thvm_grad_multi — both weights
    #define NP 2
    Term params[NP] = {w1, w2};
    Term gs[NP];
    f32 z1[18] = {0}, z2[8] = {0};
    gs[0] = thvm_tensor(ctx, z1, (Shape){.dims={2,1,3,3},.rank=4});
    gs[1] = thvm_tensor(ctx, z2, (Shape){.dims={4,2,1,1},.rank=4});

    Term grad_term = thvm_grad_multi(ctx, loss, params, gs, NP);
    Term chain = thvm_app(ctx, grad_term, term_era());
    thvm_reduce(ctx, chain);

    f32 loss_val = thvm_to_host(ctx, loss)[0];
    printf("loss = %.6f\n", loss_val);

    f32 *gw = thvm_to_host(ctx, gs[0]);
    f32 gw_copy[18];
    if (gw) memcpy(gw_copy, gw, 18*4);
    else { printf("gw1 is NULL — GRAD returned ERA\n"); memset(gw_copy, 0, 18*4); }

    // Read w2 gradient BEFORE numerical loop destroys tensors
    f32 *gw2_raw = thvm_to_host(ctx, gs[1]);
    f32 gw2_copy[8];
    if (gw2_raw) memcpy(gw2_copy, gw2_raw, 8*4);
    else { printf("gw2 is NULL — GRAD returned ERA\n"); memset(gw2_copy, 0, 8*4); }

    printf("analytic first 9: [");
    for (int i = 0; i < 9; i++) printf("%.4f%s", gw_copy[i], i<8?",":"");
    printf("]\n");

    // Numerical gradient for w1 (finite differences)
    f32 eps = 1e-3f, max_diff = 0;
    printf("\n--- w1 (traverses pool) ---\n");
    for (int j = 0; j < 5; j++) {
        int idx = j * 3;
        f32 saved = w1_data[idx];

        w1_data[idx] = saved + eps;
        thvm_reset(ctx, 0);
        { Term tx=thvm_tensor(ctx,x_data,(Shape){.dims={BS,1,6,6},.rank=4});
          Term tw=thvm_tensor(ctx,w1_data,(Shape){.dims={2,1,3,3},.rank=4});
          Term tb=thvm_tensor(ctx,b1_data,SHAPE(2));
          Term tw2=thvm_tensor(ctx,w2_data,(Shape){.dims={4,2,1,1},.rank=4});
          Term tb2=thvm_tensor(ctx,b2_data,SHAPE(4));
          Term th=thvm_conv2d(ctx,tx,tw,tb,1,s1,p0);
          th=thvm_op(ctx,UOP_RELU,th,term_era());
          th=thvm_maxpool2d(ctx,th,k2,s2);
          Term to=thvm_conv2d(ctx,th,tw2,tb2,1,s1,p0);
          f32 lp=thvm_to_host(ctx,thvm_reshape(ctx,thvm_sum_axes(ctx,to,axes,4),SHAPE(1)))[0];

          w1_data[idx] = saved - eps;
          thvm_reset(ctx, 0);
          tx=thvm_tensor(ctx,x_data,(Shape){.dims={BS,1,6,6},.rank=4});
          tw=thvm_tensor(ctx,w1_data,(Shape){.dims={2,1,3,3},.rank=4});
          tb=thvm_tensor(ctx,b1_data,SHAPE(2));
          tw2=thvm_tensor(ctx,w2_data,(Shape){.dims={4,2,1,1},.rank=4});
          tb2=thvm_tensor(ctx,b2_data,SHAPE(4));
          th=thvm_conv2d(ctx,tx,tw,tb,1,s1,p0);
          th=thvm_op(ctx,UOP_RELU,th,term_era());
          th=thvm_maxpool2d(ctx,th,k2,s2);
          to=thvm_conv2d(ctx,th,tw2,tb2,1,s1,p0);
          f32 lm=thvm_to_host(ctx,thvm_reshape(ctx,thvm_sum_axes(ctx,to,axes,4),SHAPE(1)))[0];

          w1_data[idx] = saved;
          f32 ng=(lp-lm)/(2*eps);
          f32 d=fabsf(gw_copy[idx]-ng);
          if(d>max_diff) max_diff=d;
          printf("  w1[%2d]: a=%.6f n=%.6f d=%.2e\n",idx,gw_copy[idx],ng,d);
        }
    }
    printf("w1 max_diff = %.6e  %s\n", max_diff, max_diff<0.01f?"OK":"FAIL");

    printf("\n--- w2 (after pool) ---\n");
    f32 max_d2 = 0;
    for (int j = 0; j < 4; j++) {
        f32 saved = w2_data[j];
        w2_data[j] = saved + eps;
        thvm_reset(ctx, 0);
        { Term tx=thvm_tensor(ctx,x_data,(Shape){.dims={BS,1,6,6},.rank=4});
          Term tw=thvm_tensor(ctx,w1_data,(Shape){.dims={2,1,3,3},.rank=4});
          Term tb=thvm_tensor(ctx,b1_data,SHAPE(2));
          Term tw2=thvm_tensor(ctx,w2_data,(Shape){.dims={4,2,1,1},.rank=4});
          Term tb2=thvm_tensor(ctx,b2_data,SHAPE(4));
          Term th=thvm_conv2d(ctx,tx,tw,tb,1,s1,p0);
          th=thvm_op(ctx,UOP_RELU,th,term_era());
          th=thvm_maxpool2d(ctx,th,k2,s2);
          Term to=thvm_conv2d(ctx,th,tw2,tb2,1,s1,p0);
          f32 lp=thvm_to_host(ctx,thvm_reshape(ctx,thvm_sum_axes(ctx,to,axes,4),SHAPE(1)))[0];

          w2_data[j] = saved - eps;
          thvm_reset(ctx, 0);
          tx=thvm_tensor(ctx,x_data,(Shape){.dims={BS,1,6,6},.rank=4});
          tw=thvm_tensor(ctx,w1_data,(Shape){.dims={2,1,3,3},.rank=4});
          tb=thvm_tensor(ctx,b1_data,SHAPE(2));
          tw2=thvm_tensor(ctx,w2_data,(Shape){.dims={4,2,1,1},.rank=4});
          tb2=thvm_tensor(ctx,b2_data,SHAPE(4));
          th=thvm_conv2d(ctx,tx,tw,tb,1,s1,p0);
          th=thvm_op(ctx,UOP_RELU,th,term_era());
          th=thvm_maxpool2d(ctx,th,k2,s2);
          to=thvm_conv2d(ctx,th,tw2,tb2,1,s1,p0);
          f32 lm=thvm_to_host(ctx,thvm_reshape(ctx,thvm_sum_axes(ctx,to,axes,4),SHAPE(1)))[0];

          w2_data[j] = saved;
          f32 ng=(lp-lm)/(2*eps);
          f32 d=fabsf(gw2_copy[j]-ng);
          if(d>max_d2) max_d2=d;
          printf("  w2[%d]: a=%.6f n=%.6f d=%.2e\n",j,gw2_copy[j],ng,d);
        }
    }
    printf("w2 max_diff = %.6e  %s\n", max_d2, max_d2<0.01f?"OK":"FAIL");

    int ok = (max_diff<0.01f && max_d2<0.01f);
    printf("\n%s\n", ok ? "ALL PASS" : "FAIL");
    thvm_free(ctx);
    return max_diff < 0.01f ? 0 : 1;
}
