// test_ic_grad_bisect.m — Bisect: which combination breaks IC gradient?
// Minimal architecture, IC-native path, finite-difference check on conv1 weight
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>
#include <string.h>

// Build forward graph and return loss term
static Term build_fwd(TinyHVM *ctx, f32 *xd, f32 *w1d, f32 *b1d,
                       u32 BS, u32 IH, u32 IW, int use_relu, int use_pool) {
    Term x = thvm_tensor(ctx, xd, (Shape){.dims={BS,1,IH,IW},.rank=4});
    thvm_set_requires_grad(ctx, x);
    Term w1 = thvm_tensor(ctx, w1d, (Shape){.dims={4,1,3,3},.rank=4});
    thvm_set_requires_grad(ctx, w1);
    Term b1 = thvm_tensor(ctx, b1d, SHAPE(4));

    u32 p0[]={0,0,0,0}, s1[]={1,1};
    Term h = thvm_conv2d(ctx, x, w1, b1, 1, s1, p0);
    if (use_relu) h = thvm_op(ctx, UOP_RELU, h, term_era());
    if (use_pool) {
        u32 k2[]={2,2}, s2[]={2,2};
        h = thvm_maxpool2d(ctx, h, k2, s2);
    }
    u32 ax[]={0,1,2,3};
    return thvm_reshape(ctx, thvm_sum_axes(ctx, h, ax, 4), SHAPE(1));
}

static void test_config(const char *name, u32 BS, u32 IH, u32 IW,
                          int use_relu, int use_pool) {
    TinyHVM *ctx = thvm_init("metal");
    srand(42);

    u32 n = BS*IH*IW;
    f32 *xd = malloc(n*4);
    for (u32 i=0;i<n;i++) xd[i] = 0.01f*(f32)(i%100+1);
    f32 w1d[4*1*3*3];
    for (int i=0;i<36;i++) w1d[i] = 0.05f*((f32)rand()/(f32)RAND_MAX*2-1);
    f32 b1d[4] = {0};

    // Analytic gradient via IC
    Term loss = build_fwd(ctx, xd, w1d, b1d, BS, IH, IW, use_relu, use_pool);
    Term w1_term = term_ten(1, DTYPE_F32); // tensor 1 = w1 (after x=0)
    // Find w1 tensor ID
    u32 w1_id = 0;
    for (u32 i = 0; i < ctx->tensor_count; i++)
        if (ctx->tensors[i].requires_grad && ctx->tensors[i].view.numel == 36) { w1_id = i; break; }
    w1_term = term_ten(w1_id, DTYPE_F32);

    Term params[1] = {w1_term};
    f32 z[36] = {0};
    Term gs[1] = {thvm_tensor(ctx, z, (Shape){.dims={4,1,3,3},.rank=4})};
    Term gt = thvm_grad_multi(ctx, loss, params, gs, 1);
    thvm_reduce(ctx, thvm_app(ctx, gt, term_era()));

    f32 lv = thvm_to_host(ctx, loss)[0];
    f32 *gw = thvm_to_host(ctx, gs[0]);
    f32 gw_copy[36];
    if (gw) memcpy(gw_copy, gw, 36*4); else memset(gw_copy, 0, 36*4);

    // Numerical gradient — 3 samples
    f32 eps = 1e-3f, max_d = 0;
    for (int j=0; j<3; j++) {
        int idx = j*12;
        f32 saved = w1d[idx];

        w1d[idx] = saved + eps;
        thvm_reset(ctx, 0);
        Term lp_t = build_fwd(ctx, xd, w1d, b1d, BS, IH, IW, use_relu, use_pool);
        f32 lp = thvm_to_host(ctx, lp_t)[0];

        w1d[idx] = saved - eps;
        thvm_reset(ctx, 0);
        Term lm_t = build_fwd(ctx, xd, w1d, b1d, BS, IH, IW, use_relu, use_pool);
        f32 lm = thvm_to_host(ctx, lm_t)[0];

        w1d[idx] = saved;
        f32 ng = (lp-lm)/(2*eps);
        f32 d = fabsf(gw_copy[idx] - ng);
        if (d > max_d) max_d = d;
        f32 rel = (ng != 0) ? d/fabsf(ng) : 0;
        fprintf(stderr, "  [%d] a=%.4f n=%.4f abs=%.2e rel=%.2e\n", idx, gw_copy[idx], ng, d, rel);
    }
    printf("  %-40s loss=%.3f max_d=%.2e %s\n", name, lv, max_d, max_d<0.01f?"OK":"FAIL");
    free(xd); thvm_free(ctx);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <test_id>\n", argv[0]);
        printf("  0: conv only, BS=2, 6x6\n");
        printf("  1: conv+relu, BS=2, 6x6\n");
        printf("  2: conv+pool, BS=2, 6x6\n");
        printf("  3: conv+relu+pool, BS=2, 6x6\n");
        printf("  4: conv only, BS=4, 28x28\n");
        printf("  5: conv+relu, BS=4, 28x28\n");
        printf("  6: conv+pool, BS=4, 28x28\n");
        printf("  7: conv+relu+pool, BS=4, 28x28\n");
        printf("  8: conv+relu+pool, BS=2, 8x8\n");
        printf("  9: conv+relu+pool, BS=4, 8x8\n");
        return 1;
    }
    int t = atoi(argv[1]);
    switch(t) {
        case 0: test_config("conv, BS=2, 6x6",          2,6,6,0,0); break;
        case 1: test_config("conv+relu, BS=2, 6x6",     2,6,6,1,0); break;
        case 2: test_config("conv+pool, BS=2, 6x6",     2,6,6,0,1); break;
        case 3: test_config("conv+relu+pool, BS=2, 6x6", 2,6,6,1,1); break;
        case 4: test_config("conv, BS=4, 28x28",         4,28,28,0,0); break;
        case 5: test_config("conv+relu, BS=4, 28x28",    4,28,28,1,0); break;
        case 6: test_config("conv+pool, BS=4, 28x28",    4,28,28,0,1); break;
        case 7: test_config("conv+relu+pool, BS=4, 28x28",4,28,28,1,1); break;
        case 8: test_config("conv+relu+pool, BS=2, 8x8", 2,8,8,1,1); break;
        case 9: test_config("conv+relu+pool, BS=4, 8x8", 4,8,8,1,1); break;
        case 10: { // custom: conv only, BS=2, size from argv[2]
            int sz = argc>2 ? atoi(argv[2]) : 10;
            char nm[64]; snprintf(nm,sizeof(nm),"conv, BS=2, %dx%d",sz,sz);
            test_config(nm, 2, (u32)sz, (u32)sz, 0, 0); break; }
    }
    return 0;
}
