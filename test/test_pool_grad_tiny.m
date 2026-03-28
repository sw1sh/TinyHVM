// test_pool_grad_tiny.m — Minimal maxpool gradient test
// MaxPool(2,2) on a tiny [1,1,4,4] input, check gradient against numpy.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>

int main(void) {
    TinyHVM *ctx = thvm_init(thvm_device("metal"));

    // Test 1: standalone maxpool
    // Input: [1,1,4,4] = 16 elements, known values
    f32 x_data[16];
    for (int i = 0; i < 16; i++) x_data[i] = (f32)(i + 1); // 1..16
    Term x = thvm_tensor(ctx, x_data, (Shape){.dims={1,1,4,4},.rank=4});
    thvm_set_requires_grad(ctx, x);

    // Test 2: Conv → ReLU → MaxPool chain (the failing case)
    // Conv(1,2,3,padding=0) → [1,2,2,2] → MaxPool(2,2) → [1,2,1,1]
    f32 w_data[2*1*3*3]; // [Cout=2, Cin=1, KH=3, KW=3]
    srand(42);
    for (int i = 0; i < 18; i++) w_data[i] = 0.1f * ((f32)rand()/(f32)RAND_MAX * 2 - 1);
    Term w = thvm_tensor(ctx, w_data, (Shape){.dims={2,1,3,3},.rank=4});
    thvm_set_requires_grad(ctx, w);
    f32 b_data[2] = {0};
    Term b = thvm_tensor(ctx, b_data, SHAPE(2));

    // ctx->no_fuse = 1; // fusion enabled
    u32 pad[] = {0,0,0,0}, stride[] = {1,1};
    Term h = thvm_conv2d(ctx, x, w, b, 1, stride, pad); // [1,2,2,2]
    h = thvm_op(ctx, UOP_RELU, h, term_era());

    // Test B: conv+relu+pool
    int use_pool = 1; // 0 = no pool, 1 = with pool
    u32 k[] = {2,2}, s[] = {2,2};
    Term target;
    if (use_pool) {
        target = thvm_maxpool2d(ctx, h, k, s); // [1,2,1,1]
    } else {
        target = h; // [1,2,2,2]
    }

    // Loss = sum
    u32 axes[] = {0,1,2,3};
    Term loss = thvm_reshape(ctx, thvm_sum_axes(ctx, target, axes, 4), SHAPE(1));

    // Check forward first
    printf("w tensor id = %u\n", (u32)term_val(w));
    f32 loss_fwd = thvm_to_host(ctx, loss)[0];
    printf("forward loss = %f\n", loss_fwd);
    fflush(stdout);

    // Gradient w.r.t. conv weight (flows through pool backward)
    printf("creating grad term...\n"); fflush(stdout);
    Term grad_w = thvm_grad(ctx, loss, w);
    printf("reducing grad...\n"); fflush(stdout);
    Term gw_r = thvm_reduce(ctx, grad_w);
    printf("grad reduced OK\n"); fflush(stdout);

    f32 loss_val = thvm_to_host(ctx, loss)[0];
    printf("loss = %.6f\n", loss_val);

    printf("calling to_host...\n"); fflush(stdout);
    f32 *gw_raw = thvm_to_host(ctx, gw_r);
    printf("to_host returned %p\n", (void*)gw_raw); fflush(stdout);
    // Copy gradient data before thvm_reset frees it
    f32 gw[18];
    if (gw_raw) memcpy(gw, gw_raw, 18 * sizeof(f32));
    else { printf("gw_raw is NULL!\n"); memset(gw, 0, sizeof(gw)); }
    printf("grad_w first 9 = [");
    for (int i = 0; i < 9; i++) printf("%.6f%s", gw[i], i<8?",":"");
    printf("]\n");

    printf("analytic grad done\n"); fflush(stdout);

    // Numerical gradient check (finite differences)
    f32 eps = 1e-3f;
    f32 max_diff = 0;
    for (int i = 0; i < 18; i++) {
        // f(w+eps)
        f32 saved = w_data[i];
        w_data[i] = saved + eps;
        printf("reset %d+...\n", i); fflush(stdout);
        thvm_reset(ctx, 0);
        printf("reset ok\n"); fflush(stdout);
        Term x2 = thvm_tensor(ctx, x_data, (Shape){.dims={1,1,4,4},.rank=4});
        Term w2 = thvm_tensor(ctx, w_data, (Shape){.dims={2,1,3,3},.rank=4});
        Term b2 = thvm_tensor(ctx, b_data, SHAPE(2));
        Term h2 = thvm_conv2d(ctx, x2, w2, b2, 1, stride, pad);
        h2 = thvm_op(ctx, UOP_RELU, h2, term_era());
        Term p2 = thvm_maxpool2d(ctx, h2, k, s);
        Term l2 = thvm_reshape(ctx, thvm_sum_axes(ctx, p2, axes, 4), SHAPE(1));
        f32 lp = thvm_to_host(ctx, thvm_reduce(ctx, l2))[0];

        // f(w-eps)
        w_data[i] = saved - eps;
        thvm_reset(ctx, 0);
        x2 = thvm_tensor(ctx, x_data, (Shape){.dims={1,1,4,4},.rank=4});
        w2 = thvm_tensor(ctx, w_data, (Shape){.dims={2,1,3,3},.rank=4});
        b2 = thvm_tensor(ctx, b_data, SHAPE(2));
        h2 = thvm_conv2d(ctx, x2, w2, b2, 1, stride, pad);
        h2 = thvm_op(ctx, UOP_RELU, h2, term_era());
        p2 = thvm_maxpool2d(ctx, h2, k, s);
        l2 = thvm_reshape(ctx, thvm_sum_axes(ctx, p2, axes, 4), SHAPE(1));
        f32 lm = thvm_to_host(ctx, thvm_reduce(ctx, l2))[0];

        w_data[i] = saved;
        f32 numerical = (lp - lm) / (2 * eps);
        f32 diff = fabsf(gw[i] - numerical);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3f || i < 9)
            printf("  gw[%2d]: analytic=%.6f numerical=%.6f diff=%.2e\n",
                   i, gw[i], numerical, diff);
    }
    printf("max_diff = %.6e\n", max_diff);
    printf("%s\n", max_diff < 1e-2f ? "PASS" : "FAIL");

    thvm_free(ctx);
    return max_diff < 1e-5 ? 0 : 1;
}
