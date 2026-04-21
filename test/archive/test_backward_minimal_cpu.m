// test_backward_minimal_cpu.m — bisect backward regressions on CPU only
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>

static int run_mul_keep(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1,2,3, 4,5,6};
    f32 wd[] = {0.1f, 0.2f, 0.3f};
    Term x = thvm_tensor(ctx, xd, (Shape){.dims={2,3}, .rank=2});
    Term w = thvm_tensor(ctx, wd, (Shape){.dims={3}, .rank=1});
    thvm_set_requires_grad(ctx, w);

    Term h = thvm_op(ctx, UOP_MUL, x, w);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    Term loss = thvm_sum_axes(ctx, h, (u32[]){0,1}, 2);
    Term bundle = thvm_eval(ctx, thvm_grad_keep(ctx, loss, w));
    u32 n = thvm_grad_bundle_count(ctx, bundle);
    f32 *dw = n ? thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 0)) : NULL;
    printf("mul_keep: bundle=%u dw=[%.2f,%.2f,%.2f]\n", n, dw ? dw[0] : -1.0f, dw ? dw[1] : -1.0f,
           dw ? dw[2] : -1.0f);
    thvm_free(ctx);
    return (n == 1 && dw) ? 0 : 1;
}

static int run_mm_keep(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1,2,3,4,5,6};           // [2,3]
    f32 wd[] = {1,2,3,4,5,6};           // [3,2]
    Term x = thvm_tensor(ctx, xd, SHAPE(2,3));
    Term w = thvm_tensor(ctx, wd, SHAPE(3,2));
    thvm_set_requires_grad(ctx, w);

    Term y = thvm_mm(ctx, x, w);        // [2,2]
    Term loss = thvm_sum_axes(ctx, y, (u32[]){0,1}, 2);
    Term bundle = thvm_eval(ctx, thvm_grad_keep(ctx, loss, w));
    u32 n = thvm_grad_bundle_count(ctx, bundle);
    f32 *dw = n ? thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 0)) : NULL;
    printf("mm_keep: bundle=%u dw0=%.2f\n", n, dw ? dw[0] : -1.0f);
    thvm_free(ctx);
    return (n == 1 && dw) ? 0 : 1;
}

static int run_linear_bias_keep(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1,2,3, 4,5,6};          // [2,3]
    f32 wd[] = {
        0.10f, -0.20f,  0.30f,  0.40f,
       -0.50f,  0.60f, -0.70f,  0.80f,
        0.90f, -1.00f,  1.10f, -1.20f,
    };                                   // [3,4]
    f32 bd[] = {0.05f, -0.10f, 0.15f, 0.20f}; // [4]

    Term x = thvm_tensor(ctx, xd, (Shape){.dims={2,3}, .rank=2});
    Term w = thvm_tensor(ctx, wd, (Shape){.dims={3,4}, .rank=2});
    Term b = thvm_tensor(ctx, bd, (Shape){.dims={4}, .rank=1});
    thvm_set_requires_grad(ctx, w);
    thvm_set_requires_grad(ctx, b);

    Term logits = thvm_op(ctx, UOP_ADD,
        thvm_mm(ctx, x, w),
        thvm_expand(ctx, thvm_reshape(ctx, b, SHAPE(1,4)), SHAPE(2,4)));
    Term act = thvm_op(ctx, UOP_RELU, logits, term_era());
    Term loss = thvm_sum_axes(ctx, act, (u32[]){0,1}, 2);

    Term params[] = {w, b};
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));
    u32 n = thvm_grad_bundle_count(ctx, bundle);
    printf("linear_bias_keep: bundle=%u\n", n);
    thvm_free(ctx);
    return n == 2 ? 0 : 1;
}

int main(void) {
    int fails = 0;
    fprintf(stderr, "[1] mul keep\n"); fflush(stderr);
    fails += run_mul_keep();
    fprintf(stderr, "[2] mm keep\n"); fflush(stderr);
    fails += run_mm_keep();
    fprintf(stderr, "[3] linear+bias keep\n"); fflush(stderr);
    fails += run_linear_bias_keep();
    fprintf(stderr, "minimal backward checks: %s\n", fails ? "FAIL" : "OK"); fflush(stderr);
    return fails ? 1 : 0;
}
