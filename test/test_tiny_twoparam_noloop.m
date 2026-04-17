// test_tiny_twoparam_noloop.m — diagnostic: multi-target GRAD without loop
//
// Same loss as test_tiny_twoparam_grad_loop.m but without recursion:
//   loss = sum((w + b - t)^2)
// Call thvm_grad_multi_keep once, read both grads, compare to host simulation.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 3

int main(void) {
    setbuf(stdout, NULL);
    TinyHVM *ctx = thvm_init("cpu");

    f32 wd[N] = { 0.5f, -1.0f, 0.25f };
    f32 bd[N] = { 0.1f,  0.2f, -0.3f };
    f32 td[N] = { 2.0f,  4.0f, -3.0f };

    Term w = thvm_tensor(ctx, wd, SHAPE(N));
    Term b = thvm_tensor(ctx, bd, SHAPE(N));
    Term target = thvm_tensor(ctx, td, SHAPE(N));
    thvm_set_requires_grad(ctx, w);
    thvm_set_requires_grad(ctx, b);

    Term sum_wb = thvm_op(ctx, UOP_ADD, w, b);
    Term diff   = thvm_op(ctx, UOP_SUB, sum_wb, target);
    Term sq     = thvm_op(ctx, UOP_MUL, diff, diff);
    Term loss   = thvm_sum_axes(ctx, sq, (u32[]){0}, 1);

    Term params[] = {w, b};
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));

    u32 bcount = thvm_grad_bundle_count(ctx, bundle);
    printf("bundle_count=%u (expect 2)\n", bcount);
    if (bcount != 2) { thvm_free(ctx); return 1; }

    Term gw = thvm_grad_bundle_get(ctx, bundle, 0);
    Term gb = thvm_grad_bundle_get(ctx, bundle, 1);
    printf("gw: tag=%u ext=%u val=%llu\n", (u32)term_tag(gw), (u32)term_ext(gw), (unsigned long long)term_val(gw));
    printf("gb: tag=%u ext=%u val=%llu\n", (u32)term_tag(gb), (u32)term_ext(gb), (unsigned long long)term_val(gb));

    f32 *dw = thvm_to_host(ctx, gw);
    f32 *db = thvm_to_host(ctx, gb);
    if (!dw || !db) {
        printf("grad tensors not readable (dw=%p db=%p)\n", (void*)dw, (void*)db);
        thvm_free(ctx); return 1;
    }

    // Expected: g = 2*(w + b - t)
    f32 exp_g[N];
    for (u32 i = 0; i < N; i++) exp_g[i] = 2.0f * (wd[i] + bd[i] - td[i]);

    printf("grad_w    = [%.4f, %.4f, %.4f]\n", dw[0], dw[1], dw[2]);
    printf("grad_b    = [%.4f, %.4f, %.4f]\n", db[0], db[1], db[2]);
    printf("expected  = [%.4f, %.4f, %.4f]\n", exp_g[0], exp_g[1], exp_g[2]);

    int ok_w = 1, ok_b = 1;
    for (u32 i = 0; i < N; i++) {
        if (fabsf(dw[i] - exp_g[i]) >= 1e-4f) ok_w = 0;
        if (fabsf(db[i] - exp_g[i]) >= 1e-4f) ok_b = 0;
    }
    printf("w %s, b %s\n", ok_w ? "OK" : "FAIL", ok_b ? "OK" : "FAIL");

    thvm_free(ctx);
    return (ok_w && ok_b) ? 0 : 1;
}
