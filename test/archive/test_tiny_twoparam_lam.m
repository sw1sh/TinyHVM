// test_tiny_twoparam_lam.m — diagnostic: multi-target GRAD inside a LAM-APP
//
// Like test_tiny_twoparam_noloop.m but the forward/backward lives in a
// lambda body and w/b come in as binders:
//
//   program = (λw. λb. loss(w, b) via grad_multi_keep) w0 b0
//
// Isolates whether LAM substitution + DUP chain on the binders is what
// breaks the loop version.

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

    Term w0 = thvm_tensor(ctx, wd, SHAPE(N));
    Term b0 = thvm_tensor(ctx, bd, SHAPE(N));
    Term target = thvm_tensor(ctx, td, SHAPE(N));
    thvm_set_requires_grad(ctx, w0);
    thvm_set_requires_grad(ctx, b0);

    // Build λw. λb. grad_multi_keep(sum((w+b-target)^2), [w, b])
    u64 l_w = heap_alloc(ctx, 2);
    Term w_var = term_new(TAG_VAR, 0, l_w);
    heap_set(ctx, l_w + 0, term_set_sub(w_var));
    thvm_hint_shape(ctx, w_var, SHAPE(N));

    u64 l_b = heap_alloc(ctx, 2);
    Term b_var = term_new(TAG_VAR, 0, l_b);
    heap_set(ctx, l_b + 0, term_set_sub(b_var));
    thvm_hint_shape(ctx, b_var, SHAPE(N));

    // Minimal DUP — just forward and grad projections
    Term w_fwd, w_grad;
    thvm_dup(ctx, thvm_fresh_label(ctx), w_var, &w_fwd, &w_grad);
    Term b_fwd, b_grad;
    thvm_dup(ctx, thvm_fresh_label(ctx), b_var, &b_fwd, &b_grad);

    Term sum_wb = thvm_op(ctx, UOP_ADD, w_fwd, b_fwd);
    Term diff   = thvm_op(ctx, UOP_SUB, sum_wb, target);
    Term sq     = thvm_op(ctx, UOP_MUL, diff, diff);
    Term loss   = thvm_sum_axes(ctx, sq, (u32[]){0}, 1);

    Term params[] = {w_grad, b_grad};
    Term body = thvm_grad_multi_keep(ctx, loss, params, 2);

    heap_set(ctx, l_b + 1, body);
    Term lam_b = term_new(TAG_LAM, 0, l_b);
    heap_set(ctx, l_w + 1, lam_b);
    Term lam_w = term_new(TAG_LAM, 0, l_w);

    Term program = thvm_app(ctx, thvm_app(ctx, lam_w, w0), b0);
    Term bundle = thvm_eval(ctx, program);

    u32 bcount = thvm_grad_bundle_count(ctx, bundle);
    printf("bundle_count=%u (expect 2)\n", bcount);
    if (bcount != 2) { thvm_free(ctx); return 1; }

    Term gw = thvm_grad_bundle_get(ctx, bundle, 0);
    Term gb = thvm_grad_bundle_get(ctx, bundle, 1);
    f32 *dw = thvm_to_host(ctx, gw);
    f32 *db = thvm_to_host(ctx, gb);
    if (!dw || !db) {
        printf("grads not readable\n");
        thvm_free(ctx); return 1;
    }

    f32 exp_g[N];
    for (u32 i = 0; i < N; i++) exp_g[i] = 2.0f * (wd[i] + bd[i] - td[i]);

    printf("grad_w   = [%.4f, %.4f, %.4f]\n", dw[0], dw[1], dw[2]);
    printf("grad_b   = [%.4f, %.4f, %.4f]\n", db[0], db[1], db[2]);
    printf("expected = [%.4f, %.4f, %.4f]\n", exp_g[0], exp_g[1], exp_g[2]);

    int ok_w = 1, ok_b = 1;
    for (u32 i = 0; i < N; i++) {
        if (fabsf(dw[i] - exp_g[i]) >= 1e-4f) ok_w = 0;
        if (fabsf(db[i] - exp_g[i]) >= 1e-4f) ok_b = 0;
    }
    printf("w %s, b %s\n", ok_w ? "OK" : "FAIL", ok_b ? "OK" : "FAIL");

    thvm_free(ctx);
    return (ok_w && ok_b) ? 0 : 1;
}
