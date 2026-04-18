// test_sum_sq_noref_scale.m — test_sum_sq_noref + outer MUL(loss, 0.5f).
// No REF. Direct thvm_eval on grad_multi_keep. Adds constant scaling.
//
// If PASSES: outer-MUL-scale works fine in pure DYN path; bug is
//   specific to REF+ALO path (which threads gy differently).
// If FAILS half-magnitude: bug is in BG MUL handling of constant
//   bt regardless of REF. Narrows to pure grad chain mechanism.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define M 2
#define N 4

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    setbuf(stdout, NULL);
    TinyHVM *ctx = thvm_init("cpu");

    f32 wd[N] = { 0.10f, -0.20f, 0.30f, 0.40f };
    f32 bd[N] = { 0.05f, -0.10f, 0.15f, 0.20f };

    Term w = thvm_tensor(ctx, wd, SHAPE(N));
    Term b = thvm_tensor(ctx, bd, SHAPE(N));
    thvm_set_requires_grad(ctx, w);
    thvm_set_requires_grad(ctx, b);

    Term w_fwd, w_grad;
    thvm_dup(ctx, thvm_fresh_label(ctx), w, &w_fwd, &w_grad);
    Term b_grad;
    { Term bu; thvm_dup(ctx, thvm_fresh_label(ctx), b, &bu, &b_grad); (void)bu; }

    Term w_bc = thvm_expand(ctx, thvm_reshape(ctx, w_fwd, SHAPE(1, N)), SHAPE(M, N));
    Term pooled = thvm_sum_axes(ctx, w_bc, (u32[]){0}, 1);
    Term sq = thvm_op(ctx, UOP_MUL, pooled, pooled);
    Term loss = thvm_sum_axes(ctx, sq, (u32[]){0, 1}, 2);
    // Add trailing MUL(loss, 0.5f) with constant [1,1] scalar.
    loss = thvm_reshape(ctx, loss, SHAPE(1, 1));
    Term half_const = thvm_reshape(ctx, thvm_scalar(ctx, 0.5f), SHAPE(1, 1));
    loss = thvm_op(ctx, UOP_MUL, loss, half_const);

    Term params[] = { w_grad, b_grad };
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));

    u32 count = thvm_grad_bundle_count(ctx, bundle);
    if (count != 2) { printf("FAIL: bundle count=%u\n", count); return 1; }

    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    Term gw = thvm_grad_bundle_get(ctx, bundle, 0);
    f32 *gwf = thvm_to_host_raw(ctx, gw, &dtype, &shp);
    if (!gwf) { printf("FAIL: gw not readable\n"); return 1; }

    // Expected: gw = 0.5 * 2*M^2*w[j] = M^2*w[j]
    f32 exp_gw[N];
    for (u32 j = 0; j < N; j++) exp_gw[j] = (f32)M * (f32)M * wd[j];

    printf("gw:      ");
    for (u32 p = 0; p < N; p++) printf("%8.4f ", gwf[p]);
    printf("\nexp_gw:  ");
    for (u32 p = 0; p < N; p++) printf("%8.4f ", exp_gw[p]);
    printf("\n");

    int ok = 1;
    for (u32 p = 0; p < N; p++) if (fabsf(gwf[p] - exp_gw[p]) > 1e-4f) { ok = 0; break; }
    printf("%s\n", ok ? "PASS" : "FAIL");
    thvm_free(ctx);
    return ok ? 0 : 1;
}
