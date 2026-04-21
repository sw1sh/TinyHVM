// test_sum_in_ref_dup_outside.m — REF body returns POOLED (just
// expand+sum, no DUP, no MUL). Outside the REF: user-DUP the pooled,
// MUL(dp0, dp1), sum to scalar, then grad on w.
//
// Differentiates from test_sum_dupmul_grad_ref (DUP inside REF, FAIL)
// and test_sum_sq_lam_app (everything in LAM, PASS):
//
// If PASSES: the trigger is having the DUP inside the REF body. The
//            SUM can be ALO-realized fine; the DUP itself is what
//            commutes wrong with the ALO-realized SUM.
// If FAILS:  the trigger is just "ALO-realized SUM in backward DUP
//            chain", regardless of where the DUP lives. The bug fires
//            on the DUP-of-pooled regardless of placement.

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

    // REF body: λw_p. sum(expand(reshape(w_p, [1,N]), [M,N]), axis 0)
    // Returns POOLED only — no DUP, no MUL.
    u32 body_id = ctx->def_count++;
    u64 l_w_p = heap_alloc(ctx, 2);
    Term w_p = term_new(TAG_VAR, 0, l_w_p);
    heap_set(ctx, l_w_p + 0, term_set_sub(w_p));
    thvm_hint_shape(ctx, w_p, SHAPE(N));

    Term w_bc = thvm_expand(ctx, thvm_reshape(ctx, w_p, SHAPE(1, N)), SHAPE(M, N));
    Term pooled_body = thvm_sum_axes(ctx, w_bc, (u32[]){0}, 1);    // [1, N]

    heap_set(ctx, l_w_p + 1, pooled_body);
    Term lam = term_new(TAG_LAM, 0, l_w_p);
    ctx->defs[body_id] = lam;

    // DUPs for grad targets
    Term w_fwd, w_grad;
    thvm_dup(ctx, thvm_fresh_label(ctx), w, &w_fwd, &w_grad);
    Term b_grad;
    { Term bu; thvm_dup(ctx, thvm_fresh_label(ctx), b, &bu, &b_grad); (void)bu; }

    // Call REF, get pooled. Then OUTSIDE REF: user-DUP, MUL, sum.
    Term pooled = thvm_app(ctx, thvm_ref(ctx, body_id), w_fwd);
    Term dp0, dp1;
    thvm_dup(ctx, thvm_fresh_label(ctx), pooled, &dp0, &dp1);
    Term sq = thvm_op(ctx, UOP_MUL, dp0, dp1);
    Term loss = thvm_sum_axes(ctx, sq, (u32[]){0, 1}, 2);

    Term params[] = { w_grad, b_grad };
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));

    printf("bundle tag=%u count=%u\n", (u32)term_tag(bundle),
           thvm_grad_bundle_count(ctx, bundle));

    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    Term gw = thvm_grad_bundle_get(ctx, bundle, 0);
    f32 *gwf = thvm_to_host_raw(ctx, gw, &dtype, &shp);
    if (!gwf) { printf("FAIL: gw not readable\n"); return 1; }

    f32 exp_gw[N];
    for (u32 j = 0; j < N; j++) exp_gw[j] = 2.0f * (f32)M * (f32)M * wd[j];

    printf("gw:      ");
    for (u32 p = 0; p < N; p++) printf("%8.4f ", gwf[p]);
    printf("\nexp_gw:  ");
    for (u32 p = 0; p < N; p++) printf("%8.4f ", exp_gw[p]);
    printf("\n");

    int ok = 1;
    for (u32 p = 0; p < N; p++)
        if (fabsf(gwf[p] - exp_gw[p]) > 1e-4f) { ok = 0; break; }
    printf("%s\n", ok ? "PASS" : "FAIL");
    thvm_free(ctx);
    return ok ? 0 : 1;
}
