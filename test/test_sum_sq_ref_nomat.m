// test_sum_sq_ref_nomat.m — same as test_sum_sq_ref but REF body
// returns the grad bundle directly (no MAT-CTR destructure).
// The outer caller then reads gw/gb from the bundle.
//
// If PASSES: bug needs MAT-CTR defer to trigger; REF alone + reduction
//            + self-MUL backward works.
// If FAILS: REF alone is enough; MAT-CTR isn't required to trigger.

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

    u32 train_id = ctx->def_count++;

    u64 l_k = heap_alloc(ctx, 2);
    Term k_var = term_new(TAG_VAR, 0, l_k);
    heap_set(ctx, l_k + 0, term_set_sub(k_var));
    u64 l_w_p = heap_alloc(ctx, 2);
    Term w_p = term_new(TAG_VAR, 0, l_w_p);
    heap_set(ctx, l_w_p + 0, term_set_sub(w_p));
    thvm_hint_shape(ctx, w_p, SHAPE(N));
    u64 l_b_p = heap_alloc(ctx, 2);
    Term b_p = term_new(TAG_VAR, 0, l_b_p);
    heap_set(ctx, l_b_p + 0, term_set_sub(b_p));
    thvm_hint_shape(ctx, b_p, SHAPE(N));

    Term w_fwd, w_grad;
    thvm_dup(ctx, thvm_fresh_label(ctx), w_p, &w_fwd, &w_grad);
    Term b_grad;
    { Term bu; thvm_dup(ctx, thvm_fresh_label(ctx), b_p, &bu, &b_grad); (void)bu; }

    Term w_bc = thvm_expand(ctx, thvm_reshape(ctx, w_fwd, SHAPE(1, N)), SHAPE(M, N));
    Term pooled = thvm_sum_axes(ctx, w_bc, (u32[]){0}, 1);
    Term sq = thvm_op(ctx, UOP_MUL, pooled, pooled);
    Term loss = thvm_sum_axes(ctx, sq, (u32[]){0, 1}, 2);

    Term params[] = { w_grad, b_grad };
    // REF body returns the grad bundle DIRECTLY (no MAT-CTR).
    Term body = thvm_grad_multi_keep(ctx, loss, params, 2);

    heap_set(ctx, l_b_p + 1, body);
    Term lam_b_p = term_new(TAG_LAM, 0, l_b_p);
    heap_set(ctx, l_w_p + 1, lam_b_p);
    Term lam_w_p = term_new(TAG_LAM, 0, l_w_p);
    heap_set(ctx, l_k + 1, lam_w_p);
    Term lam_k = term_new(TAG_LAM, 0, l_k);
    ctx->defs[train_id] = lam_k;

    Term prog = thvm_app(ctx,
                thvm_app(ctx,
                thvm_app(ctx, thvm_ref(ctx, train_id), thvm_scalar_u32(ctx, 1)),
                         w),
                         b);
    Term result = thvm_eval(ctx, prog);
    printf("result_tag=%u\n", term_tag(result));
    u32 count = thvm_grad_bundle_count(ctx, result);
    printf("bundle count = %u\n", count);
    if (count != 2) { printf("FAIL: expected bundle of 2\n"); return 1; }

    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    Term gw = thvm_grad_bundle_get(ctx, result, 0);
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
