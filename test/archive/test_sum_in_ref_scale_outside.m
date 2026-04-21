// test_sum_in_ref_scale_outside.m — REF body returns raw loss (sum).
// Outside: MUL by const, then grad. Tests whether the scale can happen
// OUTSIDE the REF without halving.
//
// If PASSES: half-magnitude requires the outer MUL to be INSIDE the
//   REF body. Workaround: move scaling outside.
// If FAILS: outer MUL + REF-returned sum is sufficient regardless of
//   where the MUL lives.

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

    // REF body: λw_p. sum(sum(expand(reshape(w_p))*pooled))
    u32 body_id = ctx->def_count++;
    u64 l_w_p = heap_alloc(ctx, 2);
    Term w_p = term_new(TAG_VAR, 0, l_w_p);
    heap_set(ctx, l_w_p + 0, term_set_sub(w_p));
    thvm_hint_shape(ctx, w_p, SHAPE(N));

    Term w_bc = thvm_expand(ctx, thvm_reshape(ctx, w_p, SHAPE(1, N)), SHAPE(M, N));
    Term pooled = thvm_sum_axes(ctx, w_bc, (u32[]){0}, 1);
    Term sq = thvm_op(ctx, UOP_MUL, pooled, pooled);
    Term loss_body = thvm_sum_axes(ctx, sq, (u32[]){0, 1}, 2);

    heap_set(ctx, l_w_p + 1, loss_body);
    Term lam = term_new(TAG_LAM, 0, l_w_p);
    ctx->defs[body_id] = lam;

    Term w_fwd, w_grad;
    thvm_dup(ctx, thvm_fresh_label(ctx), w, &w_fwd, &w_grad);
    Term b_grad;
    { Term bu; thvm_dup(ctx, thvm_fresh_label(ctx), b, &bu, &b_grad); (void)bu; }

    // Call REF, get raw sum. Outside: MUL by const, then grad.
    Term raw_loss = thvm_app(ctx, thvm_ref(ctx, body_id), w_fwd);
    Term raw_reshape = thvm_reshape(ctx, raw_loss, SHAPE(1, 1));
    Term half_const = thvm_reshape(ctx, thvm_scalar(ctx, 0.5f), SHAPE(1, 1));
    Term loss = thvm_op(ctx, UOP_MUL, raw_reshape, half_const);

    Term params[] = { w_grad, b_grad };
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));

    u32 count = thvm_grad_bundle_count(ctx, bundle);
    if (count != 2) { printf("FAIL: bundle count=%u\n", count); return 1; }

    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    Term gw = thvm_grad_bundle_get(ctx, bundle, 0);
    printf("gw tag=%u ext=%u val=%llu\n", (u32)term_tag(gw), (u32)term_ext(gw), (unsigned long long)term_val(gw));
    f32 *gwf = thvm_to_host_raw(ctx, gw, &dtype, &shp);
    if (!gwf) { printf("FAIL: gw not readable\n"); return 1; }

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
