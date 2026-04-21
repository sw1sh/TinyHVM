// test_add_two_sums_grad_ref.m — REF body uses TWO independent SUMs of
// expand(w_p) but combined with ADD (not MUL). Outside: square, sum.
//
// Forward: pooled_a + pooled_b = 2*M*w[j]. sq = (2*M*w[j])^2 = 4*M^2*w[j]^2.
// loss = sum(sq) = 4*M^2*sum(w^2). d(loss)/d(w[j]) = 8*M^2*w[j].
//
// ADD's BG rule is BG_GY — DUPs only gy; a_shape/b_shape derived, but
// `at`/`bt` are NOT reused in the gradient construction. If MUL-specific
// bt-DUP is the bug trigger, ADD should pass even with two SUM operands.
//
// If PASSES: bug is specifically MUL's BG rule with two SUM operands.
//            ADD backward handles pair-of-SUM just fine.
// If FAILS:  bug is any binary op with two SUM operands in backward
//            through ALO — broader than just MUL.

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

    u32 body_id = ctx->def_count++;
    u64 l_w_p = heap_alloc(ctx, 2);
    Term w_p = term_new(TAG_VAR, 0, l_w_p);
    heap_set(ctx, l_w_p + 0, term_set_sub(w_p));
    thvm_hint_shape(ctx, w_p, SHAPE(N));

    Term wa, wb;
    thvm_dup(ctx, thvm_fresh_label(ctx), w_p, &wa, &wb);

    Term wa_bc = thvm_expand(ctx, thvm_reshape(ctx, wa, SHAPE(1, N)), SHAPE(M, N));
    Term pooled_a = thvm_sum_axes(ctx, wa_bc, (u32[]){0}, 1);
    Term wb_bc = thvm_expand(ctx, thvm_reshape(ctx, wb, SHAPE(1, N)), SHAPE(M, N));
    Term pooled_b = thvm_sum_axes(ctx, wb_bc, (u32[]){0}, 1);

    // ADD instead of MUL
    Term sum_body = thvm_op(ctx, UOP_ADD, pooled_a, pooled_b);

    heap_set(ctx, l_w_p + 1, sum_body);
    Term lam = term_new(TAG_LAM, 0, l_w_p);
    ctx->defs[body_id] = lam;

    Term w_fwd, w_grad;
    thvm_dup(ctx, thvm_fresh_label(ctx), w, &w_fwd, &w_grad);
    Term b_grad;
    { Term bu; thvm_dup(ctx, thvm_fresh_label(ctx), b, &bu, &b_grad); (void)bu; }

    Term added = thvm_app(ctx, thvm_ref(ctx, body_id), w_fwd);
    Term sq = thvm_op(ctx, UOP_MUL, added, added);  // square outside REF
    Term loss = thvm_sum_axes(ctx, sq, (u32[]){0, 1}, 2);

    Term params[] = { w_grad, b_grad };
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));

    printf("bundle tag=%u count=%u\n", (u32)term_tag(bundle),
           thvm_grad_bundle_count(ctx, bundle));

    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    Term gw = thvm_grad_bundle_get(ctx, bundle, 0);
    printf("gw tag=%u ext=%u val=%llu\n", (u32)term_tag(gw), (u32)term_ext(gw), (unsigned long long)term_val(gw));
    f32 *gwf = thvm_to_host_raw(ctx, gw, &dtype, &shp);
    if (!gwf) { printf("FAIL: gw not readable (gw is NUM(0) = zero-gradient erase path)\n"); return 1; }

    f32 exp_gw[N];
    for (u32 j = 0; j < N; j++) exp_gw[j] = 8.0f * (f32)M * (f32)M * wd[j];

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
