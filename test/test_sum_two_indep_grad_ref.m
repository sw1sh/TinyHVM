// test_sum_two_indep_grad_ref.m — REF body uses TWO INDEPENDENT sums
// of expand(w_p) as MUL operands. No user DUP, no shared term.
// pooled_a = sum(expand(reshape(w,[1,N])),[M,N], axis 0);
// pooled_b = sum(expand(reshape(w,[1,N])),[M,N], axis 0);  // separate
// MUL(pooled_a, pooled_b) = pooled^2 elementwise.
//
// Since each pooled_X is a fresh TOP chain, there's no user-DUP in
// the body. Forward evaluation still requires w_p to be read twice
// (implicit), so there's still a DUP in the graph — but on w_p, not
// on pooled.
//
// If PASSES: bug is specifically user-DUP-of-SUM through ALO.
//            DUPing raw inputs (VARs) and independent SUMs work fine.
// If FAILS: bug is about ANY shared-pooled-through-backward, even
//           when both sides are independently-built SUM ALOs.

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

    // REF body: λw_p. { DUP w_p into (wa, wb);
    //                   pooled_a = sum(expand(reshape(wa,[1,N]),[M,N]), axis 0);
    //                   pooled_b = sum(expand(reshape(wb,[1,N]),[M,N]), axis 0);
    //                   return pooled_a * pooled_b } — returns [1, N]
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

    Term sq_body = thvm_op(ctx, UOP_MUL, pooled_a, pooled_b);

    heap_set(ctx, l_w_p + 1, sq_body);
    Term lam = term_new(TAG_LAM, 0, l_w_p);
    ctx->defs[body_id] = lam;

    Term w_fwd, w_grad;
    thvm_dup(ctx, thvm_fresh_label(ctx), w, &w_fwd, &w_grad);
    Term b_grad;
    { Term bu; thvm_dup(ctx, thvm_fresh_label(ctx), b, &bu, &b_grad); (void)bu; }

    Term sq = thvm_app(ctx, thvm_ref(ctx, body_id), w_fwd);
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
