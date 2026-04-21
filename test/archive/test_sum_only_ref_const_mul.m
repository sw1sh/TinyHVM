// test_sum_only_ref_const_mul.m — REF body returns POOLED (single SUM).
// Outside: MUL with non-uniform constant c=[1,2,3,4], sum, grad.
//
// Forward: pooled[0,j] = M*w[j]. sq[0,j] = M*w[j] * c[j].
//          loss = sum(sq) = M * sum(w[j] * c[j]).
//          d(loss)/d(w[j]) = M * c[j] — non-uniform, per-element.
//
// If PASSES: SUM backward through ALO works when the outer op has a
//            constant operand. Bug requires both MUL operands to go
//            through ALO.
// If FAILS:  Even with one constant operand, SUM backward through ALO
//            has some issue. Narrower isolation of SUM-only bug.

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
    f32 cd[1 * N] = { 1.0f, 2.0f, 3.0f, 4.0f };

    Term w = thvm_tensor(ctx, wd, SHAPE(N));
    Term b = thvm_tensor(ctx, bd, SHAPE(N));
    Term c_const = thvm_tensor(ctx, cd, SHAPE(1, N));
    thvm_set_requires_grad(ctx, w);
    thvm_set_requires_grad(ctx, b);

    // REF body: λw_p. sum(expand(reshape(w_p, [1,N]), [M,N]), axis 0)
    // Returns POOLED only.
    u32 body_id = ctx->def_count++;
    u64 l_w_p = heap_alloc(ctx, 2);
    Term w_p = term_new(TAG_VAR, 0, l_w_p);
    heap_set(ctx, l_w_p + 0, term_set_sub(w_p));
    thvm_hint_shape(ctx, w_p, SHAPE(N));

    Term w_bc = thvm_expand(ctx, thvm_reshape(ctx, w_p, SHAPE(1, N)), SHAPE(M, N));
    Term pooled_body = thvm_sum_axes(ctx, w_bc, (u32[]){0}, 1);

    heap_set(ctx, l_w_p + 1, pooled_body);
    Term lam = term_new(TAG_LAM, 0, l_w_p);
    ctx->defs[body_id] = lam;

    Term w_fwd, w_grad;
    thvm_dup(ctx, thvm_fresh_label(ctx), w, &w_fwd, &w_grad);
    Term b_grad;
    { Term bu; thvm_dup(ctx, thvm_fresh_label(ctx), b, &bu, &b_grad); (void)bu; }

    // Call REF, get pooled. Outside: MUL(pooled, c_const), sum, grad.
    Term pooled = thvm_app(ctx, thvm_ref(ctx, body_id), w_fwd);
    Term sq = thvm_op(ctx, UOP_MUL, pooled, c_const);
    Term loss = thvm_sum_axes(ctx, sq, (u32[]){0, 1}, 2);

    Term params[] = { w_grad, b_grad };
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));

    printf("bundle tag=%u count=%u\n", (u32)term_tag(bundle),
           thvm_grad_bundle_count(ctx, bundle));

    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    Term gw = thvm_grad_bundle_get(ctx, bundle, 0);
    f32 *gwf = thvm_to_host_raw(ctx, gw, &dtype, &shp);
    if (!gwf) { printf("FAIL: gw not readable, tag=%u\n", term_tag(gw)); return 1; }

    f32 exp_gw[N];
    for (u32 j = 0; j < N; j++) exp_gw[j] = (f32)M * cd[j];

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
