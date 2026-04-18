// test_sum_detach_sq_ref.m — same as test_sum_sq_ref but DETACH the
// SUM output before self-MUL. DETACH forces materialization to a TEN
// (atomic, no further DUP commutes through it).
//
// NOTE: DETACH breaks backward gradient flow (that's its purpose —
// treat the value as a constant). So gw will be 0 IF gradient stops
// at the DETACH. Still useful as a check: we expect the eval to at
// least succeed and return a CTR with SOMETHING per-element (not
// scalar-broadcast error values). If gw is uniform-per-row (same
// failure pattern), the bug is unrelated to the DUP commute of
// the SUM output. If gw is zeros, DETACH is actually dropping the
// backward as designed.

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

    // Forward with DETACH on the SUM output before self-MUL.
    Term w_bc = thvm_expand(ctx, thvm_reshape(ctx, w_fwd, SHAPE(1, N)), SHAPE(M, N));
    Term pooled = thvm_sum_axes(ctx, w_bc, (u32[]){0}, 1);     // [1, N]
    Term detached = thvm_detach(ctx, pooled);                   // breaks gradient
    Term sq = thvm_op(ctx, UOP_MUL, detached, detached);
    Term loss = thvm_sum_axes(ctx, sq, (u32[]){0, 1}, 2);

    Term params[] = { w_grad, b_grad };
    Term grads = thvm_grad_multi_keep(ctx, loss, params, 2);

    u64 l_gw = heap_alloc(ctx, 2);
    Term gw_var = term_new(TAG_VAR, 0, l_gw);
    heap_set(ctx, l_gw + 0, term_set_sub(gw_var));
    thvm_hint_shape(ctx, gw_var, SHAPE(N));
    u64 l_gb = heap_alloc(ctx, 2);
    Term gb_var = term_new(TAG_VAR, 0, l_gb);
    heap_set(ctx, l_gb + 0, term_set_sub(gb_var));
    thvm_hint_shape(ctx, gb_var, SHAPE(N));

    Term out_ctr = thvm_ctr(ctx, (Term[]){gw_var, gb_var}, 2);
    heap_set(ctx, l_gb + 1, out_ctr);
    Term lam_gb = term_new(TAG_LAM, 0, l_gb);
    heap_set(ctx, l_gw + 1, lam_gb);
    Term lam_gw = term_new(TAG_LAM, 0, l_gw);
    Term grad_match = thvm_mat(ctx, 2, lam_gw, term_era());
    Term body = thvm_app(ctx, grad_match, grads);

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
    if (term_tag(result) != TAG_CTR) {
        printf("FAIL: eval didn't reduce to CTR\n");
        return 1;
    }

    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    Term gw = thvm_grad_bundle_get(ctx, result, 0);
    printf("gw_term tag=%u ext=%u val=%llu\n",
           (u32)term_tag(gw), (u32)term_ext(gw),
           (unsigned long long)term_val(gw));
    f32 *gwf = thvm_to_host_raw(ctx, gw, &dtype, &shp);
    if (!gwf) {
        printf("gw_host=NULL — DETACH cut the gradient path before a TEN was deposited.\n");
        thvm_free(ctx);
        return 0;
    }

    printf("gw:      ");
    for (u32 p = 0; p < N; p++) printf("%8.4f ", gwf[p]);
    printf("\n");
    // No strict expected — DETACH breaks gradient, so gw = 0 is the
    // "expected" per-design. The key observation is whether the
    // uniform-per-row pattern reappears or not.
    int all_zero = 1, all_same = 1;
    for (u32 p = 0; p < N; p++) {
        if (fabsf(gwf[p]) > 1e-6f) all_zero = 0;
        if (fabsf(gwf[p] - gwf[0]) > 1e-6f) all_same = 0;
    }
    printf("gw is %s; %s\n",
           all_zero ? "all-zero (DETACH cut gradient as designed)"
                    : (all_same ? "uniform-nonzero (same bug pattern!)"
                                : "per-element (something interesting)"),
           all_zero ? "OK" : "CHECK");
    thvm_free(ctx);
    return 0;
}
