// test_matmul_selfmul_loss_ref.m — matmul + sum(pred*pred), no SUB.
//
// Loss = sum(pred * pred). No diff, no target subtraction.
//
// gy for pred = 2 * pred  (per-element MUL(pred,pred) backward)
// gw = x.T @ (2 * pred)  per-element
//
// If PASSES: the bug needs the SUB step or the target tensor.
// If FAILS: self-MUL + matmul-decomp + defer is sufficient to trigger.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define M 2
#define K 3
#define N 4

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    setbuf(stdout, NULL);
    TinyHVM *ctx = thvm_init("cpu");

    f32 xd[M*K] = { 1.0f, 2.0f, -1.0f, 0.5f, -1.5f, 2.0f };
    f32 wd[K*N] = {
        0.10f, -0.20f,  0.30f,  0.40f,
       -0.50f,  0.60f, -0.70f,  0.80f,
        0.90f, -1.00f,  1.10f, -1.20f,
    };
    f32 bd[N]   = { 0.05f, -0.10f, 0.15f, 0.20f };

    Term x = thvm_tensor(ctx, xd, SHAPE(M, K));
    Term w = thvm_tensor(ctx, wd, SHAPE(K, N));
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
    thvm_hint_shape(ctx, w_p, SHAPE(K, N));
    u64 l_b_p = heap_alloc(ctx, 2);
    Term b_p = term_new(TAG_VAR, 0, l_b_p);
    heap_set(ctx, l_b_p + 0, term_set_sub(b_p));
    thvm_hint_shape(ctx, b_p, SHAPE(N));

    Term w_fwd, w_grad;
    thvm_dup(ctx, thvm_fresh_label(ctx), w_p, &w_fwd, &w_grad);
    Term b_fwd, b_grad;
    thvm_dup(ctx, thvm_fresh_label(ctx), b_p, &b_fwd, &b_grad);

    // Forward: pred = matmul(x, w) + b,  loss = sum(pred * pred)
    Term pred = thvm_op(ctx, UOP_ADD,
        thvm_mm(ctx, x, w_fwd),
        thvm_expand(ctx, thvm_reshape(ctx, b_fwd, SHAPE(1, N)), SHAPE(M, N)));
    Term sq = thvm_op(ctx, UOP_MUL, pred, pred);
    Term loss = thvm_sum_axes(ctx, sq, (u32[]){0, 1}, 2);

    Term params[] = { w_grad, b_grad };
    Term grads = thvm_grad_multi_keep(ctx, loss, params, 2);

    u64 l_gw = heap_alloc(ctx, 2);
    Term gw_var = term_new(TAG_VAR, 0, l_gw);
    heap_set(ctx, l_gw + 0, term_set_sub(gw_var));
    thvm_hint_shape(ctx, gw_var, SHAPE(K, N));

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
    if (term_tag(result) != TAG_CTR) {
        printf("FAIL: expected CTR, got tag=%u\n", term_tag(result));
        return 1;
    }

    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    Term gw = thvm_grad_bundle_get(ctx, result, 0);
    Term gb = thvm_grad_bundle_get(ctx, result, 1);
    f32 *gwf = thvm_to_host_raw(ctx, gw, &dtype, &shp);
    f32 *gbf = thvm_to_host_raw(ctx, gb, &dtype, &shp);

    // Expected: gy for pred = 2 * pred (from MUL(pred,pred) BG)
    //           gw[k,j] = sum_i (2 * pred[i,j] * x[i,k])
    //           gb[j]   = sum_i (2 * pred[i,j])
    f32 pred_host[M*N];
    for (u32 i = 0; i < M; i++)
        for (u32 j = 0; j < N; j++) {
            f32 s = bd[j];
            for (u32 k = 0; k < K; k++) s += xd[i*K + k] * wd[k*N + j];
            pred_host[i*N + j] = s;
        }
    f32 exp_gw[K*N] = {0}, exp_gb[N] = {0};
    for (u32 k = 0; k < K; k++)
        for (u32 j = 0; j < N; j++) {
            f32 s = 0;
            for (u32 i = 0; i < M; i++) s += 2.0f * pred_host[i*N + j] * xd[i*K + k];
            exp_gw[k*N + j] = s;
        }
    for (u32 j = 0; j < N; j++) {
        f32 s = 0;
        for (u32 i = 0; i < M; i++) s += 2.0f * pred_host[i*N + j];
        exp_gb[j] = s;
    }

    printf("gw:      ");
    for (u32 p = 0; p < K*N; p++) printf("%8.4f ", gwf[p]);
    printf("\nexp_gw:  ");
    for (u32 p = 0; p < K*N; p++) printf("%8.4f ", exp_gw[p]);
    printf("\ngb:      ");
    for (u32 p = 0; p < N; p++) printf("%8.4f ", gbf[p]);
    printf("\nexp_gb:  ");
    for (u32 p = 0; p < N; p++) printf("%8.4f ", exp_gb[p]);
    printf("\n");

    int ok = 1;
    for (u32 p = 0; p < K*N; p++)
        if (fabsf(gwf[p] - exp_gw[p]) > 1e-4f) { ok = 0; break; }
    for (u32 p = 0; p < N; p++)
        if (fabsf(gbf[p] - exp_gb[p]) > 1e-4f) { ok = 0; break; }
    printf("%s\n", ok ? "PASS" : "FAIL");
    thvm_free(ctx);
    return ok ? 0 : 1;
}
