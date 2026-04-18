// test_matmul_grad_mat_assign.m — matmul grad + MAT-CTR destructure
// + SEQ(ASSIGN, ASSIGN), but NO recursion/loop.
//
// Program shape: MAT unpacks CTR(gw, gb) → bind them to lambda vars
// → SEQ(assign_w, assign_b). All one-shot, no REF/ALO.
//
// If this PASSES: bug is in recursion (REF/ALO/body clone).
// If this FAILS: bug is in MAT-CTR + matmul backward chain
//                (loop not required).

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
    f32 yd[M*N] = { 0.6f, -0.2f, 0.1f, 0.9f, -0.3f, 0.7f, -0.5f, 0.2f };
    f32 wd[K*N] = {
        0.10f, -0.20f,  0.30f,  0.40f,
       -0.50f,  0.60f, -0.70f,  0.80f,
        0.90f, -1.00f,  1.10f, -1.20f,
    };
    f32 bd[N]   = { 0.05f, -0.10f, 0.15f, 0.20f };
    f32 lrd[]   = { 0.05f };

    Term x  = thvm_tensor(ctx, xd, SHAPE(M, K));
    Term y  = thvm_tensor(ctx, yd, SHAPE(M, N));
    Term w  = thvm_tensor(ctx, wd, SHAPE(K, N));
    Term b  = thvm_tensor(ctx, bd, SHAPE(N));
    Term lr = thvm_tensor(ctx, lrd, SHAPE(1));
    thvm_set_requires_grad(ctx, w);
    thvm_set_requires_grad(ctx, b);

    Term w_fwd, w_grad, w_sgd, w_assign;
    {
        Term w1, w2;
        thvm_dup(ctx, thvm_fresh_label(ctx), w, &w_fwd, &w1);
        thvm_dup(ctx, thvm_fresh_label(ctx), w1, &w_grad, &w2);
        thvm_dup(ctx, thvm_fresh_label(ctx), w2, &w_sgd, &w_assign);
    }
    Term b_fwd, b_grad, b_sgd, b_assign;
    {
        Term b1, b2;
        thvm_dup(ctx, thvm_fresh_label(ctx), b, &b_fwd, &b1);
        thvm_dup(ctx, thvm_fresh_label(ctx), b1, &b_grad, &b2);
        thvm_dup(ctx, thvm_fresh_label(ctx), b2, &b_sgd, &b_assign);
    }

    Term pred = thvm_op(ctx, UOP_ADD,
        thvm_mm(ctx, x, w_fwd),
        thvm_expand(ctx, thvm_reshape(ctx, b_fwd, SHAPE(1, N)), SHAPE(M, N)));
    Term diff = thvm_op(ctx, UOP_SUB, pred, y);
    Term sq = thvm_op(ctx, UOP_MUL, diff, diff);
    Term loss = thvm_sum_axes(ctx, sq, (u32[]){0, 1}, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1, 1));
    Term inv_n = thvm_reshape(ctx, thvm_scalar(ctx, 1.0f / 8.0f), SHAPE(1, 1));
    loss = thvm_op(ctx, UOP_MUL, loss, inv_n);

    Term params[] = { w_grad, b_grad };
    // Do NOT pre-eval: force MAT-CTR to go through the defer path.
    Term grads = thvm_grad_multi_keep(ctx, loss, params, 2);

    // MAT-CTR destructure (same as loop test):
    // (λ gw. λ gb. SEQ(assign_w, assign_b)) applied to CTR(gw, gb)
    u64 l_gw = heap_alloc(ctx, 2);
    Term gw_var = term_new(TAG_VAR, 0, l_gw);
    heap_set(ctx, l_gw + 0, term_set_sub(gw_var));
    thvm_hint_shape(ctx, gw_var, SHAPE(K, N));

    u64 l_gb = heap_alloc(ctx, 2);
    Term gb_var = term_new(TAG_VAR, 0, l_gb);
    heap_set(ctx, l_gb + 0, term_set_sub(gb_var));
    thvm_hint_shape(ctx, gb_var, SHAPE(N));

    Term lr_w = thvm_expand(ctx, thvm_reshape(ctx, lr, SHAPE(1, 1)), SHAPE(K, N));
    Term lr_b = thvm_expand(ctx, lr, SHAPE(N));
    Term update_w = thvm_op(ctx, UOP_SUB, w_sgd, thvm_op(ctx, UOP_MUL, lr_w, gw_var));
    Term update_b = thvm_op(ctx, UOP_SUB, b_sgd, thvm_op(ctx, UOP_MUL, lr_b, gb_var));
    Term assign_w = thvm_assign(ctx, w_assign, update_w);
    Term assign_b = thvm_assign(ctx, b_assign, update_b);
    Term seq_body = thvm_seq(ctx, assign_w, assign_b);

    heap_set(ctx, l_gb + 1, seq_body);
    Term lam_gb = term_new(TAG_LAM, 0, l_gb);
    heap_set(ctx, l_gw + 1, lam_gb);
    Term lam_gw = term_new(TAG_LAM, 0, l_gw);

    Term grad_match = thvm_mat(ctx, 2, lam_gw, term_era());
    Term prog = thvm_app(ctx, grad_match, grads);
    // Wrap in IFZ to mimic the train loop's REF-body driven re-evaluation:
    // IFZ(counter, CTR(w,b), λ. prog) — force the program through a lambda
    // to get a second eval pass without full recursion.
    u64 l_cnt = heap_alloc(ctx, 2);
    Term cnt_var = term_new(TAG_VAR, 0, l_cnt);
    heap_set(ctx, l_cnt + 0, term_set_sub(cnt_var));
    heap_set(ctx, l_cnt + 1, prog);
    Term succ_lam = term_new(TAG_LAM, 0, l_cnt);
    Term ifz_prog = thvm_ifz(ctx, thvm_scalar_u32(ctx, 1), term_era(), succ_lam);
    thvm_eval(ctx, ifz_prog);

    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    f32 *wf = thvm_to_host_raw(ctx, w, &dtype, &shp);
    f32 *bf = thvm_to_host_raw(ctx, b, &dtype, &shp);

    // Expected computation (same as test_matmul_grad_assign)
    f32 pred_host[M*N];
    for (u32 i = 0; i < M; i++)
        for (u32 j = 0; j < N; j++) {
            f32 s = bd[j];
            for (u32 k = 0; k < K; k++) s += xd[i*K+k] * wd[k*N+j];
            pred_host[i*N+j] = s;
        }
    f32 diff_host[M*N];
    for (u32 i = 0; i < M*N; i++) diff_host[i] = pred_host[i] - yd[i];
    f32 exp_gw[K*N] = {0}, exp_gb[N] = {0};
    for (u32 k = 0; k < K; k++)
        for (u32 j = 0; j < N; j++) {
            f32 s = 0;
            for (u32 i = 0; i < M; i++) s += 0.25f * diff_host[i*N+j] * xd[i*K+k];
            exp_gw[k*N+j] = s;
        }
    for (u32 j = 0; j < N; j++) {
        f32 s = 0;
        for (u32 i = 0; i < M; i++) s += 0.25f * diff_host[i*N+j];
        exp_gb[j] = s;
    }
    f32 exp_w[K*N], exp_b[N];
    for (u32 p = 0; p < K*N; p++) exp_w[p] = wd[p] - lrd[0] * exp_gw[p];
    for (u32 p = 0; p < N; p++)   exp_b[p] = bd[p] - lrd[0] * exp_gb[p];

    printf("final_w: ");
    for (u32 p = 0; p < K*N; p++) printf("%8.4f ", wf[p]);
    printf("\nexpect:  ");
    for (u32 p = 0; p < K*N; p++) printf("%8.4f ", exp_w[p]);
    printf("\nfinal_b: ");
    for (u32 p = 0; p < N; p++) printf("%8.4f ", bf[p]);
    printf("\nexpect:  ");
    for (u32 p = 0; p < N; p++) printf("%8.4f ", exp_b[p]);
    printf("\n");

    int ok = 1;
    for (u32 p = 0; p < K*N; p++)
        if (fabsf(wf[p] - exp_w[p]) > 1e-4f) { ok = 0; break; }
    for (u32 p = 0; p < N; p++)
        if (fabsf(bf[p] - exp_b[p]) > 1e-4f) { ok = 0; break; }
    printf("%s\n", ok ? "PASS" : "FAIL");
    thvm_free(ctx);
    return ok ? 0 : 1;
}
