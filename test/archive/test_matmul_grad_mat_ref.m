// test_matmul_grad_mat_ref.m — same structure as linear_sgd_loop but
// only ONE iteration. Runs through the REF/IFZ/MAT-CTR/SEQ(ASSIGN)
// path exactly once (counter=1 → succ branch → recurse with counter=0
// → base case returns CTR).
//
// If this PASSES: the bug is in the multi-iteration interaction
//                 (cloning of train body more than once).
// If this FAILS: the bug is in REF/ALO body realization (one
//                iteration is enough to trigger it).

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

static Term build_train(TinyHVM *ctx, Term w0, Term b0,
                         Term x, Term y, Term lr, u32 n_steps) {
    u32 train_id = ctx->def_count++;

    u64 l_counter = heap_alloc(ctx, 2);
    Term counter_var = term_new(TAG_VAR, 0, l_counter);
    heap_set(ctx, l_counter + 0, term_set_sub(counter_var));

    u64 l_w = heap_alloc(ctx, 2);
    Term w_var = term_new(TAG_VAR, 0, l_w);
    heap_set(ctx, l_w + 0, term_set_sub(w_var));
    thvm_hint_shape(ctx, w_var, SHAPE(K, N));

    u64 l_b = heap_alloc(ctx, 2);
    Term b_var = term_new(TAG_VAR, 0, l_b);
    heap_set(ctx, l_b + 0, term_set_sub(b_var));
    thvm_hint_shape(ctx, b_var, SHAPE(N));

    u64 l_next = heap_alloc(ctx, 2);
    Term next_counter = term_new(TAG_VAR, 0, l_next);
    heap_set(ctx, l_next + 0, term_set_sub(next_counter));

    Term w_fwd, w_grad, w_sgd, w_assign, w_next, w_done;
    {
        Term w1, w2, w3, w4;
        thvm_dup(ctx, thvm_fresh_label(ctx), w_var, &w_fwd, &w1);
        thvm_dup(ctx, thvm_fresh_label(ctx), w1, &w_grad, &w2);
        thvm_dup(ctx, thvm_fresh_label(ctx), w2, &w_sgd, &w3);
        thvm_dup(ctx, thvm_fresh_label(ctx), w3, &w_assign, &w4);
        thvm_dup(ctx, thvm_fresh_label(ctx), w4, &w_next, &w_done);
    }
    Term b_fwd, b_grad, b_sgd, b_assign, b_next, b_done;
    {
        Term b1, b2, b3, b4;
        thvm_dup(ctx, thvm_fresh_label(ctx), b_var, &b_fwd, &b1);
        thvm_dup(ctx, thvm_fresh_label(ctx), b1, &b_grad, &b2);
        thvm_dup(ctx, thvm_fresh_label(ctx), b2, &b_sgd, &b3);
        thvm_dup(ctx, thvm_fresh_label(ctx), b3, &b_assign, &b4);
        thvm_dup(ctx, thvm_fresh_label(ctx), b4, &b_next, &b_done);
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
    Term grads = thvm_grad_multi_keep(ctx, loss, params, 2);

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

    Term rec = thvm_app(ctx,
               thvm_app(ctx,
               thvm_app(ctx, thvm_ref(ctx, train_id), next_counter),
                        w_next), b_next);
    Term seq_body = thvm_seq(ctx, assign_w, thvm_seq(ctx, assign_b, rec));

    heap_set(ctx, l_gb + 1, seq_body);
    Term lam_gb = term_new(TAG_LAM, 0, l_gb);
    heap_set(ctx, l_gw + 1, lam_gb);
    Term lam_gw = term_new(TAG_LAM, 0, l_gw);

    Term grad_match = thvm_mat(ctx, 2, lam_gw, term_era());
    Term succ_body = thvm_app(ctx, grad_match, grads);
    heap_set(ctx, l_next + 1, succ_body);
    Term succ_lam = term_new(TAG_LAM, 0, l_next);

    Term done = thvm_ctr(ctx, (Term[]){w_done, b_done}, 2);
    heap_set(ctx, l_b + 1, thvm_ifz(ctx, counter_var, done, succ_lam));
    Term lam_b = term_new(TAG_LAM, 0, l_b);
    heap_set(ctx, l_w + 1, lam_b);
    Term lam_w = term_new(TAG_LAM, 0, l_w);
    heap_set(ctx, l_counter + 1, lam_w);
    ctx->defs[train_id] = term_new(TAG_LAM, 0, l_counter);

    return thvm_app(ctx,
           thvm_app(ctx,
           thvm_app(ctx, thvm_ref(ctx, train_id), thvm_scalar_u32(ctx, n_steps)),
                    w0), b0);
}

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

    Term prog = build_train(ctx, w, b, x, y, lr, 1);
    thvm_eval(ctx, prog);

    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    f32 *wf = thvm_to_host_raw(ctx, w, &dtype, &shp);
    f32 *bf = thvm_to_host_raw(ctx, b, &dtype, &shp);

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
