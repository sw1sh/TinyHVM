// test_tiny_twoparam_sum_loop.m — simplest multi-param loop: loss = sum(w + b)
//
// Only one backward path per target (no MUL(diff,diff) splitting).
// Expected gradient for both w and b is uniform 1.0.
// If b still comes out != 1.0, the bug isn't in the 2-contribution pattern.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 3

static Term recursive_train(TinyHVM *ctx, Term w0, Term b0, Term lr, u32 n_steps) {
    u32 train_id = ctx->def_count++;
    assert(train_id < 256);

    u64 l_counter = heap_alloc(ctx, 2);
    Term counter_var = term_new(TAG_VAR, 0, l_counter);
    heap_set(ctx, l_counter + 0, term_set_sub(counter_var));

    u64 l_w = heap_alloc(ctx, 2);
    Term w_var = term_new(TAG_VAR, 0, l_w);
    heap_set(ctx, l_w + 0, term_set_sub(w_var));
    thvm_hint_shape(ctx, w_var, SHAPE(N));

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

    // loss = sum(w + b)  — gradient is 1 for both w and b (uniform)
    Term sum_wb = thvm_op(ctx, UOP_ADD, w_fwd, b_fwd);
    Term loss = thvm_sum_axes(ctx, sum_wb, (u32[]){0}, 1);

    Term params[] = {w_grad, b_grad};
    Term grads = thvm_grad_multi_keep(ctx, loss, params, 2);

    u64 l_gw = heap_alloc(ctx, 2);
    Term gw_var = term_new(TAG_VAR, 0, l_gw);
    heap_set(ctx, l_gw + 0, term_set_sub(gw_var));
    thvm_hint_shape(ctx, gw_var, SHAPE(N));

    u64 l_gb = heap_alloc(ctx, 2);
    Term gb_var = term_new(TAG_VAR, 0, l_gb);
    heap_set(ctx, l_gb + 0, term_set_sub(gb_var));
    thvm_hint_shape(ctx, gb_var, SHAPE(N));

    Term lr_vec = thvm_expand(ctx, lr, SHAPE(N));
    Term lr_w, lr_b;
    thvm_dup(ctx, thvm_fresh_label(ctx), lr_vec, &lr_w, &lr_b);

    Term update_w = thvm_op(ctx, UOP_SUB, w_sgd, thvm_op(ctx, UOP_MUL, lr_w, gw_var));
    Term update_b = thvm_op(ctx, UOP_SUB, b_sgd, thvm_op(ctx, UOP_MUL, lr_b, gb_var));
    Term assign_w = thvm_assign(ctx, w_assign, update_w);
    Term assign_b = thvm_assign(ctx, b_assign, update_b);

    Term rec = thvm_app(ctx,
               thvm_app(ctx,
               thvm_app(ctx, thvm_ref(ctx, train_id), next_counter),
                        w_next),
                        b_next);
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
                    w0),
                    b0);
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    u32 n_steps = 1;
    const char *env = getenv("THVM_TRAIN_STEPS");
    if (env && env[0]) n_steps = (u32)strtoul(env, NULL, 10);
    else if (argc > 1) n_steps = (u32)strtoul(argv[1], NULL, 10);

    TinyHVM *ctx = thvm_init("cpu");

    f32 wd[N] = { 0.5f, -1.0f, 0.25f };
    f32 bd[N] = { 0.1f,  0.2f, -0.3f };
    f32 lrd[] = { 0.1f };

    Term w = thvm_tensor(ctx, wd, SHAPE(N));
    Term b = thvm_tensor(ctx, bd, SHAPE(N));
    Term lr = thvm_tensor(ctx, lrd, SHAPE(1));
    thvm_set_requires_grad(ctx, w);
    thvm_set_requires_grad(ctx, b);

    Term program = recursive_train(ctx, w, b, lr, n_steps);
    Term result = thvm_eval(ctx, program);

    (void)result;
    u32 wdt = DTYPE_F32, bdt = DTYPE_F32;
    Shape ws = SHAPE(1), bs = SHAPE(1);
    f32 *wf = thvm_to_host_raw(ctx, w, &wdt, &ws);
    f32 *bf = thvm_to_host_raw(ctx, b, &bdt, &bs);
    if (!wf || !bf) { printf("not readable\n"); thvm_free(ctx); return 1; }

    // Expected: each step subtracts lr*1 from each component.
    // After n_steps: w_i -= n_steps * lr, b_i -= n_steps * lr.
    printf("n_steps=%u lr=%.4f\n", n_steps, lrd[0]);
    printf("final_w  = [%.4f, %.4f, %.4f]\n", wf[0], wf[1], wf[2]);
    printf("final_b  = [%.4f, %.4f, %.4f]\n", bf[0], bf[1], bf[2]);
    printf("expect_w = [%.4f, %.4f, %.4f]\n",
           wd[0] - n_steps*lrd[0], wd[1] - n_steps*lrd[0], wd[2] - n_steps*lrd[0]);
    printf("expect_b = [%.4f, %.4f, %.4f]\n",
           bd[0] - n_steps*lrd[0], bd[1] - n_steps*lrd[0], bd[2] - n_steps*lrd[0]);

    // Compute effective gradients (what the update implies)
    printf("effective g_w = [%.4f, %.4f, %.4f] (expect 1.0)\n",
           (wd[0]-wf[0])/(n_steps*lrd[0]),
           (wd[1]-wf[1])/(n_steps*lrd[0]),
           (wd[2]-wf[2])/(n_steps*lrd[0]));
    printf("effective g_b = [%.4f, %.4f, %.4f] (expect 1.0)\n",
           (bd[0]-bf[0])/(n_steps*lrd[0]),
           (bd[1]-bf[1])/(n_steps*lrd[0]),
           (bd[2]-bf[2])/(n_steps*lrd[0]));

    thvm_free(ctx);
    return 0;
}
