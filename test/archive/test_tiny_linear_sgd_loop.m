// test_tiny_linear_sgd_loop.m — functional recursive SGD with ASSIGN-based param updates
//
// train(counter)(w)(b) =
//   IFZ(counter, CTR(w,b),
//     λm. grads = GRAD_KEEP(loss(w,b), [w,b])
//          MAT grads λgw. λgb.
//            train(m)(ASSIGN(w, w-lr*gw))(ASSIGN(b, b-lr*gb)))
//
// w and b are lambda parameters threaded through each iteration.
// ASSIGN wraps the update: it's TAG_TOP (WNF) in phase 1, fires in phase 3.
// ASSIGN returns dst tensor (same tid) → GRAD targets always match.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void split3(TinyHVM *ctx, Term t, Term *a, Term *b, Term *c) {
    Term tail;
    thvm_dup(ctx, thvm_fresh_label(ctx), t, a, &tail);
    thvm_dup(ctx, thvm_fresh_label(ctx), tail, b, c);
}

static Term tiny_linear_sgd_loop(TinyHVM *ctx,
                                 Term w0, Term b0,
                                 Term x, Term y,
                                 Term lr,
                                 u32 n_steps) {
    u32 train_id = ctx->def_count++;
    assert(train_id < 256);

    // train := λcounter. λw. λb. IFZ(counter, CTR(w,b), succ)
    u64 l_counter = heap_alloc(ctx, 2);
    Term counter_var = term_new(TAG_VAR, 0, l_counter);
    heap_set(ctx, l_counter + 0, term_set_sub(counter_var));

    u64 l_w = heap_alloc(ctx, 2);
    Term w_var = term_new(TAG_VAR, 0, l_w);
    heap_set(ctx, l_w + 0, term_set_sub(w_var));
    thvm_hint_shape(ctx, w_var, SHAPE(3, 4));

    u64 l_b = heap_alloc(ctx, 2);
    Term b_var = term_new(TAG_VAR, 0, l_b);
    heap_set(ctx, l_b + 0, term_set_sub(b_var));
    thvm_hint_shape(ctx, b_var, SHAPE(4));

    u64 l_next = heap_alloc(ctx, 2);
    Term next_counter = term_new(TAG_VAR, 0, l_next);
    heap_set(ctx, l_next + 0, term_set_sub(next_counter));

    // DUP w and b for multiple uses in body:
    // forward, grad_target, sgd_update, assign_dst, recurse_param, base_case
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

    // Forward: pred = x @ w + b
    Term pred = thvm_op(ctx, UOP_ADD,
        thvm_mm(ctx, x, w_fwd),
        thvm_expand(ctx, thvm_reshape(ctx, b_fwd, SHAPE(1, 4)), SHAPE(2, 4)));

    // Mean squared error
    Term diff = thvm_op(ctx, UOP_SUB, pred, y);
    Term sq = thvm_op(ctx, UOP_MUL, diff, diff);
    Term loss = thvm_sum_axes(ctx, sq, (u32[]){0, 1}, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1, 1));
    Term inv_n = thvm_reshape(ctx, thvm_scalar(ctx, 1.0f / 8.0f), SHAPE(1, 1));
    loss = thvm_op(ctx, UOP_MUL, loss, inv_n);

    // Backward: GRAD_KEEP(loss, [w, b]) → CTR(gw, gb)
    Term params[] = {w_grad, b_grad};
    Term grads = thvm_grad_multi_keep(ctx, loss, params, 2);

    // MAT destructs CTR(gw, gb) → apply λgw. λgb. body
    u64 l_gw = heap_alloc(ctx, 2);
    Term gw_var = term_new(TAG_VAR, 0, l_gw);
    heap_set(ctx, l_gw + 0, term_set_sub(gw_var));
    thvm_hint_shape(ctx, gw_var, SHAPE(3, 4));

    u64 l_gb = heap_alloc(ctx, 2);
    Term gb_var = term_new(TAG_VAR, 0, l_gb);
    heap_set(ctx, l_gb + 0, term_set_sub(gb_var));
    thvm_hint_shape(ctx, gb_var, SHAPE(4));

    // SGD: ASSIGN(w, w - lr*gw), ASSIGN(b, b - lr*gb)
    Term lr_w = thvm_expand(ctx, thvm_reshape(ctx, lr, SHAPE(1, 1)), SHAPE(3, 4));
    Term lr_b = thvm_expand(ctx, lr, SHAPE(4));
    Term update_w = thvm_op(ctx, UOP_SUB, w_sgd, thvm_op(ctx, UOP_MUL, lr_w, gw_var));
    Term update_b = thvm_op(ctx, UOP_SUB, b_sgd, thvm_op(ctx, UOP_MUL, lr_b, gb_var));
    Term assign_w = thvm_assign(ctx, w_assign, update_w);
    Term assign_b = thvm_assign(ctx, b_assign, update_b);

    // Recursive call threads w_tail / b_tail as the next iteration's
    // w/b. The SEQ chain below forces both ASSIGNs to fire *before*
    // the recursion reads w/b, so each step's update is observed by
    // the next step — standard HVM4 FFI-style ordering. See
    // resources/hvm4_ffi_design.md.
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

    Term grad_bundle_match = thvm_mat(ctx, 2, lam_gw, term_era());
    Term succ_body = thvm_app(ctx, grad_bundle_match, grads);
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
    const char *backend = getenv("THVM_TEST_BACKEND");
    if (!backend || !backend[0]) backend = argc > 1 ? argv[1] : "metal";
    u32 n_steps = 2;
    const char *steps_env = getenv("THVM_TRAIN_STEPS");
    if (steps_env && steps_env[0]) n_steps = (u32)strtoul(steps_env, NULL, 10);
    else if (argc > 2) n_steps = (u32)strtoul(argv[2], NULL, 10);
#ifdef __APPLE__
    if (strcmp(backend, "metal") == 0 && !MTLCreateSystemDefaultDevice()) {
        fprintf(stderr, "SKIP: metal unavailable\n");
        return 0;
    }
#endif
    TinyHVM *ctx = thvm_init(backend);

    f32 xd[] = { 1.0f, 2.0f, -1.0f, 0.5f, -1.5f, 2.0f };
    f32 yd[] = { 0.6f, -0.2f, 0.1f, 0.9f, -0.3f, 0.7f, -0.5f, 0.2f };
    f32 wd[] = {
        0.10f, -0.20f,  0.30f,  0.40f,
       -0.50f,  0.60f, -0.70f,  0.80f,
        0.90f, -1.00f,  1.10f, -1.20f,
    };
    f32 bd[] = {0.05f, -0.10f, 0.15f, 0.20f};

    Term x  = thvm_tensor(ctx, xd,  (Shape){.dims={2,3}, .rank=2});
    Term y  = thvm_tensor(ctx, yd,  (Shape){.dims={2,4}, .rank=2});
    Term w  = thvm_tensor(ctx, wd,  (Shape){.dims={3,4}, .rank=2});
    Term b0 = thvm_tensor(ctx, bd,  (Shape){.dims={4},   .rank=1});
    Term lr = thvm_scalar(ctx, 0.05f);

    thvm_set_requires_grad(ctx, w);
    thvm_set_requires_grad(ctx, b0);

    Term program = tiny_linear_sgd_loop(ctx, w, b0, x, y, lr, n_steps);
    Term result = thvm_eval(ctx, program);

    printf("result_tag=%u ext=%u val=%llu\n",
           (u32)term_tag(result), (u32)term_ext(result), (unsigned long long)term_val(result));

    // Extract from CTR bundle
    if (term_tag(result) == TAG_CTR) {
        Term final_w = thvm_grad_bundle_get(ctx, result, 0);
        Term final_b = thvm_grad_bundle_get(ctx, result, 1);
        u32 wd2 = DTYPE_F32, bd2 = DTYPE_F32;
        Shape ws = SHAPE(1), bsh = SHAPE(1);
        f32 *wf = thvm_to_host_raw(ctx, final_w, &wd2, &ws);
        f32 *bf = thvm_to_host_raw(ctx, final_b, &bd2, &bsh);
        if (wf) {
            printf("final_w = [");
            for (int i = 0; i < 12; i++) printf("%.4f%s", wf[i], i == 11 ? "" : ", ");
            printf("]\n");
        }
        if (bf)
            printf("final_b = [%.4f, %.4f, %.4f, %.4f]\n", bf[0], bf[1], bf[2], bf[3]);
    }

    thvm_free(ctx);
    return 0;
}
