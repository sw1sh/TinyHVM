// test_tiny_twoparam_grad_loop.m — two-parameter elementwise SGD loop
//
// train(counter)(w)(b) =
//   IFZ(counter,
//       CTR(w, b),
//       λm. let (gw, gb) = GRAD_KEEP(loss(w,b), [w, b])
//           in train(m)(ASSIGN(w, w - lr*gw))(ASSIGN(b, b - lr*gb)))
//
// loss(w, b) = sum(((w + b) - target)^2)
//
// Pure elementwise, no matmul. Exercises:
//   - multi-target GRAD_KEEP in a loop
//   - MAT destructuring of the (gw, gb) bundle
//   - two independent ASSIGNs in SEQ per step
//
// Host-simulates the SGD trajectory and asserts exact match at n=1,2,3.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 3

static Term build_loss(TinyHVM *ctx, Term w, Term b, Term target) {
    Term sum_wb = thvm_op(ctx, UOP_ADD, w, b);
    Term diff = thvm_op(ctx, UOP_SUB, sum_wb, target);
    Term sq = thvm_op(ctx, UOP_MUL, diff, diff);
    return thvm_sum_axes(ctx, sq, (u32[]){0}, 1);
}

static Term recursive_train(TinyHVM *ctx, Term w0, Term b0, Term target, Term lr, u32 n_steps) {
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

    // 4 uses of w: fwd, grad_target, sgd_subtract, assign_dst, next/done (5 total)
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

    Term loss = build_loss(ctx, w_fwd, b_fwd, target);
    Term params[] = {w_grad, b_grad};
    Term grads = thvm_grad_bundle(ctx, loss, params, 2);

    u64 l_gw = heap_alloc(ctx, 2);
    Term gw_var = term_new(TAG_VAR, 0, l_gw);
    heap_set(ctx, l_gw + 0, term_set_sub(gw_var));
    thvm_hint_shape(ctx, gw_var, SHAPE(N));

    u64 l_gb = heap_alloc(ctx, 2);
    Term gb_var = term_new(TAG_VAR, 0, l_gb);
    heap_set(ctx, l_gb + 0, term_set_sub(gb_var));
    thvm_hint_shape(ctx, gb_var, SHAPE(N));

    Term lr_vec = thvm_expand(ctx, lr, SHAPE(N));
    // DUP lr_vec twice since both updates multiply by lr
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

static void simulate(f32 *w, f32 *b, const f32 *target, f32 lr, u32 n_steps) {
    for (u32 step = 0; step < n_steps; step++) {
        f32 diff[N];
        for (u32 i = 0; i < N; i++) diff[i] = (w[i] + b[i]) - target[i];
        for (u32 i = 0; i < N; i++) {
            f32 g = 2.0f * diff[i];
            w[i] -= lr * g;
            b[i] -= lr * g;
        }
    }
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    u32 n_steps = 3;
    const char *env = getenv("THVM_TRAIN_STEPS");
    if (env && env[0]) n_steps = (u32)strtoul(env, NULL, 10);
    else if (argc > 1) n_steps = (u32)strtoul(argv[1], NULL, 10);

    TinyHVM *ctx = thvm_init("cpu");

    f32 wd[N] = { 0.5f, -1.0f, 0.25f };
    f32 bd[N] = { 0.1f,  0.2f, -0.3f };
    f32 td[N] = { 2.0f,  4.0f, -3.0f };
    f32 lrd[] = { 0.1f };

    Term w = thvm_tensor(ctx, wd, SHAPE(N));
    Term b = thvm_tensor(ctx, bd, SHAPE(N));
    Term target = thvm_tensor(ctx, td, SHAPE(N));
    Term lr = thvm_tensor(ctx, lrd, SHAPE(1));
    thvm_set_requires_grad(ctx, w);
    thvm_set_requires_grad(ctx, b);

    printf("initial_w = [%.4f, %.4f, %.4f]\n", wd[0], wd[1], wd[2]);
    printf("initial_b = [%.4f, %.4f, %.4f]\n", bd[0], bd[1], bd[2]);
    printf("target    = [%.4f, %.4f, %.4f]\n", td[0], td[1], td[2]);
    printf("lr = [%.4f]\n", lrd[0]);
    printf("n_steps = %u\n", n_steps);

    Term program = recursive_train(ctx, w, b, target, lr, n_steps);
    Term result = thvm_eval(ctx, program);

    printf("result tag=%u ext=%u val=%llu\n",
           (u32)term_tag(result), (u32)term_ext(result),
           (unsigned long long)term_val(result));

    f32 ew[N] = { wd[0], wd[1], wd[2] };
    f32 eb[N] = { bd[0], bd[1], bd[2] };
    simulate(ew, eb, td, lrd[0], n_steps);

    u32 wdt = DTYPE_F32, bdt = DTYPE_F32;
    Shape ws = SHAPE(1), bs = SHAPE(1);
    f32 *wf = thvm_to_host_raw(ctx, w, &wdt, &ws);
    f32 *bf = thvm_to_host_raw(ctx, b, &bdt, &bs);
    assert(wf && "w must be readable");
    assert(bf && "b must be readable");

    printf("final_w = [%.4f, %.4f, %.4f]\n", wf[0], wf[1], wf[2]);
    printf("final_b = [%.4f, %.4f, %.4f]\n", bf[0], bf[1], bf[2]);
    printf("expect_w= [%.4f, %.4f, %.4f]\n", ew[0], ew[1], ew[2]);
    printf("expect_b= [%.4f, %.4f, %.4f]\n", eb[0], eb[1], eb[2]);
    int all_match = 1;
    for (u32 i = 0; i < N; i++) {
        if (fabsf(wf[i] - ew[i]) >= 1e-4f) all_match = 0;
        if (fabsf(bf[i] - eb[i]) >= 1e-4f) all_match = 0;
    }
    assert(all_match && "multi-parameter GRAD in loop must match host SGD");
    printf("PASS\n");

    thvm_free(ctx);
    return 0;
}
