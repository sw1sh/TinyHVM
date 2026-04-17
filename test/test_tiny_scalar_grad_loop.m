// test_tiny_scalar_grad_loop.m — minimal functional recursive learning loop
//
// train(counter)(w) =
//   IFZ(counter,
//       w,
//       λm. train(m)(ASSIGN(w, w - lr * d/dw loss(w))))
//
// The whole loop is constructed first and evaluated once. Each unfold builds a
// fresh forward/backward/update subgraph; ASSIGN stays lazy until eval reaches
// the staged lowering/dispatch path.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static Term build_loss(TinyHVM *ctx, Term w, Term target) {
    Term diff = thvm_op(ctx, UOP_SUB, w, target);
    Term sq = thvm_op(ctx, UOP_MUL, diff, diff);
    return thvm_sum_axes(ctx, sq, (u32[]){0}, 1);
}

static Term recursive_train(TinyHVM *ctx, Term w0, Term target, Term lr, u32 n_steps) {
    u32 train_id = ctx->def_count++;
    assert(train_id < 256);

    u64 l_counter = heap_alloc(ctx, 2);
    Term counter_var = term_new(TAG_VAR, 0, l_counter);
    heap_set(ctx, l_counter + 0, term_set_sub(counter_var));

    u64 l_w = heap_alloc(ctx, 2);
    Term w_var = term_new(TAG_VAR, 0, l_w);
    heap_set(ctx, l_w + 0, term_set_sub(w_var));
    thvm_hint_shape(ctx, w_var, SHAPE(3));

    u64 l_next = heap_alloc(ctx, 2);
    Term next_counter = term_new(TAG_VAR, 0, l_next);
    heap_set(ctx, l_next + 0, term_set_sub(next_counter));

    Term w_fwd, w_grad, w_sgd, w_assign, w_done;
    {
        Term w1, w2, w3, w4;
        thvm_dup(ctx, thvm_fresh_label(ctx), w_var, &w_fwd, &w1);
        thvm_dup(ctx, thvm_fresh_label(ctx), w1, &w_grad, &w2);
        thvm_dup(ctx, thvm_fresh_label(ctx), w2, &w_sgd, &w3);
        thvm_dup(ctx, thvm_fresh_label(ctx), w3, &w_assign, &w_done);
    }

    Term loss = build_loss(ctx, w_fwd, target);
    Term grad = thvm_grad_keep(ctx, loss, w_grad);
    Term lr_vec = thvm_expand(ctx, lr, SHAPE(3));
    Term update = thvm_op(ctx, UOP_SUB, w_sgd, thvm_op(ctx, UOP_MUL, lr_vec, grad));
    Term assign = thvm_assign(ctx, w_assign, update);
    Term rec = thvm_app(ctx,
               thvm_app(ctx, thvm_ref(ctx, train_id), next_counter),
                        assign);

    heap_set(ctx, l_next + 1, rec);
    Term succ_lam = term_new(TAG_LAM, 0, l_next);
    heap_set(ctx, l_w + 1, thvm_ifz(ctx, counter_var, w_done, succ_lam));
    Term lam_w = term_new(TAG_LAM, 0, l_w);
    heap_set(ctx, l_counter + 1, lam_w);
    ctx->defs[train_id] = term_new(TAG_LAM, 0, l_counter);

    return thvm_app(ctx,
           thvm_app(ctx, thvm_ref(ctx, train_id), thvm_scalar_u32(ctx, n_steps)),
                    w0);
}

static void simulate_expected(f32 *w, const f32 *target, f32 lr, u32 n_steps, u32 n) {
    for (u32 step = 0; step < n_steps; step++) {
        for (u32 i = 0; i < n; i++) {
            f32 gw = 2.0f * (w[i] - target[i]);
            w[i] -= lr * gw;
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

    f32 wd[] = {0.5f, -1.0f, 0.25f};
    f32 td[] = {2.0f, 4.0f, -3.0f};
    f32 lrd[] = {0.1f};

    Term w = thvm_tensor(ctx, wd, SHAPE(3));
    Term target = thvm_tensor(ctx, td, SHAPE(3));
    Term lr = thvm_tensor(ctx, lrd, SHAPE(1));
    thvm_set_requires_grad(ctx, w);

    printf("initial_w = [%.4f, %.4f, %.4f]\n", wd[0], wd[1], wd[2]);
    printf("target    = [%.4f, %.4f, %.4f]\n", td[0], td[1], td[2]);
    printf("lr = [%.4f]\n", lrd[0]);

    Term program = recursive_train(ctx, w, target, lr, n_steps);
    Term result = thvm_eval(ctx, program);

    printf("result tag=%u ext=%u val=%llu raw=0x%016llx\n",
           (u32)term_tag(result), (u32)term_ext(result),
           (unsigned long long)term_val(result),
           (unsigned long long)result);

    f32 expected[] = {wd[0], wd[1], wd[2]};
    simulate_expected(expected, td, lrd[0], n_steps, 3);

    u32 dtype = DTYPE_F32;
    Shape sh = SHAPE(1);
    f32 *wf = thvm_to_host_raw(ctx, w, &dtype, &sh);
    assert(wf && "w must be readable");

    u32 rdtype = DTYPE_F32;
    Shape rsh = SHAPE(1);
    f32 *rf = thvm_to_host_raw(ctx, result, &rdtype, &rsh);
    assert(rf && "result must be readable");

    printf("final_w = [%.4f, %.4f, %.4f]\n", wf[0], wf[1], wf[2]);
    printf("result  = [%.4f, %.4f, %.4f]\n", rf[0], rf[1], rf[2]);
    printf("expect  = [%.4f, %.4f, %.4f]\n", expected[0], expected[1], expected[2]);
    for (u32 i = 0; i < 3; i++) {
        assert(fabsf(wf[i] - expected[i]) < 1e-4f);
        assert(fabsf(rf[i] - expected[i]) < 1e-4f);
    }

    thvm_free(ctx);
    return 0;
}
