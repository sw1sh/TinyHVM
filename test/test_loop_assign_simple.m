// test_loop_assign_simple.m — minimal loop: ASSIGN(w, w*2) N times
// Uses SEQ to sequence ASSIGN before the recursive call.
//
// train := λcounter. λw.
//   IFZ(counter, w,
//     λm. SEQ(ASSIGN(w, w*2), train(m)(w)))
//
// SEQ forces ASSIGN to fire (TAG_TEN) before the continuation.
// train(m)(w) passes the same tensor w — buffer updated by ASSIGN.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <stdlib.h>

static Term simple_loop(TinyHVM *ctx, Term w0, u32 n_steps) {
    u32 train_id = ctx->def_count++;
    assert(train_id < 256);

    // train := λcounter. λw. IFZ(counter, w, λm. SEQ(ASSIGN(w, w*2), train(m)(w)))
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

    // DUP w for 4 uses: compute_src, assign_dst, recurse_param, base_case
    Term w1, w2, w3, w4;
    {
        Term wa, wb;
        thvm_dup(ctx, thvm_fresh_label(ctx), w_var, &wa, &wb);
        thvm_dup(ctx, thvm_fresh_label(ctx), wa, &w1, &w2);
        thvm_dup(ctx, thvm_fresh_label(ctx), wb, &w3, &w4);
    }
    // w1=compute_src (w*2), w2=assign_dst, w3=recurse_param, w4=base_case

    // w * 2
    Term two = thvm_expand(ctx, thvm_scalar(ctx, 2.0f), SHAPE(3));
    Term doubled = thvm_op(ctx, UOP_MUL, w1, two);

    // ASSIGN(w, w*2)
    Term assign = thvm_assign(ctx, w2, doubled);

    // Recursive call: train(m)(w) — same w, buffer updated by ASSIGN
    Term rec = thvm_app(ctx,
               thvm_app(ctx, thvm_ref(ctx, train_id), next_counter),
                        w3);

    // SEQ(ASSIGN(w, w*2), train(m)(w))
    // Forces ASSIGN to fire before the recursive call.
    Term seq_body = thvm_seq(ctx, assign, rec);

    heap_set(ctx, l_next + 1, seq_body);
    Term succ_lam = term_new(TAG_LAM, 0, l_next);

    // IFZ(counter, w, succ_lam)
    heap_set(ctx, l_w + 1, thvm_ifz(ctx, counter_var, w4, succ_lam));
    Term lam_w = term_new(TAG_LAM, 0, l_w);
    heap_set(ctx, l_counter + 1, lam_w);
    ctx->defs[train_id] = term_new(TAG_LAM, 0, l_counter);

    return thvm_app(ctx,
           thvm_app(ctx, thvm_ref(ctx, train_id), thvm_scalar_u32(ctx, n_steps)),
                    w0);
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    u32 n_steps = 3;
    const char *env = getenv("THVM_TRAIN_STEPS");
    if (env && env[0]) n_steps = (u32)strtoul(env, NULL, 10);
    else if (argc > 1) n_steps = (u32)strtoul(argv[1], NULL, 10);

    TinyHVM *ctx = thvm_init("cpu");

    f32 wd[] = {1.0f, 2.0f, 3.0f};
    Term w = thvm_tensor(ctx, wd, (Shape){.dims={3}, .rank=1});

    printf("initial_w = [%.1f, %.1f, %.1f]\n", wd[0], wd[1], wd[2]);

    Term program = simple_loop(ctx, w, n_steps);
    Term result = thvm_eval(ctx, program);

    printf("result tag=%u ext=%u val=%llu raw=0x%016llx\n",
           (u32)term_tag(result), (u32)term_ext(result),
           (unsigned long long)term_val(result),
           (unsigned long long)result);

    // Read w's buffer directly
    u32 dtype = DTYPE_F32;
    Shape sh = SHAPE(1);
    f32 *wf = thvm_to_host_raw(ctx, w, &dtype, &sh);
    if (wf)
        printf("final_w = [%.1f, %.1f, %.1f]  (expect [%.1f, %.1f, %.1f])\n",
               wf[0], wf[1], wf[2],
               wd[0] * (1 << n_steps), wd[1] * (1 << n_steps), wd[2] * (1 << n_steps));

    // Also read from result term
    u32 dtype2 = DTYPE_F32;
    Shape sh2 = SHAPE(1);
    f32 *rf = thvm_to_host_raw(ctx, result, &dtype2, &sh2);
    if (rf)
        printf("result_w = [%.1f, %.1f, %.1f]\n", rf[0], rf[1], rf[2]);

    thvm_free(ctx);
    return 0;
}
