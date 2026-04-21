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
    Term counter_var = 0;
    Term w_var = 0;
    Term next_counter = 0;

    // Build binders explicitly via thvm_lam, then fill their bodies.
    Term lam_counter = thvm_lam(ctx, &counter_var, term_new(TAG_ERA, 0, 0));
    Term lam_w       = thvm_lam(ctx, &w_var,       term_new(TAG_ERA, 0, 0));
    Term succ_lam    = thvm_lam(ctx, &next_counter,term_new(TAG_ERA, 0, 0));
    thvm_hint_shape(ctx, w_var, SHAPE(3));

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
    heap_set(ctx, term_val(succ_lam) + 1, seq_body);

    // IFZ(counter, w, succ_lam)
    heap_set(ctx, term_val(lam_w) + 1, thvm_ifz(ctx, counter_var, w4, succ_lam));
    heap_set(ctx, term_val(lam_counter) + 1, lam_w);
    ctx->defs[train_id] = lam_counter;

    return thvm_app(ctx,
           thvm_app(ctx, thvm_ref(ctx, train_id), thvm_scalar_u32(ctx, n_steps)),
                    w0);
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    u32 n_steps = 3;
    int allow_non_ten_result = 0;
    const char *env = getenv("THVM_TRAIN_STEPS");
    if (env && env[0]) n_steps = (u32)strtoul(env, NULL, 10);
    else if (argc > 1) n_steps = (u32)strtoul(argv[1], NULL, 10);
    env = getenv("THVM_ALLOW_NON_TEN_RESULT");
    if (env && env[0] && strcmp(env, "0") != 0) allow_non_ten_result = 1;

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
    assert(wf && "w buffer must be readable");
    f32 scale = (f32)(1 << n_steps);
    printf("final_w = [%.1f, %.1f, %.1f]  (expect [%.1f, %.1f, %.1f])\n",
           wf[0], wf[1], wf[2],
           wd[0] * scale, wd[1] * scale, wd[2] * scale);
    assert(wf[0] == wd[0] * scale && wf[1] == wd[1] * scale && wf[2] == wd[2] * scale
           && "w values must match expected");

    // Normal eval should return TAG_TEN. Diagnostic coarse-graph runs can opt
    // out so they can inspect the settled coarse root while still validating
    // the buffer-side effects.
    if (!allow_non_ten_result)
        assert(term_tag(result) == TAG_TEN && "result must reduce to TEN");

    // Also read from result term
    u32 dtype2 = DTYPE_F32;
    Shape sh2 = SHAPE(1);
    f32 *rf = thvm_to_host_raw(ctx, result, &dtype2, &sh2);
    if (rf)
        printf("result_w = [%.1f, %.1f, %.1f]\n", rf[0], rf[1], rf[2]);

    thvm_free(ctx);
    return 0;
}
