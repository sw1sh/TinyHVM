// test_dup_chain.m — stress the HVM4-style DUP rewrite with a 5-deep
// chain over a tensor, mixing live and ERA'd auxes. Mirrors the DUP
// chain shape in test_tiny_twoparam_grad_loop but without SGD/ASSIGN.
//
// w_var → dup → (p0, w1) → dup → (p1, w2) → dup → (p2, w3)
//       → dup → (p3, w4) → dup → (p4, p5)
//
// Live arms: p0, p2, p4. Sum(live arms) = 3 * sum(w). Grad_w = 3 * ones.
// ERA arms: p1, p3, p5. The ERAs must not leak through and kill the chain.
//
// Under the HVM4-style SUB-bit subst with explicit ERA sweeps,
// each ERA'd aux should either (a) mark its DUP cell as identity-collapse
// for the live sibling, or (b) spawn a visible detached ERA on the
// orphan clone stored by a sibling that fired first.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *backend = (argc > 1) ? argv[1] : "cpu";
    TinyHVM *ctx = thvm_init(backend);

    const u32 N = 4;
    f32 wd[4] = { 1.0f, -2.0f, 3.5f, -0.5f };
    Term w = thvm_tensor(ctx, wd, SHAPE(N));
    thvm_set_requires_grad(ctx, w);

    // Same chain shape as test_tiny_twoparam_grad_loop: each DUP splits
    // into (consumer_in_this_frame, deeper_into_chain). The consumers in
    // the "ERA'd branch" are bundled into a CTR that gets erased — this
    // matches the IFZ-losing-branch pattern.
    Term p_fwd, w1, p_grad, w2, p_sgd, w3, p_assign, w4, p_next, p_done;
    thvm_dup(ctx, thvm_fresh_label(ctx), w,  &p_fwd,    &w1);
    thvm_dup(ctx, thvm_fresh_label(ctx), w1, &p_grad,   &w2);
    thvm_dup(ctx, thvm_fresh_label(ctx), w2, &p_sgd,    &w3);
    thvm_dup(ctx, thvm_fresh_label(ctx), w3, &p_assign, &w4);
    thvm_dup(ctx, thvm_fresh_label(ctx), w4, &p_next,   &p_done);

    // Live side (4 consumers of the w chain): sum(p_fwd) + sum(p_grad) +
    // sum(p_sgd) + sum(p_assign) + sum(p_next) = 5 * sum(w).
    Term s_fwd    = thvm_sum_axes(ctx, p_fwd,    (u32[]){0}, 1);
    Term s_grad   = thvm_sum_axes(ctx, p_grad,   (u32[]){0}, 1);
    Term s_sgd    = thvm_sum_axes(ctx, p_sgd,    (u32[]){0}, 1);
    Term s_assign = thvm_sum_axes(ctx, p_assign, (u32[]){0}, 1);
    Term s_next   = thvm_sum_axes(ctx, p_next,   (u32[]){0}, 1);
    Term loss = thvm_op(ctx, UOP_ADD, s_fwd,
                thvm_op(ctx, UOP_ADD, s_grad,
                thvm_op(ctx, UOP_ADD, s_sgd,
                thvm_op(ctx, UOP_ADD, s_assign, s_next))));

    // Dead branch: wrap p_done inside a CTR then erase the CTR. Matches
    // IFZ-done-branch erasure in twoparam.
    Term done_ctr = thvm_ctr(ctx, (Term[]){p_done}, 1);
    thvm_spawn_detached_era(ctx, done_ctr);

    f32 *gw = thvm_to_host(ctx, thvm_eval(ctx, thvm_grad(ctx, loss, w)));
    if (!gw) { fprintf(stderr, "gw ptr NULL\n"); thvm_free(ctx); return 2; }

    // Expected: 5 live arms, each sum(w), so grad = 5 * [1,1,1,1].
    f32 diff = 0;
    for (u32 i = 0; i < N; i++) diff += fabsf(gw[i] - 5.0f);
    printf("backend=%s  L1 diff: %.6f  %s\n", backend, diff,
        diff < 1e-4f ? "PASS" : "FAIL (DUP chain grad drop)");
    if (diff >= 1e-4f) {
        printf("  got:      ");
        for (u32 i = 0; i < N; i++) printf("%.4f%s", gw[i], i+1==N ? "\n" : ", ");
        printf("  expected: 5.0000, 5.0000, 5.0000, 5.0000\n");
    }
    thvm_free(ctx);
    return diff < 1e-4f ? 0 : 1;
}
