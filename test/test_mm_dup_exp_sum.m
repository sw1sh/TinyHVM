// test_mm_dup_exp_sum.m — regression test for DUP-after-reduction grad drop.
//
// Bug: when `logits = MM(x, W)` is DUPed into two arms, and at least one
// arm goes through a non-trivial chain before loss (e.g., EXP → SUM), the
// backward through MM drops one arm's gradient contribution — ending up
// with only one arm reaching W.
//
// Diagnosis: `BUNDLE_ACCUM_ADD` log shows both grads reaching W's slot
// (as an UOP_ADD of two TOPs), but the final evaluated value only
// reflects one arm. Suspect interaction between MM's EXPAND+MUL+SUM
// decomposition and the bundle-add evaluation during keep-mode reduction.
//
// Verified: identical two-arm DUP + both arms `sum(l_k)` works (3× col_sum
// as expected) — only the SUM-reduced logits case fails.
//
// This bug blocks MNIST linear training — W updates freeze to ~0 while
// B trains correctly via a clean ADD+EXPAND backward path. See
// feedback_mm_dup_grad_drop memory.
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <math.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *backend = (argc > 1) ? argv[1] : "cpu";
    TinyHVM *ctx = thvm_init(backend);
    u32 B = 2, in = 2, C = 3;
    f32 xd[] = { 1.0f, 0.5f,   0.1f, -1.0f };
    f32 wd[] = {
        0.1f, 0.2f, 0.3f,
        0.05f, 0.1f, -0.2f,
    };
    Term x = thvm_tensor(ctx, xd, SHAPE(B, in));
    Term W = thvm_tensor(ctx, wd, SHAPE(in, C));
    thvm_set_requires_grad(ctx, W);

    Term logits = thvm_mm(ctx, x, W);
    Term l1, l2;
    thvm_dup(ctx, thvm_fresh_label(ctx), logits, &l1, &l2);
    // Both arms: sum(exp(l_k)). Loss = 2 * sum(exp(logits)).
    Term e1 = thvm_op(ctx, UOP_EXP, l1, term_era());
    Term e2 = thvm_op(ctx, UOP_EXP, l2, term_era());
    Term s1 = thvm_sum_axes(ctx, e1, (u32[]){0,1}, 2);
    Term s2 = thvm_sum_axes(ctx, e2, (u32[]){0,1}, 2);
    Term loss = thvm_op(ctx, UOP_ADD, s1, s2);

    f32 *dw = thvm_to_host(ctx, thvm_eval(ctx, thvm_grad(ctx, loss, W)));
    if (!dw) { fprintf(stderr, "dw ptr NULL (KERNEL not lowered)\n"); thvm_free(ctx); return 2; }

    // Expected: grad_W = 2 * x.T @ exp(logits).
    f32 lg[6];
    for (u32 i = 0; i < B; i++)
        for (u32 j = 0; j < C; j++) {
            f32 v = 0;
            for (u32 k = 0; k < in; k++) v += xd[i*in+k] * wd[k*C+j];
            lg[i*C+j] = v;
        }
    f32 exp_dw[6], diff = 0;
    for (u32 k = 0; k < in; k++)
        for (u32 j = 0; j < C; j++) {
            f32 v = 0;
            for (u32 i = 0; i < B; i++) v += xd[i*in+k] * expf(lg[i*C+j]);
            exp_dw[k*C+j] = 2.0f * v;
            diff += fabsf(dw[k*C+j] - exp_dw[k*C+j]);
        }
    printf("backend=%s  L1 diff: %.6f  %s\n", backend, diff,
        diff < 0.01f ? "PASS" : "FAIL (known DUP-after-reduction grad drop)");
    if (diff >= 0.01f) {
        printf("  got:      ");
        for (u32 i = 0; i < in*C; i++) printf("%.4f%s", dw[i], i+1==in*C ? "\n" : ", ");
        printf("  expected: ");
        for (u32 i = 0; i < in*C; i++) printf("%.4f%s", exp_dw[i], i+1==in*C ? "\n" : ", ");
    }
    thvm_free(ctx);
    return diff < 0.01f ? 0 : 1;
}
