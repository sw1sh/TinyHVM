// test_mm_dup_2arm.m — minimal MM+DUP+SUM+SUM test (MNIST pattern).
//
//   logits = MM(x, W)
//   DUP logits into (l0, l1)
//   loss = sum(l0) + sum(l1)
//
// Both arms live, both fire the DUP normally. Tests that HVM4-style
// subst_cop produces correct (r0, r1) pairs and stores the sibling's
// clone correctly in the DUP cell for second-firer pickup.
//
// Expected grad_W = 2 * x.T @ ones(B,C) = 2 * sum_i x[i,:] (broadcast).

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
    u32 B = 2, in = 2, C = 3;
    f32 xd[] = { 1.0f, 0.5f,   0.1f, -1.0f };
    f32 wd[] = { 0.1f, 0.2f, 0.3f, 0.05f, 0.1f, -0.2f };
    Term x = thvm_tensor(ctx, xd, SHAPE(B, in));
    Term W = thvm_tensor(ctx, wd, SHAPE(in, C));
    thvm_set_requires_grad(ctx, W);

    Term logits = thvm_mm(ctx, x, W);
    Term l0, l1;
    thvm_dup(ctx, thvm_fresh_label(ctx), logits, &l0, &l1);
    Term s0 = thvm_sum_axes(ctx, l0, (u32[]){0, 1}, 2);
    Term s1 = thvm_sum_axes(ctx, l1, (u32[]){0, 1}, 2);
    Term loss = thvm_op(ctx, UOP_ADD, s0, s1);

    Term params[] = { W };
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 1));
    f32 *dw = thvm_to_host(ctx, bundle);

    // Expected grad_W[k,j] = 2 * sum_i x[i,k]
    f32 exp_dw[6] = {0};
    for (u32 k = 0; k < in; k++) {
        f32 col = 0;
        for (u32 i = 0; i < B; i++) col += xd[i*in + k];
        for (u32 j = 0; j < C; j++) exp_dw[k*C + j] = 2.0f * col;
    }

    f32 diff = 0;
    for (u32 i = 0; i < in*C; i++) diff += fabsf(dw[i] - exp_dw[i]);
    printf("backend=%s  L1 diff: %.6f  %s\n", backend, diff,
        diff < 1e-3f ? "PASS" : "FAIL (MM+DUP grad drop)");
    if (diff >= 1e-3f) {
        printf("  got:      ");
        for (u32 i = 0; i < in*C; i++) printf("%.4f%s", dw[i], i+1==in*C ? "\n" : ", ");
        printf("  expected: ");
        for (u32 i = 0; i < in*C; i++) printf("%.4f%s", exp_dw[i], i+1==in*C ? "\n" : ", ");
    }
    thvm_free(ctx);
    return diff < 1e-3f ? 0 : 1;
}
