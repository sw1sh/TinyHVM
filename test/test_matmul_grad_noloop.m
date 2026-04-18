// test_matmul_grad_noloop.m — linear_sgd_loop's gradient check without
// the recursive loop or ASSIGN. If this PASSES, the bug is in the
// loop/ASSIGN machinery. If it FAILS, the bug is in the grad framework
// itself for this forward shape.

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

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    setbuf(stdout, NULL);
    TinyHVM *ctx = thvm_init("cpu");

    f32 xd[M*K]  = { 1.0f, 2.0f, -1.0f, 0.5f, -1.5f, 2.0f };
    f32 yd[M*N]  = { 0.6f, -0.2f, 0.1f, 0.9f, -0.3f, 0.7f, -0.5f, 0.2f };
    f32 wd[K*N]  = {
        0.10f, -0.20f,  0.30f,  0.40f,
       -0.50f,  0.60f, -0.70f,  0.80f,
        0.90f, -1.00f,  1.10f, -1.20f,
    };
    f32 bd[N]    = { 0.05f, -0.10f, 0.15f, 0.20f };

    Term x = thvm_tensor(ctx, xd, SHAPE(M, K));
    Term y = thvm_tensor(ctx, yd, SHAPE(M, N));
    Term w = thvm_tensor(ctx, wd, SHAPE(K, N));
    Term b = thvm_tensor(ctx, bd, SHAPE(N));
    thvm_set_requires_grad(ctx, w);
    thvm_set_requires_grad(ctx, b);

    // Exact same forward as test_tiny_linear_sgd_loop
    Term pred = thvm_op(ctx, UOP_ADD,
        thvm_mm(ctx, x, w),
        thvm_expand(ctx, thvm_reshape(ctx, b, SHAPE(1, N)), SHAPE(M, N)));
    Term diff = thvm_op(ctx, UOP_SUB, pred, y);
    Term sq = thvm_op(ctx, UOP_MUL, diff, diff);
    Term loss = thvm_sum_axes(ctx, sq, (u32[]){0, 1}, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1, 1));
    Term inv_n = thvm_reshape(ctx, thvm_scalar(ctx, 1.0f / 8.0f), SHAPE(1, 1));
    loss = thvm_op(ctx, UOP_MUL, loss, inv_n);

    Term params[] = { w, b };
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));
    u32 n = thvm_grad_bundle_count(ctx, bundle);
    printf("bundle count = %u\n", n);

    if (n != 2) { printf("FAIL: expected bundle of 2\n"); return 1; }

    Term gw_term = thvm_grad_bundle_get(ctx, bundle, 0);
    Term gb_term = thvm_grad_bundle_get(ctx, bundle, 1);
    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    f32 *gw = thvm_to_host_raw(ctx, gw_term, &dtype, &shp);
    f32 *gb = thvm_to_host_raw(ctx, gb_term, &dtype, &shp);

    // Hand-computed expected values
    // pred = x @ w + b, diff = pred - y
    // dL/d(pred) = (2/8) * diff = 0.25 * diff
    // gw[k, j] = sum_i (0.25 * diff[i, j] * x[i, k])
    // gb[j]   = sum_i (0.25 * diff[i, j])
    f32 pred_host[M*N];
    for (u32 i = 0; i < M; i++)
        for (u32 j = 0; j < N; j++) {
            f32 s = bd[j];
            for (u32 k = 0; k < K; k++) s += xd[i*K+k] * wd[k*N+j];
            pred_host[i*N + j] = s;
        }
    f32 diff_host[M*N];
    for (u32 i = 0; i < M*N; i++) diff_host[i] = pred_host[i] - yd[i];

    f32 exp_gw[K*N];
    for (u32 k = 0; k < K; k++)
        for (u32 j = 0; j < N; j++) {
            f32 s = 0.0f;
            for (u32 i = 0; i < M; i++)
                s += 0.25f * diff_host[i*N + j] * xd[i*K + k];
            exp_gw[k*N + j] = s;
        }
    f32 exp_gb[N];
    for (u32 j = 0; j < N; j++) {
        f32 s = 0.0f;
        for (u32 i = 0; i < M; i++) s += 0.25f * diff_host[i*N + j];
        exp_gb[j] = s;
    }

    printf("gw:       ");
    for (u32 p = 0; p < K*N; p++) printf("%8.4f ", gw[p]);
    printf("\nexpected: ");
    for (u32 p = 0; p < K*N; p++) printf("%8.4f ", exp_gw[p]);
    printf("\n");

    printf("gb:       ");
    for (u32 p = 0; p < N; p++) printf("%8.4f ", gb[p]);
    printf("\nexpected: ");
    for (u32 p = 0; p < N; p++) printf("%8.4f ", exp_gb[p]);
    printf("\n");

    int ok = 1;
    for (u32 p = 0; p < K*N; p++)
        if (fabsf(gw[p] - exp_gw[p]) > 1e-4f) { ok = 0; break; }
    for (u32 p = 0; p < N; p++)
        if (fabsf(gb[p] - exp_gb[p]) > 1e-4f) { ok = 0; break; }
    printf("%s\n", ok ? "PASS" : "FAIL");

    thvm_free(ctx);
    return ok ? 0 : 1;
}
