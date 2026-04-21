// test_matmul_backward_compute.m — pure-compute reproducer of the
// matmul-in-loop backward bug. No loops, no grad, no ASSIGN.
//
// Reconstructs the exact term structure the backward chain produces
// for `gw = dL/dw` in linear SGD:
//
//   gw = reshape(
//          sum_axes(
//            MUL(EXPAND(RESHAPE(diff_scaled, [M,1,N]), [M,K,N]),
//                EXPAND(RESHAPE(x,           [M,K,1]), [M,K,N])),
//            {0}
//          ),
//          [K,N]
//        )
//
// Where diff_scaled = diff / 8 (the 2*diff/8 = diff/4 split across the
// MUL(diff,diff) BG's two branches — each branch contributes diff/8).
//
// Ground truth: gw[k,j] = sum_i (diff[i,j]/8 * x[i,k])
//
// If the test prints per-element correct values, the backward bug is
// somewhere in IC setup (target registration / DUP routing / clone).
// If the test prints uniform-per-row values, the bug is in the forward
// kernel compilation / view composition itself.

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

    // Same initial tensors as test_tiny_linear_sgd_loop so the expected
    // values match the same hand calculation.
    f32 xd[M*K]   = { 1.0f, 2.0f, -1.0f, 0.5f, -1.5f, 2.0f };
    // diff_scaled = diff / 8, where diff values are from step-1 of the
    // loop: diff = pred - y with the initial weights.
    f32 diff_over_8[M*N] = {
        -2.35f / 8.0f,  2.10f / 8.0f, -2.15f / 8.0f,  2.50f / 8.0f,
         2.95f / 8.0f, -3.80f / 8.0f,  4.05f / 8.0f, -3.40f / 8.0f,
    };

    Term x    = thvm_tensor(ctx, xd,          SHAPE(M, K));
    Term diff = thvm_tensor(ctx, diff_over_8, SHAPE(M, N));

    // Build the backward-chain shape: diff → reshape → expand → MUL with
    // x-expansion → sum over M → reshape to [K,N].
    Term diff_3d   = thvm_reshape(ctx, diff, SHAPE(M, 1, N));
    Term diff_exp  = thvm_expand(ctx, diff_3d, SHAPE(M, K, N));

    Term x_3d      = thvm_reshape(ctx, x, SHAPE(M, K, 1));
    Term x_exp     = thvm_expand(ctx, x_3d, SHAPE(M, K, N));

    Term prod      = thvm_op(ctx, UOP_MUL, diff_exp, x_exp);
    Term summed    = thvm_sum_axes(ctx, prod, (u32[]){0}, 1);   // [1, K, N]
    Term gw        = thvm_reshape(ctx, summed, SHAPE(K, N));

    gw = thvm_eval(ctx, gw);

    // Print the result and compare to the hand-computed ground truth.
    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    f32 *out = thvm_to_host_raw(ctx, gw, &dtype, &shp);
    printf("gw shape = [%u", shp.dims[0]);
    for (u32 i = 1; i < shp.rank; i++) printf(",%u", shp.dims[i]);
    printf("]\n");

    // Ground truth: gw[k, j] = sum_i (diff_over_8[i, j] * x[i, k])
    f32 expected[K*N];
    for (u32 k = 0; k < K; k++)
        for (u32 j = 0; j < N; j++) {
            f32 s = 0.0f;
            for (u32 i = 0; i < M; i++)
                s += diff_over_8[i*N + j] * xd[i*K + k];
            expected[k*N + j] = s;
        }

    printf("gw:       ");
    for (u32 p = 0; p < K*N; p++) printf("%8.4f ", out[p]);
    printf("\n");
    printf("expected: ");
    for (u32 p = 0; p < K*N; p++) printf("%8.4f ", expected[p]);
    printf("\n");

    int ok = 1;
    for (u32 p = 0; p < K*N; p++)
        if (fabsf(out[p] - expected[p]) > 1e-4f) { ok = 0; break; }
    printf("%s\n", ok ? "PASS" : "FAIL");

    thvm_free(ctx);
    return ok ? 0 : 1;
}
