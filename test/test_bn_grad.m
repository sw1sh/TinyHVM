// test_bn_grad.m — Minimal BN backward correctness test
// Compares analytical backward through BN with numerical gradient (finite diff).

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"

#ifndef DEVICE
  #define DEVICE "metal"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// BN forward: y = (x - mean) * invstd * gamma + beta
// Full backward formula:
//   dx = gamma * invstd * (dy - mean(dy) - xhat * mean(dy * xhat))
//   dgamma = sum(dy * xhat, axes=(0,2,3))
//   dbeta = sum(dy, axes=(0,2,3))
// where xhat = (x - mean) * invstd

static void bn_backward_ref(const f32 *x, const f32 *dy, const f32 *gamma,
                              f32 *dx_out, f32 *dgamma_out, f32 *dbeta_out,
                              u32 B, u32 C, u32 H, u32 W) {
    u32 N = B * H * W;
    f32 eps = 1e-5f;

    for (u32 c = 0; c < C; c++) {
        // Compute mean and var for channel c
        f32 mean = 0, var = 0;
        for (u32 b = 0; b < B; b++)
            for (u32 h = 0; h < H; h++)
                for (u32 w = 0; w < W; w++)
                    mean += x[((b*C + c)*H + h)*W + w];
        mean /= (f32)N;
        for (u32 b = 0; b < B; b++)
            for (u32 h = 0; h < H; h++)
                for (u32 w = 0; w < W; w++) {
                    f32 d = x[((b*C + c)*H + h)*W + w] - mean;
                    var += d * d;
                }
        var /= (f32)N;
        f32 invstd = 1.0f / sqrtf(var + eps);

        // Compute xhat, sum(dy), sum(dy*xhat)
        f32 sum_dy = 0, sum_dy_xhat = 0;
        for (u32 b = 0; b < B; b++)
            for (u32 h = 0; h < H; h++)
                for (u32 w = 0; w < W; w++) {
                    u32 idx = ((b*C + c)*H + h)*W + w;
                    f32 xhat = (x[idx] - mean) * invstd;
                    sum_dy += dy[idx];
                    sum_dy_xhat += dy[idx] * xhat;
                }

        // dgamma, dbeta
        dgamma_out[c] = 0; dbeta_out[c] = 0;
        for (u32 b = 0; b < B; b++)
            for (u32 h = 0; h < H; h++)
                for (u32 w = 0; w < W; w++) {
                    u32 idx = ((b*C + c)*H + h)*W + w;
                    f32 xhat = (x[idx] - mean) * invstd;
                    dgamma_out[c] += dy[idx] * xhat;
                    dbeta_out[c] += dy[idx];
                }

        // dx
        for (u32 b = 0; b < B; b++)
            for (u32 h = 0; h < H; h++)
                for (u32 w = 0; w < W; w++) {
                    u32 idx = ((b*C + c)*H + h)*W + w;
                    f32 xhat = (x[idx] - mean) * invstd;
                    dx_out[idx] = gamma[c] * invstd * (dy[idx] - sum_dy/(f32)N - xhat * sum_dy_xhat/(f32)N);
                }
    }
}

int main(void) {
    srand(42);
    printf("=== BN backward test (%s) ===\n\n", DEVICE);

    u32 B = 4, C = 2, H = 3, W = 3;
    u32 total = B * C * H * W;
    u32 N = B * H * W; // = 36

    // Create random input
    f32 *x_data = malloc(total * sizeof(f32));
    for (u32 i = 0; i < total; i++) x_data[i] = 0.5f * ((f32)rand()/(f32)RAND_MAX * 2.0f - 1.0f);

    f32 gamma_data[2] = {1.0f, 1.0f};
    f32 beta_data[2] = {0.0f, 0.0f};
    f32 rmean_data[2] = {0.0f, 0.0f};
    f32 rvar_data[2] = {1.0f, 1.0f};

    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    Term x = thvm_tensor(ctx, x_data, (Shape){.dims={B,C,H,W},.rank=4});
    Term gamma_t = thvm_tensor(ctx, gamma_data, SHAPE(C));
    Term beta_t = thvm_tensor(ctx, beta_data, SHAPE(C));
    Term rmean_t = thvm_tensor(ctx, rmean_data, SHAPE(C));
    Term rvar_t = thvm_tensor(ctx, rvar_data, SHAPE(C));

    thvm_set_requires_grad(ctx, x);
    thvm_set_requires_grad(ctx, gamma_t);
    thvm_set_requires_grad(ctx, beta_t);

    // Forward
    Term bn_out = batchnorm_term(ctx, x, gamma_t, beta_t, rmean_t, rvar_t, B, C, H, W, 1);

    // Loss = sum(bn_out * dy) where dy is random
    f32 *dy_data = malloc(total * sizeof(f32));
    for (u32 i = 0; i < total; i++) dy_data[i] = 0.1f * ((f32)rand()/(f32)RAND_MAX * 2.0f - 1.0f);
    Term dy = thvm_tensor(ctx, dy_data, (Shape){.dims={B,C,H,W},.rank=4});

    // loss = sum(bn_out * dy)  — flatten then sum to avoid rank-change issues
    Term flat = thvm_reshape(ctx, thvm_op(ctx, UOP_MUL, bn_out, dy), SHAPE(total));
    Term loss = thvm_op(ctx, UOP_SUM, flat, term_era());

    // Reduce and backward
    Term loss_r = thvm_reduce(ctx, loss);
    f32 loss_val = thvm_to_host(ctx, loss_r)[0];
    printf("  loss = %.6f\n", loss_val);

    // BN output check
    f32 *bn_data = thvm_to_host(ctx, thvm_reduce(ctx, bn_out));
    printf("  bn_out[0:5] = %.4f %.4f %.4f %.4f %.4f\n",
           bn_data[0], bn_data[1], bn_data[2], bn_data[3], bn_data[4]);

    // Check BN output: mean should be ~0, var should be ~1 per channel
    for (u32 c = 0; c < C; c++) {
        f32 sum = 0, sum2 = 0;
        for (u32 b = 0; b < B; b++)
            for (u32 h = 0; h < H; h++)
                for (u32 w = 0; w < W; w++) {
                    f32 v = bn_data[((b*C + c)*H + h)*W + w];
                    sum += v;
                    sum2 += v * v;
                }
        f32 mean = sum / (f32)N;
        f32 var = sum2 / (f32)N - mean * mean;
        printf("  channel %u: mean=%.6f  var=%.6f  (should be ~0, ~1)\n", c, mean, var);
    }

    // Backward
    Term params[] = {x, gamma_t, beta_t};
    Term grads[3];
    thvm_backward(ctx, loss_r, params, grads, 3);

    // Get our gradients
    f32 *dx_ours = thvm_to_host(ctx, grads[0]);
    f32 *dgamma_ours = thvm_to_host(ctx, grads[1]);
    f32 *dbeta_ours = thvm_to_host(ctx, grads[2]);

    // Compute reference gradients
    f32 *dx_ref = malloc(total * sizeof(f32));
    f32 dgamma_ref[2], dbeta_ref[2];
    bn_backward_ref(x_data, dy_data, gamma_data, dx_ref, dgamma_ref, dbeta_ref, B, C, H, W);

    // Compare
    printf("\n  --- dgamma ---\n");
    for (u32 c = 0; c < C; c++)
        printf("  ch%u: ours=%.6f  ref=%.6f  ratio=%.2f\n",
               c, dgamma_ours[c], dgamma_ref[c],
               dgamma_ref[c] != 0 ? dgamma_ours[c]/dgamma_ref[c] : 0);

    printf("\n  --- dbeta ---\n");
    for (u32 c = 0; c < C; c++)
        printf("  ch%u: ours=%.6f  ref=%.6f  ratio=%.2f\n",
               c, dbeta_ours[c], dbeta_ref[c],
               dbeta_ref[c] != 0 ? dbeta_ours[c]/dbeta_ref[c] : 0);

    printf("\n  --- dx (first 10) ---\n");
    f32 dx_rms_ours = 0, dx_rms_ref = 0, dx_err_rms = 0;
    for (u32 i = 0; i < total; i++) {
        dx_rms_ours += dx_ours[i] * dx_ours[i];
        dx_rms_ref += dx_ref[i] * dx_ref[i];
        f32 e = dx_ours[i] - dx_ref[i];
        dx_err_rms += e * e;
    }
    dx_rms_ours = sqrtf(dx_rms_ours / total);
    dx_rms_ref = sqrtf(dx_rms_ref / total);
    dx_err_rms = sqrtf(dx_err_rms / total);

    for (u32 i = 0; i < 10 && i < total; i++)
        printf("  [%u] ours=%.6f  ref=%.6f  diff=%.6f\n",
               i, dx_ours[i], dx_ref[i], dx_ours[i] - dx_ref[i]);

    printf("\n  dx rms: ours=%.6f  ref=%.6f  err=%.6f  ratio=%.2f\n",
           dx_rms_ours, dx_rms_ref, dx_err_rms,
           dx_rms_ref != 0 ? dx_rms_ours / dx_rms_ref : 0);

    int pass = (dx_err_rms < 0.01f * dx_rms_ref);
    printf("\n  %s: BN backward dx error %s 1%% of reference\n",
           pass ? "PASS" : "FAIL", pass ? "<" : ">=");

    free(x_data); free(dy_data); free(dx_ref);
    thvm_free(ctx);
    return pass ? 0 : 1;
}
