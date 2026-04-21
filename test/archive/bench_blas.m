// bench_blas.m — Raw Accelerate BLAS BobNet (no inet, no autograd)
// This shows the theoretical floor: just BLAS matmul + manual backward.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

#define BS  69
#define IN  784
#define H   128
#define OUT 10
#define N_STEPS 20
#define WARMUP  3

static void randn(float *d, int n, float s) {
    for (int i = 0; i < n; i++)
        d[i] = s * ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
}

int main(void) {
    srand(1337);
    printf("=== Raw BLAS Benchmark (Accelerate) ===\n");
    printf("    BS=%d, %d→%d→%d, %d steps\n\n", BS, IN, H, OUT, N_STEPS);

    float *w1 = malloc(IN * H * sizeof(float));
    float *w2 = malloc(H * OUT * sizeof(float));
    float *x  = malloc(BS * IN * sizeof(float));
    float *y  = calloc(BS * OUT, sizeof(float));

    randn(w1, IN * H, sqrtf(2.0f / (IN + H)));
    randn(w2, H * OUT, sqrtf(2.0f / (H + OUT)));
    randn(x, BS * IN, 1.0f);
    for (int i = 0; i < BS; i++) y[i * OUT + (i % OUT)] = 1.0f;

    // Temporaries
    float *z1      = malloc(BS * H * sizeof(float));    // x @ w1
    float *h       = malloc(BS * H * sizeof(float));    // relu(z1)
    float *logits  = malloc(BS * OUT * sizeof(float));   // h @ w2
    float *diff    = malloc(BS * OUT * sizeof(float));
    float *dlogits = malloc(BS * OUT * sizeof(float));
    float *dh      = malloc(BS * H * sizeof(float));
    float *dz1     = malloc(BS * H * sizeof(float));
    float *dw1     = malloc(IN * H * sizeof(float));
    float *dw2     = malloc(H * OUT * sizeof(float));

    double total_ms = 0;

    for (int step = 0; step < N_STEPS + WARMUP; step++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        // Forward: z1 = x @ w1
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    BS, H, IN, 1.0f, x, IN, w1, H, 0.0f, z1, H);
        // relu
        for (int i = 0; i < BS * H; i++) h[i] = z1[i] > 0 ? z1[i] : 0;
        // logits = h @ w2
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    BS, OUT, H, 1.0f, h, H, w2, OUT, 0.0f, logits, OUT);

        // MSE loss
        float loss = 0;
        for (int i = 0; i < BS * OUT; i++) {
            diff[i] = logits[i] - y[i];
            loss += diff[i] * diff[i];
        }
        loss /= BS;

        // Backward
        float inv_bs = 2.0f / BS;
        for (int i = 0; i < BS * OUT; i++) dlogits[i] = inv_bs * diff[i];

        // dw2 = h^T @ dlogits
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    H, OUT, BS, 1.0f, h, H, dlogits, OUT, 0.0f, dw2, OUT);
        // dh = dlogits @ w2^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    BS, H, OUT, 1.0f, dlogits, OUT, w2, OUT, 0.0f, dh, H);
        // dz1 = dh * relu_mask
        for (int i = 0; i < BS * H; i++) dz1[i] = dh[i] * (z1[i] > 0 ? 1.0f : 0.0f);
        // dw1 = x^T @ dz1
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    IN, H, BS, 1.0f, x, IN, dz1, H, 0.0f, dw1, H);

        // SGD
        for (int i = 0; i < IN * H; i++) w1[i] -= 0.001f * dw1[i];
        for (int i = 0; i < H * OUT; i++) w2[i] -= 0.001f * dw2[i];

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        if (step >= WARMUP) total_ms += ms;
        printf("  step %2d: loss=%.4f  %.3fms%s\n", step, loss, ms,
               step < WARMUP ? " (warmup)" : "");
    }

    double avg = total_ms / N_STEPS;
    printf("\n  Average: %.3f ms/step (%d steps)\n", avg, N_STEPS);
    printf("  Throughput: %.0f samples/sec\n", BS * 1000.0 / avg);

    free(w1); free(w2); free(x); free(y);
    free(z1); free(h); free(logits); free(diff);
    free(dlogits); free(dh); free(dz1); free(dw1); free(dw2);
    return 0;
}
