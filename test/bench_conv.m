// bench_conv.m — Conv gradient benchmark for TinyHVM
// Compares forward + backward of a small CNN on CPU and Metal
//
// Architecture: Conv(1→8, 3x3, pad=1) → ReLU → Conv(8→16, 3x3) → ReLU → flatten → MM → MSE
// Input: [BS, 1, 8, 8]  (small spatial to keep it fast)

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BS     16
#define CIN    1
#define H      8
#define W      8
#define C1     8     // conv1 output channels
#define C2     16    // conv2 output channels
#define KH     3
#define KW     3
#define NCLASS 10

#define N_STEPS  10
#define WARMUP   3
#define LR       0.001f

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const char *dev = (argc > 1) ? argv[1] : "cpu";
    printf("=== Conv Gradient Benchmark (TinyHVM, %s) ===\n", dev);
    printf("    BS=%d, Conv(%d→%d,3x3)→ReLU→Conv(%d→%d,3x3)→ReLU→FC→MSE\n",
           BS, CIN, C1, C1, C2);
    printf("    Input: [%d,%d,%d,%d], %d steps\n\n", BS, CIN, H, W, N_STEPS);

    srand(42);
    #define RANDF() (((float)rand()/(float)RAND_MAX) * 2.f - 1.f)

    // Weights
    // Conv1: [C1, CIN, KH, KW] = [8, 1, 3, 3] = 72 params
    float w1[C1 * CIN * KH * KW];
    float scale1 = sqrtf(2.f / (float)(CIN * KH * KW));
    for (int i = 0; i < C1*CIN*KH*KW; i++) w1[i] = RANDF() * scale1;

    // Conv2: [C2, C1, KH, KW] = [16, 8, 3, 3] = 1152 params
    float w2[C2 * C1 * KH * KW];
    float scale2 = sqrtf(2.f / (float)(C1 * KH * KW));
    for (int i = 0; i < C2*C1*KH*KW; i++) w2[i] = RANDF() * scale2;

    // After conv1(pad=1,s=1): [BS, 8, 8, 8] → after conv2(no pad,s=1): [BS, 16, 6, 6]
    // Flatten: [BS, 16*6*6] = [BS, 576]
    int flat_dim = C2 * 6 * 6;

    // FC: [flat_dim, NCLASS] = [576, 10]
    float *fc = malloc(flat_dim * NCLASS * sizeof(float));
    float scale_fc = sqrtf(2.f / (float)(flat_dim + NCLASS));
    for (int i = 0; i < flat_dim * NCLASS; i++) fc[i] = RANDF() * scale_fc;

    // Input data
    float *x_data = malloc(BS * CIN * H * W * sizeof(float));
    for (int i = 0; i < BS * CIN * H * W; i++) x_data[i] = RANDF() * 0.5f;

    // Labels (one-hot)
    float *y_data = calloc(BS * NCLASS, sizeof(float));
    for (int i = 0; i < BS; i++) y_data[i * NCLASS + (i % NCLASS)] = 1.f;

    double times[N_STEPS];
    u32 stride1[] = {1, 1}, pad1[] = {1, 1, 1, 1};  // same padding
    u32 stride2[] = {1, 1}, pad2[] = {0, 0, 0, 0};  // no padding

    for (int step = 0; step < N_STEPS + WARMUP; step++) {
        TinyHVM *ctx = thvm_init(dev);
        double t0 = now_ms();

        // Load weights
        Term tw1 = thvm_tensor(ctx, w1, SHAPE(C1, CIN, KH, KW));
        Term tw2 = thvm_tensor(ctx, w2, SHAPE(C2, C1, KH, KW));
        Term tfc = thvm_tensor(ctx, fc, SHAPE(flat_dim, NCLASS));
        thvm_set_requires_grad(ctx, tw1);
        thvm_set_requires_grad(ctx, tw2);
        thvm_set_requires_grad(ctx, tfc);

        // Load data
        Term tx = thvm_tensor(ctx, x_data, SHAPE(BS, CIN, H, W));
        Term ty = thvm_tensor(ctx, y_data, SHAPE(BS, NCLASS));

        // Forward: conv1 → relu → conv2 → relu → flatten → FC
        Term h1 = thvm_conv2d(ctx, tx, tw1, term_era(), 1, stride1, pad1);
        h1 = thvm_op(ctx, UOP_RELU, h1, term_era());

        Term h2 = thvm_conv2d(ctx, h1, tw2, term_era(), 1, stride2, pad2);
        h2 = thvm_op(ctx, UOP_RELU, h2, term_era());

        // Flatten
        Term flat = thvm_reshape(ctx, h2, SHAPE(BS, flat_dim));

        // FC
        Term logits = thvm_op(ctx, UOP_MM, flat, tfc);

        // MSE loss
        Term diff = thvm_op(ctx, UOP_SUB, logits, ty);
        Term sq = thvm_op(ctx, UOP_MUL, diff, diff);
        u32 ax[] = {0, 1};
        Term sum_loss = thvm_sum_axes(ctx, sq, ax, 2);
        f32 bs_f = (f32)BS;
        Term loss = thvm_op(ctx, UOP_DIV,
            thvm_reshape(ctx, sum_loss, SHAPE(1)),
            thvm_tensor(ctx, &bs_f, SHAPE(1)));

        // Backward
        Term gw1 = thvm_grad(ctx, loss, tw1);
        Term gw2 = thvm_grad(ctx, loss, tw2);
        Term gfc = thvm_grad(ctx, loss, tfc);

        // Reduce all grads (force computation)
        Term gw1_r = thvm_reduce(ctx, gw1);
        Term gw2_r = thvm_reduce(ctx, gw2);
        Term gfc_r = thvm_reduce(ctx, gfc);

        double t1 = now_ms();
        double ms = t1 - t0;
        float lv = thvm_to_host(ctx, thvm_reduce(ctx, loss))[0];

        // SGD update (read back, update, will be loaded fresh next step)
        if (term_tag(gw1_r) == TAG_TEN) {
            float *g = thvm_to_host(ctx, gw1_r);
            for (int i = 0; i < C1*CIN*KH*KW; i++) w1[i] -= LR * g[i];
        }
        if (term_tag(gw2_r) == TAG_TEN) {
            float *g = thvm_to_host(ctx, gw2_r);
            for (int i = 0; i < C2*C1*KH*KW; i++) w2[i] -= LR * g[i];
        }
        if (term_tag(gfc_r) == TAG_TEN) {
            float *g = thvm_to_host(ctx, gfc_r);
            for (int i = 0; i < flat_dim*NCLASS; i++) fc[i] -= LR * g[i];
        }

        const char *tag = (step < WARMUP) ? " (warmup)" : "";
        if (step >= WARMUP) times[step - WARMUP] = ms;
        printf("  step %2d: loss=%.4f  %.2fms  tensors=%u  heap=%llu%s\n",
               step, lv, ms, ctx->tensor_count, ctx->heap_pos, tag);

        thvm_free(ctx);
    }

    double avg = 0;
    for (int i = 0; i < N_STEPS; i++) avg += times[i];
    avg /= N_STEPS;
    printf("\n  Average: %.2f ms/step (%d steps)\n", avg, N_STEPS);
    printf("  Throughput: %.0f samples/sec\n", BS * 1000.0 / avg);

    free(x_data); free(y_data); free(fc);
    return 0;
}
