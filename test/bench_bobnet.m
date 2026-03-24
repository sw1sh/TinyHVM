// bench_bobnet.m — Benchmark TinyBobNet: forward + backward + SGD step
//
// Matches tinygrad's TinyBobNet: x.dot(l1).relu().dot(l2)
// Reports: time per step (forward + backward + update)

#define DEVICE "cpu"
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

static void randn_init(f32 *data, u32 n, f32 scale) {
    for (u32 i = 0; i < n; i++)
        data[i] = scale * ((f32)rand() / (f32)RAND_MAX * 2.0f - 1.0f);
}

int main(int argc, char **argv) {
    const char *dev = (argc > 1) ? argv[1] : DEVICE;
    srand(1337);

    // Match tinygrad test_mnist.py: BS=69, 784→128→10
    u32 BS = 69, IN = 784, H = 128, OUT = 10;
    u32 N_STEPS = 20;
    u32 WARMUP = 3;

    printf("=== BobNet Benchmark (TinyHVM, %s) ===\n", dev);
    printf("    BS=%u, %u→%u→%u, %u steps\n\n", BS, IN, H, OUT, N_STEPS);

    TinyHVM *ctx = thvm_init(thvm_device(dev));

    // Weights
    f32 *w1d = malloc(IN * H * sizeof(f32));
    f32 *w2d = malloc(H * OUT * sizeof(f32));
    randn_init(w1d, IN * H, sqrtf(2.f / (f32)(IN + H)));
    randn_init(w2d, H * OUT, sqrtf(2.f / (f32)(H + OUT)));
    Term W1 = thvm_tensor(ctx, w1d, SHAPE(IN, H));
    Term W2 = thvm_tensor(ctx, w2d, SHAPE(H, OUT));
    thvm_set_requires_grad(ctx, W1);
    thvm_set_requires_grad(ctx, W2);
    free(w1d); free(w2d);

    // Input + one-hot labels
    f32 *xd = malloc(BS * IN * sizeof(f32));
    f32 *yd = calloc(BS * OUT, sizeof(f32));
    randn_init(xd, BS * IN, 1.f);
    for (u32 i = 0; i < BS; i++) yd[i * OUT + (i % OUT)] = 1.f;

    u32 n_w = ctx->tensor_count;
    double total_ms = 0;

    for (u32 step = 0; step < N_STEPS + WARMUP; step++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        // Forward: x @ W1 → relu → @ W2
        Term x = thvm_tensor(ctx, xd, SHAPE(BS, IN));
        Term h = thvm_op(ctx, UOP_RELU,
                 thvm_op(ctx, UOP_MM, x, W1), term_era());
        Term logits = thvm_op(ctx, UOP_MM, h, W2);

        // MSE loss
        Term y = thvm_tensor(ctx, yd, SHAPE(BS, OUT));
        Term diff = thvm_op(ctx, UOP_SUB, logits, y);
        Term sq = thvm_op(ctx, UOP_MUL, diff, diff);
        u32 ax[] = {0, 1};
        Term total = thvm_sum_axes(ctx, sq, ax, 2);
        f32 inv = 1.f / (f32)BS;
        Term loss = thvm_reshape(ctx, thvm_op(ctx, UOP_MUL, total,
                    thvm_tensor(ctx, &inv, SHAPE(1))), SHAPE(1));

        // Backward
        Term g1 = thvm_reduce(ctx, thvm_grad(ctx, loss, W1));
        Term g2 = thvm_reduce(ctx, thvm_grad(ctx, loss, W2));

        // SGD update
        u32 w1id = (u32)term_val(W1), w2id = (u32)term_val(W2);
        f32 *w1h = thvm_to_host(ctx, W1);
        f32 *g1h = thvm_to_host(ctx, g1);
        for (u32 i = 0; i < IN * H; i++) w1h[i] -= 0.001f * g1h[i];
        ctx->backend->buf_write(ctx->tensors[w1id].buf_id, w1h, IN * H * sizeof(f32));

        f32 *w2h = thvm_to_host(ctx, W2);
        f32 *g2h = thvm_to_host(ctx, g2);
        for (u32 i = 0; i < H * OUT; i++) w2h[i] -= 0.001f * g2h[i];
        ctx->backend->buf_write(ctx->tensors[w2id].buf_id, w2h, H * OUT * sizeof(f32));

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

        if (step >= WARMUP) total_ms += ms;

        // Read loss for sanity
        f32 lv = thvm_to_host(ctx, thvm_reduce(ctx, loss))[0];
        printf("  step %2u: loss=%.4f  %.2fms%s\n", step, lv, ms,
               step < WARMUP ? " (warmup)" : "");

        thvm_reset(ctx, n_w);
    }

    double avg = total_ms / (double)N_STEPS;
    printf("\n  Average: %.2f ms/step (%u steps)\n", avg, N_STEPS);
    printf("  Throughput: %.0f samples/sec\n", (double)BS * 1000.0 / avg);

    free(xd); free(yd);
    thvm_free(ctx);
    return 0;
}
