// test_mnist.m — MNIST training as recursive inet
// Same pattern as test_train.m (XOR): entire training chunk is ONE lazy
// term, ONE thvm_reduce() call, no C training loop within a chunk.
//
// Each chunk of 50 steps is a recursive inet program:
//   train = λcounter. IFZ(counter, ERA, λm. forward→loss→grad→assign→recurse)
// Batch slicing via dynamic SHRINK: pairs tensor computed from counter.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "train_helpers.h"

#ifndef DEVICE
  #define DEVICE "cpu"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ============================================================
// MNIST IDX file loader
// ============================================================

static u32 read_u32_be(FILE *f) {
    u8 buf[4];
    fread(buf, 1, 4, f);
    return ((u32)buf[0] << 24) | ((u32)buf[1] << 16) |
           ((u32)buf[2] << 8)  |  (u32)buf[3];
}

static f32 *load_images(const char *path, u32 *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    read_u32_be(f); *n = read_u32_be(f);
    u32 rows = read_u32_be(f), cols = read_u32_be(f);
    u32 pixels = rows * cols;
    u8 *raw = malloc(*n * pixels);
    fread(raw, 1, *n * pixels, f); fclose(f);
    f32 *data = malloc(*n * pixels * sizeof(f32));
    for (u32 i = 0; i < *n * pixels; i++) data[i] = raw[i] / 255.0f;
    free(raw);
    return data;
}

static f32 *load_labels_onehot(const char *path, u32 *n, u8 **raw_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    read_u32_be(f); *n = read_u32_be(f);
    *raw_out = malloc(*n);
    fread(*raw_out, 1, *n, f); fclose(f);
    f32 *onehot = calloc(*n * 10, sizeof(f32));
    for (u32 i = 0; i < *n; i++) onehot[i * 10 + (*raw_out)[i]] = 1.0f;
    return onehot;
}

static void xavier_init(f32 *data, u32 fan_in, u32 fan_out, u32 n) {
    f32 scale = sqrtf(6.0f / (f32)(fan_in + fan_out));
    for (u32 i = 0; i < n; i++)
        data[i] = ((f32)rand() / (f32)RAND_MAX * 2.0f - 1.0f) * scale;
}

// ============================================================
// Main
// ============================================================

int main(void) {
    printf("=== MNIST — recursive inet training (%s) ===\n\n", DEVICE);
    srand(42);

    // Load data
    u32 n_train, n_test;
    u8 *train_raw, *test_raw;
    f32 *train_images = load_images("data/train-images-idx3-ubyte", &n_train);
    f32 *train_onehot = load_labels_onehot("data/train-labels-idx1-ubyte", &n_train, &train_raw);
    f32 *test_images  = load_images("data/t10k-images-idx3-ubyte", &n_test);
    f32 *test_onehot  = load_labels_onehot("data/t10k-labels-idx1-ubyte", &n_test, &test_raw);
    (void)test_onehot;
    printf("  Train: %u, Test: %u\n\n", n_train, n_test);

    u32 BS = 64, H = 128;
    u32 n_batches = n_train / BS;
    u32 n_epochs = 10;
    f32 lr_val = 0.1f;

    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    // Weights: 784 → H → 10
    f32 *w1d = malloc(784 * H * sizeof(f32)); xavier_init(w1d, 784, H, 784 * H);
    f32 *b1d = calloc(H, sizeof(f32));
    f32 *w2d = malloc(H * 10 * sizeof(f32));  xavier_init(w2d, H, 10, H * 10);
    f32 *b2d = calloc(10, sizeof(f32));

    Term W1 = thvm_tensor(ctx, w1d, SHAPE(784, H));
    Term B1 = thvm_tensor(ctx, b1d, SHAPE(1, H));
    Term W2 = thvm_tensor(ctx, w2d, SHAPE(H, 10));
    Term B2 = thvm_tensor(ctx, b2d, SHAPE(1, 10));
    thvm_set_requires_grad(ctx, W1);
    thvm_set_requires_grad(ctx, B1);
    thvm_set_requires_grad(ctx, W2);
    thvm_set_requires_grad(ctx, B2);
    free(w1d); free(b1d); free(w2d); free(b2d);

    // Entire training set as persistent tensors
    Term X_all = thvm_tensor(ctx, train_images, SHAPE(n_train, 784));
    Term Y_all = thvm_tensor(ctx, train_onehot, SHAPE(n_train, 10));

    // Persistent constants
    Term LR = thvm_tensor(ctx, &lr_val, SHAPE(1));
    f32 inv_n = 1.0f / (f32)(BS * 10);
    Term INV_N = thvm_tensor(ctx, &inv_n, SHAPE(1));
    f32 bs_f = (f32)BS;
    Term BS_ten = thvm_tensor(ctx, &bs_f, SHAPE(1));

    // Broadcast masks for dynamic SHRINK pairs construction
    // pairs_x = start * MASK_S + end * MASK_E + FIXED  →  [start, end, 0, 784]
    f32 ms[] = {1,0,0,0}, me[] = {0,1,0,0};
    f32 fx[] = {0,0,0,784}, fy[] = {0,0,0,10};
    Term MASK_S   = thvm_tensor(ctx, ms, SHAPE(4));
    Term MASK_E   = thvm_tensor(ctx, me, SHAPE(4));
    Term FIXED_X  = thvm_tensor(ctx, fx, SHAPE(4));
    Term FIXED_Y  = thvm_tensor(ctx, fy, SHAPE(4));

    u32 n_weights = ctx->tensor_count;

    // ── Training: chunked recursive inet ─────────────────────
    u32 chunk_size = 50;
    u32 n_chunks = n_batches / chunk_size;
    printf("  %u epochs × %u chunks × %u steps (BS=%u)\n\n",
           n_epochs, n_chunks, chunk_size, BS);

    for (u32 epoch = 0; epoch < n_epochs; epoch++) {
        for (u32 c = 0; c < n_chunks; c++) {
            f32 sb = (f32)(c * chunk_size);
            Term SB_ten = thvm_tensor(ctx, &sb, SHAPE(1));
            f32 ns = (f32)chunk_size;
            Term NS_ten = thvm_tensor(ctx, &ns, SHAPE(1));

            Term program = mnist_train_program(ctx,
                W1, B1, W2, B2, X_all, Y_all, LR, INV_N,
                MASK_S, MASK_E, FIXED_X,
                MASK_S, MASK_E, FIXED_Y,
                BS_ten, NS_ten, SB_ten,
                BS, H, (int)chunk_size);

            thvm_reduce(ctx, program);

            // Invalidate host caches so next chunk reads updated weights
            for (u32 i = 0; i < n_weights; i++) {
                if (ctx->tensors[i].host_ptr) {
                    free(ctx->tensors[i].host_ptr);
                    ctx->tensors[i].host_ptr = NULL;
                }
            }
            thvm_reset(ctx, n_weights);
        }
        printf("  Epoch %u complete — itrs=%llu\n\n", epoch + 1,
               (unsigned long long)ctx->itrs);
    }

    // ── Test accuracy ────────────────────────────────────────
    printf("  Evaluating test set...\n");
    u32 test_correct = 0;
    u32 test_batches = n_test / BS;
    for (u32 b = 0; b < test_batches; b++) {
        u32 offset = b * BS;
        Term X = thvm_tensor(ctx, &test_images[offset * 784], SHAPE(BS, 784));
        Term z1  = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, X, W1), B1);
        Term h   = thvm_op(ctx, UOP_RELU, z1, term_era());
        Term out = thvm_reduce(ctx, thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, W2), B2));
        f32 *od = thvm_to_host(ctx, out);
        for (u32 i = 0; i < BS; i++) {
            u32 pred = 0; f32 mx = od[i * 10];
            for (u32 j = 1; j < 10; j++)
                if (od[i * 10 + j] > mx) { mx = od[i * 10 + j]; pred = j; }
            if (pred == test_raw[offset + i]) test_correct++;
        }
        thvm_reset(ctx, n_weights);
    }

    f32 acc = 100.0f * (f32)test_correct / (f32)(test_batches * BS);
    printf("\n  Test accuracy: %.1f%% (%u/%u)\n", acc, test_correct, test_batches * BS);
    printf("\n  %s: MNIST test accuracy %s 90%%\n",
           acc > 90 ? "PASS" : "FAIL", acc > 90 ? ">" : "<");

    thvm_free(ctx);
    free(train_images); free(train_onehot); free(train_raw);
    free(test_images);  free(test_onehot);  free(test_raw);
    return acc > 90 ? 0 : 1;
}
