// test_mnist.m — MNIST digit classification (inet training)
// 2-layer MLP: [784] → [128] → [10], relu activation
// MSE loss on one-hot targets, SGD via ASSIGN
//
// Each training step is ONE inet term:
//   LOG_PRINT(loss) → ASSIGN(W1) → ASSIGN(B1) → ASSIGN(W2) → ASSIGN(B2) → ERA
// Single thvm_reduce() per step. No host-side SGD. No manual loss reading.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif

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

// Load MNIST images: returns f32 array [n × 784], normalized to [0,1]
static f32 *load_images(const char *path, u32 *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    read_u32_be(f);  // magic
    *n = read_u32_be(f);
    u32 rows = read_u32_be(f);
    u32 cols = read_u32_be(f);
    u32 pixels = rows * cols;  // 784

    u8 *raw = malloc(*n * pixels);
    fread(raw, 1, *n * pixels, f);
    fclose(f);

    f32 *data = malloc(*n * pixels * sizeof(f32));
    for (u32 i = 0; i < *n * pixels; i++)
        data[i] = raw[i] / 255.0f;
    free(raw);
    return data;
}

// Load MNIST labels: returns one-hot f32 array [n × 10]
static f32 *load_labels(const char *path, u32 *n, u8 **raw_labels) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    read_u32_be(f);  // magic
    *n = read_u32_be(f);

    *raw_labels = malloc(*n);
    fread(*raw_labels, 1, *n, f);
    fclose(f);

    f32 *onehot = calloc(*n * 10, sizeof(f32));
    for (u32 i = 0; i < *n; i++)
        onehot[i * 10 + (*raw_labels)[i]] = 1.0f;
    return onehot;
}

// ============================================================
// Xavier initialization
// ============================================================

static void xavier_init(f32 *data, u32 fan_in, u32 fan_out, u32 n) {
    f32 scale = sqrtf(6.0f / (f32)(fan_in + fan_out));
    for (u32 i = 0; i < n; i++)
        data[i] = ((f32)rand() / (f32)RAND_MAX * 2.0f - 1.0f) * scale;
}

// ============================================================
// Main
// ============================================================

int main(void) {
    printf("=== MNIST Training — inet steps (%s) ===\n\n", DEVICE);
    srand(42);

    // Load data
    u32 n_train, n_test;
    u8 *train_raw_labels, *test_raw_labels;
    f32 *train_images = load_images("data/train-images-idx3-ubyte", &n_train);
    f32 *train_onehot = load_labels("data/train-labels-idx1-ubyte", &n_train, &train_raw_labels);
    f32 *test_images  = load_images("data/t10k-images-idx3-ubyte", &n_test);
    f32 *test_onehot  = load_labels("data/t10k-labels-idx1-ubyte", &n_test, &test_raw_labels);
    (void)test_onehot;

    printf("  Train: %u images, Test: %u images\n\n", n_train, n_test);

    u32 batch_size = 64;
    u32 n_epochs = 10;
    f32 lr = 0.1f;  // higher lr since loss is averaged (divided by BS*10)

    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    // Initialize weights: 784→128→10
    u32 H = 128;
    f32 *w1_data = malloc(784 * H * sizeof(f32));
    f32 *b1_data = calloc(H, sizeof(f32));
    f32 *w2_data = malloc(H * 10 * sizeof(f32));
    f32 *b2_data = calloc(10, sizeof(f32));

    xavier_init(w1_data, 784, H, 784 * H);
    xavier_init(w2_data, H, 10, H * 10);

    Term W1 = thvm_tensor(ctx, w1_data, SHAPE(784, H));
    Term B1 = thvm_tensor(ctx, b1_data, SHAPE(1, H));
    Term W2 = thvm_tensor(ctx, w2_data, SHAPE(H, 10));
    Term B2 = thvm_tensor(ctx, b2_data, SHAPE(1, 10));

    thvm_set_requires_grad(ctx, W1);
    thvm_set_requires_grad(ctx, B1);
    thvm_set_requires_grad(ctx, W2);
    thvm_set_requires_grad(ctx, B2);

    // Persistent scalars
    Term LR = thvm_tensor(ctx, &lr, SHAPE(1));
    f32 inv_n = 1.0f / (f32)(batch_size * 10);
    Term INV_N = thvm_tensor(ctx, &inv_n, SHAPE(1));

    u32 n_weights = ctx->tensor_count;  // all persistent tensors
    free(w1_data); free(b1_data); free(w2_data); free(b2_data);

    // Training
    u32 n_batches = n_train / batch_size;
    printf("  Training %u epochs × %u batches, BS=%u\n\n", n_epochs, n_batches, batch_size);

    for (u32 epoch = 0; epoch < n_epochs; epoch++) {
        for (u32 b = 0; b < n_batches; b++) {
            u32 offset = b * batch_size;

            // Create batch tensors
            Term X = thvm_tensor(ctx, &train_images[offset * 784], SHAPE(batch_size, 784));
            Term Y = thvm_tensor(ctx, &train_onehot[offset * 10],  SHAPE(batch_size, 10));

            // Forward: h = relu(X·W1 + B1), out = h·W2 + B2
            Term z1  = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, X, W1), B1);
            Term h   = thvm_op(ctx, UOP_RELU, z1, term_era());
            Term out = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, W2), B2);

            // MSE loss → scalar: sum((out - Y)^2) / (BS * 10)
            Term diff = thvm_op(ctx, UOP_SUB, out, Y);
            Term sq   = thvm_op(ctx, UOP_MUL, diff, diff);
            u32 axes[] = {0, 1};
            Term loss = thvm_sum_axes(ctx, sq, axes, 2);
            loss = thvm_op(ctx, UOP_MUL, loss, INV_N);
            loss = thvm_reshape(ctx, loss, SHAPE(1));

            // Force loss (drives forward pass), then use reduced TEN for GRAD
            Term loss_ten = thvm_reduce(ctx, loss);

            // Log loss inside inet
            if (b % 100 == 0)
                thvm_reduce(ctx, thvm_log_print(ctx, loss_ten));

            // Gradients (lazy — walk provenance of the realized loss)
            Term gW1 = thvm_grad(ctx, loss_ten, W1);
            Term gB1 = thvm_grad(ctx, loss_ten, B1);
            Term gW2 = thvm_grad(ctx, loss_ten, W2);
            Term gB2 = thvm_grad(ctx, loss_ten, B2);

            // SGD via ASSIGN: W -= LR * grad  (in-place, no host roundtrip)
            thvm_reduce(ctx, thvm_assign(ctx, W1,
                thvm_op(ctx, UOP_SUB, W1, thvm_op(ctx, UOP_MUL, LR, gW1))));
            thvm_reduce(ctx, thvm_assign(ctx, B1,
                thvm_op(ctx, UOP_SUB, B1, thvm_op(ctx, UOP_MUL, LR, gB1))));
            thvm_reduce(ctx, thvm_assign(ctx, W2,
                thvm_op(ctx, UOP_SUB, W2, thvm_op(ctx, UOP_MUL, LR, gW2))));
            thvm_reduce(ctx, thvm_assign(ctx, B2,
                thvm_op(ctx, UOP_SUB, B2, thvm_op(ctx, UOP_MUL, LR, gB2))));

            thvm_reset(ctx, n_weights);

            if (b % 100 == 0)
                printf("  epoch %u batch %u/%u\n", epoch+1, b, n_batches);
        }
        printf("  Epoch %u complete\n\n", epoch+1);
    }

    // Test accuracy
    printf("  Evaluating test set...\n");
    u32 test_correct = 0;
    u32 test_batches = n_test / batch_size;
    for (u32 b = 0; b < test_batches; b++) {
        u32 offset = b * batch_size;
        Term X = thvm_tensor(ctx, &test_images[offset * 784], SHAPE(batch_size, 784));

        Term z1  = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, X, W1), B1);
        Term h   = thvm_op(ctx, UOP_RELU, z1, term_era());
        Term out = thvm_reduce(ctx, thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, W2), B2));

        f32 *out_data = thvm_to_host(ctx, out);
        for (u32 i = 0; i < batch_size; i++) {
            u32 pred = 0;
            f32 max_val = out_data[i * 10];
            for (u32 j = 1; j < 10; j++) {
                if (out_data[i * 10 + j] > max_val) {
                    max_val = out_data[i * 10 + j];
                    pred = j;
                }
            }
            if (pred == test_raw_labels[offset + i]) test_correct++;
        }
        thvm_reset(ctx, n_weights);
    }

    f32 test_acc = 100.0f * (f32)test_correct / (f32)(test_batches * batch_size);
    printf("\n  Test accuracy: %.1f%% (%u/%u)\n", test_acc, test_correct, test_batches * batch_size);

    int pass = test_acc > 90.0f;
    printf("\n  %s: MNIST test accuracy %s 90%%\n", pass ? "PASS" : "FAIL",
           pass ? ">" : "<");

    // Cleanup
    thvm_free(ctx);
    free(train_images); free(train_onehot); free(train_raw_labels);
    free(test_images);  free(test_onehot);  free(test_raw_labels);

    return pass ? 0 : 1;
}
