// test_mnist_cnn.m — CNN matching tinygrad's beautiful_mnist.py
// Architecture: Conv2d(1,32,5)→ReLU→Conv2d(32,32,5)→ReLU→BN(32)→MaxPool
//             → Conv2d(32,64,3)→ReLU→Conv2d(64,64,3)→ReLU→BN(64)→MaxPool
//             → Flatten → Linear(576,10)
// Loss: softmax cross-entropy
// Optimizer: Adam (lr=0.001)
//
// Usage: ./test_mnist_cnn

#include "../src/tinyhvm.c"
#include "../src/cpu.c"
#ifdef __APPLE__
  #include "../src/metal.m"
#endif
#include "../src/layers.c"

#ifndef DEVICE
  #define DEVICE "cpu"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// ============================================================
// MNIST loader (same as test_mnist.m)
// ============================================================

static u32 read_u32_be(FILE *f) {
    u8 buf[4];
    fread(buf, 1, 4, f);
    return ((u32)buf[0] << 24) | ((u32)buf[1] << 16) |
           ((u32)buf[2] << 8)  |  (u32)buf[3];
}

static f32 *load_images(const char *path, u32 *n) {
    FILE *f = fopen(path, "rb"); assert(f);
    read_u32_be(f); *n = read_u32_be(f);
    u32 rows = read_u32_be(f), cols = read_u32_be(f);
    u32 px = rows * cols;
    u8 *raw = malloc(*n * px);
    fread(raw, 1, *n * px, f); fclose(f);
    f32 *data = malloc(*n * px * sizeof(f32));
    for (u32 i = 0; i < *n * px; i++) data[i] = raw[i] / 255.0f;
    free(raw);
    return data;
}

static u8 *load_labels(const char *path, u32 *n) {
    FILE *f = fopen(path, "rb"); assert(f);
    read_u32_be(f); *n = read_u32_be(f);
    u8 *labels = malloc(*n);
    fread(labels, 1, *n, f); fclose(f);
    return labels;
}

// ============================================================
// Xavier init
// ============================================================

static void xavier_init(f32 *data, u32 fan_in, u32 fan_out, u32 n) {
    f32 scale = sqrtf(2.0f / (f32)(fan_in + fan_out));
    for (u32 i = 0; i < n; i++)
        data[i] = ((f32)rand() / (f32)RAND_MAX * 2.0f - 1.0f) * scale;
}

// ============================================================
// Main
// ============================================================

int main(void) {
    printf("=== Beautiful MNIST (TinyHVM CNN, %s) ===\n\n", DEVICE);
    srand(42);

    // Load data
    u32 n_train, n_test;
    f32 *train_images = load_images("data/train-images-idx3-ubyte", &n_train);
    u8  *train_labels = load_labels("data/train-labels-idx1-ubyte", &n_train);
    f32 *test_images  = load_images("data/t10k-images-idx3-ubyte", &n_test);
    u8  *test_labels  = load_labels("data/t10k-labels-idx1-ubyte", &n_test);
    printf("  Train: %u, Test: %u\n\n", n_train, n_test);

    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    // ========== Allocate parameters ==========
    // Conv1: (1,32,5,5) → 800 weights + 32 bias
    u32 conv1_w_n = 32*1*5*5;
    f32 *conv1_w_data = malloc(conv1_w_n * sizeof(f32));
    xavier_init(conv1_w_data, 1*5*5, 32*5*5, conv1_w_n);
    u32 conv1_w = tensor_from_data(ctx, conv1_w_data, (Shape){.dims={32,1,5,5}, .rank=4});
    f32 *conv1_b_data = calloc(32, sizeof(f32));
    u32 conv1_b = tensor_from_data(ctx, conv1_b_data, SHAPE(32));

    // Conv2: (32,32,5,5)
    u32 conv2_w_n = 32*32*5*5;
    f32 *conv2_w_data = malloc(conv2_w_n * sizeof(f32));
    xavier_init(conv2_w_data, 32*5*5, 32*5*5, conv2_w_n);
    u32 conv2_w = tensor_from_data(ctx, conv2_w_data, (Shape){.dims={32,32,5,5}, .rank=4});
    f32 *conv2_b_data = calloc(32, sizeof(f32));
    u32 conv2_b = tensor_from_data(ctx, conv2_b_data, SHAPE(32));

    // BN1: gamma=1, beta=0, running_mean=0, running_var=1
    f32 *bn1_gamma_data = malloc(32 * sizeof(f32));
    for (u32 i = 0; i < 32; i++) bn1_gamma_data[i] = 1.0f;
    u32 bn1_gamma = tensor_from_data(ctx, bn1_gamma_data, SHAPE(32));
    f32 *bn1_beta_data = calloc(32, sizeof(f32));
    u32 bn1_beta = tensor_from_data(ctx, bn1_beta_data, SHAPE(32));
    f32 *bn1_rm_data = calloc(32, sizeof(f32));
    u32 bn1_rmean = tensor_from_data(ctx, bn1_rm_data, SHAPE(32));
    f32 *bn1_rv_data = malloc(32 * sizeof(f32));
    for (u32 i = 0; i < 32; i++) bn1_rv_data[i] = 1.0f;
    u32 bn1_rvar = tensor_from_data(ctx, bn1_rv_data, SHAPE(32));

    // Conv3: (32,64,3,3)
    u32 conv3_w_n = 64*32*3*3;
    f32 *conv3_w_data = malloc(conv3_w_n * sizeof(f32));
    xavier_init(conv3_w_data, 32*3*3, 64*3*3, conv3_w_n);
    u32 conv3_w = tensor_from_data(ctx, conv3_w_data, (Shape){.dims={64,32,3,3}, .rank=4});
    f32 *conv3_b_data = calloc(64, sizeof(f32));
    u32 conv3_b = tensor_from_data(ctx, conv3_b_data, SHAPE(64));

    // Conv4: (64,64,3,3)
    u32 conv4_w_n = 64*64*3*3;
    f32 *conv4_w_data = malloc(conv4_w_n * sizeof(f32));
    xavier_init(conv4_w_data, 64*3*3, 64*3*3, conv4_w_n);
    u32 conv4_w = tensor_from_data(ctx, conv4_w_data, (Shape){.dims={64,64,3,3}, .rank=4});
    f32 *conv4_b_data = calloc(64, sizeof(f32));
    u32 conv4_b = tensor_from_data(ctx, conv4_b_data, SHAPE(64));

    // BN2: gamma=1, beta=0, running_mean=0, running_var=1
    f32 *bn2_gamma_data = malloc(64 * sizeof(f32));
    for (u32 i = 0; i < 64; i++) bn2_gamma_data[i] = 1.0f;
    u32 bn2_gamma = tensor_from_data(ctx, bn2_gamma_data, SHAPE(64));
    f32 *bn2_beta_data = calloc(64, sizeof(f32));
    u32 bn2_beta = tensor_from_data(ctx, bn2_beta_data, SHAPE(64));
    f32 *bn2_rm_data = calloc(64, sizeof(f32));
    u32 bn2_rmean = tensor_from_data(ctx, bn2_rm_data, SHAPE(64));
    f32 *bn2_rv_data = malloc(64 * sizeof(f32));
    for (u32 i = 0; i < 64; i++) bn2_rv_data[i] = 1.0f;
    u32 bn2_rvar = tensor_from_data(ctx, bn2_rv_data, SHAPE(64));

    // Linear: (576,10)
    u32 fc_w_n = 576 * 10;
    f32 *fc_w_data = malloc(fc_w_n * sizeof(f32));
    xavier_init(fc_w_data, 576, 10, fc_w_n);
    u32 fc_w = tensor_from_data(ctx, fc_w_data, SHAPE(576, 10));
    f32 *fc_b_data = calloc(10, sizeof(f32));
    u32 fc_b = tensor_from_data(ctx, fc_b_data, SHAPE(10));

    u32 n_weights = ctx->tensor_count;

    // Free init data
    free(conv1_w_data); free(conv1_b_data); free(conv2_w_data); free(conv2_b_data);
    free(bn1_gamma_data); free(bn1_beta_data); free(bn1_rm_data); free(bn1_rv_data);
    free(conv3_w_data); free(conv3_b_data); free(conv4_w_data); free(conv4_b_data);
    free(bn2_gamma_data); free(bn2_beta_data); free(bn2_rm_data); free(bn2_rv_data);
    free(fc_w_data); free(fc_b_data);

    // ========== Adam optimizer ==========
    // Params: conv1_w, conv1_b, conv2_w, conv2_b, bn1_gamma, bn1_beta,
    //         conv3_w, conv3_b, conv4_w, conv4_b, bn2_gamma, bn2_beta,
    //         fc_w, fc_b
    #define N_PARAMS 14
    Adam opt = adam_init(0.001f, N_PARAMS);
    u32 param_ids[] = {conv1_w, conv1_b, conv2_w, conv2_b, bn1_gamma, bn1_beta,
                       conv3_w, conv3_b, conv4_w, conv4_b, bn2_gamma, bn2_beta,
                       fc_w, fc_b};
    u32 param_sizes[] = {32*1*5*5, 32, 32*32*5*5, 32, 32, 32,
                         64*32*3*3, 64, 64*64*3*3, 64, 64, 64,
                         576*10, 10};
    for (u32 i = 0; i < N_PARAMS; i++)
        adam_add_param(&opt, i, param_ids[i], param_sizes[i]);

    // ========== Training loop ==========
    u32 BS = 64;  // batch size (beautiful_mnist uses 512, we use smaller for speed)
    u32 n_steps = 70;  // same as beautiful_mnist

    printf("  Training %u steps, BS=%u...\n\n", n_steps, BS);

    for (u32 step = 0; step < n_steps; step++) {
        clock_t t0 = clock();

        // Random batch
        u32 sample_indices[BS]; // using stack since BS is small
        (void)sample_indices;
        f32 *batch_x = malloc(BS * 1 * 28 * 28 * sizeof(f32));
        u8  *batch_y = malloc(BS);

        for (u32 i = 0; i < BS; i++) {
            u32 idx = (u32)(rand() % (int)n_train);
            memcpy(&batch_x[i * 784], &train_images[idx * 784], 784 * sizeof(f32));
            batch_y[i] = train_labels[idx];
        }

        // Create input tensor [BS,1,28,28]
        u32 x = tensor_from_data(ctx, batch_x, (Shape){.dims={BS,1,28,28}, .rank=4});

        // ===== Forward pass =====
        // Conv block 1: Conv(1,32,5) → ReLU → Conv(32,32,5) → ReLU → BN → MaxPool
        Conv2dCache cc1 = conv2d_forward(ctx, x, conv1_w, conv1_b,
                                          BS, 1, 28, 28, 32, 5, 5);
        // cc1.out: [BS,32,24,24]
        u32 r1 = relu_forward(ctx, cc1.out_id, BS*32*24*24);

        Conv2dCache cc2 = conv2d_forward(ctx, r1, conv2_w, conv2_b,
                                          BS, 32, 24, 24, 32, 5, 5);
        // cc2.out: [BS,32,20,20]
        u32 r2 = relu_forward(ctx, cc2.out_id, BS*32*20*20);

        BatchNormCache bn1c = batchnorm_forward(ctx, r2, bn1_gamma, bn1_beta,
                                                 bn1_rmean, bn1_rvar,
                                                 BS, 32, 20, 20, 1);
        // bn1c.out: [BS,32,20,20]
        MaxPool2dCache mp1 = maxpool2d_forward(ctx, bn1c.out_id, BS, 32, 20, 20);
        // mp1.out: [BS,32,10,10]

        // Conv block 2: Conv(32,64,3) → ReLU → Conv(64,64,3) → ReLU → BN → MaxPool
        Conv2dCache cc3 = conv2d_forward(ctx, mp1.out_id, conv3_w, conv3_b,
                                          BS, 32, 10, 10, 64, 3, 3);
        // cc3.out: [BS,64,8,8]
        u32 r3 = relu_forward(ctx, cc3.out_id, BS*64*8*8);

        Conv2dCache cc4 = conv2d_forward(ctx, r3, conv4_w, conv4_b,
                                          BS, 64, 8, 8, 64, 3, 3);
        // cc4.out: [BS,64,6,6]
        u32 r4 = relu_forward(ctx, cc4.out_id, BS*64*6*6);

        BatchNormCache bn2c = batchnorm_forward(ctx, r4, bn2_gamma, bn2_beta,
                                                 bn2_rmean, bn2_rvar,
                                                 BS, 64, 6, 6, 1);
        // bn2c.out: [BS,64,6,6]
        MaxPool2dCache mp2 = maxpool2d_forward(ctx, bn2c.out_id, BS, 64, 6, 6);
        // mp2.out: [BS,64,3,3]

        // Flatten: [BS,64,3,3] → [BS, 576]
        // No actual op needed — just reinterpret the buffer
        u32 flat = mp2.out_id;
        ctx->tensors[flat].view.shape = SHAPE(BS, 576);
        ctx->tensors[flat].view.strides[0] = 576;
        ctx->tensors[flat].view.strides[1] = 1;

        // Linear: [BS,576] → [BS,10]
        u32 logits = linear_forward(ctx, flat, fc_w, fc_b, BS, 576, 10);

        // ===== Loss =====
        CrossEntropyResult ce = cross_entropy_forward(ctx, logits, batch_y, BS, 10);

        // ===== Backward pass =====
        // d_logits = (softmax - onehot) / B
        u32 d_logits = cross_entropy_backward(ctx, &ce, batch_y);

        // Linear backward
        LinearGrads lg = linear_backward(ctx, d_logits, flat, fc_w, BS, 576, 10);
        u32 grad_ids[N_PARAMS]; // will fill as we go
        grad_ids[12] = lg.d_weight;  // fc_w grad
        grad_ids[13] = lg.d_bias;    // fc_b grad

        // Unflatten gradient: [BS,576] → [BS,64,3,3]
        u32 d_flat = lg.d_input;
        ctx->tensors[d_flat].view.shape = (Shape){.dims={BS,64,3,3}, .rank=4};
        ctx->tensors[d_flat].view.strides[0] = 576;
        ctx->tensors[d_flat].view.strides[1] = 9;
        ctx->tensors[d_flat].view.strides[2] = 3;
        ctx->tensors[d_flat].view.strides[3] = 1;

        // MaxPool2 backward
        u32 d_mp2 = maxpool2d_backward(ctx, d_flat, &mp2);

        // BN2 backward
        u32 d_bn2, d_bn2_gamma, d_bn2_beta;
        batchnorm_backward(ctx, d_mp2, &bn2c, bn2_gamma, &d_bn2, &d_bn2_gamma, &d_bn2_beta);
        grad_ids[10] = d_bn2_gamma;
        grad_ids[11] = d_bn2_beta;

        // ReLU4 backward
        u32 d_r4 = relu_backward(ctx, d_bn2, cc4.out_id, BS*64*6*6);

        // Conv4 backward
        Conv2dGrads cg4 = conv2d_backward(ctx, d_r4, &cc4, conv4_w);
        grad_ids[8] = cg4.d_weight;
        grad_ids[9] = cg4.d_bias;

        // ReLU3 backward
        u32 d_r3 = relu_backward(ctx, cg4.d_input, cc3.out_id, BS*64*8*8);

        // Conv3 backward
        Conv2dGrads cg3 = conv2d_backward(ctx, d_r3, &cc3, conv3_w);
        grad_ids[6] = cg3.d_weight;
        grad_ids[7] = cg3.d_bias;

        // MaxPool1 backward
        u32 d_mp1 = maxpool2d_backward(ctx, cg3.d_input, &mp1);

        // BN1 backward
        u32 d_bn1, d_bn1_gamma, d_bn1_beta;
        batchnorm_backward(ctx, d_mp1, &bn1c, bn1_gamma, &d_bn1, &d_bn1_gamma, &d_bn1_beta);
        grad_ids[4] = d_bn1_gamma;
        grad_ids[5] = d_bn1_beta;

        // ReLU2 backward
        u32 d_r2 = relu_backward(ctx, d_bn1, cc2.out_id, BS*32*20*20);

        // Conv2 backward
        Conv2dGrads cg2 = conv2d_backward(ctx, d_r2, &cc2, conv2_w);
        grad_ids[2] = cg2.d_weight;
        grad_ids[3] = cg2.d_bias;

        // ReLU1 backward
        u32 d_r1 = relu_backward(ctx, cg2.d_input, cc1.out_id, BS*32*24*24);

        // Conv1 backward
        Conv2dGrads cg1 = conv2d_backward(ctx, d_r1, &cc1, conv1_w);
        grad_ids[0] = cg1.d_weight;
        grad_ids[1] = cg1.d_bias;

        // ===== Adam step =====
        adam_step(ctx, &opt, grad_ids);

        clock_t t1 = clock();
        f32 ms = 1000.0f * (f32)(t1 - t0) / (f32)CLOCKS_PER_SEC;

        if (step % 10 == 0 || step == n_steps - 1) {
            printf("  step %2u/%u  loss=%.4f  (%.0fms)\n", step, n_steps, ce.loss, ms);
        }

        // Reset ephemeral tensors
        thvm_reset(ctx, n_weights);
        free(batch_x); free(batch_y);
    }

    // ========== Test accuracy ==========
    printf("\n  Evaluating test set...\n");
    u32 test_correct = 0;
    u32 test_bs = 64;
    u32 test_batches = n_test / test_bs;

    for (u32 b = 0; b < test_batches; b++) {
        u32 off = b * test_bs;
        // Reshape images to [BS,1,28,28]
        f32 *bx = malloc(test_bs * 784 * sizeof(f32));
        memcpy(bx, &test_images[off * 784], test_bs * 784 * sizeof(f32));
        u32 x = tensor_from_data(ctx, bx, (Shape){.dims={test_bs,1,28,28}, .rank=4});

        // Forward (eval mode — BN uses running stats)
        Conv2dCache cc1 = conv2d_forward(ctx, x, conv1_w, conv1_b,
                                          test_bs, 1, 28, 28, 32, 5, 5);
        u32 r1 = relu_forward(ctx, cc1.out_id, test_bs*32*24*24);
        Conv2dCache cc2 = conv2d_forward(ctx, r1, conv2_w, conv2_b,
                                          test_bs, 32, 24, 24, 32, 5, 5);
        u32 r2 = relu_forward(ctx, cc2.out_id, test_bs*32*20*20);
        BatchNormCache bn1c = batchnorm_forward(ctx, r2, bn1_gamma, bn1_beta,
                                                 bn1_rmean, bn1_rvar,
                                                 test_bs, 32, 20, 20, 0);  // eval mode
        MaxPool2dCache mp1 = maxpool2d_forward(ctx, bn1c.out_id, test_bs, 32, 20, 20);

        Conv2dCache cc3 = conv2d_forward(ctx, mp1.out_id, conv3_w, conv3_b,
                                          test_bs, 32, 10, 10, 64, 3, 3);
        u32 r3 = relu_forward(ctx, cc3.out_id, test_bs*64*8*8);
        Conv2dCache cc4 = conv2d_forward(ctx, r3, conv4_w, conv4_b,
                                          test_bs, 64, 8, 8, 64, 3, 3);
        u32 r4 = relu_forward(ctx, cc4.out_id, test_bs*64*6*6);
        BatchNormCache bn2c = batchnorm_forward(ctx, r4, bn2_gamma, bn2_beta,
                                                 bn2_rmean, bn2_rvar,
                                                 test_bs, 64, 6, 6, 0);  // eval mode
        MaxPool2dCache mp2 = maxpool2d_forward(ctx, bn2c.out_id, test_bs, 64, 6, 6);

        u32 flat = mp2.out_id;
        ctx->tensors[flat].view.shape = SHAPE(test_bs, 576);
        ctx->tensors[flat].view.strides[0] = 576;
        ctx->tensors[flat].view.strides[1] = 1;

        u32 logits = linear_forward(ctx, flat, fc_w, fc_b, test_bs, 576, 10);
        f32 *out = buf_read_f32(ctx, logits, test_bs * 10);

        for (u32 i = 0; i < test_bs; i++) {
            u32 pred = 0;
            f32 mv = out[i * 10];
            for (u32 j = 1; j < 10; j++)
                if (out[i * 10 + j] > mv) { mv = out[i * 10 + j]; pred = j; }
            if (pred == test_labels[off + i]) test_correct++;
        }
        free(out);
        // BN cache cleanup (eval mode doesn't alloc x_hat/inv_std that need freeing)
        free(bn1c.x_hat); free(bn1c.inv_std);
        free(bn2c.x_hat); free(bn2c.inv_std);
        thvm_reset(ctx, n_weights);
        free(bx);
    }

    f32 test_acc = 100.0f * (f32)test_correct / (f32)(test_batches * test_bs);
    printf("\n  Test accuracy: %.1f%% (%u/%u)\n", test_acc, test_correct, test_batches * test_bs);

    int pass = test_acc > 95.0f;  // tinygrad targets 99%, we'll accept 95% initially
    printf("\n  %s: CNN test accuracy %s 95%%\n", pass ? "PASS" : "FAIL",
           pass ? ">" : "<");

    // Cleanup
    adam_free(&opt);
    thvm_free(ctx);
    free(train_images); free(train_labels);
    free(test_images);  free(test_labels);
    return pass ? 0 : 1;
}
