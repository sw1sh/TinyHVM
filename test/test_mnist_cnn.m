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
  #define DEVICE "metal"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// ============================================================
// MNIST loader (IDX format)
// ============================================================

static u32 read_u32_be(FILE *f) {
    u8 buf[4]; fread(buf, 1, 4, f);
    return ((u32)buf[0]<<24) | ((u32)buf[1]<<16) | ((u32)buf[2]<<8) | buf[3];
}

static f32 *load_mnist_images(const char *path, u32 *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); exit(1); }
    read_u32_be(f); *n = read_u32_be(f);
    u32 rows = read_u32_be(f), cols = read_u32_be(f);
    u32 sz = *n * rows * cols;
    u8 *raw = malloc(sz);
    fread(raw, 1, sz, f); fclose(f);
    f32 *out = malloc(sz * sizeof(f32));
    for (u32 i = 0; i < sz; i++) out[i] = (f32)raw[i] / 255.0f;
    free(raw);
    return out;
}

static u8 *load_mnist_labels(const char *path, u32 *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); exit(1); }
    read_u32_be(f); *n = read_u32_be(f);
    u8 *out = malloc(*n);
    fread(out, 1, *n, f); fclose(f);
    return out;
}

// Xavier initialization
static void xavier_init(f32 *data, u32 fan_in, u32 fan_out, u32 n) {
    f32 scale = sqrtf(2.0f / (f32)(fan_in + fan_out));
    for (u32 i = 0; i < n; i++)
        data[i] = scale * ((f32)rand() / (f32)RAND_MAX * 2.0f - 1.0f);
}

// ============================================================
// Forward pass (shared between train and eval)
// ============================================================

// Holds all intermediate results needed for backward
typedef struct {
    ConvResult  cc1, cc2, cc3, cc4;
    BNResult    bn1c, bn2c;
    PoolResult  mp1, mp2;
    u32 r1, r2, r3, r4;   // ReLU outputs (tensor IDs)
    u32 flat, logits;
} ForwardCache;

static ForwardCache forward_pass(TinyHVM *ctx,
    u32 x, u32 BS,
    u32 conv1_w, u32 conv1_b, u32 conv2_w, u32 conv2_b,
    u32 bn1_gamma, u32 bn1_beta, u32 bn1_rmean, u32 bn1_rvar,
    u32 conv3_w, u32 conv3_b, u32 conv4_w, u32 conv4_b,
    u32 bn2_gamma, u32 bn2_beta, u32 bn2_rmean, u32 bn2_rvar,
    u32 fc_w, u32 fc_b, int training) {

    ForwardCache fc = {0};

    // Conv block 1: Conv(1,32,5) → ReLU → Conv(32,32,5) → ReLU → BN → MaxPool
    fc.cc1 = conv2d(ctx, x, conv1_w, conv1_b, BS, 1, 28, 28, 32, 5, 5);
    fc.r1 = relu_backward(ctx, fc.cc1.out_id, fc.cc1.out_id, BS*32*24*24); // relu_forward = relu_backward pattern with self
    // Actually let's just use the existing relu which returns a Term
    // We need a simpler relu that takes/returns tensor IDs
    {
        Term r = thvm_op(ctx, UOP_RELU, term_ten(fc.cc1.out_id, DTYPE_F32), term_era());
        r = thvm_reduce(ctx, r);
        fc.r1 = (u32)term_val(r);
    }

    fc.cc2 = conv2d(ctx, fc.r1, conv2_w, conv2_b, BS, 32, 24, 24, 32, 5, 5);
    {
        Term r = thvm_op(ctx, UOP_RELU, term_ten(fc.cc2.out_id, DTYPE_F32), term_era());
        r = thvm_reduce(ctx, r);
        fc.r2 = (u32)term_val(r);
    }

    fc.bn1c = batchnorm(ctx, fc.r2, bn1_gamma, bn1_beta, bn1_rmean, bn1_rvar,
                        BS, 32, 20, 20, training);
    fc.mp1 = maxpool2d(ctx, fc.bn1c.out_id, BS, 32, 20, 20);

    // Conv block 2: Conv(32,64,3) → ReLU → Conv(64,64,3) → ReLU → BN → MaxPool
    fc.cc3 = conv2d(ctx, fc.mp1.out_id, conv3_w, conv3_b, BS, 32, 10, 10, 64, 3, 3);
    {
        Term r = thvm_op(ctx, UOP_RELU, term_ten(fc.cc3.out_id, DTYPE_F32), term_era());
        r = thvm_reduce(ctx, r);
        fc.r3 = (u32)term_val(r);
    }

    fc.cc4 = conv2d(ctx, fc.r3, conv4_w, conv4_b, BS, 64, 8, 8, 64, 3, 3);
    {
        Term r = thvm_op(ctx, UOP_RELU, term_ten(fc.cc4.out_id, DTYPE_F32), term_era());
        r = thvm_reduce(ctx, r);
        fc.r4 = (u32)term_val(r);
    }

    fc.bn2c = batchnorm(ctx, fc.r4, bn2_gamma, bn2_beta, bn2_rmean, bn2_rvar,
                        BS, 64, 6, 6, training);
    fc.mp2 = maxpool2d(ctx, fc.bn2c.out_id, BS, 64, 6, 6);

    // Flatten: [BS,64,3,3] → [BS,576]
    fc.flat = fc.mp2.out_id;
    ctx->tensors[fc.flat].view.shape = SHAPE(BS, 576);
    ctx->tensors[fc.flat].view.strides[0] = 576;
    ctx->tensors[fc.flat].view.strides[1] = 1;

    // Linear: [BS,576] → [BS,10]
    Term logits_t = linear(ctx, term_ten(fc.flat, DTYPE_F32),
                           term_ten(fc_w, DTYPE_F32), term_ten(fc_b, DTYPE_F32),
                           BS, 576, 10);
    logits_t = thvm_reduce(ctx, logits_t);
    fc.logits = (u32)term_val(logits_t);

    return fc;
}

// ============================================================
// Main
// ============================================================

int main(void) {
    srand(42);
    printf("=== CNN MNIST (beautiful_mnist) ===\n\n");

    // Load data
    u32 n_train, n_test;
    f32 *train_images = load_mnist_images("data/train-images-idx3-ubyte", &n_train);
    u8  *train_labels = load_mnist_labels("data/train-labels-idx1-ubyte", &n_train);
    f32 *test_images  = load_mnist_images("data/t10k-images-idx3-ubyte", &n_test);
    u8  *test_labels  = load_mnist_labels("data/t10k-labels-idx1-ubyte", &n_test);
    printf("  Train: %u, Test: %u\n\n", n_train, n_test);

    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    // ========== Weight initialization ==========
    // Conv1: (32, 1, 5, 5)
    u32 c1n = 32*1*5*5;
    f32 *c1w = malloc(c1n * sizeof(f32)); xavier_init(c1w, 25, 32, c1n);
    u32 conv1_w = tensor_from(ctx, c1w, (Shape){.dims={32,1,5,5}, .rank=4});
    f32 *c1b = calloc(32, sizeof(f32));
    u32 conv1_b = tensor_from(ctx, c1b, SHAPE(32));
    free(c1w); free(c1b);

    // Conv2: (32, 32, 5, 5)
    u32 c2n = 32*32*5*5;
    f32 *c2w = malloc(c2n * sizeof(f32)); xavier_init(c2w, 32*25, 32, c2n);
    u32 conv2_w = tensor_from(ctx, c2w, (Shape){.dims={32,32,5,5}, .rank=4});
    f32 *c2b = calloc(32, sizeof(f32));
    u32 conv2_b = tensor_from(ctx, c2b, SHAPE(32));
    free(c2w); free(c2b);

    // BN1: gamma=1, beta=0, running_mean=0, running_var=1
    f32 bn1g[32], bn1b_d[32], bn1rm[32], bn1rv[32];
    for (u32 i = 0; i < 32; i++) { bn1g[i]=1; bn1b_d[i]=0; bn1rm[i]=0; bn1rv[i]=1; }
    u32 bn1_gamma = tensor_from(ctx, bn1g, SHAPE(32));
    u32 bn1_beta  = tensor_from(ctx, bn1b_d, SHAPE(32));
    u32 bn1_rmean = tensor_from(ctx, bn1rm, SHAPE(32));
    u32 bn1_rvar  = tensor_from(ctx, bn1rv, SHAPE(32));

    // Conv3: (64, 32, 3, 3)
    u32 c3n = 64*32*3*3;
    f32 *c3w = malloc(c3n * sizeof(f32)); xavier_init(c3w, 32*9, 64, c3n);
    u32 conv3_w = tensor_from(ctx, c3w, (Shape){.dims={64,32,3,3}, .rank=4});
    f32 *c3b = calloc(64, sizeof(f32));
    u32 conv3_b = tensor_from(ctx, c3b, SHAPE(64));
    free(c3w); free(c3b);

    // Conv4: (64, 64, 3, 3)
    u32 c4n = 64*64*3*3;
    f32 *c4w = malloc(c4n * sizeof(f32)); xavier_init(c4w, 64*9, 64, c4n);
    u32 conv4_w = tensor_from(ctx, c4w, (Shape){.dims={64,64,3,3}, .rank=4});
    f32 *c4b = calloc(64, sizeof(f32));
    u32 conv4_b = tensor_from(ctx, c4b, SHAPE(64));
    free(c4w); free(c4b);

    // BN2
    f32 bn2g[64], bn2b_d[64], bn2rm[64], bn2rv[64];
    for (u32 i = 0; i < 64; i++) { bn2g[i]=1; bn2b_d[i]=0; bn2rm[i]=0; bn2rv[i]=1; }
    u32 bn2_gamma = tensor_from(ctx, bn2g, SHAPE(64));
    u32 bn2_beta  = tensor_from(ctx, bn2b_d, SHAPE(64));
    u32 bn2_rmean = tensor_from(ctx, bn2rm, SHAPE(64));
    u32 bn2_rvar  = tensor_from(ctx, bn2rv, SHAPE(64));

    // Linear: (576, 10)
    u32 fcn = 576*10;
    f32 *fcw = malloc(fcn * sizeof(f32)); xavier_init(fcw, 576, 10, fcn);
    u32 fc_w = tensor_from(ctx, fcw, SHAPE(576, 10));
    f32 *fcb = calloc(10, sizeof(f32));
    u32 fc_b = tensor_from(ctx, fcb, SHAPE(10));
    free(fcw); free(fcb);

    // ========== Adam optimizer ==========
    #define N_PARAMS 14
    Adam opt = adam_init(ctx, 0.001f, N_PARAMS);
    u32 param_ids[] = {conv1_w, conv1_b, conv2_w, conv2_b, bn1_gamma, bn1_beta,
                       conv3_w, conv3_b, conv4_w, conv4_b, bn2_gamma, bn2_beta,
                       fc_w, fc_b};
    u32 param_sizes[] = {32*1*5*5, 32, 32*32*5*5, 32, 32, 32,
                         64*32*3*3, 64, 64*64*3*3, 64, 64, 64,
                         576*10, 10};
    for (u32 i = 0; i < N_PARAMS; i++)
        adam_add_param(ctx, &opt, i, param_ids[i], param_sizes[i]);

    // IMPORTANT: n_weights must be set AFTER adam_add_param allocates
    // GPU m/v buffers, so they survive thvm_reset(n_weights)
    u32 n_weights = ctx->tensor_count;

    // ========== Training loop ==========
    u32 BS = 128;
    u32 n_steps = 70;
    f32 lr_max = 0.001f, lr_min = 0.0001f;

    printf("  Training %u steps, BS=%u...\n\n", n_steps, BS);

    // Profiling accumulators (nanoseconds)
    #include <mach/mach_time.h>
    mach_timebase_info_data_t tbi;
    mach_timebase_info(&tbi);
    #define NS(t) ((t) * tbi.numer / tbi.denom)

    u64 prof_batch = 0, prof_fwd = 0, prof_loss = 0;
    u64 prof_bwd_linear = 0, prof_bwd_pool_bn = 0, prof_bwd_conv = 0;
    u64 prof_adam = 0, prof_reset = 0;

    #define PROF_START() u64 _ps = mach_absolute_time()
    #define PROF_ACC(acc) do { acc += mach_absolute_time() - _ps; _ps = mach_absolute_time(); } while(0)

    for (u32 step = 0; step < n_steps; step++) {
        f32 progress = (f32)step / (f32)n_steps;
        opt.lr = lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf(3.14159f * progress));
        clock_t t0 = clock();
        PROF_START();

        // Random batch
        f32 *batch_x = malloc(BS * 784 * sizeof(f32));
        u8  *batch_y = malloc(BS);
        for (u32 i = 0; i < BS; i++) {
            u32 idx = (u32)(rand() % (int)n_train);
            memcpy(&batch_x[i*784], &train_images[idx*784], 784*sizeof(f32));
            batch_y[i] = train_labels[idx];
        }
        u32 x = tensor_from(ctx, batch_x, (Shape){.dims={BS,1,28,28}, .rank=4});
        PROF_ACC(prof_batch);

        // Start batch — all compute dispatches accumulate until flush
        ctx->backend->begin_batch();
        ForwardCache fc = forward_pass(ctx, x, BS,
            conv1_w, conv1_b, conv2_w, conv2_b,
            bn1_gamma, bn1_beta, bn1_rmean, bn1_rvar,
            conv3_w, conv3_b, conv4_w, conv4_b,
            bn2_gamma, bn2_beta, bn2_rmean, bn2_rvar,
            fc_w, fc_b, 1);
        PROF_ACC(prof_fwd);

        // Loss
        CEResult ce = cross_entropy(ctx, term_ten(fc.logits, DTYPE_F32), batch_y, BS, 10);
        PROF_ACC(prof_loss);

        // Backward
        Term d_logits = cross_entropy_backward(ctx, &ce, batch_y);
        d_logits = thvm_reduce(ctx, d_logits);
        u32 d_logits_id = (u32)term_val(d_logits);

        // Linear backward
        LinearGrads lg = linear_backward(ctx, d_logits_id, fc.flat, fc_w, BS, 576, 10);
        u32 grad_ids[N_PARAMS];
        grad_ids[12] = lg.d_weight;
        grad_ids[13] = lg.d_bias;

        // Unflatten
        u32 d_flat = lg.d_input;
        ctx->tensors[d_flat].view.shape = (Shape){.dims={BS,64,3,3}, .rank=4};
        ctx->tensors[d_flat].view.strides[0] = 576;
        ctx->tensors[d_flat].view.strides[1] = 9;
        ctx->tensors[d_flat].view.strides[2] = 3;
        ctx->tensors[d_flat].view.strides[3] = 1;
        PROF_ACC(prof_bwd_linear);

        // MaxPool2 backward
        u32 d_mp2 = maxpool2d_backward(ctx, d_flat, &fc.mp2);
        // BN2 backward
        BNGrads bng2 = batchnorm_backward(ctx, d_mp2, &fc.bn2c, bn2_gamma);
        grad_ids[10] = bng2.d_gamma;
        grad_ids[11] = bng2.d_beta;
        PROF_ACC(prof_bwd_pool_bn);

        // ReLU4
        u32 d_r4 = relu_backward(ctx, bng2.d_input, fc.cc4.out_id, BS*64*6*6);
        // Conv4
        ConvGrads cg4 = conv2d_backward(ctx, d_r4, &fc.cc4, conv4_w);
        grad_ids[8] = cg4.d_weight;
        grad_ids[9] = cg4.d_bias;
        // ReLU3
        u32 d_r3 = relu_backward(ctx, cg4.d_input, fc.cc3.out_id, BS*64*8*8);
        // Conv3
        ConvGrads cg3 = conv2d_backward(ctx, d_r3, &fc.cc3, conv3_w);
        grad_ids[6] = cg3.d_weight;
        grad_ids[7] = cg3.d_bias;
        // MaxPool1
        u32 d_mp1 = maxpool2d_backward(ctx, cg3.d_input, &fc.mp1);
        // BN1
        BNGrads bng1 = batchnorm_backward(ctx, d_mp1, &fc.bn1c, bn1_gamma);
        grad_ids[4] = bng1.d_gamma;
        grad_ids[5] = bng1.d_beta;
        // ReLU2
        u32 d_r2 = relu_backward(ctx, bng1.d_input, fc.cc2.out_id, BS*32*20*20);
        // Conv2
        ConvGrads cg2 = conv2d_backward(ctx, d_r2, &fc.cc2, conv2_w);
        grad_ids[2] = cg2.d_weight;
        grad_ids[3] = cg2.d_bias;
        // ReLU1
        u32 d_r1 = relu_backward(ctx, cg2.d_input, fc.cc1.out_id, BS*32*24*24);
        // Conv1
        ConvGrads cg1 = conv2d_backward(ctx, d_r1, &fc.cc1, conv1_w);
        grad_ids[0] = cg1.d_weight;
        grad_ids[1] = cg1.d_bias;
        PROF_ACC(prof_bwd_conv);

        // Adam step
        adam_step(ctx, &opt, grad_ids);
        PROF_ACC(prof_adam);

        // Flush all pending compute before reset
        ctx->backend->end_batch();

        f32 ms = 1000.0f * (f32)(clock() - t0) / (f32)CLOCKS_PER_SEC;
        if (step % 10 == 0 || step == n_steps - 1)
            printf("  step %3u/%u  loss=%.4f  lr=%.5f  (%.0fms)\n", step, n_steps, ce.loss, opt.lr, ms);

        thvm_reset(ctx, n_weights);
        PROF_ACC(prof_reset);

        free(batch_x); free(batch_y);
    }

    // Print profiling breakdown
    f32 steps_f = (f32)n_steps;
    printf("\n  === Profile (avg per step) ===\n");
    printf("  batch_prep: %6.1f ms\n", (f32)NS(prof_batch) / 1e6f / steps_f);
    printf("  forward:    %6.1f ms\n", (f32)NS(prof_fwd) / 1e6f / steps_f);
    printf("  loss:       %6.1f ms\n", (f32)NS(prof_loss) / 1e6f / steps_f);
    printf("  bwd_linear: %6.1f ms\n", (f32)NS(prof_bwd_linear) / 1e6f / steps_f);
    printf("  bwd_bn+mp:  %6.1f ms\n", (f32)NS(prof_bwd_pool_bn) / 1e6f / steps_f);
    printf("  bwd_conv:   %6.1f ms\n", (f32)NS(prof_bwd_conv) / 1e6f / steps_f);
    printf("  adam:       %6.1f ms\n", (f32)NS(prof_adam) / 1e6f / steps_f);
    printf("  reset:      %6.1f ms\n", (f32)NS(prof_reset) / 1e6f / steps_f);
    f32 total = (f32)NS(prof_batch + prof_fwd + prof_loss + prof_bwd_linear + prof_bwd_pool_bn + prof_bwd_conv + prof_adam + prof_reset) / 1e6f / steps_f;
    printf("  TOTAL:      %6.1f ms\n", total);

    // ========== Test accuracy ==========
    printf("\n  Evaluating test set...\n");
    u32 test_correct = 0;
    u32 test_bs = 64;
    u32 test_batches = n_test / test_bs;

    for (u32 b = 0; b < test_batches; b++) {
        u32 off = b * test_bs;
        f32 *bx = malloc(test_bs * 784 * sizeof(f32));
        memcpy(bx, &test_images[off * 784], test_bs * 784 * sizeof(f32));
        u32 x = tensor_from(ctx, bx, (Shape){.dims={test_bs,1,28,28}, .rank=4});

        ForwardCache fc = forward_pass(ctx, x, test_bs,
            conv1_w, conv1_b, conv2_w, conv2_b,
            bn1_gamma, bn1_beta, bn1_rmean, bn1_rvar,
            conv3_w, conv3_b, conv4_w, conv4_b,
            bn2_gamma, bn2_beta, bn2_rmean, bn2_rvar,
            fc_w, fc_b, 0);  // eval mode

        f32 *out = buf_read(ctx, fc.logits, test_bs * 10);
        for (u32 i = 0; i < test_bs; i++) {
            u32 pred = 0;
            f32 mv = out[i * 10];
            for (u32 j = 1; j < 10; j++)
                if (out[i * 10 + j] > mv) { mv = out[i * 10 + j]; pred = j; }
            if (pred == test_labels[off + i]) test_correct++;
        }
        free(out);
        free(fc.bn1c.x_hat); free(fc.bn1c.inv_std);
        free(fc.bn2c.x_hat); free(fc.bn2c.inv_std);
        thvm_reset(ctx, n_weights);
        free(bx);
    }

    f32 test_acc = 100.0f * (f32)test_correct / (f32)(test_batches * test_bs);
    printf("\n  Test accuracy: %.1f%% (%u/%u)\n", test_acc, test_correct, test_batches * test_bs);

    int pass = test_acc > 90.0f;  // lower threshold for 70 steps
    printf("\n  %s: CNN test accuracy %s 90%%\n", pass ? "PASS" : "FAIL",
           pass ? ">" : "<");

    adam_free(&opt);
    thvm_free(ctx);
    free(train_images); free(train_labels);
    free(test_images);  free(test_labels);
    return pass ? 0 : 1;
}
