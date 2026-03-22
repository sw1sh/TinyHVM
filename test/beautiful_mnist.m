// beautiful_mnist.m — CNN matching tinygrad's beautiful_mnist.py
// Architecture: Conv(1,32,5)→ReLU→Conv(32,32,5)→ReLU→BN(32)→MaxPool
//             → Conv(32,64,3)→ReLU→Conv(64,64,3)→ReLU→BN(64)→MaxPool
//             → Flatten→Linear(576,10)
//
// Usage: ./beautiful_mnist

#include "../src/tinyhvm.c"
#include "../src/cpu.c"
#ifdef __APPLE__
  #include "../src/metal.m"
#endif
#include "../src/layers.c"
#include "../src/nn/datasets.c"
#include "../src/nn/profiler.h"

#ifndef DEVICE
  #define DEVICE "metal"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Xavier initialization
static void xavier_init(f32 *data, u32 fan_in, u32 fan_out, u32 n) {
    f32 scale = sqrtf(2.0f / (f32)(fan_in + fan_out));
    for (u32 i = 0; i < n; i++)
        data[i] = scale * ((f32)rand() / (f32)RAND_MAX * 2.0f - 1.0f);
}

// ============================================================
// Model weights
// ============================================================

typedef struct {
    u32 w, b;
} Conv2dLayer;

typedef struct {
    u32 gamma, beta, rmean, rvar;
} BNLayer;

typedef struct {
    u32 w, b;
} LinearLayer;

static Conv2dLayer conv_layer(TinyHVM *ctx, u32 ci, u32 co, u32 k) {
    u32 n = co * ci * k * k;
    f32 *w = malloc(n * sizeof(f32)); xavier_init(w, ci*k*k, co, n);
    u32 wt = tensor_from(ctx, w, (Shape){.dims={co,ci,k,k}, .rank=4});
    f32 *b = calloc(co, sizeof(f32));
    u32 bt = tensor_from(ctx, b, SHAPE(co));
    free(w); free(b);
    return (Conv2dLayer){wt, bt};
}

static BNLayer bn_layer(TinyHVM *ctx, u32 c) {
    f32 *g = malloc(c * sizeof(f32)), *b = calloc(c, sizeof(f32));
    f32 *rm = calloc(c, sizeof(f32)), *rv = malloc(c * sizeof(f32));
    for (u32 i = 0; i < c; i++) { g[i] = 1; rv[i] = 1; }
    BNLayer l = {
        tensor_from(ctx, g, SHAPE(c)), tensor_from(ctx, b, SHAPE(c)),
        tensor_from(ctx, rm, SHAPE(c)), tensor_from(ctx, rv, SHAPE(c)),
    };
    free(g); free(b); free(rm); free(rv);
    return l;
}

static LinearLayer linear_layer(TinyHVM *ctx, u32 in_f, u32 out_f) {
    u32 n = in_f * out_f;
    f32 *w = malloc(n * sizeof(f32)); xavier_init(w, in_f, out_f, n);
    u32 wt = tensor_from(ctx, w, SHAPE(in_f, out_f));
    f32 *b = calloc(out_f, sizeof(f32));
    u32 bt = tensor_from(ctx, b, SHAPE(out_f));
    free(w); free(b);
    return (LinearLayer){wt, bt};
}

// ============================================================
// Forward pass
// ============================================================

typedef struct {
    ConvResult cc1, cc2, cc3, cc4;
    BNResult   bn1c, bn2c;
    PoolResult mp1, mp2;
    u32 r1, r2, r3, r4, flat, logits;
} ForwardCache;

static u32 relu_fwd(TinyHVM *ctx, u32 x) {
    Term r = thvm_op(ctx, UOP_RELU, term_ten(x, DTYPE_F32), term_era());
    return (u32)term_val(thvm_reduce(ctx, r));
}

static ForwardCache forward(TinyHVM *ctx, u32 x, u32 BS,
    Conv2dLayer c1, Conv2dLayer c2, BNLayer bn1,
    Conv2dLayer c3, Conv2dLayer c4, BNLayer bn2,
    LinearLayer fc, int training) {

    ForwardCache f = {0};

    // Block 1: Conv→ReLU→Conv→ReLU→BN→MaxPool
    f.cc1 = conv2d(ctx, x, c1.w, c1.b, BS, 1, 28, 28, 32, 5, 5);
    f.r1 = relu_fwd(ctx, f.cc1.out_id);
    f.cc2 = conv2d(ctx, f.r1, c2.w, c2.b, BS, 32, 24, 24, 32, 5, 5);
    f.r2 = relu_fwd(ctx, f.cc2.out_id);
    f.bn1c = batchnorm(ctx, f.r2, bn1.gamma, bn1.beta, bn1.rmean, bn1.rvar,
                        BS, 32, 20, 20, training);
    f.mp1 = maxpool2d(ctx, f.bn1c.out_id, BS, 32, 20, 20);

    // Block 2: Conv→ReLU→Conv→ReLU→BN→MaxPool
    f.cc3 = conv2d(ctx, f.mp1.out_id, c3.w, c3.b, BS, 32, 10, 10, 64, 3, 3);
    f.r3 = relu_fwd(ctx, f.cc3.out_id);
    f.cc4 = conv2d(ctx, f.r3, c4.w, c4.b, BS, 64, 8, 8, 64, 3, 3);
    f.r4 = relu_fwd(ctx, f.cc4.out_id);
    f.bn2c = batchnorm(ctx, f.r4, bn2.gamma, bn2.beta, bn2.rmean, bn2.rvar,
                        BS, 64, 6, 6, training);
    f.mp2 = maxpool2d(ctx, f.bn2c.out_id, BS, 64, 6, 6);

    // Flatten → Linear
    f.flat = f.mp2.out_id;
    ctx->tensors[f.flat].view = view_create(SHAPE(BS, 576));
    Term logits = linear(ctx, term_ten(f.flat, DTYPE_F32),
                         term_ten(fc.w, DTYPE_F32), term_ten(fc.b, DTYPE_F32),
                         BS, 576, 10);
    f.logits = (u32)term_val(thvm_reduce(ctx, logits));
    return f;
}

// ============================================================
// Main
// ============================================================

int main(void) {
    srand(42);
    printf("=== beautiful_mnist (TinyHVM) ===\n\n");

    MNISTData data = mnist_load("data");
    printf("  Train: %u, Test: %u\n\n", data.n_train, data.n_test);

    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    // Model
    Conv2dLayer c1 = conv_layer(ctx, 1, 32, 5);
    Conv2dLayer c2 = conv_layer(ctx, 32, 32, 5);
    BNLayer bn1 = bn_layer(ctx, 32);
    Conv2dLayer c3 = conv_layer(ctx, 32, 64, 3);
    Conv2dLayer c4 = conv_layer(ctx, 64, 64, 3);
    BNLayer bn2 = bn_layer(ctx, 64);
    LinearLayer fc = linear_layer(ctx, 576, 10);

    // Optimizer
    #define N_PARAMS 14
    u32 param_ids[] = {c1.w, c1.b, c2.w, c2.b, bn1.gamma, bn1.beta,
                       c3.w, c3.b, c4.w, c4.b, bn2.gamma, bn2.beta,
                       fc.w, fc.b};
    u32 param_sizes[] = {32*1*5*5, 32, 32*32*5*5, 32, 32, 32,
                         64*32*3*3, 64, 64*64*3*3, 64, 64, 64,
                         576*10, 10};
    Adam opt = adam_init(ctx, 0.001f, N_PARAMS);
    for (u32 i = 0; i < N_PARAMS; i++)
        adam_add_param(ctx, &opt, i, param_ids[i], param_sizes[i]);
    u32 n_weights = ctx->tensor_count;

    // Train
    u32 BS = 128, n_steps = 70;
    f32 lr_max = 0.001f, lr_min = 0.0001f;
    prof_init();
    printf("  Training %u steps, BS=%u...\n\n", n_steps, BS);

    for (u32 step = 0; step < n_steps; step++) {
        f32 progress = (f32)step / (f32)n_steps;
        opt.lr = lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf(3.14159f * progress));
        clock_t t0 = clock();
        PROF_START();

        // Sample batch
        f32 *bx = malloc(BS * 784 * sizeof(f32));
        u8 *by = malloc(BS);
        for (u32 i = 0; i < BS; i++) {
            u32 idx = (u32)(rand() % (int)data.n_train);
            memcpy(&bx[i*784], &data.train_images[idx*784], 784*sizeof(f32));
            by[i] = data.train_labels[idx];
        }
        u32 x = tensor_from(ctx, bx, (Shape){.dims={BS,1,28,28}, .rank=4});
        PROF_ACC("batch");

        ctx->backend->begin_batch();

        // Forward + loss
        ForwardCache fwd = forward(ctx, x, BS, c1, c2, bn1, c3, c4, bn2, fc, 1);
        PROF_ACC("forward");
        CEResult ce = cross_entropy(ctx, term_ten(fwd.logits, DTYPE_F32), by, BS, 10);
        PROF_ACC("loss");

        // Backward
        u32 dl = (u32)term_val(thvm_reduce(ctx, cross_entropy_backward(ctx, &ce, by)));
        u32 grad_ids[N_PARAMS];

        LinearGrads lg = linear_backward(ctx, dl, fwd.flat, fc.w, BS, 576, 10);
        grad_ids[12] = lg.d_weight; grad_ids[13] = lg.d_bias;

        u32 df = lg.d_input;
        ctx->tensors[df].view.shape = (Shape){.dims={BS,64,3,3}, .rank=4};
        ctx->tensors[df].view.strides[0]=576; ctx->tensors[df].view.strides[1]=9;
        ctx->tensors[df].view.strides[2]=3;   ctx->tensors[df].view.strides[3]=1;
        PROF_ACC("bwd_linear");

        u32 d_mp2 = maxpool2d_backward(ctx, df, &fwd.mp2);
        BNGrads bg2 = batchnorm_backward(ctx, d_mp2, &fwd.bn2c, bn2.gamma);
        grad_ids[10] = bg2.d_gamma; grad_ids[11] = bg2.d_beta;

        u32 dr4 = relu_backward(ctx, bg2.d_input, fwd.cc4.out_id, BS*64*6*6);
        ConvGrads g4 = conv2d_backward(ctx, dr4, &fwd.cc4, c4.w);
        grad_ids[8] = g4.d_weight; grad_ids[9] = g4.d_bias;
        u32 dr3 = relu_backward(ctx, g4.d_input, fwd.cc3.out_id, BS*64*8*8);
        ConvGrads g3 = conv2d_backward(ctx, dr3, &fwd.cc3, c3.w);
        grad_ids[6] = g3.d_weight; grad_ids[7] = g3.d_bias;

        u32 d_mp1 = maxpool2d_backward(ctx, g3.d_input, &fwd.mp1);
        BNGrads bg1 = batchnorm_backward(ctx, d_mp1, &fwd.bn1c, bn1.gamma);
        grad_ids[4] = bg1.d_gamma; grad_ids[5] = bg1.d_beta;

        u32 dr2 = relu_backward(ctx, bg1.d_input, fwd.cc2.out_id, BS*32*20*20);
        ConvGrads g2 = conv2d_backward(ctx, dr2, &fwd.cc2, c2.w);
        grad_ids[2] = g2.d_weight; grad_ids[3] = g2.d_bias;
        u32 dr1 = relu_backward(ctx, g2.d_input, fwd.cc1.out_id, BS*32*24*24);
        ConvGrads g1 = conv2d_backward(ctx, dr1, &fwd.cc1, c1.w);
        grad_ids[0] = g1.d_weight; grad_ids[1] = g1.d_bias;
        PROF_ACC("bwd_conv");

        adam_step(ctx, &opt, grad_ids);
        PROF_ACC("adam");

        ctx->backend->end_batch();

        f32 ms = 1000.0f * (f32)(clock() - t0) / (f32)CLOCKS_PER_SEC;
        if (step % 10 == 0 || step == n_steps - 1)
            printf("  step %3u/%u  loss=%.4f  lr=%.5f  (%.0fms)\n",
                   step, n_steps, ce.loss, opt.lr, ms);

        thvm_reset(ctx, n_weights);
        PROF_ACC("reset");
        PROF_STEP();
        free(bx); free(by);
    }
    prof_report();

    // Eval
    printf("\n  Evaluating test set...\n");
    u32 correct = 0, tbs = 64, tb = data.n_test / tbs;
    for (u32 b = 0; b < tb; b++) {
        f32 *bx = malloc(tbs * 784 * sizeof(f32));
        memcpy(bx, &data.test_images[b*tbs*784], tbs * 784 * sizeof(f32));
        u32 x = tensor_from(ctx, bx, (Shape){.dims={tbs,1,28,28}, .rank=4});
        ForwardCache fwd = forward(ctx, x, tbs, c1, c2, bn1, c3, c4, bn2, fc, 0);
        f32 *out = buf_read(ctx, fwd.logits, tbs * 10);
        for (u32 i = 0; i < tbs; i++) {
            u32 pred = 0; f32 mv = out[i*10];
            for (u32 j = 1; j < 10; j++)
                if (out[i*10+j] > mv) { mv = out[i*10+j]; pred = j; }
            if (pred == data.test_labels[b*tbs+i]) correct++;
        }
        free(out);
        free(fwd.bn1c.x_hat); free(fwd.bn1c.inv_std);
        free(fwd.bn2c.x_hat); free(fwd.bn2c.inv_std);
        thvm_reset(ctx, n_weights); free(bx);
    }
    f32 acc = 100.0f * (f32)correct / (f32)(tb * tbs);
    printf("\n  Test accuracy: %.1f%% (%u/%u)\n", acc, correct, tb * tbs);
    printf("\n  %s: CNN test accuracy %s 90%%\n", acc > 90 ? "PASS" : "FAIL",
           acc > 90 ? ">" : "<");

    adam_free(&opt);
    mnist_free(&data);
    thvm_free(ctx);
    return acc > 90 ? 0 : 1;
}
