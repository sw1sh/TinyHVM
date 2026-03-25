// test_mnist.m — MNIST training (MLP or CNN)
//
// Usage:
//   ./test_mnist              # MLP, one-program inet, no C loop
//   ./test_mnist mlp          # same
//   ./test_mnist cnn          # CNN with IC autograd + Adam
//
// MLP: 784→128→10, ReLU, cross-entropy, SGD.
//   Entire epoch is ONE recursive inet program, ONE thvm_reduce().
//
// CNN: Conv(1,32,5)→ReLU→Conv(32,32,5)→ReLU→BN→MaxPool
//    → Conv(32,64,3)→ReLU→Conv(64,64,3)→ReLU→BN→MaxPool
//    → Flatten→Linear(576,10). Adam, cosine LR.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include "../src/nn/datasets.c"
#include "train_helpers.h"

#ifndef DEVICE
  #define DEVICE "metal"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Kaiming uniform init: uniform(-bound, bound) where bound = 1/sqrt(fan_in)
// Matches tinygrad's Conv2d/Linear default initialization.
static void kaiming_init(f32 *data, u32 fan_in, u32 n) {
    f32 bound = 1.0f / sqrtf((f32)fan_in);
    for (u32 i = 0; i < n; i++)
        data[i] = bound * ((f32)rand() / (f32)RAND_MAX * 2.0f - 1.0f);
}

static Term make_weight(TinyHVM *ctx, Shape s, u32 fan_in, u32 fan_out) {
    (void)fan_out;
    u32 n = 1;
    for (u32 i = 0; i < s.rank; i++) n *= s.dims[i];
    f32 *data = malloc(n * sizeof(f32));
    kaiming_init(data, fan_in, n);
    Term t = thvm_tensor(ctx, data, s);
    free(data);
    return t;
}

// Load weights from binary file (dumped from tinygrad for exact parity)
static f32 *load_weights_bin(const char *path, u32 *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    f32 *data = malloc((size_t)sz);
    fread(data, 1, (size_t)sz, f);
    fclose(f);
    *out_size = (u32)(sz / sizeof(f32));
    return data;
}

static Term make_zeros(TinyHVM *ctx, u32 n) {
    f32 *z = calloc(n, sizeof(f32));
    Term t = thvm_tensor(ctx, z, SHAPE(n));
    free(z);
    return t;
}

static Term make_ones(TinyHVM *ctx, u32 n) {
    f32 *o = malloc(n * sizeof(f32));
    for (u32 i = 0; i < n; i++) o[i] = 1.0f;
    Term t = thvm_tensor(ctx, o, SHAPE(n));
    free(o);
    return t;
}

static Term relu_fn(TinyHVM *ctx, Term x) {
    return thvm_op(ctx, UOP_RELU, x, term_era());
}

// ============================================================
// MLP: one recursive inet program, one reduce
// ============================================================

static int run_mlp(MNISTData *data) {
    printf("=== MNIST MLP — recursive inet (%s) ===\n", DEVICE);
    printf("  No C loop. ONE program, ONE reduce.\n\n");

    u32 BS = 128, H = 128;
    u32 n_batches = data->n_train / BS;
    f32 lr_val = 0.1f;

    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    // Weights
    f32 *w1d = malloc(784 * H * sizeof(f32)); kaiming_init(w1d, 784, 784 * H);
    f32 *b1d = calloc(H, sizeof(f32));
    f32 *w2d = malloc(H * 10 * sizeof(f32));  kaiming_init(w2d, H, H * 10);
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

    Term X_all = thvm_tensor(ctx, data->train_images, SHAPE(data->n_train, 784));
    Term Y_all;
    { // one-hot labels
        f32 *oh = calloc(data->n_train * 10, sizeof(f32));
        for (u32 i = 0; i < data->n_train; i++) oh[i * 10 + data->train_labels[i]] = 1.0f;
        Y_all = thvm_tensor(ctx, oh, SHAPE(data->n_train, 10));
        free(oh);
    }

    Term LR = thvm_tensor(ctx, &lr_val, SHAPE(1));
    f32 inv_n = 1.0f / (f32)BS;
    Term INV_N = thvm_tensor(ctx, &inv_n, SHAPE(1));
    f32 bs_f = (f32)BS;
    Term BS_ten = thvm_tensor(ctx, &bs_f, SHAPE(1));

    f32 ms[] = {1,0,0,0}, me[] = {0,1,0,0};
    f32 fx[] = {0,0,0,784}, fy[] = {0,0,0,10};
    Term MASK_S  = thvm_tensor(ctx, ms, SHAPE(4));
    Term MASK_E  = thvm_tensor(ctx, me, SHAPE(4));
    Term FIXED_X = thvm_tensor(ctx, fx, SHAPE(4));
    Term FIXED_Y = thvm_tensor(ctx, fy, SHAPE(4));

    u32 n_weights = ctx->tensor_count;

    // ONE program, ONE reduce
    f32 sb = 0.0f;
    Term SB_ten = thvm_tensor(ctx, &sb, SHAPE(1));
    f32 ns = (f32)n_batches;
    Term NS_ten = thvm_tensor(ctx, &ns, SHAPE(1));

    Term program = mnist_train_program(ctx,
        W1, B1, W2, B2, X_all, Y_all, LR, INV_N,
        MASK_S, MASK_E, FIXED_X, MASK_S, MASK_E, FIXED_Y,
        BS_ten, NS_ten, SB_ten, BS, H, (int)n_batches);

    printf("  reducing %u-step inet (1 epoch, BS=%u)...\n", n_batches, BS);
    @autoreleasepool {
        thvm_reduce(ctx, program);
    }
    printf("  done — itrs=%llu\n", (unsigned long long)ctx->itrs);
    printf("  peak: %u tensors, %llu heap slots\n\n",
           ctx->tensor_count, (unsigned long long)ctx->heap_pos);

    // Eval
    printf("  Evaluating test set...\n");
    u32 correct = 0, test_bs = BS, tb = data->n_test / test_bs;
    for (u32 b = 0; b < tb; b++) {
        u32 off = b * test_bs;
        Term X = thvm_tensor(ctx, &data->test_images[off * 784], SHAPE(test_bs, 784));
        Term z1 = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, X, W1), B1);
        Term h  = thvm_op(ctx, UOP_RELU, z1, term_era());
        Term out = thvm_reduce(ctx, thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, W2), B2));
        f32 *od = thvm_to_host(ctx, out);
        for (u32 i = 0; i < test_bs; i++) {
            u32 pred = 0; f32 mx = od[i * 10];
            for (u32 j = 1; j < 10; j++)
                if (od[i * 10 + j] > mx) { mx = od[i * 10 + j]; pred = j; }
            if (pred == data->test_labels[off + i]) correct++;
        }
        thvm_reset(ctx, n_weights);
    }

    f32 acc = 100.0f * (f32)correct / (f32)(tb * test_bs);
    printf("\n  Test accuracy: %.1f%% (%u/%u)\n", acc, correct, tb * test_bs);
    printf("  %s: MLP accuracy %s 90%%\n", acc > 90 ? "PASS" : "FAIL", acc > 90 ? ">" : "<");

    thvm_free(ctx);
    return acc > 90 ? 0 : 1;
}

// ============================================================
// CNN: IC autograd + Adam, per-step C loop (conv needs reset)
// ============================================================

static int run_cnn(MNISTData *data) {
    printf("=== MNIST CNN — IC autograd (%s) ===\n\n", DEVICE);

    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    // Try to load tinygrad-dumped weights for exact parity
    u32 wsize = 0;
    f32 *wbin = load_weights_bin("/tmp/tinygrad_weights/init_thvm.bin", &wsize);
    f32 *wp = wbin;  // cursor

    // Helper: load next n floats from binary or use fallback
    #define LOAD_OR_INIT(shape, fan_in, fan_out) \
        (wp ? ({ u32 _n = 1; for (u32 _i=0;_i<(shape).rank;_i++) _n*=(shape).dims[_i]; \
                 Term _t = thvm_tensor(ctx, wp, (shape)); wp += _n; _t; }) \
            : make_weight(ctx, (shape), (fan_in), (fan_out)))
    #define LOAD_OR_ZEROS(n) \
        (wp ? ({ f32 *_p = wp; wp += (n); thvm_tensor(ctx, _p, SHAPE(n)); }) \
            : make_zeros(ctx, (n)))
    #define LOAD_OR_ONES(n) \
        (wp ? ({ f32 *_p = wp; wp += (n); thvm_tensor(ctx, _p, SHAPE(n)); }) \
            : make_ones(ctx, (n)))

    if (wbin) printf("  Loaded tinygrad weights (%u floats) for exact parity\n", wsize);
    else printf("  Using Kaiming uniform init (no tinygrad weights found)\n");

    #define N_LAYERS 14
    Layer model[N_LAYERS] = {
        {.type=LAYER_CONV2D, .conv={LOAD_OR_INIT(((Shape){.dims={32,1,5,5},.rank=4}), 25, 32),
                                     LOAD_OR_ZEROS(32), 1, 32, 5}},
        {.type=LAYER_FN, .fn=relu_fn},
        {.type=LAYER_CONV2D, .conv={LOAD_OR_INIT(((Shape){.dims={32,32,5,5},.rank=4}), 800, 32),
                                     LOAD_OR_ZEROS(32), 32, 32, 5}},
        {.type=LAYER_FN, .fn=relu_fn},
        {.type=LAYER_BN, .bn={LOAD_OR_ONES(32), LOAD_OR_ZEROS(32),
                               LOAD_OR_ZEROS(32), LOAD_OR_ONES(32), 32, 20, 20}},
        {.type=LAYER_MAXPOOL, .pool={2}},
        {.type=LAYER_CONV2D, .conv={LOAD_OR_INIT(((Shape){.dims={64,32,3,3},.rank=4}), 288, 64),
                                     LOAD_OR_ZEROS(64), 32, 64, 3}},
        {.type=LAYER_FN, .fn=relu_fn},
        {.type=LAYER_CONV2D, .conv={LOAD_OR_INIT(((Shape){.dims={64,64,3,3},.rank=4}), 576, 64),
                                     LOAD_OR_ZEROS(64), 64, 64, 3}},
        {.type=LAYER_FN, .fn=relu_fn},
        {.type=LAYER_BN, .bn={LOAD_OR_ONES(64), LOAD_OR_ZEROS(64),
                               LOAD_OR_ZEROS(64), LOAD_OR_ONES(64), 64, 6, 6}},
        {.type=LAYER_MAXPOOL, .pool={2}},
        {.type=LAYER_FLATTEN, .flat={576}},
        {.type=LAYER_LINEAR, .lin={LOAD_OR_INIT(SHAPE(576, 10), 576, 10),
                                    LOAD_OR_ZEROS(10), 576, 10}},
    };
    if (wbin) free(wbin);
    #undef LOAD_OR_INIT
    #undef LOAD_OR_ZEROS
    #undef LOAD_OR_ONES

    // Extract params
    Term params[N_LAYERS * 2];
    u32 param_sizes[N_LAYERS * 2];
    u32 n_params = 0;
    for (u32 i = 0; i < N_LAYERS; i++) {
        Layer *l = &model[i];
        switch (l->type) {
            case LAYER_CONV2D:
                params[n_params] = l->conv.w; param_sizes[n_params] = ctx->tensors[(u32)term_val(l->conv.w)].view.numel; n_params++;
                params[n_params] = l->conv.b; param_sizes[n_params] = ctx->tensors[(u32)term_val(l->conv.b)].view.numel; n_params++;
                break;
            case LAYER_BN:
                params[n_params] = l->bn.gamma; param_sizes[n_params] = ctx->tensors[(u32)term_val(l->bn.gamma)].view.numel; n_params++;
                params[n_params] = l->bn.beta;  param_sizes[n_params] = ctx->tensors[(u32)term_val(l->bn.beta)].view.numel;  n_params++;
                break;
            case LAYER_LINEAR:
                params[n_params] = l->lin.w; param_sizes[n_params] = ctx->tensors[(u32)term_val(l->lin.w)].view.numel; n_params++;
                params[n_params] = l->lin.b; param_sizes[n_params] = ctx->tensors[(u32)term_val(l->lin.b)].view.numel; n_params++;
                break;
            default: break;
        }
    }
    for (u32 i = 0; i < n_params; i++)
        thvm_set_requires_grad(ctx, params[i]);

    u32 BS = 128;
    Term train_data = thvm_tensor(ctx, data->train_images,
        (Shape){.dims={data->n_train, 1, 28, 28}, .rank=4});

    Adam opt = adam_init(ctx, 0.001f, n_params);
    for (u32 i = 0; i < n_params; i++)
        adam_add_param(ctx, &opt, i, (u32)term_val(params[i]), param_sizes[i]);

    u32 n_weights = ctx->tensor_count;
    u32 n_batches = data->n_train / BS;
    u32 n_steps = 70;
    f32 lr_max = 0.001f, lr_min = 0.0001f;
    printf("  %u params, %u steps, BS=%u\n\n", n_params, n_steps, BS);

    struct timespec train_start; clock_gettime(CLOCK_MONOTONIC, &train_start);

    for (u32 step = 0; step < n_steps; step++) {
      @autoreleasepool {
        f32 progress = (f32)step / (f32)n_steps;
        opt.lr = lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf(3.14159f * progress));
        struct timespec t0_wall; clock_gettime(CLOCK_MONOTONIC, &t0_wall);

        if (ctx->backend->begin_batch) ctx->backend->begin_batch();

        u32 bi = step % n_batches;
        Term x = thvm_shrink(ctx, train_data,
            (u32[]){bi*BS, (bi+1)*BS, 0, 1, 0, 28, 0, 28}, 4);
        thvm_set_requires_grad(ctx, x);
        u8 *by = &data->train_labels[bi * BS];

        Term logits = thvm_sequential(ctx, x, model, N_LAYERS, BS, 1);
        Term loss = cross_entropy_loss(ctx, logits, by, BS, 10);

        if (ctx->backend->end_batch) ctx->backend->end_batch();

        Term loss_r = thvm_reduce(ctx, loss);
        f32 loss_val = thvm_to_host(ctx, loss_r)[0];

        Term grad_terms[n_params];
        thvm_backward(ctx, loss_r, params, grad_terms, n_params);
        u32 grad_ids[n_params];
        for (u32 i = 0; i < n_params; i++)
            grad_ids[i] = (term_tag(grad_terms[i]) == TAG_TEN) ? (u32)term_val(grad_terms[i]) : 0;

        adam_step(ctx, &opt, grad_ids);

        extern u32 total_dispatches;
        if (step == 0) {
            printf("    GPU dispatches: %u (tinygrad: ~186)\n", total_dispatches);
            total_dispatches = 0;
        }

        // Invalidate host caches so next step reads updated weights
        for (u32 i = 0; i < n_params; i++) {
            u32 pid = (u32)term_val(params[i]);
            if (ctx->tensors[pid].host_ptr) {
                free(ctx->tensors[pid].host_ptr);
                ctx->tensors[pid].host_ptr = NULL;
            }
        }

        struct timespec t1_wall; clock_gettime(CLOCK_MONOTONIC, &t1_wall);
        f32 ms = (f32)(t1_wall.tv_sec - t0_wall.tv_sec)*1000.0f +
                 (f32)(t1_wall.tv_nsec - t0_wall.tv_nsec)/1e6f;
        if (step % 10 == 0 || step == n_steps - 1)
            printf("  step %3u/%u  loss=%.4f  lr=%.5f  (%.0fms)\n",
                   step, n_steps, loss_val, opt.lr, ms);

        thvm_reset(ctx, n_weights);
      }
    }

    struct timespec train_end; clock_gettime(CLOCK_MONOTONIC, &train_end);
    f32 train_s = (f32)(train_end.tv_sec - train_start.tv_sec) +
                  (f32)(train_end.tv_nsec - train_start.tv_nsec)/1e9f;
    printf("\n  Train: %.1fs (%.0fms/step avg)\n", train_s, train_s*1000.0f/(f32)n_steps);

    // Eval
    struct timespec eval_start; clock_gettime(CLOCK_MONOTONIC, &eval_start);
    printf("  Evaluating test set...\n");
    Term test_data = thvm_tensor(ctx, data->test_images,
        (Shape){.dims={data->n_test, 1, 28, 28}, .rank=4});
    u32 eval_keep = ctx->tensor_count;
    u32 correct = 0, tbs = 64, tb = data->n_test / tbs;
    for (u32 b = 0; b < tb; b++) {
        Term x = thvm_shrink(ctx, test_data,
            (u32[]){b*tbs, (b+1)*tbs, 0, 1, 0, 28, 0, 28}, 4);
        Term logits = thvm_sequential(ctx, x, model, N_LAYERS, tbs, 0);
        f32 batch_acc = thvm_eval_accuracy(ctx, logits, &data->test_labels[b*tbs], tbs, 10);
        correct += (u32)(batch_acc * (f32)tbs / 100.0f);
        thvm_reset(ctx, eval_keep);
    }

    struct timespec eval_end; clock_gettime(CLOCK_MONOTONIC, &eval_end);
    f32 eval_s = (f32)(eval_end.tv_sec - eval_start.tv_sec) +
                 (f32)(eval_end.tv_nsec - eval_start.tv_nsec)/1e9f;

    f32 acc = 100.0f * (f32)correct / (f32)(tb * tbs);
    printf("\n  Test accuracy: %.1f%% (%u/%u)\n", acc, correct, tb * tbs);
    printf("  Eval: %.1fs  Total: %.1fs\n", eval_s, train_s + eval_s);
    printf("  %s: CNN accuracy %s 90%%\n", acc > 90 ? "PASS" : "FAIL", acc > 90 ? ">" : "<");

    adam_free(&opt);
    thvm_free(ctx);
    return acc > 90 ? 0 : 1;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char **argv) {
    srand(42);
    const char *arch = (argc > 1) ? argv[1] : "mlp";
    int cnn = (strcmp(arch, "cnn") == 0);

    MNISTData data = mnist_load("data");
    printf("  Train: %u, Test: %u\n\n", data.n_train, data.n_test);

    int rc = cnn ? run_cnn(&data) : run_mlp(&data);

    mnist_free(&data);
    return rc;
}
