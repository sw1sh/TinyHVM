// test_cnn_small.m — Small CNN: Conv(1,8,3)→ReLU→Flatten→Linear(8*26*26,10)
// Per-step C loop with thvm_grad + SGD. Tests convergence + dispatch count.

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

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    srand(42);
    MNISTData data = mnist_load("data");
    printf("  Train: %u, Test: %u\n\n", data.n_train, data.n_test);

    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    u32 BS = 32;

    u32 cw_n = 8*1*3*3;
    f32 *cw = malloc(cw_n * sizeof(f32));
    for (u32 i = 0; i < cw_n; i++) cw[i] = 0.1f * ((f32)rand()/(f32)RAND_MAX * 2 - 1);
    Term conv_w = thvm_tensor(ctx, cw, (Shape){.dims={8,1,3,3},.rank=4});
    free(cw);
    f32 *cb = calloc(8, sizeof(f32));
    Term conv_b = thvm_tensor(ctx, cb, SHAPE(8));
    free(cb);

    u32 flat_f = 8 * 26 * 26;
    u32 lw_n = flat_f * 10;
    f32 *lw = malloc(lw_n * sizeof(f32));
    f32 bound = 1.0f / sqrtf((f32)flat_f);
    for (u32 i = 0; i < lw_n; i++) lw[i] = bound * ((f32)rand()/(f32)RAND_MAX * 2 - 1);
    Term lin_w = thvm_tensor(ctx, lw, SHAPE(flat_f, 10));
    free(lw);
    f32 *lb = calloc(10, sizeof(f32));
    Term lin_b = thvm_tensor(ctx, lb, SHAPE(10));
    free(lb);

    #define N_PARAMS 4
    Term params[N_PARAMS] = { conv_w, conv_b, lin_w, lin_b };
    for (u32 i = 0; i < N_PARAMS; i++)
        thvm_set_requires_grad(ctx, params[i]);

    Term train_data = thvm_tensor(ctx, data.train_images,
        (Shape){.dims={data.n_train, 1, 28, 28}, .rank=4});

    u32 n_weights = ctx->tensor_count;
    u32 n_steps = 20;

    for (u32 step = 0; step < n_steps; step++) {
        struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);

        u32 bi = step % (data.n_train / BS);
        Term x = thvm_shrink(ctx, train_data,
            (u32[]){bi*BS, (bi+1)*BS, 0, 1, 0, 28, 0, 28}, 4);
        thvm_set_requires_grad(ctx, x);
        u8 *by = &data.train_labels[bi * BS];

        u32 padding[] = {0,0,0,0}, stride[] = {1,1};
        Term h = thvm_conv2d(ctx, x, conv_w, conv_b, 1, stride, padding);
        h = thvm_op(ctx, UOP_RELU, h, term_era());
        h = thvm_reshape(ctx, h, SHAPE(BS, flat_f));
        Term logits = thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MM, h, lin_w),
            thvm_expand(ctx, thvm_reshape(ctx, lin_b, SHAPE(1, 10)), SHAPE(BS, 10)));
        Term loss = cross_entropy_loss(ctx, logits, by, BS, 10);

        // Pure IC: ONE reduce drives grad + SGD.
        // Grad slots: pre-allocated zero tensors for gradient deposits.
        Term grad_slots[N_PARAMS];
        u32 param_sizes[] = {8*1*3*3, 8, flat_f*10, 10};
        for (int i = 0; i < N_PARAMS; i++) {
            f32 *z = calloc(param_sizes[i], sizeof(f32));
            grad_slots[i] = thvm_tensor(ctx, z, ctx->tensors[(u32)term_val(params[i])].view.shape);
            free(z);
        }

        // grad_term: walks provenance once, deposits into grad_slots via ASSIGN
        Term grad_term = thvm_grad_multi(ctx, loss, params, grad_slots, N_PARAMS);

        // SGD chain: reads from grad_slots (filled by grad_term's ASSIGNs)
        f32 lr = 0.01f;
        Term lr_t = thvm_tensor(ctx, &lr, SHAPE(1));
        Term sgd_chain = term_era();
        for (int i = N_PARAMS - 1; i >= 0; i--) {
            Term new_p = thvm_op(ctx, UOP_SUB, params[i],
                thvm_op(ctx, UOP_MUL, lr_t, grad_slots[i]));
            sgd_chain = thvm_app(ctx, thvm_assign(ctx, params[i], new_p), sgd_chain);
        }

        // ONE reduce: APP(grad_term, sgd_chain)
        // grad_term runs first (deposits), returns ERA. APP-ERA → sgd_chain runs.
        thvm_reduce(ctx, thvm_app(ctx, grad_term, sgd_chain));
        f32 loss_val = thvm_to_host(ctx, loss)[0];

        struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        f32 ms = (f32)(t1.tv_sec-t0.tv_sec)*1000+(f32)(t1.tv_nsec-t0.tv_nsec)/1e6f;

        extern u32 total_dispatches;
        extern void print_dispatch_breakdown(void);
        if (step < 5 || step % 5 == 0 || step == n_steps - 1)
            printf("  step %2u: loss=%.4f dispatches=%u tensors=%u heap=%llu (%.0fms)\n",
                   step, loss_val, total_dispatches, ctx->tensor_count,
                   (unsigned long long)ctx->heap_pos, ms);
        if (step == 1) print_dispatch_breakdown();
        total_dispatches = 0;

        // Invalidate host caches for updated params
        for (u32 i = 0; i < N_PARAMS; i++) {
            u32 pid = (u32)term_val(params[i]);
            if (ctx->tensors[pid].host_ptr) {
                free(ctx->tensors[pid].host_ptr);
                ctx->tensors[pid].host_ptr = NULL;
            }
        }
        thvm_reset(ctx, n_weights);
    }

    // Quick eval
    printf("\n  Evaluating...\n");
    Term test_data = thvm_tensor(ctx, data.test_images,
        (Shape){.dims={data.n_test, 1, 28, 28}, .rank=4});
    u32 eval_keep = ctx->tensor_count;
    u32 correct = 0, tbs = 32, tb = data.n_test / tbs;
    for (u32 b = 0; b < tb; b++) {
        Term x = thvm_shrink(ctx, test_data,
            (u32[]){b*tbs, (b+1)*tbs, 0, 1, 0, 28, 0, 28}, 4);
        u32 padding[] = {0,0,0,0}, stride[] = {1,1};
        Term h = thvm_conv2d(ctx, x, conv_w, conv_b, 1, stride, padding);
        h = thvm_op(ctx, UOP_RELU, h, term_era());
        h = thvm_reshape(ctx, h, SHAPE(tbs, flat_f));
        Term logits = thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MM, h, lin_w),
            thvm_expand(ctx, thvm_reshape(ctx, lin_b, SHAPE(1, 10)), SHAPE(tbs, 10)));
        f32 acc = thvm_eval_accuracy(ctx, logits, &data.test_labels[b*tbs], tbs, 10);
        correct += (u32)(acc * (f32)tbs / 100.0f);
        thvm_reset(ctx, eval_keep);
    }
    f32 acc = 100.0f * (f32)correct / (f32)(tb * tbs);
    printf("  Test accuracy: %.1f%% (%u/%u)\n", acc, correct, tb*tbs);

    thvm_free(ctx);
    mnist_free(&data);
    printf("  DONE\n");
    return 0;
}
