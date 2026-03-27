// test_cnn_small.m — Minimal CNN test: 3 steps, BS=32, with explicit DUP
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

    for (u32 step = 0; step < 3; step++) {
        struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);

        Term x = thvm_shrink(ctx, train_data,
            (u32[]){step*BS, (step+1)*BS, 0, 1, 0, 28, 0, 28}, 4);
        thvm_set_requires_grad(ctx, x);
        u8 *by = &data.train_labels[step * BS];

        // No DUP — direct (working baseline with GRAD3_DUP changes)
        u32 padding[] = {0,0,0,0}, stride[] = {1,1};
        Term h = thvm_conv2d(ctx, x, conv_w, conv_b, 1, stride, padding);
        h = thvm_op(ctx, UOP_RELU, h, term_era());
        h = thvm_reshape(ctx, h, SHAPE(BS, flat_f));
        Term logits = thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MM, h, lin_w),
            thvm_expand(ctx, thvm_reshape(ctx, lin_b, SHAPE(1, 10)), SHAPE(BS, 10)));
        Term loss = cross_entropy_loss(ctx, logits, by, BS, 10);

        f32 lr = 0.01f;
        Term lr_t = thvm_tensor(ctx, &lr, SHAPE(1));
        Term chain = term_era();
        for (int i = N_PARAMS - 1; i >= 0; i--) {
            Term g = thvm_grad(ctx, loss, params[i]);
            Term new_p = thvm_op(ctx, UOP_SUB, params[i],
                thvm_op(ctx, UOP_MUL, lr_t, g));
            chain = thvm_app(ctx, thvm_assign(ctx, params[i], new_p), chain);
        }
        thvm_reduce(ctx, chain);
        f32 loss_val = thvm_to_host(ctx, loss)[0];

        struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        f32 ms = (f32)(t1.tv_sec-t0.tv_sec)*1000+(f32)(t1.tv_nsec-t0.tv_nsec)/1e6f;

        extern u32 total_dispatches;
        printf("  step %u: loss=%.4f dispatches=%u tensors=%u heap=%llu (%.0fms)\n",
               step, loss_val, total_dispatches, ctx->tensor_count,
               (unsigned long long)ctx->heap_pos, ms);
        total_dispatches = 0;

        thvm_reset(ctx, n_weights);
    }

    thvm_free(ctx);
    mnist_free(&data);
    printf("  DONE\n");
    return 0;
}
