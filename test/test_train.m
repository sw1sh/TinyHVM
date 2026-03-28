// test_train.m — HVM-faithful training test
//
// The ENTIRE training sequence is ONE lazy term:
//
//   program = thvm_train_step(ctx, N, W1, B1, W2, B2, X, Y, LR)
//
// thvm_reduce(program) drives all N steps internally:
//   each step builds forward→grad→assign, tail-recurses to step n-1
//   when n==0 it returns ERA (training complete)
//
// No C training loop, no explicit backward call.
// The inet is the program.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "train_helpers.h"
#include <stdio.h>

#ifndef DEVICE
  #define DEVICE "cpu"
#endif

int main(void) {
    printf("=== HVM train: thvm_reduce(train_step(N)) — no C loop ===\n\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    f32 x_d[] = {0,0, 0,1, 1,0, 1,1};
    f32 y_d[] = {0, 1, 1, 0};
    Term X = thvm_tensor(ctx, x_d, SHAPE(4,2));
    Term Y = thvm_tensor(ctx, y_d, SHAPE(4,1));

    f32 w1d[] = { 0.5f,-0.3f, 0.2f, 0.4f,-0.1f, 0.6f,-0.4f, 0.3f};
    f32 b1d[] = {0,0,0,0}, w2d[] = {0.5f,-0.3f,0.2f,0.4f}, b2d[] = {0};
    Term W1 = thvm_tensor(ctx, w1d, SHAPE(2,4));
    Term B1 = thvm_tensor(ctx, b1d, SHAPE(1,4));
    Term W2 = thvm_tensor(ctx, w2d, SHAPE(4,1));
    Term B2 = thvm_tensor(ctx, b2d, SHAPE(1,1));
    thvm_set_requires_grad(ctx, W1);
    thvm_set_requires_grad(ctx, B1);
    thvm_set_requires_grad(ctx, W2);
    thvm_set_requires_grad(ctx, B2);

    f32 lr_val = 0.5f;
    Term LR = thvm_tensor(ctx, &lr_val, SHAPE(1));

    // ── Build recursive inet program — NO computation, NO loop ─────────────
    // Loss is logged each step via LOG_PRINT inside the inet.
    int N = 50;
    Term program = train_program(ctx, W1, B1, W2, B2, X, Y, LR, N);

    // ── ONE reduction drives the entire N-step training ───────────────────
    printf("  reducing %d-step inet...\n", N);
    thvm_reduce(ctx, program);
    printf("  done — itrs=%llu\n", ctx->itrs);

    printf("\n=== DONE ===\n");
    thvm_free(ctx);
    return 0;
}
