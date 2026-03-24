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
#include <stdio.h>

#ifndef DEVICE
  #define DEVICE "cpu"
#endif

static f32 eval_loss(TinyHVM *ctx, Term X, Term W1, Term B1,
                     Term W2, Term B2, Term Y, int nw) {
    Term z1  = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, X, W1), B1);
    Term h   = thvm_op(ctx, UOP_RELU, z1, term_era());
    Term out = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, W2), B2);
    Term diff = thvm_op(ctx, UOP_SUB, out, Y);
    Term sq  = thvm_op(ctx, UOP_MUL, diff, diff);
    u32 ax[] = {0, 1};
    Term loss = thvm_reduce(ctx, thvm_sum_axes(ctx, sq, ax, 2));
    f32 v = (term_tag(loss) == TAG_TEN) ? thvm_to_host(ctx, loss)[0] / 4.f : 1e9f;
    thvm_reset(ctx, nw);
    return v;
}

int main(void) {
    printf("=== HVM train: thvm_reduce(train_step(N)) — no C loop ===\n\n");
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

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
    int nw = (int)ctx->tensor_count;

    f32 lr_val = 0.5f;
    Term LR = thvm_tensor(ctx, &lr_val, SHAPE(1));

    f32 loss_before = eval_loss(ctx, X, W1, B1, W2, B2, Y, nw);
    printf("  loss before: %.6f\n", loss_before);

    // ── Build the program — NO computation yet ────────────────────────────
    // This is ONE lazy term encoding all N training steps.
    int N = 50;
    Term program = thvm_train_step(ctx, N, W1, B1, W2, B2, X, Y, LR);

    // ── ONE reduction drives the full training ────────────────────────────
    printf("  reducing %d-step inet...\n", N);
    thvm_reduce(ctx, program);
    printf("  done — itrs=%u\n", ctx->itrs);
    thvm_reset(ctx, nw);

    f32 loss_after = eval_loss(ctx, X, W1, B1, W2, B2, Y, nw);
    printf("  loss after:  %.6f\n\n", loss_after);

    int pass = loss_after < loss_before * 0.5f;
    printf("%s\n", pass
        ? "=== PASS: inet training reduced loss ==="
        : "=== FAIL ===");
    thvm_free(ctx);
    return pass ? 0 : 1;
}
