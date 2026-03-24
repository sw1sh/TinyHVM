// Minimal debug: 2 steps, CPU backend, trace SUB operand values
#define DEVICE "cpu"
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#include "../src/backend/metal/_.m"
#include "train_helpers.h"

int main(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    f32 x_d[] = {0,0, 0,1, 1,0, 1,1};
    f32 y_d[] = {0, 1, 1, 0};
    Term X = thvm_tensor(ctx, x_d, SHAPE(4,2));
    Term Y = thvm_tensor(ctx, y_d, SHAPE(4,1));

    f32 w1d[] = {0.5f,-0.3f, 0.2f, 0.4f,-0.1f, 0.6f,-0.4f, 0.3f};
    f32 b1d[] = {0,0,0,0}, w2d[] = {0.5f,-0.3f,0.2f,0.4f}, b2d[] = {0};
    Term W1 = thvm_tensor(ctx, w1d, SHAPE(2,4));
    Term B1 = thvm_tensor(ctx, b1d, SHAPE(1,4));
    Term W2 = thvm_tensor(ctx, w2d, SHAPE(4,1));
    Term B2 = thvm_tensor(ctx, b2d, SHAPE(1,1));
    thvm_set_requires_grad(ctx, W1);
    thvm_set_requires_grad(ctx, B1);
    thvm_set_requires_grad(ctx, W2);
    thvm_set_requires_grad(ctx, B2);
    f32 lr = 0.5f;
    Term LR = thvm_tensor(ctx, &lr, SHAPE(1));

    printf("W1 id=%u, B1 id=%u, W2 id=%u, B2 id=%u, LR id=%u\n",
           (u32)term_val(W1), (u32)term_val(B1), (u32)term_val(W2),
           (u32)term_val(B2), (u32)term_val(LR));

    printf("before: W1[0]=%.6f\n", thvm_to_host(ctx, W1)[0]);

    // Build 2-step lazy recursive program (no C loop!)
    Term program = train_program(ctx, W1, B1, W2, B2, X, Y, LR, 2);

    printf("reducing...\n");
    thvm_reduce(ctx, program);
    // Invalidate host cache
    for (u32 i = 0; i < ctx->tensor_count; i++)
        if (ctx->tensors[i].host_ptr) { free(ctx->tensors[i].host_ptr); ctx->tensors[i].host_ptr = NULL; }
    printf("after:  W1[0]=%.6f\n", thvm_to_host(ctx, W1)[0]);

    thvm_free(ctx);
    return 0;
}
