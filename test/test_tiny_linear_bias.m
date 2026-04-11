// test_tiny.m — phase-1 graph case for a tiny linear layer with bias
// loss = sum(relu(x @ W + b)), backward for W and b
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <math.h>

int main(void) {
    srand(42);
    TinyHVM *ctx = thvm_init("cpu");

    // Tiny linear layer: x=[2,3], W=[3,4], b=[4]
    f32 xd[] = {1,2,3, 4,5,6};
    f32 wd[] = {
         0.10f, -0.20f,  0.30f,  0.40f,
        -0.50f,  0.60f, -0.70f,  0.80f,
         0.90f, -1.00f,  1.10f, -1.20f,
    };
    f32 bd[] = {0.05f, -0.10f, 0.15f, 0.20f};
    Term x = thvm_tensor(ctx, xd, (Shape){.dims={2,3}, .rank=2});
    Term w = thvm_tensor(ctx, wd, (Shape){.dims={3,4}, .rank=2});
    Term b = thvm_tensor(ctx, bd, (Shape){.dims={4}, .rank=1});
    thvm_set_requires_grad(ctx, w);
    thvm_set_requires_grad(ctx, b);

    // Forward: loss = sum(relu(x @ W + b))
    Term logits = thvm_op(ctx, UOP_ADD,
        thvm_mm(ctx, x, w),
        thvm_expand(ctx, thvm_reshape(ctx, b, SHAPE(1,4)), SHAPE(2,4)));
    Term act = thvm_op(ctx, UOP_RELU, logits, term_era());
    Term loss = thvm_sum_axes(ctx, act, (u32[]){0, 1}, 2);

    Term grad = thvm_grad_multi(ctx, loss, (Term[]){w, b}, NULL, 2);
    Term log_loss = thvm_log_print(ctx, loss);
    Term program = thvm_app(ctx, log_loss, grad);
    thvm_eval(ctx, program);

    thvm_free(ctx);
    return 0;
}
