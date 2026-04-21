// test_tiny.m — minimal graph for visual debugging
// loss = sum(relu(x * w)); plain GRAD is drop-mode unless explicitly kept
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

    // Small tensors: x=[2,3], w=[3,1]
    f32 xd[] = {1,2,3, 4,5,6};
    f32 wd[] = {0.1, 0.2, 0.3};
    Term x = thvm_tensor(ctx, xd, (Shape){.dims={2,3}, .rank=2});
    Term w = thvm_tensor(ctx, wd, (Shape){.dims={3}, .rank=1});
    thvm_set_requires_grad(ctx, w);

    // Forward: loss = sum(relu(x * w))
    Term h = thvm_op(ctx, UOP_MUL, x, w);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    Term loss = thvm_sum_axes(ctx, h, (u32[]){0, 1}, 2);

    // Plain GRAD is drop-mode — backward walk has no consumer so the seed
    // stays a NUM(1.0) literal (no scalar tensor allocation). With the
    // forward no longer observed by a separate LOG_PRINT, `loss` has a
    // single consumer (GRAD) and no external DUP is inserted.
    Term grad = thvm_grad_multi(ctx, loss, (Term[]){w}, NULL, 1);
    thvm_eval(ctx, grad);

    thvm_free(ctx);
    return 0;
}
