// test_tiny.m — minimal graph for visual debugging
// loss = sum(relu(x * w)), backward for w
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
    int step_graph = getenv("THVM_STEP_GRAPH") != NULL;

    // Small tensors: x=[2,3], w=[3,1]
    f32 xd[] = {1,2,3, 4,5,6};
    f32 wd[] = {0.1, 0.2, 0.3};
    Term x = thvm_tensor(ctx, xd, (Shape){.dims={2,3}, .rank=2});
    Term w = thvm_tensor(ctx, wd, (Shape){.dims={3}, .rank=1});
    thvm_set_requires_grad(ctx, w);

    u8 saved_nga = ctx->no_grad_alloc;
    if (step_graph) ctx->no_grad_alloc = 1;

    // Forward: loss = sum(relu(x * w))
    Term h = thvm_op(ctx, UOP_MUL, x, w);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    Term loss = thvm_sum_axes(ctx, h, (u32[]){0, 1}, 2);
    ctx->no_grad_alloc = saved_nga;

    // Backward
    if (step_graph) {
        Term grad = thvm_grad(ctx, loss, w);
        thvm_eval(ctx, grad);
    } else {
        f32 zero = 0;
        Term gw = thvm_tensor(ctx, &zero, (Shape){.dims={3}, .rank=1});
        Term grad = thvm_grad_multi(ctx, loss, (Term[]){w}, (Term[]){gw}, 1);
        thvm_eval(ctx, grad);

        f32 *dw = thvm_to_host(ctx, gw);
        printf("grad_w = [%.2f, %.2f, %.2f]\n", dw[0], dw[1], dw[2]);
    }

    thvm_free(ctx);
    return 0;
}
