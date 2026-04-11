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
    int step_graph = getenv("THVM_STEP_GRAPH") != NULL;

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

    // For phase-1 graph dumps, keep forward view construction lazy so the
    // init net shows the actual source tensors instead of hidden alias tids.
    u8 saved_nga = ctx->no_grad_alloc;
    if (step_graph) ctx->no_grad_alloc = 1;

    // Forward: loss = sum(relu(x @ W + b))
    Term logits = thvm_op(ctx, UOP_ADD,
        thvm_mm(ctx, x, w),
        thvm_expand(ctx, thvm_reshape(ctx, b, SHAPE(1,4)), SHAPE(2,4)));
    Term act = thvm_op(ctx, UOP_RELU, logits, term_era());
    Term loss = thvm_sum_axes(ctx, act, (u32[]){0, 1}, 2);
    ctx->no_grad_alloc = saved_nga;

    // Backward
    if (step_graph) {
        // Phase-1 only: do not preallocate gradient tensors. The graph should
        // expose only tensors that are actually part of the visible init net.
        Term grad = thvm_grad_multi(ctx, loss, (Term[]){w, b}, NULL, 2);
        thvm_eval(ctx, grad);
    } else {
        f32 zw[12] = {0};
        f32 zb[4] = {0};
        Term gw = thvm_tensor(ctx, zw, (Shape){.dims={3,4}, .rank=2});
        Term gb = thvm_tensor(ctx, zb, (Shape){.dims={4}, .rank=1});
        Term grad = thvm_grad_multi(ctx, loss, (Term[]){w, b}, (Term[]){gw, gb}, 2);
        thvm_eval(ctx, grad);

        f32 *dw = thvm_to_host(ctx, gw);
        f32 *db = thvm_to_host(ctx, gb);
        printf("grad_w = [");
        for (int i = 0; i < 12; i++) {
            printf("%.2f%s", dw[i], i == 11 ? "" : ", ");
        }
        printf("]\n");
        printf("grad_b = [%.2f, %.2f, %.2f, %.2f]\n", db[0], db[1], db[2], db[3]);
    }

    thvm_free(ctx);
    return 0;
}
