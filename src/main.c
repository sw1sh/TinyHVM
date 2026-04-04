// main.c — TinyHVM CLI entry point

#include "tinyhvm.c"
#include "gpu_cpu.c"

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    printf("TinyHVM v0.2\n");
    printf("heap: %llu terms (%llu MB)\n",
           (unsigned long long)HEAP_CAP,
           (unsigned long long)(HEAP_CAP * 8 / (1024 * 1024)));
    printf("tags: %d, uops: %d\n\n", TAG_COUNT, UOP_COUNT);

    TinyHVM *ctx = thvm_init(&gpu_cpu_backend);

    // x = [[1, 2, 3], [4, 5, 6]]  (2x3)
    f32 x_data[] = {1, 2, 3, 4, 5, 6};
    u32 x_shape[] = {2, 3};
    Term x = thvm_tensor(ctx, x_data, x_shape, 2);

    // w = [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]]  (3x2)
    f32 w_data[] = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f};
    u32 w_shape[] = {3, 2};
    Term w = thvm_tensor(ctx, w_data, w_shape, 2);

    // b = [-0.1, 0.2]  (1x2) — will be broadcast to (2x2)!
    f32 b_data[] = {-0.1f, 0.2f};
    u32 b_shape[] = {1, 2};
    Term b = thvm_tensor(ctx, b_data, b_shape, 2);

    // z = relu(matmul(x, w) + b)  — b is auto-broadcast!
    Term xw  = thvm_mm(ctx, x, w);
    Term xwb = thvm_op(ctx, UOP_ADD, xw, b);
    Term z   = thvm_op(ctx, UOP_RELU, xwb, term_era());

    printf("Before reduction:\n  z = ");
    thvm_print_term(ctx, z);
    printf("\n");

    Term result = thvm_reduce(ctx, z);
    printf("\nAfter reduction:\n  z = ");
    thvm_print_term(ctx, result);
    printf("\n");

    f32 *out = thvm_to_host(ctx, result);
    if (out) {
        u32 id = (u32)term_val(result);
        View *v = &ctx->tensors[id].view;
        printf("\nResult [%u x %u]:\n", v->shape[0], v->shape[1]);
        for (u32 i = 0; i < v->shape[0]; i++) {
            printf("  [");
            for (u32 j = 0; j < v->shape[1]; j++)
                printf(" %.4f", out[i * v->shape[1] + j]);
            printf(" ]\n");
        }
    }

    printf("\nInteractions: %llu\n", (unsigned long long)ctx->itrs);
    thvm_free(ctx);
    return 0;
}
