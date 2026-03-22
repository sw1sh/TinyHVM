// main.c — TinyHVM CLI entry point
// For now, just runs the forward pass test inline.

#include "tinyhvm.c"
#include "gpu_cpu.c"

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    printf("TinyHVM v0.1\n");
    printf("heap: %llu terms (%llu MB)\n",
           (unsigned long long)HEAP_CAP,
           (unsigned long long)(HEAP_CAP * 8 / (1024 * 1024)));
    printf("tags: %d, uops: %d\n", TAG_COUNT, UOP_COUNT);

    // Quick smoke test: relu(matmul(x, w) + b)
    TinyHVM *ctx = thvm_init(&gpu_cpu_backend);

    // x = [[1, 2, 3], [4, 5, 6]]  (2x3)
    f32 x_data[] = {1, 2, 3, 4, 5, 6};
    u32 x_shape[] = {2, 3};
    Term x = thvm_tensor(ctx, x_data, x_shape, 2);

    // w = [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]]  (3x2)
    f32 w_data[] = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f};
    u32 w_shape[] = {3, 2};
    Term w = thvm_tensor(ctx, w_data, w_shape, 2);

    // b = [-0.1, 0.2]  (2,) — will need broadcast, for now use (1x2)
    // Actually for simplicity, expand b to (2x2) matching output
    f32 b_data[] = {-0.1f, 0.2f, -0.1f, 0.2f};
    u32 b_shape[] = {2, 2};
    Term b = thvm_tensor(ctx, b_data, b_shape, 2);

    // z = relu(matmul(x, w) + b)
    Term xw  = thvm_op(ctx, UOP_MM, x, w);
    Term xwb = thvm_op(ctx, UOP_ADD, xw, b);
    Term z   = thvm_op(ctx, UOP_RELU, xwb, term_era());

    printf("\nBefore reduction:\n  z = ");
    thvm_print_term(ctx, z);
    printf("\n");

    // Realize
    Term result = thvm_reduce(ctx, z);
    printf("\nAfter reduction:\n  z = ");
    thvm_print_term(ctx, result);
    printf("\n");

    // Read back
    f32 *out = thvm_to_host(ctx, result);
    if (out) {
        u32 id = (u32)term_val(result);
        TensorMeta *m = &ctx->tensors[id];
        printf("\nResult [%u x %u]:\n", m->shape[0], m->shape[1]);
        for (u32 i = 0; i < m->shape[0]; i++) {
            printf("  [");
            for (u32 j = 0; j < m->shape[1]; j++) {
                printf(" %.4f", out[i * m->shape[1] + j]);
            }
            printf(" ]\n");
        }
    }

    printf("\nInteractions: %llu\n", (unsigned long long)ctx->itrs);

    thvm_free(ctx);
    return 0;
}
