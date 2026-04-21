// test_tiny_linear_bias_keep.m — explicit multi-grad bundle readback
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *backend = getenv("THVM_TEST_BACKEND");
    if (!backend || !backend[0]) backend = argc > 1 ? argv[1] : "cpu";
#ifdef __APPLE__
    if (strcmp(backend, "metal") == 0 && !MTLCreateSystemDefaultDevice()) {
        fprintf(stderr, "SKIP: metal unavailable\n");
        return 0;
    }
#endif
    TinyHVM *ctx = thvm_init(backend);

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

    Term logits = thvm_op(ctx, UOP_ADD,
        thvm_mm(ctx, x, w),
        thvm_expand(ctx, thvm_reshape(ctx, b, SHAPE(1,4)), SHAPE(2,4)));
    Term act = thvm_op(ctx, UOP_RELU, logits, term_era());
    Term loss = thvm_sum_axes(ctx, act, (u32[]){0, 1}, 2);

    Term params[] = {w, b};
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));
    if (thvm_grad_bundle_count(ctx, bundle) == 2) {
        f32 *dw = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 0));
        f32 *db = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 1));
        if (dw && db) {
            printf("grad_w = [");
            for (int i = 0; i < 12; i++) printf("%.2f%s", dw[i], i == 11 ? "" : ", ");
            printf("]\n");
            printf("grad_b = [%.2f, %.2f, %.2f, %.2f]\n", db[0], db[1], db[2], db[3]);
        }
    }

    thvm_free(ctx);
    return 0;
}
