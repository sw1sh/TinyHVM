// Phase-2 checkpoint test for linear+bias keep path on CPU.
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>

int main(void) {
    TinyHVM *ctx = thvm_init("cpu");
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
    Term keep = thvm_grad_multi_keep(ctx, loss, params, 2);

    fprintf(stderr, "phase2: eval begin\n"); fflush(stderr);
    Term bundle = thvm_eval(ctx, keep);
    fprintf(stderr, "phase2: eval done\n"); fflush(stderr);

    u32 n = thvm_grad_bundle_count(ctx, bundle);
    fprintf(stderr, "phase2: bundle n=%u\n", n); fflush(stderr);

    if (n > 0) {
      fprintf(stderr, "phase2: get g0\n"); fflush(stderr);
      Term g0 = thvm_grad_bundle_get(ctx, bundle, 0);
      fprintf(stderr, "phase2: host g0\n"); fflush(stderr);
      f32 *dw = thvm_to_host(ctx, g0);
      fprintf(stderr, "phase2: host g0 done %.3f\n", dw ? dw[0] : -1.0f); fflush(stderr);
    }
    if (n > 1) {
      fprintf(stderr, "phase2: get g1\n"); fflush(stderr);
      Term g1 = thvm_grad_bundle_get(ctx, bundle, 1);
      fprintf(stderr, "phase2: host g1\n"); fflush(stderr);
      f32 *db = thvm_to_host(ctx, g1);
      fprintf(stderr, "phase2: host g1 done %.3f\n", db ? db[0] : -1.0f); fflush(stderr);
    }

    thvm_free(ctx);
    return 0;
}
