// Stage-by-stage hang localization for linear+bias keep-backward on CPU.
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>

int main(void) {
    fprintf(stderr, "stage: init\n"); fflush(stderr);
    TinyHVM *ctx = thvm_init("cpu");

    f32 xd[] = {1,2,3, 4,5,6};
    f32 wd[] = {
         0.10f, -0.20f,  0.30f,  0.40f,
        -0.50f,  0.60f, -0.70f,  0.80f,
         0.90f, -1.00f,  1.10f, -1.20f,
    };
    f32 bd[] = {0.05f, -0.10f, 0.15f, 0.20f};

    fprintf(stderr, "stage: tensors\n"); fflush(stderr);
    Term x = thvm_tensor(ctx, xd, (Shape){.dims={2,3}, .rank=2});
    Term w = thvm_tensor(ctx, wd, (Shape){.dims={3,4}, .rank=2});
    Term b = thvm_tensor(ctx, bd, (Shape){.dims={4}, .rank=1});
    thvm_set_requires_grad(ctx, w);
    thvm_set_requires_grad(ctx, b);

    fprintf(stderr, "stage: forward graph\n"); fflush(stderr);
    Term logits = thvm_op(ctx, UOP_ADD,
        thvm_mm(ctx, x, w),
        thvm_expand(ctx, thvm_reshape(ctx, b, SHAPE(1,4)), SHAPE(2,4)));
    Term act = thvm_op(ctx, UOP_RELU, logits, term_era());
    Term loss = thvm_sum_axes(ctx, act, (u32[]){0, 1}, 2);

    fprintf(stderr, "stage: grad graph\n"); fflush(stderr);
    Term params[] = {w, b};
    Term keep = thvm_grad_multi_keep(ctx, loss, params, 2);

    fprintf(stderr, "stage: eval begin\n"); fflush(stderr);
    Term bundle = thvm_eval(ctx, keep);
    fprintf(stderr, "stage: eval done (tag=%u ext=%u)\n", term_tag(bundle), term_ext(bundle)); fflush(stderr);

    fprintf(stderr, "stage: bundle count\n"); fflush(stderr);
    u32 n = thvm_grad_bundle_count(ctx, bundle);
    fprintf(stderr, "stage: bundle count done n=%u\n", n); fflush(stderr);
    if (term_tag(bundle) == TAG_CTR) {
      u64 bl = term_val(bundle);
      fprintf(stderr, "stage: raw slot0 tag=%u ext=%u val=%llu\n",
              term_tag(heap_read(ctx, bl+0)), term_ext(heap_read(ctx, bl+0)),
              (unsigned long long)term_val(heap_read(ctx, bl+0))); fflush(stderr);
      fprintf(stderr, "stage: raw slot1 tag=%u ext=%u val=%llu\n",
              term_tag(heap_read(ctx, bl+1)), term_ext(heap_read(ctx, bl+1)),
              (unsigned long long)term_val(heap_read(ctx, bl+1))); fflush(stderr);
    }

    if (n > 1) {
      fprintf(stderr, "stage: bundle get 1\n"); fflush(stderr);
      Term g1 = thvm_grad_bundle_get(ctx, bundle, 1);
      fprintf(stderr, "stage: bundle get 1 done tag=%u\n", term_tag(g1)); fflush(stderr);
      fprintf(stderr, "stage: to_host 1\n"); fflush(stderr);
      f32 *db = thvm_to_host(ctx, g1);
      fprintf(stderr, "stage: to_host 1 done db0=%.3f\n", db ? db[0] : -1.0f); fflush(stderr);
    }

    if (n > 0) {
      fprintf(stderr, "stage: bundle get 0\n"); fflush(stderr);
      Term g0 = thvm_grad_bundle_get(ctx, bundle, 0);
      fprintf(stderr, "stage: bundle get 0 done tag=%u\n", term_tag(g0)); fflush(stderr);
      fprintf(stderr, "stage: to_host 0\n"); fflush(stderr);
      f32 *dw = thvm_to_host(ctx, g0);
      fprintf(stderr, "stage: to_host 0 done dw0=%.3f\n", dw ? dw[0] : -1.0f); fflush(stderr);
    }

    fprintf(stderr, "stage: free\n"); fflush(stderr);
    thvm_free(ctx);
    fprintf(stderr, "stage: done\n"); fflush(stderr);
    return 0;
}
