// test_tiny_single_param_keep.m — explicit gradient bundle readback
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>

int main(void) {
    TinyHVM *ctx = thvm_init("cpu");

    f32 xd[] = {1,2,3, 4,5,6};
    f32 wd[] = {0.1f, 0.2f, 0.3f};
    Term x = thvm_tensor(ctx, xd, (Shape){.dims={2,3}, .rank=2});
    Term w = thvm_tensor(ctx, wd, (Shape){.dims={3}, .rank=1});
    thvm_set_requires_grad(ctx, w);

    Term h = thvm_op(ctx, UOP_MUL, x, w);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    Term loss = thvm_sum_axes(ctx, h, (u32[]){0, 1}, 2);

    Term bundle = thvm_eval(ctx, thvm_grad_keep(ctx, loss, w));
    if (thvm_grad_bundle_count(ctx, bundle) == 1) {
        f32 *dw = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 0));
        if (dw) printf("grad_w = [%.2f, %.2f, %.2f]\n", dw[0], dw[1], dw[2]);
    }

    thvm_free(ctx);
    return 0;
}
