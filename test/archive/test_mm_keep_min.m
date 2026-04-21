#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>

int main(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1,2,3,4,5,6}; // [2,3]
    f32 wd[] = {1,2,3,4,5,6}; // [3,2]
    Term x = thvm_tensor(ctx, xd, SHAPE(2,3));
    Term w = thvm_tensor(ctx, wd, SHAPE(3,2));
    thvm_set_requires_grad(ctx, w);
    Term y = thvm_mm(ctx, x, w);
    Term loss = thvm_sum_axes(ctx, y, (u32[]){0,1}, 2);
    Term bundle = thvm_eval(ctx, thvm_grad_keep(ctx, loss, w));
    u32 n = thvm_grad_bundle_count(ctx, bundle);
    printf("bundle_count=%u\n", n);
    if (n) {
        f32 *dw = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 0));
        printf("dw0=%.2f\n", dw ? dw[0] : -1.0f);
    }
    thvm_free(ctx);
    return 0;
}
