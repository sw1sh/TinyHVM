#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>

int main(void) {
    TinyHVM *ctx = thvm_init("cpu");

    f32 xd[] = {1.0f, 2.0f, 3.0f};
    f32 yd[] = {2.0f, 2.0f, 2.0f};
    Shape s = {.dims = {3}, .rank = 1};

    Term x = thvm_tensor(ctx, xd, s);
    Term y = thvm_tensor(ctx, yd, s);
    Term mul = thvm_op(ctx, UOP_MUL, x, y);
    Term add = thvm_op(ctx, UOP_ADD, mul, y);

    Term out = thvm_eval(ctx, add);
    u32 dtype = DTYPE_F32;
    Shape out_shape = SHAPE(1);
    f32 *vals = thvm_to_host_raw(ctx, out, &dtype, &out_shape);
    if (vals) {
        printf("result = [%.1f, %.1f, %.1f]\n", vals[0], vals[1], vals[2]);
    } else {
        printf("result unavailable\n");
    }

    thvm_free(ctx);
    return 0;
}
