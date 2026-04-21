#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <assert.h>
#include <stdio.h>

int main(void) {
    TinyHVM *ctx = thvm_init("cpu");

    f32 x[] = {1,2,3,4,5,6,7,8,9};
    Term tx = thvm_tensor(ctx, x, (Shape){.dims={1,1,3,3}, .rank=4});

    // Force the fuser to carry a non-trivial ShapeTracker through lowering.
    Term pooled = thvm_pool(ctx, tx, (u32[]){2,2}, (u32[]){1,1}, 2);
    Term x_rs = thvm_reshape(ctx, pooled, shape_of((u32[]){1,1,1,1,2,2,2,2}, 8));
    Term doubled = thvm_op(ctx, UOP_ADD, x_rs, x_rs);
    Term total = thvm_sum_axes(ctx, doubled, (u32[]){0,1,2,3,4,5,6,7}, 8);
    total = thvm_reshape(ctx, total, SHAPE(1));

    Term out = thvm_eval(ctx, total);
    if (term_tag(out) != TAG_TEN)
        out = thvm_force_tensor_term(ctx, out);
    assert(term_tag(out) == TAG_TEN);

    u32 dtype = DTYPE_F32;
    Shape out_shape = SHAPE(1);
    f32 *vals = thvm_to_host_raw(ctx, out, &dtype, &out_shape);
    assert(vals);
    printf("multiview_total = %.1f\n", vals[0]);
    assert(vals[0] == 160.0f);

    thvm_free(ctx);
    return 0;
}
