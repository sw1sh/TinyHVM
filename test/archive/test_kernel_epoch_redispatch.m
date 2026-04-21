#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <assert.h>
#include <stdio.h>

int main(void) {
    TinyHVM *ctx = thvm_init("cpu");

    f32 wd[] = {1.0f, 2.0f, 3.0f};
    Shape s = {.dims = {3}, .rank = 1};
    Term w = thvm_tensor(ctx, wd, s);
    Term two = thvm_expand(ctx, thvm_scalar(ctx, 2.0f), SHAPE(3));

    // Reuse the same lazy kernel term three times:
    // 1. materialize and cache it
    // 2. consume it again unchanged via ASSIGN source (cache hit expected)
    // 3. revisit it after ASSIGN mutates w (epoch-stale redispatch expected)
    Term k = thvm_op(ctx, UOP_MUL, w, two);
    Term program = thvm_seq(ctx, k,
                   thvm_seq(ctx, thvm_assign(ctx, w, k), k));

    Term out = thvm_eval(ctx, program);
    assert(term_tag(out) == TAG_TEN);

    u32 dtype = DTYPE_F32;
    Shape out_shape = SHAPE(1);
    f32 *out_vals = thvm_to_host_raw(ctx, out, &dtype, &out_shape);
    assert(out_vals);
    printf("result = [%.1f, %.1f, %.1f]\n", out_vals[0], out_vals[1], out_vals[2]);
    assert(out_vals[0] == 4.0f && out_vals[1] == 8.0f && out_vals[2] == 12.0f);

    u32 w_dtype = DTYPE_F32;
    Shape w_shape = SHAPE(1);
    f32 *w_vals = thvm_to_host_raw(ctx, w, &w_dtype, &w_shape);
    assert(w_vals);
    printf("final_w = [%.1f, %.1f, %.1f]\n", w_vals[0], w_vals[1], w_vals[2]);
    assert(w_vals[0] == 2.0f && w_vals[1] == 4.0f && w_vals[2] == 6.0f);

    thvm_free(ctx);
    return 0;
}
