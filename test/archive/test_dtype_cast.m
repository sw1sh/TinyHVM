// test_dtype_cast.m — typed tensor construction, CAST, and raw readback smoke
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>

int main(void) {
    TinyHVM *ctx = thvm_init("cpu");

    u32 uv = 7;
    Term tu = thvm_tensor_u32(ctx, &uv, SHAPE(1));
    u32 *hu = thvm_to_host_u32(ctx, tu);
    if (hu) printf("u32=%u\n", hu[0]);

    f32 xf[] = {1.9f, -2.1f, 3.0f};
    Term x = thvm_tensor(ctx, xf, (Shape){.dims={3}, .rank=1});
    i32 *xi = thvm_to_host_i32(ctx, thvm_cast(ctx, x, DTYPE_I32));
    if (xi) printf("cast_i32=[%d,%d,%d]\n", xi[0], xi[1], xi[2]);

    i32 iv[] = {3, -4};
    Term ti = thvm_tensor_i32(ctx, iv, (Shape){.dims={2}, .rank=1});
    f32 *hf = thvm_to_host(ctx, thvm_cast(ctx, ti, DTYPE_F32));
    if (hf) printf("cast_f32=[%.1f,%.1f]\n", hf[0], hf[1]);

    i32 aa[] = {10, -20};
    i32 bb[] = {-3, 4};
    Term ta = thvm_tensor_i32(ctx, aa, (Shape){.dims={2}, .rank=1});
    Term tb = thvm_tensor_i32(ctx, bb, (Shape){.dims={2}, .rank=1});
    i32 *hs = thvm_to_host_i32(ctx, thvm_eval(ctx, thvm_op(ctx, UOP_ADD, ta, tb)));
    if (hs) printf("sched_add_i32=[%d,%d]\n", hs[0], hs[1]);

    i32 cc[] = {1, 2, 3};
    Term tc = thvm_tensor_i32(ctx, cc, (Shape){.dims={3}, .rank=1});
    i32 *hr = thvm_to_host_i32(ctx, thvm_eval(ctx, thvm_sum_axes(ctx, tc, (u32[]){0}, 1)));
    if (hr) printf("sched_sum_i32=[%d]\n", hr[0]);

    i32 ones[] = {1, 1, 1};
    Term tones = thvm_tensor_i32(ctx, ones, (Shape){.dims={3}, .rank=1});
    i32 *hca = thvm_to_host_i32(ctx,
        thvm_eval(ctx, thvm_op(ctx, UOP_ADD, thvm_cast(ctx, x, DTYPE_I32), tones)));
    if (hca) printf("sched_cast_add_i32=[%d,%d,%d]\n", hca[0], hca[1], hca[2]);

    i32 condv[] = {0, 1, 0};
    i32 thenv[] = {10, 20, 30};
    i32 elsev[] = {1, 2, 3};
    Term tcond = thvm_tensor_i32(ctx, condv, (Shape){.dims={3}, .rank=1});
    Term tthen = thvm_tensor_i32(ctx, thenv, (Shape){.dims={3}, .rank=1});
    Term telse = thvm_tensor_i32(ctx, elsev, (Shape){.dims={3}, .rank=1});
    i32 *hw = thvm_to_host_i32(ctx, thvm_where(ctx, tcond, tthen, telse));
    if (hw) printf("where_i32=[%d,%d,%d]\n", hw[0], hw[1], hw[2]);

    i32 logitsv[] = {1, 9, 2, 8, 7, 3};
    Term tlogits = thvm_tensor_i32(ctx, logitsv, (Shape){.dims={2,3}, .rank=2});
    u32 *preds = thvm_to_host_u32(ctx, thvm_argmax(ctx, tlogits, 2, 3));
    if (preds) printf("argmax_i32=[%u,%u]\n", preds[0], preds[1]);
    f32 acc = thvm_eval_accuracy(ctx, tlogits, (u8[]){1,0}, 2, 3);
    printf("acc_i32=%.1f\n", acc);

    Term seed = thvm_scalar_typed(ctx, 1.0f, DTYPE_F16);
    u32 seed_dtype = 0;
    void *seed_raw = thvm_to_host_raw(ctx, seed, &seed_dtype, NULL);
    if (seed_raw) printf("seed=%s:%.1f\n", dtype_name(seed_dtype), dtype_load_as_f32(seed_raw, seed_dtype, 0));

    thvm_free(ctx);
    return 0;
}
