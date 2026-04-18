// test_sum_ref_eval_forward.m — REF body returns pooled (just SUM).
// No grad, no DUP, no MUL. Evaluate the REF, read the result TEN,
// check its shape and values.
//
// If pooled = [1,4] with correct values [M*w[0], ..., M*w[3]]:
//   The SUM itself through ALO is correct. Bug must be downstream
//   (DUP-commute, BG MUL handling).
// If pooled = [1,1] singleton or wrong values:
//   The ALO-realized SUM produces a corrupted TEN. Bug is in SUM
//   realization through ALO.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define M 2
#define N 4

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    setbuf(stdout, NULL);
    TinyHVM *ctx = thvm_init("cpu");

    f32 wd[N] = { 0.10f, -0.20f, 0.30f, 0.40f };
    Term w = thvm_tensor(ctx, wd, SHAPE(N));

    // REF body: λw_p. sum(expand(reshape(w_p, [1,N])), [M,N], axis 0)
    u32 body_id = ctx->def_count++;
    u64 l_w_p = heap_alloc(ctx, 2);
    Term w_p = term_new(TAG_VAR, 0, l_w_p);
    heap_set(ctx, l_w_p + 0, term_set_sub(w_p));
    thvm_hint_shape(ctx, w_p, SHAPE(N));

    Term w_bc = thvm_expand(ctx, thvm_reshape(ctx, w_p, SHAPE(1, N)), SHAPE(M, N));
    Term pooled_body = thvm_sum_axes(ctx, w_bc, (u32[]){0}, 1);

    heap_set(ctx, l_w_p + 1, pooled_body);
    Term lam = term_new(TAG_LAM, 0, l_w_p);
    ctx->defs[body_id] = lam;

    // Call REF, eval to TEN
    Term prog = thvm_app(ctx, thvm_ref(ctx, body_id), w);
    Term result = thvm_eval(ctx, prog);

    printf("result tag=%u\n", (u32)term_tag(result));
    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    f32 *out = thvm_to_host_raw(ctx, result, &dtype, &shp);
    if (!out) { printf("FAIL: result not readable\n"); return 1; }

    printf("shape = [");
    for (u32 i = 0; i < shp.rank; i++) printf("%s%u", i?",":"", shp.dims[i]);
    printf("]\n");

    printf("values: ");
    u32 total = 1;
    for (u32 i = 0; i < shp.rank; i++) total *= shp.dims[i];
    for (u32 i = 0; i < total; i++) printf("%8.4f ", out[i]);
    printf("\n");

    // Expected: [1, 4] with pooled[0, j] = M * w[j]
    f32 exp_vals[N];
    for (u32 j = 0; j < N; j++) exp_vals[j] = (f32)M * wd[j];

    printf("expect: ");
    for (u32 j = 0; j < N; j++) printf("%8.4f ", exp_vals[j]);
    printf("\n");

    int ok = 1;
    if (shp.rank != 2 || shp.dims[0] != 1 || shp.dims[1] != N) {
        printf("FAIL: shape mismatch\n");
        ok = 0;
    } else {
        for (u32 j = 0; j < N; j++) {
            if (fabsf(out[j] - exp_vals[j]) > 1e-4f) { ok = 0; break; }
        }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    thvm_free(ctx);
    return ok ? 0 : 1;
}
