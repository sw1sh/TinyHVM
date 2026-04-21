// test_sum_sq_forward_ref.m — forward-only REF body that computes
// X = sum_axes(expand(w), {0}) then returns X * X.
// No grad, no MAT-CTR. If this returns wrong values (scalar-broadcast
// instead of per-element), the bug is PURE IC — in how a REF body
// realizes MUL(X, X) when X is a SUM output.

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

    u32 body_id = ctx->def_count++;
    u64 l_w_p = heap_alloc(ctx, 2);
    Term w_p = term_new(TAG_VAR, 0, l_w_p);
    heap_set(ctx, l_w_p + 0, term_set_sub(w_p));
    thvm_hint_shape(ctx, w_p, SHAPE(N));

    // X = sum_axes(expand(reshape(w, [1,N]), [M,N]), {0})  → shape [1,N]
    // result = X * X  → shape [1,N]
    Term w_bc = thvm_expand(ctx, thvm_reshape(ctx, w_p, SHAPE(1, N)), SHAPE(M, N));
    Term pooled = thvm_sum_axes(ctx, w_bc, (u32[]){0}, 1);   // [1, N]
    Term sq = thvm_op(ctx, UOP_MUL, pooled, pooled);
    heap_set(ctx, l_w_p + 1, sq);
    Term lam = term_new(TAG_LAM, 0, l_w_p);
    ctx->defs[body_id] = lam;

    Term prog = thvm_app(ctx, thvm_ref(ctx, body_id), w);
    Term result = thvm_eval(ctx, prog);
    printf("result_tag=%u val=%llu\n",
           (u32)term_tag(result), (unsigned long long)term_val(result));

    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    f32 *out = thvm_to_host_raw(ctx, result, &dtype, &shp);
    if (!out) { printf("FAIL: result not readable\n"); return 1; }

    // Expected: pooled[0, j] = M * w[j]. sq[0, j] = (M * w[j])^2.
    f32 exp[N];
    for (u32 j = 0; j < N; j++) {
        f32 p = (f32)M * wd[j];
        exp[j] = p * p;
    }

    printf("sq:      ");
    for (u32 j = 0; j < N; j++) printf("%8.4f ", out[j]);
    printf("\nexpect:  ");
    for (u32 j = 0; j < N; j++) printf("%8.4f ", exp[j]);
    printf("\n");

    int ok = 1;
    for (u32 j = 0; j < N; j++)
        if (fabsf(out[j] - exp[j]) > 1e-4f) { ok = 0; break; }
    printf("%s\n", ok ? "PASS" : "FAIL");
    thvm_free(ctx);
    return ok ? 0 : 1;
}
