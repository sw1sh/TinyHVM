// test_dup_sum_forward_ref.m — REF body with EXPLICIT user DUP on a
// SUM output, then MUL of the two DUP projections. Forward-only,
// no grad. Does ALO-realized DUP⊳SUM commute produce correct values?
//
// REF body: λw. { pooled = sum(expand(w), {0});
//                 dup pooled into (dp0, dp1);
//                 return dp0 * dp1 }
// Returns shape [1, N] with each element = pooled[0, j]^2 = (M*w[j])^2
//
// If PASSES: user-written DUP⊳SUM works in ALO context. Bug is
//            specifically in grad's BG rule's DUP structure.
// If FAILS: DUP⊳SUM commute is broken in ALO context independent
//           of grad. The bug is at the IC rule level.

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

    // REF body builds: expand(w) -> sum -> dup -> mul(dp0, dp1)
    u32 body_id = ctx->def_count++;
    u64 l_w_p = heap_alloc(ctx, 2);
    Term w_p = term_new(TAG_VAR, 0, l_w_p);
    heap_set(ctx, l_w_p + 0, term_set_sub(w_p));
    thvm_hint_shape(ctx, w_p, SHAPE(N));

    Term w_bc = thvm_expand(ctx, thvm_reshape(ctx, w_p, SHAPE(1, N)), SHAPE(M, N));
    Term pooled = thvm_sum_axes(ctx, w_bc, (u32[]){0}, 1);    // [1, N]
    Term p0, p1;
    thvm_dup(ctx, thvm_fresh_label(ctx), pooled, &p0, &p1);
    Term sq = thvm_op(ctx, UOP_MUL, p0, p1);

    heap_set(ctx, l_w_p + 1, sq);
    Term lam = term_new(TAG_LAM, 0, l_w_p);
    ctx->defs[body_id] = lam;

    Term prog = thvm_app(ctx, thvm_ref(ctx, body_id), w);
    Term result = thvm_eval(ctx, prog);

    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    f32 *out = thvm_to_host_raw(ctx, result, &dtype, &shp);
    if (!out) { printf("FAIL: result not readable, tag=%u\n", term_tag(result)); return 1; }

    printf("shape = [");
    for (u32 i = 0; i < shp.rank; i++) printf("%s%u", i?",":"", shp.dims[i]);
    printf("]\n");

    // Expected: pooled[0, j] = M * w[j]. sq[0, j] = (M*w[j])^2.
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
