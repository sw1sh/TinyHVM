// test_sum_sq_ref_scale_fwd.m — forward-only version of
// test_sum_sq_ref_scale. REF body returns the SCALED loss directly,
// no grad or MAT-CTR. Verifies the scaled forward computation in
// REF+ALO produces correct value.
//
// If correct: forward is fine in scaled REF; halving bug is backward-only.
// If wrong: the scaling MUL inside REF+ALO is miscomputing at forward
//   time, and the half-magnitude in grad is a downstream symptom.

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

    // REF body: λw_p. { pooled = sum(expand(reshape(w_p))); sq = pooled*pooled;
    //                   raw_loss = sum(sq); reshape; MUL(const 0.5) }
    u32 body_id = ctx->def_count++;
    u64 l_w_p = heap_alloc(ctx, 2);
    Term w_p = term_new(TAG_VAR, 0, l_w_p);
    heap_set(ctx, l_w_p + 0, term_set_sub(w_p));
    thvm_hint_shape(ctx, w_p, SHAPE(N));

    Term w_bc = thvm_expand(ctx, thvm_reshape(ctx, w_p, SHAPE(1, N)), SHAPE(M, N));
    Term pooled = thvm_sum_axes(ctx, w_bc, (u32[]){0}, 1);
    Term sq = thvm_op(ctx, UOP_MUL, pooled, pooled);
    Term raw_loss = thvm_sum_axes(ctx, sq, (u32[]){0, 1}, 2);
    Term loss_rs = thvm_reshape(ctx, raw_loss, SHAPE(1, 1));
    Term half_const = thvm_reshape(ctx, thvm_scalar(ctx, 0.5f), SHAPE(1, 1));
    Term scaled_loss_body = thvm_op(ctx, UOP_MUL, loss_rs, half_const);

    heap_set(ctx, l_w_p + 1, scaled_loss_body);
    Term lam = term_new(TAG_LAM, 0, l_w_p);
    ctx->defs[body_id] = lam;

    Term prog = thvm_app(ctx, thvm_ref(ctx, body_id), w);
    Term result = thvm_eval(ctx, prog);

    printf("result tag=%u\n", (u32)term_tag(result));
    u32 dtype = DTYPE_F32;
    Shape shp = SHAPE(1);
    f32 *out = thvm_to_host_raw(ctx, result, &dtype, &shp);
    if (!out) { printf("FAIL: result not readable\n"); return 1; }

    // Expected: sum((M*w[j])^2) * 0.5 = 0.5 * M^2 * sum(w^2)
    f32 exp_val = 0.0f;
    for (u32 j = 0; j < N; j++) exp_val += wd[j] * wd[j];
    exp_val *= (f32)M * (f32)M * 0.5f;

    printf("result = %.6f (shape=[", out[0]);
    for (u32 i = 0; i < shp.rank; i++) printf("%s%u", i?",":"", shp.dims[i]);
    printf("])\n");
    printf("expect = %.6f\n", exp_val);

    int ok = fabsf(out[0] - exp_val) < 1e-4f;
    printf("%s\n", ok ? "PASS" : "FAIL");
    thvm_free(ctx);
    return ok ? 0 : 1;
}
