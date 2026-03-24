// test_reduce.m — reduce op tests with axis control
//
// Modeled on tinygrad/test/test_ops.py reduce tests

#define DEVICE "cpu"
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <math.h>

static int n_pass = 0, n_fail = 0;
#define ATOL 1e-4f

static int chk(const char *tag, f32 got, f32 exp, f32 atol) {
    if (fabsf(got - exp) > atol) {
        printf("  FAIL %s: got %.6f, expected %.6f\n", tag, got, exp);
        n_fail++; return 0;
    }
    return 1;
}

static f32 *eval_v(TinyHVM *ctx, Term t, u32 *n) {
    Term r = thvm_reduce(ctx, t);
    if (term_tag(r) != TAG_TEN) { *n = 0; return NULL; }
    *n = ctx->tensors[(u32)term_val(r)].view.numel;
    return thvm_to_host(ctx, r);
}

// ── Sum tests ──────────────────────────────────────────────────

static void test_sum_axis0(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[] = {1,2,3, 4,5,6};
    Term t = thvm_tensor(ctx, d, SHAPE(2, 3));
    u32 ax[] = {0};
    Term r = thvm_sum_axes(ctx, t, ax, 1);
    u32 n; f32 *out = eval_v(ctx, r, &n);
    int ok = (n == 3) && chk("ax0[0]", out[0], 5.f, ATOL)
                      && chk("ax0[1]", out[1], 7.f, ATOL)
                      && chk("ax0[2]", out[2], 9.f, ATOL);
    if (ok) { printf("  test_sum_axis0: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

static void test_sum_axis1(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[] = {1,2,3, 4,5,6};
    Term t = thvm_tensor(ctx, d, SHAPE(2, 3));
    u32 ax[] = {1};
    Term r = thvm_sum_axes(ctx, t, ax, 1);
    u32 n; f32 *out = eval_v(ctx, r, &n);
    int ok = (n == 2) && chk("ax1[0]", out[0], 6.f, ATOL)
                      && chk("ax1[1]", out[1], 15.f, ATOL);
    if (ok) { printf("  test_sum_axis1: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

static void test_sum_all(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[] = {1,2,3, 4,5,6};
    Term t = thvm_tensor(ctx, d, SHAPE(2, 3));
    u32 ax[] = {0, 1};
    Term r = thvm_sum_axes(ctx, t, ax, 2);
    u32 n; f32 *out = eval_v(ctx, r, &n);
    int ok = (n == 1) && chk("all", out[0], 21.f, ATOL);
    if (ok) { printf("  test_sum_all: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

static void test_sum_3d(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[24]; for (int i = 0; i < 24; i++) d[i] = 1.f;
    Term t = thvm_tensor(ctx, d, SHAPE(2, 3, 4));
    u32 ax[] = {1};
    Term r = thvm_sum_axes(ctx, t, ax, 1);
    u32 n; f32 *out = eval_v(ctx, r, &n);
    int ok = (n == 8);
    for (int i = 0; i < 8 && ok; i++)
        ok = chk("3d", out[i], 3.f, ATOL);
    if (ok) { printf("  test_sum_3d: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── Reduce max ─────────────────────────────────────────────────

static void test_rmax(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[] = {1,5,3, 4,2,6};
    Term t = thvm_tensor(ctx, d, SHAPE(2, 3));
    Term r = thvm_op(ctx, UOP_RMAX, t, term_era());
    u32 n; f32 *out = eval_v(ctx, r, &n);
    int ok = (n == 2) && chk("rmax[0]", out[0], 5.f, ATOL)
                      && chk("rmax[1]", out[1], 6.f, ATOL);
    if (ok) { printf("  test_rmax: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

static void test_rmax_negative(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[] = {-3, -1, -5, -2, -4, -1};
    Term t = thvm_tensor(ctx, d, SHAPE(2, 3));
    Term r = thvm_op(ctx, UOP_RMAX, t, term_era());
    u32 n; f32 *out = eval_v(ctx, r, &n);
    int ok = (n == 2) && chk("neg[0]", out[0], -1.f, ATOL)
                      && chk("neg[1]", out[1], -1.f, ATOL);
    if (ok) { printf("  test_rmax_negative: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── Backward: reduce ───────────────────────────────────────────

static void test_grad_sum(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[] = {1, 2, 3, 4, 5, 6};
    Term t = thvm_tensor(ctx, d, SHAPE(2, 3));
    thvm_set_requires_grad(ctx, t);
    u32 ax[] = {0, 1};
    Term loss = thvm_reshape(ctx, thvm_sum_axes(ctx, t, ax, 2), SHAPE(1));

    Term g = thvm_grad(ctx, loss, t);
    u32 n; f32 *gv = eval_v(ctx, g, &n);
    int ok = (n == 6);
    for (int i = 0; i < 6 && ok; i++)
        ok = chk("gs", gv[i], 1.f, ATOL);
    if (ok) { printf("  test_grad_sum: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── Backward: softmax cross-entropy ────────────────────────────

static void test_softmax_grad(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 xd[] = {2.f, 1.f, 0.1f};
    f32 yd[] = {1, 0, 0};
    Term x = thvm_tensor(ctx, xd, SHAPE(1, 3));
    Term y = thvm_tensor(ctx, yd, SHAPE(1, 3));
    thvm_set_requires_grad(ctx, x);

    Term xmax = thvm_op(ctx, UOP_RMAX, x, term_era());
    Term shifted = thvm_op(ctx, UOP_SUB, x, thvm_expand(ctx, xmax, SHAPE(1, 3)));
    Term e = thvm_op(ctx, UOP_EXP, shifted, term_era());
    Term esum = thvm_op(ctx, UOP_SUM, e, term_era());
    Term probs = thvm_op(ctx, UOP_DIV, e, thvm_expand(ctx, esum, SHAPE(1, 3)));

    f32 eps = 1e-7f;
    Term clamped = thvm_op(ctx, UOP_MAX, probs,
                   thvm_expand(ctx, thvm_tensor(ctx, &eps, SHAPE(1,1)), SHAPE(1, 3)));
    Term log_p = thvm_op(ctx, UOP_LOG, clamped, term_era());
    Term masked = thvm_op(ctx, UOP_MUL, y, log_p);
    u32 ax[] = {0, 1};
    Term loss = thvm_op(ctx, UOP_NEG,
                thvm_reshape(ctx, thvm_sum_axes(ctx, masked, ax, 2), SHAPE(1)),
                term_era());

    f32 e0 = expf(0), e1 = expf(-1), e2 = expf(-1.9f);
    f32 es = e0 + e1 + e2;
    f32 expected[] = {e0/es - 1.f, e1/es, e2/es};

    Term gx = thvm_grad(ctx, loss, x);
    u32 n; f32 *gv = eval_v(ctx, gx, &n);
    int ok = (n == 3);
    for (int i = 0; i < 3 && ok; i++)
        ok = chk("sm", gv[i], expected[i], 5e-3f);
    if (ok) { printf("  test_softmax_grad: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── Backward: MSE ──────────────────────────────────────────────

static void test_mse_grad(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 pred[] = {1.f, 2.f, 3.f};
    f32 tgt[] = {1.5f, 2.5f, 2.0f};
    Term p = thvm_tensor(ctx, pred, SHAPE(1, 3));
    Term t = thvm_tensor(ctx, tgt, SHAPE(1, 3));
    thvm_set_requires_grad(ctx, p);

    Term diff = thvm_op(ctx, UOP_SUB, p, t);
    Term sq = thvm_op(ctx, UOP_MUL, diff, diff);
    u32 ax[] = {0, 1};
    Term loss = thvm_reshape(ctx, thvm_sum_axes(ctx, sq, ax, 2), SHAPE(1));

    f32 expected[] = {2*(1.f-1.5f), 2*(2.f-2.5f), 2*(3.f-2.f)};
    Term gp = thvm_grad(ctx, loss, p);
    u32 n; f32 *gv = eval_v(ctx, gp, &n);
    int ok = (n == 3);
    for (int i = 0; i < 3 && ok; i++)
        ok = chk("mse", gv[i], expected[i], ATOL);
    if (ok) { printf("  test_mse_grad: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── Main ───────────────────────────────────────────────────────

int main(void) {
    printf("=== TinyHVM Reduce Tests (tinygrad-style) ===\n\n");

    printf("── Sum ──\n");
    test_sum_axis0();
    test_sum_axis1();
    test_sum_all();
    test_sum_3d();

    printf("\n── Reduce Max ──\n");
    test_rmax();
    test_rmax_negative();

    printf("\n── Backward ──\n");
    test_grad_sum();
    test_softmax_grad();
    test_mse_grad();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
