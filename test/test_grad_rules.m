// test_grad_rules.m — per-rule GRAD/TOP correctness.
//
// Each rule exercised in isolation:
//   1. Build small forward with requires_grad leaves.
//   2. Wrap in sum() to get a scalar loss (so the seed is 1.0).
//   3. thvm_grad_keep / thvm_grad_multi_keep, eval, read bundle.
//   4. Assert per-element against the analytical gradient.
//
// When a rule is broken, the relevant sub-test fails loudly with
// "got vs expected" so the owning rule is obvious.
//
// Each sub-test is hermetic: its own TinyHVM context, tiny tensors,
// no shared state. A failure in one rule does not cascade.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <math.h>
#include <stdio.h>
#include <string.h>

#define TOL 1e-4f

static int almost_eq(const f32 *got, const f32 *exp, u32 n, f32 tol) {
    for (u32 i = 0; i < n; i++) {
        f32 d = got[i] - exp[i];
        if (d < 0) d = -d;
        if (d > tol) return 0;
    }
    return 1;
}

static int report(const char *name, const f32 *got, const f32 *exp, u32 n) {
    int ok = almost_eq(got, exp, n, TOL);
    printf("%-12s %s", name, ok ? "PASS" : "FAIL");
    if (!ok) {
        printf("\n  got: ");
        for (u32 i = 0; i < n; i++) printf("%.4f ", got[i]);
        printf("\n  exp: ");
        for (u32 i = 0; i < n; i++) printf("%.4f ", exp[i]);
    }
    printf("\n");
    return ok ? 0 : 1;
}

// Helper: run grad_keep on a single-param loss, copy bundle slot 0.
static int read_grad_1(TinyHVM *ctx, Term loss, Term p, f32 *out, u32 n) {
    Term bundle = thvm_eval(ctx, thvm_grad_keep(ctx, loss, p));
    if (thvm_grad_bundle_count(ctx, bundle) != 1) return -1;
    Term g = thvm_grad_bundle_get(ctx, bundle, 0);
    f32 *d = thvm_to_host(ctx, g);
    if (!d) return -1;
    memcpy(out, d, n * sizeof(f32));
    return 0;
}

// Helper: sum all axes of an op result so the loss is scalar.
static Term sum_all(TinyHVM *ctx, Term y, u32 rank) {
    u32 axes[8]; for (u32 i = 0; i < rank; i++) axes[i] = i;
    return thvm_sum_axes(ctx, y, axes, rank);
}

// ─── binary elementwise ─────────────────────────────────────────────

static int test_add(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_ADD, a, b);
    Term loss = sum_all(ctx, y, 1);
    f32 got[3]; if (read_grad_1(ctx, loss, a, got, 3) < 0) { printf("ADD read FAIL\n"); thvm_free(ctx); return 1; }
    f32 exp[3] = {1, 1, 1};  // d(sum(a+b))/da = 1
    int r = report("ADD", got, exp, 3);
    thvm_free(ctx); return r;
}

static int test_sub(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);
    Term y = thvm_op(ctx, UOP_SUB, a, b);
    Term loss = sum_all(ctx, y, 1);
    Term params[] = {a, b};
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));
    int fails = 0;
    if (thvm_grad_bundle_count(ctx, bundle) != 2) { printf("SUB count FAIL\n"); thvm_free(ctx); return 1; }
    f32 *ga = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 0));
    f32 *gb = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 1));
    f32 ea[3] = {1,1,1}, eb[3] = {-1,-1,-1};
    fails += report("SUB.a", ga, ea, 3);
    fails += report("SUB.b", gb, eb, 3);
    thvm_free(ctx); return fails;
}

static int test_mul(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);
    Term y = thvm_op(ctx, UOP_MUL, a, b);
    Term loss = sum_all(ctx, y, 1);
    Term params[] = {a, b};
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));
    int fails = 0;
    f32 *ga = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 0));
    f32 *gb = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 1));
    // d(sum(a*b))/da = b, /db = a
    fails += report("MUL.a", ga, bd, 3);
    fails += report("MUL.b", gb, ad, 3);
    thvm_free(ctx); return fails;
}

static int test_div(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {2, 4, 6}, bd[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);
    Term y = thvm_op(ctx, UOP_DIV, a, b);
    Term loss = sum_all(ctx, y, 1);
    Term params[] = {a, b};
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));
    int fails = 0;
    f32 *ga = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 0));
    f32 *gb = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 1));
    f32 ea[3], eb[3];
    for (int i = 0; i < 3; i++) { ea[i] = 1.0f/bd[i]; eb[i] = -ad[i]/(bd[i]*bd[i]); }
    fails += report("DIV.a", ga, ea, 3);
    fails += report("DIV.b", gb, eb, 3);
    thvm_free(ctx); return fails;
}

static int test_max(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 5, 3}, bd[] = {4, 2, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);
    Term y = thvm_op(ctx, UOP_MAX, a, b);
    Term loss = sum_all(ctx, y, 1);
    Term params[] = {a, b};
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));
    int fails = 0;
    f32 *ga = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 0));
    f32 *gb = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 1));
    // where a>=b: ga=1, gb=0; else ga=0, gb=1
    f32 ea[3] = {0, 1, 0}, eb[3] = {1, 0, 1};
    fails += report("MAX.a", ga, ea, 3);
    fails += report("MAX.b", gb, eb, 3);
    thvm_free(ctx); return fails;
}

// ─── unary elementwise ──────────────────────────────────────────────

static int test_neg(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_NEG, a, term_era());
    Term loss = sum_all(ctx, y, 1);
    f32 got[3]; read_grad_1(ctx, loss, a, got, 3);
    f32 exp[3] = {-1, -1, -1};
    int r = report("NEG", got, exp, 3);
    thvm_free(ctx); return r;
}

static int test_exp(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {0, 1, 2};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_EXP, a, term_era());
    Term loss = sum_all(ctx, y, 1);
    f32 got[3]; read_grad_1(ctx, loss, a, got, 3);
    f32 exp[3] = {expf(0), expf(1), expf(2)};
    int r = report("EXP", got, exp, 3);
    thvm_free(ctx); return r;
}

static int test_log(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 4};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_LOG, a, term_era());
    Term loss = sum_all(ctx, y, 1);
    f32 got[3]; read_grad_1(ctx, loss, a, got, 3);
    f32 exp[3] = {1.0f, 0.5f, 0.25f};
    int r = report("LOG", got, exp, 3);
    thvm_free(ctx); return r;
}

static int test_sqrt(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 4, 9};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_SQRT, a, term_era());
    Term loss = sum_all(ctx, y, 1);
    f32 got[3]; read_grad_1(ctx, loss, a, got, 3);
    f32 exp[3] = {0.5f, 0.25f, 1.0f/6.0f};
    int r = report("SQRT", got, exp, 3);
    thvm_free(ctx); return r;
}

static int test_relu(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {-1, 0, 1, 2};
    Term a = thvm_tensor(ctx, ad, SHAPE(4));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_RELU, a, term_era());
    Term loss = sum_all(ctx, y, 1);
    f32 got[4]; read_grad_1(ctx, loss, a, got, 4);
    f32 exp[4] = {0, 0, 1, 1};
    int r = report("RELU", got, exp, 4);
    thvm_free(ctx); return r;
}

// ─── reductions ─────────────────────────────────────────────────────

static int test_sum(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3, 4};
    Term a = thvm_tensor(ctx, ad, SHAPE(4));
    thvm_set_requires_grad(ctx, a);
    Term loss = thvm_sum_axes(ctx, a, (u32[]){0}, 1);
    f32 got[4]; read_grad_1(ctx, loss, a, got, 4);
    f32 exp[4] = {1, 1, 1, 1};
    int r = report("SUM", got, exp, 4);
    thvm_free(ctx); return r;
}

// ─── movement ops ──────────────────────────────────────────────────

static int test_reshape(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3, 4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(6));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_reshape(ctx, a, SHAPE(2, 3));
    Term loss = sum_all(ctx, y, 2);
    f32 got[6]; read_grad_1(ctx, loss, a, got, 6);
    f32 exp[6] = {1, 1, 1, 1, 1, 1};
    int r = report("RESHAPE", got, exp, 6);
    thvm_free(ctx); return r;
}

static int test_expand(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_expand(ctx, thvm_reshape(ctx, a, SHAPE(1, 3)), SHAPE(4, 3));
    Term loss = sum_all(ctx, y, 2);
    f32 got[3]; read_grad_1(ctx, loss, a, got, 3);
    f32 exp[3] = {4, 4, 4};  // expanded 4x along axis 0
    int r = report("EXPAND", got, exp, 3);
    thvm_free(ctx); return r;
}

// ─── non-differentiable ───────────────────────────────────────────

static int test_cmp(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {2, 2, 2};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_CMP, a, b);
    Term loss = sum_all(ctx, y, 1);
    f32 got[3]; read_grad_1(ctx, loss, a, got, 3);
    f32 exp[3] = {0, 0, 0};
    int r = report("CMP", got, exp, 3);
    thvm_free(ctx); return r;
}

int main(void) {
    int fails = 0;
    // binary
    fails += test_add();
    fails += test_sub();
    fails += test_mul();
    fails += test_div();
    fails += test_max();
    // unary
    fails += test_neg();
    fails += test_exp();
    fails += test_log();
    fails += test_sqrt();
    fails += test_relu();
    // reductions
    fails += test_sum();
    // movement
    fails += test_reshape();
    fails += test_expand();
    // non-diff
    fails += test_cmp();

    printf("\ntotal failures: %d\n", fails);
    return fails ? 1 : 0;
}
