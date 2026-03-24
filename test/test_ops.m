// test_ops.m — tinygrad-style op tests for TinyHVM
//
// For each op, tests:
//   1. Forward pass: compare output against expected values
//   2. Backward pass: compare gradient against expected values
//
// Modeled on tinygrad/test/test_ops.py

#define DEVICE "cpu"
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <math.h>

// ── Helpers ─────────────────────────────────────────────────────

static int n_pass = 0, n_fail = 0;

#define ATOL 1e-4f

static int check_close(const char *name, f32 got, f32 expected, f32 atol) {
    f32 diff = fabsf(got - expected);
    if (diff > atol) {
        printf("  FAIL %s: got %.6f, expected %.6f (diff=%.6f)\n", name, got, expected, diff);
        n_fail++;
        return 0;
    }
    return 1;
}

// Reduce a term, return the f32 scalar value
static f32 eval_scalar(TinyHVM *ctx, Term t) {
    Term r = thvm_reduce(ctx, t);
    if (term_tag(r) != TAG_TEN) return NAN;
    return thvm_to_host(ctx, r)[0];
}

// Evaluate tensor, return pointer to values (caller reads numel elements)
static f32 *eval_buf(TinyHVM *ctx, Term t, u32 *out_numel) {
    Term r = thvm_reduce(ctx, t);
    if (term_tag(r) != TAG_TEN) { *out_numel = 0; return NULL; }
    u32 tid = (u32)term_val(r);
    *out_numel = ctx->tensors[tid].view.numel;
    return thvm_to_host(ctx, r);
}

// Create a fresh ctx for each test
static TinyHVM *fresh_ctx(void) {
    return thvm_init(thvm_device(DEVICE));
}

// ── Forward-Only Tests ─────────────────────────────────────────

// test_add: [1,2,3] + [4,5,6] = [5,7,9]
static void test_add(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,2,3}, b[] = {4,5,6};
    Term ta = thvm_tensor(ctx, a, SHAPE(3));
    Term tb = thvm_tensor(ctx, b, SHAPE(3));
    Term result = thvm_op(ctx, UOP_ADD, ta, tb);
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    int ok = (n == 3) && check_close("add[0]", out[0], 5.f, ATOL)
                      && check_close("add[1]", out[1], 7.f, ATOL)
                      && check_close("add[2]", out[2], 9.f, ATOL);
    if (ok) { printf("  test_add: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_sub: [5,7,9] - [4,5,6] = [1,2,3]
static void test_sub(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {5,7,9}, b[] = {4,5,6};
    Term ta = thvm_tensor(ctx, a, SHAPE(3));
    Term tb = thvm_tensor(ctx, b, SHAPE(3));
    Term result = thvm_op(ctx, UOP_SUB, ta, tb);
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    int ok = (n == 3) && check_close("sub[0]", out[0], 1.f, ATOL)
                      && check_close("sub[1]", out[1], 2.f, ATOL)
                      && check_close("sub[2]", out[2], 3.f, ATOL);
    if (ok) { printf("  test_sub: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_mul: [2,3,4] * [5,6,7] = [10,18,28]
static void test_mul(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {2,3,4}, b[] = {5,6,7};
    Term ta = thvm_tensor(ctx, a, SHAPE(3));
    Term tb = thvm_tensor(ctx, b, SHAPE(3));
    Term result = thvm_op(ctx, UOP_MUL, ta, tb);
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    int ok = (n == 3) && check_close("mul[0]", out[0], 10.f, ATOL)
                      && check_close("mul[1]", out[1], 18.f, ATOL)
                      && check_close("mul[2]", out[2], 28.f, ATOL);
    if (ok) { printf("  test_mul: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_div: [10,18,28] / [5,6,7] = [2,3,4]
static void test_div(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {10,18,28}, b[] = {5,6,7};
    Term ta = thvm_tensor(ctx, a, SHAPE(3));
    Term tb = thvm_tensor(ctx, b, SHAPE(3));
    Term result = thvm_op(ctx, UOP_DIV, ta, tb);
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    int ok = (n == 3) && check_close("div[0]", out[0], 2.f, ATOL)
                      && check_close("div[1]", out[1], 3.f, ATOL)
                      && check_close("div[2]", out[2], 4.f, ATOL);
    if (ok) { printf("  test_div: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_neg: -[1,-2,3] = [-1,2,-3]
static void test_neg(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,-2,3};
    Term ta = thvm_tensor(ctx, a, SHAPE(3));
    Term result = thvm_op(ctx, UOP_NEG, ta, term_era());
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    int ok = (n == 3) && check_close("neg[0]", out[0], -1.f, ATOL)
                      && check_close("neg[1]", out[1],  2.f, ATOL)
                      && check_close("neg[2]", out[2], -3.f, ATOL);
    if (ok) { printf("  test_neg: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_relu: relu([-2,-1,0,1,2]) = [0,0,0,1,2]
static void test_relu(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {-2,-1,0,1,2};
    Term ta = thvm_tensor(ctx, a, SHAPE(5));
    Term result = thvm_op(ctx, UOP_RELU, ta, term_era());
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    int ok = (n == 5) && check_close("relu[0]", out[0], 0.f, ATOL)
                      && check_close("relu[1]", out[1], 0.f, ATOL)
                      && check_close("relu[2]", out[2], 0.f, ATOL)
                      && check_close("relu[3]", out[3], 1.f, ATOL)
                      && check_close("relu[4]", out[4], 2.f, ATOL);
    if (ok) { printf("  test_relu: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_exp: exp([0, 1, -1]) ≈ [1.0, 2.71828, 0.36788]
static void test_exp(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {0.f, 1.f, -1.f};
    Term ta = thvm_tensor(ctx, a, SHAPE(3));
    Term result = thvm_op(ctx, UOP_EXP, ta, term_era());
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    int ok = (n == 3) && check_close("exp[0]", out[0], 1.f, ATOL)
                      && check_close("exp[1]", out[1], expf(1.f), ATOL)
                      && check_close("exp[2]", out[2], expf(-1.f), ATOL);
    if (ok) { printf("  test_exp: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_log: log([1, e, e^2]) ≈ [0, 1, 2]
static void test_log(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1.f, expf(1.f), expf(2.f)};
    Term ta = thvm_tensor(ctx, a, SHAPE(3));
    Term result = thvm_op(ctx, UOP_LOG, ta, term_era());
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    int ok = (n == 3) && check_close("log[0]", out[0], 0.f, ATOL)
                      && check_close("log[1]", out[1], 1.f, ATOL)
                      && check_close("log[2]", out[2], 2.f, ATOL);
    if (ok) { printf("  test_log: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_sqrt: sqrt([1, 4, 9, 16]) = [1, 2, 3, 4]
static void test_sqrt(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,4,9,16};
    Term ta = thvm_tensor(ctx, a, SHAPE(4));
    Term result = thvm_op(ctx, UOP_SQRT, ta, term_era());
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    int ok = (n == 4) && check_close("sqrt[0]", out[0], 1.f, ATOL)
                      && check_close("sqrt[1]", out[1], 2.f, ATOL)
                      && check_close("sqrt[2]", out[2], 3.f, ATOL)
                      && check_close("sqrt[3]", out[3], 4.f, ATOL);
    if (ok) { printf("  test_sqrt: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_max: max([1,5,3], [4,2,6]) = [4,5,6]
static void test_max(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,5,3}, b[] = {4,2,6};
    Term ta = thvm_tensor(ctx, a, SHAPE(3));
    Term tb = thvm_tensor(ctx, b, SHAPE(3));
    Term result = thvm_op(ctx, UOP_MAX, ta, tb);
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    int ok = (n == 3) && check_close("max[0]", out[0], 4.f, ATOL)
                      && check_close("max[1]", out[1], 5.f, ATOL)
                      && check_close("max[2]", out[2], 6.f, ATOL);
    if (ok) { printf("  test_max: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_cmp: [1,5,3] > [4,2,6] = [0,1,0]
static void test_cmp(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,5,3}, b[] = {4,2,6};
    Term ta = thvm_tensor(ctx, a, SHAPE(3));
    Term tb = thvm_tensor(ctx, b, SHAPE(3));
    Term result = thvm_op(ctx, UOP_CMP, ta, tb);
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    int ok = (n == 3) && check_close("cmp[0]", out[0], 0.f, ATOL)
                      && check_close("cmp[1]", out[1], 1.f, ATOL)
                      && check_close("cmp[2]", out[2], 0.f, ATOL);
    if (ok) { printf("  test_cmp: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── Reduce Tests ───────────────────────────────────────────────

// test_sum: sum([[1,2],[3,4]], axis=1) = [3, 7]
static void test_sum(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,2,3,4};
    Term ta = thvm_tensor(ctx, a, SHAPE(2,2));
    u32 axes[] = {1};
    Term result = thvm_sum_axes(ctx, ta, axes, 1);
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    int ok = (n == 2) && check_close("sum[0]", out[0], 3.f, ATOL)
                      && check_close("sum[1]", out[1], 7.f, ATOL);
    if (ok) { printf("  test_sum: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_sum_all: sum([[1,2],[3,4]]) over all axes = 10
static void test_sum_all(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,2,3,4};
    Term ta = thvm_tensor(ctx, a, SHAPE(2,2));
    u32 axes[] = {0, 1};
    Term result = thvm_sum_axes(ctx, ta, axes, 2);
    f32 v = eval_scalar(ctx, result);
    if (check_close("sum_all", v, 10.f, ATOL)) {
        printf("  test_sum_all: PASS\n"); n_pass++;
    }
    thvm_free(ctx);
}

// test_rmax: max([[1,4],[3,2]], axis=1) = [4, 3] (reduce max)
static void test_rmax(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,4,3,2};
    Term ta = thvm_tensor(ctx, a, SHAPE(2,2));
    Term result = thvm_op(ctx, UOP_RMAX, ta, term_era());
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    // RMAX reduces last axis by default → [2,1]
    int ok = (n == 2) && check_close("rmax[0]", out[0], 4.f, ATOL)
                      && check_close("rmax[1]", out[1], 3.f, ATOL);
    if (ok) { printf("  test_rmax: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── Movement Tests ─────────────────────────────────────────────

// test_reshape: reshape([1,2,3,4,5,6], (2,3)) → [[1,2,3],[4,5,6]]
static void test_reshape(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,2,3,4,5,6};
    Term ta = thvm_tensor(ctx, a, SHAPE(6));
    Term result = thvm_reshape(ctx, ta, SHAPE(2, 3));
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    int ok = (n == 6);
    for (int i = 0; i < 6 && ok; i++)
        ok = check_close("reshape", out[i], (f32)(i+1), ATOL);
    if (ok) { printf("  test_reshape: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_permute: [[1,2,3],[4,5,6]] transposed → [[1,4],[2,5],[3,6]]
static void test_permute(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,2,3,4,5,6};
    Term ta = thvm_tensor(ctx, a, SHAPE(2, 3));
    u32 axes[] = {1, 0};
    Term result = thvm_permute(ctx, ta, axes, 2);
    // Force contiguous: permute only rearranges strides; add 0 materializes
    f32 zero = 0.f;
    result = thvm_op(ctx, UOP_ADD, result, 
             thvm_expand(ctx, thvm_tensor(ctx, &zero, SHAPE(1,1)), SHAPE(3, 2)));
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    f32 expected[] = {1,4,2,5,3,6};
    int ok = (n == 6);
    for (int i = 0; i < 6 && ok; i++)
        ok = check_close("permute", out[i], expected[i], ATOL);
    if (ok) { printf("  test_permute: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_expand: expand([1,2,3] shape (1,3) → (4,3)) = broadcast rows
static void test_expand(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,2,3};
    Term ta = thvm_tensor(ctx, a, SHAPE(1, 3));
    Term result = thvm_expand(ctx, ta, SHAPE(4, 3));
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    int ok = (n == 12);
    for (int r = 0; r < 4 && ok; r++)
        for (int c = 0; c < 3 && ok; c++)
            ok = check_close("expand", out[r*3+c], (f32)(c+1), ATOL);
    if (ok) { printf("  test_expand: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── Matmul Tests ───────────────────────────────────────────────

// test_matmul: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
static void test_matmul(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,2,3,4}, b[] = {5,6,7,8};
    Term ta = thvm_tensor(ctx, a, SHAPE(2, 2));
    Term tb = thvm_tensor(ctx, b, SHAPE(2, 2));
    Term result = thvm_op(ctx, UOP_MM, ta, tb);
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    f32 expected[] = {19,22,43,50};
    int ok = (n == 4);
    for (int i = 0; i < 4 && ok; i++)
        ok = check_close("mm", out[i], expected[i], ATOL);
    if (ok) { printf("  test_matmul: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_matmul_rect: (2,3) @ (3,2) = (2,2)
static void test_matmul_rect(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,2,3, 4,5,6};       // [2,3]
    f32 b[] = {7,8, 9,10, 11,12};   // [3,2]
    Term ta = thvm_tensor(ctx, a, SHAPE(2, 3));
    Term tb = thvm_tensor(ctx, b, SHAPE(3, 2));
    Term result = thvm_op(ctx, UOP_MM, ta, tb);
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    // row0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
    // row1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
    f32 expected[] = {58,64,139,154};
    int ok = (n == 4);
    for (int i = 0; i < 4 && ok; i++)
        ok = check_close("mm_rect", out[i], expected[i], ATOL);
    if (ok) { printf("  test_matmul_rect: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── Broadcast Tests ────────────────────────────────────────────

// test_broadcast_add: (2,3) + (1,3) → broadcasts row
static void test_broadcast_add(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,2,3, 4,5,6};  // [2,3]
    f32 b[] = {10,20,30};       // [1,3]
    Term ta = thvm_tensor(ctx, a, SHAPE(2, 3));
    Term tb = thvm_tensor(ctx, b, SHAPE(1, 3));
    Term bc = thvm_expand(ctx, tb, SHAPE(2, 3));
    Term result = thvm_op(ctx, UOP_ADD, ta, bc);
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    f32 expected[] = {11,22,33, 14,25,36};
    int ok = (n == 6);
    for (int i = 0; i < 6 && ok; i++)
        ok = check_close("bcast_add", out[i], expected[i], ATOL);
    if (ok) { printf("  test_broadcast_add: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── Gradient Tests ─────────────────────────────────────────────

// test_grad_add: z = sum(a + b), dz/da = [1,1,1], dz/db = [1,1,1]
static void test_grad_add(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,2,3}, b[] = {4,5,6};
    Term ta = thvm_tensor(ctx, a, SHAPE(1,3));
    Term tb = thvm_tensor(ctx, b, SHAPE(1,3));
    thvm_set_requires_grad(ctx, ta);
    thvm_set_requires_grad(ctx, tb);
    Term z = thvm_op(ctx, UOP_ADD, ta, tb);
    u32 axes[] = {0, 1};
    Term loss = thvm_sum_axes(ctx, z, axes, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1));

    Term ga = thvm_grad(ctx, loss, ta);
    Term gb = thvm_grad(ctx, loss, tb);
    u32 na, nb; f32 *ga_v = eval_buf(ctx, ga, &na);
    f32 *gb_v = eval_buf(ctx, gb, &nb);
    int ok = (na == 3) && (nb == 3);
    for (int i = 0; i < 3 && ok; i++) {
        ok = check_close("grad_add_a", ga_v[i], 1.f, ATOL)
           && check_close("grad_add_b", gb_v[i], 1.f, ATOL);
    }
    if (ok) { printf("  test_grad_add: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_grad_mul: z = sum(a * b), dz/da = b, dz/db = a
static void test_grad_mul(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {2,3,4}, b[] = {5,6,7};
    Term ta = thvm_tensor(ctx, a, SHAPE(1,3));
    Term tb = thvm_tensor(ctx, b, SHAPE(1,3));
    thvm_set_requires_grad(ctx, ta);
    thvm_set_requires_grad(ctx, tb);
    Term z = thvm_op(ctx, UOP_MUL, ta, tb);
    u32 axes[] = {0, 1};
    Term loss = thvm_sum_axes(ctx, z, axes, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1));

    Term ga = thvm_grad(ctx, loss, ta);
    Term gb = thvm_grad(ctx, loss, tb);
    u32 na, nb; f32 *ga_v = eval_buf(ctx, ga, &na);
    f32 *gb_v = eval_buf(ctx, gb, &nb);
    int ok = (na == 3) && (nb == 3);
    for (int i = 0; i < 3 && ok; i++) {
        ok = check_close("grad_mul_a", ga_v[i], b[i], ATOL)
           && check_close("grad_mul_b", gb_v[i], a[i], ATOL);
    }
    if (ok) { printf("  test_grad_mul: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_grad_relu: z = sum(relu(a)), dz/da = (a > 0 ? 1 : 0)
static void test_grad_relu(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {-2, -1, 0, 1, 2};
    Term ta = thvm_tensor(ctx, a, SHAPE(1, 5));
    thvm_set_requires_grad(ctx, ta);
    Term h = thvm_op(ctx, UOP_RELU, ta, term_era());
    u32 axes[] = {0, 1};
    Term loss = thvm_sum_axes(ctx, h, axes, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1));

    Term ga = thvm_grad(ctx, loss, ta);
    u32 na; f32 *ga_v = eval_buf(ctx, ga, &na);
    f32 expected[] = {0, 0, 0, 1, 1};  // relu grad: 0 for x<=0, 1 for x>0
    int ok = (na == 5);
    for (int i = 0; i < 5 && ok; i++)
        ok = check_close("grad_relu", ga_v[i], expected[i], ATOL);
    if (ok) { printf("  test_grad_relu: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_grad_matmul: z = sum(A @ B), dz/dA = ones @ B^T, dz/dB = A^T @ ones
static void test_grad_matmul(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1,2,3,4}; // [2,2]
    f32 b[] = {5,6,7,8}; // [2,2]
    Term ta = thvm_tensor(ctx, a, SHAPE(2, 2));
    Term tb = thvm_tensor(ctx, b, SHAPE(2, 2));
    thvm_set_requires_grad(ctx, ta);
    thvm_set_requires_grad(ctx, tb);
    Term mm = thvm_op(ctx, UOP_MM, ta, tb);
    u32 axes[] = {0, 1};
    Term loss = thvm_sum_axes(ctx, mm, axes, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1));

    // dz/dA = ones(2,2) @ B^T = [[5+6, 7+8],[5+6, 7+8]] = [[11,15],[11,15]]
    Term ga = thvm_grad(ctx, loss, ta);
    u32 na; f32 *ga_v = eval_buf(ctx, ga, &na);
    f32 expected_a[] = {11,15,11,15};
    int ok = (na == 4);
    for (int i = 0; i < 4 && ok; i++)
        ok = check_close("grad_mm_a", ga_v[i], expected_a[i], ATOL);

    // dz/dB = A^T @ ones(2,2) = [[1+3, 1+3],[2+4, 2+4]] = [[4,4],[6,6]]
    Term gb = thvm_grad(ctx, loss, tb);
    u32 nb; f32 *gb_v = eval_buf(ctx, gb, &nb);
    f32 expected_b[] = {4,4,6,6};
    for (int i = 0; i < 4 && ok; i++)
        ok = check_close("grad_mm_b", gb_v[i], expected_b[i], ATOL);

    if (ok) { printf("  test_grad_matmul: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_grad_exp: z = sum(exp(a)), dz/da = exp(a)
static void test_grad_exp(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {0.f, 1.f, -1.f};
    Term ta = thvm_tensor(ctx, a, SHAPE(1, 3));
    thvm_set_requires_grad(ctx, ta);
    Term e = thvm_op(ctx, UOP_EXP, ta, term_era());
    u32 axes[] = {0, 1};
    Term loss = thvm_sum_axes(ctx, e, axes, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1));

    Term ga = thvm_grad(ctx, loss, ta);
    u32 na; f32 *ga_v = eval_buf(ctx, ga, &na);
    int ok = (na == 3);
    for (int i = 0; i < 3 && ok; i++)
        ok = check_close("grad_exp", ga_v[i], expf(a[i]), ATOL);
    if (ok) { printf("  test_grad_exp: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_grad_log: z = sum(log(a)), dz/da = 1/a
static void test_grad_log(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 a[] = {1.f, 2.f, 4.f};
    Term ta = thvm_tensor(ctx, a, SHAPE(1, 3));
    thvm_set_requires_grad(ctx, ta);
    Term lg = thvm_op(ctx, UOP_LOG, ta, term_era());
    u32 axes[] = {0, 1};
    Term loss = thvm_sum_axes(ctx, lg, axes, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1));

    Term ga = thvm_grad(ctx, loss, ta);
    u32 na; f32 *ga_v = eval_buf(ctx, ga, &na);
    int ok = (na == 3);
    for (int i = 0; i < 3 && ok; i++)
        ok = check_close("grad_log", ga_v[i], 1.f / a[i], ATOL);
    if (ok) { printf("  test_grad_log: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// test_grad_chain: z = sum(relu(a * b + c)), multi-op backward
static void test_grad_chain(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 av[] = {1, -2, 3}, bv[] = {2, 3, -1}, cv[] = {0.5f, 0.5f, 0.5f};
    Term a = thvm_tensor(ctx, av, SHAPE(1, 3));
    Term b = thvm_tensor(ctx, bv, SHAPE(1, 3));
    Term c = thvm_tensor(ctx, cv, SHAPE(1, 3));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);
    thvm_set_requires_grad(ctx, c);

    // z = relu(a*b + c) = relu([2.5, -5.5, -2.5]) = [2.5, 0, 0]
    Term ab = thvm_op(ctx, UOP_MUL, a, b);
    Term s  = thvm_op(ctx, UOP_ADD, ab, c);
    Term h  = thvm_op(ctx, UOP_RELU, s, term_era());
    u32 axes[] = {0, 1};
    Term loss = thvm_sum_axes(ctx, h, axes, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1));

    // Manually:
    // pre_relu = [2.5, -5.5, -2.5]
    // relu_mask = [1, 0, 0]
    // dz/ds = relu_mask = [1, 0, 0]
    // dz/da = relu_mask * b = [2, 0, 0]
    // dz/db = relu_mask * a = [1, 0, 0]
    // dz/dc = relu_mask = [1, 0, 0]

    Term ga = thvm_grad(ctx, loss, a);
    Term gb = thvm_grad(ctx, loss, b);
    Term gc = thvm_grad(ctx, loss, c);
    u32 na, nb, nc;
    f32 *ga_v = eval_buf(ctx, ga, &na);
    f32 *gb_v = eval_buf(ctx, gb, &nb);
    f32 *gc_v = eval_buf(ctx, gc, &nc);

    f32 exp_a[] = {2, 0, 0}, exp_b[] = {1, 0, 0}, exp_c[] = {1, 0, 0};
    int ok = (na == 3) && (nb == 3) && (nc == 3);
    for (int i = 0; i < 3 && ok; i++) {
        ok = check_close("grad_chain_a", ga_v[i], exp_a[i], ATOL)
           && check_close("grad_chain_b", gb_v[i], exp_b[i], ATOL)
           && check_close("grad_chain_c", gc_v[i], exp_c[i], ATOL);
    }
    if (ok) { printf("  test_grad_chain: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── WHERE test ─────────────────────────────────────────────────

static void test_where(void) {
    TinyHVM *ctx = fresh_ctx();
    f32 cond_v[] = {1, 0, 1, 0};
    f32 a_v[]    = {10, 20, 30, 40};
    f32 b_v[]    = {-1, -2, -3, -4};
    Term cond = thvm_tensor(ctx, cond_v, SHAPE(4));
    Term a    = thvm_tensor(ctx, a_v, SHAPE(4));
    Term b    = thvm_tensor(ctx, b_v, SHAPE(4));
    Term result = thvm_where(ctx, cond, a, b);
    u32 n; f32 *out = eval_buf(ctx, result, &n);
    f32 expected[] = {10, -2, 30, -4};
    int ok = (n == 4);
    for (int i = 0; i < 4 && ok; i++)
        ok = check_close("where", out[i], expected[i], ATOL);
    if (ok) { printf("  test_where: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── Softmax test (compound: max → sub → exp → sum → div) ──────

static void test_softmax(void) {
    TinyHVM *ctx = fresh_ctx();
    // softmax([1, 2, 3]) ≈ [0.0900, 0.2447, 0.6652]
    f32 a[] = {1.f, 2.f, 3.f};
    Term ta = thvm_tensor(ctx, a, SHAPE(1, 3));
    // manual softmax
    Term xmax = thvm_op(ctx, UOP_RMAX, ta, term_era());          // [1,1]
    Term shifted = thvm_op(ctx, UOP_SUB, ta,
                   thvm_expand(ctx, xmax, SHAPE(1, 3)));
    Term e = thvm_op(ctx, UOP_EXP, shifted, term_era());         // [1,3]
    Term esum = thvm_op(ctx, UOP_SUM, e, term_era());            // [1,1]
    Term probs = thvm_op(ctx, UOP_DIV, e,
                 thvm_expand(ctx, esum, SHAPE(1, 3)));
    u32 n; f32 *out = eval_buf(ctx, probs, &n);

    // Expected: exp(x-max) / sum_exp
    f32 e0 = expf(1-3), e1 = expf(2-3), e2 = expf(3-3);
    f32 es = e0 + e1 + e2;
    f32 expected[] = {e0/es, e1/es, e2/es};
    int ok = (n == 3);
    for (int i = 0; i < 3 && ok; i++)
        ok = check_close("softmax", out[i], expected[i], ATOL);
    if (ok) { printf("  test_softmax: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── Main ───────────────────────────────────────────────────────

int main(void) {
    printf("=== TinyHVM Op Tests (tinygrad-style) ===\n\n");

    printf("── Forward ──\n");
    test_add();
    test_sub();
    test_mul();
    test_div();
    test_neg();
    test_relu();
    test_exp();
    test_log();
    test_sqrt();
    test_max();
    test_cmp();

    printf("\n── Reduce ──\n");
    test_sum();
    test_sum_all();
    test_rmax();

    printf("\n── Movement ──\n");
    test_reshape();
    test_permute();
    test_expand();

    printf("\n── Matmul ──\n");
    test_matmul();
    test_matmul_rect();

    printf("\n── Broadcast ──\n");
    test_broadcast_add();

    printf("\n── Compound ──\n");
    test_where();
    test_softmax();

    printf("\n── Backward ──\n");
    test_grad_add();
    test_grad_mul();
    test_grad_relu();
    test_grad_exp();
    test_grad_log();
    test_grad_matmul();
    test_grad_chain();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
