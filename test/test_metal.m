// test_metal.m — Metal backend tests
//
// Modeled on tinygrad/test/device/test_metal.py
// Tests: backend init, buffer alloc/free lifecycle, read/write roundtrip,
// batch command buffer, GPU compute parity with CPU.

#define DEVICE "metal"
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#include "../src/backend/metal/_.m"
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

// ── 1. Backend initializes correctly ───────────────────────────

static void test_backend_init(void) {
    TinyHVM *ctx = thvm_init(DEVICE);
    assert(ctx != NULL);
    assert(ctx->backend != NULL);
    assert(ctx->backend->buf_alloc != NULL);
    assert(ctx->backend->buf_free != NULL);
    assert(ctx->backend->buf_write != NULL);
    assert(ctx->backend->buf_read != NULL);
    printf("  test_backend_init: PASS\n"); n_pass++;
    thvm_free(ctx);
}

// ── 2. Buffer alloc + write + read roundtrip ───────────────────

static void test_buf_roundtrip(void) {
    TinyHVM *ctx = thvm_init(DEVICE);
    f32 data[] = {3.14f, 2.71f, 1.41f, 1.73f};
    u32 bid = ctx->backend->buf_alloc(4 * sizeof(f32));
    assert(bid > 0);
    ctx->backend->buf_write(bid, data, 4 * sizeof(f32));

    f32 out[4] = {0};
    ctx->backend->buf_read(bid, out, 4 * sizeof(f32));
    int ok = 1;
    for (int i = 0; i < 4 && ok; i++)
        ok = chk("roundtrip", out[i], data[i], 1e-6f);
    if (ok) { printf("  test_buf_roundtrip: PASS\n"); n_pass++; }

    ctx->backend->buf_free(bid);
    thvm_free(ctx);
}

// ── 3. Tensor roundtrip through Metal ──────────────────────────

static void test_tensor_roundtrip(void) {
    TinyHVM *ctx = thvm_init(DEVICE);
    f32 data[] = {1, 2, 3, 4, 5, 6};
    Term t = thvm_tensor(ctx, data, SHAPE(2, 3));
    f32 *host = thvm_to_host(ctx, t);
    int ok = 1;
    for (int i = 0; i < 6 && ok; i++)
        ok = chk("tensor_rt", host[i], data[i], 1e-6f);
    if (ok) { printf("  test_tensor_roundtrip: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── 4. Metal ADD parity with expected values ───────────────────

static void test_metal_add(void) {
    TinyHVM *ctx = thvm_init(DEVICE);
    f32 a[] = {1, 2, 3, 4}, b[] = {10, 20, 30, 40};
    Term ta = thvm_tensor(ctx, a, SHAPE(4));
    Term tb = thvm_tensor(ctx, b, SHAPE(4));
    Term r = thvm_op(ctx, UOP_ADD, ta, tb);
    u32 n; f32 *out = eval_v(ctx, r, &n);
    f32 expected[] = {11, 22, 33, 44};
    int ok = (n == 4);
    for (int i = 0; i < 4 && ok; i++)
        ok = chk("metal_add", out[i], expected[i], ATOL);
    if (ok) { printf("  test_metal_add: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── 5. Metal matmul ────────────────────────────────────────────

static void test_metal_matmul(void) {
    TinyHVM *ctx = thvm_init(DEVICE);
    f32 a[] = {1,2,3,4}, b[] = {5,6,7,8};
    Term ta = thvm_tensor(ctx, a, SHAPE(2, 2));
    Term tb = thvm_tensor(ctx, b, SHAPE(2, 2));
    Term r = thvm_mm(ctx, ta, tb);
    u32 n; f32 *out = eval_v(ctx, r, &n);
    f32 expected[] = {19, 22, 43, 50};
    int ok = (n == 4);
    for (int i = 0; i < 4 && ok; i++)
        ok = chk("metal_mm", out[i], expected[i], ATOL);
    if (ok) { printf("  test_metal_matmul: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── 6. Metal exp/log roundtrip ─────────────────────────────────

static void test_metal_exp_log(void) {
    TinyHVM *ctx = thvm_init(DEVICE);
    f32 a[] = {1.f, 2.f, 3.f};
    Term ta = thvm_tensor(ctx, a, SHAPE(3));
    Term e = thvm_op(ctx, UOP_EXP, ta, term_era());
    Term l = thvm_op(ctx, UOP_LOG, e, term_era());
    u32 n; f32 *out = eval_v(ctx, l, &n);
    // log(exp(x)) = x
    int ok = (n == 3);
    for (int i = 0; i < 3 && ok; i++)
        ok = chk("exp_log", out[i], a[i], 1e-3f);
    if (ok) { printf("  test_metal_exp_log: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── 7. Metal reduce sum ────────────────────────────────────────

static void test_metal_sum(void) {
    TinyHVM *ctx = thvm_init(DEVICE);
    f32 a[] = {1, 2, 3, 4, 5, 6};
    Term ta = thvm_tensor(ctx, a, SHAPE(2, 3));
    u32 ax[] = {0, 1};
    Term r = thvm_sum_axes(ctx, ta, ax, 2);
    u32 n; f32 *out = eval_v(ctx, r, &n);
    int ok = (n == 1) && chk("metal_sum", out[0], 21.f, ATOL);
    if (ok) { printf("  test_metal_sum: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── 8. Metal backward — gradient through add ───────────────────

static void test_metal_grad(void) {
    TinyHVM *ctx = thvm_init(DEVICE);
    f32 a[] = {1, 2, 3}, b[] = {4, 5, 6};
    Term ta = thvm_tensor(ctx, a, SHAPE(1, 3));
    Term tb = thvm_tensor(ctx, b, SHAPE(1, 3));
    thvm_set_requires_grad(ctx, ta);
    thvm_set_requires_grad(ctx, tb);
    Term z = thvm_op(ctx, UOP_ADD, ta, tb);
    u32 ax[] = {0, 1};
    Term loss = thvm_reshape(ctx, thvm_sum_axes(ctx, z, ax, 2), SHAPE(1));

    Term ga = thvm_grad(ctx, loss, ta);
    Term gb = thvm_grad(ctx, loss, tb);
    u32 na, nb;
    f32 *ga_v = eval_v(ctx, ga, &na);
    f32 *gb_v = eval_v(ctx, gb, &nb);
    int ok = (na == 3) && (nb == 3);
    for (int i = 0; i < 3 && ok; i++) {
        ok = chk("metal_grad_a", ga_v[i], 1.f, ATOL)
           && chk("metal_grad_b", gb_v[i], 1.f, ATOL);
    }
    if (ok) { printf("  test_metal_grad: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── 9. Batch command buffer ────────────────────────────────────

static void test_batch_commands(void) {
    TinyHVM *ctx = thvm_init(DEVICE);
    if (ctx->backend->begin_batch) ctx->backend->begin_batch();

    f32 a[] = {1, 2, 3, 4};
    Term ta = thvm_tensor(ctx, a, SHAPE(4));
    // Chain: add → mul → neg
    f32 one = 1.f;
    Term r = thvm_op(ctx, UOP_ADD, ta, thvm_expand(ctx,
             thvm_tensor(ctx, &one, SHAPE(1)), SHAPE(4)));
    r = thvm_op(ctx, UOP_MUL, r, r);
    r = thvm_op(ctx, UOP_NEG, r, term_era());

    if (ctx->backend->end_batch) ctx->backend->end_batch();

    u32 n; f32 *out = eval_v(ctx, r, &n);
    // -(x+1)^2: -4, -9, -16, -25
    f32 expected[] = {-4, -9, -16, -25};
    int ok = (n == 4);
    for (int i = 0; i < 4 && ok; i++)
        ok = chk("batch", out[i], expected[i], ATOL);
    if (ok) { printf("  test_batch_commands: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── 10. CPU vs Metal parity ────────────────────────────────────

static void test_cpu_metal_parity(void) {
    f32 a[] = {1.5f, -2.3f, 0.7f, 4.1f};
    f32 b[] = {0.5f, 1.2f, -0.8f, 2.0f};

    // CPU result
    TinyHVM *cpu_ctx = thvm_init("cpu");
    Term ca = thvm_tensor(cpu_ctx, a, SHAPE(2, 2));
    Term cb = thvm_tensor(cpu_ctx, b, SHAPE(2, 2));
    Term cr = thvm_mm(cpu_ctx, ca, cb);
    u32 cn; f32 *cpu_out = eval_v(cpu_ctx, cr, &cn);
    f32 cpu_vals[4]; for (int i = 0; i < 4; i++) cpu_vals[i] = cpu_out[i];
    thvm_free(cpu_ctx);

    // Metal result
    TinyHVM *mtl_ctx = thvm_init("metal");
    Term ma = thvm_tensor(mtl_ctx, a, SHAPE(2, 2));
    Term mb = thvm_tensor(mtl_ctx, b, SHAPE(2, 2));
    Term mr = thvm_mm(mtl_ctx, ma, mb);
    u32 mn; f32 *mtl_out = eval_v(mtl_ctx, mr, &mn);

    int ok = (cn == 4) && (mn == 4);
    for (int i = 0; i < 4 && ok; i++)
        ok = chk("parity", mtl_out[i], cpu_vals[i], 1e-3f);
    if (ok) { printf("  test_cpu_metal_parity: PASS\n"); n_pass++; }
    thvm_free(mtl_ctx);
}

// ── Main ───────────────────────────────────────────────────────

int main(void) {
    printf("=== TinyHVM Metal Backend Tests ===\n\n");

    printf("── Backend ──\n");
    test_backend_init();
    test_buf_roundtrip();
    test_tensor_roundtrip();

    printf("\n── Compute ──\n");
    test_metal_add();
    test_metal_matmul();
    test_metal_exp_log();
    test_metal_sum();

    printf("\n── Autograd on Metal ──\n");
    test_metal_grad();

    printf("\n── Command Batching ──\n");
    test_batch_commands();

    printf("\n── Parity ──\n");
    test_cpu_metal_parity();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
