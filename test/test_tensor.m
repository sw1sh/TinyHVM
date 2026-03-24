// test_tensor.m — tensor lifecycle, backward chains, shapes, and metadata
//
// Modeled on tinygrad/test/test_tensor.py

#define DEVICE "cpu"
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <math.h>

static int n_pass = 0, n_fail = 0;
#define ATOL 1e-3f

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

static f32 eval_s(TinyHVM *ctx, Term t) {
    Term r = thvm_reduce(ctx, t);
    return (term_tag(r) == TAG_TEN) ? thvm_to_host(ctx, r)[0] : NAN;
}

// ── 1. Shape metadata ──────────────────────────────────────────

static void test_shape_after_reshape(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[] = {1,2,3,4,5,6};
    Term t = thvm_tensor(ctx, d, SHAPE(6));
    u32 tid = (u32)term_val(t);
    assert(ctx->tensors[tid].view.shape.rank == 1);
    assert(ctx->tensors[tid].view.shape.dims[0] == 6);

    Term r = thvm_reshape(ctx, t, SHAPE(2, 3));
    u32 rid = (u32)term_val(r);
    assert(ctx->tensors[rid].view.shape.rank == 2);
    assert(ctx->tensors[rid].view.shape.dims[0] == 2);
    assert(ctx->tensors[rid].view.shape.dims[1] == 3);
    assert(ctx->tensors[rid].view.numel == 6);
    printf("  test_shape_after_reshape: PASS\n"); n_pass++;
    thvm_free(ctx);
}

static void test_shape_after_expand(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[] = {1, 2, 3};
    Term t = thvm_tensor(ctx, d, SHAPE(1, 3));
    Term e = thvm_expand(ctx, t, SHAPE(4, 3));
    u32 eid = (u32)term_val(e);
    assert(ctx->tensors[eid].view.shape.rank == 2);
    assert(ctx->tensors[eid].view.shape.dims[0] == 4);
    assert(ctx->tensors[eid].view.shape.dims[1] == 3);
    assert(ctx->tensors[eid].view.numel == 12);
    assert(ctx->tensors[eid].view.strides[0] == 0);  // broadcast
    printf("  test_shape_after_expand: PASS\n"); n_pass++;
    thvm_free(ctx);
}

static void test_numel(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[24]; for (int i = 0; i < 24; i++) d[i] = (f32)i;
    Term t = thvm_tensor(ctx, d, SHAPE(2, 3, 4));
    u32 tid = (u32)term_val(t);
    assert(ctx->tensors[tid].view.numel == 24);
    assert(ctx->tensors[tid].view.shape.rank == 3);
    printf("  test_numel: PASS\n"); n_pass++;
    thvm_free(ctx);
}

// ── 2. Backward pass — simple chain ────────────────────────────

static void test_backward_chain(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 xd[] = {1,0, 0,1};
    f32 wd[] = {0.5f, -0.3f, 0.2f, 0.4f};
    f32 bd[] = {0.1f, -0.1f};

    Term X = thvm_tensor(ctx, xd, SHAPE(2, 2));
    Term W = thvm_tensor(ctx, wd, SHAPE(2, 2));
    Term B = thvm_tensor(ctx, bd, SHAPE(1, 2));
    thvm_set_requires_grad(ctx, W);
    thvm_set_requires_grad(ctx, B);

    Term xw = thvm_op(ctx, UOP_MM, X, W);
    Term z = thvm_op(ctx, UOP_RELU, thvm_op(ctx, UOP_ADD, xw,
             thvm_expand(ctx, B, SHAPE(2, 2))), term_era());
    u32 ax[] = {0, 1};
    Term loss = thvm_reshape(ctx, thvm_sum_axes(ctx, z, ax, 2), SHAPE(1));

    f32 fwd = eval_s(ctx, loss);
    int ok = chk("chain_fwd", fwd, 1.2f, ATOL);

    Term gW = thvm_grad(ctx, loss, W);
    Term gB = thvm_grad(ctx, loss, B);
    u32 nw, nb;
    f32 *gw = eval_v(ctx, gW, &nw);
    f32 *gb = eval_v(ctx, gB, &nb);

    f32 exp_gw[] = {1, 0, 1, 1};
    ok = ok && (nw == 4);
    for (int i = 0; i < 4 && ok; i++)
        ok = chk("chain_gW", gw[i], exp_gw[i], ATOL);

    ok = ok && (nb == 2) && chk("chain_gB[0]", gb[0], 2.f, ATOL)
                         && chk("chain_gB[1]", gb[1], 1.f, ATOL);

    if (ok) { printf("  test_backward_chain: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── 3. Diamond model gradient ──────────────────────────────────

static void test_diamond_grad(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 ud[] = {1, 2}, vd[] = {3, 4}, wd[] = {5, 6};
    Term U = thvm_tensor(ctx, ud, SHAPE(1, 2));
    Term V = thvm_tensor(ctx, vd, SHAPE(1, 2));
    Term W = thvm_tensor(ctx, wd, SHAPE(1, 2));
    thvm_set_requires_grad(ctx, U);
    thvm_set_requires_grad(ctx, V);
    thvm_set_requires_grad(ctx, W);

    Term x = thvm_op(ctx, UOP_MUL, U, V);
    Term y = thvm_op(ctx, UOP_MUL, U, W);
    Term s = thvm_op(ctx, UOP_ADD, x, y);
    Term h = thvm_op(ctx, UOP_RELU, s, term_era());
    Term out = thvm_op(ctx, UOP_MUL, h, y);
    u32 ax[] = {0, 1};
    Term loss = thvm_reshape(ctx, thvm_sum_axes(ctx, out, ax, 2), SHAPE(1));

    f32 fwd = eval_s(ctx, loss);
    int ok = chk("diamond_fwd", fwd, 280.f, ATOL);

    Term gU = thvm_grad(ctx, loss, U);
    Term gV = thvm_grad(ctx, loss, V);
    Term gW2 = thvm_grad(ctx, loss, W);
    u32 nu, nv, nw2;
    f32 *gu = eval_v(ctx, gU, &nu);
    f32 *gv = eval_v(ctx, gV, &nv);
    f32 *gw = eval_v(ctx, gW2, &nw2);

    ok = ok && (nu == 2) && chk("gU[0]", gu[0], 80.f, ATOL)
                         && chk("gU[1]", gu[1], 240.f, ATOL);
    ok = ok && (nv == 2) && chk("gV[0]", gv[0], 5.f, ATOL)
                         && chk("gV[1]", gv[1], 24.f, ATOL);
    ok = ok && (nw2 == 2) && chk("gW[0]", gw[0], 13.f, ATOL)
                          && chk("gW[1]", gw[1], 64.f, ATOL);

    if (ok) { printf("  test_diamond_grad: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── 4. Clone preserves values ──────────────────────────────────

static void test_clone(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[] = {1, 2, 3, 4};
    Term t = thvm_tensor(ctx, d, SHAPE(2, 2));
    // Clone by forcing a contiguous copy via add-0
    f32 z = 0.f;
    Term copy = thvm_op(ctx, UOP_ADD, t,
                thvm_expand(ctx, thvm_tensor(ctx, &z, SHAPE(1,1)), SHAPE(2, 2)));
    Term cloned = thvm_reduce(ctx, copy);
    u32 orig_id = (u32)term_val(t);
    u32 clone_id = (u32)term_val(cloned);
    assert(orig_id != clone_id);  // different tensor IDs

    f32 *orig = thvm_to_host(ctx, t);
    f32 *cp = thvm_to_host(ctx, cloned);
    int ok = 1;
    for (int i = 0; i < 4 && ok; i++)
        ok = chk("clone", cp[i], orig[i], 1e-6f);
    if (ok) { printf("  test_clone: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── 5. Assign ──────────────────────────────────────────────────

static void test_assign(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[] = {1, 2, 3, 4};
    f32 upd[] = {10, 20, 30, 40};
    Term dst = thvm_tensor(ctx, d, SHAPE(4));
    Term src = thvm_tensor(ctx, upd, SHAPE(4));
    thvm_reduce(ctx, thvm_assign(ctx, dst, src));

    f32 *buf = thvm_to_host(ctx, dst);
    int ok = 1;
    for (int i = 0; i < 4 && ok; i++)
        ok = chk("assign", buf[i], upd[i], 1e-6f);
    if (ok) { printf("  test_assign: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── 6. Requires_grad flag ──────────────────────────────────────

static void test_requires_grad(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[] = {1, 2};
    Term t = thvm_tensor(ctx, d, SHAPE(2));
    u32 tid = (u32)term_val(t);
    assert(ctx->tensors[tid].requires_grad == 0);
    thvm_set_requires_grad(ctx, t);
    assert(ctx->tensors[tid].requires_grad == 1);
    printf("  test_requires_grad: PASS\n"); n_pass++;
    thvm_free(ctx);
}

// ── 7. Refcount starts at 1 ───────────────────────────────────

static void test_refcount(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[] = {1};
    Term t = thvm_tensor(ctx, d, SHAPE(1));
    u32 tid = (u32)term_val(t);
    assert(ctx->tensors[tid].refcount == 1);
    printf("  test_refcount: PASS\n"); n_pass++;
    thvm_free(ctx);
}

// ── 8. Backward through div ────────────────────────────────────

static void test_grad_div(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 av[] = {6.f, 12.f}, bv[] = {2.f, 3.f};
    Term a = thvm_tensor(ctx, av, SHAPE(1, 2));
    Term b = thvm_tensor(ctx, bv, SHAPE(1, 2));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);

    Term q = thvm_op(ctx, UOP_DIV, a, b);
    u32 ax[] = {0, 1};
    Term loss = thvm_reshape(ctx, thvm_sum_axes(ctx, q, ax, 2), SHAPE(1));

    f32 fwd = eval_s(ctx, loss);
    int ok = chk("div_fwd", fwd, 7.f, ATOL);

    Term ga = thvm_grad(ctx, loss, a);
    Term gb = thvm_grad(ctx, loss, b);
    u32 na, nb;
    f32 *ga_v = eval_v(ctx, ga, &na);
    f32 *gb_v = eval_v(ctx, gb, &nb);
    ok = ok && (na == 2) && chk("gA[0]", ga_v[0], 0.5f, ATOL)
                         && chk("gA[1]", ga_v[1], 1.f/3.f, ATOL);
    ok = ok && (nb == 2) && chk("gB[0]", gb_v[0], -1.5f, ATOL)
                         && chk("gB[1]", gb_v[1], -12.f/9.f, ATOL);
    if (ok) { printf("  test_grad_div: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── 9. Backward through sub ────────────────────────────────────

static void test_grad_sub(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 av[] = {5, 3}, bv[] = {2, 1};
    Term a = thvm_tensor(ctx, av, SHAPE(1, 2));
    Term b = thvm_tensor(ctx, bv, SHAPE(1, 2));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);

    Term s = thvm_op(ctx, UOP_SUB, a, b);
    u32 ax[] = {0, 1};
    Term loss = thvm_reshape(ctx, thvm_sum_axes(ctx, s, ax, 2), SHAPE(1));

    Term ga = thvm_grad(ctx, loss, a);
    Term gb = thvm_grad(ctx, loss, b);
    u32 na, nb;
    f32 *ga_v = eval_v(ctx, ga, &na);
    f32 *gb_v = eval_v(ctx, gb, &nb);
    int ok = (na == 2) && (nb == 2);
    ok = ok && chk("gA[0]", ga_v[0], 1.f, ATOL)
           && chk("gA[1]", ga_v[1], 1.f, ATOL)
           && chk("gB[0]", gb_v[0], -1.f, ATOL)
           && chk("gB[1]", gb_v[1], -1.f, ATOL);
    if (ok) { printf("  test_grad_sub: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── 10. Backward through sqrt ──────────────────────────────────

static void test_grad_sqrt(void) {
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 av[] = {4.f, 9.f, 16.f};
    Term a = thvm_tensor(ctx, av, SHAPE(1, 3));
    thvm_set_requires_grad(ctx, a);

    Term s = thvm_op(ctx, UOP_SQRT, a, term_era());
    u32 ax[] = {0, 1};
    Term loss = thvm_reshape(ctx, thvm_sum_axes(ctx, s, ax, 2), SHAPE(1));

    Term ga = thvm_grad(ctx, loss, a);
    u32 na; f32 *ga_v = eval_v(ctx, ga, &na);
    int ok = (na == 3);
    for (int i = 0; i < 3 && ok; i++)
        ok = chk("grad_sqrt", ga_v[i], 0.5f / sqrtf(av[i]), ATOL);
    if (ok) { printf("  test_grad_sqrt: PASS\n"); n_pass++; }
    thvm_free(ctx);
}

// ── Main ───────────────────────────────────────────────────────

int main(void) {
    printf("=== TinyHVM Tensor Tests (tinygrad-style) ===\n\n");

    printf("── Shape / Metadata ──\n");
    test_shape_after_reshape();
    test_shape_after_expand();
    test_numel();
    test_requires_grad();
    test_refcount();

    printf("\n── Lifecycle ──\n");
    test_clone();
    test_assign();

    printf("\n── Backward: chains ──\n");
    test_backward_chain();
    test_grad_div();
    test_grad_sub();
    test_grad_sqrt();

    printf("\n── Backward: diamond ──\n");
    test_diamond_grad();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
