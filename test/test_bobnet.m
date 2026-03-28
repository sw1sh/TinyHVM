// test_bobnet.m — TinyBobNet training tests (lightweight MLP)
//
// Modeled on tinygrad/test/models/test_mnist.py (TinyBobNet) and
// tinygrad/test/models/test_train.py (train_one_step pattern).
//
// Tests: forward shape, SGD one-step loss decrease, multi-step
// convergence, gradient health, and GC after training steps.

#define DEVICE "cpu"
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int n_pass = 0, n_fail = 0;

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

static void randn_init(f32 *data, u32 n, f32 scale) {
    for (u32 i = 0; i < n; i++)
        data[i] = scale * ((f32)rand() / (f32)RAND_MAX * 2.0f - 1.0f);
}

// ── TinyBobNet: x.dot(l1).relu().dot(l2) ──────────────────────

typedef struct { Term l1, l2; } BobNet;

static BobNet bobnet_init(TinyHVM *ctx, u32 in_d, u32 hid, u32 out_d) {
    f32 *w1 = malloc(in_d * hid * sizeof(f32));
    f32 *w2 = malloc(hid * out_d * sizeof(f32));
    randn_init(w1, in_d * hid, sqrtf(2.f / (f32)(in_d + hid)));
    randn_init(w2, hid * out_d, sqrtf(2.f / (f32)(hid + out_d)));
    BobNet net;
    net.l1 = thvm_tensor(ctx, w1, SHAPE(in_d, hid));
    net.l2 = thvm_tensor(ctx, w2, SHAPE(hid, out_d));
    thvm_set_requires_grad(ctx, net.l1);
    thvm_set_requires_grad(ctx, net.l2);
    free(w1); free(w2);
    return net;
}

static Term bobnet_fwd(TinyHVM *ctx, BobNet *net, Term x) {
    return thvm_op(ctx, UOP_MM,
           thvm_op(ctx, UOP_RELU, thvm_op(ctx, UOP_MM, x, net->l1), term_era()),
           net->l2);
}

static Term mse_loss(TinyHVM *ctx, Term logits, f32 *onehot, u32 bs, u32 nc) {
    Term y = thvm_tensor(ctx, onehot, SHAPE(bs, nc));
    Term diff = thvm_op(ctx, UOP_SUB, logits, y);
    Term sq = thvm_op(ctx, UOP_MUL, diff, diff);
    u32 ax[] = {0, 1};
    Term total = thvm_sum_axes(ctx, sq, ax, 2);
    f32 inv = 1.f / (f32)bs;
    return thvm_reshape(ctx, thvm_op(ctx, UOP_MUL, total,
           thvm_tensor(ctx, &inv, SHAPE(1))), SHAPE(1));
}

static void sgd_step(TinyHVM *ctx, Term w, Term grad_reduced, f32 lr) {
    u32 wid = (u32)term_val(w);
    u32 numel = ctx->tensors[wid].view.numel;
    f32 *wh = thvm_to_host(ctx, w);
    f32 *gh = thvm_to_host(ctx, grad_reduced);
    for (u32 i = 0; i < numel; i++) wh[i] -= lr * gh[i];
    ctx->backend->buf_write(ctx->tensors[wid].buf_id, wh, numel * sizeof(f32));
}

// ── Test 1: forward shape ──────────────────────────────────────

static void test_forward_shape(void) {
    TinyHVM *ctx = thvm_init(DEVICE);
    u32 BS = 4, IN = 8, H = 16, OUT = 3;
    BobNet net = bobnet_init(ctx, IN, H, OUT);
    f32 xd[4 * 8]; randn_init(xd, BS * IN, 1.f);
    Term x = thvm_tensor(ctx, xd, SHAPE(BS, IN));
    u32 n; eval_v(ctx, bobnet_fwd(ctx, &net, x), &n);
    if (n == BS * OUT) { printf("  test_forward_shape: PASS\n"); n_pass++; }
    else { printf("  FAIL test_forward_shape: got %u expected %u\n", n, BS*OUT); n_fail++; }
    thvm_free(ctx);
}

// ── Test 2: one SGD step decreases loss ────────────────────────

static void test_sgd_onestep(void) {
    srand(1337);
    TinyHVM *ctx = thvm_init(DEVICE);
    u32 BS = 4, IN = 8, H = 16, OUT = 3;
    BobNet net = bobnet_init(ctx, IN, H, OUT);

    f32 xd[32], yd[12];
    randn_init(xd, BS * IN, 1.f);
    memset(yd, 0, sizeof(yd));
    for (u32 i = 0; i < BS; i++) yd[i * OUT + (i % OUT)] = 1.f;

    // Before
    Term x0 = thvm_tensor(ctx, xd, SHAPE(BS, IN));
    f32 loss0 = eval_s(ctx, mse_loss(ctx, bobnet_fwd(ctx, &net, x0), yd, BS, OUT));

    // Gradients
    Term x1 = thvm_tensor(ctx, xd, SHAPE(BS, IN));
    Term loss = mse_loss(ctx, bobnet_fwd(ctx, &net, x1), yd, BS, OUT);
    Term g1 = thvm_reduce(ctx, thvm_grad(ctx, loss, net.l1));
    Term g2 = thvm_reduce(ctx, thvm_grad(ctx, loss, net.l2));
    sgd_step(ctx, net.l1, g1, 0.01f);
    sgd_step(ctx, net.l2, g2, 0.01f);

    // After
    Term x2 = thvm_tensor(ctx, xd, SHAPE(BS, IN));
    f32 loss1 = eval_s(ctx, mse_loss(ctx, bobnet_fwd(ctx, &net, x2), yd, BS, OUT));

    printf("    loss: %.4f → %.4f\n", loss0, loss1);
    if (loss1 < loss0) { printf("  test_sgd_onestep: PASS\n"); n_pass++; }
    else { printf("  FAIL test_sgd_onestep\n"); n_fail++; }
    thvm_free(ctx);
}

// ── Test 3: multi-step convergence ─────────────────────────────

static void test_sgd_multistep(void) {
    srand(42);
    TinyHVM *ctx = thvm_init(DEVICE);
    u32 BS = 8, IN = 8, H = 32, OUT = 4;
    BobNet net = bobnet_init(ctx, IN, H, OUT);

    f32 xd[64], yd[32];
    randn_init(xd, BS * IN, 1.f);
    memset(yd, 0, sizeof(yd));
    for (u32 i = 0; i < BS; i++) yd[i * OUT + (i % OUT)] = 1.f;

    u32 n_w = ctx->tensor_count;
    f32 first = 0, last = 0;
    for (u32 step = 0; step < 50; step++) {
        Term x = thvm_tensor(ctx, xd, SHAPE(BS, IN));
        Term loss = mse_loss(ctx, bobnet_fwd(ctx, &net, x), yd, BS, OUT);
        f32 lv = eval_s(ctx, loss);
        if (step == 0) first = lv;
        if (step == 49) last = lv;

        Term g1 = thvm_reduce(ctx, thvm_grad(ctx, loss, net.l1));
        Term g2 = thvm_reduce(ctx, thvm_grad(ctx, loss, net.l2));
        sgd_step(ctx, net.l1, g1, 0.05f);
        sgd_step(ctx, net.l2, g2, 0.05f);
        thvm_reset(ctx, n_w);
    }
    printf("    loss: %.4f → %.4f (%.0f%%↓)\n", first, last,
           100.f * (first - last) / first);
    if (last < first * 0.5f) { printf("  test_sgd_multistep: PASS\n"); n_pass++; }
    else { printf("  FAIL test_sgd_multistep\n"); n_fail++; }
    thvm_free(ctx);
}

// ── Test 4: gradient health ────────────────────────────────────

static void test_gradient_health(void) {
    srand(1234);
    TinyHVM *ctx = thvm_init(DEVICE);
    u32 BS = 4, IN = 8, H = 16, OUT = 3;
    BobNet net = bobnet_init(ctx, IN, H, OUT);

    f32 xd[32], yd[12];
    randn_init(xd, BS * IN, 1.f);
    memset(yd, 0, sizeof(yd));
    for (u32 i = 0; i < BS; i++) yd[i * OUT + (i % OUT)] = 1.f;

    Term x = thvm_tensor(ctx, xd, SHAPE(BS, IN));
    Term loss = mse_loss(ctx, bobnet_fwd(ctx, &net, x), yd, BS, OUT);
    Term g1r = thvm_reduce(ctx, thvm_grad(ctx, loss, net.l1));
    Term g2r = thvm_reduce(ctx, thvm_grad(ctx, loss, net.l2));

    int ok = (term_tag(g1r) == TAG_TEN) && (term_tag(g2r) == TAG_TEN);
    u32 n1 = ctx->tensors[(u32)term_val(g1r)].view.numel;
    u32 n2 = ctx->tensors[(u32)term_val(g2r)].view.numel;
    ok = ok && (n1 == IN * H) && (n2 == H * OUT);

    f32 *gv = thvm_to_host(ctx, g1r);
    f32 norm = 0;
    for (u32 i = 0; i < n1; i++) norm += gv[i] * gv[i];
    norm = sqrtf(norm);
    ok = ok && isfinite(norm) && (norm > 1e-8f);

    if (ok) { printf("  test_gradient_health: PASS (norm=%.4f)\n", norm); n_pass++; }
    else { printf("  FAIL test_gradient_health\n"); n_fail++; }
    thvm_free(ctx);
}

// ── Test 5: GC after training steps ────────────────────────────

static void test_train_gc(void) {
    srand(99);
    TinyHVM *ctx = thvm_init(DEVICE);
    u32 BS = 4, IN = 8, H = 16, OUT = 3;
    BobNet net = bobnet_init(ctx, IN, H, OUT);

    f32 xd[32], yd[12];
    randn_init(xd, BS * IN, 1.f);
    memset(yd, 0, sizeof(yd));
    for (u32 i = 0; i < BS; i++) yd[i * OUT + (i % OUT)] = 1.f;

    u32 n_w = ctx->tensor_count;
    for (int s = 0; s < 5; s++) {
        Term x = thvm_tensor(ctx, xd, SHAPE(BS, IN));
        eval_s(ctx, mse_loss(ctx, bobnet_fwd(ctx, &net, x), yd, BS, OUT));
        thvm_reset(ctx, n_w);
    }
    if (ctx->tensor_count == n_w) {
        printf("  test_train_gc: PASS (tensors=%u)\n", n_w); n_pass++;
    } else {
        printf("  FAIL test_train_gc: leak %u→%u\n", n_w, ctx->tensor_count); n_fail++;
    }
    thvm_free(ctx);
}

// ── Main ───────────────────────────────────────────────────────

int main(void) {
    printf("=== TinyHVM BobNet Tests (tinygrad-style) ===\n\n");

    printf("── Forward ──\n");
    test_forward_shape();

    printf("\n── Training ──\n");
    test_sgd_onestep();
    test_sgd_multistep();

    printf("\n── Gradient ──\n");
    test_gradient_health();

    printf("\n── GC ──\n");
    test_train_gc();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
