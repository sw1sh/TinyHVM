// test_tensor_nn_backward.m — backward checks for Tensor NN wrappers (cpu + metal)
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#include "../src/backend/metal/_.m"
#include "../src/nn/_.c"
#include <math.h>
#include <stdio.h>

static int g_fail = 0;

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail = 1; } } while (0)
#define CHECK_NEAR(a,b,eps,msg) CHECK(fabsf((a)-(b)) <= (eps), msg)

static int run_for_device(const char *device) {
    fprintf(stderr, "== backward device=%s ==\n", device);

    // ---- conv2d backward wrt weight+bias ----
    {
        TinyHVM *ctx = thvm_init(device);
        f32 xdat[] = {1,2,3,4,5,6,7,8,9}; // [1,1,3,3]
        f32 wdat[] = {1,0,0,1};           // [1,1,2,2]
        f32 bdat[] = {1};                 // [1]
        Tensor x = tensor_from_f32(ctx, xdat, SHAPE(1,1,3,3));
        Tensor w = tensor_from_f32(ctx, wdat, SHAPE(1,1,2,2));
        Tensor b = tensor_from_f32(ctx, bdat, SHAPE(1));
        thvm_set_requires_grad(ctx, w.term);
        thvm_set_requires_grad(ctx, b.term);

        u32 s[] = {1,1}, p[] = {0,0,0,0};
        Tensor y = tensor_conv2d(x, w, &b, 1, s, p);
        Tensor loss = tensor_sum_axes(y, (u32[]){0,1,2,3}, 4);
        Term params[] = {w.term, b.term};
        Tensor bundle = tensor_from_term(ctx, thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss.term, params, 2)));
        CHECK(tensor_bundle_count(bundle) == 2u, "conv bundle count");
        f32 *dw = tensor_to_host_f32(tensor_bundle_get(bundle, 0));
        f32 *db = tensor_to_host_f32(tensor_bundle_get(bundle, 1));
        CHECK(dw != NULL, "conv dw host");
        CHECK(db != NULL, "conv db host");
        if (dw && device[0] == 'm') {
            fprintf(stderr, "conv metal dw = [%f, %f, %f, %f]\n", dw[0], dw[1], dw[2], dw[3]);
        }
        if (dw) {
            CHECK_NEAR(dw[0], 12.0f, 1e-3f, "conv dw00");
            CHECK_NEAR(dw[1], 16.0f, 1e-3f, "conv dw01");
            CHECK_NEAR(dw[2], 24.0f, 1e-3f, "conv dw10");
            CHECK_NEAR(dw[3], 28.0f, 1e-3f, "conv dw11");
        }
        if (db) CHECK_NEAR(db[0], 4.0f, 1e-3f, "conv db");
        thvm_free(ctx);
    }

    // ---- maxpool2d backward wrt input ----
    {
        TinyHVM *ctx = thvm_init(device);
        f32 xdat[] = {1,2,3,4}; // [1,1,2,2]
        Tensor x = tensor_from_f32(ctx, xdat, SHAPE(1,1,2,2));
        thvm_set_requires_grad(ctx, x.term);
        u32 k[] = {2,2}, s[] = {1,1};
        Tensor y = tensor_maxpool2d(x, k, s);
        Tensor loss = tensor_sum_axes(y, (u32[]){0,1,2,3}, 4);
        Tensor bundle = tensor_from_term(ctx, thvm_eval(ctx, thvm_grad_keep(ctx, loss.term, x.term)));
        u32 n = tensor_bundle_count(bundle);
        CHECK(n == 1u, "pool bundle count");
        Tensor gx_t = tensor_bundle_get(bundle, 0);
        f32 *gx = tensor_to_host_f32(gx_t);
        CHECK(gx != NULL, "pool gx host");
        if (!gx) fprintf(stderr, "pool: bundle_count=%u tag=%u val=%llu\n",
                         n, term_tag(gx_t.term), (unsigned long long)term_val(gx_t.term));
        if (gx) {
            CHECK_NEAR(gx[0], 0.0f, 1e-3f, "pool gx0");
            CHECK_NEAR(gx[1], 0.0f, 1e-3f, "pool gx1");
            CHECK_NEAR(gx[2], 0.0f, 1e-3f, "pool gx2");
            CHECK_NEAR(gx[3], 1.0f, 1e-3f, "pool gx3");
        }
        thvm_free(ctx);
    }

    // ---- batchnorm(eval) backward wrt gamma+beta ----
    {
        TinyHVM *ctx = thvm_init(device);
        f32 xdat[] = {1,2,3,4};
        f32 gdat[] = {1}, bdat[] = {0}, mdat[] = {0}, vdat[] = {1};
        Tensor x = tensor_from_f32(ctx, xdat, SHAPE(1,1,2,2));
        Tensor g = tensor_from_f32(ctx, gdat, SHAPE(1));
        Tensor b = tensor_from_f32(ctx, bdat, SHAPE(1));
        Tensor m = tensor_from_f32(ctx, mdat, SHAPE(1));
        Tensor v = tensor_from_f32(ctx, vdat, SHAPE(1));
        thvm_set_requires_grad(ctx, g.term);
        thvm_set_requires_grad(ctx, b.term);

        Tensor y = tensor_batchnorm(x, g, b, m, v, 1,1,2,2, 0);
        Tensor loss = tensor_sum_axes(y, (u32[]){0,1,2,3}, 4);
        Term params[] = {g.term, b.term};
        Tensor bundle = tensor_from_term(ctx, thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss.term, params, 2)));
        CHECK(tensor_bundle_count(bundle) == 2u, "bn bundle count");
        f32 *dg = tensor_to_host_f32(tensor_bundle_get(bundle, 0));
        f32 *db = tensor_to_host_f32(tensor_bundle_get(bundle, 1));
        CHECK(dg != NULL, "bn dg host");
        CHECK(db != NULL, "bn db host");
        if (dg) CHECK_NEAR(dg[0], 10.0f / sqrtf(1.0f + 1e-5f), 5e-3f, "bn dgamma");
        if (db) CHECK_NEAR(db[0], 4.0f, 1e-3f, "bn dbeta");
        thvm_free(ctx);
    }

    // ---- softmax+cross-entropy backward wrt logits ----
    {
        TinyHVM *ctx = thvm_init(device);
        f32 ldat[] = {2.0f, 1.0f, 0.1f}; // [1,3]
        f32 one_hot_dat[] = {1.0f, 0.0f, 0.0f};
        Tensor logits = tensor_from_f32(ctx, ldat, SHAPE(1,3));
        Tensor oh = tensor_from_f32(ctx, one_hot_dat, SHAPE(1,3));
        thvm_set_requires_grad(ctx, logits.term);

        Tensor p = tensor_softmax(logits, 1, 3);
        Tensor picked = tensor_mul(p, oh);
        Tensor loss = tensor_sum_axes(picked, (u32[]){0,1}, 2);
        Tensor bundle = tensor_from_term(ctx, thvm_eval(ctx, thvm_grad_keep(ctx, loss.term, logits.term)));
        u32 n = tensor_bundle_count(bundle);
        CHECK(n == 1u, "softmax bundle count");
        Tensor g_t = tensor_bundle_get(bundle, 0);
        f32 *g = tensor_to_host_f32(g_t);
        CHECK(g != NULL, "softmax grad host");
        if (!g) fprintf(stderr, "softmax: bundle_count=%u tag=%u val=%llu\n",
                        n, term_tag(g_t.term), (unsigned long long)term_val(g_t.term));
        thvm_free(ctx);
    }

    // ---- cross-entropy backward wrt logits ----
    {
        TinyHVM *ctx = thvm_init(device);
        f32 ldat[] = {2.0f, 1.0f, 0.1f}; // [1,3]
        u8 label[] = {0};
        Tensor logits = tensor_from_f32(ctx, ldat, SHAPE(1,3));
        thvm_set_requires_grad(ctx, logits.term);

        Tensor ce = tensor_cross_entropy(logits, label, 1, 3);
        Tensor bundle = tensor_from_term(ctx, thvm_eval(ctx, thvm_grad_keep(ctx, ce.term, logits.term)));
        u32 n = tensor_bundle_count(bundle);
        CHECK(n == 1u, "ce bundle count");
        Tensor g_t = tensor_bundle_get(bundle, 0);
        f32 *g = tensor_to_host_f32(g_t);
        CHECK(g != NULL, "ce grad host");
        if (!g) fprintf(stderr, "ce: bundle_count=%u tag=%u val=%llu\n",
                        n, term_tag(g_t.term), (unsigned long long)term_val(g_t.term));
        if (g) {
            f32 e0 = expf(2.0f), e1 = expf(1.0f), e2 = expf(0.1f), s = e0 + e1 + e2;
            CHECK_NEAR(g[0], (e0/s) - 1.0f, 2e-3f, "ce grad0");
            CHECK_NEAR(g[1], (e1/s),         2e-3f, "ce grad1");
            CHECK_NEAR(g[2], (e2/s),         2e-3f, "ce grad2");
        }
        thvm_free(ctx);
    }

    return g_fail ? 1 : 0;
}

int main(void) {
    int rc = 0;
    rc |= run_for_device("cpu");
#ifdef __APPLE__
    if (MTLCreateSystemDefaultDevice()) {
        rc |= run_for_device("metal");
    } else {
        fprintf(stderr, "SKIP metal (no default device)\n");
    }
#endif
    if (rc) {
        fprintf(stderr, "test_tensor_nn_backward: FAIL\n");
        return 1;
    }
    fprintf(stderr, "test_tensor_nn_backward: OK\n");
    return 0;
}
