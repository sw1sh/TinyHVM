// test_tensor_nn_layers.m — high-level Tensor NN wrappers parity checks
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
    fprintf(stderr, "== device=%s ==\n", device);

    // --- conv2d forward ---
    {
        TinyHVM *ctx = thvm_init(device);
        f32 xdat[] = {1,2,3,4,5,6,7,8,9}; // [1,1,3,3]
        f32 wdat[] = {1,0,0,1};           // [1,1,2,2]
        f32 bdat[] = {1};
        Tensor x = tensor_from_f32(ctx, xdat, SHAPE(1,1,3,3));
        Tensor w = tensor_from_f32(ctx, wdat, SHAPE(1,1,2,2));
        Tensor b = tensor_from_f32(ctx, bdat, SHAPE(1));
        u32 s[] = {1,1}, p[] = {0,0,0,0};
        Tensor y = tensor_conv2d(x, w, &b, 1, s, p);
        f32 *out = tensor_to_host_f32(tensor_realize(y));
        CHECK(out != NULL, "conv forward to_host");
        CHECK_NEAR(out[0], 7.0f, 1e-3f, "conv[0]");
        CHECK_NEAR(out[1], 9.0f, 1e-3f, "conv[1]");
        CHECK_NEAR(out[2], 13.0f, 1e-3f, "conv[2]");
        CHECK_NEAR(out[3], 15.0f, 1e-3f, "conv[3]");

        thvm_free(ctx);
    }

    // --- maxpool forward ---
    {
        TinyHVM *ctx = thvm_init(device);
        f32 xdat[] = {1,2,3,4}; // [1,1,2,2]
        Tensor x = tensor_from_f32(ctx, xdat, SHAPE(1,1,2,2));
        thvm_set_requires_grad(ctx, x.term);
        u32 k[] = {2,2}, s[] = {1,1};
        Tensor y = tensor_maxpool2d(x, k, s);
        f32 *out = tensor_to_host_f32(tensor_realize(y));
        CHECK(out != NULL, "pool forward to_host");
        CHECK_NEAR(out[0], 4.0f, 1e-3f, "pool max");

        thvm_free(ctx);
    }

    // --- batchnorm eval forward ---
    {
        TinyHVM *ctx = thvm_init(device);
        f32 xdat[] = {1,2,3,4};
        f32 gdat[] = {1}, bdat[] = {0}, mdat[] = {0}, vdat[] = {1};
        Tensor x = tensor_from_f32(ctx, xdat, SHAPE(1,1,2,2));
        Tensor g = tensor_from_f32(ctx, gdat, SHAPE(1));
        Tensor b = tensor_from_f32(ctx, bdat, SHAPE(1));
        Tensor m = tensor_from_f32(ctx, mdat, SHAPE(1));
        Tensor v = tensor_from_f32(ctx, vdat, SHAPE(1));
        Tensor y = tensor_batchnorm(x, g, b, m, v, 1,1,2,2, 0);
        f32 *yy = tensor_to_host_f32(tensor_realize(y));
        CHECK(yy != NULL, "bn to_host");
        CHECK_NEAR(yy[0], 1.0f, 5e-3f, "bn y0");
        CHECK_NEAR(yy[3], 4.0f, 5e-3f, "bn y3");
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
        fprintf(stderr, "test_tensor_nn_layers: FAIL\n");
        return 1;
    }
    fprintf(stderr, "test_tensor_nn_layers: OK\n");
    return 0;
}
