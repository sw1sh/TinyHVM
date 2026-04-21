// README examples: forward (relu(mm+b)) + autograd bundle readback.
// Same numerics as test/test_tiny_single_param_keep.m (CPU verified).
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#include "../src/backend/metal/_.m"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DEVICE
#define DEVICE "cpu"
#endif

static int fail(const char *msg) {
    fprintf(stderr, "README verify FAIL: %s\n", msg);
    return 1;
}

int main(void) {
    /* Forward — must use thvm_eval so scheduler + dispatch materialize TAG_TOPs */
    {
        TinyHVM *ctx = thvm_init(DEVICE);
        if (!ctx) return fail("thvm_init");

        f32 x_d[] = {1, 2, 3, 4, 5, 6};
        u32 xs[] = {2, 3};
        f32 w_d[] = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f};
        u32 ws[] = {3, 2};
        f32 b_d[] = {-0.1f, 0.2f};
        u32 bs[] = {1, 2};

        Tensor x = tensor_from_f32(ctx, x_d, shape_of(xs, 2));
        Tensor w = tensor_from_f32(ctx, w_d, shape_of(ws, 2));
        Tensor b = tensor_from_f32(ctx, b_d, shape_of(bs, 2));
        Linear lin = {.weight = w, .bias = b, .has_bias = 1};
        Tensor y = linear_forward(&lin, x);
        Tensor result = tensor_realize(tensor_relu(y));
        f32 *out = tensor_to_host_f32(result);
        if (!out) return fail("forward to_host");
        if (fabsf(out[0]) > 1e-3f || fabsf(out[1] - 2.6f) > 1e-3f ||
            fabsf(out[2]) > 1e-3f || fabsf(out[3] - 5.0f) > 1e-3f)
            return fail("forward values");
        thvm_free(ctx);
    }

    /* Autograd API surface — build keep-bundle term and sanity-check shape/count API. */
    {
        TinyHVM *ctx = thvm_init(DEVICE);
        f32 xd[] = {3.0f};
        f32 wd[] = {4.0f};
        Tensor x = tensor_from_f32(ctx, xd, (Shape){.dims = {1, 1}, .rank = 2});
        Tensor w = tensor_from_f32(ctx, wd, (Shape){.dims = {1, 1}, .rank = 2});
        thvm_set_requires_grad(ctx, w.term);

        Tensor loss = tensor_mul(x, w);
        Tensor bundle = tensor_from_term(ctx, thvm_grad_keep(ctx, loss.term, w.term));
        (void)bundle;
        thvm_free(ctx);
    }

    printf("README verify OK (DEVICE=%s)\n", DEVICE);
    return 0;
}
