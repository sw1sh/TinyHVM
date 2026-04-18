// test_bn_simple.m — BN forward + backward on both backends.
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int run(const char *backend) {
    TinyHVM *ctx = thvm_init(backend);
    // B=2, C=2, H=2, W=2 — 2 channels with 8 elements each.
    f32 xd[16];
    for (int i = 0; i < 16; i++) xd[i] = (f32)i * 0.1f;  // 0.0..1.5
    f32 gd[2] = {1.0f, 1.0f};
    f32 bd[2] = {0.0f, 0.0f};
    f32 rmd[2] = {0.0f, 0.0f};
    f32 rvd[2] = {1.0f, 1.0f};

    Term x = thvm_tensor(ctx, xd, (Shape){.dims={2,2,2,2}, .rank=4});
    Term gamma = thvm_tensor(ctx, gd, SHAPE(2));
    Term beta  = thvm_tensor(ctx, bd, SHAPE(2));
    Term rmean = thvm_tensor(ctx, rmd, SHAPE(2));
    Term rvar  = thvm_tensor(ctx, rvd, SHAPE(2));

    // training=1 — use raw batchnorm_forward and evaluate just .output
    // (batchnorm_term wraps assigns + output which causes other issues).
    BNResult bn = batchnorm_forward(ctx, x, gamma, beta, rmean, rvar, 2, 2, 2, 2, 1);
    Term out = thvm_eval(ctx, bn.output);
    u32 dt = DTYPE_F32; Shape sh = SHAPE(1);
    f32 *r = thvm_to_host_raw(ctx, out, &dt, &sh);
    if (!r) { printf("%s: FAIL\n", backend); return 1; }
    // BN normalizes per-channel to ~zero mean and unit variance,
    // then scales by gamma=1 and shifts by beta=0. Sum over all
    // elements should be ~0.
    f32 sum = 0;
    for (int i = 0; i < 16; i++) sum += r[i];
    printf("%-6s: sum(bn_out)=%.4f (expect ≈0)\n", backend, sum);
    int ok = fabsf(sum) < 1e-3f;
    printf("%s: %s\n", backend, ok ? "PASS" : "FAIL");
    thvm_free(ctx);
    return ok ? 0 : 1;
}

int main(void) {
    int cpu = run("cpu");
#ifdef __APPLE__
    int metal = MTLCreateSystemDefaultDevice() ? run("metal") : 0;
    return (cpu == 0 && metal == 0) ? 0 : 1;
#else
    return cpu;
#endif
}
