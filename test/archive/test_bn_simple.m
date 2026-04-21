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

    // Forward via batchnorm_term (wraps running-stats assigns via SEQ).
    Term y = batchnorm_term(ctx, x, gamma, beta, rmean, rvar, 2, 2, 2, 2, 1);
    Term out = thvm_eval(ctx, y);
    if (term_tag(out) != TAG_TEN) { printf("%s: FAIL fwd tag=%u\n", backend, term_tag(out)); return 1; }
    f32 *r = thvm_to_host(ctx, out);
    f32 sum = 0; for (int i = 0; i < 16; i++) sum += r[i];

    // Fresh ctx for the backward check — avoids re-using consumed g/b.
    thvm_free(ctx);
    ctx = thvm_init(backend);
    Term x2 = thvm_tensor(ctx, xd, (Shape){.dims={2,2,2,2}, .rank=4});
    Term g2 = thvm_tensor(ctx, gd, SHAPE(2));
    Term b2 = thvm_tensor(ctx, bd, SHAPE(2));
    Term rm2 = thvm_tensor(ctx, rmd, SHAPE(2));
    Term rv2 = thvm_tensor(ctx, rvd, SHAPE(2));
    thvm_set_requires_grad(ctx, g2);
    thvm_set_requires_grad(ctx, b2);
    Term g_fwd, g_grad; thvm_dup(ctx, thvm_fresh_label(ctx), g2, &g_fwd, &g_grad);
    Term b_fwd, b_grad; thvm_dup(ctx, thvm_fresh_label(ctx), b2, &b_fwd, &b_grad);

    BNResult bn = batchnorm_forward(ctx, x2, g_fwd, b_fwd, rm2, rv2, 2, 2, 2, 2, 1);
    Term loss = thvm_sum_axes(ctx, bn.output, (u32[]){0,1,2,3}, 4);
    Term params[] = {g_grad, b_grad};
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));
    if (term_tag(bundle) != TAG_CTR) { printf("%s: FAIL bwd bundle tag=%u\n", backend, term_tag(bundle)); return 1; }
    u32 dt = DTYPE_F32; Shape sh = SHAPE(1);
    f32 *gg = thvm_to_host_raw(ctx, thvm_grad_bundle_get(ctx, bundle, 0), &dt, &sh);
    f32 *gb = thvm_to_host_raw(ctx, thvm_grad_bundle_get(ctx, bundle, 1), &dt, &sh);
    if (!gg || !gb) { printf("%s: FAIL readback\n", backend); return 1; }

    printf("%-6s: fwd sum=%.4f  bwd gg=[%.3f,%.3f] gb=[%.3f,%.3f]\n",
        backend, sum, gg[0], gg[1], gb[0], gb[1]);
    int ok = fabsf(sum) < 1e-3f
          && fabsf(gg[0]) < 1e-3f && fabsf(gg[1]) < 1e-3f
          && fabsf(gb[0] - 8.0f) < 1e-3f && fabsf(gb[1] - 8.0f) < 1e-3f;
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
