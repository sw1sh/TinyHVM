// test_conv_bwd_simple.m — conv forward + backward on both backends.
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
    f32 x[] = {1,2,3, 4,5,6, 7,8,9};
    f32 w[] = {1,0, 0,1};
    f32 bd[] = {0};
    Term tx = thvm_tensor(ctx, x, (Shape){.dims={1,1,3,3}, .rank=4});
    Term tw = thvm_tensor(ctx, w, (Shape){.dims={1,1,2,2}, .rank=4});
    Term tb = thvm_tensor(ctx, bd, SHAPE(1));
    thvm_set_requires_grad(ctx, tw);
    thvm_set_requires_grad(ctx, tb);

    Term w_fwd, w_grad;
    thvm_dup(ctx, thvm_fresh_label(ctx), tw, &w_fwd, &w_grad);
    Term b_fwd, b_grad;
    thvm_dup(ctx, thvm_fresh_label(ctx), tb, &b_fwd, &b_grad);

    Term h = thvm_conv2d(ctx, tx, w_fwd, b_fwd, 1, (u32[]){1,1}, (u32[]){0,0,0,0});
    Term loss = thvm_sum_axes(ctx, h, (u32[]){0,1,2,3}, 4);

    Term params[] = {w_grad, b_grad};
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));
    if (term_tag(bundle) != TAG_CTR) { printf("%s: FAIL not CTR\n", backend); return 1; }

    u32 dt = DTYPE_F32; Shape sh = SHAPE(1);
    f32 *gw = thvm_to_host_raw(ctx, thvm_grad_bundle_get(ctx, bundle, 0), &dt, &sh);
    f32 *gb = thvm_to_host_raw(ctx, thvm_grad_bundle_get(ctx, bundle, 1), &dt, &sh);
    if (!gw || !gb) { printf("%s: FAIL readback\n", backend); return 1; }

    // Conv output = 2x2 sliding 2x2 kernel over 3x3 input.
    // d(loss)/d(w[i,j]) = sum over output positions of x[output_pos + (i,j)].
    // For kernel (2x2) at positions (0,0),(0,1),(1,0),(1,1):
    //   dw[0,0] = x[0,0]+x[0,1]+x[1,0]+x[1,1] = 1+2+4+5 = 12
    //   dw[0,1] = x[0,1]+x[0,2]+x[1,1]+x[1,2] = 2+3+5+6 = 16
    //   dw[1,0] = x[1,0]+x[1,1]+x[2,0]+x[2,1] = 4+5+7+8 = 24
    //   dw[1,1] = x[1,1]+x[1,2]+x[2,1]+x[2,2] = 5+6+8+9 = 28
    // db = sum(ones at 4 output positions) = 4
    f32 exp_gw[] = {12, 16, 24, 28};
    f32 exp_gb[] = {4};

    printf("%-6s: gw=[%.1f,%.1f,%.1f,%.1f] exp [12,16,24,28] gb=%.1f exp 4\n",
        backend, gw[0], gw[1], gw[2], gw[3], gb[0]);
    int ok = 1;
    for (int i = 0; i < 4; i++) if (fabsf(gw[i] - exp_gw[i]) > 1e-3f) ok = 0;
    if (fabsf(gb[0] - exp_gb[0]) > 1e-3f) ok = 0;
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
