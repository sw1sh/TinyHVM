// test_conv_simple.m — direct eval of conv2d output, both backends.
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
    f32 b[] = {0};
    Term tx = thvm_tensor(ctx, x, (Shape){.dims={1,1,3,3}, .rank=4});
    Term tw = thvm_tensor(ctx, w, (Shape){.dims={1,1,2,2}, .rank=4});
    Term tb = thvm_tensor(ctx, b, SHAPE(1));
    Term h = thvm_conv2d(ctx, tx, tw, tb, 1, (u32[]){1,1}, (u32[]){0,0,0,0});
    Term out = thvm_eval(ctx, h);
    u32 dt = DTYPE_F32;
    Shape shp = SHAPE(1);
    f32 *r = thvm_to_host_raw(ctx, out, &dt, &shp);
    if (!r) { printf("%s: FAIL (not readable)\n", backend); return 1; }

    // Expected: [1+5, 2+6, 4+8, 5+9] = [6, 8, 12, 14]
    f32 exp[] = {6, 8, 12, 14};
    printf("%-6s: [%.1f, %.1f, %.1f, %.1f]  expect [6, 8, 12, 14]\n",
        backend, r[0], r[1], r[2], r[3]);
    int ok = 1;
    for (int i = 0; i < 4; i++) if (fabsf(r[i] - exp[i]) > 1e-3f) ok = 0;
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
