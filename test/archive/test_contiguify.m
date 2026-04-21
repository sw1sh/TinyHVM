// test_contiguify.m — Verify Metal contiguify matches CPU for the specific
// view pattern that appears in conv backward
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    srand(42);

    // Create data on both backends
    u32 n = 4*4*12*12; // [4,4,12,12]
    f32 *data = malloc(n*4);
    for (u32 i=0;i<n;i++) data[i] = (f32)(i+1) * 0.01f;

    // Test: RESHAPE [4,4,12,12] → [1,4,1,4,1,12,1,12]
    // This creates a view with specific strides. Then EXPAND → contiguify.
    for (int gpu = 0; gpu <= 1; gpu++) {
        const char *dev = gpu ? "metal" : "cpu";
        TinyHVM *ctx = thvm_init(dev);
        Term x = thvm_tensor(ctx, data, (Shape){.dims={4,4,12,12},.rank=4});

        // Reshape to [1,4,1,4,1,12,1,12]
        Term r1 = thvm_reshape(ctx, x, (Shape){.dims={1,4,1,4,1,12,1,12},.rank=8});
        // Expand to [1,4,1,4,4,12,4,12]
        Term r2 = thvm_expand(ctx, r1, (Shape){.dims={1,4,1,4,4,12,4,12},.rank=8});
        // Reshape to [4,4,48,48] — this triggers contiguify on Metal
        Term r3 = thvm_reshape(ctx, r2, (Shape){.dims={4,4,48,48},.rank=4});

        // Force evaluation
        thvm_reduce(ctx, r3);
        f32 *out = thvm_to_host(ctx, r3);
        u32 on = 4*4*48*48;
        double norm = 0;
        for (u32 i=0;i<on;i++) norm += out[i]*out[i];
        printf("%-6s reshape→expand→reshape: numel=%u norm=%.4f first=[%.4f,%.4f,%.4f,%.4f]\n",
            dev, on, sqrt(norm), out[0], out[1], out[2], out[3]);
        // Check last elements too
        printf("        last=[%.4f,%.4f,%.4f,%.4f]\n",
            out[on-4], out[on-3], out[on-2], out[on-1]);

        thvm_free(ctx);
    }

    free(data);
    return 0;
}
