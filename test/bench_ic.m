#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <time.h>
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
int main(void) {
    printf("=== IC Reduction Benchmark (CPU) ===\n\n");
    u32 lens[] = {100, 1000, 10000, 100000};
    for (int li = 0; li < 4; li++) {
        u32 len = lens[li];
        TinyHVM *ctx = thvm_init("cpu");
        f32 v = 1.0f; Term t = thvm_tensor(ctx, &v, SHAPE(1));
        for (u32 i = 0; i < len; i++) t = thvm_op(ctx, UOP_NEG, t, term_era());
        ctx->itrs = 0;
        double t0 = now();
        Term r = thvm_reduce(ctx, t);
        double elapsed = now() - t0;
        printf("NEG %6u: %7llu itrs, %7.1fms, %5.1f Mitrs/s\n",
            len, (unsigned long long)ctx->itrs, elapsed*1000,
            elapsed > 0 ? ctx->itrs/elapsed/1e6 : 0);
        thvm_free(ctx);
    }
    printf("\n");
    for (u32 n = 100; n <= 100000; n *= 10) {
        TinyHVM *ctx = thvm_init("cpu");
        f32 v = 0.0f; Term t = thvm_tensor(ctx, &v, SHAPE(1));
        for (u32 i = 0; i < n; i++) t = thvm_op(ctx, UOP_ADD, t, thvm_tensor(ctx, &(f32){1.0f}, SHAPE(1)));
        ctx->itrs = 0;
        double t0 = now();
        Term r = thvm_reduce(ctx, t);
        double elapsed = now() - t0;
        f32 *rv = thvm_to_host(ctx, r);
        printf("ADD %6u: %7llu itrs, %7.1fms, %5.1f Mitrs/s  sum=%.0f\n",
            n, (unsigned long long)ctx->itrs, elapsed*1000,
            elapsed > 0 ? ctx->itrs/elapsed/1e6 : 0, rv?rv[0]:-1);
        thvm_free(ctx);
    }
    return 0;
}
