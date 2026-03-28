// bench_ic.m — IC reduction rate benchmark
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <time.h>
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(void) {
    printf("=== TinyHVM IC Reduction Rate ===\n\n");

    // TAG_TEN path (tensor-backed, full overhead)
    { TinyHVM *ctx = thvm_init("cpu"); u32 N=100000;
      f32 v=1.0f; Term t = thvm_tensor(ctx, &v, SHAPE(1));
      for (u32 i=0;i<N;i++) t = thvm_op_raw(ctx, UOP_NEG, t, term_era());
      ctx->itrs=0; double t0=now(); thvm_reduce(ctx,t); double e=now()-t0;
      printf("TAG_TEN NEG %6u: %5.1f Mitrs/s (%3.0f ns/itr)\n", N, ctx->itrs/e/1e6, e/ctx->itrs*1e9);
      thvm_free(ctx); }

    // TAG_NUM path (inline scalar, no tensors)
    { TinyHVM *ctx = thvm_init("cpu"); u32 N=100000;
      Term t = term_num_f32(1.0f);
      for (u32 i=0;i<N;i++) t = thvm_op_raw(ctx, UOP_NEG, t, term_era());
      ctx->itrs=0; double t0=now(); thvm_reduce(ctx,t); double e=now()-t0;
      printf("TAG_NUM NEG %6u: %5.1f Mitrs/s (%3.0f ns/itr)\n", N, ctx->itrs/e/1e6, e/ctx->itrs*1e9);
      thvm_free(ctx); }

    // TAG_NUM ADD chain
    { TinyHVM *ctx = thvm_init("cpu"); u32 N=100000;
      Term t = term_num_f32(0.0f);
      for (u32 i=0;i<N;i++) t = thvm_op_raw(ctx, UOP_ADD, t, term_num_f32(1.0f));
      ctx->itrs=0; double t0=now(); Term r=thvm_reduce(ctx,t); double e=now()-t0;
      f32 rv = term_as_f32(r);
      printf("TAG_NUM ADD %6u: %5.1f Mitrs/s (%3.0f ns/itr) sum=%.0f\n", N, ctx->itrs/e/1e6, e/ctx->itrs*1e9, rv);
      thvm_free(ctx); }

    // TAG_TEN ADD for comparison
    { TinyHVM *ctx = thvm_init("cpu"); u32 N=100000;
      f32 v=0.0f; Term t = thvm_tensor(ctx, &v, SHAPE(1));
      for (u32 i=0;i<N;i++) t = thvm_op_raw(ctx, UOP_ADD, t, thvm_tensor(ctx, &(f32){1.0f}, SHAPE(1)));
      ctx->itrs=0; double t0=now(); Term r=thvm_reduce(ctx,t); double e=now()-t0;
      f32 *rv=thvm_to_host(ctx,r);
      printf("TAG_TEN ADD %6u: %5.1f Mitrs/s (%3.0f ns/itr) sum=%.0f\n", N, ctx->itrs/e/1e6, e/ctx->itrs*1e9, rv?rv[0]:-1);
      thvm_free(ctx); }

    // Scaling: TAG_NUM NEG at different chain lengths
    printf("\nScaling (TAG_NUM NEG):\n");
    for (u32 N = 1000; N <= 1000000; N *= 10) {
        TinyHVM *ctx = thvm_init("cpu");
        Term t = term_num_f32(1.0f);
        for (u32 i=0;i<N;i++) t = thvm_op_raw(ctx, UOP_NEG, t, term_era());
        ctx->itrs=0; double t0=now(); thvm_reduce(ctx,t); double e=now()-t0;
        printf("  N=%7u: %6.1f Mitrs/s (%4.1f ns/itr)\n", N, ctx->itrs/e/1e6, e/ctx->itrs*1e9);
        thvm_free(ctx);
    }

    printf("\nHVM4 reference: ~200 Mitrs/s (single-threaded)\n");
    return 0;
}
