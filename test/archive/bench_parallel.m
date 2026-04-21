// bench_parallel.m — Multi-threaded IC throughput benchmark
// Independent chains per thread (no shared state) to measure scaling potential
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/parallel/reduce.c"
#include <stdio.h>
#include <time.h>

int main(void) {
    printf("=== Parallel IC Throughput (independent chains per thread) ===\n\n");
    printf("Each thread: build + reduce a TAG_NUM ADD chain of 10K ops\n\n");

    u32 chain_len = 10000;
    for (u32 nt = 1; nt <= 12; nt = (nt < 4) ? nt + 1 : (nt < 8) ? nt + 2 : nt + 4) {
        parallel_bench(nt, chain_len);
    }

    printf("\nSingle-thread reference: ~130 Mitrs/s (TAG_NUM ADD)\n");
    printf("Perfect scaling at 12 threads: ~1560 Mitrs/s\n");
    return 0;
}
