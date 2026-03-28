// bench_shared_parallel.m — Shared-context parallel IC benchmark
// All threads reduce subtrees of the SAME heap (shared TinyHVM context).

#define HEAP_CAP (1ULL << 24)  // 16M terms (128MB) for large trees

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/parallel/shared.c"
#include <stdio.h>
#include <time.h>

// Balanced binary tree: 2^depth leaves (1.0f), 2^depth - 1 ADD nodes.
static Term build_tree(TinyHVM *ctx, u32 depth) {
    if (depth == 0) return term_num_f32(1.0f);
    return thvm_op_raw(ctx, UOP_ADD,
                       build_tree(ctx, depth - 1),
                       build_tree(ctx, depth - 1));
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    printf("=== Shared-Context Parallel IC (balanced binary tree) ===\n\n");

    u32 depths[] = {18, 20, 22};
    for (u32 di = 0; di < 3; di++) {
        u32 depth = depths[di];
        u64 n_itrs = (1ULL << depth) - 1;
        printf("Tree depth=%u  interactions=%llu\n",
               depth, (unsigned long long)n_itrs);

        // Single-threaded baseline
        double base_ms;
        {
            TinyHVM *ctx = thvm_init("cpu");
            Term root = build_tree(ctx, depth);
            double t0 = now_ms();
            Term r = thvm_reduce(ctx, root);
            base_ms = now_ms() - t0;
            printf("   1 thr: %6.1f ms  sum=%.0f  (%.0f Mitrs/s)\n",
                   base_ms, term_as_f32(r), n_itrs / (base_ms / 1000) / 1e6);
            thvm_free(ctx);
        }

        // Parallel (shared-context)
        u32 thread_counts[] = {2, 4, 6, 8, 12};
        for (u32 ti = 0; ti < 5; ti++) {
            u32 nt = thread_counts[ti];
            TinyHVM *ctx = thvm_init("cpu");
            Term root = build_tree(ctx, depth);

            double t0 = now_ms();
            Term r = parallel_reduce_tree(ctx, root, nt);
            double ms = now_ms() - t0;

            printf("  %2u thr: %6.1f ms  sum=%.0f  (%.0f Mitrs/s, %.1f×)\n",
                   nt, ms, term_as_f32(r),
                   n_itrs / (ms / 1000) / 1e6, base_ms / ms);
            thvm_free(ctx);
        }
        printf("\n");
    }
    return 0;
}
