// bench_hybrid.m — Benchmark hybrid clone-based parallel collapse
#import <Foundation/Foundation.h>
#include "tinyhvm.c"
#include "backend/cpu/_.c"
#ifdef __APPLE__
#include "backend/metal/_.m"
#endif
#include <stdio.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// === Workload A: ADD chain of SUPs (heavy DUP sharing) ===
static Term build_bitsum(TinyHVM *ctx, u32 n_bits) {
    Term bits[32];
    for (u32 i = 0; i < n_bits; i++)
        bits[i] = thvm_sup(ctx, thvm_fresh_label(ctx), term_num_u32(0), term_num_u32(1));
    Term sum = bits[n_bits - 1];
    for (int i = (int)n_bits - 2; i >= 0; i--)
        sum = thvm_op2(ctx, 0, bits[i], sum);
    return sum;
}

// === Workload B: Balanced SUP tree with leaf computation (NO DUP sharing) ===
// Each leaf is a chain of OP2 adds that take `ops` steps to reduce.
// Subtrees are fully independent — ideal for parallel clone.
static Term build_balanced(TinyHVM *ctx, u32 depth, u32 leaf_ops, u32 *leaf_id) {
    if (depth == 0) {
        Term t = term_num_u32((*leaf_id)++);
        for (u32 i = 0; i < leaf_ops; i++)
            t = thvm_op2(ctx, 0, t, term_num_u32(1));
        return t;
    }
    return thvm_sup(ctx, thvm_fresh_label(ctx),
        build_balanced(ctx, depth - 1, leaf_ops, leaf_id),
        build_balanced(ctx, depth - 1, leaf_ops, leaf_id));
}

static void bench_workload(const char *name, u32 depth, u32 leaf_ops, int use_bitsum) {
    u32 expected = 1u << depth;
    double times[4];

    // Serial
    {
        TinyHVM *ctx = thvm_init("cpu");
        Term tree;
        if (use_bitsum) {
            tree = build_bitsum(ctx, depth);
        } else {
            u32 lid = 0;
            tree = build_balanced(ctx, depth, leaf_ops, &lid);
        }
        double t0 = now_ms();
        CollapseResult cr = thvm_collapse(ctx, tree);
        times[0] = now_ms() - t0;
        if (cr.count != expected) {
            printf("  FAIL: serial count=%u expected=%u\n", cr.count, expected);
            thvm_collapse_free(&cr); thvm_free(ctx); return;
        }
        thvm_collapse_free(&cr); thvm_free(ctx);
    }

    // Parallel: 2, 4, 8
    u32 tcs[] = {2, 4, 8};
    for (u32 ti = 0; ti < 3; ti++) {
        TinyHVM *ctx = thvm_init("cpu");
        Term tree;
        if (use_bitsum) {
            tree = build_bitsum(ctx, depth);
        } else {
            u32 lid = 0;
            tree = build_balanced(ctx, depth, leaf_ops, &lid);
        }
        double t0 = now_ms();
        CollapseResult cr = thvm_collapse_par(ctx, tree, tcs[ti]);
        times[1 + ti] = now_ms() - t0;
        if (cr.count != expected) {
            printf("  FAIL: %ut count=%u expected=%u\n", tcs[ti], cr.count, expected);
            thvm_collapse_free(&cr); thvm_free(ctx); return;
        }
        thvm_collapse_free(&cr); thvm_free(ctx);
    }

    printf("  %-40s %8.1f  %8.1f (%.2fx)  %8.1f (%.2fx)  %8.1f (%.2fx)\n",
           name, times[0],
           times[1], times[0]/times[1],
           times[2], times[0]/times[2],
           times[3], times[0]/times[3]);
}

int main(void) {
    printf("=== Hybrid Clone-Based Parallel Collapse Benchmark ===\n\n");
    printf("  %-40s %8s  %14s  %14s  %14s\n",
           "Workload", "serial", "2-thread", "4-thread", "8-thread");
    printf("  %-40s %8s  %14s  %14s  %14s\n",
           "----------------------------------------", "--------",
           "--------------", "--------------", "--------------");

    // A: ADD chain (DUP-heavy): clone duplicates shared substructure → redundant work
    printf("\n  [A] ADD chain of SUPs (heavy DUP sharing — worst case for clone):\n");
    bench_workload("N=12 bitsum (4K results)", 12, 0, 1);
    bench_workload("N=14 bitsum (16K results)", 14, 0, 1);
    bench_workload("N=16 bitsum (64K results)", 16, 0, 1);

    // B: Balanced tree (no DUPs): subtrees fully independent → ideal for clone
    printf("\n  [B] Balanced SUP tree, independent leaves (best case for clone):\n");
    bench_workload("depth=12 ops=50 (4K leaves)", 12, 50, 0);
    bench_workload("depth=14 ops=50 (16K leaves)", 14, 50, 0);
    bench_workload("depth=16 ops=50 (64K leaves)", 16, 50, 0);
    bench_workload("depth=14 ops=200 (16K leaves, more work)", 14, 200, 0);
    bench_workload("depth=16 ops=200 (64K leaves, more work)", 16, 200, 0);

    printf("\n  Interpretation:\n");
    printf("  - [A] Clone duplicates DUP-shared structure → redundant computation → slower.\n");
    printf("  - [B] Independent subtrees → no redundancy → speedup proportional to threads.\n");

    return 0;
}
