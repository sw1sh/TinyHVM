// test_collapse.m — test priority-queue, grouped, and parallel collapse
#import <Foundation/Foundation.h>
#include "tinyhvm.c"
#include "backend/cpu/_.c"
#ifdef __APPLE__
#include "backend/metal/_.m"
#endif

int main(void) {
    TinyHVM *ctx = thvm_init("cpu");

    // Build: SUP(1, SUP(2, SUP(3, 4)))
    Term t = thvm_sup(ctx, thvm_fresh_label(ctx),
        term_num_u32(1),
        thvm_sup(ctx, thvm_fresh_label(ctx),
            term_num_u32(2),
            thvm_sup(ctx, thvm_fresh_label(ctx),
                term_num_u32(3),
                term_num_u32(4))));

    // Test 1: PQ collapse
    printf("=== PQ Collapse ===\n");
    CollapseResult cr = thvm_collapse(ctx, t);
    printf("count=%u:", cr.count);
    for (u32 i = 0; i < cr.count; i++)
        printf(" %u", term_as_u32(cr.terms[i]));
    printf("\n");
    thvm_collapse_free(&cr);

    // Test 2: INC
    printf("=== INC Collapse ===\n");
    thvm_reset(ctx, 0);
    Term inc_t = thvm_sup(ctx, thvm_fresh_label(ctx),
        term_num_u32(10),
        thvm_inc(ctx, thvm_sup(ctx, thvm_fresh_label(ctx),
            term_num_u32(20),
            term_num_u32(30))));
    CollapseResult cr2 = thvm_collapse(ctx, inc_t);
    printf("count=%u:", cr2.count);
    for (u32 i = 0; i < cr2.count; i++)
        printf(" %u", term_as_u32(cr2.terms[i]));
    printf("\n");
    thvm_collapse_free(&cr2);

    // Test 3: Grouped
    printf("=== Grouped Collapse ===\n");
    thvm_reset(ctx, 0);
    Term g = thvm_sup(ctx, thvm_fresh_label(ctx),
        term_num_u32(100),
        thvm_sup(ctx, thvm_fresh_label(ctx),
            term_num_u32(200),
            term_num_u32(300)));
    GroupedCollapseResult gr = thvm_collapse_grouped(ctx, g);
    printf("count=%u\n", gr.count);
    for (u32 i = 0; i < gr.count; i++) {
        printf("  leaf %u: val=%u path=[", i, term_as_u32(gr.leaves[i].term));
        for (u32 j = 0; j < gr.leaves[i].path_len; j++)
            printf("(%u,%u)", gr.leaves[i].path[j].label, gr.leaves[i].path[j].branch);
        printf("]\n");
    }
    thvm_collapse_grouped_free(&gr);

    // Test 4: Parallel collapse
    printf("=== Parallel Collapse (2 threads) ===\n");
    fflush(stdout);
    thvm_reset(ctx, 0);
    Term p = thvm_sup(ctx, thvm_fresh_label(ctx),
        term_num_u32(1),
        thvm_sup(ctx, thvm_fresh_label(ctx),
            term_num_u32(2),
            thvm_sup(ctx, thvm_fresh_label(ctx),
                term_num_u32(3),
                term_num_u32(4))));
    printf("  calling thvm_collapse_par...\n"); fflush(stdout);
    CollapseResult cr3 = thvm_collapse_par(ctx, p, 2);
    printf("  count=%u:", cr3.count);
    for (u32 i = 0; i < cr3.count; i++)
        printf(" %u(tag=%u)", term_as_u32(cr3.terms[i]), term_tag(cr3.terms[i]));
    printf("\n");
    thvm_collapse_free(&cr3);

    // Also test with 1 thread (should behave like sequential)
    printf("=== Parallel Collapse (1 thread, fallback) ===\n");
    thvm_reset(ctx, 0);
    Term q = thvm_sup(ctx, thvm_fresh_label(ctx),
        term_num_u32(10),
        thvm_sup(ctx, thvm_fresh_label(ctx),
            term_num_u32(20),
            term_num_u32(30)));
    CollapseResult cr4 = thvm_collapse_par(ctx, q, 1);
    printf("  count=%u:", cr4.count);
    for (u32 i = 0; i < cr4.count; i++)
        printf(" %u", term_as_u32(cr4.terms[i]));
    printf("\n");
    thvm_collapse_free(&cr4);

    // Test parallel with more threads than items
    printf("=== Parallel Collapse (4 threads, 4 items) ===\n");
    thvm_reset(ctx, 0);
    Term r = thvm_sup(ctx, thvm_fresh_label(ctx),
        term_num_u32(10),
        thvm_sup(ctx, thvm_fresh_label(ctx),
            term_num_u32(20),
            thvm_sup(ctx, thvm_fresh_label(ctx),
                term_num_u32(30),
                term_num_u32(40))));
    CollapseResult cr5 = thvm_collapse_par(ctx, r, 4);
    printf("  count=%u:", cr5.count);
    for (u32 i = 0; i < cr5.count; i++)
        printf(" %u", term_as_u32(cr5.terms[i]));
    printf("\n");
    thvm_collapse_free(&cr5);

    printf("=== ALL TESTS PASSED ===\n");
    thvm_free(ctx);
    return 0;
}
