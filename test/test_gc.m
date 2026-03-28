// test_gc.m — Refcount GC tests for tensor lifetime management
//
// Tests that tensors are automatically freed when their refcount
// drops to zero through inet interactions (APP-TEN, clone, etc).

#define DEVICE "cpu"
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#include "../src/backend/metal/_.m"
#include "train_helpers.h"
#include <stdio.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); return 1; } \
} while(0)

// ── Test 1: Fresh tensor has refcount 1 ──────────────────────
static int test_initial_refcount(void) {
    TinyHVM *ctx = thvm_init(DEVICE);
    f32 data[] = {1.0f, 2.0f, 3.0f};
    Term t = thvm_tensor(ctx, data, SHAPE(3));
    u32 id = (u32)term_val(t);

    ASSERT(ctx->tensors[id].refcount == 1, "initial refcount should be 1");
    printf("  test_initial_refcount: PASS (rc=%u)\n", ctx->tensors[id].refcount);
    thvm_free(ctx);
    return 0;
}

// ── Test 2: Clone increments refcount ────────────────────────
static int test_clone_increments_refcount(void) {
    TinyHVM *ctx = thvm_init(DEVICE);
    f32 data[] = {1.0f, 2.0f};
    Term t = thvm_tensor(ctx, data, SHAPE(2));
    u32 id = (u32)term_val(t);

    ASSERT(ctx->tensors[id].refcount == 1, "before clone: rc should be 1");

    Term lam = thvm_lam(ctx, NULL, t);
    Term cloned = term_clone(ctx, lam);
    (void)cloned;

    ASSERT(ctx->tensors[id].refcount == 2, "after clone: rc should be 2");
    printf("  test_clone_increments_refcount: PASS (rc=%u)\n", ctx->tensors[id].refcount);
    thvm_free(ctx);
    return 0;
}

// ── Test 3: APP-TEN decrements refcount ──────────────────────
static int test_app_ten_decrements_refcount(void) {
    TinyHVM *ctx = thvm_init(DEVICE);
    f32 data[] = {42.0f};
    Term ten = thvm_tensor(ctx, data, SHAPE(1));
    u32 id = (u32)term_val(ten);

    ASSERT(ctx->tensors[id].refcount == 1, "before APP-TEN: rc should be 1");

    Term app = thvm_app(ctx, ten, term_era());
    Term result = thvm_reduce(ctx, app);
    (void)result;

    ASSERT(ctx->tensors[id].refcount == 0, "after APP-TEN: rc should be 0");
    // buf_id not zeroed — buffer freeing deferred to thvm_reset (view-shared buf safety)
    printf("  test_app_ten_decrements_refcount: PASS (rc=%u)\n",
           ctx->tensors[id].refcount);
    thvm_free(ctx);
    return 0;
}

// ── Test 4: Clone + APP-TEN = net zero (rc goes 1→2→1) ──────
static int test_clone_then_discard(void) {
    TinyHVM *ctx = thvm_init(DEVICE);
    f32 data[] = {7.0f};
    Term ten = thvm_tensor(ctx, data, SHAPE(1));
    u32 id = (u32)term_val(ten);

    Term lam = thvm_lam(ctx, NULL, ten);
    Term cloned = term_clone(ctx, lam);

    ASSERT(ctx->tensors[id].refcount == 2, "after clone: rc should be 2");

    u64 cloned_loc = term_val(cloned);
    Term cloned_ten = heap_read(ctx, cloned_loc + 1);
    Term app = thvm_app(ctx, cloned_ten, term_era());
    thvm_reduce(ctx, app);

    ASSERT(ctx->tensors[id].refcount == 1, "after discard clone: rc should be 1");
    ASSERT(ctx->tensors[id].buf_id != 0, "buf should still exist (original ref)");
    printf("  test_clone_then_discard: PASS (rc=%u)\n", ctx->tensors[id].refcount);
    thvm_free(ctx);
    return 0;
}

// ── Test 5: APP chain with multiple intermediate tensors ─────
static int test_app_chain_frees_intermediates(void) {
    TinyHVM *ctx = thvm_init(DEVICE);

    f32 d1[] = {1.0f}, d2[] = {2.0f}, d3[] = {3.0f};
    Term t1 = thvm_tensor(ctx, d1, SHAPE(1));
    Term t2 = thvm_tensor(ctx, d2, SHAPE(1));
    Term t3 = thvm_tensor(ctx, d3, SHAPE(1));
    u32 id1 = (u32)term_val(t1);
    u32 id2 = (u32)term_val(t2);
    u32 id3 = (u32)term_val(t3);

    Term chain = thvm_app(ctx, t1, thvm_app(ctx, t2, t3));
    Term result = thvm_reduce(ctx, chain);

    ASSERT(ctx->tensors[id1].refcount == 0, "t1 should be freed");
    ASSERT(ctx->tensors[id2].refcount == 0, "t2 should be freed");
    ASSERT(ctx->tensors[id3].refcount == 1, "t3 should survive (it's the result)");
    ASSERT(term_tag(result) == TAG_TEN && (u32)term_val(result) == id3,
           "result should be t3");
    printf("  test_app_chain_frees_intermediates: PASS (rc1=%u rc2=%u rc3=%u)\n",
           ctx->tensors[id1].refcount, ctx->tensors[id2].refcount,
           ctx->tensors[id3].refcount);
    thvm_free(ctx);
    return 0;
}

// ── Test 6: Recursive training tensor pressure ───────────────
// Run 5 training steps, then check how many tensors were freed by GC.
static int test_training_tensor_pressure(void) {
    TinyHVM *ctx = thvm_init(DEVICE);

    f32 x_d[] = {0,0, 0,1, 1,0, 1,1};
    f32 y_d[] = {0, 1, 1, 0};
    f32 w1_d[] = {0.5f,-0.3f, 0.2f, 0.4f,-0.1f, 0.6f,-0.4f, 0.3f};
    f32 b1_d[] = {0,0,0,0};
    f32 w2_d[] = {0.5f,-0.3f,0.2f,0.4f};
    f32 b2_d[] = {0};
    f32 lr_d[] = {0.1f};

    Term X  = thvm_tensor(ctx, x_d,  SHAPE(4, 2));
    Term Y  = thvm_tensor(ctx, y_d,  SHAPE(4, 1));
    Term W1 = thvm_tensor(ctx, w1_d, SHAPE(2, 4));
    Term B1 = thvm_tensor(ctx, b1_d, SHAPE(1, 4));
    Term W2 = thvm_tensor(ctx, w2_d, SHAPE(4, 1));
    Term B2 = thvm_tensor(ctx, b2_d, SHAPE(1, 1));
    Term LR = thvm_tensor(ctx, lr_d, SHAPE(1));

    u32 tensors_before = ctx->tensor_count;

    Term program = train_program(ctx, W1, B1, W2, B2, X, Y, LR, 5);
    thvm_reduce(ctx, program);

    u32 tensors_after = ctx->tensor_count;
    u32 freed = 0, live = 0;
    for (u32 i = 0; i < tensors_after; i++) {
        if (ctx->tensors[i].refcount == 0) freed++;
        else live++;
    }

    printf("  test_training_tensor_pressure:\n");
    printf("    tensors allocated: %u (before=%u)\n", tensors_after, tensors_before);
    printf("    live (rc>0): %u, freed (rc=0): %u\n", live, freed);
    printf("    NOTE: intermediate compute tensors not yet freed —\n");
    printf("          only APP-TEN discards trigger decref currently.\n");
    // This test documents the current state. As more decref hooks are added
    // (e.g. in TOP consumption, ERA-TEN interaction), freed count will grow.
    printf("  PASS (observation recorded)\n");
    thvm_free(ctx);
    return 0;
}

int main(void) {
    printf("=== Tensor Refcount GC Tests ===\n\n");
    int fail = 0;
    fail |= test_initial_refcount();
    fail |= test_clone_increments_refcount();
    fail |= test_app_ten_decrements_refcount();
    fail |= test_clone_then_discard();
    fail |= test_app_chain_frees_intermediates();
    fail |= test_training_tensor_pressure();
    printf("\n%s\n", fail ? "=== SOME TESTS FAILED ===" : "=== ALL TESTS PASSED ===");
    return fail;
}

