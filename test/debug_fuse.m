// debug_fuse.m — test a single fused kernel against unfused
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
    TinyHVM *ctx = thvm_init("metal");

    // Simple test: c = a * b + a (3 ops, 2 leaves)
    f32 ad[] = {1,2,3,4,5,6,7,8};
    f32 bd[] = {0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0};
    Term A = thvm_tensor(ctx, ad, SHAPE(2,4));
    Term B = thvm_tensor(ctx, bd, SHAPE(2,4));

    // Build lazy tree: MUL(A, B) + A
    Term mul_ab = thvm_op(ctx, UOP_MUL, A, B);
    Term result = thvm_op(ctx, UOP_ADD, mul_ab, A);
    // Expected: (a*b)+a = a*(b+1)
    // [0]: 1*0.5+1=1.5, [1]: 2*1+2=4, [2]: 3*1.5+3=7.5, [3]: 4*2+4=12

    printf("Test 1: MUL(A,B) + A\n");

    // Unfused
    Term unf = thvm_reduce(ctx, thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MUL, A, B), A));
    f32 *ud = thvm_to_host(ctx, unf);
    printf("  unfused: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
           ud[0],ud[1],ud[2],ud[3],ud[4],ud[5],ud[6],ud[7]);

    // Fused via fuse_or_reduce
    // Re-enable fusion: build fresh lazy tree
    Term mul2 = thvm_op(ctx, UOP_MUL, A, B);
    Term res2 = thvm_op(ctx, UOP_ADD, mul2, A);
    u32 fused_id = fuse_or_reduce(ctx, res2);
    f32 *fd = thvm_to_host(ctx, term_ten(fused_id, DTYPE_F32));
    printf("  fused:   [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
           fd[0],fd[1],fd[2],fd[3],fd[4],fd[5],fd[6],fd[7]);

    // Test 2: with broadcast — a[2,4] * scalar[1]
    printf("\nTest 2: MUL(A, scalar) + B\n");
    f32 sv = 2.0f;
    Term S = thvm_tensor(ctx, &sv, SHAPE(1));
    Term mul3 = thvm_op(ctx, UOP_MUL, A, S);
    Term res3 = thvm_op(ctx, UOP_ADD, mul3, B);
    // Expected: a*2+b = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]

    Term unf3 = thvm_reduce(ctx, thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MUL, A, S), B));
    f32 *ud3 = thvm_to_host(ctx, unf3);
    printf("  unfused: [%.2f, %.2f, %.2f, %.2f]\n", ud3[0],ud3[1],ud3[2],ud3[3]);

    Term mul3b = thvm_op(ctx, UOP_MUL, A, S);
    Term res3b = thvm_op(ctx, UOP_ADD, mul3b, B);
    u32 fused3 = fuse_or_reduce(ctx, res3b);
    f32 *fd3 = thvm_to_host(ctx, term_ten(fused3, DTYPE_F32));
    printf("  fused:   [%.2f, %.2f, %.2f, %.2f]\n", fd3[0],fd3[1],fd3[2],fd3[3]);

    // Test 3: NEG(MUL(A, B)) — unary + binary
    printf("\nTest 3: NEG(MUL(A, B))\n");
    Term unf4 = thvm_reduce(ctx, thvm_op(ctx, UOP_NEG, thvm_op(ctx, UOP_MUL, A, B), term_era()));
    f32 *ud4 = thvm_to_host(ctx, unf4);
    printf("  unfused: [%.2f, %.2f, %.2f, %.2f]\n", ud4[0],ud4[1],ud4[2],ud4[3]);

    Term nmul = thvm_op(ctx, UOP_NEG, thvm_op(ctx, UOP_MUL, A, B), term_era());
    u32 fused4 = fuse_or_reduce(ctx, nmul);
    f32 *fd4 = thvm_to_host(ctx, term_ten(fused4, DTYPE_F32));
    printf("  fused:   [%.2f, %.2f, %.2f, %.2f]\n", fd4[0],fd4[1],fd4[2],fd4[3]);

    thvm_free(ctx);
    return 0;
}
