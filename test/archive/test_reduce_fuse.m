// test_reduce_fuse.m — Verify fused-reduce codegen: SUM(MUL(a,b), axes)
// Tests both non-trailing (axis=0) and trailing (axis=1) reductions on [4,3].

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include "../src/nn/datasets.c"
#include "train_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef DEVICE
  #define DEVICE "metal"
#endif

static int all_pass = 1;

static void check(const char *name, const f32 *got, const f32 *exp, u32 n) {
    float max_err = 0.0f;
    for (u32 i = 0; i < n; i++) {
        float e = fabsf(got[i] - exp[i]);
        if (e > max_err) max_err = e;
    }
    int ok = (max_err < 1e-4f);
    printf("  %-40s %s  (max_err=%.2e)\n", name, ok ? "PASS" : "FAIL", (double)max_err);
    if (!ok) {
        printf("    got: ");
        for (u32 i = 0; i < n; i++) printf("%.6f ", got[i]);
        printf("\n    exp: ");
        for (u32 i = 0; i < n; i++) printf("%.6f ", exp[i]);
        printf("\n");
        all_pass = 0;
    }
}

int main(void) {
    printf("test_reduce_fuse\n");

    TinyHVM *ctx = thvm_init(DEVICE);

    // ----------------------------------------------------------------
    // Known-value [4,3] tensors
    // a[i][j] = (i*3 + j + 1)       → 1..12
    // b[i][j] = (i*3 + j + 1) * 0.5 → 0.5..6.0
    // ----------------------------------------------------------------
    f32 a_data[12], b_data[12];
    for (int i = 0; i < 12; i++) {
        a_data[i] = (f32)(i + 1);
        b_data[i] = (f32)(i + 1) * 0.5f;
    }

    // ================================================================
    // Test 1: SUM(MUL(a, b), axes=[0])  — reduce over axis 0 (rows)
    // Expected: c[j] = sum_i (a[i][j] * b[i][j])
    //   col 0: 1*0.5 + 4*2 + 7*3.5 + 10*5   = 0.5+8+24.5+50    = 83.0
    //   col 1: 2*1   + 5*2.5 + 8*4   + 11*5.5= 2+12.5+32+60.5  = 107.0
    //   col 2: 3*1.5 + 6*3   + 9*4.5 + 12*6 = 4.5+18+40.5+72   = 135.0
    // ================================================================
    {
        Term ta = thvm_tensor(ctx, a_data, SHAPE(4,3));
        Term tb = thvm_tensor(ctx, b_data, SHAPE(4,3));
        Term mul = thvm_op(ctx, UOP_MUL, ta, tb);

        f32 axes0_f[1] = {0.0f};
        Term axes0 = thvm_tensor(ctx, axes0_f, SHAPE(1));
        Term sum0 = thvm_op(ctx, UOP_SUM, mul, axes0);
        Term out0 = thvm_reduce(ctx, sum0);

        f32 *res0 = thvm_to_host(ctx, out0);
        f32 exp0[3] = {83.0f, 107.0f, 135.0f};
        check("SUM(MUL(a,b), axis=0) [non-trailing]", res0, exp0, 3);
    }

    // ================================================================
    // Test 2: SUM(MUL(a, b), axes=[1])  — reduce over axis 1 (cols)
    // Expected: r[i] = sum_j (a[i][j] * b[i][j])
    //   row 0: 1*0.5 + 2*1   + 3*1.5  = 0.5+2+4.5     = 7.0
    //   row 1: 4*2   + 5*2.5 + 6*3    = 8+12.5+18      = 38.5
    //   row 2: 7*3.5 + 8*4   + 9*4.5  = 24.5+32+40.5   = 97.0
    //   row 3: 10*5  + 11*5.5 + 12*6  = 50+60.5+72     = 182.5
    // ================================================================
    {
        Term ta = thvm_tensor(ctx, a_data, SHAPE(4,3));
        Term tb = thvm_tensor(ctx, b_data, SHAPE(4,3));
        Term mul = thvm_op(ctx, UOP_MUL, ta, tb);

        f32 axes1_f[1] = {1.0f};
        Term axes1 = thvm_tensor(ctx, axes1_f, SHAPE(1));
        Term sum1 = thvm_op(ctx, UOP_SUM, mul, axes1);
        Term out1 = thvm_reduce(ctx, sum1);

        f32 *res1 = thvm_to_host(ctx, out1);
        f32 exp1[4] = {7.0f, 38.5f, 97.0f, 182.5f};
        check("SUM(MUL(a,b), axis=1) [trailing]", res1, exp1, 4);
    }

    // ================================================================
    // Test 3: SUM(MUL(a, b), axes=[0,1])  — reduce over both axes
    // Expected: scalar = sum of all a[i][j]*b[i][j]
    //   = 7 + 38.5 + 97 + 182.5 = 325.0
    // ================================================================
    {
        Term ta = thvm_tensor(ctx, a_data, SHAPE(4,3));
        Term tb = thvm_tensor(ctx, b_data, SHAPE(4,3));
        Term mul = thvm_op(ctx, UOP_MUL, ta, tb);

        f32 axes_both_f[2] = {0.0f, 1.0f};
        Term axes_both = thvm_tensor(ctx, axes_both_f, SHAPE(2));
        Term sum_both = thvm_op(ctx, UOP_SUM, mul, axes_both);
        Term out_both = thvm_reduce(ctx, sum_both);

        f32 *res_both = thvm_to_host(ctx, out_both);
        f32 exp_both[1] = {325.0f};
        check("SUM(MUL(a,b), axes=[0,1]) [multi-axis]", res_both, exp_both, 1);
    }

    // ================================================================
    // Test 4: SUM(ADD(a, b), axis=0)  — different elementwise op
    // Expected: c[j] = sum_i (a[i][j] + b[i][j])
    //   col 0: (1+0.5)+(4+2)+(7+3.5)+(10+5) = 1.5+6+10.5+15 = 33.0
    //   col 1: (2+1)+(5+2.5)+(8+4)+(11+5.5) = 3+7.5+12+16.5 = 39.0
    //   col 2: (3+1.5)+(6+3)+(9+4.5)+(12+6) = 4.5+9+13.5+18 = 45.0
    // ================================================================
    {
        Term ta = thvm_tensor(ctx, a_data, SHAPE(4,3));
        Term tb = thvm_tensor(ctx, b_data, SHAPE(4,3));
        Term add = thvm_op(ctx, UOP_ADD, ta, tb);

        f32 axes0_f[1] = {0.0f};
        Term axes0 = thvm_tensor(ctx, axes0_f, SHAPE(1));
        Term sum0 = thvm_op(ctx, UOP_SUM, add, axes0);
        Term out0 = thvm_reduce(ctx, sum0);

        f32 *res0 = thvm_to_host(ctx, out0);
        f32 exp0[3] = {33.0f, 39.0f, 45.0f};
        check("SUM(ADD(a,b), axis=0)", res0, exp0, 3);
    }

    // ================================================================
    // Test 5: Larger [8,4] tensor, axis=0 to stress non-trailing case
    // a[i][j] = i*4+j+1, b[i][j] = 1.0 (all ones → SUM(MUL(a,1)) = col sums)
    // Expected: c[j] = sum_i a[i][j]
    //   col 0: 1+5+9+13+17+21+25+29  = 120
    //   col 1: 2+6+10+14+18+22+26+30 = 128
    //   col 2: 3+7+11+15+19+23+27+31 = 136
    //   col 3: 4+8+12+16+20+24+28+32 = 144
    // ================================================================
    {
        f32 big_a[32], big_b[32];
        for (int i = 0; i < 32; i++) { big_a[i] = (f32)(i + 1); big_b[i] = 1.0f; }

        Term ta = thvm_tensor(ctx, big_a, SHAPE(8,4));
        Term tb = thvm_tensor(ctx, big_b, SHAPE(8,4));
        Term mul = thvm_op(ctx, UOP_MUL, ta, tb);

        f32 axes0_f[1] = {0.0f};
        Term axes0 = thvm_tensor(ctx, axes0_f, SHAPE(1));
        Term sum0 = thvm_op(ctx, UOP_SUM, mul, axes0);
        Term out0 = thvm_reduce(ctx, sum0);

        f32 *res0 = thvm_to_host(ctx, out0);
        f32 exp0[4] = {120.0f, 128.0f, 136.0f, 144.0f};
        check("SUM(MUL(a,ones), axis=0) [8x4 non-trailing]", res0, exp0, 4);
    }

    thvm_free(ctx);

    printf("\n%s\n", all_pass ? "ALL PASS" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
