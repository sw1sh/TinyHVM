// test_metal.c — Metal parity test
// Runs the same core tests as test_term.c on the Metal GPU backend.

#import <Foundation/Foundation.h>
#include "../src/tinyhvm.c"
#include "../src/gpu_metal.m"

#include <stdio.h>

static int tests_passed = 0, tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s (line %d)\n", msg, __LINE__); } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) ASSERT(fabsf((a)-(b)) < (tol), msg)

// ---- Forward ops ----

static void test_matmul(void) {
    printf("test_matmul:\n");
    TinyHVM *ctx = thvm_init(&gpu_metal_backend);

    f32 a[] = {1,2,3,4};
    f32 b[] = {1,0,0,1};  // identity
    Term A = thvm_tensor(ctx, a, SHAPE(2, 2));
    Term B = thvm_tensor(ctx, b, SHAPE(2, 2));
    Term C = thvm_op(ctx, UOP_MM, A, B);
    f32 *out = thvm_to_host(ctx, C);
    ASSERT_NEAR(out[0], 1.0f, 1e-4f, "mm[0,0]=1");
    ASSERT_NEAR(out[1], 2.0f, 1e-4f, "mm[0,1]=2");
    ASSERT_NEAR(out[2], 3.0f, 1e-4f, "mm[1,0]=3");
    ASSERT_NEAR(out[3], 4.0f, 1e-4f, "mm[1,1]=4");

    thvm_free(ctx);
}

static void test_relu(void) {
    printf("test_relu:\n");
    TinyHVM *ctx = thvm_init(&gpu_metal_backend);

    f32 x[] = {-2, 3, -1, 4};
    Term X = thvm_tensor(ctx, x, SHAPE(1, 4));
    Term Y = thvm_op(ctx, UOP_RELU, X, term_era());
    f32 *out = thvm_to_host(ctx, Y);
    ASSERT_NEAR(out[0], 0.0f, 1e-5f, "relu(-2)=0");
    ASSERT_NEAR(out[1], 3.0f, 1e-5f, "relu(3)=3");
    ASSERT_NEAR(out[2], 0.0f, 1e-5f, "relu(-1)=0");
    ASSERT_NEAR(out[3], 4.0f, 1e-5f, "relu(4)=4");

    thvm_free(ctx);
}

static void test_broadcast_add(void) {
    printf("test_broadcast_add:\n");
    TinyHVM *ctx = thvm_init(&gpu_metal_backend);

    f32 a[] = {1,2,3,4,5,6};  // [2,3]
    f32 b[] = {10,20,30};      // [1,3]
    Term A = thvm_tensor(ctx, a, SHAPE(2, 3));
    Term B = thvm_tensor(ctx, b, SHAPE(1, 3));
    Term C = thvm_op(ctx, UOP_ADD, A, B);
    f32 *out = thvm_to_host(ctx, C);
    ASSERT_NEAR(out[0], 11.0f, 1e-4f, "1+10=11");
    ASSERT_NEAR(out[1], 22.0f, 1e-4f, "2+20=22");
    ASSERT_NEAR(out[2], 33.0f, 1e-4f, "3+30=33");
    ASSERT_NEAR(out[3], 14.0f, 1e-4f, "4+10=14");
    ASSERT_NEAR(out[4], 25.0f, 1e-4f, "5+20=25");
    ASSERT_NEAR(out[5], 36.0f, 1e-4f, "6+30=36");

    thvm_free(ctx);
}

static void test_full_forward(void) {
    printf("test_full_forward (relu(mm(x,w)+b)):\n");
    TinyHVM *ctx = thvm_init(&gpu_metal_backend);

    f32 x_data[] = {1, 2};
    f32 w_data[] = {0.5f, -0.5f, 1.0f, 1.0f};
    f32 b_data[] = {0.1f, -0.1f};

    Term x = thvm_tensor(ctx, x_data, SHAPE(1, 2));
    Term w = thvm_tensor(ctx, w_data, SHAPE(2, 2));
    Term b = thvm_tensor(ctx, b_data, SHAPE(1, 2));

    Term mm  = thvm_op(ctx, UOP_MM, x, w);
    Term add = thvm_op(ctx, UOP_ADD, mm, b);
    Term out = thvm_op(ctx, UOP_RELU, add, term_era());

    f32 *result = thvm_to_host(ctx, out);
    // mm = [1*0.5+2*1.0, 1*(-0.5)+2*1.0] = [2.5, 1.5]
    // add = [2.6, 1.4]
    // relu = [2.6, 1.4]
    ASSERT_NEAR(result[0], 2.6f, 1e-4f, "relu(mm+b)[0]=2.6");
    ASSERT_NEAR(result[1], 1.4f, 1e-4f, "relu(mm+b)[1]=1.4");

    thvm_free(ctx);
}

// ---- Autograd ----

static void test_grad_x2(void) {
    printf("test_grad_x2 (d(x^2)/dx on Metal):\n");
    TinyHVM *ctx = thvm_init(&gpu_metal_backend);

    f32 x_d[] = {3.0f};
    Term x = thvm_tensor(ctx, x_d, SHAPE(1, 1));
    thvm_set_requires_grad(ctx, x);

    thvm_clear_tape(ctx);
    thvm_start_recording(ctx);
    Term yr = thvm_reduce(ctx, thvm_op(ctx, UOP_MUL, x, x));
    thvm_stop_recording(ctx);

    f32 *fwd = thvm_to_host(ctx, yr);
    ASSERT_NEAR(fwd[0], 9.0f, 1e-4f, "3*3=9");

    f32 *g = thvm_to_host(ctx, thvm_grad(ctx, yr, x));
    ASSERT(g != NULL, "grad exists");
    ASSERT_NEAR(g[0], 6.0f, 1e-4f, "d(x^2)/dx=6");

    thvm_free(ctx);
}

static void test_grad_mm(void) {
    printf("test_grad_mm (matmul gradient on Metal):\n");
    TinyHVM *ctx = thvm_init(&gpu_metal_backend);

    f32 a_d[] = {1,2,3,4};
    f32 b_d[] = {5,6,7,8};
    Term a = thvm_tensor(ctx, a_d, SHAPE(2, 2));
    Term b = thvm_tensor(ctx, b_d, SHAPE(2, 2));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);

    thvm_clear_tape(ctx);
    thvm_start_recording(ctx);
    Term yr = thvm_reduce(ctx, thvm_op(ctx, UOP_MM, a, b));
    thvm_stop_recording(ctx);

    f32 *fwd = thvm_to_host(ctx, yr);
    ASSERT_NEAR(fwd[0], 19.0f, 1e-3f, "mm[0,0]=19");
    ASSERT_NEAR(fwd[3], 50.0f, 1e-3f, "mm[1,1]=50");

    f32 *ga = thvm_to_host(ctx, thvm_grad(ctx, yr, a));
    ASSERT_NEAR(ga[0], 11.0f, 1e-3f, "grad_A[0,0]=11");
    ASSERT_NEAR(ga[1], 15.0f, 1e-3f, "grad_A[0,1]=15");

    thvm_free(ctx);
}

int main(void) {
    @autoreleasepool {
        printf("=== TinyHVM Metal Parity Test ===\n\n");

        test_matmul();
        test_relu();
        test_broadcast_add();
        test_full_forward();
        test_grad_x2();
        test_grad_mm();

        printf("\n=== Results: %d passed, %d failed ===\n",
               tests_passed, tests_failed);
        return tests_failed > 0 ? 1 : 0;
    }
}
