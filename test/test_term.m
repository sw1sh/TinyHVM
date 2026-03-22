// test_term.m — TinyHVM test suite
// Change DEVICE to "metal" to run all tests on GPU.

#include "../src/tinyhvm.c"
#include "../src/gpu_cpu.c"
#include "../src/gpu_metal.m"

#define DEVICE "cpu"

#include <stdio.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s (line %d)\n", msg, __LINE__); tests_failed++; } \
    else { tests_passed++; } \
} while(0)

#define ASSERT_EQ(a, b, msg) ASSERT((a) == (b), msg)
#define ASSERT_NEAR(a, b, eps, msg) ASSERT(fabsf((a) - (b)) < (eps), msg)

// ---- Term packing ----

static void test_term_packing(void) {
    printf("test_term_packing:\n");
    Term t = term_new(TAG_NUM, 42, 12345ULL);
    ASSERT_EQ(term_tag(t), TAG_NUM, "tag round-trip");
    ASSERT_EQ(term_ext(t), 42, "ext round-trip");
    ASSERT_EQ(term_val(t), 12345ULL, "val round-trip");

    Term t2 = term_new(TAG_MASK, EXT_MASK, VAL_MASK);
    ASSERT_EQ(term_tag(t2), TAG_MASK, "max tag");
    ASSERT_EQ(term_ext(t2), EXT_MASK, "max ext");
    ASSERT_EQ(term_val(t2), VAL_MASK, "max val");

    Term t3 = term_new(0, 0, 0);
    ASSERT_EQ(term_tag(t3), 0, "zero tag");

    Term t4 = term_set_sub(t);
    ASSERT(term_is_sub(t4), "sub bit set");
    ASSERT(!term_is_sub(t), "sub bit clear");
    ASSERT_EQ(term_tag(t4), TAG_NUM, "sub preserves tag");
}

static void test_num_encoding(void) {
    printf("test_num_encoding:\n");
    Term u = term_num_u32(42);
    ASSERT_EQ(term_as_u32(u), 42, "u32 value");
    Term f = term_num_f32(3.14f);
    ASSERT_NEAR(term_as_f32(f), 3.14f, 1e-6f, "f32 value");
    Term fn = term_num_f32(-0.5f);
    ASSERT_NEAR(term_as_f32(fn), -0.5f, 1e-6f, "f32 neg");
}

// ---- View ----

static void test_view_create(void) {
    printf("test_view_create:\n");
    u32 shape[] = {2, 3, 4};
    View v = view_create(shape_of(shape, 3));
    ASSERT_EQ(v.shape.rank, 3, "ndim");
    ASSERT_EQ(v.shape.dims[0], 2, "shape[0]");
    ASSERT_EQ(v.shape.dims[1], 3, "shape[1]");
    ASSERT_EQ(v.shape.dims[2], 4, "shape[2]");
    ASSERT_EQ(v.strides[0], 12, "stride[0] = 3*4");
    ASSERT_EQ(v.strides[1], 4, "stride[1] = 4");
    ASSERT_EQ(v.strides[2], 1, "stride[2] = 1");
    ASSERT_EQ(v.numel, 24, "numel = 2*3*4");
    ASSERT_EQ(v.contiguous, 1, "contiguous");
}

static void test_view_broadcast(void) {
    printf("test_view_broadcast:\n");
    u32 sa[] = {2, 3}, sb[] = {1, 3};
    View a = view_create(shape_of(sa, 2));
    View b = view_create(shape_of(sb, 2));
    View oa, ob;
    u32 out_shape[MAX_DIM], out_ndim;
    int ok = view_broadcast(&a, &b, &oa, &ob, out_shape, &out_ndim);
    ASSERT(ok, "broadcast succeeds");
    ASSERT_EQ(out_ndim, 2, "broadcast ndim");
    ASSERT_EQ(out_shape[0], 2, "broadcast shape[0]");
    ASSERT_EQ(out_shape[1], 3, "broadcast shape[1]");
    ASSERT_EQ(ob.strides[0], 0, "broadcast stride=0 for dim 0 of b");
    ASSERT_EQ(ob.strides[1], 1, "broadcast stride=1 for dim 1 of b");
}

static void test_view_permute(void) {
    printf("test_view_permute:\n");
    u32 shape[] = {2, 3};
    View v = view_create(shape_of(shape, 2));
    u32 axes[] = {1, 0};
    View p = view_permute(v, axes);
    ASSERT_EQ(p.shape.dims[0], 3, "permuted shape[0]");
    ASSERT_EQ(p.shape.dims[1], 2, "permuted shape[1]");
    ASSERT_EQ(p.strides[0], 1, "permuted stride[0]");
    ASSERT_EQ(p.strides[1], 3, "permuted stride[1]");
    ASSERT_EQ(p.contiguous, 0, "not contiguous");
}

// ---- Reduction ----

static void test_op2_reduce(void) {
    printf("test_op2_reduce:\n");
    TinyHVM *ctx = thvm_init(NULL);
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc, term_num_u32(3));
    heap_set(ctx, loc + 1, term_num_u32(5));
    Term result = thvm_reduce(ctx, term_new(TAG_OP2, 0, loc));
    ASSERT_EQ(term_as_u32(result), 8, "3 + 5 = 8");
    thvm_free(ctx);
}

// ---- Tensor ops ----

static void test_matmul_identity(void) {
    printf("test_matmul_identity:\n");
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 x_d[] = {1,2,3,4}; u32 s[] = {2,2};
    f32 i_d[] = {1,0,0,1};
    Term x = thvm_tensor(ctx, x_d, shape_of(s, 2));
    Term w = thvm_tensor(ctx, i_d, shape_of(s, 2));
    Term r = thvm_reduce(ctx, thvm_op(ctx, UOP_MM, x, w));
    f32 *out = thvm_to_host(ctx, r);
    ASSERT(out != NULL, "mm returns data");
    ASSERT_NEAR(out[0], 1, 1e-5f, "mm I [0,0]");
    ASSERT_NEAR(out[1], 2, 1e-5f, "mm I [0,1]");
    ASSERT_NEAR(out[2], 3, 1e-5f, "mm I [1,0]");
    ASSERT_NEAR(out[3], 4, 1e-5f, "mm I [1,1]");
    thvm_free(ctx);
}

static void test_relu(void) {
    printf("test_relu:\n");
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    f32 d[] = {-1, 2, -3, 4}; u32 s[] = {1, 4};
    Term t = thvm_tensor(ctx, d, shape_of(s, 2));
    Term r = thvm_reduce(ctx, thvm_op(ctx, UOP_RELU, t, term_era()));
    f32 *out = thvm_to_host(ctx, r);
    ASSERT_NEAR(out[0], 0, 1e-5f, "relu(-1)=0");
    ASSERT_NEAR(out[1], 2, 1e-5f, "relu(2)=2");
    ASSERT_NEAR(out[2], 0, 1e-5f, "relu(-3)=0");
    ASSERT_NEAR(out[3], 4, 1e-5f, "relu(4)=4");
    thvm_free(ctx);
}

// ---- Broadcasting ----

static void test_broadcast_add(void) {
    printf("test_broadcast_add:\n");
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    // a = [[1,2,3],[4,5,6]] (2x3)
    f32 a_d[] = {1,2,3,4,5,6}; u32 sa[] = {2, 3};
    Term a = thvm_tensor(ctx, a_d, shape_of(sa, 2));

    // b = [10, 20, 30] (1x3) — broadcast to (2x3)
    f32 b_d[] = {10, 20, 30}; u32 sb[] = {1, 3};
    Term b = thvm_tensor(ctx, b_d, shape_of(sb, 2));

    Term r = thvm_reduce(ctx, thvm_op(ctx, UOP_ADD, a, b));
    f32 *out = thvm_to_host(ctx, r);
    ASSERT(out != NULL, "broadcast add returns data");
    // Expected: [[11,22,33],[14,25,36]]
    ASSERT_NEAR(out[0], 11, 1e-5f, "bc_add[0,0]");
    ASSERT_NEAR(out[1], 22, 1e-5f, "bc_add[0,1]");
    ASSERT_NEAR(out[2], 33, 1e-5f, "bc_add[0,2]");
    ASSERT_NEAR(out[3], 14, 1e-5f, "bc_add[1,0]");
    ASSERT_NEAR(out[4], 25, 1e-5f, "bc_add[1,1]");
    ASSERT_NEAR(out[5], 36, 1e-5f, "bc_add[1,2]");

    thvm_free(ctx);
}

static void test_broadcast_column(void) {
    printf("test_broadcast_column:\n");
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    // a = [[1,2],[3,4]] (2x2)
    f32 a_d[] = {1,2,3,4}; u32 sa[] = {2, 2};
    Term a = thvm_tensor(ctx, a_d, shape_of(sa, 2));

    // b = [[10],[20]] (2x1) — broadcast to (2x2)
    f32 b_d[] = {10, 20}; u32 sb[] = {2, 1};
    Term b = thvm_tensor(ctx, b_d, shape_of(sb, 2));

    Term r = thvm_reduce(ctx, thvm_op(ctx, UOP_ADD, a, b));
    f32 *out = thvm_to_host(ctx, r);
    ASSERT(out != NULL, "column broadcast returns data");
    // Expected: [[11,12],[23,24]]
    ASSERT_NEAR(out[0], 11, 1e-5f, "col_bc[0,0]");
    ASSERT_NEAR(out[1], 12, 1e-5f, "col_bc[0,1]");
    ASSERT_NEAR(out[2], 23, 1e-5f, "col_bc[1,0]");
    ASSERT_NEAR(out[3], 24, 1e-5f, "col_bc[1,1]");

    thvm_free(ctx);
}

// ---- Full forward ----

static void test_full_forward(void) {
    printf("test_full_forward (relu(mm(x,w)+b) with broadcast):\n");
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    f32 x_d[] = {1,2,3,4,5,6}; u32 xs[] = {2,3};
    f32 w_d[] = {0.1f,-0.2f, 0.3f,0.4f, -0.5f,0.6f}; u32 ws[] = {3,2};
    f32 b_d[] = {-0.1f, 0.2f}; u32 bs[] = {1,2};  // broadcast!

    Term x = thvm_tensor(ctx, x_d, shape_of(xs, 2));
    Term w = thvm_tensor(ctx, w_d, shape_of(ws, 2));
    Term b = thvm_tensor(ctx, b_d, shape_of(bs, 2));

    Term z = thvm_op(ctx, UOP_RELU,
                thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, x, w), b),
                term_era());

    Term result = thvm_reduce(ctx, z);
    ASSERT_EQ(term_tag(result), TAG_TEN, "forward produces tensor");

    f32 *out = thvm_to_host(ctx, result);
    ASSERT(out != NULL, "forward returns data");
    ASSERT_NEAR(out[0], 0.0f,  1e-4f, "z[0,0]=0 (relu clips)");
    ASSERT_NEAR(out[1], 2.6f,  1e-4f, "z[0,1]=2.6");
    ASSERT_NEAR(out[2], 0.0f,  1e-4f, "z[1,0]=0 (relu clips)");
    ASSERT_NEAR(out[3], 5.0f,  1e-4f, "z[1,1]=5.0");

    printf("  interactions: %llu\n", (unsigned long long)ctx->itrs);
    thvm_free(ctx);
}

// ---- Autograd (graph-level only) ----

static void test_grad_x2(void) {
    printf("test_grad_x2 (d(x^2)/dx = 2x at x=3):\n");
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    f32 x_d[] = {3.0f}; u32 xs[] = {1, 1};
    Term x = thvm_tensor(ctx, x_d, shape_of(xs, 2));
    thvm_set_requires_grad(ctx, x);

    thvm_start_recording(ctx);
    Term yr = thvm_reduce(ctx, thvm_op(ctx, UOP_MUL, x, x));
    thvm_stop_recording(ctx);

    f32 *fwd = thvm_to_host(ctx, yr);
    ASSERT_NEAR(fwd[0], 9.0f, 1e-5f, "3*3 = 9");

    Term dy = thvm_grad(ctx, yr, x);
    f32 *g = thvm_to_host(ctx, dy);
    ASSERT(g != NULL, "grad exists");
    ASSERT_NEAR(g[0], 6.0f, 1e-5f, "d(x^2)/dx = 2x = 6");

    thvm_free(ctx);
}

static void test_grad_add(void) {
    printf("test_grad_add (d(a+b)/da = 1):\n");
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    f32 a_d[] = {2.0f}; u32 s[] = {1, 1};
    f32 b_d[] = {5.0f};
    Term a = thvm_tensor(ctx, a_d, shape_of(s, 2));
    Term b = thvm_tensor(ctx, b_d, shape_of(s, 2));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);

    thvm_start_recording(ctx);
    Term yr = thvm_reduce(ctx, thvm_op(ctx, UOP_ADD, a, b));
    thvm_stop_recording(ctx);

    f32 *fwd = thvm_to_host(ctx, yr);
    ASSERT_NEAR(fwd[0], 7.0f, 1e-5f, "2+5 = 7");

    f32 *ga = thvm_to_host(ctx, thvm_grad(ctx, yr, a));
    f32 *gb = thvm_to_host(ctx, thvm_grad(ctx, yr, b));
    ASSERT_NEAR(ga[0], 1.0f, 1e-5f, "d(a+b)/da = 1");
    ASSERT_NEAR(gb[0], 1.0f, 1e-5f, "d(a+b)/db = 1");

    thvm_free(ctx);
}

static void test_grad_relu(void) {
    printf("test_grad_relu:\n");
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    f32 x_d[] = {-2.0f, 3.0f, -1.0f, 4.0f}; u32 xs[] = {1, 4};
    Term x = thvm_tensor(ctx, x_d, shape_of(xs, 2));
    thvm_set_requires_grad(ctx, x);

    thvm_start_recording(ctx);
    Term yr = thvm_reduce(ctx, thvm_op(ctx, UOP_RELU, x, term_era()));
    thvm_stop_recording(ctx);

    f32 *fwd = thvm_to_host(ctx, yr);
    ASSERT_NEAR(fwd[0], 0.0f, 1e-5f, "relu(-2)=0");
    ASSERT_NEAR(fwd[1], 3.0f, 1e-5f, "relu(3)=3");

    f32 *g = thvm_to_host(ctx, thvm_grad(ctx, yr, x));
    ASSERT(g != NULL, "relu grad exists");
    ASSERT_NEAR(g[0], 0.0f, 1e-5f, "relu grad(-2)=0");
    ASSERT_NEAR(g[1], 1.0f, 1e-5f, "relu grad(3)=1");
    ASSERT_NEAR(g[2], 0.0f, 1e-5f, "relu grad(-1)=0");
    ASSERT_NEAR(g[3], 1.0f, 1e-5f, "relu grad(4)=1");

    thvm_free(ctx);
}

static void test_grad_mm(void) {
    printf("test_grad_mm (matmul gradient):\n");
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    f32 a_d[] = {1,2,3,4}; u32 s[] = {2, 2};
    f32 b_d[] = {5,6,7,8};
    Term a = thvm_tensor(ctx, a_d, shape_of(s, 2));
    Term b = thvm_tensor(ctx, b_d, shape_of(s, 2));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);

    thvm_start_recording(ctx);
    Term yr = thvm_reduce(ctx, thvm_op(ctx, UOP_MM, a, b));
    thvm_stop_recording(ctx);

    f32 *fwd = thvm_to_host(ctx, yr);
    ASSERT_NEAR(fwd[0], 19.0f, 1e-4f, "mm[0,0]=19");
    ASSERT_NEAR(fwd[1], 22.0f, 1e-4f, "mm[0,1]=22");
    ASSERT_NEAR(fwd[2], 43.0f, 1e-4f, "mm[1,0]=43");
    ASSERT_NEAR(fwd[3], 50.0f, 1e-4f, "mm[1,1]=50");

    // grad_A = ones . B^T = [[11,15],[11,15]]
    f32 *ga = thvm_to_host(ctx, thvm_grad(ctx, yr, a));
    ASSERT(ga != NULL, "mm grad_a exists");
    ASSERT_NEAR(ga[0], 11.0f, 1e-4f, "grad_A[0,0]=11");
    ASSERT_NEAR(ga[1], 15.0f, 1e-4f, "grad_A[0,1]=15");
    ASSERT_NEAR(ga[2], 11.0f, 1e-4f, "grad_A[1,0]=11");
    ASSERT_NEAR(ga[3], 15.0f, 1e-4f, "grad_A[1,1]=15");

    // grad_B = A^T . ones = [[4,4],[6,6]]
    f32 *gb = thvm_to_host(ctx, thvm_grad(ctx, yr, b));
    ASSERT(gb != NULL, "mm grad_b exists");
    ASSERT_NEAR(gb[0], 4.0f, 1e-4f, "grad_B[0,0]=4");
    ASSERT_NEAR(gb[1], 4.0f, 1e-4f, "grad_B[0,1]=4");
    ASSERT_NEAR(gb[2], 6.0f, 1e-4f, "grad_B[1,0]=6");
    ASSERT_NEAR(gb[3], 6.0f, 1e-4f, "grad_B[1,1]=6");

    thvm_free(ctx);
}

static void test_grad_of_grad(void) {
    printf("test_grad_of_grad (d²(x³)/dx² = 6x at x=2 => 12):\n");
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    f32 x_d[] = {2.0f}; u32 xs[] = {1, 1};
    Term x = thvm_tensor(ctx, x_d, shape_of(xs, 2));
    thvm_set_requires_grad(ctx, x);

    // Forward: x^3 — tape accumulates
    thvm_start_recording(ctx);
    Term t1 = thvm_op(ctx, UOP_MUL, x, x);
    Term y  = thvm_op(ctx, UOP_MUL, t1, x);
    Term yr = thvm_reduce(ctx, y);
    // DON'T stop recording — tape accumulates into grad reduction

    f32 *fwd = thvm_to_host(ctx, yr);
    ASSERT_NEAR(fwd[0], 8.0f, 1e-4f, "x^3 = 8");

    // First grad: dy/dx (lazy) — reduce it while recording
    Term dy_dx = thvm_grad(ctx, yr, x);
    Term dy_val = thvm_reduce(ctx, dy_dx);  // grad ops get taped
    thvm_stop_recording(ctx);

    f32 *g1 = thvm_to_host(ctx, dy_val);
    ASSERT(g1 != NULL, "first grad exists");
    ASSERT_NEAR(g1[0], 12.0f, 1e-4f, "d(x^3)/dx = 3x^2 = 12");

    // Second grad: d²y/dx² — walks FULL tape (forward + grad ops)
    Term d2y = thvm_grad(ctx, dy_val, x);
    f32 *g2 = thvm_to_host(ctx, d2y);
    ASSERT(g2 != NULL, "second grad exists");
    ASSERT_NEAR(g2[0], 12.0f, 1e-4f, "d²(x³)/dx² = 6x = 12");

    thvm_free(ctx);
}

int main(void) {
    printf("=== TinyHVM Test Suite v3 ===\n\n");

    test_term_packing();
    test_num_encoding();
    test_view_create();
    test_view_broadcast();
    test_view_permute();
    test_op2_reduce();
    test_matmul_identity();
    test_relu();
    test_broadcast_add();
    test_broadcast_column();
    test_full_forward();
    test_grad_x2();
    test_grad_add();
    test_grad_relu();
    test_grad_mm();
    test_grad_of_grad();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
