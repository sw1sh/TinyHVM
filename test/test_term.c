// test_term.c — Unit tests for TinyHVM term packing and basic ops

#include "../src/tinyhvm.c"
#include "../src/gpu_cpu.c"

#include <stdio.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_EQ(a, b, msg) ASSERT((a) == (b), msg)
#define ASSERT_NEAR(a, b, eps, msg) ASSERT(fabsf((a) - (b)) < (eps), msg)

static void test_term_packing(void) {
    printf("test_term_packing:\n");

    // Basic round-trip
    Term t = term_new(TAG_NUM, 42, 12345ULL);
    ASSERT_EQ(term_tag(t), TAG_NUM, "tag round-trip");
    ASSERT_EQ(term_ext(t), 42, "ext round-trip");
    ASSERT_EQ(term_val(t), 12345ULL, "val round-trip");

    // Max values
    Term t2 = term_new(TAG_MASK, EXT_MASK, VAL_MASK);
    ASSERT_EQ(term_tag(t2), TAG_MASK, "max tag");
    ASSERT_EQ(term_ext(t2), EXT_MASK, "max ext");
    ASSERT_EQ(term_val(t2), VAL_MASK, "max val");

    // Zero
    Term t3 = term_new(0, 0, 0);
    ASSERT_EQ(term_tag(t3), 0, "zero tag");
    ASSERT_EQ(term_ext(t3), 0, "zero ext");
    ASSERT_EQ(term_val(t3), 0ULL, "zero val");

    // SUB bit
    Term t4 = term_set_sub(t);
    ASSERT(term_is_sub(t4), "sub bit set");
    ASSERT(!term_is_sub(t), "sub bit not set");
    ASSERT_EQ(term_tag(t4), TAG_NUM, "sub preserves tag");
}

static void test_num_encoding(void) {
    printf("test_num_encoding:\n");

    // u32
    Term u = term_num_u32(42);
    ASSERT_EQ(term_tag(u), TAG_NUM, "u32 tag");
    ASSERT_EQ(term_ext(u), NUM_U32, "u32 ext");
    ASSERT_EQ(term_as_u32(u), 42, "u32 value");

    // f32
    Term f = term_num_f32(3.14f);
    ASSERT_EQ(term_tag(f), TAG_NUM, "f32 tag");
    ASSERT_EQ(term_ext(f), NUM_F32, "f32 ext");
    ASSERT_NEAR(term_as_f32(f), 3.14f, 1e-6f, "f32 value");

    // f32 negative
    Term fn = term_num_f32(-0.5f);
    ASSERT_NEAR(term_as_f32(fn), -0.5f, 1e-6f, "f32 negative");

    // f32 zero
    Term fz = term_num_f32(0.0f);
    ASSERT_NEAR(term_as_f32(fz), 0.0f, 1e-6f, "f32 zero");
}

static void test_tensor_constructors(void) {
    printf("test_tensor_constructors:\n");

    Term ten = term_ten(7, DTYPE_F32);
    ASSERT_EQ(term_tag(ten), TAG_TEN, "ten tag");
    ASSERT_EQ(term_val(ten), 7ULL, "ten buf_id");
    ASSERT_EQ(term_ext(ten), DTYPE_F32, "ten dtype");

    Term top = term_top(UOP_MM, 100);
    ASSERT_EQ(term_tag(top), TAG_TOP, "top tag");
    ASSERT_EQ(term_ext(top), UOP_MM, "top uop");
    ASSERT_EQ(term_val(top), 100ULL, "top loc");
}

static void test_heap_basic(void) {
    printf("test_heap_basic:\n");

    TinyHVM *ctx = thvm_init(NULL);

    u64 loc = heap_alloc(ctx, 2);
    ASSERT(loc > 0, "alloc returns non-zero");

    heap_set(ctx, loc, term_num_u32(42));
    heap_set(ctx, loc + 1, term_num_f32(3.14f));

    ASSERT_EQ(term_as_u32(heap_read(ctx, loc)), 42, "heap read u32");
    ASSERT_NEAR(term_as_f32(heap_read(ctx, loc + 1)), 3.14f, 1e-6f, "heap read f32");

    thvm_free(ctx);
}

static void test_op2_reduce(void) {
    printf("test_op2_reduce:\n");

    TinyHVM *ctx = thvm_init(NULL);

    // 3 + 5 = 8
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc, term_num_u32(3));
    heap_set(ctx, loc + 1, term_num_u32(5));
    Term op = term_new(TAG_OP2, 0, loc);  // opr=0 (add)

    Term result = thvm_reduce(ctx, op);
    ASSERT_EQ(term_tag(result), TAG_NUM, "op2 produces num");
    ASSERT_EQ(term_as_u32(result), 8, "3 + 5 = 8");
    ASSERT_EQ(ctx->itrs, 1ULL, "one interaction");

    // 7 * 6 = 42
    u64 loc2 = heap_alloc(ctx, 2);
    heap_set(ctx, loc2, term_num_u32(7));
    heap_set(ctx, loc2 + 1, term_num_u32(6));
    Term op2 = term_new(TAG_OP2, 2, loc2);  // opr=2 (mul)

    Term result2 = thvm_reduce(ctx, op2);
    ASSERT_EQ(term_as_u32(result2), 42, "7 * 6 = 42");

    thvm_free(ctx);
}

static void test_tensor_forward(void) {
    printf("test_tensor_forward:\n");

    TinyHVM *ctx = thvm_init(&gpu_cpu_backend);

    // x = [[1,2],[3,4]] (2x2), w = [[1,0],[0,1]] (2x2 identity)
    f32 x_data[] = {1, 2, 3, 4};
    f32 w_data[] = {1, 0, 0, 1};
    u32 shape[] = {2, 2};

    Term x = thvm_tensor(ctx, x_data, shape, 2);
    Term w = thvm_tensor(ctx, w_data, shape, 2);

    // matmul(x, I) should = x
    Term xw = thvm_op(ctx, UOP_MM, x, w);
    Term result = thvm_reduce(ctx, xw);
    ASSERT_EQ(term_tag(result), TAG_TEN, "mm produces tensor");

    f32 *out = thvm_to_host(ctx, result);
    ASSERT(out != NULL, "to_host returns data");
    ASSERT_NEAR(out[0], 1.0f, 1e-5f, "mm identity [0,0]");
    ASSERT_NEAR(out[1], 2.0f, 1e-5f, "mm identity [0,1]");
    ASSERT_NEAR(out[2], 3.0f, 1e-5f, "mm identity [1,0]");
    ASSERT_NEAR(out[3], 4.0f, 1e-5f, "mm identity [1,1]");

    thvm_free(ctx);
}

static void test_relu_chain(void) {
    printf("test_relu_chain:\n");

    TinyHVM *ctx = thvm_init(&gpu_cpu_backend);

    // relu([-1, 2, -3, 4]) = [0, 2, 0, 4]
    f32 data[] = {-1, 2, -3, 4};
    u32 shape[] = {1, 4};
    Term t = thvm_tensor(ctx, data, shape, 2);

    Term r = thvm_op(ctx, UOP_RELU, t, term_era());
    Term result = thvm_reduce(ctx, r);

    f32 *out = thvm_to_host(ctx, result);
    ASSERT(out != NULL, "relu returns data");
    ASSERT_NEAR(out[0], 0.0f, 1e-5f, "relu(-1)=0");
    ASSERT_NEAR(out[1], 2.0f, 1e-5f, "relu(2)=2");
    ASSERT_NEAR(out[2], 0.0f, 1e-5f, "relu(-3)=0");
    ASSERT_NEAR(out[3], 4.0f, 1e-5f, "relu(4)=4");

    thvm_free(ctx);
}

static void test_full_forward(void) {
    printf("test_full_forward (relu(mm(x,w)+b)):\n");

    TinyHVM *ctx = thvm_init(&gpu_cpu_backend);

    // x [2x3]
    f32 x_data[] = {1, 2, 3, 4, 5, 6};
    u32 x_shape[] = {2, 3};
    Term x = thvm_tensor(ctx, x_data, x_shape, 2);

    // w [3x2]
    f32 w_data[] = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f};
    u32 w_shape[] = {3, 2};
    Term w = thvm_tensor(ctx, w_data, w_shape, 2);

    // b [2x2] (pre-broadcast)
    f32 b_data[] = {-0.1f, 0.2f, -0.1f, 0.2f};
    u32 b_shape[] = {2, 2};
    Term b = thvm_tensor(ctx, b_data, b_shape, 2);

    // z = relu(matmul(x, w) + b)
    Term xw  = thvm_op(ctx, UOP_MM, x, w);
    Term xwb = thvm_op(ctx, UOP_ADD, xw, b);
    Term z   = thvm_op(ctx, UOP_RELU, xwb, term_era());

    Term result = thvm_reduce(ctx, z);
    ASSERT_EQ(term_tag(result), TAG_TEN, "full forward produces tensor");

    f32 *out = thvm_to_host(ctx, result);
    ASSERT(out != NULL, "full forward returns data");

    // Expected (computed manually):
    // x@w = [[1*0.1+2*0.3+3*(-0.5), 1*(-0.2)+2*0.4+3*0.6],
    //        [4*0.1+5*0.3+6*(-0.5), 4*(-0.2)+5*0.4+6*0.6]]
    //     = [[0.1+0.6-1.5, -0.2+0.8+1.8],
    //        [0.4+1.5-3.0, -0.8+2.0+3.6]]
    //     = [[-0.8, 2.4],
    //        [-1.1, 4.8]]
    // +b  = [[-0.9, 2.6],
    //        [-1.2, 5.0]]
    // relu= [[0.0, 2.6],
    //        [0.0, 5.0]]
    ASSERT_NEAR(out[0], 0.0f,  1e-4f, "z[0,0] = 0.0");
    ASSERT_NEAR(out[1], 2.6f,  1e-4f, "z[0,1] = 2.6");
    ASSERT_NEAR(out[2], 0.0f,  1e-4f, "z[1,0] = 0.0");
    ASSERT_NEAR(out[3], 5.0f,  1e-4f, "z[1,1] = 5.0");

    printf("  interactions: %llu\n", (unsigned long long)ctx->itrs);

    thvm_free(ctx);
}

int main(void) {
    printf("=== TinyHVM Test Suite ===\n\n");

    test_term_packing();
    test_num_encoding();
    test_tensor_constructors();
    test_heap_basic();
    test_op2_reduce();
    test_tensor_forward();
    test_relu_chain();
    test_full_forward();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
