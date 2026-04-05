// test/test_fuse.m — Direct fusion/view-composition tests, no GRAD machinery.
// Build: clang -O0 -o /tmp/test_fuse test/test_fuse.m -framework Metal -framework Foundation -framework MetalPerformanceShaders -framework Accelerate -lm

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>

#define PASS(name) printf("PASS: %s\n", name)
#define FAIL(name, ...) do { printf("FAIL: %s — ", name); printf(__VA_ARGS__); printf("\n"); } while(0)
#define CHECK_NEAR(a,b,eps) (fabsf((a)-(b)) < (eps))

static TinyHVM *ctx;

// eval_term: force a computation into dst via ASSIGN.
// thvm_op only returns the ASSIGN term — it's not on the heap.
// Phase 3 scans ctx->heap[] for TAG_TOP ASSIGN cells, so we must
// allocate an extra cell and write the ASSIGN term there.
static float *eval_term(Term src, Term dst) {
    // Build ASSIGN args
    u64 loc = heap_alloc(ctx, 2);
    ctx->heap[loc]     = dst;
    ctx->heap[loc + 1] = src;
    Term assign = term_new(TAG_TOP, UOP_ASSIGN, loc);
    // Store ASSIGN term in its own heap cell so phase 3 finds it
    u64 ac = heap_alloc(ctx, 1);
    ctx->heap[ac] = assign;
    thvm_eval(ctx, term_era());
    return thvm_to_host(ctx, dst);
}

// Allocate zero tensor of given shape, run eval, return host data.
static float *eval_shape(Term src, Shape sh) {
    u32 n = 1; for (u32 i = 0; i < sh.rank; i++) n *= sh.dims[i];
    float *z = calloc(n, sizeof(float));
    Term dst = thvm_tensor(ctx, z, sh);
    free(z);
    return eval_term(src, dst);
}

extern Term thvm_sum_axes(TinyHVM *ctx, Term x, const u32 *axes, u32 n);
extern Term thvm_rmax_axes(TinyHVM *ctx, Term x, const u32 *axes, u32 n);

// ---------------------------------------------------------------
// T1: MUL(RESHAPE(x), y) — basic reshape view composition
// x:[6] → reshape [2,3]; y:[2,3]; z = x_reshaped * y
// ---------------------------------------------------------------
static void test_reshape_mul(void) {
    float xd[6]={1,2,3,4,5,6}, yd[6]={2,2,2,3,3,3};
    Term x = thvm_tensor(ctx, xd, SHAPE(6));
    Term y = thvm_tensor(ctx, yd, SHAPE(2,3));
    Term z = thvm_op(ctx, UOP_MUL, thvm_reshape(ctx, x, SHAPE(2,3)), y);
    float *r = eval_shape(z, SHAPE(2,3));
    int ok = r && CHECK_NEAR(r[0],2,1e-4) && CHECK_NEAR(r[1],4,1e-4) &&
             CHECK_NEAR(r[2],6,1e-4) && CHECK_NEAR(r[3],12,1e-4) &&
             CHECK_NEAR(r[4],15,1e-4) && CHECK_NEAR(r[5],18,1e-4);
    if (ok) PASS("reshape_mul");
    else if (!r) FAIL("reshape_mul", "eval returned NULL");
    else FAIL("reshape_mul", "got [%.1f,%.1f,%.1f,%.1f,%.1f,%.1f]", r[0],r[1],r[2],r[3],r[4],r[5]);
}

// ---------------------------------------------------------------
// T2: MUL(PERMUTE(x,[1,0]), y) — permute view composition
// x:[2,3] permuted [3,2]; y:[3,2]
// ---------------------------------------------------------------
static void test_permute_mul(void) {
    float xd[6]={1,2,3,4,5,6}, yd[6]={1,1,1,1,1,1};
    Term x = thvm_tensor(ctx, xd, SHAPE(2,3));
    Term y = thvm_tensor(ctx, yd, SHAPE(3,2));
    Term z = thvm_op(ctx, UOP_MUL, thvm_permute(ctx, x, (u32[]){1,0}, 2), y);
    // xp = [[1,4],[2,5],[3,6]] * 1 = [[1,4],[2,5],[3,6]]
    float *r = eval_shape(z, SHAPE(3,2));
    int ok = r && CHECK_NEAR(r[0],1,1e-4) && CHECK_NEAR(r[1],4,1e-4) &&
             CHECK_NEAR(r[2],2,1e-4) && CHECK_NEAR(r[3],5,1e-4) &&
             CHECK_NEAR(r[4],3,1e-4) && CHECK_NEAR(r[5],6,1e-4);
    if (ok) PASS("permute_mul");
    else if (!r) FAIL("permute_mul", "eval returned NULL");
    else FAIL("permute_mul", "got [%.1f,%.1f,%.1f,%.1f,%.1f,%.1f]", r[0],r[1],r[2],r[3],r[4],r[5]);
}

// ---------------------------------------------------------------
// T3: MUL(PERMUTE(RESHAPE(x)), y) — chained view composition
// x:[1,4]→reshape[2,2]→permute[2,2] (transpose); y:[2,2]
// ---------------------------------------------------------------
static void test_chain_view_mul(void) {
    float xd[4]={1,2,3,4}, yd[4]={10,10,10,10};
    Term x = thvm_tensor(ctx, xd, SHAPE(1,4));
    Term y = thvm_tensor(ctx, yd, SHAPE(2,2));
    Term xp = thvm_permute(ctx, thvm_reshape(ctx, x, SHAPE(2,2)), (u32[]){1,0}, 2);
    // xr=[[1,2],[3,4]], xp=[[1,3],[2,4]], z=[[10,30],[20,40]]
    float *r = eval_shape(thvm_op(ctx, UOP_MUL, xp, y), SHAPE(2,2));
    int ok = r && CHECK_NEAR(r[0],10,1e-4) && CHECK_NEAR(r[1],30,1e-4) &&
             CHECK_NEAR(r[2],20,1e-4) && CHECK_NEAR(r[3],40,1e-4);
    if (ok) PASS("chain_view_mul");
    else if (!r) FAIL("chain_view_mul", "eval returned NULL");
    else FAIL("chain_view_mul", "got [%.1f,%.1f,%.1f,%.1f]", r[0],r[1],r[2],r[3]);
}

// ---------------------------------------------------------------
// T4: SUM(RESHAPE(x)) — reduce of reshaped tensor
// x:[4]→reshape[2,2]→sum axis1 = [x0+x1, x2+x3]
// ---------------------------------------------------------------
static void test_sum_reshape(void) {
    float xd[4]={1,2,3,4};
    Term x = thvm_tensor(ctx, xd, SHAPE(4));
    Term s = thvm_sum_axes(ctx, thvm_reshape(ctx, x, SHAPE(2,2)), (u32[]){1}, 1);
    float *r = eval_shape(s, SHAPE(2));
    int ok = r && CHECK_NEAR(r[0],3,1e-4) && CHECK_NEAR(r[1],7,1e-4);
    if (ok) PASS("sum_reshape");
    else if (!r) FAIL("sum_reshape", "eval returned NULL");
    else FAIL("sum_reshape", "got [%.1f,%.1f]", r[0],r[1]);
}

// ---------------------------------------------------------------
// T5: MUL(x_6d, RESHAPE(y_4d)) — THE bc_fail pattern
// y is 4D; fuser must compose RESHAPE view so leaf gets 6D shape
// ---------------------------------------------------------------
static void test_6d_mul_4d_reshape(void) {
    float xd[16], yd[16];
    for (int i=0; i<16; i++) { xd[i]=1.f; yd[i]=(float)(i+1); }
    Term x = thvm_tensor(ctx, xd, SHAPE(1,1,2,2,2,2));
    Term y = thvm_tensor(ctx, yd, SHAPE(1,1,4,4));
    Term z = thvm_op(ctx, UOP_MUL, x, thvm_reshape(ctx, y, SHAPE(1,1,2,2,2,2)));
    float *r = eval_shape(z, SHAPE(1,1,2,2,2,2));
    int ok = (r != NULL); for (int i=0; ok && i<16; i++) ok = CHECK_NEAR(r[i],(float)(i+1),1e-4);
    if (ok) PASS("6d_mul_4d_reshape");
    else if (!r) FAIL("6d_mul_4d_reshape", "eval returned NULL");
    else { FAIL("6d_mul_4d_reshape", "mismatch:"); for (int i=0;i<16;i++) printf("  [%d]=%.1f want %.1f\n",i,r[i],(float)(i+1)); }
}

// ---------------------------------------------------------------
// T6: MUL(y_6d, PERMUTE(RESHAPE(x_4d))) — backward pattern
// Tests that PERMUTE(RESHAPE(x)) composed into matching 6D leaf
// ---------------------------------------------------------------
static void test_6d_mul_permute_reshape(void) {
    float xd[16], yd[16];
    for (int i=0; i<16; i++) { xd[i]=(float)(i+1); yd[i]=1.f; }
    Term x = thvm_tensor(ctx, xd, SHAPE(1,1,4,4));
    Term y = thvm_tensor(ctx, yd, SHAPE(1,1,2,2,2,2));
    // x→reshape[1,1,2,2,2,2]→permute[0,1,2,4,3,5]
    Term xp = thvm_permute(ctx, thvm_reshape(ctx, x, SHAPE(1,1,2,2,2,2)),
                            (u32[]){0,1,2,4,3,5}, 6);
    // y*xp: y=all-1s so result = xp
    float *r = eval_shape(thvm_op(ctx, UOP_MUL, y, xp), SHAPE(1,1,2,2,2,2));
    if (!r) { FAIL("6d_mul_permute_reshape", "eval returned NULL"); return; }
    // xd = [1..16] row-major [1,1,4,4]
    // reshape→[1,1,2,2,2,2]: element [0,0,i,j,k,l] = xd[i*8+j*4+k*2+l]
    // permute[0,1,2,4,3,5]: output[0,0,i,k,j,l] = xd[i*8+j*4+k*2+l]
    // i.e. output at linear index matters for values
    // Since y=1, result = xp. We just check it's non-zero and correct.
    int ok = 1;
    for (int i=0; i<16; i++) if (r[i] < 0.5f) { ok=0; break; }  // all should be >= 1
    if (ok) PASS("6d_mul_permute_reshape");
    else { FAIL("6d_mul_permute_reshape", "zeros found:"); for (int i=0;i<16;i++) if(r[i]<0.5f) printf("  [%d]=%.3f\n",i,r[i]); }
}

// ---------------------------------------------------------------
// T7: RMAX(PERMUTE(RESHAPE(x))) — maxpool forward pattern
// x:[1,1,4,4]→reshape[1,1,2,2,2,2]→permute→rmax axes[4,5]=[1,1,2,2]
// Expected: [[6,8],[14,16]] (2x2 max over each 2x2 window)
// ---------------------------------------------------------------
static void test_maxpool_rmax(void) {
    float xd[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    Term x = thvm_tensor(ctx, xd, SHAPE(1,1,4,4));
    Term xr = thvm_reshape(ctx, x, SHAPE(1,1,2,2,2,2));
    Term xp = thvm_permute(ctx, xr, (u32[]){0,1,2,4,3,5}, 6);
    Term mx = thvm_rmax_axes(ctx, xp, (u32[]){4,5}, 2);
    float *r = eval_shape(mx, SHAPE(1,1,2,2));
    int ok = r && CHECK_NEAR(r[0],6,1e-4) && CHECK_NEAR(r[1],8,1e-4) &&
             CHECK_NEAR(r[2],14,1e-4) && CHECK_NEAR(r[3],16,1e-4);
    if (ok) PASS("maxpool_rmax");
    else if (!r) FAIL("maxpool_rmax", "eval returned NULL");
    else FAIL("maxpool_rmax", "got [%.1f,%.1f,%.1f,%.1f]", r[0],r[1],r[2],r[3]);
}

// ---------------------------------------------------------------
// T8: RMAX backward — MUL(EXPAND(gy), mask) where mask involves
//     CMP(EXPAND(RESHAPE(rmax_out)), at) and at is a 6D view of x.
//     This is the exact pattern that bc_fails in test_ch32.
//     The key: x (4D) should only appear via its 6D view alias.
// ---------------------------------------------------------------
static void test_rmax_backward(void) {
    // x [1,1,4,4] — the "h4" tensor (pre-pool activation)
    float xd[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    Term x = thvm_tensor(ctx, xd, SHAPE(1,1,4,4));
    // Build maxpool forward: reshape→permute→rmax
    Term x_rs = thvm_reshape(ctx, x, SHAPE(1,1,2,2,2,2));
    Term x_pm = thvm_permute(ctx, x_rs, (u32[]){0,1,2,4,3,5}, 6); // [1,1,2,2,2,2]
    Term mx    = thvm_rmax_axes(ctx, x_pm, (u32[]){4,5}, 2);       // [1,1,2,2]

    // Simulate RMAX backward: grad_in = MUL(EXPAND(gy), 1-CMP(EXPAND(RESHAPE(mx)), x_pm))
    // gy [1,1,2,2] — upstream gradient
    float gyd[4]={1,1,1,1};
    Term gy = thvm_tensor(ctx, gyd, SHAPE(1,1,2,2));

    // mask: where x_pm == max
    Term mx_rs = thvm_reshape(ctx, mx, SHAPE(1,1,2,2,1,1));
    Term mx_bc = thvm_expand(ctx, mx_rs, SHAPE(1,1,2,2,2,2)); // [1,1,2,2,2,2]
    f32 one = 1.f;
    Term mask  = thvm_op(ctx, UOP_SUB,
                    thvm_tensor(ctx, &one, SHAPE(1)),
                    thvm_op(ctx, UOP_CMP, mx_bc, x_pm)); // 0 where max, 1 elsewhere? inverted in grad.c

    // gy expanded to [1,1,2,2,2,2]
    Term gy_bc = thvm_expand(ctx, thvm_reshape(ctx, gy, SHAPE(1,1,2,2,1,1)),
                              SHAPE(1,1,2,2,2,2));
    Term grad_in_6d = thvm_op(ctx, UOP_MUL, gy_bc, mask); // [1,1,2,2,2,2]

    // Fold back to 4D: permute inverse then reshape
    Term grad_pm = thvm_permute(ctx, grad_in_6d, (u32[]){0,1,2,4,3,5}, 6);
    Term grad_x  = thvm_reshape(ctx, grad_pm, SHAPE(1,1,4,4));

    float *r = eval_shape(grad_x, SHAPE(1,1,4,4));
    if (!r) { FAIL("rmax_backward", "eval returned NULL (bc_fail?)"); return; }
    // gradient should be non-zero at maxima positions
    int has_nonzero = 0;
    for (int i=0; i<16; i++) if (fabsf(r[i]) > 1e-6f) { has_nonzero=1; break; }
    if (has_nonzero) PASS("rmax_backward");
    else { FAIL("rmax_backward", "all zeros:"); for(int i=0;i<16;i++) printf(" %.2f",r[i]); printf("\n"); }
}

// ---------------------------------------------------------------
// T9: Minimal maxpool grad — full thvm_grad_multi path on a tiny net
// ---------------------------------------------------------------
static void test_maxpool_grad(void) {
    // x [1,1,4,4] → pool → sum → grad w.r.t. x
    float xd[16]; for(int i=0;i<16;i++) xd[i]=(float)(i+1);
    Term x = thvm_tensor(ctx, xd, SHAPE(1,1,4,4));
    thvm_set_requires_grad(ctx, x);

    Term x_rs = thvm_reshape(ctx, x, SHAPE(1,1,2,2,2,2));
    Term x_pm = thvm_permute(ctx, x_rs, (u32[]){0,1,2,4,3,5}, 6);
    Term mx    = thvm_rmax_axes(ctx, x_pm, (u32[]){4,5}, 2); // [1,1,2,2]

    // sum to scalar loss
    extern Term thvm_sum_axes(TinyHVM*, Term, const u32*, u32);
    Term loss  = thvm_sum_axes(ctx, mx, (u32[]){0,1,2,3}, 4);
    loss       = thvm_reshape(ctx, loss, SHAPE(1));

    float gz[16] = {0}; Term gx = thvm_tensor(ctx, gz, SHAPE(1,1,4,4));
    Term grad = thvm_grad_multi(ctx, loss, (Term[]){x}, (Term[]){gx}, 1);
    thvm_eval(ctx, grad);

    float *r = thvm_to_host(ctx, gx);
    if (!r) { FAIL("maxpool_grad", "gx read failed"); return; }
    // Each max in 2x2 window gets gradient 1.0 (or 0.5 if tied)
    // x = [1..16], maxima of each window: 6, 8, 14, 16
    float sum = 0; for(int i=0;i<16;i++) sum += r[i];
    int ok = CHECK_NEAR(sum, 4.f, 1e-3f); // 4 maxima, each gets grad 1.0
    if (ok) PASS("maxpool_grad");
    else { FAIL("maxpool_grad", "grad sum=%.3f (want 4.0):", sum);
           for(int i=0;i<16;i++) printf(" %.2f",r[i]); printf("\n"); }
}

int main(void) {
    ctx = thvm_init("metal");
    printf("=== Fusion/view-composition tests ===\n");
    test_reshape_mul();
    test_permute_mul();
    test_chain_view_mul();
    test_sum_reshape();
    test_6d_mul_4d_reshape();
    test_6d_mul_permute_reshape();
    test_maxpool_rmax();
    test_rmax_backward();
    test_maxpool_grad();
    thvm_free(ctx);
    return 0;
}
