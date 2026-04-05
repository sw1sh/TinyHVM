#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>
static void check(const char *dev) {
    srand(42);
    TinyHVM *ctx = thvm_init(dev);
    // Simple 3D broadcast MUL + SUM
    f32 d1[] = {1,2,3,4,5,6}; // [2,3]
    f32 d2[] = {10,20,30};     // [1,3] broadcast
    Term a = thvm_tensor(ctx, d1, SHAPE(2,3));
    Term b = thvm_tensor(ctx, d2, SHAPE(1,3));
    Term prod = thvm_op(ctx, UOP_MUL, a, b);
    extern Term thvm_sum_axes(TinyHVM*, Term, const u32*, u32);
    Term sum = thvm_sum_axes(ctx, prod, (u32[]){1}, 1);
    sum = thvm_reshape(ctx, sum, SHAPE(2));
    f32 z[2]={0}; Term dst=thvm_tensor(ctx,z,SHAPE(2));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=sum;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32 *r=thvm_to_host(ctx,dst);
    // Expected: [1*10+2*20+3*30, 4*10+5*20+6*30] = [140, 320]
    printf("%-6s: [%.1f, %.1f] (expect [140.0, 320.0])\n", dev, r[0], r[1]);
    thvm_free(ctx);
}
int main(void){check("cpu");check("metal");return 0;}
