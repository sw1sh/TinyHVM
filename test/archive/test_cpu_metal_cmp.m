#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>

int main(void) {
    // Simple: MUL two known tensors, compare CPU vs Metal
    f32 a[] = {1,2,3,4,5,6};
    f32 b[] = {2,3,4,5,6,7};
    
    for (int dev = 0; dev < 2; dev++) {
        TinyHVM *ctx = thvm_init(dev == 0 ? "cpu" : "metal");
        Term ta = thvm_tensor(ctx, a, SHAPE(2,3));
        Term tb = thvm_tensor(ctx, b, SHAPE(2,3));
        Term prod = thvm_op(ctx, UOP_MUL, ta, tb);
        extern Term thvm_sum_axes(TinyHVM*, Term, const u32*, u32);
        Term sum = thvm_sum_axes(ctx, prod, (u32[]){1}, 1);
        Term out = thvm_reshape(ctx, sum, SHAPE(2));
        f32 z[2] = {0}; Term dst = thvm_tensor(ctx, z, SHAPE(2));
        u64 loc = heap_alloc(ctx, 2); ctx->heap[loc] = dst; ctx->heap[loc+1] = out;
        u64 ac = heap_alloc(ctx, 1); ctx->heap[ac] = term_new(TAG_TOP, UOP_ASSIGN, loc);
        thvm_eval(ctx, term_era());
        f32 *r = thvm_to_host(ctx, dst);
        printf("%-6s: [%.4f, %.4f]  (expect [20.0000, 86.0000])\n",
            dev==0?"cpu":"metal", r[0], r[1]);
        thvm_free(ctx);
    }
    return 0;
}
