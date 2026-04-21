#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
int main(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4,5,6}; f32 bd[] = {2,2,2,3,3,3};
    Term a = thvm_tensor(ctx, ad, SHAPE(2,3));
    Term b = thvm_tensor(ctx, bd, SHAPE(2,3));
    Term prod = thvm_op(ctx, UOP_MUL, a, b);
    extern Term thvm_sum_axes(TinyHVM*, Term, const u32*, u32);
    Term sum = thvm_sum_axes(ctx, prod, (u32[]){1}, 1);
    Term out = thvm_reshape(ctx, sum, SHAPE(2));
    f32 z[2] = {0}; Term dst = thvm_tensor(ctx, z, SHAPE(2));
    u64 loc = heap_alloc(ctx, 2); ctx->heap[loc] = dst; ctx->heap[loc+1] = out;
    u64 ac = heap_alloc(ctx, 1); ctx->heap[ac] = term_new(TAG_TOP, UOP_ASSIGN, loc);
    thvm_eval(ctx, term_era());
    f32 *r = thvm_to_host(ctx, dst);
    int ok = (r[0] > 11.9f && r[0] < 12.1f && r[1] > 44.9f && r[1] < 45.1f);
    printf("%s: cpu_dispatch [%.1f, %.1f] (expect [12.0, 45.0])\n", ok?"PASS":"FAIL", r[0], r[1]);
    thvm_free(ctx); return ok ? 0 : 1;
}
