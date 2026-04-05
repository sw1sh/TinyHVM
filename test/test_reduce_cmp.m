#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>
static void check(const char *dev) {
    TinyHVM *ctx = thvm_init(dev);
    // Large reduce: sum 10000 random values
    srand(42);
    u32 n = 10000;
    f32 *d = malloc(n*4); for(u32 i=0;i<n;i++) d[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    Term x = thvm_tensor(ctx, d, SHAPE(n)); free(d);
    extern Term thvm_sum_axes(TinyHVM*, Term, const u32*, u32);
    Term sum = thvm_sum_axes(ctx, x, (u32[]){0}, 1);
    sum = thvm_reshape(ctx, sum, SHAPE(1));
    f32 z=0; Term dst=thvm_tensor(ctx,&z,SHAPE(1));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=sum;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32 *r=thvm_to_host(ctx,dst);
    printf("%-6s sum(10000): %.6f\n", dev, r[0]);
    thvm_free(ctx);
}
int main(void){check("cpu");check("metal");return 0;}
