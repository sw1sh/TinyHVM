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
    // Simulate conv: [4,1,12,12,3,3] * [1,1,32,1,1,1,3,3] -> SUM(axes 5,6,7) -> [4,1,32,12,12]
    // Use expand to broadcast
    u32 n1 = 4*1*12*12*3*3; // 5184
    f32 *d1 = malloc(n1*4); for(u32 i=0;i<n1;i++) d1[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    Term a = thvm_tensor(ctx, d1, (Shape){.dims={4,1,1,1,12,12,3,3},.rank=8}); free(d1);
    Term a_exp = thvm_expand(ctx, a, (Shape){.dims={4,1,32,12,12,1,3,3},.rank=8});
    u32 n2 = 32*1*3*3; // 288
    f32 *d2 = malloc(n2*4); for(u32 i=0;i<n2;i++) d2[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    Term b = thvm_tensor(ctx, d2, (Shape){.dims={1,1,32,1,1,1,3,3},.rank=8}); free(d2);
    Term prod = thvm_op(ctx, UOP_MUL, a_exp, b);
    extern Term thvm_sum_axes(TinyHVM*, Term, const u32*, u32);
    Term sum = thvm_sum_axes(ctx, prod, (u32[]){5,6,7}, 3);
    sum = thvm_reshape(ctx, sum, SHAPE(4,32,12,12));
    // Sum all to scalar for comparison
    Term total = thvm_sum_axes(ctx, sum, (u32[]){0,1,2,3}, 4);
    total = thvm_reshape(ctx, total, SHAPE(1));
    f32 z=0; Term dst=thvm_tensor(ctx,&z,SHAPE(1));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=total;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32 *r=thvm_to_host(ctx,dst);
    printf("%-6s conv_like_sum: %.4f\n", dev, r[0]);
    thvm_free(ctx);
}
int main(void){check("cpu");check("metal");return 0;}
