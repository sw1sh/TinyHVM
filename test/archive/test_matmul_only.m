#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
// Simple: 2x3 @ 3x2 matmul
int main(void){
    TinyHVM*ctx=thvm_init("cpu");
    f32 a[]={1,2,3,4,5,6}; // 2x3
    f32 b[]={1,0,0,1,1,1}; // 3x2
    Term ta=thvm_tensor(ctx,a,SHAPE(2,3));
    Term tb=thvm_tensor(ctx,b,SHAPE(3,2));
    Term c=thvm_mm(ctx,ta,tb);
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term s=thvm_sum_axes(ctx,c,(u32[]){0,1},2);
    s=thvm_reshape(ctx,s,SHAPE(1));
    f32 z=0;Term dst=thvm_tensor(ctx,&z,SHAPE(1));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=s;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32*r=thvm_to_host(ctx,dst);
    // Expected: [[1+0+3, 0+2+3],[4+0+6, 0+5+6]] = [[4,5],[10,11]] → sum=30
    printf("matmul sum: %.1f (expect 30.0)\n", r[0]);
    thvm_free(ctx);
    return 0;
}
