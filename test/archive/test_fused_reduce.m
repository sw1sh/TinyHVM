#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>
// Test: MUL(a,b) → SUM(axes=[0,3,4]) → RESHAPE
// a: [4,1,32,12,12,1,3,3] (all ones)
// b: [1,1,32,1,1,1,3,3] (all ones)
// Expected: MUL = all ones. SUM over [0,3,4] = 4*12*12 = 576 per output element.
// Output shape: [1,1,32,1,1,1,3,3] = 288 elements, each = 576.
// Total sum = 288 * 576 = 165888.
int main(void){
    TinyHVM*ctx=thvm_init("cpu");
    u32 an = 4*1*32*12*12*1*3*3; // 165888
    u32 bn = 1*1*32*1*1*1*3*3;   // 288
    f32 *ad=malloc(an*4);for(u32 i=0;i<an;i++)ad[i]=1.f;
    f32 *bd=malloc(bn*4);for(u32 i=0;i<bn;i++)bd[i]=1.f;
    Term a=thvm_tensor(ctx,ad,shape_of((u32[]){4,1,32,12,12,1,3,3},8));free(ad);
    Term b=thvm_tensor(ctx,bd,shape_of((u32[]){1,1,32,1,1,1,3,3},8));free(bd);
    Term prod=thvm_op(ctx,UOP_MUL,a,b);
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term summed=thvm_sum_axes(ctx,prod,(u32[]){0,3,4},3);
    Term out=thvm_reshape(ctx,summed,shape_of((u32[]){1,1,32,1,1,1,3,3},8));
    // Sum all output elements
    Term total=thvm_sum_axes(ctx,out,(u32[]){0,1,2,3,4,5,6,7},8);
    total=thvm_reshape(ctx,total,SHAPE(1));
    f32 z=0;Term dst=thvm_tensor(ctx,&z,SHAPE(1));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=total;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32*r=thvm_to_host(ctx,dst);
    printf("total: %.1f (expect 165888.0)\n", r[0]);
    thvm_free(ctx);return 0;
}
