#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
int main(void){
    TinyHVM*ctx=thvm_init("cpu");
    // 1 batch, 1 chan, 3x3 input, 1 filter 2x2
    f32 x[]={1,2,3,4,5,6,7,8,9}; // 3x3
    f32 w[]={1,0,0,1};            // 2x2
    f32 b[]={0};
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={1,1,3,3},.rank=4});
    Term tw=thvm_tensor(ctx,w,(Shape){.dims={1,1,2,2},.rank=4});
    Term tb=thvm_tensor(ctx,b,SHAPE(1));
    Term h=thvm_conv2d(ctx,tx,tw,tb,1,(u32[]){1,1},(u32[]){0,0,0,0});
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term s=thvm_sum_axes(ctx,h,(u32[]){0,1,2,3},4);
    s=thvm_reshape(ctx,s,SHAPE(1));
    f32 z=0;Term dst=thvm_tensor(ctx,&z,SHAPE(1));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=s;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32*r=thvm_to_host(ctx,dst);
    // Conv output 2x2: [1*1+2*0+4*0+5*1, 2*1+3*0+5*0+6*1, 4*1+5*0+7*0+8*1, 5*1+6*0+8*0+9*1]
    //                 = [6, 8, 12, 14] → sum = 40
    printf("conv sum: %.1f (expect 40.0)\n", r[0]);
    thvm_free(ctx);
    return 0;
}
