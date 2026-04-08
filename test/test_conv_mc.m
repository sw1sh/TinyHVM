#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
int main(void){
    TinyHVM*ctx=thvm_init("cpu");
    // BS=1 Cin=1 H=3 W=3 Cout=2 K=2
    f32 x[]={1,2,3,4,5,6,7,8,9};
    f32 w[]={1,0,0,1, 0,1,1,0}; // 2 filters [2,1,2,2]
    f32 b[]={0,0};
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={1,1,3,3},.rank=4});
    Term tw=thvm_tensor(ctx,w,(Shape){.dims={2,1,2,2},.rank=4});
    Term tb=thvm_tensor(ctx,b,SHAPE(2));
    Term h=thvm_conv2d(ctx,tx,tw,tb,1,(u32[]){1,1},(u32[]){0,0,0,0});
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term loss=thvm_sum_axes(ctx,h,(u32[]){0,1,2,3},4);
    loss=thvm_reshape(ctx,loss,SHAPE(1));
    f32 z=0; Term dst=thvm_tensor(ctx,&z,SHAPE(1));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=loss;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32*r=thvm_to_host(ctx,dst);
    // Filter 0: [1,0;0,1] → [6,8,12,14] sum=40
    // Filter 1: [0,1;1,0] → [7,11,11,13] sum=42  ← wait, let me recalculate
    // f1: x[0,1]+x[1,0]=2+4=6, x[0,2]+x[1,1]=3+5=8, x[1,1]+x[2,0]=5+7=12, x[1,2]+x[2,1]=6+8=14 → sum=40
    // f2: x[0,1]+x[1,0]=2+4=6, x[0,2]+x[1,1]=3+5=8, x[1,1]+x[2,0]=5+7=12, x[1,2]+x[2,1]=6+8=14 → WAIT
    // f2=[0,1;1,0]: out[oy,ox] = x[oy,ox+1]*1 + x[oy+1,ox]*1
    // (0,0): x[0,1]+x[1,0]=2+4=6, (0,1): x[0,2]+x[1,1]=3+5=8
    // (1,0): x[1,1]+x[2,0]=5+7=12, (1,1): x[1,2]+x[2,1]=6+8=14
    // Both filters give same output! sum = 40+40 = 80
    printf("conv_mc sum: %.1f (expect 80.0)\n", r[0]);
    thvm_free(ctx);return 0;
}
