#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
int main(void){
    TinyHVM*ctx=thvm_init("cpu");
    f32 x[]={1,2,3,4,5,6,7,8,9};
    f32 w[]={1,0,0,1};
    f32 b[]={0};
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={1,1,3,3},.rank=4});
    Term tw=thvm_tensor(ctx,w,(Shape){.dims={1,1,2,2},.rank=4});
    Term tb=thvm_tensor(ctx,b,SHAPE(1));
    Term h=thvm_conv2d(ctx,tx,tw,tb,1,(u32[]){1,1},(u32[]){0,0,0,0});
    // Force eval via ASSIGN
    f32 z[4]={0}; Term dst=thvm_tensor(ctx,z,(Shape){.dims={1,1,2,2},.rank=4});
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=h;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32*r=thvm_to_host(ctx,dst);
    printf("conv out: [%.1f, %.1f, %.1f, %.1f] (expect [6, 8, 12, 14])\n", r[0],r[1],r[2],r[3]);
    thvm_free(ctx);
    return 0;
}
