#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>
int main(void){
    srand(42);TinyHVM*ctx=thvm_init("cpu");
    u32 BS=4,Cin=1,H=14,W=14,Cout=32,K=3;
    u32 wn=Cout*Cin*K*K;
    // Just forward + simple sum to check if reduce fuses correctly
    u32 xn=BS*Cin*H*W;
    f32 *xd=malloc(xn*4);for(u32 i=0;i<xn;i++)xd[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,Cin,H,W},.rank=4});free(xd);
    f32 b=1.f/sqrtf((f32)(Cin*K*K));
    f32 *wd=malloc(wn*4);for(u32 i=0;i<wn;i++)wd[i]=b*((f32)rand()/(f32)RAND_MAX*2-1);
    Term w=thvm_tensor(ctx,wd,(Shape){.dims={Cout,Cin,K,K},.rank=4});free(wd);
    f32 *bz=calloc(Cout,4);Term b1=thvm_tensor(ctx,bz,SHAPE(Cout));free(bz);
    Term h=thvm_conv2d(ctx,x,w,b1,1,(u32[]){1,1},(u32[]){0,0,0,0});
    // Sum all to scalar
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term loss=thvm_sum_axes(ctx,h,(u32[]){0,1,2,3},4);
    loss=thvm_reshape(ctx,loss,SHAPE(1));
    f32 z=0; Term dst=thvm_tensor(ctx,&z,SHAPE(1));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=loss;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32*r=thvm_to_host(ctx,dst);
    printf("forward sum: %.4f (expect 2590.3853)\n", r[0]);
    thvm_free(ctx);return 0;
}
