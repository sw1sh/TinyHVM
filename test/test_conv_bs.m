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
    // BS=4, Cin=1, H=14, W=14, Cout=32, K=3 (same as conv1)
    u32 BS=4,Cin=1,H=14,W=14,Cout=32,K=3;
    u32 xn=BS*Cin*H*W;
    f32 *xd=malloc(xn*4);for(u32 i=0;i<xn;i++)xd[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,Cin,H,W},.rank=4});free(xd);
    f32 b=1.f/sqrtf((f32)(Cin*K*K));
    u32 wn=Cout*Cin*K*K;
    f32 *wd=malloc(wn*4);for(u32 i=0;i<wn;i++)wd[i]=b*((f32)rand()/(f32)RAND_MAX*2-1);
    Term w=thvm_tensor(ctx,wd,(Shape){.dims={Cout,Cin,K,K},.rank=4});free(wd);
    f32 *bz=calloc(Cout,4);Term b1=thvm_tensor(ctx,bz,SHAPE(Cout));free(bz);
    Term h=thvm_conv2d(ctx,x,w,b1,1,(u32[]){1,1},(u32[]){0,0,0,0});
    // Get first 4 output elements (not fused sum — just read raw output)
    Term flat=thvm_reshape(ctx,h,SHAPE(BS*Cout*12*12));
    f32 *dst=calloc(BS*Cout*12*12,4);
    Term dt=thvm_tensor(ctx,dst,SHAPE(BS*Cout*12*12));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dt;ctx->heap[loc+1]=flat;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32*r=thvm_to_host(ctx,dt);
    double sum=0;for(u32 i=0;i<BS*Cout*12*12;i++)sum+=r[i];
    printf("conv sum: %.4f (expect 2590.3853)\n",sum);
    free(dst);thvm_free(ctx);return 0;
}
