#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>
static Term mkw(TinyHVM*c,Shape s,u32 fi){u32 n=1;for(u32 i=0;i<s.rank;i++)n*=s.dims[i];f32*d=malloc(n*4);f32 b=1.f/sqrtf((f32)fi);for(u32 i=0;i<n;i++)d[i]=b*((f32)rand()/(f32)RAND_MAX*2-1);Term t=thvm_tensor(c,d,s);free(d);return t;}
static void check(const char *dev) {
    srand(42);
    TinyHVM *ctx = thvm_init(dev);
    u32 BS=4,Cin=1,H=14,W=14,Cout=32,K=3;
    u32 xn=BS*Cin*H*W;
    f32 *xd=malloc(xn*4);for(u32 i=0;i<xn;i++)xd[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,Cin,H,W},.rank=4});free(xd);
    Term w=mkw(ctx,(Shape){.dims={Cout,Cin,K,K},.rank=4},Cin*K*K);
    f32 bz[32]={0}; Term b1=thvm_tensor(ctx,bz,SHAPE(Cout));
    // Just compute conv1 output
    Term h=thvm_conv2d(ctx,x,w,b1,1,(u32[]){1,1},(u32[]){0,0,0,0});
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    // Force eval via ASSIGN
    extern Term thvm_sum_axes(TinyHVM*, Term, const u32*, u32);
    Term loss=thvm_sum_axes(ctx,h,(u32[]){0,1,2,3},4);
    loss=thvm_reshape(ctx,loss,SHAPE(1));
    f32 z=0; Term dst=thvm_tensor(ctx,&z,SHAPE(1));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=loss;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32 *r=thvm_to_host(ctx,dst);
    printf("%-6s conv1_relu_sum: %.6f\n", dev, r[0]);
    thvm_free(ctx);
}
int main(void){check("cpu");check("metal");return 0;}
