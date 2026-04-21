#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>
static void test(const char *dev){
    srand(42);TinyHVM*ctx=thvm_init(dev);
    u32 BS=4,Cin=1,H=14,W=14,Cout=32,K=3;
    u32 xn=BS*Cin*H*W,wn=Cout*Cin*K*K;
    f32 *xd=malloc(xn*4);for(u32 i=0;i<xn;i++)xd[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,Cin,H,W},.rank=4});free(xd);
    thvm_set_requires_grad(ctx,x);
    f32 b=1.f/sqrtf((f32)(Cin*K*K));
    f32 *wd=malloc(wn*4);for(u32 i=0;i<wn;i++)wd[i]=b*((f32)rand()/(f32)RAND_MAX*2-1);
    Term w=thvm_tensor(ctx,wd,(Shape){.dims={Cout,Cin,K,K},.rank=4});free(wd);
    thvm_set_requires_grad(ctx,w);
    f32 *bz=calloc(Cout,4);Term b1=thvm_tensor(ctx,bz,SHAPE(Cout));free(bz);
    thvm_set_requires_grad(ctx,b1);
    // Forward: conv + relu + sum
    Term h=thvm_conv2d(ctx,x,w,b1,1,(u32[]){1,1},(u32[]){0,0,0,0});
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term loss=thvm_sum_axes(ctx,h,(u32[]){0,1,2,3},4);
    loss=thvm_reshape(ctx,loss,SHAPE(1));
    // Grad
    f32*gw1=calloc(wn,4);Term gw=thvm_tensor(ctx,gw1,(Shape){.dims={Cout,Cin,K,K},.rank=4});free(gw1);
    Term grad=thvm_grad_multi(ctx,loss,(Term[]){w},(Term[]){gw},1);
    thvm_eval(ctx,grad);
    f32*dw=thvm_to_host(ctx,gw);
    double n1=0;for(u32 i=0;i<wn;i++)n1+=dw[i]*dw[i];
    printf("%-6s: dw=%.2f\n",dev,sqrt(n1));
    thvm_free(ctx);
}
int main(void){test("cpu");test("metal");return 0;}
