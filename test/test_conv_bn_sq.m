#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static void test(const char *dev) {
    srand(42);
    TinyHVM *ctx = thvm_init(dev);
    u32 BS=4,Cin=1,H=8,W=8,Cout=2,K=3;
    u32 xn=BS*Cin*H*W,wn=Cout*Cin*K*K;
    f32 xd[256]; for(u32 i=0;i<xn;i++)xd[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    f32 wd[18]; for(u32 i=0;i<wn;i++)wd[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,Cin,H,W},.rank=4});
    Term w=thvm_tensor(ctx,wd,(Shape){.dims={Cout,Cin,K,K},.rank=4});
    thvm_set_requires_grad(ctx,x); thvm_set_requires_grad(ctx,w);
    // Conv → ReLU → BN → square → sum (non-trivial loss)
    Term h=thvm_conv2d(ctx,x,w,term_era(),1,(u32[]){1,1},(u32[]){0,0,0,0});
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    u32 OH=H-K+1;
    f32 gd[]={1,1};Term gamma=thvm_tensor(ctx,gd,SHAPE(Cout));
    f32 bd[]={0,0};Term beta=thvm_tensor(ctx,bd,SHAPE(Cout));
    f32 rmd[]={0,0};Term rm=thvm_tensor(ctx,rmd,SHAPE(Cout));
    f32 rvd[]={1,1};Term rv=thvm_tensor(ctx,rvd,SHAPE(Cout));
    BNResult bn=batchnorm_forward(ctx,h,gamma,beta,rm,rv,BS,Cout,OH,OH,1);
    // Square the BN output for non-trivial gradient
    Term sq = thvm_op(ctx, UOP_MUL, bn.output, bn.output);
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term loss=thvm_sum_axes(ctx,sq,(u32[]){0,1,2,3},4);
    loss=thvm_reshape(ctx,loss,SHAPE(1));
    f32 gwd[18]={0};Term gw=thvm_tensor(ctx,gwd,(Shape){.dims={Cout,Cin,K,K},.rank=4});
    Term grad=thvm_grad_multi(ctx,loss,(Term[]){w},(Term[]){gw},1);
    thvm_reduce(ctx,thvm_app(ctx,grad,bn.assigns));
    f32*dw=thvm_to_host(ctx,gw);
    double dn=0;for(u32 i=0;i<wn;i++)dn+=dw[i]*dw[i];
    printf("%-6s: dw_norm=%.4f dw[0..3]=[%.4f,%.4f,%.4f,%.4f]\n",
        dev,sqrt(dn),dw[0],dw[1],dw[2],dw[3]);
    thvm_free(ctx);
}
int main(void){test("cpu");test("metal");return 0;}
