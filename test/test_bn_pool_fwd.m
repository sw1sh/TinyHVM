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
    u32 BS=4,Cin=1,Cout=4,K=3,H=14,W=14;
    u32 xn=BS*Cin*H*W;
    f32 xd[4*14*14]; for(u32 i=0;i<xn;i++)xd[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    f32 wd[4*9]; for(u32 i=0;i<4*9;i++)wd[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,Cin,H,W},.rank=4});
    Term w=thvm_tensor(ctx,wd,(Shape){.dims={Cout,Cin,K,K},.rank=4});
    thvm_set_requires_grad(ctx,x); thvm_set_requires_grad(ctx,w);
    Term h=thvm_conv2d(ctx,x,w,term_era(),1,(u32[]){1,1},(u32[]){0,0,0,0});
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    u32 OH=H-K+1;
    // BN
    f32 gd[]={1,1,1,1};Term gamma=thvm_tensor(ctx,gd,SHAPE(Cout));
    f32 bd[]={0,0,0,0};Term beta=thvm_tensor(ctx,bd,SHAPE(Cout));
    f32 rmd[]={0,0,0,0};Term rm=thvm_tensor(ctx,rmd,SHAPE(Cout));
    f32 rvd[]={1,1,1,1};Term rv=thvm_tensor(ctx,rvd,SHAPE(Cout));
    BNResult bn=batchnorm_forward(ctx,h,gamma,beta,rm,rv,BS,Cout,OH,OH,1);
    h=bn.output;
    // Pool
    h=thvm_maxpool2d(ctx,h,(u32[]){2,2},(u32[]){2,2});
    thvm_reduce(ctx,h);
    f32*out=thvm_to_host(ctx,h);
    u32 on=BS*Cout*OH/2*OH/2;
    double norm=0;for(u32 i=0;i<on;i++)norm+=out[i]*out[i];
    printf("%-6s: pool_out norm=%.6f first=[%.6f,%.6f,%.6f,%.6f]\n",
        dev,sqrt(norm),out[0],out[1],out[2],out[3]);
    thvm_free(ctx);
}
int main(void){test("cpu");test("metal");return 0;}
