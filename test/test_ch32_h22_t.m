#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static Term mkw(TinyHVM*c,Shape s,u32 fi){u32 n=1;for(u32 i=0;i<s.rank;i++)n*=s.dims[i];f32*d=malloc(n*4);f32 b=1.f/sqrtf((f32)fi);for(u32 i=0;i<n;i++)d[i]=b*((f32)rand()/(f32)RAND_MAX*2-1);Term t=thvm_tensor(c,d,s);free(d);return t;}
static Term mkz(TinyHVM*c,u32 n){f32*z=calloc(n,4);Term t=thvm_tensor(c,z,SHAPE(n));free(z);return t;}
static Term mkones(TinyHVM*c,u32 n){f32*o=malloc(n*4);for(u32 i=0;i<n;i++)o[i]=1.f;Term t=thvm_tensor(c,o,SHAPE(n));free(o);return t;}
static void test(const char *dev){
    srand(42);TinyHVM*ctx=thvm_init(dev);u32 BS=4,Cin=1,H=22,W=22,Cout=32,K=3;
    u32 xn=BS*Cin*H*W,wn=Cout*Cin*K*K,wn2=Cout*Cout*K*K;
    f32 *xd=malloc(xn*4);for(u32 i=0;i<xn;i++)xd[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,Cin,H,W},.rank=4});free(xd);thvm_set_requires_grad(ctx,x);
    Term w=mkw(ctx,(Shape){.dims={Cout,Cin,K,K},.rank=4},Cin*K*K);thvm_set_requires_grad(ctx,w);
    Term b1=mkz(ctx,Cout);thvm_set_requires_grad(ctx,b1);
    Term w2=mkw(ctx,(Shape){.dims={Cout,Cout,K,K},.rank=4},Cout*K*K);thvm_set_requires_grad(ctx,w2);
    Term b2=mkz(ctx,Cout);thvm_set_requires_grad(ctx,b2);
    Term bn_g=mkones(ctx,Cout),bn_b=mkz(ctx,Cout),bn_rm=mkz(ctx,Cout),bn_rv=mkones(ctx,Cout);
    u32 OH=H-K+1;u32 OH2=OH-K+1;
    Term h=thvm_conv2d(ctx,x,w,b1,1,(u32[]){1,1},(u32[]){0,0,0,0});
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_conv2d(ctx,h,w2,b2,1,(u32[]){1,1},(u32[]){0,0,0,0});
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult bn=batchnorm_forward(ctx,h,bn_g,bn_b,bn_rm,bn_rv,BS,Cout,OH2,OH2,1);
    h=bn.output;h=thvm_maxpool2d(ctx,h,(u32[]){2,2},(u32[]){2,2});
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    u32 PH=OH2/2; h=thvm_reshape(ctx,h,SHAPE(BS,Cout*PH*PH));
    Term loss=thvm_sum_axes(ctx,h,(u32[]){0,1},2);loss=thvm_reshape(ctx,loss,SHAPE(1));
    f32*gw1=calloc(wn,4);Term gw=thvm_tensor(ctx,gw1,(Shape){.dims={Cout,Cin,K,K},.rank=4});free(gw1);
    f32*gw2d=calloc(wn2,4);Term gw2=thvm_tensor(ctx,gw2d,(Shape){.dims={Cout,Cout,K,K},.rank=4});free(gw2d);
    Term grad=thvm_grad_multi(ctx,loss,(Term[]){w,w2},(Term[]){gw,gw2},2);
    thvm_reduce(ctx,thvm_app(ctx,grad,bn.assigns));
    f32*dw=thvm_to_host(ctx,gw);f32*dw2=thvm_to_host(ctx,gw2);
    double n1=0,n2=0;for(u32 i=0;i<wn;i++)n1+=dw[i]*dw[i];for(u32 i=0;i<wn2;i++)n2+=dw2[i]*dw2[i];
    printf("%-6s: w1=%.2f w2=%.2f\n",dev,sqrt(n1),sqrt(n2));
    thvm_free(ctx);
}
int main(void){test("cpu");test("metal");return 0;}
