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
static void test(const char *dev) {
    srand(42); TinyHVM *ctx = thvm_init(dev);
    u32 BS=4;
    Term cw1=mkw(ctx,(Shape){.dims={4,1,5,5},.rank=4},25),cb1=mkz(ctx,4);
    Term cw2=mkw(ctx,(Shape){.dims={4,4,5,5},.rank=4},100),cb2=mkz(ctx,4);
    Term cw3=mkw(ctx,(Shape){.dims={8,4,3,3},.rank=4},36),cb3=mkz(ctx,8);
    Term cw4=mkw(ctx,(Shape){.dims={8,8,3,3},.rank=4},72),cb4=mkz(ctx,8);
    Term bn1_g=mkones(ctx,4),bn1_b=mkz(ctx,4),bn1_rm=mkz(ctx,4),bn1_rv=mkones(ctx,4);
    Term bn2_g=mkones(ctx,8),bn2_b=mkz(ctx,8),bn2_rm=mkz(ctx,8),bn2_rv=mkones(ctx,8);
    f32 xd[4*1*28*28]; for(u32 i=0;i<sizeof(xd)/4;i++) xd[i]=(f32)rand()/(f32)RAND_MAX;
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,1,28,28},.rank=4});
    thvm_set_requires_grad(ctx,x);
    u32 s1[]={1,1},p0[]={0,0,0,0},k2[]={2,2},s2[]={2,2};
    Term h=thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0); h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0); h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult bn1=batchnorm_forward(ctx,h,bn1_g,bn1_b,bn1_rm,bn1_rv,BS,4,20,20,1);
    h=bn1.output; h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_conv2d(ctx,h,cw3,cb3,1,s1,p0); h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_conv2d(ctx,h,cw4,cb4,1,s1,p0); h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult bn2=batchnorm_forward(ctx,h,bn2_g,bn2_b,bn2_rm,bn2_rv,BS,8,6,6,1);
    h=bn2.output; h=thvm_maxpool2d(ctx,h,k2,s2);
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    h=thvm_reshape(ctx,h,SHAPE(BS,8*3*3)); // 8*3*3=72
    Term loss=thvm_sum_axes(ctx,h,(u32[]){0,1},2);
    loss=thvm_reshape(ctx,loss,SHAPE(1));
    #define NP 8
    Term params[NP]={cw1,cb1,cw2,cb2,cw3,cb3,cw4,cb4};
    u32 psz[NP]={4*25,4,4*4*25,4,8*4*9,8,8*8*9,8};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,params[i]);
    Term gs[NP]; for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
        gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
    Term grad=thvm_grad_multi(ctx,loss,params,gs,NP);
    thvm_reduce(ctx,thvm_app(ctx,thvm_app(ctx,grad,bn1.assigns),bn2.assigns));
    printf("%-6s: ",dev);
    const char*n[NP]={"cw1","cb1","cw2","cb2","cw3","cb3","cw4","cb4"};
    for(int i=0;i<NP;i++){f32*gd=thvm_to_host(ctx,gs[i]);double nm=0;
        for(u32 j=0;j<psz[i];j++)nm+=gd[j]*gd[j];printf("%s=%.1f ",n[i],sqrt(nm));}
    printf("\n"); thvm_free(ctx);
    #undef NP
}
int main(void){test("cpu");test("metal");return 0;}
