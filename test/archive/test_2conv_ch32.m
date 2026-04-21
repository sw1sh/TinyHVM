#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include "../src/nn/datasets.c"
#include "train_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static Term mkw(TinyHVM*c,Shape s,u32 fi){u32 n=1;for(u32 i=0;i<s.rank;i++)n*=s.dims[i];f32*d=malloc(n*4);f32 b=1.f/sqrtf((f32)fi);for(u32 i=0;i<n;i++)d[i]=b*((f32)rand()/(f32)RAND_MAX*2-1);Term t=thvm_tensor(c,d,s);free(d);return t;}
static Term mkz(TinyHVM*c,u32 n){f32*z=calloc(n,4);Term t=thvm_tensor(c,z,SHAPE(n));free(z);return t;}
static Term mkones(TinyHVM*c,u32 n){f32*o=malloc(n*4);for(u32 i=0;i<n;i++)o[i]=1.f;Term t=thvm_tensor(c,o,SHAPE(n));free(o);return t;}
static void test(const char *dev){
    srand(42);MNISTData data=mnist_load("data");
    TinyHVM*ctx=thvm_init(dev);u32 BS=4;
    Term cw1=mkw(ctx,(Shape){.dims={32,1,3,3},.rank=4},25),cb1=mkz(ctx,32);
    Term cw2=mkw(ctx,(Shape){.dims={32,32,3,3},.rank=4},800),cb2=mkz(ctx,32);
    Term bn_g=mkones(ctx,32),bn_b=mkz(ctx,32),bn_rm=mkz(ctx,32),bn_rv=mkones(ctx,32);
    u32 ff=32*5*5; Term lw=mkw(ctx,SHAPE(ff,10),ff),lb=mkz(ctx,10);
    Term all[]={cw1,cb1,cw2,cb2,bn_g,bn_b,lw,lb};
    for(u32 i=0;i<8;i++) thvm_set_requires_grad(ctx,all[i]);
    Term td=thvm_tensor(ctx,data.train_images,(Shape){.dims={data.n_train,1,28,28},.rank=4});
    Term x=thvm_shrink(ctx,td,(u32[]){0,BS,0,1,0,28,0,28},4);
    thvm_set_requires_grad(ctx,x);
    u32 s1[]={1,1},p0[]={0,0,0,0},k2[]={2,2},s2[]={2,2};
    Term h=thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0);h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult bn=batchnorm_forward(ctx,h,bn_g,bn_b,bn_rm,bn_rv,BS,32,12,12,1);
    h=bn.output;h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_reshape(ctx,h,SHAPE(BS,ff));
    Term logits=thvm_op(ctx,UOP_ADD,thvm_mm(ctx,h,lw),
        thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
    Term loss=cross_entropy_loss(ctx,logits,&data.train_labels[0],BS,10);
    #define NP 4
    Term params[NP]={cw1,cb1,cw2,cb2};u32 psz[NP]={32*9,32,32*32*9,32};
    Term gs[NP];for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
        gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
    Term grad=thvm_grad_multi(ctx,loss,params,gs,NP);
    thvm_reduce(ctx,thvm_app(ctx,grad,bn.assigns));
    printf("%-6s: ",dev);const char*n[NP]={"cw1","cb1","cw2","cb2"};
    for(int i=0;i<NP;i++){f32*gd=thvm_to_host(ctx,gs[i]);double nm=0;
        for(u32 j=0;j<psz[i];j++)nm+=gd[j]*gd[j];printf("%s=%.2f ",n[i],sqrt(nm));}
    printf("\n");thvm_free(ctx);
    #undef NP
}
int main(void){test("cpu");test("metal");return 0;}
