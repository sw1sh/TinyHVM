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
static Term mkw(TinyHVM*c,Shape s,u32 fi){u32 n=1;for(u32 i=0;i<s.rank;i++)n*=s.dims[i];f32*d=malloc(n*4);f32 b=sqrtf(2.f/(f32)fi);for(u32 i=0;i<n;i++)d[i]=b*((f32)rand()/(f32)RAND_MAX*2-1)*.5f;Term t=thvm_tensor(c,d,s);free(d);return t;}
static Term mkz(TinyHVM*c,u32 n){f32*z=calloc(n,4);Term t=thvm_tensor(c,z,SHAPE(n));free(z);return t;}
static Term mkones(TinyHVM*c,u32 n){f32*o=malloc(n*4);for(u32 i=0;i<n;i++)o[i]=1.f;Term t=thvm_tensor(c,o,SHAPE(n));free(o);return t;}
int main(void) {
    srand(42);MNISTData data=mnist_load("data");
    TinyHVM*ctx=thvm_init("metal");u32 BS=64,n_hid=32,n_in=784;
    Term w1=mkw(ctx,SHAPE(n_in,n_hid),n_in),b1=mkz(ctx,n_hid);
    Term bn_g=mkones(ctx,n_hid),bn_b=mkz(ctx,n_hid),bn_rm=mkz(ctx,n_hid),bn_rv=mkones(ctx,n_hid);
    Term w2=mkw(ctx,SHAPE(n_hid,10),n_hid),b2=mkz(ctx,10);
    #define NP 6
    Term p[NP]={w1,b1,bn_g,bn_b,w2,b2};
    u32 psz[]={n_in*n_hid,n_hid,n_hid,n_hid,n_hid*10,10};
    for(u32 i=0;i<NP;i++)thvm_set_requires_grad(ctx,p[i]);
    Term td=thvm_tensor(ctx,data.train_images,(Shape){.dims={data.n_train,1,28,28},.rank=4});
    u32 nw=ctx->tensor_count;
    const char*names[]={"w1","b1","bn_g","bn_b","w2","b2"};
    @autoreleasepool{
        Term x=thvm_shrink(ctx,td,(u32[]){0,BS,0,1,0,28,0,28},4);
        thvm_set_requires_grad(ctx,x);x=thvm_reshape(ctx,x,SHAPE(BS,n_in));
        Term h=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,x,w1),thvm_expand(ctx,thvm_reshape(ctx,b1,SHAPE(1,n_hid)),SHAPE(BS,n_hid)));
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        h=thvm_reshape(ctx,h,(Shape){.dims={BS,n_hid,1,1},.rank=4});
        BNResult bn=batchnorm_forward(ctx,h,bn_g,bn_b,bn_rm,bn_rv,BS,n_hid,1,1,1);
        h=thvm_reshape(ctx,bn.output,SHAPE(BS,n_hid));
        Term lo=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,w2),thvm_expand(ctx,thvm_reshape(ctx,b2,SHAPE(1,10)),SHAPE(BS,10)));
        Term loss=cross_entropy_loss(ctx,lo,&data.train_labels[0],BS,10);
        Term gs[NP];for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(p[i])].view.shape);free(z);}
        Term gt=thvm_grad_multi(ctx,loss,p,gs,NP);
        thvm_reduce(ctx,thvm_app(ctx,gt,bn.assigns));
        f32 lv=thvm_to_host(ctx,loss)[0];
        printf("BN Metal gradient check (loss=%.4f):\n",lv);
        for(int i=0;i<NP;i++){
            f32*g=thvm_to_host(ctx,gs[i]);
            f32 mx=0;int nans=0;
            for(u32 j=0;j<psz[i];j++){if(isnan(g[j])||isinf(g[j]))nans++;if(fabsf(g[j])>mx)mx=fabsf(g[j]);}
            printf("  %4s: maxabs=%.6f nans=%d %s\n",names[i],mx,nans,(mx>0&&mx<100&&!nans)?"OK":"BAD");
        }
    }
    thvm_free(ctx);return 0;
}
