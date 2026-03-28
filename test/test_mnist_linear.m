// test_mnist_linear.m — Fastest path to 90%: logistic regression
// Linear(784,10) = 7850 params. Normalizes raw pixels by /255.
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
#include <time.h>
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

int main(void) {
    srand(42);
    MNISTData data=mnist_load("data");
    TinyHVM*ctx=thvm_init("metal");
    u32 BS=256, nb=data.n_train/BS, n_in=28*28;

    f32*wd=malloc(n_in*10*4);f32 k=1.f/sqrtf((f32)n_in);
    for(u32 i=0;i<n_in*10;i++)wd[i]=k*((f32)rand()/(f32)RAND_MAX*2-1);
    Term W=thvm_tensor(ctx,wd,SHAPE(n_in,10));free(wd);
    f32 bz[10]={0};Term B=thvm_tensor(ctx,bz,SHAPE(10));
    thvm_set_requires_grad(ctx,W);thvm_set_requires_grad(ctx,B);

    // Normalization constant: 1/255
    f32 inv255=1.f/255.f;
    Term norm=thvm_tensor(ctx,&inv255,SHAPE(1,1));

    Term td=thvm_tensor(ctx,data.train_images,(Shape){.dims={data.n_train,1,28,28},.rank=4});
    u32 nw=ctx->tensor_count;
    #define NP 2
    Term p[NP]={W,B};u32 psz[]={n_in*10,10};

    printf("=== MNIST Linear (7850 params, BS=%u) ===\n\n",BS);
    double t0=now_s();u32 step=0;
    f32 targets[]={85,90,92};double ttimes[3]={-1,-1,-1};

    for(u32 ep=0;ep<5;ep++){
        f32 lr=0.1f*powf(0.7f,(f32)ep);
        for(u32 bi=0;bi<nb;bi++){@autoreleasepool{
            Term x=thvm_shrink(ctx,td,(u32[]){bi*BS,(bi+1)*BS,0,1,0,28,0,28},4);
            thvm_set_requires_grad(ctx,x);
            x=thvm_reshape(ctx,x,SHAPE(BS,n_in));
            x=thvm_op(ctx,UOP_MUL,x,thvm_expand(ctx,norm,SHAPE(BS,n_in))); // normalize
            Term lo=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,x,W),
                thvm_expand(ctx,thvm_reshape(ctx,B,SHAPE(1,10)),SHAPE(BS,10)));
            Term loss=cross_entropy_loss(ctx,lo,&data.train_labels[bi*BS],BS,10);
            Term gs[NP];for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(p[i])].view.shape);free(z);}
            Term gt=thvm_grad_multi(ctx,loss,p,gs,NP);
            Term lt=thvm_tensor(ctx,&lr,SHAPE(1));
            Term sgd=term_era();for(int i=NP-1;i>=0;i--)sgd=thvm_app(ctx,thvm_assign(ctx,p[i],thvm_op(ctx,UOP_SUB,p[i],thvm_op(ctx,UOP_MUL,lt,gs[i]))),sgd);
            thvm_reduce(ctx,thvm_app(ctx,gt,sgd));
            extern u32 total_dispatches;total_dispatches=0;
            for(u32 i=0;i<NP;i++){u32 pid=(u32)term_val(p[i]);if(ctx->tensors[pid].host_ptr){free(ctx->tensors[pid].host_ptr);ctx->tensors[pid].host_ptr=NULL;}}
            thvm_reset(ctx,nw);step++;
        }}
        Term test_data=thvm_tensor(ctx,data.test_images,(Shape){.dims={data.n_test,1,28,28},.rank=4});
        u32 ek=ctx->tensor_count;u32 cor=0,tbs=256,tb=data.n_test/tbs;
        for(u32 b2=0;b2<tb;b2++){
            Term x=thvm_shrink(ctx,test_data,(u32[]){b2*tbs,(b2+1)*tbs,0,1,0,28,0,28},4);
            x=thvm_reshape(ctx,x,SHAPE(tbs,n_in));
            x=thvm_op(ctx,UOP_MUL,x,thvm_expand(ctx,norm,SHAPE(tbs,n_in)));
            Term lo=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,x,W),thvm_expand(ctx,thvm_reshape(ctx,B,SHAPE(1,10)),SHAPE(tbs,10)));
            f32 acc=thvm_eval_accuracy(ctx,lo,&data.test_labels[b2*tbs],tbs,10);
            cor+=(u32)(acc*(f32)tbs/100.f);thvm_reset(ctx,ek);
        }
        f32 acc=100.f*(f32)cor/(f32)(tb*tbs);double el=now_s()-t0;
        for(int t=0;t<3;t++)if(ttimes[t]<0&&acc>=targets[t])ttimes[t]=el;
        printf("  Epoch %u: %.1f%% in %.1fs (lr=%.4f, %.1f ms/step)\n",ep,acc,el,lr,el*1000/step);
    }
    printf("\nTime-to-accuracy:\n");
    for(int t=0;t<3;t++){
        if(ttimes[t]>=0)printf("  %.0f%%: %.1fs\n",targets[t],ttimes[t]);
        else printf("  %.0f%%: not reached\n",targets[t]);
    }
    thvm_free(ctx);return 0;
}
