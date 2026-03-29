// test_mnist_fast.m — CNN + BatchNorm + Adam, targeting tinygrad parity
// Conv(1→16,3)→ReLU→BN→Pool→Conv(16→32,3)→ReLU→BN→Pool
//   →Conv(32→64,3,p1)→ReLU→Conv(64→64,3,p1)→ReLU→Dense(1600→10)
// BS=64, Adam, 70 steps per epoch. Safe (no JIT, thvm_to_host for sync).
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

static Term mkw(TinyHVM*c,Shape s,u32 fi){u32 n=1;for(u32 i=0;i<s.rank;i++)n*=s.dims[i];f32*d=malloc(n*4);f32 b=sqrtf(2.f/(f32)fi);for(u32 i=0;i<n;i++)d[i]=b*((f32)rand()/(f32)RAND_MAX*2-1)*.5f;Term t=thvm_tensor(c,d,s);free(d);return t;}
static Term mkz(TinyHVM*c,u32 n){f32*z=calloc(n,4);Term t=thvm_tensor(c,z,SHAPE(n));free(z);return t;}
static Term mkones(TinyHVM*c,u32 n){f32*o=malloc(n*4);for(u32 i=0;i<n;i++)o[i]=1.f;Term t=thvm_tensor(c,o,SHAPE(n));free(o);return t;}
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

int main(void) {
    srand(42); MNISTData data=mnist_load("data");
    TinyHVM*ctx=thvm_init("metal"); u32 BS=64;

    // Conv weights
    Term cw1=mkw(ctx,(Shape){.dims={16,1,3,3},.rank=4},9),cb1=mkz(ctx,16);
    Term cw2=mkw(ctx,(Shape){.dims={32,16,3,3},.rank=4},144),cb2=mkz(ctx,32);
    Term cw3=mkw(ctx,(Shape){.dims={64,32,3,3},.rank=4},288),cb3=mkz(ctx,64);
    Term cw4=mkw(ctx,(Shape){.dims={64,64,3,3},.rank=4},576),cb4=mkz(ctx,64);

    // BN params: gamma=1, beta=0, rmean=0, rvar=1
    Term bn1_g=mkones(ctx,16), bn1_b=mkz(ctx,16), bn1_rm=mkz(ctx,16), bn1_rv=mkones(ctx,16);
    Term bn2_g=mkones(ctx,32), bn2_b=mkz(ctx,32), bn2_rm=mkz(ctx,32), bn2_rv=mkones(ctx,32);

    // Dense
    u32 ff=64*5*5;
    Term lw=mkw(ctx,SHAPE(ff,10),ff),lb=mkz(ctx,10);

    // All trainable params (conv weights + biases + BN gamma/beta + dense)
    // BN running_mean/var are NOT trainable (updated via ASSIGN, not grad)
    #define NP 14
    Term params[NP]={cw1,cb1,cw2,cb2,cw3,cb3,cw4,cb4,
                     bn1_g,bn1_b,bn2_g,bn2_b,lw,lb};
    u32 psz[NP]={16*9,16, 32*16*9,32, 64*32*9,64, 64*64*9,64,
                 16,16, 32,32, ff*10,10};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,params[i]);

    // Adam
    Adam opt=adam_init(ctx,0.001f,NP);
    for(u32 i=0;i<NP;i++) adam_add_param(ctx,&opt,i,(u32)term_val(params[i]),psz[i]);

    Term td=thvm_tensor(ctx,data.train_images,(Shape){.dims={data.n_train,1,28,28},.rank=4});
    u32 nw=ctx->tensor_count;

    u32 n_steps=200;
    printf("=== TinyHVM CNN + BatchNorm + Adam ===\n");
    printf("  Conv(16)→BN→Pool→Conv(32)→BN→Pool→Conv(64,p1)→Conv(64,p1)→Dense(%u)\n",ff);
    printf("  BS=%u, Adam(lr=0.001), %u steps\n\n",BS,n_steps);

    double t0=now_s(); f32 test_acc=0;
    for(u32 step=0;step<n_steps;step++){@autoreleasepool{
        u32 bi=rand()%(data.n_train/BS);
        Term x=thvm_shrink(ctx,td,(u32[]){bi*BS,(bi+1)*BS,0,1,0,28,0,28},4);
        thvm_set_requires_grad(ctx,x);
        u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2},p1[]={1,1,1,1};

        // Conv1 → ReLU → BN1 → Pool
        Term h=thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        BNResult bn1=batchnorm_forward(ctx,h,bn1_g,bn1_b,bn1_rm,bn1_rv,BS,16,26,26,1);
        h=bn1.output;
        h=thvm_maxpool2d(ctx,h,k2,s2);

        // Conv2 → ReLU → BN2 → Pool
        h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        BNResult bn2=batchnorm_forward(ctx,h,bn2_g,bn2_b,bn2_rm,bn2_rv,BS,32,11,11,1);
        h=bn2.output;
        h=thvm_maxpool2d(ctx,h,k2,s2);

        // Conv3 → ReLU → Conv4 → ReLU → Dense
        h=thvm_conv2d(ctx,h,cw3,cb3,1,s1,p1);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        h=thvm_conv2d(ctx,h,cw4,cb4,1,s1,p1);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        h=thvm_reshape(ctx,h,SHAPE(BS,ff));
        Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
            thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
        Term loss=cross_entropy_loss(ctx,logits,&data.train_labels[bi*BS],BS,10);

        // Gradient + Adam
        Term gs[NP];
        for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
            gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
        Term grad_term=thvm_grad_multi(ctx,loss,params,gs,NP);
        u32 gids[NP]; for(u32 i=0;i<NP;i++) gids[i]=(u32)term_val(gs[i]);
        Term adam_chain=adam_step_lazy(ctx,&opt,gids);
        // Chain: grad deposits → BN running stat updates → Adam updates
        Term bn_assigns=thvm_app(ctx,bn1.assigns,bn2.assigns);
        thvm_reduce(ctx,thvm_app(ctx,grad_term,thvm_app(ctx,bn_assigns,adam_chain)));

        if(step<3||step%20==0||step==n_steps-1){
            f32 lv=thvm_to_host(ctx,loss)[0];
            printf("  step %3u: loss=%5.2f (%.1fs)\n",step,lv,now_s()-t0);
        }
        extern u32 total_dispatches;
        extern u32 total_dispatches;
        total_dispatches=0;
        // Don't free host_ptr — shape data is still valid after Adam updates
        thvm_reset(ctx,nw);
    }}

    // Eval
    printf("\n  Evaluating...\n");
    Term test_data=thvm_tensor(ctx,data.test_images,(Shape){.dims={data.n_test,1,28,28},.rank=4});
    u32 ek=ctx->tensor_count; u32 cor=0,tbs=64,tb=data.n_test/tbs;
    for(u32 b=0;b<tb;b++){
        Term tx=thvm_shrink(ctx,test_data,(u32[]){b*tbs,(b+1)*tbs,0,1,0,28,0,28},4);
        u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2},p1[]={1,1,1,1};
        Term th=thvm_conv2d(ctx,tx,cw1,cb1,1,s1,p0);
        th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=batchnorm_forward(ctx,th,bn1_g,bn1_b,bn1_rm,bn1_rv,tbs,16,26,26,0).output;
        th=thvm_maxpool2d(ctx,th,k2,s2);
        th=thvm_conv2d(ctx,th,cw2,cb2,1,s1,p0);
        th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=batchnorm_forward(ctx,th,bn2_g,bn2_b,bn2_rm,bn2_rv,tbs,32,11,11,0).output;
        th=thvm_maxpool2d(ctx,th,k2,s2);
        th=thvm_conv2d(ctx,th,cw3,cb3,1,s1,p1);
        th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=thvm_conv2d(ctx,th,cw4,cb4,1,s1,p1);
        th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=thvm_reshape(ctx,th,SHAPE(tbs,ff));
        Term tlo=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,th,lw),
            thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(tbs,10)));
        f32 acc=thvm_eval_accuracy(ctx,tlo,&data.test_labels[b*tbs],tbs,10);
        cor+=(u32)(acc*(f32)tbs/100.f);thvm_reset(ctx,ek);
    }
    test_acc=100.f*(f32)cor/(f32)(tb*tbs);
    double total=now_s()-t0;
    printf("  Test accuracy: %.1f%% in %.1fs (%u steps, %.0fms/step)\n\n",
           test_acc,total,n_steps,total*1000/n_steps);
    printf("  tinygrad: 98.3%% in 5.5s (70 steps, BS=512, BN+Adam)\n");

    adam_free(&opt);
    thvm_free(ctx);
    return 0;
}
