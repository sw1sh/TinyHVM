// test_beautiful_mnist.m — Match tinygrad's beautiful_mnist.py exactly
// Conv(1→32,5)→ReLU→Conv(32→32,5)→ReLU→BN(32)→Pool→
// Conv(32→64,3)→ReLU→Conv(64→64,3)→ReLU→BN(64)→Pool→Linear(576,10)
// BS=512, 70 steps, Adam
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

// tinygrad init: uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
static Term mkw(TinyHVM*c,Shape s,u32 fi){u32 n=1;for(u32 i=0;i<s.rank;i++)n*=s.dims[i];f32*d=malloc(n*4);f32 b=1.f/sqrtf((f32)fi);for(u32 i=0;i<n;i++)d[i]=b*((f32)rand()/(f32)RAND_MAX*2-1);Term t=thvm_tensor(c,d,s);free(d);return t;}
static Term mkz(TinyHVM*c,u32 n){f32*z=calloc(n,4);Term t=thvm_tensor(c,z,SHAPE(n));free(z);return t;}
static Term mkones(TinyHVM*c,u32 n){f32*o=malloc(n*4);for(u32 i=0;i<n;i++)o[i]=1.f;Term t=thvm_tensor(c,o,SHAPE(n));free(o);return t;}
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

int main(void) {
    srand(42); MNISTData data=mnist_load("data");
    TinyHVM*ctx=thvm_init("metal");
    u32 BS=128; // tinygrad uses 512
    u32 n_steps=500;

    // Model: beautiful_mnist architecture
    // Conv1: (1→32, 5×5), Conv2: (32→32, 5×5), BN1(32), Pool(2×2)
    Term cw1=mkw(ctx,(Shape){.dims={32,1,5,5},.rank=4},25),cb1=mkz(ctx,32);
    Term cw2=mkw(ctx,(Shape){.dims={32,32,5,5},.rank=4},800),cb2=mkz(ctx,32);
    // Conv3: (32→64, 3×3), Conv4: (64→64, 3×3), BN2(64), Pool(2×2)
    Term cw3=mkw(ctx,(Shape){.dims={64,32,3,3},.rank=4},288),cb3=mkz(ctx,64);
    Term cw4=mkw(ctx,(Shape){.dims={64,64,3,3},.rank=4},576),cb4=mkz(ctx,64);
    // BN params
    Term bn1_g=mkones(ctx,32),bn1_b=mkz(ctx,32),bn1_rm=mkz(ctx,32),bn1_rv=mkones(ctx,32);
    Term bn2_g=mkones(ctx,64),bn2_b=mkz(ctx,64),bn2_rm=mkz(ctx,64),bn2_rv=mkones(ctx,64);
    // Dense: 64*3*3=576 → 10
    u32 ff=64*3*3; // 576
    Term lw=mkw(ctx,SHAPE(ff,10),ff),lb=mkz(ctx,10);

    #define NP 14
    Term params[NP]={cw1,cb1,cw2,cb2,cw3,cb3,cw4,cb4,bn1_g,bn1_b,bn2_g,bn2_b,lw,lb};
    u32 psz[NP]={32*25,32, 32*32*25,32, 64*32*9,64, 64*64*9,64, 32,32, 64,64, ff*10,10};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,params[i]);
    Adam opt=adam_init(ctx,0.0005f,NP);
    for(u32 i=0;i<NP;i++) adam_add_param(ctx,&opt,i,(u32)term_val(params[i]),psz[i]);

    Term td=thvm_tensor(ctx,data.train_images,(Shape){.dims={data.n_train,1,28,28},.rank=4});
    u32 nw=ctx->tensor_count;

    printf("=== TinyHVM beautiful_mnist ===\n");
    printf("  Conv(32,k5)→Conv(32,k5)→BN→Pool→Conv(64,k3)→Conv(64,k3)→BN→Pool→Dense(%u)\n",ff);
    printf("  BS=%u, Adam(lr=0.001), %u steps\n\n",BS,n_steps);

    double t0=now_s();
    for(u32 step=0;step<n_steps;step++){@autoreleasepool{
        u32 bi=rand()%(data.n_train/BS);
        Term x=thvm_shrink(ctx,td,(u32[]){bi*BS,(bi+1)*BS,0,1,0,28,0,28},4);
        thvm_set_requires_grad(ctx,x);
        u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};

        // Conv1(5×5,no pad)→ReLU: 28→24
        Term h=thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        // Conv2(5×5,no pad)→ReLU: 24→20
        h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        // BN1→Pool: 20→10
        BNResult bn1=batchnorm_forward(ctx,h,bn1_g,bn1_b,bn1_rm,bn1_rv,BS,32,20,20,1);
        h=bn1.output;
        h=thvm_maxpool2d(ctx,h,k2,s2);

        // Conv3(3×3,no pad)→ReLU: 10→8
        h=thvm_conv2d(ctx,h,cw3,cb3,1,s1,p0);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        // Conv4(3×3,no pad)→ReLU: 8→6
        h=thvm_conv2d(ctx,h,cw4,cb4,1,s1,p0);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        // BN2→Pool: 6→3
        BNResult bn2=batchnorm_forward(ctx,h,bn2_g,bn2_b,bn2_rm,bn2_rv,BS,64,6,6,1);
        h=bn2.output;
        h=thvm_maxpool2d(ctx,h,k2,s2);

        // Dense
        h=thvm_reshape(ctx,h,SHAPE(BS,ff));
        Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
            thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
        Term loss=cross_entropy_loss(ctx,logits,&data.train_labels[bi*BS],BS,10);

        // Backward + Adam
        Term gs[NP];
        for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
            gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
        Term grad_term=thvm_grad_multi(ctx,loss,params,gs,NP);
        u32 gids[NP]; for(u32 i=0;i<NP;i++) gids[i]=(u32)term_val(gs[i]);
        Term bn_assigns=thvm_app(ctx,bn1.assigns,bn2.assigns);
        thvm_reduce(ctx,thvm_app(ctx,grad_term,bn_assigns));
        adam_step_direct(ctx,&opt,gids);

        if(step<3||step%10==0||step==n_steps-1){
            f32 lv=thvm_to_host(ctx,loss)[0];
            printf("  step %3u: loss=%5.2f (%.1fs)\n",step,lv,now_s()-t0);
        }
        extern u32 total_dispatches; total_dispatches=0;
        thvm_reset(ctx,nw);
    }}

    // Eval
    printf("\n  Evaluating...\n");
    Term test_data=thvm_tensor(ctx,data.test_images,(Shape){.dims={data.n_test,1,28,28},.rank=4});
    u32 ek=ctx->tensor_count; u32 cor=0,tbs=64,tb=data.n_test/tbs;
    for(u32 b=0;b<tb;b++){
        Term tx=thvm_shrink(ctx,test_data,(u32[]){b*tbs,(b+1)*tbs,0,1,0,28,0,28},4);
        u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
        Term th=thvm_conv2d(ctx,tx,cw1,cb1,1,s1,p0);
        th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=thvm_conv2d(ctx,th,cw2,cb2,1,s1,p0);
        th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=batchnorm_forward(ctx,th,bn1_g,bn1_b,bn1_rm,bn1_rv,tbs,32,20,20,0).output;
        th=thvm_maxpool2d(ctx,th,k2,s2);
        th=thvm_conv2d(ctx,th,cw3,cb3,1,s1,p0);
        th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=thvm_conv2d(ctx,th,cw4,cb4,1,s1,p0);
        th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=batchnorm_forward(ctx,th,bn2_g,bn2_b,bn2_rm,bn2_rv,tbs,64,6,6,0).output;
        th=thvm_maxpool2d(ctx,th,k2,s2);
        th=thvm_reshape(ctx,th,SHAPE(tbs,ff));
        Term tlo=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,th,lw),
            thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(tbs,10)));
        f32 acc=thvm_eval_accuracy(ctx,tlo,&data.test_labels[b*tbs],tbs,10);
        cor+=(u32)(acc*(f32)tbs/100.f);thvm_reset(ctx,ek);
    }
    f32 test_acc=100.f*(f32)cor/(f32)(tb*tbs);
    double total=now_s()-t0;
    printf("  Test accuracy: %.1f%% in %.1fs (%u steps, %.0fms/step)\n\n",
           test_acc,total,n_steps,total*1000/n_steps);
    printf("  tinygrad beautiful_mnist: 98.3%% in 5.5s (70 steps, BS=512)\n");

    adam_free(&opt);
    thvm_free(ctx);
    return 0;
}
