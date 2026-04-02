// test_beautiful_mnist.m — Match tinygrad's beautiful_mnist.py
// Conv(1→32,5)→ReLU→Conv(32→32,5)→ReLU→BN(32)→Pool→
// Conv(32→64,3)→ReLU→Conv(64→64,3)→ReLU→BN(64)→Pool→Linear(576,10)
// BS=128, 70 steps, Adam, JIT capture/replay
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

static Term mkw(TinyHVM*c,Shape s,u32 fi){u32 n=1;for(u32 i=0;i<s.rank;i++)n*=s.dims[i];f32*d=malloc(n*4);f32 b=1.f/sqrtf((f32)fi);for(u32 i=0;i<n;i++)d[i]=b*((f32)rand()/(f32)RAND_MAX*2-1);Term t=thvm_tensor(c,d,s);free(d);return t;}
static Term mkz(TinyHVM*c,u32 n){f32*z=calloc(n,4);Term t=thvm_tensor(c,z,SHAPE(n));free(z);return t;}
static Term mkones(TinyHVM*c,u32 n){f32*o=malloc(n*4);for(u32 i=0;i<n;i++)o[i]=1.f;Term t=thvm_tensor(c,o,SHAPE(n));free(o);return t;}
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

int main(void) {
    srand(42); MNISTData data=mnist_load("data");
    TinyHVM*ctx=thvm_init("metal");
    u32 BS=128;
    u32 n_steps=70;

    // Model
    Term cw1=mkw(ctx,(Shape){.dims={32,1,5,5},.rank=4},25),cb1=mkz(ctx,32);
    Term cw2=mkw(ctx,(Shape){.dims={32,32,5,5},.rank=4},800),cb2=mkz(ctx,32);
    Term cw3=mkw(ctx,(Shape){.dims={64,32,3,3},.rank=4},288),cb3=mkz(ctx,64);
    Term cw4=mkw(ctx,(Shape){.dims={64,64,3,3},.rank=4},576),cb4=mkz(ctx,64);
    Term bn1_g=mkones(ctx,32),bn1_b=mkz(ctx,32),bn1_rm=mkz(ctx,32),bn1_rv=mkones(ctx,32);
    Term bn2_g=mkones(ctx,64),bn2_b=mkz(ctx,64),bn2_rm=mkz(ctx,64),bn2_rv=mkones(ctx,64);
    u32 ff=64*3*3;
    Term lw=mkw(ctx,SHAPE(ff,10),ff),lb=mkz(ctx,10);

    #define NP 14
    Term params[NP]={cw1,cb1,cw2,cb2,cw3,cb3,cw4,cb4,bn1_g,bn1_b,bn2_g,bn2_b,lw,lb};
    u32 psz[NP]={32*25,32, 32*32*25,32, 64*32*9,64, 64*64*9,64, 32,32, 64,64, ff*10,10};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,params[i]);

    // Pre-allocate input/label buffers for JIT
    f32*xbuf=calloc(BS*784,4);
    Term x_buf=thvm_tensor(ctx,xbuf,SHAPE(BS,1,28,28));free(xbuf);
    thvm_set_requires_grad(ctx,x_buf);
    f32*ohbuf=calloc(BS*10,4);
    Term oh=thvm_tensor(ctx,ohbuf,SHAPE(BS,10));free(ohbuf);

    Adam opt=adam_init(ctx,0.0005f,NP);
    for(u32 i=0;i<NP;i++) adam_add_param(ctx,&opt,i,(u32)term_val(params[i]),psz[i]);

    Term gs[NP];
    for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
        gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}

    // Build graph once
    u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
    Term h=thvm_conv2d(ctx,x_buf,cw1,cb1,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult bnr1=batchnorm_forward(ctx,h,bn1_g,bn1_b,bn1_rm,bn1_rv,BS,32,20,20,1);
    h=thvm_maxpool2d(ctx,bnr1.output,k2,s2);
    h=thvm_conv2d(ctx,h,cw3,cb3,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_conv2d(ctx,h,cw4,cb4,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult bnr2=batchnorm_forward(ctx,h,bn2_g,bn2_b,bn2_rm,bn2_rv,BS,64,6,6,1);
    h=thvm_maxpool2d(ctx,bnr2.output,k2,s2);
    h=thvm_reshape(ctx,h,SHAPE(BS,ff));
    Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
        thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));

    // Cross-entropy with pre-allocated one-hot
    Term probs=softmax(ctx,logits,BS,10);
    f32 eps=1e-7f;
    Term eps_t=thvm_expand(ctx,thvm_tensor(ctx,&eps,SHAPE(1,1)),SHAPE(BS,10));
    Term clamped=thvm_op(ctx,UOP_MAX,probs,eps_t);
    Term log_probs=thvm_op(ctx,UOP_LOG,clamped,term_era());
    Term masked=thvm_op(ctx,UOP_MUL,oh,log_probs);
    Term sum_all=thvm_sum_axes(ctx,masked,(u32[]){0,1},2);
    Term neg=thvm_op(ctx,UOP_NEG,sum_all,term_era());
    f32 inv_B=1.f/(f32)BS;Term scale=thvm_tensor(ctx,&inv_B,SHAPE(1,1));
    Term loss=thvm_op(ctx,UOP_MUL,neg,scale);

    // Backward + BN assigns + Adam
    Term grad_term=thvm_grad_multi(ctx,loss,params,gs,NP);
    u32 gids[NP]; for(u32 i=0;i<NP;i++) gids[i]=(u32)term_val(gs[i]);
    Term adam_chain=adam_step_lazy(ctx,&opt,gids);
    Term bn_assigns=thvm_app(ctx,bnr1.assigns,bnr2.assigns);
    Term train_step=thvm_app(ctx,grad_term,thvm_app(ctx,bn_assigns,adam_chain));

    u32 n_persistent=ctx->tensor_count;
    u32 x_bid=(u32)term_val(x_buf), oh_bid=(u32)term_val(oh);
    f32*oh_data=calloc(BS*10,sizeof(f32));

    printf("=== TinyHVM beautiful_mnist (JIT) ===\n");
    printf("  Conv(32,k5)→Conv(32,k5)→BN→Pool→Conv(64,k3)→Conv(64,k3)→BN→Pool→Dense(%u)\n",ff);
    printf("  BS=%u, Adam(lr=0.0005), %u steps, JIT capture/replay\n\n",BS,n_steps);

    // Step 0: capture
    u32 bi0=rand()%(data.n_train/BS);
    ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
        &data.train_images[bi0*BS*784], BS*784*sizeof(f32));
    memset(oh_data,0,BS*10*sizeof(f32));
    for(u32 i=0;i<BS;i++) oh_data[i*10+data.train_labels[bi0*BS+i]]=1.f;
    ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh_data, BS*10*sizeof(f32));
    for(int i=0;i<NP;i++){u32 gid=(u32)term_val(gs[i]);
        memset(metal_pool.bufs[ctx->tensors[gid].buf_id].contents,0,psz[i]*sizeof(f32));}

    extern u32 total_dispatches; total_dispatches=0;
    extern void print_dispatch_breakdown(void);
    jit_begin_capture(n_persistent);
    double t0=now_s();
    thvm_reduce(ctx,train_step);
    jit_end_capture();
    f32 lv=thvm_to_host(ctx,loss)[0];
    jit_flush();
    print_dispatch_breakdown();
    printf("  step   0: loss=%5.2f (%.1fs) [capture]\n",lv,now_s()-t0);
    total_dispatches=0;
    thvm_reset(ctx,n_persistent);

    // Steps 1+: replay
    for(u32 step=1;step<n_steps;step++){
        jit_flush();
        u32 bi=rand()%(data.n_train/BS);
        ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
            &data.train_images[bi*BS*784], BS*784*sizeof(f32));
        memset(oh_data,0,BS*10*sizeof(f32));
        for(u32 i=0;i<BS;i++) oh_data[i*10+data.train_labels[bi*BS+i]]=1.f;
        ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh_data, BS*10*sizeof(f32));
        { extern void jit_restore_consts(void); jit_restore_consts(); }
        fprintf(stderr, "  consts restored, zeroing grads...\n");
        for(int i=0;i<NP;i++){u32 gid=(u32)term_val(gs[i]);
            memset(BUF_CONTENTS(ctx->tensors[gid].buf_id),0,psz[i]*sizeof(f32));}
        opt.t = step + 1;
        f32 bc1v = 1.0f - powf(opt.beta1, (f32)opt.t);
        f32 bc2v = 1.0f - powf(opt.beta2, (f32)opt.t);
        memcpy(BUF_CONTENTS(ctx->tensors[opt.bc1_tid].buf_id), &bc1v, 4);
        memcpy(BUF_CONTENTS(ctx->tensors[opt.bc2_tid].buf_id), &bc2v, 4);
        { extern void jit_replay_commands(void); jit_replay_commands(); }
        if(step<3||step%10==0||step==n_steps-1){
            jit_flush();
            lv=thvm_to_host(ctx,loss)[0];
            printf("  step %3u: loss=%5.2f (%.1fs)\n",step,lv,now_s()-t0);
        }
    }
    jit_flush();
    double train_time=now_s()-t0;
    printf("  JIT: %.1fs for %u steps (%.0f ms/step)\n\n",train_time,n_steps,train_time*1000/n_steps);

    // Eval
    printf("  Evaluating...\n");
    u32 ek=ctx->tensor_count; u32 cor=0,tbs=64,tb=data.n_test/tbs;
    for(u32 b=0;b<tb;b++){
        u32 pe[]={0,0,0,0},se[]={1,1},ke[]={2,2},sse[]={2,2};
        Term tx=thvm_tensor(ctx,&data.test_images[b*tbs*784],(Shape){.dims={tbs,1,28,28},.rank=4});
        Term th=thvm_conv2d(ctx,tx,cw1,cb1,1,se,pe);
        th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=thvm_conv2d(ctx,th,cw2,cb2,1,se,pe);
        th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=batchnorm_forward(ctx,th,bn1_g,bn1_b,bn1_rm,bn1_rv,tbs,32,20,20,0).output;
        th=thvm_maxpool2d(ctx,th,ke,sse);
        th=thvm_conv2d(ctx,th,cw3,cb3,1,se,pe);
        th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=thvm_conv2d(ctx,th,cw4,cb4,1,se,pe);
        th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=batchnorm_forward(ctx,th,bn2_g,bn2_b,bn2_rm,bn2_rv,tbs,64,6,6,0).output;
        th=thvm_maxpool2d(ctx,th,ke,sse);
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

    free(oh_data);adam_free(&opt);thvm_free(ctx);
    return 0;
}
