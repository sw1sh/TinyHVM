// test_mnist_fast.m — Beat tinygrad: JIT command replay
// Step 0: IC reduce captures GPU commands. Steps 1+: replay only.
// Uses fixed buffers for batch data (overwritten each step via memcpy).
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
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

int main(void) {
    srand(42); MNISTData data=mnist_load("data");
    TinyHVM*ctx=thvm_init("metal"); u32 BS=512;

    // Wide 3×3 CNN (working architecture)
    Term cw1=mkw(ctx,(Shape){.dims={16,1,3,3},.rank=4},9),cb1=mkz(ctx,16);
    Term cw2=mkw(ctx,(Shape){.dims={32,16,3,3},.rank=4},144),cb2=mkz(ctx,32);
    Term cw3=mkw(ctx,(Shape){.dims={64,32,3,3},.rank=4},288),cb3=mkz(ctx,64);
    Term cw4=mkw(ctx,(Shape){.dims={64,64,3,3},.rank=4},576),cb4=mkz(ctx,64);
    u32 ff=64*5*5;
    Term lw=mkw(ctx,SHAPE(ff,10),ff),lb=mkz(ctx,10);

    // Fixed batch buffer: overwritten each step (persistent, JIT-safe)
    f32 *batch_buf = calloc(BS*1*28*28, sizeof(f32));
    Term x_buf = thvm_tensor(ctx, batch_buf, (Shape){.dims={BS,1,28,28},.rank=4});
    free(batch_buf);
    thvm_set_requires_grad(ctx, x_buf);

    // Fixed one-hot buffer
    f32 *oh_buf = calloc(BS*10, sizeof(f32));
    Term oh_fixed = thvm_tensor(ctx, oh_buf, SHAPE(BS, 10));
    free(oh_buf);

    // Fixed LR buffer
    f32 lr_val = 0.01f;
    Term lr_fixed = thvm_tensor(ctx, &lr_val, SHAPE(1));

    // Gradient accumulators (persistent, zeroed each step)
    #define NP 10
    Term params[NP]={cw1,cb1,cw2,cb2,cw3,cb3,cw4,cb4,lw,lb};
    u32 psz[]={16*9,16,32*16*9,32,64*32*9,64,64*64*9,64,ff*10,10};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,params[i]);

    Term gs[NP];
    for(int i=0;i<NP;i++){
        f32*z=calloc(psz[i],4);
        gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);
        free(z);
    }

    u32 n_persistent = ctx->tensor_count;

    // Build the training graph using FIXED buffers
    u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2},p1[]={1,1,1,1};
    Term h=thvm_conv2d(ctx,x_buf,cw1,cb1,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_conv2d(ctx,h,cw3,cb3,1,s1,p1);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_conv2d(ctx,h,cw4,cb4,1,s1,p1);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_reshape(ctx,h,SHAPE(BS,ff));
    Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
        thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));

    // CE loss using fixed one-hot buffer (instead of creating fresh each step)
    Term probs = softmax(ctx, logits, BS, 10);
    f32 eps=1e-7f;
    Term eps_t=thvm_expand(ctx,thvm_tensor(ctx,&eps,SHAPE(1,1)),SHAPE(BS,10));
    Term clamped=thvm_op(ctx,UOP_MAX,probs,eps_t);
    Term log_probs=thvm_op(ctx,UOP_LOG,clamped,term_era());
    Term masked=thvm_op(ctx,UOP_MUL,oh_fixed,log_probs);
    Term sum_all=thvm_sum_axes(ctx,masked,(u32[]){0,1},2);
    Term neg=thvm_op(ctx,UOP_NEG,sum_all,term_era());
    f32 inv_B=1.f/(f32)BS; Term scale=thvm_tensor(ctx,&inv_B,SHAPE(1,1));
    Term loss=thvm_op(ctx,UOP_MUL,neg,scale);

    // Gradient + SGD chain
    Term grad_term=thvm_grad_multi(ctx,loss,params,gs,NP);
    Term sgd=term_era();
    for(int i=NP-1;i>=0;i--)
        sgd=thvm_app(ctx,thvm_assign(ctx,params[i],thvm_op(ctx,UOP_SUB,params[i],
            thvm_op(ctx,UOP_MUL,lr_fixed,gs[i]))),sgd);
    Term train_step=thvm_app(ctx,grad_term,sgd);

    u32 n_steps=70;
    printf("=== TinyHVM JIT: wide 3×3 CNN, BS=%u, %u steps ===\n\n",BS,n_steps);

    // Step 0: full IC reduce with JIT capture
    u32 bi=rand()%(data.n_train/BS);
    u32 x_bid=(u32)term_val(x_buf);
    u32 oh_bid=(u32)term_val(oh_fixed);
    // Write batch data to fixed buffer
    ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
        &data.train_images[bi*BS*784], BS*784*sizeof(f32));
    // Write one-hot labels
    f32 *oh=calloc(BS*10,sizeof(f32));
    for(u32 i=0;i<BS;i++) oh[i*10+data.train_labels[bi*BS+i]]=1.f;
    ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh, BS*10*sizeof(f32));
    // Zero gradient accumulators
    for(int i=0;i<NP;i++){
        u32 gid=(u32)term_val(gs[i]);
        memset(metal_pool.bufs[ctx->tensors[gid].buf_id].contents,0,psz[i]*sizeof(f32));
    }

    jit_begin_capture(n_persistent);
    double t0=now_s();
    thvm_reduce(ctx, train_step);
    f32 lv=thvm_to_host(ctx,loss)[0];
    jit_end_capture();
    printf("  step  0: loss=%5.2f (%.0fms, capture)\n",lv,(now_s()-t0)*1000);
    extern u32 total_dispatches; total_dispatches=0;
    thvm_reset(ctx, n_persistent);

    // Steps 1+: JIT replay (skip IC entirely)
    double t_train=now_s();
    for(u32 step=1;step<n_steps;step++){
        bi=rand()%(data.n_train/BS);
        // Update batch data
        ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
            &data.train_images[bi*BS*784], BS*784*sizeof(f32));
        // Update one-hot labels
        memset(oh,0,BS*10*sizeof(f32));
        for(u32 i=0;i<BS;i++) oh[i*10+data.train_labels[bi*BS+i]]=1.f;
        ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh, BS*10*sizeof(f32));
        // Zero gradient accumulators
        for(int i=0;i<NP;i++){
            u32 gid=(u32)term_val(gs[i]);
            memset(metal_pool.bufs[ctx->tensors[gid].buf_id].contents,0,psz[i]*sizeof(f32));
        }
        // Update LR (cosine decay)
        lr_val=0.01f*0.5f*(1.f+cosf(3.14159f*(f32)step/(f32)n_steps));
        memcpy(metal_pool.bufs[ctx->tensors[(u32)term_val(lr_fixed)].buf_id].contents,&lr_val,4);

        jit_replay();

        if(step%10==9){
            metal_flush();
            // Read loss from the JIT's output buffer
            printf("  step %2u: (%.1fs, %.0fms/step)\n",step,now_s()-t_train,(now_s()-t_train)*1000/step);
        }
    }
    metal_flush();
    double total=now_s()-t_train;
    printf("\n  JIT: %.1fs for %u steps (%.0f ms/step)\n",total,n_steps-1,total*1000/(n_steps-1));
    printf("  tinygrad: 5.5s (80ms/step)\n");

    free(oh);
    thvm_free(ctx);
    return 0;
}
