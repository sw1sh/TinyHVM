// test_jit_bn.m — JIT replay with BN+Adam, BS=64
// Step 0: IC capture (full fwd+bwd+adam). Steps 1+: replay.
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
    // Normalize to [0,1] for stable training with Adam
    for(u32 i=0;i<data.n_train*784;i++) data.train_images[i]/=255.f;
    for(u32 i=0;i<data.n_test*784;i++) data.test_images[i]/=255.f;
    TinyHVM *ctx=thvm_init("metal"); u32 BS=64;

    // Model: 4-conv + BN + Adam
    Term cw1=mkw(ctx,(Shape){.dims={16,1,3,3},.rank=4},9),cb1=mkz(ctx,16);
    Term cw2=mkw(ctx,(Shape){.dims={32,16,3,3},.rank=4},144),cb2=mkz(ctx,32);
    Term cw3=mkw(ctx,(Shape){.dims={64,32,3,3},.rank=4},288),cb3=mkz(ctx,64);
    Term cw4=mkw(ctx,(Shape){.dims={64,64,3,3},.rank=4},576),cb4=mkz(ctx,64);
    Term bn1_g=mkones(ctx,16),bn1_b=mkz(ctx,16),bn1_rm=mkz(ctx,16),bn1_rv=mkones(ctx,16);
    Term bn2_g=mkones(ctx,32),bn2_b=mkz(ctx,32),bn2_rm=mkz(ctx,32),bn2_rv=mkones(ctx,32);
    u32 ff=64*5*5;
    Term lw=mkw(ctx,SHAPE(ff,10),ff),lb=mkz(ctx,10);

    // Fixed buffers for batch data + one-hot
    f32 *batch_buf=calloc(BS*784,4);
    Term x_buf=thvm_tensor(ctx,batch_buf,SHAPE(BS,1,28,28));free(batch_buf);
    thvm_set_requires_grad(ctx,x_buf);
    f32 *oh_buf=calloc(BS*10,4);
    Term oh_fixed=thvm_tensor(ctx,oh_buf,SHAPE(BS,10));free(oh_buf);

    #define NP 14
    Term params[NP]={cw1,cb1,cw2,cb2,cw3,cb3,cw4,cb4,bn1_g,bn1_b,bn2_g,bn2_b,lw,lb};
    u32 psz[NP]={16*9,16,32*16*9,32,64*32*9,64,64*64*9,64,16,16,32,32,ff*10,10};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,params[i]);

    // Adam optimizer (persistent state)
    Adam opt=adam_init(ctx,0.001f,NP);
    for(u32 i=0;i<NP;i++) adam_add_param(ctx,&opt,i,(u32)term_val(params[i]),psz[i]);

    // Gradient accumulators (persistent)
    Term gs[NP];
    for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
        gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}

    // Build training graph
    u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2},p1[]={1,1,1,1};
    Term h=thvm_conv2d(ctx,x_buf,cw1,cb1,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult bn1=batchnorm_forward(ctx,h,bn1_g,bn1_b,bn1_rm,bn1_rv,BS,16,26,26,1);
    h=bn1.output; h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult bn2=batchnorm_forward(ctx,h,bn2_g,bn2_b,bn2_rm,bn2_rv,BS,32,11,11,1);
    h=bn2.output; h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_conv2d(ctx,h,cw3,cb3,1,s1,p1);h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_conv2d(ctx,h,cw4,cb4,1,s1,p1);h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_reshape(ctx,h,SHAPE(BS,ff));
    Term logits=thvm_op(ctx,UOP_ADD,thvm_mm(ctx,h,lw),
        thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));

    // Loss using fixed one-hot
    Term probs=softmax(ctx,logits,BS,10);
    f32 eps=1e-7f;
    Term eps_t=thvm_expand(ctx,thvm_tensor(ctx,&eps,SHAPE(1,1)),SHAPE(BS,10));
    Term clamped=thvm_op(ctx,UOP_MAX,probs,eps_t);
    Term log_probs=thvm_op(ctx,UOP_LOG,clamped,term_era());
    Term masked=thvm_op(ctx,UOP_MUL,oh_fixed,log_probs);
    Term sum_all=thvm_sum_axes(ctx,masked,(u32[]){0,1},2);
    Term neg=thvm_op(ctx,UOP_NEG,sum_all,term_era());
    f32 inv_B=1.f/(f32)BS;Term scale=thvm_tensor(ctx,&inv_B,SHAPE(1,1));
    Term loss=thvm_op(ctx,UOP_MUL,neg,scale);

    // Gradient + Adam + BN assigns
    Term grad_term=thvm_grad_multi(ctx,loss,params,gs,NP);
    u32 gids[NP]; for(u32 i=0;i<NP;i++) gids[i]=(u32)term_val(gs[i]);
    Term adam_chain=adam_step_lazy(ctx,&opt,gids);
    Term bn_assigns=thvm_app(ctx,bn1.assigns,bn2.assigns);
    Term train_step=thvm_app(ctx,grad_term,thvm_app(ctx,bn_assigns,adam_chain));

    u32 n_persistent=ctx->tensor_count;
    u32 n_steps=300;
    printf("=== JIT BN+Adam BS=%u, %u steps ===\n",BS,n_steps);
    printf("n_persistent=%u, pool_count=%u\n\n",n_persistent,metal_pool.count);

    // Step 0: capture
    u32 bi=0;
    u32 x_bid=(u32)term_val(x_buf), oh_bid=(u32)term_val(oh_fixed);
    ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
        &data.train_images[bi*BS*784], BS*784*sizeof(f32));
    f32 *oh=calloc(BS*10,sizeof(f32));
    for(u32 i=0;i<BS;i++) oh[i*10+data.train_labels[bi*BS+i]]=1.f;
    ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh, BS*10*sizeof(f32));
    for(int i=0;i<NP;i++){u32 gid=(u32)term_val(gs[i]);
        memset(metal_pool.bufs[ctx->tensors[gid].buf_id].contents,0,psz[i]*sizeof(f32));}

    jit_begin_capture(n_persistent);
    double t0=now_s();
    thvm_reduce(ctx,train_step);
    f32 lv=thvm_to_host(ctx,loss)[0];
    jit_end_capture();
    jit_flush();
    {u32 gb=ctx->tensors[(u32)term_val(gs[0])].buf_id;
     f32*gd=(f32*)metal_pool.bufs[gb].contents;
     f32 gs0=0;for(u32 j=0;j<psz[0];j++)gs0+=fabsf(gd[j]);
     printf("step  0: loss=%5.2f |g_cw1|=%.2f (capture, %.0fms)\n",lv,gs0,(now_s()-t0)*1000);
    }

    // Reset for replay
    extern u32 total_dispatches; total_dispatches=0;
    thvm_reset(ctx,n_persistent);

    // Steps 1+: JIT replay
    double t_train=now_s();
    for(u32 step=1;step<n_steps;step++){
        jit_flush();
        bi=rand()%(data.n_train/BS);
        ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
            &data.train_images[bi*BS*784], BS*784*sizeof(f32));
        memset(oh,0,BS*10*sizeof(f32));
        for(u32 i=0;i<BS;i++) oh[i*10+data.train_labels[bi*BS+i]]=1.f;
        ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh, BS*10*sizeof(f32));
        for(int i=0;i<NP;i++){u32 gid=(u32)term_val(gs[i]);
            memset(metal_pool.bufs[ctx->tensors[gid].buf_id].contents,0,psz[i]*sizeof(f32));}

        // Update Adam bias correction for this step
        opt.t = step + 1;
        f32 bc1v = 1.0f - powf(opt.beta1, (f32)opt.t);
        f32 bc2v = 1.0f - powf(opt.beta2, (f32)opt.t);
        memcpy(metal_pool.bufs[ctx->tensors[opt.bc1_tid].buf_id].contents, &bc1v, 4);
        memcpy(metal_pool.bufs[ctx->tensors[opt.bc2_tid].buf_id].contents, &bc2v, 4);

        jit_replay();

        if(step<3||step%50==0||step==n_steps-1){
            jit_flush();
            u32 gbid0=ctx->tensors[(u32)term_val(gs[0])].buf_id;
            f32*gd0=(f32*)metal_pool.bufs[gbid0].contents;
            f32 gs0=0;for(u32 j=0;j<psz[0];j++)gs0+=fabsf(gd0[j]);
            int gnan=isnan(gd0[0])||isnan(gd0[1]);
            u32 wbid=ctx->tensors[(u32)term_val(cw1)].buf_id;
            f32*wd=(f32*)metal_pool.bufs[wbid].contents;
            // Check Adam m/v for cw1
            u32 mbid=ctx->tensors[opt.m_bufs[0]].buf_id;
            f32*md=(f32*)metal_pool.bufs[mbid].contents;
            u32 vbid=ctx->tensors[opt.v_bufs[0]].buf_id;
            f32*vd=(f32*)metal_pool.bufs[vbid].contents;
            // Check bc values
            f32 bc1_cur, bc2_cur;
            memcpy(&bc1_cur, metal_pool.bufs[ctx->tensors[opt.bc1_tid].buf_id].contents, 4);
            memcpy(&bc2_cur, metal_pool.bufs[ctx->tensors[opt.bc2_tid].buf_id].contents, 4);
            printf("step %3u: cw1[0]=%.6f bc1=%.4f bc2=%.6f (%.0fms/step)\n",
                   step,wd[0],bc1_cur,bc2_cur,(now_s()-t_train)*1000/step);
        }
    }
    jit_flush();
    double total=now_s()-t_train;
    printf("\nJIT: %.1fs for %u steps (%.0f ms/step)\n",total,n_steps-1,total*1000/(n_steps-1));

    // Eval
    printf("\nEvaluating...\n");
    Term test_data=thvm_tensor(ctx,data.test_images,(Shape){.dims={data.n_test,1,28,28},.rank=4});
    u32 ek=ctx->tensor_count;u32 cor=0,tbs=64,tb=data.n_test/tbs;
    for(u32 b2=0;b2<tb;b2++){
        Term tx=thvm_shrink(ctx,test_data,(u32[]){b2*tbs,(b2+1)*tbs,0,1,0,28,0,28},4);
        Term th=thvm_conv2d(ctx,tx,cw1,cb1,1,s1,p0);th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=batchnorm_forward(ctx,th,bn1_g,bn1_b,bn1_rm,bn1_rv,tbs,16,26,26,1).output; // training mode for eval (test)
        th=thvm_maxpool2d(ctx,th,k2,s2);
        th=thvm_conv2d(ctx,th,cw2,cb2,1,s1,p0);th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=batchnorm_forward(ctx,th,bn2_g,bn2_b,bn2_rm,bn2_rv,tbs,32,11,11,1).output; // training mode
        th=thvm_maxpool2d(ctx,th,k2,s2);
        th=thvm_conv2d(ctx,th,cw3,cb3,1,s1,p1);th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=thvm_conv2d(ctx,th,cw4,cb4,1,s1,p1);th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=thvm_reshape(ctx,th,SHAPE(tbs,ff));
        Term tlo=thvm_op(ctx,UOP_ADD,thvm_mm(ctx,th,lw),
            thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(tbs,10)));
        f32 acc=thvm_eval_accuracy(ctx,tlo,&data.test_labels[b2*tbs],tbs,10);
        cor+=(u32)(acc*(f32)tbs/100.f);thvm_reset(ctx,ek);
    }
    printf("Test accuracy: %.1f%%\n",100.f*(f32)cor/(f32)(tb*tbs));
    printf("\nnon-JIT: 96.4%% in 34.6s (200 steps, 173ms/step, Adam)\n");
    printf("tinygrad: 98.3%% in 5.5s (70 steps, BS=512, BN+Adam)\n");

    free(oh);adam_free(&opt);thvm_free(ctx);return 0;
}
