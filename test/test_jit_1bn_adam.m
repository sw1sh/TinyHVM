// test_jit_1bn_adam.m — JIT: 1-conv(1→16)+BN+Pool+Dense, Adam, normalized [0,1]
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
    for(u32 i=0;i<data.n_train*784;i++) data.train_images[i]/=255.f;
    for(u32 i=0;i<data.n_test*784;i++) data.test_images[i]/=255.f;
    TinyHVM *ctx=thvm_init("metal"); u32 BS=64, C=16;

    Term cw=mkw(ctx,(Shape){.dims={C,1,3,3},.rank=4},9),cb=mkz(ctx,C);
    Term bn_g=mkones(ctx,C),bn_b=mkz(ctx,C),bn_rm=mkz(ctx,C),bn_rv=mkones(ctx,C);
    u32 ff=C*13*13;
    Term lw=mkw(ctx,SHAPE(ff,10),ff),lb=mkz(ctx,10);

    f32*xbuf=calloc(BS*784,4);
    Term x_buf=thvm_tensor(ctx,xbuf,SHAPE(BS,1,28,28));free(xbuf);
    thvm_set_requires_grad(ctx,x_buf);
    f32*ohbuf=calloc(BS*10,4);
    Term oh=thvm_tensor(ctx,ohbuf,SHAPE(BS,10));free(ohbuf);

    #define NP 6
    Term params[NP]={cw,cb,bn_g,bn_b,lw,lb};
    u32 psz[NP]={C*9,C,C,C,ff*10,10};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,params[i]);

    Adam opt=adam_init(ctx,0.001f,NP);
    for(u32 i=0;i<NP;i++) adam_add_param(ctx,&opt,i,(u32)term_val(params[i]),psz[i]);

    Term gs[NP];
    for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
        gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}

    // Forward
    u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
    Term h=thvm_conv2d(ctx,x_buf,cw,cb,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult bn=batchnorm_forward(ctx,h,bn_g,bn_b,bn_rm,bn_rv,BS,C,26,26,1);
    h=bn.output; h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_reshape(ctx,h,SHAPE(BS,ff));
    Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
        thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
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

    Term grad_term=thvm_grad_multi(ctx,loss,params,gs,NP);
    u32 gids[NP]; for(u32 i=0;i<NP;i++) gids[i]=(u32)term_val(gs[i]);
    Term adam_chain=adam_step_lazy(ctx,&opt,gids);
    Term train_step=thvm_app(ctx,grad_term,thvm_app(ctx,bn.assigns,adam_chain));

    u32 n_persistent=ctx->tensor_count;
    u32 x_bid=(u32)term_val(x_buf), oh_bid=(u32)term_val(oh);
    u32 bi=0;
    u32 bi_cap=426;
    // Print initial weights to verify
    {u32 cwb=ctx->tensors[(u32)term_val(cw)].buf_id;
     if(batch_dirty)metal_flush();
     f32*cwd=(f32*)metal_pool.bufs[cwb].contents;
     u32 lwb=ctx->tensors[(u32)term_val(lw)].buf_id;
     f32*lwd=(f32*)metal_pool.bufs[lwb].contents;
     printf("INIT: cw[0]=%.8f lw[0]=%.8f lw[100]=%.8f\n",cwd[0],lwd[0],lwd[100]);
    }
    ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
        &data.train_images[bi_cap*BS*784], BS*784*sizeof(f32));
    f32*oh_data=calloc(BS*10,sizeof(f32));
    for(u32 i=0;i<BS;i++) oh_data[i*10+data.train_labels[bi_cap*BS+i]]=1.f;
    ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh_data, BS*10*sizeof(f32));
    for(int i=0;i<NP;i++){u32 gid=(u32)term_val(gs[i]);
        memset(metal_pool.bufs[ctx->tensors[gid].buf_id].contents,0,psz[i]*sizeof(f32));}

    jit_begin_capture(n_persistent);
    double t0=now_s();
    thvm_reduce(ctx,train_step);
    jit_end_capture();
    f32 lv=thvm_to_host(ctx,loss)[0];
    jit_flush();
    {u32 lwb2=ctx->tensors[(u32)term_val(lw)].buf_id;
     f32*lwd2=(f32*)metal_pool.bufs[lwb2].contents;
     u32 glb2=ctx->tensors[(u32)term_val(gs[NP-2])].buf_id;
     f32*gld2=(f32*)metal_pool.bufs[glb2].contents;
     printf("JIT step 0: loss=%.6f lw[0]=%.8f lw[100]=%.8f g[0]=%.8f g[100]=%.8f\n",
            lv,lwd2[0],lwd2[100],gld2[0],gld2[100]);
    }

    extern u32 total_dispatches; total_dispatches=0;
    thvm_reset(ctx,n_persistent);
    srand(42); // Reset random seed to match non-JIT batch sequence

    double t_train=now_s();
    u32 n_steps=200;
    for(u32 step=1;step<n_steps;step++){
        jit_flush();
        {static u32 batches[]={426,750,10,839,845,822,305,875,377,198,276,579,446,165,382,127,895,443,267,575,642,178,626,566,191,363,755,750,93,543,753,422,599,516,170,151,210,736,79,194,857,349,329,24,469,347,27,522,707,762,425,3,6,679,891,167,789,594,183,293,101,668,424,108,906,697,591,186,754,384,238,536,610,726,935,844,337,826,41,237,122,894,511,221,754,821,774,568,685,190,263,602,444,530,132,42,168,308,885,852,9,153,234,358,552,350,160,395,331,13,50,62,728,455,80,397,818,183,807,11,790,729,376,220,694,524,779,124,492,636,63,14,641,503,333,790,624,430,563,206,373,787,678,715,122,569,153,81,506,168,913,343,236,208,167,166,295,153,550,92,532,712,3,150,3,673,507,907,189,855,622,817,871,98,270,517,768,415,67,809,769,471,426,186,249,899,188,27,381,338,180,404,182,182,585,696,843,530,914,254}; bi=(step<200)?batches[step]:rand()%(data.n_train/BS);}
        ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
            &data.train_images[bi*BS*784], BS*784*sizeof(f32));
        memset(oh_data,0,BS*10*sizeof(f32));
        for(u32 i=0;i<BS;i++) oh_data[i*10+data.train_labels[bi*BS+i]]=1.f;
        ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh_data, BS*10*sizeof(f32));
        // Split replay: restore consts first, then override per-step values
        { extern void jit_restore_consts(void); jit_restore_consts(); }
        for(int i=0;i<NP;i++){u32 gid=(u32)term_val(gs[i]);
            memset(BUF_CONTENTS(ctx->tensors[gid].buf_id),0,psz[i]*sizeof(f32));}
        opt.t = step + 1;
        f32 bc1v = 1.0f - powf(opt.beta1, (f32)opt.t);
        f32 bc2v = 1.0f - powf(opt.beta2, (f32)opt.t);
        memcpy(BUF_CONTENTS(ctx->tensors[opt.bc1_tid].buf_id), &bc1v, 4);
        memcpy(BUF_CONTENTS(ctx->tensors[opt.bc2_tid].buf_id), &bc2v, 4);
        { extern void jit_replay_commands(void); jit_replay_commands(); }
        if(step<3||step%100==0||step==n_steps-1){
            jit_flush();
            // Check dense layer weight sum and grad sum
            u32 lwb=ctx->tensors[(u32)term_val(lw)].buf_id;
            f32*lwd=(f32*)metal_pool.bufs[lwb].contents;
            f32 lwsum=0;for(u32 j=0;j<ff*10;j++)lwsum+=lwd[j];
            u32 glb=ctx->tensors[(u32)term_val(gs[NP-2])].buf_id;
            f32*gld=(f32*)metal_pool.bufs[glb].contents;
            f32 gsum=0;for(u32 j=0;j<ff*10;j++)gsum+=fabsf(gld[j]);
            {u32 glb2=ctx->tensors[(u32)term_val(gs[NP-2])].buf_id;
             f32*g2=(f32*)metal_pool.bufs[glb2].contents;
             u32 lwb2=ctx->tensors[(u32)term_val(lw)].buf_id;
             f32*lw2=(f32*)metal_pool.bufs[lwb2].contents;
             printf("step %3u: lw[0]=%.8f g[0]=%.8f\n",step,lw2[0],g2[0]);}
        }
    }
    jit_flush();
    printf("JIT: %.1fs for %u steps (%.0f ms/step)\n",now_s()-t_train,n_steps-1,(now_s()-t_train)*1000/(n_steps-1));

    // Eval
    Term test_data=thvm_tensor(ctx,data.test_images,(Shape){.dims={data.n_test,1,28,28},.rank=4});
    u32 ek=ctx->tensor_count;u32 cor=0,tbs=64,tb=data.n_test/tbs;
    for(u32 b=0;b<tb;b++){
        u32 p0e[]={0,0,0,0},s1e[]={1,1},k2e[]={2,2},s2e[]={2,2};
        Term tx=thvm_shrink(ctx,test_data,(u32[]){b*tbs,(b+1)*tbs,0,1,0,28,0,28},4);
        Term th=thvm_conv2d(ctx,tx,cw,cb,1,s1e,p0e);th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=batchnorm_forward(ctx,th,bn_g,bn_b,bn_rm,bn_rv,tbs,C,26,26,1).output; // training mode for eval
        th=thvm_maxpool2d(ctx,th,k2e,s2e);
        th=thvm_reshape(ctx,th,SHAPE(tbs,ff));
        Term tlo=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,th,lw),
            thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(tbs,10)));
        f32 acc=thvm_eval_accuracy(ctx,tlo,&data.test_labels[b*tbs],tbs,10);
        cor+=(u32)(acc*(f32)tbs/100.f);thvm_reset(ctx,ek);
    }
    printf("Test accuracy: %.1f%%\n",100.f*(f32)cor/(f32)(tb*tbs));
    free(oh_data);adam_free(&opt);thvm_free(ctx);return 0;
}
