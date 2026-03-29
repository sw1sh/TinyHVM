// test_jit_bn_mini.m — Minimal BN JIT: Conv(1→8) + BN + Pool + Dense(8*13*13 → 10)
// Smallest possible model with BN to debug JIT replay gradient vanishing.
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
    TinyHVM *ctx=thvm_init("metal"); u32 BS=64;

    // 2-conv+1-BN: Conv1(1→8) + BN + Pool + Conv2(8→16) + Pool + Dense
    u32 C1=8, C2=16;
    Term cw=mkw(ctx,(Shape){.dims={C1,1,3,3},.rank=4},9),cb=mkz(ctx,C1);
    Term bn_g=mkones(ctx,C1),bn_b=mkz(ctx,C1),bn_rm=mkz(ctx,C1),bn_rv=mkones(ctx,C1);
    Term cw2=mkw(ctx,(Shape){.dims={C2,C1,3,3},.rank=4},C1*9),cb2=mkz(ctx,C2);
    u32 ff=C2*5*5; // conv1→26, pool→13, conv2→11, pool→5
    Term lw=mkw(ctx,SHAPE(ff,10),ff),lb=mkz(ctx,10);

    // Fixed input/label buffers
    f32 *xbuf=calloc(BS*784,4);
    Term x_buf=thvm_tensor(ctx,xbuf,SHAPE(BS,1,28,28));free(xbuf);
    thvm_set_requires_grad(ctx,x_buf);
    f32 *ohbuf=calloc(BS*10,4);
    Term oh=thvm_tensor(ctx,ohbuf,SHAPE(BS,10));free(ohbuf);

    #define NP 8
    Term params[NP]={cw,cb,bn_g,bn_b,cw2,cb2,lw,lb};
    u32 psz[NP]={C1*9,C1,C1,C1,C2*C1*9,C2,ff*10,10};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,params[i]);

    // SGD with small LR (simpler than Adam for debugging)
    f32 lr_val=0.01f;
    Term lr=thvm_tensor(ctx,&lr_val,SHAPE(1));

    // Grad accumulators
    Term gs[NP];
    for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
        gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}

    // Forward: Conv1 → ReLU → BN1 → Pool → Conv2 → ReLU → BN2 → Pool → Dense
    u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
    Term h=thvm_conv2d(ctx,x_buf,cw,cb,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult bn=batchnorm_forward(ctx,h,bn_g,bn_b,bn_rm,bn_rv,BS,C1,26,26,1);
    h=bn.output; h=thvm_maxpool2d(ctx,h,k2,s2);
    // Conv2 + BN2
    Term bn2_g=mkones(ctx,C2),bn2_b=mkz(ctx,C2),bn2_rm=mkz(ctx,C2),bn2_rv=mkones(ctx,C2);
    h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult bn2=batchnorm_forward(ctx,h,bn2_g,bn2_b,bn2_rm,bn2_rv,BS,C2,11,11,1);
    h=bn2.output; h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_reshape(ctx,h,SHAPE(BS,ff));
    Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
        thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));

    // Loss
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

    // Grad + SGD + BN assigns
    Term grad_term=thvm_grad_multi(ctx,loss,params,gs,NP);
    Term sgd=term_era();
    for(int i=NP-1;i>=0;i--)
        sgd=thvm_app(ctx,thvm_assign(ctx,params[i],thvm_op(ctx,UOP_SUB,params[i],
            thvm_op(ctx,UOP_MUL,lr,gs[i]))),sgd);
    Term bn_assigns=thvm_app(ctx,bn.assigns,bn2.assigns);
    // GRAD ONLY - no SGD for debugging
    Term train_step=grad_term;

    u32 n_persistent=ctx->tensor_count;

    // Step 0: non-JIT reference (same batch)
    u32 bi=0;
    u32 x_bid=(u32)term_val(x_buf), oh_bid=(u32)term_val(oh);
    ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
        &data.train_images[bi*BS*784], BS*784*sizeof(f32));
    f32 *oh_data=calloc(BS*10,sizeof(f32));
    for(u32 i=0;i<BS;i++) oh_data[i*10+data.train_labels[bi*BS+i]]=1.f;
    ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh_data, BS*10*sizeof(f32));
    for(int i=0;i<NP;i++){u32 gid=(u32)term_val(gs[i]);
        memset(metal_pool.bufs[ctx->tensors[gid].buf_id].contents,0,psz[i]*sizeof(f32));}

    // JIT capture
    jit_begin_capture(n_persistent);
    thvm_reduce(ctx,train_step);
    f32 lv=thvm_to_host(ctx,loss)[0];
    jit_end_capture();
    printf("Capture: loss=%.4f (%u cmds, %u slots)\n",lv,jit.n_cmds,jit.n_slots);

    // Check capture gradients
    jit_flush();
    for(int i=0;i<NP;i++){
        u32 gbid=ctx->tensors[(u32)term_val(gs[i])].buf_id;
        f32*gd=(f32*)metal_pool.bufs[gbid].contents;
        f32 gsum=0;for(u32 j=0;j<psz[i];j++)gsum+=fabsf(gd[j]);
        printf("  cap gs[%d] |grad|=%.4f\n",i,gsum);
    }

    // Reset
    extern u32 total_dispatches; total_dispatches=0;
    thvm_reset(ctx,n_persistent);

    // Replay with SAME batch data
    jit_flush();
    // Re-write same batch data
    ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
        &data.train_images[bi*BS*784], BS*784*sizeof(f32));
    memset(oh_data,0,BS*10*sizeof(f32));
    for(u32 i=0;i<BS;i++) oh_data[i*10+data.train_labels[bi*BS+i]]=1.f;
    ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh_data, BS*10*sizeof(f32));
    for(int i=0;i<NP;i++){u32 gid=(u32)term_val(gs[i]);
        memset(metal_pool.bufs[ctx->tensors[gid].buf_id].contents,0,psz[i]*sizeof(f32));}

    #if 0 // Disabled: NON-JIT verification
    // This tells us if the NaN is JIT-specific or numerical
    {
        thvm_reset(ctx,n_persistent); // reset ephemeral
        ctx->prescan_done = 0;
        // Write same batch data
        ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
            &data.train_images[0], BS*784*sizeof(f32));
        memset(oh_data,0,BS*10*sizeof(f32));
        for(u32 i=0;i<BS;i++) oh_data[i*10+data.train_labels[i]]=1.f;
        ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh_data, BS*10*sizeof(f32));
        for(int i=0;i<NP;i++){u32 gid=(u32)term_val(gs[i]);
            memset(metal_pool.bufs[ctx->tensors[gid].buf_id].contents,0,psz[i]*sizeof(f32));}

        // Build fresh graph (same structure, current weights)
        u32 p02[]={0,0,0,0},s12[]={1,1},k22[]={2,2},s22[]={2,2};
        Term h2=thvm_conv2d(ctx,x_buf,cw1,cb1,1,s12,p02);
        h2=thvm_op(ctx,UOP_RELU,h2,term_era());
        BNResult bn1b=batchnorm_forward(ctx,h2,bn1_g,bn1_b,bn1_rm,bn1_rv,BS,C1,26,26,1);
        h2=bn1b.output; h2=thvm_maxpool2d(ctx,h2,k22,s22);
        h2=thvm_conv2d(ctx,h2,cw2,cb2,1,s12,p02);
        h2=thvm_op(ctx,UOP_RELU,h2,term_era());
        BNResult bn2b=batchnorm_forward(ctx,h2,bn2_g,bn2_b,bn2_rm,bn2_rv,BS,C2,11,11,1);
        h2=bn2b.output; h2=thvm_maxpool2d(ctx,h2,k22,s22);
        h2=thvm_reshape(ctx,h2,SHAPE(BS,ff));
        Term logits2=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h2,lw),
            thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
        Term probs2=softmax(ctx,logits2,BS,10);
        f32 eps2=1e-7f;
        Term eps_t2=thvm_expand(ctx,thvm_tensor(ctx,&eps2,SHAPE(1,1)),SHAPE(BS,10));
        Term clamped2=thvm_op(ctx,UOP_MAX,probs2,eps_t2);
        Term log_probs2=thvm_op(ctx,UOP_LOG,clamped2,term_era());
        Term masked2=thvm_op(ctx,UOP_MUL,oh,log_probs2);
        Term sum_all2=thvm_sum_axes(ctx,masked2,(u32[]){0,1},2);
        Term neg2=thvm_op(ctx,UOP_NEG,sum_all2,term_era());
        f32 inv_B2=1.f/(f32)BS;Term scale2=thvm_tensor(ctx,&inv_B2,SHAPE(1,1));
        Term loss2=thvm_op(ctx,UOP_MUL,neg2,scale2);
        Term grad2=thvm_grad_multi(ctx,loss2,params,gs,NP);
        thvm_reduce(ctx,grad2);
        printf("\nNon-JIT verification (same weights, same batch):\n");
        for(int pi=0;pi<NP;pi++){
            u32 gb=ctx->tensors[(u32)term_val(gs[pi])].buf_id;
            f32*gd=(f32*)metal_pool.bufs[gb].contents;
            f32 gsi=0;for(u32 j=0;j<psz[pi];j++)gsi+=fabsf(gd[j]);
            int nn=0;for(u32 j=0;j<psz[pi];j++)if(isnan(gd[j]))nn++;
            printf("  gs[%d] |g|=%.4f nan=%d/%d\n",pi,gsi,nn,psz[pi]);
        }
    }

    #endif // Disabled: NON-JIT verification

    // Run 200 replay steps
    double t_train=now_s();
    u32 n_steps=200;
    for(u32 step=1;step<n_steps;step++){
        jit_flush();
        bi=0; // same batch as capture for debugging
        ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
            &data.train_images[bi*BS*784], BS*784*sizeof(f32));
        memset(oh_data,0,BS*10*sizeof(f32));
        for(u32 i=0;i<BS;i++) oh_data[i*10+data.train_labels[bi*BS+i]]=1.f;
        ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh_data, BS*10*sizeof(f32));
        for(int i=0;i<NP;i++){u32 gid=(u32)term_val(gs[i]);
            memset(metal_pool.bufs[ctx->tensors[gid].buf_id].contents,0,psz[i]*sizeof(f32));}
        // (persistent change tracking removed — using simpler approach below)
        // Reset BN running stats to initial values (rm=0, rv=1)
        {u32 rb;
         rb=ctx->tensors[(u32)term_val(bn_rm)].buf_id; memset(metal_pool.bufs[rb].contents,0,C1*4);
         rb=ctx->tensors[(u32)term_val(bn_rv)].buf_id;
         for(u32 j=0;j<C1;j++) ((f32*)metal_pool.bufs[rb].contents)[j]=1.f;
         rb=ctx->tensors[(u32)term_val(bn2_rm)].buf_id; memset(metal_pool.bufs[rb].contents,0,C2*4);
         rb=ctx->tensors[(u32)term_val(bn2_rv)].buf_id;
         for(u32 j=0;j<C2;j++) ((f32*)metal_pool.bufs[rb].contents)[j]=1.f;
        }
        if(step<=2){
            u32 cwb=ctx->tensors[(u32)term_val(cw)].buf_id;
            u32 g0b=ctx->tensors[(u32)term_val(gs[0])].buf_id;
            f32*wd=(f32*)metal_pool.bufs[cwb].contents;
            f32*gd=(f32*)metal_pool.bufs[g0b].contents;
            fprintf(stderr, "  [PRE step %u] cw_buf=%u cw[0]=%.6f  gs0_buf=%u gs0[0]=%.6f\n",
                    step, cwb, wd[0], g0b, gd[0]);
        }
        jit_replay();
        if(step==1){
            jit_flush();
            printf("step 1 grads:\n");
            for(int pi=0;pi<NP;pi++){
                u32 gb=ctx->tensors[(u32)term_val(gs[pi])].buf_id;
                f32*gd=(f32*)metal_pool.bufs[gb].contents;
                f32 gsi=0;for(u32 j=0;j<psz[pi];j++)gsi+=fabsf(gd[j]);
                int n_nan=0;for(u32 j=0;j<psz[pi];j++)if(isnan(gd[j]))n_nan++;
                printf("  gs[%d] |g|=%.4f nan=%d/%d [%.4f,%.4f,%.4f]\n",
                       pi,gsi,n_nan,psz[pi],gd[0],psz[pi]>1?gd[1]:0,psz[pi]>2?gd[2]:0);
            }
        }
        if(step<3||step%50==0||step==n_steps-1){
            jit_flush();
            u32 gb=ctx->tensors[(u32)term_val(gs[0])].buf_id;
            f32*gd=(f32*)metal_pool.bufs[gb].contents;
            f32 gs0=0;for(u32 j=0;j<psz[0];j++)gs0+=fabsf(gd[j]);
            printf("step %3u: |g|=%.2f%s (%.0fms/step)\n",step,gs0,isnan(gd[0])?" NaN!":"",
                   (now_s()-t_train)*1000/step);
        }
    }
    jit_flush();
    printf("\nJIT: %.1fs for %u steps (%.0f ms/step)\n",now_s()-t_train,n_steps-1,(now_s()-t_train)*1000/(n_steps-1));

    // Eval
    printf("\nEvaluating...\n");
    Term test_data=thvm_tensor(ctx,data.test_images,(Shape){.dims={data.n_test,1,28,28},.rank=4});
    u32 ek=ctx->tensor_count;u32 cor=0,tbs=64,tb=data.n_test/tbs;
    for(u32 b2=0;b2<tb;b2++){
        u32 p0e[]={0,0,0,0},s1e[]={1,1},k2e[]={2,2},s2e[]={2,2};
        Term tx=thvm_shrink(ctx,test_data,(u32[]){b2*tbs,(b2+1)*tbs,0,1,0,28,0,28},4);
        Term th=thvm_conv2d(ctx,tx,cw,cb,1,s1e,p0e);th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=batchnorm_forward(ctx,th,bn_g,bn_b,bn_rm,bn_rv,tbs,C1,26,26,0).output;
        th=thvm_maxpool2d(ctx,th,k2e,s2e);
        th=thvm_conv2d(ctx,th,cw2,cb2,1,s1e,p0e);th=thvm_op(ctx,UOP_RELU,th,term_era());
        th=batchnorm_forward(ctx,th,bn2_g,bn2_b,bn2_rm,bn2_rv,tbs,C2,11,11,0).output;
        th=thvm_maxpool2d(ctx,th,k2e,s2e);
        th=thvm_reshape(ctx,th,SHAPE(tbs,ff));
        Term tlo=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,th,lw),
            thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(tbs,10)));
        f32 acc=thvm_eval_accuracy(ctx,tlo,&data.test_labels[b2*tbs],tbs,10);
        cor+=(u32)(acc*(f32)tbs/100.f);thvm_reset(ctx,ek);
    }
    printf("Test accuracy: %.1f%%\n",100.f*(f32)cor/(f32)(tb*tbs));

    free(oh_data);thvm_free(ctx);return 0;
}
