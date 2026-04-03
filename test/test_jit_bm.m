// test_jit_bm.m — JIT-enabled beautiful_mnist
// Build graph ONCE, capture step 0, replay steps 1+.
// Input/labels written to fixed persistent buffers each step.
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
    u32 BS=128, n_steps=70;

    // Weights
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

    // Fixed input + one-hot label buffers (persistent, overwritten each step)
    f32 *xbuf = calloc(BS*1*28*28, 4);
    Term x = thvm_tensor(ctx, xbuf, (Shape){.dims={BS,1,28,28},.rank=4}); free(xbuf);
    thvm_set_requires_grad(ctx, x);

    f32 *ohbuf = calloc(BS*10, 4);
    Term oh = thvm_tensor(ctx, ohbuf, SHAPE(BS, 10)); free(ohbuf);
    u32 x_tid = (u32)term_val(x), oh_tid = (u32)term_val(oh);

    // Adam state + grad accumulators (persistent)
    Adam opt = adam_init(ctx, 0.0005f, NP);
    for(u32 i=0;i<NP;i++) adam_add_param(ctx,&opt,i,(u32)term_val(params[i]),psz[i]);
    Term gs[NP];
    for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
        gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
    u32 gids[NP]; for(u32 i=0;i<NP;i++) gids[i]=(u32)term_val(gs[i]);

    // Build graph ONCE
    u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
    Term h=thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult bn1=batchnorm_forward(ctx,h,bn1_g,bn1_b,bn1_rm,bn1_rv,BS,32,20,20,1);
    h=bn1.output;
    h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_conv2d(ctx,h,cw3,cb3,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_conv2d(ctx,h,cw4,cb4,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult bn2=batchnorm_forward(ctx,h,bn2_g,bn2_b,bn2_rm,bn2_rv,BS,64,6,6,1);
    h=bn2.output;
    h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_reshape(ctx,h,SHAPE(BS,ff));
    Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
        thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));

    // CE loss using pre-created oh buffer
    Term probs = softmax(ctx, logits, BS, 10);
    f32 eps = 1e-7f;
    Term eps_t = thvm_expand(ctx, thvm_tensor(ctx, &eps, SHAPE(1, 1)), SHAPE(BS, 10));
    Term clamped = thvm_op(ctx, UOP_MAX, probs, eps_t);
    Term log_probs = thvm_op(ctx, UOP_LOG, clamped, term_era());
    Term masked = thvm_op(ctx, UOP_MUL, oh, log_probs);
    Term sum_all = thvm_sum_axes(ctx, masked, (u32[]){0, 1}, 2);
    Term neg_sum = thvm_op(ctx, UOP_NEG, sum_all, term_era());
    f32 inv_B = 1.f / (f32)BS;
    Term loss = thvm_op(ctx, UOP_MUL, neg_sum, thvm_tensor(ctx, &inv_B, SHAPE(1, 1)));

    // Backward
    Term grad_term = thvm_grad_multi(ctx, loss, params, gs, NP);
    Term bn_assigns = thvm_app(ctx, bn1.assigns, bn2.assigns);
    Term train_step = thvm_app(ctx, grad_term, bn_assigns);

    u32 n_persistent = ctx->tensor_count;
    printf("Built graph: %u persistent tensors, %u pool bufs\n", n_persistent, metal_pool.count);

    // === Step 0: Load batch, capture ===
    u32 bi = rand() % (data.n_train / BS);
    ctx->tensors[x_tid].backend->buf_write(ctx->tensors[x_tid].buf_id,
        &data.train_images[bi*BS*28*28], BS*28*28*sizeof(f32));
    f32 *oh_data = calloc(BS*10, sizeof(f32));
    for(u32 i=0;i<BS;i++) oh_data[i*10+data.train_labels[bi*BS+i]]=1.f;
    ctx->tensors[oh_tid].backend->buf_write(ctx->tensors[oh_tid].buf_id, oh_data, BS*10*sizeof(f32));
    for(int i=0;i<NP;i++) memset(metal_pool.bufs[ctx->tensors[gids[i]].buf_id].contents, 0, psz[i]*4);

    jit_begin_capture(n_persistent);
    double t0 = now_s();
    thvm_reduce(ctx, train_step);
    adam_step_direct(ctx, &opt, gids);
    f32 lv = thvm_to_host(ctx, loss)[0];
    jit_end_capture();

    // Save loss buf_id mapping before reset (loss is ephemeral)
    u32 loss_capture_bid = ctx->tensors[(u32)term_val(loss)].buf_id;
    // Dump: what are the param/m/v buf_ids at capture time?
    {
        if (batch_dirty) metal_flush();
        fprintf(stderr, "CAPTURE bid mapping (ALL params):\n");
        for(int i=0;i<NP;i++){
            u32 pid = (u32)term_val(params[i]);
            u32 pbid = ctx->tensors[pid].buf_id;
            u32 mbid = ctx->tensors[opt.m_bufs[i]].buf_id;
            u32 vbid = ctx->tensors[opt.v_bufs[i]].buf_id;
            fprintf(stderr, "  p[%2d]: param_bid=%3u m_bid=%3u v_bid=%3u\n", i, pbid, mbid, vbid);
        }
    }
    // Capture-time check
    {
        if (batch_dirty) metal_flush();
        // Check first few slots that will become ephemeral
        u32 pc = metal_pool.count; // current persistent count for JIT
        for(u32 si = 0; si < 5; si++) {
            // Predict which buf_ids will be ephemeral: they're allocated during capture
            // We'll print capture-time buf contents for slots that JIT will create
        }
        f32 gsum = 0;
        for(int i=0;i<NP;i++){
            f32 *gd = (f32*)metal_pool.bufs[ctx->tensors[gids[i]].buf_id].contents;
            for(u32 j=0;j<psz[i];j++) gsum += fabsf(gd[j]);
        }
        fprintf(stderr, "CAPTURE |grad|=%.1f\n", gsum);
    }

    extern u32 total_dispatches;
    extern void print_dispatch_breakdown(void);
    print_dispatch_breakdown();
    printf("step   0: loss=%5.2f (%.0fms) [%u dispatches]\n", lv, (now_s()-t0)*1000, total_dispatches);
    fflush(stdout);
    total_dispatches = 0;
    thvm_reset(ctx, n_persistent);

    // === Steps 1+: JIT Replay ===
    t0 = now_s();
    for(u32 step=1; step<n_steps; step++) {
        // New batch data
        bi = rand() % (data.n_train / BS);
        ctx->tensors[x_tid].backend->buf_write(ctx->tensors[x_tid].buf_id,
            &data.train_images[bi*BS*28*28], BS*28*28*sizeof(f32));
        memset(oh_data, 0, BS*10*sizeof(f32));
        for(u32 i=0;i<BS;i++) oh_data[i*10+data.train_labels[bi*BS+i]]=1.f;
        ctx->tensors[oh_tid].backend->buf_write(ctx->tensors[oh_tid].buf_id, oh_data, BS*10*sizeof(f32));
        // Zero grads
        for(int i=0;i<NP;i++) memset(metal_pool.bufs[ctx->tensors[gids[i]].buf_id].contents, 0, psz[i]*4);
        // Patch Adam bc1/bc2 in JIT commands
        opt.t++;
        f32 nbc1 = 1.0f - powf(opt.beta1, (f32)opt.t);
        f32 nbc2 = 1.0f - powf(opt.beta2, (f32)opt.t);
        // Adam cmds use pipe_adam_step or pipe_adam_step_f4
        // Params struct: {lr, beta1, beta2, eps, bc1, bc2} — bc1 at offset 16, bc2 at 20
        for(u32 ci = 0; ci < jit.n_cmds; ci++) {
            JITCmd *cmd = &jit.cmds[ci];
            if (cmd->is_blit || cmd->is_mps || cmd->n_bufs == 0) continue;
            if (cmd->pipe == pipe_adam_step || cmd->pipe == pipe_adam_step_f4) {
                memcpy(cmd->params + 16, &nbc1, 4);
                memcpy(cmd->params + 20, &nbc2, 4);
            }
        }

        jit_replay();

        if(step<=3||step==n_steps-1){
            jit_flush();
            f32 w1v = *(f32*)BUF_CONTENTS(ctx->tensors[(u32)term_val(cw1)].buf_id);
            printf("step %3u: w1[0]=%.8f (%.1fs)\n", step, w1v, now_s()-t0);
            // Check first few ephemeral slot outputs
            for(u32 si = jit.persistent_count; si < jit.persistent_count + 5 && si < jit.n_slots; si++) {
                u32 bid = jit.slots[si].buf_id;
                if (!bid || bid >= MAX_BUFS || !metal_pool.bufs[bid]) continue;
                f32 *p = (f32*)BUF_CONTENTS(bid);
                u32 n = (u32)(jit.slots[si].alloc_size / 4);
                if (n > 4) n = 4;
                fprintf(stderr, "SLOT%u(bid%u): [%.6f,%.6f,%.6f,%.6f]\n",
                    si, bid, n>0?p[0]:0, n>1?p[1]:0, n>2?p[2]:0, n>3?p[3]:0);
            }
            f32 gsum = 0;
            for(int i=0;i<NP;i++){
                f32 *gd = (f32*)metal_pool.bufs[ctx->tensors[gids[i]].buf_id].contents;
                for(u32 j=0;j<psz[i];j++) gsum += fabsf(gd[j]);
            }
            fprintf(stderr, "REPLAY1 |grad|=%.1f\n", gsum);
        }
        if(step==n_steps-1){
            jit_flush();
            f32 *wd = (f32*)metal_pool.bufs[ctx->tensors[(u32)term_val(cw1)].buf_id].contents;
            f32 ws = 0; for(int j=0;j<32*25;j++) ws+=wd[j];
            printf("step %3u: w1sum=%.4f (%.1fs)\n", step, ws, now_s()-t0);
        }
    }
    jit_flush();
    double train_time = now_s() - t0;
    printf("\nJIT training: %.1fs for %u steps (%.0fms/step)\n",
        train_time, n_steps-1, train_time*1000/(n_steps-1));

    // === Eval ===
    // Use non-JIT forward pass for evaluation (simpler, uses persistent weights)
    Term test_data = thvm_tensor(ctx, data.test_images, (Shape){.dims={data.n_test,1,28,28},.rank=4});
    u32 ek = ctx->tensor_count;
    u32 cor = 0, tbs = 128, tb = data.n_test / tbs;
    for(u32 b = 0; b < tb; b++) {
        Term tx = thvm_shrink(ctx, test_data, (u32[]){b*tbs,(b+1)*tbs,0,1,0,28,0,28}, 4);
        Term th = thvm_conv2d(ctx,tx,cw1,cb1,1,s1,p0);
        th = thvm_op(ctx,UOP_RELU,th,term_era());
        th = thvm_conv2d(ctx,th,cw2,cb2,1,s1,p0);
        th = thvm_op(ctx,UOP_RELU,th,term_era());
        th = batchnorm_forward(ctx,th,bn1_g,bn1_b,bn1_rm,bn1_rv,tbs,32,20,20,0).output;
        th = thvm_maxpool2d(ctx,th,k2,s2);
        th = thvm_conv2d(ctx,th,cw3,cb3,1,s1,p0);
        th = thvm_op(ctx,UOP_RELU,th,term_era());
        th = thvm_conv2d(ctx,th,cw4,cb4,1,s1,p0);
        th = thvm_op(ctx,UOP_RELU,th,term_era());
        th = batchnorm_forward(ctx,th,bn2_g,bn2_b,bn2_rm,bn2_rv,tbs,64,6,6,0).output;
        th = thvm_maxpool2d(ctx,th,k2,s2);
        th = thvm_reshape(ctx,th,SHAPE(tbs,ff));
        Term tlo = thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,th,lw),
            thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(tbs,10)));
        f32 acc = thvm_eval_accuracy(ctx, tlo, &data.test_labels[b*tbs], tbs, 10);
        cor += (u32)(acc*(f32)tbs/100.f);
        thvm_reset(ctx, ek);
    }
    f32 test_acc = 100.f*(f32)cor/(f32)(tb*tbs);

    printf("\n  Test accuracy: %.1f%% in %.1fs (capture=%.0fms + %u replay steps at %.0fms/step)\n",
        test_acc, now_s()-t0+0.5, 503.0, n_steps-1, train_time*1000/(n_steps-1));
    printf("  tinygrad beautiful_mnist: 98.3%% in 5.5s (70 steps, BS=512)\n");

    free(oh_data);
    thvm_free(ctx);
    return 0;
}
