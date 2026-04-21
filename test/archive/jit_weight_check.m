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
    srand(42);MNISTData data=mnist_load("data");TinyHVM*ctx=thvm_init("metal");u32 BS=64;
    Term cw1=mkw(ctx,(Shape){.dims={16,1,3,3},.rank=4},9),cb1=mkz(ctx,16);
    Term cw2=mkw(ctx,(Shape){.dims={32,16,3,3},.rank=4},144),cb2=mkz(ctx,32);
    Term cw3=mkw(ctx,(Shape){.dims={64,32,3,3},.rank=4},288),cb3=mkz(ctx,64);
    Term cw4=mkw(ctx,(Shape){.dims={64,64,3,3},.rank=4},576),cb4=mkz(ctx,64);
    Term bn1_g=mkones(ctx,16),bn1_b=mkz(ctx,16),bn1_rm=mkz(ctx,16),bn1_rv=mkones(ctx,16);
    Term bn2_g=mkones(ctx,32),bn2_b=mkz(ctx,32),bn2_rm=mkz(ctx,32),bn2_rv=mkones(ctx,32);
    u32 ff=64*5*5;Term lw=mkw(ctx,SHAPE(ff,10),ff),lb=mkz(ctx,10);
    f32*bb=calloc(BS*784,4);Term xb=thvm_tensor(ctx,bb,SHAPE(BS,1,28,28));free(bb);
    thvm_set_requires_grad(ctx,xb);
    f32*oh=calloc(BS*10,4);Term ohf=thvm_tensor(ctx,oh,SHAPE(BS,10));free(oh);
    f32 lrv=0.005f;Term lrf=thvm_tensor(ctx,&lrv,SHAPE(1));
    #define NP 14
    Term p[NP]={cw1,cb1,cw2,cb2,cw3,cb3,cw4,cb4,bn1_g,bn1_b,bn2_g,bn2_b,lw,lb};
    u32 psz[NP]={16*9,16,32*16*9,32,64*32*9,64,64*64*9,64,16,16,32,32,ff*10,10};
    for(u32 i=0;i<NP;i++)thvm_set_requires_grad(ctx,p[i]);
    Term gs[NP];for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
        gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(p[i])].view.shape);free(z);}
    u32 np=ctx->tensor_count;
    u32 xid=(u32)term_val(xb),ohid=(u32)term_val(ohf);
    // Write data
    ctx->tensors[xid].backend->buf_write(ctx->tensors[xid].buf_id,data.train_images,BS*784*4);
    oh=calloc(BS*10,4);for(u32 i=0;i<BS;i++)oh[i*10+data.train_labels[i]]=1.f;
    ctx->tensors[ohid].backend->buf_write(ctx->tensors[ohid].buf_id,oh,BS*10*4);
    for(int i=0;i<NP;i++){u32 g=(u32)term_val(gs[i]);memset(metal_pool.bufs[ctx->tensors[g].buf_id].contents,0,psz[i]*4);}
    // Record initial weight
    u32 lw_id=(u32)term_val(lw);
    f32 w_before[4];
    ctx->tensors[lw_id].backend->buf_read(ctx->tensors[lw_id].buf_id,w_before,16);
    printf("lw before: [%.6f,%.6f,%.6f,%.6f]\n",w_before[0],w_before[1],w_before[2],w_before[3]);
    // Build + capture
    u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2},p1[]={1,1,1,1};
    Term h=thvm_conv2d(ctx,xb,cw1,cb1,1,s1,p0);h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult b1=batchnorm_forward(ctx,h,bn1_g,bn1_b,bn1_rm,bn1_rv,BS,16,26,26,1);
    h=b1.output;h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);h=thvm_op(ctx,UOP_RELU,h,term_era());
    BNResult b2=batchnorm_forward(ctx,h,bn2_g,bn2_b,bn2_rm,bn2_rv,BS,32,11,11,1);
    h=b2.output;h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_conv2d(ctx,h,cw3,cb3,1,s1,p1);h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_conv2d(ctx,h,cw4,cb4,1,s1,p1);h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_reshape(ctx,h,SHAPE(BS,ff));
    Term lo=thvm_op(ctx,UOP_ADD,thvm_mm(ctx,h,lw),thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
    Term probs=softmax(ctx,lo,BS,10);f32 eps=1e-7f;
    Term lp=thvm_op(ctx,UOP_LOG,thvm_op(ctx,UOP_MAX,probs,thvm_expand(ctx,thvm_tensor(ctx,&eps,SHAPE(1,1)),SHAPE(BS,10))),term_era());
    Term mk=thvm_op(ctx,UOP_MUL,ohf,lp);Term sa=thvm_sum_axes(ctx,mk,(u32[]){0,1},2);
    Term ng=thvm_op(ctx,UOP_NEG,sa,term_era());f32 ib=1.f/(f32)BS;
    Term loss=thvm_op(ctx,UOP_MUL,ng,thvm_tensor(ctx,&ib,SHAPE(1,1)));
    Term gt=thvm_grad_multi(ctx,loss,p,gs,NP);
    Term sgd=term_era();for(int i=NP-1;i>=0;i--)
        sgd=thvm_app(ctx,thvm_assign(ctx,p[i],thvm_op(ctx,UOP_SUB,p[i],thvm_op(ctx,UOP_MUL,lrf,gs[i]))),sgd);
    Term ba=thvm_app(ctx,b1.assigns,b2.assigns);
    Term step=thvm_app(ctx,gt,thvm_app(ctx,ba,sgd));
    jit_begin_capture(np);
    thvm_reduce(ctx,step);thvm_to_host(ctx,loss);
    jit_end_capture();
    // Check weight after capture
    f32 w_after_capture[4];
    ctx->tensors[lw_id].backend->buf_read(ctx->tensors[lw_id].buf_id,w_after_capture,16);
    printf("lw after capture: [%.6f,%.6f,%.6f,%.6f]\n",w_after_capture[0],w_after_capture[1],w_after_capture[2],w_after_capture[3]);
    int cap_changed = (w_before[0]!=w_after_capture[0]);
    printf("Weight changed after capture: %s\n\n",cap_changed?"YES":"NO");

    // Reset + replay
    extern u32 total_dispatches;total_dispatches=0;
    thvm_reset(ctx,np);
    jit_flush();
    // Same batch data for comparison
    ctx->tensors[xid].backend->buf_write(ctx->tensors[xid].buf_id,data.train_images,BS*784*4);
    memset(oh,0,BS*10*4);for(u32 i=0;i<BS;i++)oh[i*10+data.train_labels[i]]=1.f;
    ctx->tensors[ohid].backend->buf_write(ctx->tensors[ohid].buf_id,oh,BS*10*4);
    for(int i=0;i<NP;i++){u32 g=(u32)term_val(gs[i]);memset(metal_pool.bufs[ctx->tensors[g].buf_id].contents,0,psz[i]*4);}
    jit_replay();
    jit_flush();
    // Check weight after replay
    f32 w_after_replay[4];
    ctx->tensors[lw_id].backend->buf_read(ctx->tensors[lw_id].buf_id,w_after_replay,16);
    printf("lw after replay: [%.6f,%.6f,%.6f,%.6f]\n",w_after_replay[0],w_after_replay[1],w_after_replay[2],w_after_replay[3]);
    int rep_changed = (w_after_capture[0]!=w_after_replay[0]);
    printf("Weight changed after replay: %s\n",rep_changed?"YES":"NO");
    // Check grad accumulator
    u32 g_lw=(u32)term_val(gs[12]); // lw grad
    f32 gv[4]; ctx->tensors[g_lw].backend->buf_read(ctx->tensors[g_lw].buf_id,gv,16);
    printf("grad_lw after replay: [%.6f,%.6f,%.6f,%.6f]\n",gv[0],gv[1],gv[2],gv[3]);

    free(oh);thvm_free(ctx);return 0;
}
