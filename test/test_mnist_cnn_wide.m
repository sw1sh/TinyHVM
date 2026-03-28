// test_mnist_cnn_wide.m — Wide 4-conv CNN, push MNIST accuracy
// Conv(1,16,3)→ReLU→Pool→Conv(16,32,3)→ReLU→Pool
//   → Conv(32,64,3,p=1)→ReLU→Conv(64,64,3,p=1)→ReLU→Flatten→Linear(64*5*5,10)
// 5 epochs, BS=64, cosine LR decay
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

static Term make_w(TinyHVM *c, Shape s, u32 fi) {
    u32 n=1; for(u32 i=0;i<s.rank;i++) n*=s.dims[i];
    f32 *d=malloc(n*4); f32 b=sqrtf(2.0f/(f32)fi);
    for(u32 i=0;i<n;i++) d[i]=b*((f32)rand()/(f32)RAND_MAX*2-1)*0.5f;
    Term t=thvm_tensor(c,d,s); free(d); return t;
}
static Term make_z(TinyHVM *c, u32 n) { f32*z=calloc(n,4); Term t=thvm_tensor(c,z,SHAPE(n)); free(z); return t; }

static void shuffle(u32 *a, u32 n) {
    for (u32 i=n-1;i>0;i--) { u32 j=rand()%(i+1); u32 t=a[i]; a[i]=a[j]; a[j]=t; }
}

int main(void) {
    srand(42);
    MNISTData data = mnist_load("data");
    TinyHVM *ctx = thvm_init("metal");
    u32 BS = 64;
    u32 n_batches = data.n_train / BS;

    // Wide architecture: 16→32→64→64
    Term cw1=make_w(ctx,(Shape){.dims={16,1,3,3},.rank=4},9);
    Term cb1=make_z(ctx,16);
    Term cw2=make_w(ctx,(Shape){.dims={32,16,3,3},.rank=4},144);
    Term cb2=make_z(ctx,32);
    Term cw3=make_w(ctx,(Shape){.dims={64,32,3,3},.rank=4},288);
    Term cb3=make_z(ctx,64);
    Term cw4=make_w(ctx,(Shape){.dims={64,64,3,3},.rank=4},576);
    Term cb4=make_z(ctx,64);
    u32 flat_f=64*5*5;
    Term lw=make_w(ctx,SHAPE(flat_f,10),flat_f);
    Term lb=make_z(ctx,10);

    #define NP 10
    Term params[NP]={cw1,cb1,cw2,cb2,cw3,cb3,cw4,cb4,lw,lb};
    u32 psz[]={16*9,16, 32*16*9,32, 64*32*9,64, 64*64*9,64, flat_f*10,10};
    u32 total_params=0; for(int i=0;i<NP;i++) total_params+=psz[i];
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,params[i]);
    Term train_data=thvm_tensor(ctx,data.train_images,(Shape){.dims={data.n_train,1,28,28},.rank=4});
    u32 n_weights=ctx->tensor_count;

    u32 *batch_order = malloc(n_batches * sizeof(u32));
    for (u32 i=0;i<n_batches;i++) batch_order[i]=i;

    u32 n_epochs = 5;
    f32 lr_init = 0.001f;

    printf("=== MNIST Wide 4-Conv CNN ===\n");
    printf("  Channels: 16→32→64→64, Dense: %u→10\n", flat_f);
    printf("  Params: %u, Epochs: %u, BS: %u\n\n", total_params, n_epochs, BS);

    struct timespec t_start; clock_gettime(CLOCK_MONOTONIC, &t_start);
    u32 total_steps = n_epochs * n_batches;
    u32 step = 0;

    for (u32 epoch = 0; epoch < n_epochs; epoch++) {
        shuffle(batch_order, n_batches);

        for (u32 bi_idx = 0; bi_idx < n_batches; bi_idx++) {
          @autoreleasepool {
            // Cosine LR decay
            f32 progress = (f32)step / (f32)total_steps;
            f32 lr = lr_init * 0.5f * (1.0f + cosf(3.14159f * progress));

            u32 bi = batch_order[bi_idx];
            Term x=thvm_shrink(ctx,train_data,(u32[]){bi*BS,(bi+1)*BS,0,1,0,28,0,28},4);
            thvm_set_requires_grad(ctx,x);
            u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2},p1[]={1,1,1,1};
            Term h=thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0);
            h=thvm_op(ctx,UOP_RELU,h,term_era());
            h=thvm_maxpool2d(ctx,h,k2,s2);
            h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);
            h=thvm_op(ctx,UOP_RELU,h,term_era());
            h=thvm_maxpool2d(ctx,h,k2,s2);
            h=thvm_conv2d(ctx,h,cw3,cb3,1,s1,p1);
            h=thvm_op(ctx,UOP_RELU,h,term_era());
            h=thvm_conv2d(ctx,h,cw4,cb4,1,s1,p1);
            h=thvm_op(ctx,UOP_RELU,h,term_era());
            h=thvm_reshape(ctx,h,SHAPE(BS,flat_f));
            Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
                thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
            Term loss=cross_entropy_loss(ctx,logits,&data.train_labels[bi*BS],BS,10);

            Term gs[NP];
            for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
                gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
            Term grad_term=thvm_grad_multi(ctx,loss,params,gs,NP);
            Term lr_t=thvm_tensor(ctx,&lr,SHAPE(1));
            Term sgd=term_era();
            for(int i=NP-1;i>=0;i--)
                sgd=thvm_app(ctx,thvm_assign(ctx,params[i],thvm_op(ctx,UOP_SUB,params[i],
                    thvm_op(ctx,UOP_MUL,lr_t,gs[i]))),sgd);
            thvm_reduce(ctx,thvm_app(ctx,grad_term,sgd));

            if (step < 3 || (step % 200 == 0 && step > 0) || bi_idx == n_batches-1) {
                f32 lv=thvm_to_host(ctx,loss)[0];
                printf("  [%u/%u] step %4u: loss=%.4f lr=%.6f\n",
                       epoch, n_epochs, step, lv, lr);
            }
            extern u32 total_dispatches; total_dispatches=0;
            for(u32 i=0;i<NP;i++){u32 pid=(u32)term_val(params[i]);
                if(ctx->tensors[pid].host_ptr){free(ctx->tensors[pid].host_ptr);ctx->tensors[pid].host_ptr=NULL;}}
            thvm_reset(ctx,n_weights);
            step++;
          }
        }

        // Eval
        Term test_data=thvm_tensor(ctx,data.test_images,(Shape){.dims={data.n_test,1,28,28},.rank=4});
        u32 ek=ctx->tensor_count;
        u32 correct=0, tbs=64, tb=data.n_test/tbs;
        for(u32 b=0;b<tb;b++){
            Term x=thvm_shrink(ctx,test_data,(u32[]){b*tbs,(b+1)*tbs,0,1,0,28,0,28},4);
            u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2},p1[]={1,1,1,1};
            Term h=thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0);
            h=thvm_op(ctx,UOP_RELU,h,term_era());
            h=thvm_maxpool2d(ctx,h,k2,s2);
            h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);
            h=thvm_op(ctx,UOP_RELU,h,term_era());
            h=thvm_maxpool2d(ctx,h,k2,s2);
            h=thvm_conv2d(ctx,h,cw3,cb3,1,s1,p1);
            h=thvm_op(ctx,UOP_RELU,h,term_era());
            h=thvm_conv2d(ctx,h,cw4,cb4,1,s1,p1);
            h=thvm_op(ctx,UOP_RELU,h,term_era());
            h=thvm_reshape(ctx,h,SHAPE(tbs,flat_f));
            Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
                thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(tbs,10)));
            f32 acc=thvm_eval_accuracy(ctx,logits,&data.test_labels[b*tbs],tbs,10);
            correct+=(u32)(acc*(f32)tbs/100.0f);
            thvm_reset(ctx,ek);
        }
        printf("  ══ Epoch %u: %.1f%% (%u/%u) ══\n\n", epoch,
               100.0f*(f32)correct/(f32)(tb*tbs), correct, tb*tbs);
    }

    struct timespec t_end; clock_gettime(CLOCK_MONOTONIC, &t_end);
    f32 total_s = (f32)(t_end.tv_sec-t_start.tv_sec)+(f32)(t_end.tv_nsec-t_start.tv_nsec)/1e9f;
    printf("Total: %.0fs (%.1f ms/step, %u steps)\n", total_s, total_s*1000/(f32)total_steps, total_steps);
    free(batch_order);
    thvm_free(ctx);
    return 0;
}
