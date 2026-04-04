// test_mnist_cnn.m — Push MNIST CNN to its limits
// 4-conv architecture: Conv(1,8,3)→ReLU→Pool→Conv(8,16,3)→ReLU→Pool
//   → Conv(16,32,3,p=1)→ReLU→Conv(32,32,3,p=1)→ReLU→Flatten→Linear(800,10)
// Full training: 3 epochs, BS=64, lr decay, shuffled batches
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
    f32 *d=malloc(n*4); f32 b=sqrtf(2.0f/(f32)fi); // He init
    for(u32 i=0;i<n;i++) d[i]=b*((f32)rand()/(f32)RAND_MAX*2-1)*0.5f;
    Term t=thvm_tensor(c,d,s); free(d); return t;
}
static Term make_z(TinyHVM *c, u32 n) { f32*z=calloc(n,4); Term t=thvm_tensor(c,z,SHAPE(n)); free(z); return t; }

// Fisher-Yates shuffle for batch indices
static void shuffle(u32 *arr, u32 n) {
    for (u32 i = n - 1; i > 0; i--) {
        u32 j = rand() % (i + 1);
        u32 tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

int main(void) {
    srand(42);
    MNISTData data = mnist_load("data");
    TinyHVM *ctx = thvm_init("metal");
    u32 BS = 256;
    u32 n_batches = data.n_train / BS;

    // Architecture: 4 conv layers + 1 dense
    Term cw1=make_w(ctx,(Shape){.dims={8,1,3,3},.rank=4},9);
    Term cb1=make_z(ctx,8);
    Term cw2=make_w(ctx,(Shape){.dims={16,8,3,3},.rank=4},72);
    Term cb2=make_z(ctx,16);
    Term cw3=make_w(ctx,(Shape){.dims={32,16,3,3},.rank=4},144);
    Term cb3=make_z(ctx,32);
    Term cw4=make_w(ctx,(Shape){.dims={32,32,3,3},.rank=4},288);
    Term cb4=make_z(ctx,32);
    u32 flat_f=32*5*5;
    Term lw=make_w(ctx,SHAPE(flat_f,10),flat_f);
    Term lb=make_z(ctx,10);

    #define NP 10
    Term params[NP]={cw1,cb1,cw2,cb2,cw3,cb3,cw4,cb4,lw,lb};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,params[i]);
    Term train_data=thvm_tensor(ctx,data.train_images,(Shape){.dims={data.n_train,1,28,28},.rank=4});

    // Adam optimizer
    u32 psz[]={72,8,1152,16,4608,32,9216,32,flat_f*10,10};
    Adam opt = adam_init(ctx, 0.001f, NP);
    for (u32 i = 0; i < NP; i++)
        adam_add_param(ctx, &opt, i, (u32)term_val(params[i]), psz[i]);

    u32 n_weights=ctx->tensor_count;

    u32 *batch_order = malloc(n_batches * sizeof(u32));
    for (u32 i = 0; i < n_batches; i++) batch_order[i] = i;

    u32 n_epochs = 3;
    u32 total_steps = n_epochs * n_batches;

    printf("=== MNIST 4-Conv CNN ===\n");
    printf("  Architecture: Conv(1→8,3)→Pool→Conv(8→16,3)→Pool→Conv(16→32,3,p1)→Conv(32→32,3,p1)→Dense(800→10)\n");
    printf("  Training: %u epochs × %u batches (BS=%u), Adam lr=%.4f\n", n_epochs, n_batches, BS, opt.lr);
    printf("  Params: %u (cw: 72+1152+4608+9216, cb: 8+16+32+32, dense: %u+10)\n\n", 72+1152+4608+9216+8+16+32+32+flat_f*10+10, flat_f*10);

    struct timespec t_start; clock_gettime(CLOCK_MONOTONIC, &t_start);
    u32 step = 0;

    for (u32 epoch = 0; epoch < n_epochs; epoch++) {
        shuffle(batch_order, n_batches);
        f32 epoch_loss = 0;
        u32 epoch_steps = 0;

        for (u32 bi_idx = 0; bi_idx < n_batches; bi_idx++) {
          @autoreleasepool {
            u32 bi = batch_order[bi_idx];
            Term x=thvm_shrink(ctx,train_data,(u32[]){bi*BS,(bi+1)*BS,0,1,0,28,0,28},4);
            thvm_set_requires_grad(ctx,x);
            u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
            u32 p1[]={1,1,1,1};
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
            Term logits=thvm_op(ctx,UOP_ADD,thvm_mm(ctx,h,lw),
                thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
            Term loss=cross_entropy_loss(ctx,logits,&data.train_labels[bi*BS],BS,10);

            Term gs[NP];
            for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
                gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
            Term grad_term=thvm_grad_multi(ctx,loss,params,gs,NP);
            // Reduce grads first (populate gs[]), then Adam
            thvm_reduce(ctx,grad_term);
            u32 grad_ids[NP];
            for(int i=0;i<NP;i++) grad_ids[i]=(u32)term_val(gs[i]);
            Term adam_chain=adam_step_lazy(ctx,&opt,grad_ids);
            thvm_reduce(ctx,adam_chain);

            // Only read loss for reporting (avoid GPU sync on every step)
            if (step < 5 || step % 100 == 0 || bi_idx == n_batches - 1) {
                f32 lv=thvm_to_host(ctx,loss)[0];
                epoch_loss += lv;
                epoch_steps++;
                if (step < 5 || step % 200 == 0)
                    printf("  epoch %u step %4u/%u: loss=%.4f\n",
                           epoch, bi_idx, n_batches, lv);
            }

            extern u32 total_dispatches;
            total_dispatches=0;
            for(u32 i=0;i<NP;i++){u32 pid=(u32)term_val(params[i]);
                if(ctx->tensors[pid].host_ptr){free(ctx->tensors[pid].host_ptr);ctx->tensors[pid].host_ptr=NULL;}}
            thvm_reset(ctx,n_weights);
            step++;
          }
        }
        printf("  epoch %u done: avg_loss=%.4f (sampled %u steps)\n", epoch, epoch_loss/epoch_steps, epoch_steps);

        // Evaluate after each epoch
        Term test_data=thvm_tensor(ctx,data.test_images,(Shape){.dims={data.n_test,1,28,28},.rank=4});
        u32 ek=ctx->tensor_count;
        u32 correct=0, tbs=64, tb=data.n_test/tbs;
        for(u32 b=0;b<tb;b++){
            Term x=thvm_shrink(ctx,test_data,(u32[]){b*tbs,(b+1)*tbs,0,1,0,28,0,28},4);
            u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
            u32 p1[]={1,1,1,1};
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
            Term logits=thvm_op(ctx,UOP_ADD,thvm_mm(ctx,h,lw),
                thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(tbs,10)));
            f32 acc=thvm_eval_accuracy(ctx,logits,&data.test_labels[b*tbs],tbs,10);
            correct+=(u32)(acc*(f32)tbs/100.0f);
            thvm_reset(ctx,ek);
        }
        printf("  → Test accuracy: %.1f%% (%u/%u)\n\n", 100.0f*(f32)correct/(f32)(tb*tbs), correct, tb*tbs);
    }

    struct timespec t_end; clock_gettime(CLOCK_MONOTONIC, &t_end);
    f32 total_s = (f32)(t_end.tv_sec-t_start.tv_sec)+(f32)(t_end.tv_nsec-t_start.tv_nsec)/1e9f;
    printf("Total: %.1fs (%.1f ms/step avg)\n", total_s, total_s*1000.0f/(f32)total_steps);

    free(batch_order);
    thvm_free(ctx);
    return 0;
}
