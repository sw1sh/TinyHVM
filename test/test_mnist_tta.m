// test_mnist_tta.m — Time-To-Accuracy benchmark
// Measures wall-clock time to reach 90%, 95%, 97%, 98% test accuracy
// Wide 4-conv, BS=128, cosine LR + warmup + weight decay
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
static void shuf(u32*a,u32 n){for(u32 i=n-1;i>0;i--){u32 j=rand()%(i+1);u32 t=a[i];a[i]=a[j];a[j]=t;}}
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

int main(void) {
    srand(42);
    MNISTData data = mnist_load("data");
    TinyHVM *ctx = thvm_init("metal");
    u32 BS = 128;
    u32 n_batches = data.n_train / BS;

    Term cw1=mkw(ctx,(Shape){.dims={16,1,3,3},.rank=4},9),cb1=mkz(ctx,16);
    Term cw2=mkw(ctx,(Shape){.dims={32,16,3,3},.rank=4},144),cb2=mkz(ctx,32);
    Term cw3=mkw(ctx,(Shape){.dims={64,32,3,3},.rank=4},288),cb3=mkz(ctx,64);
    Term cw4=mkw(ctx,(Shape){.dims={64,64,3,3},.rank=4},576),cb4=mkz(ctx,64);
    u32 ff=64*5*5;Term lw=mkw(ctx,SHAPE(ff,10),ff),lb=mkz(ctx,10);
    #define NP 10
    Term p[NP]={cw1,cb1,cw2,cb2,cw3,cb3,cw4,cb4,lw,lb};
    u32 psz[]={16*9,16,32*16*9,32,64*32*9,64,64*64*9,64,ff*10,10};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,p[i]);
    Term td=thvm_tensor(ctx,data.train_images,(Shape){.dims={data.n_train,1,28,28},.rank=4});
    u32 nw=ctx->tensor_count;

    u32 *bo = malloc(n_batches*sizeof(u32));
    for(u32 i=0;i<n_batches;i++) bo[i]=i;

    u32 max_epochs = 10;
    u32 total_steps = max_epochs * n_batches;
    f32 lr_max = 0.003f, wd = 0.0001f;
    u32 warmup = 100;

    // Milestones
    f32 targets[] = {90.0f, 95.0f, 97.0f, 98.0f};
    double target_times[4] = {-1,-1,-1,-1};
    u32 target_steps[4] = {0,0,0,0};
    u32 n_targets = 4;

    printf("=== TinyHVM Time-To-Accuracy (MNIST, 76K params, BS=%u) ===\n\n", BS);
    double t_start = now_s();
    u32 step = 0;

    for (u32 epoch = 0; epoch < max_epochs; epoch++) {
        shuf(bo, n_batches);
        for (u32 bi_idx = 0; bi_idx < n_batches; bi_idx++) {
          @autoreleasepool {
            f32 lr = (step < warmup) ? lr_max*(f32)step/(f32)warmup :
                lr_max * 0.5f * (1.f + cosf(3.14159f * (f32)(step-warmup)/(f32)(total_steps-warmup)));
            u32 bi = bo[bi_idx];
            Term x=thvm_shrink(ctx,td,(u32[]){bi*BS,(bi+1)*BS,0,1,0,28,0,28},4);
            thvm_set_requires_grad(ctx,x);
            u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2},p1[]={1,1,1,1};
            Term h=thvm_maxpool2d(ctx,thvm_op(ctx,UOP_RELU,thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0),term_era()),k2,s2);
            h=thvm_maxpool2d(ctx,thvm_op(ctx,UOP_RELU,thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0),term_era()),k2,s2);
            h=thvm_op(ctx,UOP_RELU,thvm_conv2d(ctx,h,cw3,cb3,1,s1,p1),term_era());
            h=thvm_op(ctx,UOP_RELU,thvm_conv2d(ctx,h,cw4,cb4,1,s1,p1),term_era());
            h=thvm_reshape(ctx,h,SHAPE(BS,ff));
            Term lo=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
            Term loss=cross_entropy_loss(ctx,lo,&data.train_labels[bi*BS],BS,10);
            Term gs[NP];for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(p[i])].view.shape);free(z);}
            Term gt=thvm_grad_multi(ctx,loss,p,gs,NP);
            Term lr_t=thvm_tensor(ctx,&lr,SHAPE(1));
            Term sgd=term_era();
            for(int i=NP-1;i>=0;i--){
                int is_w=(i%2==0);
                Term upd;
                if(is_w&&wd>0){f32 wv=wd;Term wt=thvm_tensor(ctx,&wv,SHAPE(1));
                    upd=thvm_op(ctx,UOP_SUB,p[i],thvm_op(ctx,UOP_MUL,lr_t,thvm_op(ctx,UOP_ADD,gs[i],thvm_op(ctx,UOP_MUL,wt,p[i]))));}
                else upd=thvm_op(ctx,UOP_SUB,p[i],thvm_op(ctx,UOP_MUL,lr_t,gs[i]));
                sgd=thvm_app(ctx,thvm_assign(ctx,p[i],upd),sgd);
            }
            thvm_reduce(ctx,thvm_app(ctx,gt,sgd));
            extern u32 total_dispatches;total_dispatches=0;
            for(u32 i=0;i<NP;i++){u32 pid=(u32)term_val(p[i]);if(ctx->tensors[pid].host_ptr){free(ctx->tensors[pid].host_ptr);ctx->tensors[pid].host_ptr=NULL;}}
            thvm_reset(ctx,nw);
            step++;
          }
        }

        // Eval after each epoch
        Term test_data=thvm_tensor(ctx,data.test_images,(Shape){.dims={data.n_test,1,28,28},.rank=4});
        u32 ek=ctx->tensor_count;
        u32 correct=0,tbs=128,tb=data.n_test/tbs;
        for(u32 b=0;b<tb;b++){
            Term x=thvm_shrink(ctx,test_data,(u32[]){b*tbs,(b+1)*tbs,0,1,0,28,0,28},4);
            u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2},p1[]={1,1,1,1};
            Term h=thvm_maxpool2d(ctx,thvm_op(ctx,UOP_RELU,thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0),term_era()),k2,s2);
            h=thvm_maxpool2d(ctx,thvm_op(ctx,UOP_RELU,thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0),term_era()),k2,s2);
            h=thvm_op(ctx,UOP_RELU,thvm_conv2d(ctx,h,cw3,cb3,1,s1,p1),term_era());
            h=thvm_op(ctx,UOP_RELU,thvm_conv2d(ctx,h,cw4,cb4,1,s1,p1),term_era());
            h=thvm_reshape(ctx,h,SHAPE(tbs,ff));
            Term lo=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(tbs,10)));
            f32 acc=thvm_eval_accuracy(ctx,lo,&data.test_labels[b*tbs],tbs,10);
            correct+=(u32)(acc*(f32)tbs/100.f);
            thvm_reset(ctx,ek);
        }
        f32 acc=100.f*(f32)correct/(f32)(tb*tbs);
        double elapsed = now_s() - t_start;

        // Check milestones
        for(u32 t=0;t<n_targets;t++){
            if(target_times[t]<0 && acc>=targets[t]){
                target_times[t]=elapsed;
                target_steps[t]=step;
                printf("  *** %.0f%% reached at %.1fs (epoch %u, step %u) ***\n",targets[t],elapsed,epoch,step);
            }
        }
        printf("  Epoch %u: %.1f%% (%.1fs elapsed, %u steps)\n\n", epoch, acc, elapsed, step);

        // Early stop if all targets hit
        int all=1; for(u32 t=0;t<n_targets;t++) if(target_times[t]<0) all=0;
        if(all) break;
    }

    printf("=== Time-To-Accuracy Summary ===\n");
    printf("  Model: 4-conv (16→32→64→64), 76K params, BS=%u\n", BS);
    printf("  Hardware: M3 Max, Metal GPU\n\n");
    for(u32 t=0;t<n_targets;t++){
        if(target_times[t]>=0)
            printf("  %.0f%%: %5.1fs  (%u steps, %.1f ms/step)\n",
                   targets[t],target_times[t],target_steps[t],target_times[t]*1000/target_steps[t]);
        else
            printf("  %.0f%%: not reached\n",targets[t]);
    }
    printf("\n");
    free(bo);
    thvm_free(ctx);
    return 0;
}
