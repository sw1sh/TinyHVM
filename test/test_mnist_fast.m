// test_mnist_fast.m — Match tinygrad speed: Adam + BS=512 + wide 3×3 model
// Same architecture as test_4conv that works, but with Adam instead of SGD.
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
    srand(42);
    MNISTData data = mnist_load("data");
    TinyHVM *ctx = thvm_init("metal");
    u32 BS = 512;

    // Wide 3×3: 16→32→64→64 (same as test_4conv that works)
    Term cw1=mkw(ctx,(Shape){.dims={16,1,3,3},.rank=4},9);   Term cb1=mkz(ctx,16);
    Term cw2=mkw(ctx,(Shape){.dims={32,16,3,3},.rank=4},144); Term cb2=mkz(ctx,32);
    Term cw3=mkw(ctx,(Shape){.dims={64,32,3,3},.rank=4},288); Term cb3=mkz(ctx,64);
    Term cw4=mkw(ctx,(Shape){.dims={64,64,3,3},.rank=4},576); Term cb4=mkz(ctx,64);
    u32 flat_f = 64*5*5; // 1600
    Term lw=mkw(ctx,SHAPE(flat_f,10),flat_f); Term lb=mkz(ctx,10);

    #define NP 10
    Term params[NP]={cw1,cb1,cw2,cb2,cw3,cb3,cw4,cb4,lw,lb};
    u32 psz[]={16*9,16, 32*16*9,32, 64*32*9,64, 64*64*9,64, flat_f*10,10};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,params[i]);

    // Adam optimizer
    Adam opt = adam_init(ctx, 0.001f, NP);
    for(u32 i=0;i<NP;i++) adam_add_param(ctx, &opt, i, (u32)term_val(params[i]), psz[i]);

    Term td = thvm_tensor(ctx, data.train_images, (Shape){.dims={data.n_train,1,28,28},.rank=4});
    u32 n_weights = ctx->tensor_count;

    u32 n_steps = 70;
    printf("=== TinyHVM vs tinygrad: 3×3 wide CNN + Adam ===\n");
    printf("  Conv(1→16,3)→Pool→Conv(16→32,3)→Pool→Conv(32→64,3,p1)→Conv(64→64,3,p1)→Dense(%u,10)\n", flat_f);
    printf("  BS=%u, Adam(lr=0.001), %u steps\n\n", BS, n_steps);

    double t_start = now_s();
    f32 test_acc = 0;

    for (u32 step = 0; step < n_steps; step++) {
      @autoreleasepool {
        u32 bi = rand() % (data.n_train / BS);
        Term x = thvm_shrink(ctx, td, (u32[]){bi*BS,(bi+1)*BS,0,1,0,28,0,28}, 4);
        thvm_set_requires_grad(ctx, x);

        u32 p0[]={0,0,0,0}, s1[]={1,1}, k2[]={2,2}, s2[]={2,2}, p1[]={1,1,1,1};
        Term h = thvm_conv2d(ctx, x, cw1, cb1, 1, s1, p0);
        h = thvm_op(ctx, UOP_RELU, h, term_era());
        h = thvm_maxpool2d(ctx, h, k2, s2);
        h = thvm_conv2d(ctx, h, cw2, cb2, 1, s1, p0);
        h = thvm_op(ctx, UOP_RELU, h, term_era());
        h = thvm_maxpool2d(ctx, h, k2, s2);
        h = thvm_conv2d(ctx, h, cw3, cb3, 1, s1, p1);
        h = thvm_op(ctx, UOP_RELU, h, term_era());
        h = thvm_conv2d(ctx, h, cw4, cb4, 1, s1, p1);
        h = thvm_op(ctx, UOP_RELU, h, term_era());
        h = thvm_reshape(ctx, h, SHAPE(BS, flat_f));
        Term logits = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, lw),
            thvm_expand(ctx, thvm_reshape(ctx, lb, SHAPE(1,10)), SHAPE(BS,10)));
        Term loss = cross_entropy_loss(ctx, logits, &data.train_labels[bi*BS], BS, 10);

        // Backward + Adam as one lazy IC graph
        Term gs[NP];
        for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
            gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
        Term grad_term = thvm_grad_multi(ctx, loss, params, gs, NP);
        u32 grad_ids[NP];
        for(u32 i=0;i<NP;i++) grad_ids[i] = (u32)term_val(gs[i]);
        Term adam_chain = adam_step_lazy(ctx, &opt, grad_ids);
        thvm_reduce(ctx, thvm_app(ctx, grad_term, adam_chain));

        double elapsed = now_s() - t_start;
        if (step < 3 || step % 10 == 9) {
            f32 lv = thvm_to_host(ctx, loss)[0];
            if (step % 10 == 9) {
                Term test_data=thvm_tensor(ctx,data.test_images,(Shape){.dims={data.n_test,1,28,28},.rank=4});
                u32 ek=ctx->tensor_count; u32 cor=0, tbs=512, tb=data.n_test/tbs;
                for(u32 b=0;b<tb;b++){
                    Term tx=thvm_shrink(ctx,test_data,(u32[]){b*tbs,(b+1)*tbs,0,1,0,28,0,28},4);
                    Term th=thvm_maxpool2d(ctx,thvm_op(ctx,UOP_RELU,thvm_conv2d(ctx,tx,cw1,cb1,1,s1,p0),term_era()),k2,s2);
                    th=thvm_maxpool2d(ctx,thvm_op(ctx,UOP_RELU,thvm_conv2d(ctx,th,cw2,cb2,1,s1,p0),term_era()),k2,s2);
                    th=thvm_op(ctx,UOP_RELU,thvm_conv2d(ctx,th,cw3,cb3,1,s1,p1),term_era());
                    th=thvm_op(ctx,UOP_RELU,thvm_conv2d(ctx,th,cw4,cb4,1,s1,p1),term_era());
                    th=thvm_reshape(ctx,th,SHAPE(tbs,flat_f));
                    Term tlo=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,th,lw),
                        thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(tbs,10)));
                    f32 acc=thvm_eval_accuracy(ctx,tlo,&data.test_labels[b*tbs],tbs,10);
                    cor+=(u32)(acc*(f32)tbs/100.f);thvm_reset(ctx,ek);
                }
                test_acc=100.f*(f32)cor/(f32)(tb*tbs);
            }
            printf("  step %2u: loss=%5.2f  acc=%5.1f%%  (%.1fs, %.0fms/step)\n",
                   step, lv, test_acc, elapsed, elapsed*1000/(step+1));
        }

        extern u32 total_dispatches; total_dispatches=0;
        for(u32 i=0;i<NP;i++){u32 pid=(u32)term_val(params[i]);
            if(ctx->tensors[pid].host_ptr){free(ctx->tensors[pid].host_ptr);ctx->tensors[pid].host_ptr=NULL;}}
        thvm_reset(ctx, n_weights);
      }
    }

    double total = now_s() - t_start;
    printf("\n  Final: %.1f%% in %.1fs (%u steps, %.0f ms/step)\n", test_acc, total, n_steps, total*1000/n_steps);
    printf("  tinygrad: 98.27%% in 5.5s (70 steps, BS=512, BatchNorm+Adam)\n");

    adam_free(&opt);
    thvm_free(ctx);
    return 0;
}
