// bench_scaling.m — Find GPU vs CPU crossover for GEMM and MLP training
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
#include <Accelerate/Accelerate.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("=== GEMM Scaling: Metal GPU vs Accelerate CPU ===\n\n");
    printf("  %-6s  %10s  %10s  %8s\n", "N", "Metal(ms)", "CPU(ms)", "Winner");
    printf("  %-6s  %10s  %10s  %8s\n", "------", "----------", "----------", "--------");

    int sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
    for (int si = 0; si < 7; si++) {
        int N = sizes[si];
        float *A = malloc(N*N*4), *B = malloc(N*N*4), *C = malloc(N*N*4);
        for (int i = 0; i < N*N; i++) { A[i] = 0.01f*(rand()%100); B[i] = 0.01f*(rand()%100); }

        // CPU (Accelerate)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
        int iters = (N <= 256) ? 50 : (N <= 1024) ? 20 : (N <= 2048) ? 5 : 2;
        double t0 = now();
        for (int i = 0; i < iters; i++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
        double cpu_ms = (now() - t0) / iters * 1000.0;

        // GPU (Metal MPS)
        double gpu_ms;
        if (N <= 8192) {
            TinyHVM *ctx = thvm_init("metal");
            Term TA = thvm_tensor(ctx, A, SHAPE(N, N));
            Term TB = thvm_tensor(ctx, B, SHAPE(N, N));
            // Warmup
            for (int i = 0; i < 2; i++) { Term TC = thvm_op(ctx, UOP_MM, TA, TB); thvm_reduce(ctx, TC); thvm_to_host(ctx, TC); }
            t0 = now();
            for (int i = 0; i < iters; i++) {
                Term TC = thvm_op(ctx, UOP_MM, TA, TB);
                thvm_reduce(ctx, TC);
                thvm_to_host(ctx, TC); // sync
            }
            gpu_ms = (now() - t0) / iters * 1000.0;
            thvm_free(ctx);
        } else { gpu_ms = -1; }

        const char *winner = gpu_ms > 0 ? (gpu_ms < cpu_ms ? "GPU" : "CPU") : "N/A";
        if (gpu_ms > 0)
            printf("  %-6d  %8.2f ms  %8.2f ms  %8s\n", N, gpu_ms, cpu_ms, winner);
        else
            printf("  %-6d  %10s  %8.2f ms  %8s\n", N, "N/A", cpu_ms, winner);
        free(A); free(B); free(C);
    }

    // MLP Training scaling by batch size
    printf("\n=== MLP Train Step Scaling (784→128→ReLU→10, CE loss) ===\n\n");
    printf("  %-6s  %10s  %10s  %8s\n", "BS", "Metal(ms)", "CPU(ms)", "Winner");
    printf("  %-6s  %10s  %10s  %8s\n", "------", "----------", "----------", "--------");

    MNISTData data = mnist_load("data");
    int batches[] = {8, 32, 64, 128, 256, 512};
    for (int bi = 0; bi < 6; bi++) {
        u32 BS = batches[bi];
        if (BS > data.n_train) continue;

        // GPU training step
        TinyHVM *ctx = thvm_init("metal");
        u32 D=784, H=128, C=10;
        srand(42);
        f32 *w1d=malloc(D*H*4),*b1d=calloc(H,4),*w2d=malloc(H*C*4),*b2d=calloc(C,4);
        for(u32 i=0;i<D*H;i++) w1d[i]=0.01f*((f32)rand()/(f32)RAND_MAX*2-1);
        for(u32 i=0;i<H*C;i++) w2d[i]=0.01f*((f32)rand()/(f32)RAND_MAX*2-1);
        Term W1=thvm_tensor(ctx,w1d,SHAPE(D,H)); Term B1=thvm_tensor(ctx,b1d,SHAPE(1,H));
        Term W2=thvm_tensor(ctx,w2d,SHAPE(H,C)); Term B2=thvm_tensor(ctx,b2d,SHAPE(1,C));
        thvm_set_requires_grad(ctx,W1); thvm_set_requires_grad(ctx,B1);
        thvm_set_requires_grad(ctx,W2); thvm_set_requires_grad(ctx,B2);
        Term td=thvm_tensor(ctx,data.train_images,SHAPE(data.n_train,D));
        u32 keep=ctx->tensor_count;
        #define NP 4
        Term params[]={W1,B1,W2,B2}; u32 psz[]={D*H,H,H*C,C};

        int iters = 10;
        // Warmup
        for(int s=0;s<3;s++){
            Term X=thvm_shrink(ctx,td,(u32[]){0,BS,0,D},2); thvm_set_requires_grad(ctx,X);
            Term h=thvm_op(ctx,UOP_RELU,thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,X,W1),thvm_expand(ctx,B1,SHAPE(BS,H))),term_era());
            Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,W2),thvm_expand(ctx,B2,SHAPE(BS,C)));
            Term loss=cross_entropy_loss(ctx,logits,data.train_labels,BS,C);
            Term gs[NP]; for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
            Term gt=thvm_grad_multi(ctx,loss,params,gs,NP);
            f32 lr=0.01f; Term lr_t=thvm_tensor(ctx,&lr,SHAPE(1));
            Term sgd=term_era();
            for(int i=NP-1;i>=0;i--) sgd=thvm_app(ctx,thvm_assign(ctx,params[i],thvm_op(ctx,UOP_SUB,params[i],thvm_op(ctx,UOP_MUL,lr_t,gs[i]))),sgd);
            thvm_reduce(ctx,thvm_app(ctx,gt,sgd));
            for(u32 i=0;i<NP;i++){u32 pid=(u32)term_val(params[i]);if(ctx->tensors[pid].host_ptr){free(ctx->tensors[pid].host_ptr);ctx->tensors[pid].host_ptr=NULL;}}
            thvm_reset(ctx,keep);
        }
        double t0 = now();
        for(int s=0;s<iters;s++){
            u32 bi2=s%(data.n_train/BS);
            Term X=thvm_shrink(ctx,td,(u32[]){bi2*BS,(bi2+1)*BS,0,D},2); thvm_set_requires_grad(ctx,X);
            Term h=thvm_op(ctx,UOP_RELU,thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,X,W1),thvm_expand(ctx,B1,SHAPE(BS,H))),term_era());
            Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,W2),thvm_expand(ctx,B2,SHAPE(BS,C)));
            Term loss=cross_entropy_loss(ctx,logits,&data.train_labels[bi2*BS],BS,C);
            Term gs[NP]; for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
            Term gt=thvm_grad_multi(ctx,loss,params,gs,NP);
            f32 lr=0.01f; Term lr_t=thvm_tensor(ctx,&lr,SHAPE(1));
            Term sgd=term_era();
            for(int i=NP-1;i>=0;i--) sgd=thvm_app(ctx,thvm_assign(ctx,params[i],thvm_op(ctx,UOP_SUB,params[i],thvm_op(ctx,UOP_MUL,lr_t,gs[i]))),sgd);
            thvm_reduce(ctx,thvm_app(ctx,gt,sgd));
            thvm_to_host(ctx, loss); // sync
            for(u32 i=0;i<NP;i++){u32 pid=(u32)term_val(params[i]);if(ctx->tensors[pid].host_ptr){free(ctx->tensors[pid].host_ptr);ctx->tensors[pid].host_ptr=NULL;}}
            thvm_reset(ctx,keep);
        }
        double gpu_ms = (now() - t0) / iters * 1000.0;
        free(w1d);free(b1d);free(w2d);free(b2d);
        thvm_free(ctx);

        // CPU training step (using TinyHVM CPU backend)
        ctx = thvm_init("cpu");
        srand(42);
        w1d=malloc(D*H*4);b1d=calloc(H,4);w2d=malloc(H*C*4);b2d=calloc(C,4);
        for(u32 i=0;i<D*H;i++) w1d[i]=0.01f*((f32)rand()/(f32)RAND_MAX*2-1);
        for(u32 i=0;i<H*C;i++) w2d[i]=0.01f*((f32)rand()/(f32)RAND_MAX*2-1);
        W1=thvm_tensor(ctx,w1d,SHAPE(D,H)); B1=thvm_tensor(ctx,b1d,SHAPE(1,H));
        W2=thvm_tensor(ctx,w2d,SHAPE(H,C)); B2=thvm_tensor(ctx,b2d,SHAPE(1,C));
        thvm_set_requires_grad(ctx,W1); thvm_set_requires_grad(ctx,B1);
        thvm_set_requires_grad(ctx,W2); thvm_set_requires_grad(ctx,B2);
        td=thvm_tensor(ctx,data.train_images,SHAPE(data.n_train,D));
        keep=ctx->tensor_count;
        params[0]=W1;params[1]=B1;params[2]=W2;params[3]=B2;

        t0 = now();
        for(int s=0;s<iters;s++){
            u32 bi2=s%(data.n_train/BS);
            Term X=thvm_shrink(ctx,td,(u32[]){bi2*BS,(bi2+1)*BS,0,D},2); thvm_set_requires_grad(ctx,X);
            Term h=thvm_op(ctx,UOP_RELU,thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,X,W1),thvm_expand(ctx,B1,SHAPE(BS,H))),term_era());
            Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,W2),thvm_expand(ctx,B2,SHAPE(BS,C)));
            Term loss=cross_entropy_loss(ctx,logits,&data.train_labels[bi2*BS],BS,C);
            Term gs[NP]; for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
            Term gt=thvm_grad_multi(ctx,loss,params,gs,NP);
            f32 lr=0.01f; Term lr_t=thvm_tensor(ctx,&lr,SHAPE(1));
            Term sgd=term_era();
            for(int i=NP-1;i>=0;i--) sgd=thvm_app(ctx,thvm_assign(ctx,params[i],thvm_op(ctx,UOP_SUB,params[i],thvm_op(ctx,UOP_MUL,lr_t,gs[i]))),sgd);
            thvm_reduce(ctx,thvm_app(ctx,gt,sgd));
            for(u32 i=0;i<NP;i++){u32 pid=(u32)term_val(params[i]);if(ctx->tensors[pid].host_ptr){free(ctx->tensors[pid].host_ptr);ctx->tensors[pid].host_ptr=NULL;}}
            thvm_reset(ctx,keep);
        }
        double cpu_ms = (now() - t0) / iters * 1000.0;
        free(w1d);free(b1d);free(w2d);free(b2d);
        thvm_free(ctx);

        printf("  %-6d  %8.2f ms  %8.2f ms  %8s\n", BS, gpu_ms, cpu_ms,
               gpu_ms < cpu_ms ? "GPU" : "CPU");
    }

    mnist_free(&data);
    return 0;
}
