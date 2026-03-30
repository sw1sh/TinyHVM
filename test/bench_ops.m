// bench_ops.m — Side-by-side kernel benchmarks (TinyHVM)
// Run with THVM_DEBUG=2 for per-kernel GPU timing
// Compare output with tinygrad's DEBUG=2
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
#include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec*1e3 + t.tv_nsec/1e6;
}

int main(void) {
    TinyHVM *ctx = thvm_init("metal");
    u32 keep = ctx->tensor_count;
    printf("=== TinyHVM Op Benchmarks (THVM_DEBUG=%s) ===\n\n",
           getenv("THVM_DEBUG") ?: "0");

    // --- 1. Elementwise ADD [4096] ---
    {
        u32 N=4096;
        f32 *a=malloc(N*4),*b=malloc(N*4);
        for(u32 i=0;i<N;i++){a[i]=(f32)i*0.01f;b[i]=(f32)i*0.02f;}
        // Warmup
        for(int w=0;w<3;w++){
            Term ta=thvm_tensor(ctx,a,SHAPE(N)),tb=thvm_tensor(ctx,b,SHAPE(N));
            thvm_reduce(ctx,thvm_op(ctx,UOP_ADD,ta,tb));
            thvm_reset(ctx,keep);
        }
        gc_reset();
        double t0=now_ms();
        for(int i=0;i<100;i++){
            Term ta=thvm_tensor(ctx,a,SHAPE(N)),tb=thvm_tensor(ctx,b,SHAPE(N));
            thvm_reduce(ctx,thvm_op(ctx,UOP_ADD,ta,tb));
            thvm_reset(ctx,keep);
        }
        double t1=now_ms();
        printf("ew_add [4096]:       %7.2fms/100  (%.2fus/iter)\n", t1-t0, (t1-t0)*10);
        free(a);free(b);
    }

    // --- 2. Elementwise MUL [512,512] ---
    {
        u32 N=512*512;
        f32 *a=malloc(N*4),*b=malloc(N*4);
        for(u32 i=0;i<N;i++){a[i]=1.001f;b[i]=0.999f;}
        for(int w=0;w<3;w++){
            Term ta=thvm_tensor(ctx,a,SHAPE(512,512)),tb=thvm_tensor(ctx,b,SHAPE(512,512));
            thvm_reduce(ctx,thvm_op(ctx,UOP_MUL,ta,tb));
            thvm_reset(ctx,keep);
        }
        gc_reset();
        double t0=now_ms();
        for(int i=0;i<50;i++){
            Term ta=thvm_tensor(ctx,a,SHAPE(512,512)),tb=thvm_tensor(ctx,b,SHAPE(512,512));
            thvm_reduce(ctx,thvm_op(ctx,UOP_MUL,ta,tb));
            thvm_reset(ctx,keep);
        }
        double t1=now_ms();
        printf("ew_mul [512,512]:    %7.2fms/50   (%.2fus/iter)\n", t1-t0, (t1-t0)*20);
        free(a);free(b);
    }

    // --- 3. Reduce SUM [512,512] → [512] ---
    {
        u32 N=512*512;
        f32 *a=malloc(N*4);
        for(u32 i=0;i<N;i++) a[i]=0.001f;
        for(int w=0;w<3;w++){
            Term ta=thvm_tensor(ctx,a,SHAPE(512,512));
            thvm_reduce(ctx,thvm_sum_axes(ctx,ta,(u32[]){1},1));
            thvm_reset(ctx,keep);
        }
        gc_reset();
        double t0=now_ms();
        for(int i=0;i<50;i++){
            Term ta=thvm_tensor(ctx,a,SHAPE(512,512));
            thvm_reduce(ctx,thvm_sum_axes(ctx,ta,(u32[]){1},1));
            thvm_reset(ctx,keep);
        }
        double t1=now_ms();
        printf("reduce_sum [512→1]:  %7.2fms/50   (%.2fus/iter)\n", t1-t0, (t1-t0)*20);
        free(a);
    }

    // --- 4. Matmul [64,784]×[784,128] ---
    {
        u32 M=64,K=784,N=128;
        f32 *a=malloc(M*K*4),*b=malloc(K*N*4);
        for(u32 i=0;i<M*K;i++) a[i]=0.01f;
        for(u32 i=0;i<K*N;i++) b[i]=0.01f;
        for(int w=0;w<3;w++){
            Term ta=thvm_tensor(ctx,a,SHAPE(M,K)),tb=thvm_tensor(ctx,b,SHAPE(K,N));
            thvm_reduce(ctx,thvm_op(ctx,UOP_MM,ta,tb));
            thvm_reset(ctx,keep);
        }
        gc_reset();
        double t0=now_ms();
        for(int i=0;i<50;i++){
            Term ta=thvm_tensor(ctx,a,SHAPE(M,K)),tb=thvm_tensor(ctx,b,SHAPE(K,N));
            thvm_reduce(ctx,thvm_op(ctx,UOP_MM,ta,tb));
            thvm_reset(ctx,keep);
        }
        double t1=now_ms();
        printf("matmul [64,784]×[784,128]: %7.2fms/50 (%.2fus/iter)\n", t1-t0, (t1-t0)*20);
        free(a);free(b);
    }

    // --- 5. Fused MUL+ADD [65536] (x*w+b) ---
    {
        u32 N=65536;
        f32 *x=malloc(N*4),*w=malloc(N*4),*b=malloc(N*4);
        for(u32 i=0;i<N;i++){x[i]=0.5f;w[i]=1.0f;b[i]=0.1f;}
        for(int ww=0;ww<3;ww++){
            Term tx=thvm_tensor(ctx,x,SHAPE(N)),tw=thvm_tensor(ctx,w,SHAPE(N)),tb=thvm_tensor(ctx,b,SHAPE(N));
            thvm_reduce(ctx,thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MUL,tx,tw),tb));
            thvm_reset(ctx,keep);
        }
        gc_reset();
        double t0=now_ms();
        for(int i=0;i<50;i++){
            Term tx=thvm_tensor(ctx,x,SHAPE(N)),tw=thvm_tensor(ctx,w,SHAPE(N)),tb=thvm_tensor(ctx,b,SHAPE(N));
            thvm_reduce(ctx,thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MUL,tx,tw),tb));
            thvm_reset(ctx,keep);
        }
        double t1=now_ms();
        printf("fused_mul_add [65536]:     %7.2fms/50 (%.2fus/iter)\n", t1-t0, (t1-t0)*20);
        free(x);free(w);free(b);
    }

    printf("\n");
    thvm_profile_kernels_summary();
    thvm_free(ctx);
    return 0;
}
