// bench_cross_framework.m — TinyHVM benchmarks matching C-ML cross_framework format
// Tests: GEMM, MLP forward, MLP train step, Conv2d forward
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

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("{\n");

    // ── GEMM ──
    {
        u32 sizes[] = {512, 1024, 2048};
        printf("  \"gemm\": [\n");
        for (int si = 0; si < 3; si++) {
            u32 N = sizes[si];
            TinyHVM *ctx = thvm_init("metal");
            f32 *ad = malloc(N*N*4), *bd = malloc(N*N*4);
            for (u32 i = 0; i < N*N; i++) { ad[i] = 0.01f*(f32)(rand()%100); bd[i] = 0.01f*(f32)(rand()%100); }
            Term A = thvm_tensor(ctx, ad, SHAPE(N, N));
            Term B = thvm_tensor(ctx, bd, SHAPE(N, N));

            // Warmup
            for (int i = 0; i < 3; i++) {
                Term C = thvm_op(ctx, UOP_MM, A, B);
                thvm_reduce(ctx, C);
            }
            if (ctx->backends[1] && ctx->backends[1]->end_batch) ctx->backends[1]->end_batch();

            // Benchmark (sync each iteration for accurate timing)
            int iters = (N <= 512) ? 50 : (N <= 1024) ? 20 : 10;
            double t0 = now();
            for (int i = 0; i < iters; i++) {
                Term C = thvm_op(ctx, UOP_MM, A, B);
                thvm_reduce(ctx, C);
                thvm_to_host(ctx, C); // force GPU sync
            }
            double elapsed = now() - t0;
            double ms = elapsed / iters * 1000.0;
            double gflops = 2.0 * N * N * N / (elapsed / iters) / 1e9;
            printf("    {\"N\": %u, \"ms\": %.2f, \"gflops\": %.1f}%s\n",
                   N, ms, gflops, si < 2 ? "," : "");
            free(ad); free(bd);
            thvm_free(ctx);
        }
        printf("  ],\n");
    }

    // ── MLP Forward ──
    {
        TinyHVM *ctx = thvm_init("metal");
        u32 BS = 64, D = 784, H = 128, C = 10;
        srand(42);
        f32 *xd = malloc(BS*D*4), *w1d = malloc(D*H*4), *b1d = calloc(H,4);
        f32 *w2d = malloc(H*C*4), *b2d = calloc(C,4);
        for (u32 i = 0; i < BS*D; i++) xd[i] = 0.01f*(f32)(rand()%100);
        for (u32 i = 0; i < D*H; i++) w1d[i] = 0.01f*((f32)rand()/(f32)RAND_MAX*2-1);
        for (u32 i = 0; i < H*C; i++) w2d[i] = 0.01f*((f32)rand()/(f32)RAND_MAX*2-1);

        Term X = thvm_tensor(ctx, xd, SHAPE(BS, D));
        Term W1 = thvm_tensor(ctx, w1d, SHAPE(D, H));
        Term B1 = thvm_tensor(ctx, b1d, SHAPE(1, H));
        Term W2 = thvm_tensor(ctx, w2d, SHAPE(H, C));
        Term B2 = thvm_tensor(ctx, b2d, SHAPE(1, C));
        u32 keep = ctx->tensor_count;

        // Warmup
        for (int i = 0; i < 5; i++) {
            Term h = thvm_op(ctx, UOP_RELU,
                thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, X, W1),
                    thvm_expand(ctx, B1, SHAPE(BS, H))), term_era());
            Term out = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, W2),
                thvm_expand(ctx, B2, SHAPE(BS, C)));
            thvm_reduce(ctx, out);
            thvm_reset(ctx, keep);
        }

        int iters = 100;
        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Term h = thvm_op(ctx, UOP_RELU,
                thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, X, W1),
                    thvm_expand(ctx, B1, SHAPE(BS, H))), term_era());
            Term out = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, W2),
                thvm_expand(ctx, B2, SHAPE(BS, C)));
            thvm_reduce(ctx, out);
            thvm_to_host(ctx, out); // force sync
            thvm_reset(ctx, keep);
        }
        double elapsed = now() - t0;
        double ms = elapsed / iters * 1000.0;
        double sps = BS * iters / elapsed;
        printf("  \"mlp_forward\": {\"batch\": %u, \"ms\": %.2f, \"samples_per_sec\": %.0f},\n", BS, ms, sps);
        free(xd); free(w1d); free(b1d); free(w2d); free(b2d);
        thvm_free(ctx);
    }

    // ── MLP Training Step ──
    {
        MNISTData data = mnist_load("data");
        TinyHVM *ctx = thvm_init("metal");
        u32 BS = 64, D = 784, H = 128, C = 10;
        srand(42);
        f32 *w1d = malloc(D*H*4), *b1d = calloc(H,4), *w2d = malloc(H*C*4), *b2d = calloc(C,4);
        for (u32 i = 0; i < D*H; i++) w1d[i] = 0.01f*((f32)rand()/(f32)RAND_MAX*2-1);
        for (u32 i = 0; i < H*C; i++) w2d[i] = 0.01f*((f32)rand()/(f32)RAND_MAX*2-1);

        Term W1=thvm_tensor(ctx,w1d,SHAPE(D,H)); Term B1=thvm_tensor(ctx,b1d,SHAPE(1,H));
        Term W2=thvm_tensor(ctx,w2d,SHAPE(H,C)); Term B2=thvm_tensor(ctx,b2d,SHAPE(1,C));
        thvm_set_requires_grad(ctx,W1); thvm_set_requires_grad(ctx,B1);
        thvm_set_requires_grad(ctx,W2); thvm_set_requires_grad(ctx,B2);
        Term td = thvm_tensor(ctx, data.train_images, SHAPE(data.n_train, D));
        u32 keep = ctx->tensor_count;

        #define NP 4
        Term params[] = {W1,B1,W2,B2};
        u32 psz[] = {D*H,H,H*C,C};

        int iters = 20;
        // Warmup
        for (int s = 0; s < 3; s++) {
            Term X = thvm_shrink(ctx, td, (u32[]){0,BS,0,D}, 2);
            Term h = thvm_op(ctx, UOP_RELU,
                thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, X, W1),
                    thvm_expand(ctx, B1, SHAPE(BS,H))), term_era());
            Term logits = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, W2),
                thvm_expand(ctx, B2, SHAPE(BS,C)));
            Term loss = cross_entropy_loss(ctx, logits, data.train_labels, BS, C);
            Term gs[NP]; for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
            Term gt = thvm_grad_multi(ctx, loss, params, gs, NP);
            f32 lr=0.01f; Term lr_t=thvm_tensor(ctx,&lr,SHAPE(1));
            Term sgd=term_era();
            for(int i=NP-1;i>=0;i--) sgd=thvm_app(ctx,thvm_assign(ctx,params[i],thvm_op(ctx,UOP_SUB,params[i],thvm_op(ctx,UOP_MUL,lr_t,gs[i]))),sgd);
            thvm_reduce(ctx, thvm_app(ctx, gt, sgd));
            for(u32 i=0;i<NP;i++){u32 pid=(u32)term_val(params[i]);if(ctx->tensors[pid].host_ptr){free(ctx->tensors[pid].host_ptr);ctx->tensors[pid].host_ptr=NULL;}}
            thvm_reset(ctx, keep);
        }

        double t0 = now();
        for (int s = 0; s < iters; s++) {
            u32 bi = s % (data.n_train / BS);
            Term X = thvm_shrink(ctx, td, (u32[]){bi*BS,(bi+1)*BS,0,D}, 2);
            thvm_set_requires_grad(ctx, X);
            Term h = thvm_op(ctx, UOP_RELU,
                thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, X, W1),
                    thvm_expand(ctx, B1, SHAPE(BS,H))), term_era());
            Term logits = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, W2),
                thvm_expand(ctx, B2, SHAPE(BS,C)));
            Term loss = cross_entropy_loss(ctx, logits, &data.train_labels[bi*BS], BS, C);
            Term gs[NP]; for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
            Term gt = thvm_grad_multi(ctx, loss, params, gs, NP);
            f32 lr=0.01f; Term lr_t=thvm_tensor(ctx,&lr,SHAPE(1));
            Term sgd=term_era();
            for(int i=NP-1;i>=0;i--) sgd=thvm_app(ctx,thvm_assign(ctx,params[i],thvm_op(ctx,UOP_SUB,params[i],thvm_op(ctx,UOP_MUL,lr_t,gs[i]))),sgd);
            thvm_reduce(ctx, thvm_app(ctx, gt, sgd));
            for(u32 i=0;i<NP;i++){u32 pid=(u32)term_val(params[i]);if(ctx->tensors[pid].host_ptr){free(ctx->tensors[pid].host_ptr);ctx->tensors[pid].host_ptr=NULL;}}
            thvm_reset(ctx, keep);
        }
        if (ctx->backends[1] && ctx->backends[1]->end_batch) ctx->backends[1]->end_batch();
        double elapsed = now() - t0;
        double ms = elapsed / iters * 1000.0;
        printf("  \"mlp_train_step\": {\"batch\": %u, \"ms\": %.2f, \"samples_per_sec\": %.0f},\n",
               BS, ms, BS * iters / elapsed);
        free(w1d); free(b1d); free(w2d); free(b2d);
        thvm_free(ctx); mnist_free(&data);
    }

    // ── Conv2d Forward ──
    {
        TinyHVM *ctx = thvm_init("metal");
        u32 BS = 8, Cin = 3, H = 32, W = 32, Cout = 16, K = 3;
        srand(42);
        f32 *xd = malloc(BS*Cin*H*W*4), *wd = malloc(Cout*Cin*K*K*4), *bd = calloc(Cout,4);
        for (u32 i = 0; i < BS*Cin*H*W; i++) xd[i] = 0.01f*(f32)(rand()%100);
        for (u32 i = 0; i < Cout*Cin*K*K; i++) wd[i] = 0.1f*((f32)rand()/(f32)RAND_MAX*2-1);
        Term X = thvm_tensor(ctx, xd, (Shape){.dims={BS,Cin,H,W},.rank=4});
        Term Wc = thvm_tensor(ctx, wd, (Shape){.dims={Cout,Cin,K,K},.rank=4});
        Term Bc = thvm_tensor(ctx, bd, SHAPE(Cout));
        u32 keep = ctx->tensor_count;

        // Warmup
        u32 pad[] = {0,0,0,0}, str[] = {1,1};
        for (int i = 0; i < 3; i++) {
            Term out = thvm_conv2d(ctx, X, Wc, Bc, 1, str, pad);
            thvm_reduce(ctx, out);
            thvm_reset(ctx, keep);
        }

        int iters = 50;
        double t0 = now();
        for (int i = 0; i < iters; i++) {
            Term out = thvm_conv2d(ctx, X, Wc, Bc, 1, str, pad);
            thvm_reduce(ctx, out);
            thvm_reset(ctx, keep);
        }
        if (ctx->backends[1] && ctx->backends[1]->end_batch) ctx->backends[1]->end_batch();
        double elapsed = now() - t0;
        double ms = elapsed / iters * 1000.0;
        printf("  \"conv2d_forward\": {\"batch\": %u, \"cin\": %u, \"cout\": %u, \"hw\": %u, \"kernel\": %u, \"ms\": %.2f}\n",
               BS, Cin, Cout, H, K, ms);
        free(xd); free(wd); free(bd);
        thvm_free(ctx);
    }

    printf("}\n");
    return 0;
}
