// bench_jit.m — JIT capture/replay benchmark for CNN training
// Step 0: full IC evaluation + JIT capture
// Steps 1+: JIT replay (bypasses IC trampoline entirely)
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
#include <time.h>

int main(void) {
    srand(42);
    MNISTData data = mnist_load("data");
    TinyHVM *ctx = thvm_init("metal");
    u32 BS = 32;

    f32 *cw = malloc(72*4); for(u32 i=0;i<72;i++) cw[i]=0.1f*((f32)rand()/(f32)RAND_MAX*2-1);
    Term conv_w = thvm_tensor(ctx, cw, (Shape){.dims={8,1,3,3},.rank=4}); free(cw);
    f32 *cb = calloc(8,4); Term conv_b = thvm_tensor(ctx, cb, SHAPE(8)); free(cb);
    u32 ff = 8*26*26;
    f32 *lw = malloc(ff*10*4); f32 bnd = 1.f/sqrtf((f32)ff);
    for(u32 i=0;i<ff*10;i++) lw[i]=bnd*((f32)rand()/(f32)RAND_MAX*2-1);
    Term lin_w = thvm_tensor(ctx, lw, SHAPE(ff,10)); free(lw);
    f32 *lb = calloc(10,4); Term lin_b = thvm_tensor(ctx, lb, SHAPE(10)); free(lb);

    #define NP 4
    Term params[NP] = {conv_w, conv_b, lin_w, lin_b};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx, params[i]);
    Term train_data = thvm_tensor(ctx, data.train_images, (Shape){.dims={data.n_train,1,28,28},.rank=4});
    u32 n_weights = ctx->tensor_count;

    // ── Step 0: Full IC evaluation + JIT capture ──
    printf("Step 0: IC evaluation + JIT capture\n");
    jit_begin_capture(n_weights);
    {
        struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
        Term x = thvm_shrink(ctx, train_data, (u32[]){0,BS,0,1,0,28,0,28}, 4);
        thvm_set_requires_grad(ctx, x);
        u32 pad[]={0,0,0,0}, str[]={1,1};
        Term h = thvm_conv2d(ctx, x, conv_w, conv_b, 1, str, pad);
        h = thvm_op(ctx, UOP_RELU, h, term_era());
        h = thvm_reshape(ctx, h, SHAPE(BS, ff));
        Term logits = thvm_op(ctx, UOP_ADD, thvm_mm(ctx, h, lin_w),
            thvm_expand(ctx, thvm_reshape(ctx, lin_b, SHAPE(1,10)), SHAPE(BS,10)));
        Term loss = cross_entropy_loss(ctx, logits, data.train_labels, BS, 10);
        Term gs[NP]; u32 psz[]={72,8,ff*10,10};
        for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
        Term gt = thvm_grad_multi(ctx, loss, params, gs, NP);
        f32 lr=0.01f; Term lr_t = thvm_tensor(ctx, &lr, SHAPE(1));
        Term sgd = term_era();
        for(int i=NP-1;i>=0;i--) sgd = thvm_app(ctx, thvm_assign(ctx, params[i],
            thvm_op(ctx, UOP_SUB, params[i], thvm_op(ctx, UOP_MUL, lr_t, gs[i]))), sgd);
        thvm_reduce(ctx, thvm_app(ctx, gt, sgd));
        f32 loss_val = thvm_to_host(ctx, loss)[0];
        struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        f32 ms = (f32)(t1.tv_sec-t0.tv_sec)*1000+(f32)(t1.tv_nsec-t0.tv_nsec)/1e6f;
        extern u32 total_dispatches;
        printf("  loss=%.4f dispatches=%u (%.0fms)\n", loss_val, total_dispatches, ms);
        total_dispatches = 0;
        for(u32 i=0;i<NP;i++){u32 pid=(u32)term_val(params[i]);if(ctx->tensors[pid].host_ptr){free(ctx->tensors[pid].host_ptr);ctx->tensors[pid].host_ptr=NULL;}}
        thvm_reset(ctx, n_weights);
    }
    jit_end_capture();

    // ── Steps 1+: JIT replay ──
    printf("\nJIT Replay steps:\n");
    u32 n_steps = 20;
    struct timespec t_start; clock_gettime(CLOCK_MONOTONIC, &t_start);
    for(u32 step = 1; step < n_steps; step++) {
        struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
        jit_replay();
        if (ctx->backends[1] && ctx->backends[1]->end_batch) ctx->backends[1]->end_batch();
        struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        f32 ms = (f32)(t1.tv_sec-t0.tv_sec)*1000+(f32)(t1.tv_nsec-t0.tv_nsec)/1e6f;
        extern u32 total_dispatches;
        if (step < 5 || step == n_steps-1)
            printf("  step %2u: dispatches=%u (%.1fms)\n", step, total_dispatches, ms);
        total_dispatches = 0;
    }
    struct timespec t_end; clock_gettime(CLOCK_MONOTONIC, &t_end);
    f32 replay_ms = ((f32)(t_end.tv_sec-t_start.tv_sec)*1000+(f32)(t_end.tv_nsec-t_start.tv_nsec)/1e6f) / (n_steps-1);
    printf("\nJIT replay: %.1fms/step avg (vs IC: ~5ms/step)\n", replay_ms);

    thvm_free(ctx); mnist_free(&data);
    return 0;
}
