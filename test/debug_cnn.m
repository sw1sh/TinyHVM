// debug_cnn.m — profile single CNN step: per-layer forward, backward, memory
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
#include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include "../src/nn/datasets.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

static Term relu_fn(TinyHVM *c, Term x) { return thvm_op(c, UOP_RELU, x, term_era()); }
static Term make_w(TinyHVM *c, Shape s, u32 fi, u32 fo) {
    u32 n=1; for(u32 i=0;i<s.rank;i++) n*=s.dims[i];
    f32 sc=sqrtf(2.0f/(f32)(fi+fo)), *d=malloc(n*sizeof(f32));
    for(u32 i=0;i<n;i++) d[i]=sc*((f32)rand()/(f32)RAND_MAX*2-1);
    Term t=thvm_tensor(c,d,s); free(d); return t;
}
static Term make_z(TinyHVM *c, u32 n) { f32*z=calloc(n,sizeof(f32)); Term t=thvm_tensor(c,z,SHAPE(n)); free(z); return t; }
static Term make_o(TinyHVM *c, u32 n) { f32*o=malloc(n*sizeof(f32)); for(u32 i=0;i<n;i++)o[i]=1; Term t=thvm_tensor(c,o,SHAPE(n)); free(o); return t; }

int main(void) {
    srand(42);
    MNISTData data = mnist_load("data");
    TinyHVM *ctx = thvm_init("metal");
    u32 BS = 128;

    Layer model[] = {
        {.type=LAYER_CONV2D, .conv={make_w(ctx,(Shape){.dims={32,1,5,5},.rank=4},25,32), make_z(ctx,32), 1,32,5}},
        {.type=LAYER_FN, .fn=relu_fn},
        {.type=LAYER_CONV2D, .conv={make_w(ctx,(Shape){.dims={32,32,5,5},.rank=4},800,32), make_z(ctx,32), 32,32,5}},
        {.type=LAYER_FN, .fn=relu_fn},
        {.type=LAYER_BN, .bn={make_o(ctx,32), make_z(ctx,32), make_z(ctx,32), make_o(ctx,32), 32, 20, 20}},
        {.type=LAYER_MAXPOOL, .pool={2}},
        {.type=LAYER_CONV2D, .conv={make_w(ctx,(Shape){.dims={64,32,3,3},.rank=4},288,64), make_z(ctx,64), 32,64,3}},
        {.type=LAYER_FN, .fn=relu_fn},
        {.type=LAYER_CONV2D, .conv={make_w(ctx,(Shape){.dims={64,64,3,3},.rank=4},576,64), make_z(ctx,64), 64,64,3}},
        {.type=LAYER_FN, .fn=relu_fn},
        {.type=LAYER_BN, .bn={make_o(ctx,64), make_z(ctx,64), make_z(ctx,64), make_o(ctx,64), 64, 6, 6}},
        {.type=LAYER_MAXPOOL, .pool={2}},
        {.type=LAYER_FLATTEN, .flat={576}},
        {.type=LAYER_LINEAR, .lin={make_w(ctx,SHAPE(576,10),576,10), make_z(ctx,10), 576,10}},
    };
    u32 NL = 14;
    Term params[28]; u32 np = 0;
    for (u32 i=0;i<NL;i++) {
        Layer *l=&model[i];
        if(l->type==LAYER_CONV2D){params[np++]=l->conv.w;params[np++]=l->conv.b;}
        else if(l->type==LAYER_BN){params[np++]=l->bn.gamma;params[np++]=l->bn.beta;}
        else if(l->type==LAYER_LINEAR){params[np++]=l->lin.w;params[np++]=l->lin.b;}
    }
    for(u32 i=0;i<np;i++) thvm_set_requires_grad(ctx,params[i]);
    Term train_data = thvm_tensor(ctx, data.train_images, (Shape){.dims={data.n_train,1,28,28},.rank=4});
    u32 n_weights = ctx->tensor_count;
    printf("setup: %u params, %u persistent tensors\n\n", np, n_weights); fflush(stdout);

    // === Build forward graph (lazy) ===
    Term x = thvm_shrink(ctx, train_data, (u32[]){0,BS,0,1,0,28,0,28}, 4);
    thvm_set_requires_grad(ctx, x);

    printf("Forward graph construction (lazy, no reduce):\n"); fflush(stdout);
    const char *lnames[] = {"conv1","relu1","conv2","relu2","bn1","mpool1","conv3","relu3","conv4","relu4","bn2","mpool2","flat","linear"};
    for (u32 i = 0; i < NL; i++) {
        u32 tc0 = ctx->tensor_count;
        u64 hp0 = ctx->heap_pos;
        Layer *l = &model[i];
        switch (l->type) {
            case LAYER_CONV2D: { u32 p[]={0,0,0,0}; u32 s[]={1,1};
                x = thvm_conv2d(ctx, x, l->conv.w, l->conv.b, 1, s, p); break; }
            case LAYER_FN: x = l->fn(ctx, x); break;
            case LAYER_BN: x = batchnorm_term(ctx, x, l->bn.gamma, l->bn.beta,
                l->bn.rmean, l->bn.rvar, BS, l->bn.c, l->bn.h, l->bn.w, 1); break;
            case LAYER_MAXPOOL: { u32 k[]={l->pool.ks,l->pool.ks}; u32 s[]={l->pool.ks,l->pool.ks};
                x = thvm_maxpool2d(ctx, x, k, s); break; }
            case LAYER_FLATTEN: x = thvm_reshape(ctx, x, SHAPE(BS, l->flat.flat_features)); break;
            case LAYER_LINEAR: x = linear(ctx, x, l->lin.w, l->lin.b, BS, l->lin.in_f, l->lin.out_f); break;
        }
        printf("  %-8s +%3u tensors  +%5llu heap  (total: %u t, %llu h)\n",
            lnames[i], ctx->tensor_count-tc0, (unsigned long long)(ctx->heap_pos-hp0),
            ctx->tensor_count, (unsigned long long)ctx->heap_pos);
        fflush(stdout);
    }

    // Loss (lazy)
    u32 tc0 = ctx->tensor_count;
    Term loss = cross_entropy_loss(ctx, x, data.train_labels, BS, 10);
    printf("  loss     +%3u tensors  (total: %u)\n\n", ctx->tensor_count-tc0, ctx->tensor_count);
    fflush(stdout);

    // === Reduce (executes everything) ===
    clock_t t0 = clock();
    Term loss_r = thvm_reduce(ctx, loss);
    f32 lv = thvm_to_host(ctx, loss_r)[0];
    clock_t t1 = clock();
    printf("Reduce: %.0fms  loss=%.4f  tensors=%u\n\n", 1000.0*(t1-t0)/(double)CLOCKS_PER_SEC, lv, ctx->tensor_count);
    fflush(stdout);

    // === Backward ===
    t0 = clock();
    Term grads[np];
    thvm_backward(ctx, loss_r, params, grads, np);
    t1 = clock();
    u32 tc_bw = ctx->tensor_count;
    printf("Backward: %.0fms  +%u tensors\n", 1000.0*(t1-t0)/(double)CLOCKS_PER_SEC, tc_bw - (u32)term_val(loss_r) - 1);
    for (u32 i=0;i<np;i++) {
        if (term_tag(grads[i])!=TAG_TEN) { printf("  [%u] ERA\n",i); continue; }
        f32 *gd=thvm_to_host(ctx,grads[i]);
        u32 gn=ctx->tensors[(u32)term_val(grads[i])].view.numel;
        f32 s=0; for(u32 j=0;j<gn;j++) s+=gd[j]*gd[j];
        printf("  [%2u] norm=%8.4f\n", i, sqrtf(s));
    }
    fflush(stdout);

    printf("\nTotal: %u tensors  %llu heap\n", ctx->tensor_count, (unsigned long long)ctx->heap_pos);
    thvm_reset(ctx, n_weights);
    printf("After reset: %u tensors\n", ctx->tensor_count);

    thvm_free(ctx);
    mnist_free(&data);
    return 0;
}
