// test_jit_minimal.m — Minimal JIT test: dense layer + SGD, no BN/conv
// Verifies weights update correctly during JIT replay
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
    u32 BS = 64;

    // Simple dense: 784 → 10
    u32 ff = 784;
    Term w = mkw(ctx, SHAPE(ff, 10), ff);
    Term b = mkz(ctx, 10);
    thvm_set_requires_grad(ctx, w);
    thvm_set_requires_grad(ctx, b);

    // Fixed input/label buffers
    f32 *xbuf = calloc(BS * 784, 4);
    Term x = thvm_tensor(ctx, xbuf, SHAPE(BS, 784)); free(xbuf);
    thvm_set_requires_grad(ctx, x);
    f32 *ohbuf = calloc(BS * 10, 4);
    Term oh = thvm_tensor(ctx, ohbuf, SHAPE(BS, 10)); free(ohbuf);
    f32 lr_val = 0.01f;
    Term lr = thvm_tensor(ctx, &lr_val, SHAPE(1));

    // Grad accumulators
    #define NP 2
    Term params[NP] = {w, b};
    u32 psz[NP] = {ff * 10, 10};
    Term gs[NP];
    for (int i = 0; i < NP; i++) {
        f32 *z = calloc(psz[i], 4);
        gs[i] = thvm_tensor(ctx, z, ctx->tensors[(u32)term_val(params[i])].view.shape);
        free(z);
    }

    // Build graph
    Term logits = thvm_op(ctx, UOP_ADD, thvm_mm(ctx, x, w),
        thvm_expand(ctx, thvm_reshape(ctx, b, SHAPE(1, 10)), SHAPE(BS, 10)));
    Term probs = softmax(ctx, logits, BS, 10);
    f32 eps = 1e-7f;
    Term eps_t = thvm_expand(ctx, thvm_tensor(ctx, &eps, SHAPE(1, 1)), SHAPE(BS, 10));
    Term clamped = thvm_op(ctx, UOP_MAX, probs, eps_t);
    Term log_probs = thvm_op(ctx, UOP_LOG, clamped, term_era());
    Term masked = thvm_op(ctx, UOP_MUL, oh, log_probs);
    Term sum_all = thvm_sum_axes(ctx, masked, (u32[]){0, 1}, 2);
    Term neg = thvm_op(ctx, UOP_NEG, sum_all, term_era());
    f32 inv_B = 1.f / (f32)BS;
    Term scale = thvm_tensor(ctx, &inv_B, SHAPE(1, 1));
    Term loss = thvm_op(ctx, UOP_MUL, neg, scale);

    Term grad_term = thvm_grad_multi(ctx, loss, params, gs, NP);
    Term sgd = term_era();
    for (int i = NP - 1; i >= 0; i--)
        sgd = thvm_app(ctx, thvm_assign(ctx, params[i],
            thvm_op(ctx, UOP_SUB, params[i],
                thvm_op(ctx, UOP_MUL, lr, gs[i]))), sgd);
    Term train_step = thvm_app(ctx, grad_term, sgd);

    // n_persistent AFTER graph construction
    u32 n_persistent = ctx->tensor_count;
    printf("n_persistent=%u, metal_pool.count=%u\n", n_persistent, metal_pool.count);

    // Load first batch
    u32 x_bid = (u32)term_val(x), oh_bid = (u32)term_val(oh);
    u32 bi = 0;
    ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
        &data.train_images[bi * BS * 784], BS * 784 * sizeof(f32));
    f32 *oh_data = calloc(BS * 10, sizeof(f32));
    for (u32 i = 0; i < BS; i++) oh_data[i * 10 + data.train_labels[bi * BS + i]] = 1.f;
    ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh_data, BS * 10 * sizeof(f32));
    for (int i = 0; i < NP; i++) {
        u32 gid = (u32)term_val(gs[i]);
        memset(metal_pool.bufs[ctx->tensors[gid].buf_id].contents, 0, psz[i] * sizeof(f32));
    }

    // Read initial weights
    u32 w_bufid = ctx->tensors[(u32)term_val(w)].buf_id;
    jit_flush();
    f32 *wdata = (f32 *)metal_pool.bufs[w_bufid].contents;
    printf("w before: [%f,%f,%f,%f]\n", wdata[0], wdata[1], wdata[2], wdata[3]);

    // Capture
    jit_begin_capture(n_persistent);
    double t0 = now_s();
    thvm_reduce(ctx, train_step);
    f32 lv = thvm_to_host(ctx, loss)[0];
    jit_end_capture();
    printf("capture: loss=%.4f (%.0fms)\n", lv, (now_s() - t0) * 1000);

    jit_flush();
    wdata = (f32 *)metal_pool.bufs[w_bufid].contents;
    printf("w after capture: [%f,%f,%f,%f]\n", wdata[0], wdata[1], wdata[2], wdata[3]);

    // Reset
    extern u32 total_dispatches; total_dispatches = 0;
    thvm_reset(ctx, n_persistent);

    // Verify persistent buffers still intact
    jit_flush();
    wdata = (f32 *)metal_pool.bufs[w_bufid].contents;
    printf("w after reset: [%f,%f,%f,%f]\n", wdata[0], wdata[1], wdata[2], wdata[3]);

    // Replay step 1
    jit_flush();
    bi = 1;
    ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
        &data.train_images[bi * BS * 784], BS * 784 * sizeof(f32));
    memset(oh_data, 0, BS * 10 * sizeof(f32));
    for (u32 i = 0; i < BS; i++) oh_data[i * 10 + data.train_labels[bi * BS + i]] = 1.f;
    ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh_data, BS * 10 * sizeof(f32));
    for (int i = 0; i < NP; i++) {
        u32 gid = (u32)term_val(gs[i]);
        memset(metal_pool.bufs[ctx->tensors[gid].buf_id].contents, 0, psz[i] * sizeof(f32));
    }

    jit_replay();
    jit_flush();

    wdata = (f32 *)metal_pool.bufs[w_bufid].contents;
    u32 gw_bufid = ctx->tensors[(u32)term_val(gs[0])].buf_id;
    f32 *gwdata = (f32 *)metal_pool.bufs[gw_bufid].contents;
    printf("w after replay1: [%f,%f,%f,%f]\n", wdata[0], wdata[1], wdata[2], wdata[3]);
    printf("grad_w after replay1: [%f,%f,%f,%f]\n", gwdata[0], gwdata[1], gwdata[2], gwdata[3]);

    // Replay step 2
    bi = 2;
    ctx->tensors[x_bid].backend->buf_write(ctx->tensors[x_bid].buf_id,
        &data.train_images[bi * BS * 784], BS * 784 * sizeof(f32));
    memset(oh_data, 0, BS * 10 * sizeof(f32));
    for (u32 i = 0; i < BS; i++) oh_data[i * 10 + data.train_labels[bi * BS + i]] = 1.f;
    ctx->tensors[oh_bid].backend->buf_write(ctx->tensors[oh_bid].buf_id, oh_data, BS * 10 * sizeof(f32));
    for (int i = 0; i < NP; i++) {
        u32 gid = (u32)term_val(gs[i]);
        memset(metal_pool.bufs[ctx->tensors[gid].buf_id].contents, 0, psz[i] * sizeof(f32));
    }

    jit_replay();
    jit_flush();

    wdata = (f32 *)metal_pool.bufs[w_bufid].contents;
    gwdata = (f32 *)metal_pool.bufs[gw_bufid].contents;
    printf("w after replay2: [%f,%f,%f,%f]\n", wdata[0], wdata[1], wdata[2], wdata[3]);
    printf("grad_w after replay2: [%f,%f,%f,%f]\n", gwdata[0], gwdata[1], gwdata[2], gwdata[3]);

    free(oh_data);
    thvm_free(ctx);
    return 0;
}
