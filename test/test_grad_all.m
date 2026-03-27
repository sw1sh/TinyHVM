// test_grad_all.m — Test thvm_grad_all (multi-target GRAD)
// Compare against per-param thvm_grad for correctness

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

int main(void) {
    MNISTData data = mnist_load("data");
    u32 BS = 32;
    srand(42);
    TinyHVM *ctx = thvm_init(thvm_device("metal"));

    u32 cw_n = 8*1*3*3;
    f32 *cwd = malloc(cw_n*4);
    for(u32 i=0;i<cw_n;i++) cwd[i]=0.1f*((f32)rand()/(f32)RAND_MAX*2-1);
    Term cw = thvm_tensor(ctx, cwd, (Shape){.dims={8,1,3,3},.rank=4});
    f32 *cbd = calloc(8,4);
    Term cb = thvm_tensor(ctx, cbd, SHAPE(8));
    u32 flat_f = 8*26*26;
    f32 *lwd = malloc(flat_f*10*4);
    f32 bound = 1.0f/sqrtf((f32)flat_f);
    for(u32 i=0;i<flat_f*10;i++) lwd[i]=bound*((f32)rand()/(f32)RAND_MAX*2-1);
    Term lw = thvm_tensor(ctx, lwd, SHAPE(flat_f,10));
    f32 *lbd = calloc(10,4);
    Term lb = thvm_tensor(ctx, lbd, SHAPE(10));

    #define NP 4
    Term params[NP] = {cw, cb, lw, lb};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx, params[i]);

    Term X = thvm_tensor(ctx, data.train_images, (Shape){.dims={BS,1,28,28},.rank=4});
    u32 n_weights = ctx->tensor_count;

    // Build forward + loss
    u32 p0[]={0,0,0,0}, s1[]={1,1};
    Term h = thvm_conv2d(ctx, X, cw, cb, 1, s1, p0);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    h = thvm_reshape(ctx, h, SHAPE(BS, flat_f));
    Term logits = thvm_op(ctx, UOP_ADD,
        thvm_op(ctx, UOP_MM, h, lw),
        thvm_expand(ctx, thvm_reshape(ctx, lb, SHAPE(1,10)), SHAPE(BS,10)));
    Term loss = cross_entropy_loss(ctx, logits, data.train_labels, BS, 10);
    thvm_reduce(ctx, loss);
    printf("loss=%.4f\n", thvm_to_host(ctx, loss)[0]);

    // Method 1: thvm_grad_all (single walk)
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    Term grads_all[NP];
    thvm_grad_all(ctx, loss, params, grads_all, NP);
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
    f32 ms_all = (f32)(t1.tv_sec-t0.tv_sec)*1000+(f32)(t1.tv_nsec-t0.tv_nsec)/1e6f;
    extern u32 total_dispatches;
    u32 disp_all = total_dispatches;
    total_dispatches = 0;

    printf("\ngrad_all: %.0fms, %u dispatches\n", ms_all, disp_all);
    const char *gn[NP] = {"cw","cb","lw","lb"};
    for(u32 i=0;i<NP;i++){
        if(term_tag(grads_all[i])!=TAG_TEN){printf("  %s: tag=%u (not TEN)\n",gn[i],term_tag(grads_all[i]));continue;}
        f32 *v = thvm_to_host(ctx, grads_all[i]);
        printf("  %s: [%.4e, %.4e, %.4e, %.4e]\n", gn[i], v[0],v[1],v[2],v[3]);
    }

    // Method 2: per-param thvm_grad (separate walks)
    thvm_reset(ctx, n_weights);
    // Rebuild forward (tensors were freed by reset)
    h = thvm_conv2d(ctx, X, cw, cb, 1, s1, p0);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    h = thvm_reshape(ctx, h, SHAPE(BS, flat_f));
    logits = thvm_op(ctx, UOP_ADD,
        thvm_op(ctx, UOP_MM, h, lw),
        thvm_expand(ctx, thvm_reshape(ctx, lb, SHAPE(1,10)), SHAPE(BS,10)));
    loss = cross_entropy_loss(ctx, logits, data.train_labels, BS, 10);
    thvm_reduce(ctx, loss);

    struct timespec t2; clock_gettime(CLOCK_MONOTONIC, &t2);
    Term grads_sep[NP];
    for(u32 i=0;i<NP;i++)
        grads_sep[i] = thvm_reduce(ctx, thvm_grad(ctx, loss, params[i]));
    struct timespec t3; clock_gettime(CLOCK_MONOTONIC, &t3);
    f32 ms_sep = (f32)(t3.tv_sec-t2.tv_sec)*1000+(f32)(t3.tv_nsec-t2.tv_nsec)/1e6f;
    u32 disp_sep = total_dispatches;

    printf("\ngrad_sep: %.0fms, %u dispatches\n", ms_sep, disp_sep);
    for(u32 i=0;i<NP;i++){
        if(term_tag(grads_sep[i])!=TAG_TEN) continue;
        f32 *v = thvm_to_host(ctx, grads_sep[i]);
        printf("  %s: [%.4e, %.4e, %.4e, %.4e]\n", gn[i], v[0],v[1],v[2],v[3]);
    }

    // Compare
    printf("\nComparison:\n");
    int ok = 1;
    for(u32 i=0;i<NP;i++){
        if(term_tag(grads_all[i])!=TAG_TEN || term_tag(grads_sep[i])!=TAG_TEN){
            printf("  %s: SKIP (not both TEN)\n",gn[i]); ok=0; continue;
        }
        f32 *va = thvm_to_host(ctx, grads_all[i]);
        f32 *vs = thvm_to_host(ctx, grads_sep[i]);
        u32 n = ctx->tensors[(u32)term_val(grads_all[i])].view.numel;
        f32 maxd = 0;
        for(u32 j=0;j<n;j++){f32 d=fabsf(va[j]-vs[j]); if(d>maxd) maxd=d;}
        printf("  %s: max_diff=%.2e %s\n", gn[i], maxd, maxd<1e-5?"OK":"FAIL");
        if(maxd>=1e-5) ok=0;
    }
    printf("\n%s  speedup=%.1fx\n", ok?"PASS":"FAIL", ms_sep/ms_all);

    free(cwd);free(cbd);free(lwd);free(lbd);
    thvm_free(ctx); mnist_free(&data);
    return 0;
}
