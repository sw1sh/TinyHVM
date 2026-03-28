// test_3conv_numgrad.m — Numerical gradient check for 3-conv+pool CNN
// Checks ALL 8 params via finite differences. No numpy dependency.
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

static Term make_w(TinyHVM *c, Shape s, u32 fi) {
    u32 n=1; for(u32 i=0;i<s.rank;i++) n*=s.dims[i];
    f32 *d=malloc(n*4); f32 b=1.0f/sqrtf((f32)fi);
    for(u32 i=0;i<n;i++) d[i]=b*((f32)rand()/(f32)RAND_MAX*2-1);
    Term t=thvm_tensor(c,d,s); free(d); return t;
}

static f32 forward_loss(TinyHVM *ctx, f32 *x_data, u32 BS,
                         f32 *cw1d, f32 *cb1d, f32 *cw2d, f32 *cb2d,
                         f32 *cw3d, f32 *cb3d, f32 *lwd, f32 *lbd,
                         u8 *labels) {
    thvm_reset(ctx, 0);
    Term x = thvm_tensor(ctx, x_data, (Shape){.dims={BS,1,28,28},.rank=4});
    Term cw1 = thvm_tensor(ctx, cw1d, (Shape){.dims={8,1,3,3},.rank=4});
    Term cb1 = thvm_tensor(ctx, cb1d, SHAPE(8));
    Term cw2 = thvm_tensor(ctx, cw2d, (Shape){.dims={16,8,3,3},.rank=4});
    Term cb2 = thvm_tensor(ctx, cb2d, SHAPE(16));
    Term cw3 = thvm_tensor(ctx, cw3d, (Shape){.dims={32,16,3,3},.rank=4});
    Term cb3 = thvm_tensor(ctx, cb3d, SHAPE(32));
    u32 flat_f = 32*5*5;
    Term lw = thvm_tensor(ctx, lwd, SHAPE(flat_f, 10));
    Term lb = thvm_tensor(ctx, lbd, SHAPE(10));

    u32 p0[]={0,0,0,0}, s1[]={1,1}, k2[]={2,2}, s2[]={2,2};
    u32 p1[]={1,1,1,1}; // padding=1 for conv3
    Term h = thvm_conv2d(ctx, x, cw1, cb1, 1, s1, p0);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    h = thvm_maxpool2d(ctx, h, k2, s2);
    h = thvm_conv2d(ctx, h, cw2, cb2, 1, s1, p0);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    h = thvm_maxpool2d(ctx, h, k2, s2);
    h = thvm_conv2d(ctx, h, cw3, cb3, 1, s1, p1);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    h = thvm_reshape(ctx, h, SHAPE(BS, flat_f));
    Term logits = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, lw),
        thvm_expand(ctx, thvm_reshape(ctx, lb, SHAPE(1,10)), SHAPE(BS,10)));
    Term loss = cross_entropy_loss(ctx, logits, labels, BS, 10);
    return thvm_to_host(ctx, loss)[0];
}

int main(void) {
    srand(42);
    MNISTData data = mnist_load("data");
    TinyHVM *ctx = thvm_init(thvm_device("metal"));
    u32 BS = 4; // small batch for speed

    // Allocate weight arrays
    u32 sizes[] = {8*1*3*3, 8, 16*8*3*3, 16, 32*16*3*3, 32, 32*5*5*10, 10};
    const char *names[] = {"cw1","cb1","cw2","cb2","cw3","cb3","lw","lb"};
    f32 *wdata[8];
    for (int i = 0; i < 8; i++) wdata[i] = malloc(sizes[i] * 4);

    // Init
    f32 bounds[] = {1.0f/3, 0, 1.0f/sqrtf(72), 0, 1.0f/sqrtf(144), 0, 1.0f/sqrtf(800), 0};
    for (int p = 0; p < 8; p++) {
        if (bounds[p] == 0) { memset(wdata[p], 0, sizes[p]*4); continue; }
        for (u32 i = 0; i < sizes[p]; i++)
            wdata[p][i] = bounds[p]*((f32)rand()/(f32)RAND_MAX*2-1);
    }

    f32 *x_data = malloc(BS * 784 * 4);
    memcpy(x_data, data.train_images, BS * 784 * 4);
    u8 *labels = data.train_labels;

    // Analytic gradient
    thvm_reset(ctx, 0);
    Term x = thvm_tensor(ctx, x_data, (Shape){.dims={BS,1,28,28},.rank=4});
    thvm_set_requires_grad(ctx, x);
    Term params[8];
    Shape shapes[] = {{.dims={8,1,3,3},.rank=4}, {.dims={8},.rank=1},
                      {.dims={16,8,3,3},.rank=4}, {.dims={16},.rank=1},
                      {.dims={32,16,3,3},.rank=4}, {.dims={32},.rank=1},
                      {.dims={800,10},.rank=2}, {.dims={10},.rank=1}};
    for (int i = 0; i < 8; i++) {
        params[i] = thvm_tensor(ctx, wdata[i], shapes[i]);
        thvm_set_requires_grad(ctx, params[i]);
    }

    u32 flat_f = 32*5*5;
    u32 p0[]={0,0,0,0}, s1[]={1,1}, k2[]={2,2}, s2[]={2,2};
    u32 p1[]={1,1,1,1};
    Term h = thvm_conv2d(ctx, x, params[0], params[1], 1, s1, p0);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    h = thvm_maxpool2d(ctx, h, k2, s2);
    h = thvm_conv2d(ctx, h, params[2], params[3], 1, s1, p0);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    h = thvm_maxpool2d(ctx, h, k2, s2);
    h = thvm_conv2d(ctx, h, params[4], params[5], 1, s1, p1);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    h = thvm_reshape(ctx, h, SHAPE(BS, flat_f));
    Term logits = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, params[6]),
        thvm_expand(ctx, thvm_reshape(ctx, params[7], SHAPE(1,10)), SHAPE(BS,10)));
    Term loss = cross_entropy_loss(ctx, logits, labels, BS, 10);

    // Use thvm_grad_multi (IC-native path)
    u32 psz[] = {72,8,1152,16,4608,32,8000,10};
    Term gs[8];
    for (int i=0;i<8;i++){f32*z=calloc(psz[i],4);
        gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
    Term grad_term = thvm_grad_multi(ctx, loss, params, gs, 8);
    thvm_reduce(ctx, thvm_app(ctx, grad_term, term_era()));

    f32 loss_val = thvm_to_host(ctx, loss)[0];
    printf("loss = %.6f\n", loss_val);

    // Read analytic gradients BEFORE numerical loop
    f32 *agrads[8];
    for (int i = 0; i < 8; i++) {
        f32 *g = thvm_to_host(ctx, gs[i]);
        agrads[i] = malloc(psz[i] * 4);
        if (g) memcpy(agrads[i], g, psz[i]*4);
        else memset(agrads[i], 0, psz[i]*4);
    }

    // Numerical gradient (finite differences) — sample 5 elements per param
    f32 eps = 1e-3f;
    int all_ok = 1;
    for (int p = 0; p < 8; p++) {
        f32 max_diff = 0;
        u32 n_check = sizes[p] < 10 ? sizes[p] : 5;
        u32 stride = sizes[p] / n_check;
        for (u32 j = 0; j < n_check; j++) {
            u32 idx = j * stride;
            f32 saved = wdata[p][idx];
            wdata[p][idx] = saved + eps;
            f32 lp = forward_loss(ctx, x_data, BS, wdata[0],wdata[1],wdata[2],wdata[3],
                                   wdata[4],wdata[5],wdata[6],wdata[7], labels);
            wdata[p][idx] = saved - eps;
            f32 lm = forward_loss(ctx, x_data, BS, wdata[0],wdata[1],wdata[2],wdata[3],
                                   wdata[4],wdata[5],wdata[6],wdata[7], labels);
            wdata[p][idx] = saved;
            f32 numg = (lp - lm) / (2*eps);
            f32 d = fabsf(agrads[p][idx] - numg);
            if (d > max_diff) max_diff = d;
        }
        int ok = max_diff < 0.01f;
        printf("  %s: max_diff=%.2e  %s\n", names[p], max_diff, ok?"OK":"FAIL");
        if (!ok) all_ok = 0;
    }
    printf("\n%s\n", all_ok ? "ALL PASS" : "FAIL");

    for (int i=0;i<8;i++){free(wdata[i]);free(agrads[i]);}
    free(x_data);
    thvm_free(ctx);
    return all_ok ? 0 : 1;
}
