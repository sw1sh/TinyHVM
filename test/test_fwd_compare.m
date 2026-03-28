// test_fwd_compare.m — Compare forward pass with tinygrad
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include "../src/nn/datasets.c"

#ifndef DEVICE
  #define DEVICE "metal"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static Term relu_fn(TinyHVM *ctx, Term x) { return thvm_op(ctx, UOP_RELU, x, term_era()); }

static void print_stats(TinyHVM *ctx, Term t, const char *name) {
    t = thvm_reduce(ctx, t);
    f32 *d = thvm_to_host(ctx, t);
    u32 n = ctx->tensors[(u32)term_val(t)].view.numel;
    f32 sum = 0, sum2 = 0;
    for (u32 i = 0; i < n; i++) { sum += d[i]; sum2 += d[i]*d[i]; }
    f32 mean = sum / (f32)n, std = sqrtf(sum2/(f32)n - mean*mean);
    printf("  %-20s mean=%.6f std=%.6f", name, mean, std);
}

int main(void) {
    printf("=== Forward comparison (%s) ===\n\n", DEVICE);

    u32 wn; f32 *wbin;
    { FILE *f = fopen("/tmp/tinygrad_weights/init_thvm.bin", "rb");
      if (!f) { printf("ERROR: no weights\n"); return 1; }
      fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
      wbin = malloc((size_t)sz); fread(wbin, 1, (size_t)sz, f); fclose(f);
      wn = (u32)(sz / sizeof(f32));
    }
    f32 *wp = wbin;

    MNISTData data = mnist_load("data");
    TinyHVM *ctx = thvm_init(DEVICE);

    #define T(shape) ({ Term _t = thvm_tensor(ctx, wp, (shape)); \
        u32 _n=1; for(u32 _i=0;_i<(shape).rank;_i++) _n*=(shape).dims[_i]; wp+=_n; _t; })

    // Build model layer by layer, run one layer at a time for comparison
    u32 BS = 128;
    Term x = thvm_tensor(ctx, data.train_images, (Shape){.dims={BS,1,28,28},.rank=4});

    // Conv1 + ReLU
    Term c1w=T(((Shape){.dims={32,1,5,5},.rank=4})), c1b=T(SHAPE(32));
    Layer l_c1 = {.type=LAYER_CONV2D, .conv={c1w, c1b, 1, 32, 5}};
    Layer l_r1 = {.type=LAYER_FN, .fn=relu_fn};
    Term h = thvm_sequential(ctx, x, (Layer[]){l_c1, l_r1}, 2, BS, 1);
    print_stats(ctx, h, "conv1+relu:");
    printf("  (tg: 12.726076, 30.410383)\n");

    // Conv2 + ReLU
    Term c2w=T(((Shape){.dims={32,32,5,5},.rank=4})), c2b=T(SHAPE(32));
    Layer l_c2 = {.type=LAYER_CONV2D, .conv={c2w, c2b, 32, 32, 5}};
    Layer l_r2 = {.type=LAYER_FN, .fn=relu_fn};
    h = thvm_sequential(ctx, h, (Layer[]){l_c2, l_r2}, 2, BS, 1);
    print_stats(ctx, h, "conv2+relu:");
    printf("  (tg: 7.819910, 13.551962)\n");

    // BN1
    Term bn1g=T(SHAPE(32)), bn1b=T(SHAPE(32)), bn1rm=T(SHAPE(32)), bn1rv=T(SHAPE(32));
    Layer l_bn1 = {.type=LAYER_BN, .bn={bn1g, bn1b, bn1rm, bn1rv, 32, 20, 20}};
    h = thvm_sequential(ctx, h, (Layer[]){l_bn1}, 1, BS, 1);
    print_stats(ctx, h, "bn1:");
    printf("  (tg: 0.000000, 1.000000)\n");

    // Pool1
    Layer l_p1 = {.type=LAYER_MAXPOOL, .pool={2}};
    h = thvm_sequential(ctx, h, (Layer[]){l_p1}, 1, BS, 1);
    print_stats(ctx, h, "pool1:");
    printf("  (tg: 0.445291, 1.332044)\n");

    // Conv3 + ReLU
    Term c3w=T(((Shape){.dims={64,32,3,3},.rank=4})), c3b=T(SHAPE(64));
    Layer l_c3 = {.type=LAYER_CONV2D, .conv={c3w, c3b, 32, 64, 3}};
    Layer l_r3 = {.type=LAYER_FN, .fn=relu_fn};
    h = thvm_sequential(ctx, h, (Layer[]){l_c3, l_r3}, 2, BS, 1);
    print_stats(ctx, h, "conv3+relu:");
    printf("  (tg: 0.358148, 0.538818)\n");

    // Conv4 + ReLU
    Term c4w=T(((Shape){.dims={64,64,3,3},.rank=4})), c4b=T(SHAPE(64));
    Layer l_c4 = {.type=LAYER_CONV2D, .conv={c4w, c4b, 64, 64, 3}};
    Layer l_r4 = {.type=LAYER_FN, .fn=relu_fn};
    h = thvm_sequential(ctx, h, (Layer[]){l_c4, l_r4}, 2, BS, 1);
    print_stats(ctx, h, "conv4+relu:");
    printf("  (tg: 0.171727, 0.242304)\n");

    // BN2
    Term bn2g=T(SHAPE(64)), bn2b=T(SHAPE(64)), bn2rm=T(SHAPE(64)), bn2rv=T(SHAPE(64));
    Layer l_bn2 = {.type=LAYER_BN, .bn={bn2g, bn2b, bn2rm, bn2rv, 64, 6, 6}};
    h = thvm_sequential(ctx, h, (Layer[]){l_bn2}, 1, BS, 1);
    print_stats(ctx, h, "bn2:");
    printf("  (tg: -0.000000, 0.998685)\n");

    // Pool2
    Layer l_p2 = {.type=LAYER_MAXPOOL, .pool={2}};
    h = thvm_sequential(ctx, h, (Layer[]){l_p2}, 1, BS, 1);
    print_stats(ctx, h, "pool2:");
    printf("  (tg: 0.790940, 1.289854)\n");

    // FC
    Term fcw=T(SHAPE(576,10)), fcb=T(SHAPE(10));
    Layer l_flat = {.type=LAYER_FLATTEN, .flat={576}};
    Layer l_fc = {.type=LAYER_LINEAR, .lin={fcw, fcb, 576, 10}};
    h = thvm_sequential(ctx, h, (Layer[]){l_flat, l_fc}, 2, BS, 1);
    print_stats(ctx, h, "logits:");
    printf("  (tg: 0.117092, 0.741963)\n");

    h = thvm_reduce(ctx, h);
    f32 *ld = thvm_to_host(ctx, h);
    printf("  logits[0,:5]: %.6f %.6f %.6f %.6f %.6f\n", ld[0],ld[1],ld[2],ld[3],ld[4]);
    printf("  tg ref:       0.755676 0.789514 -1.110338 1.148737 1.433470\n");

    #undef T
    free(wbin);
    mnist_free(&data);
    thvm_free(ctx);
    return 0;
}
