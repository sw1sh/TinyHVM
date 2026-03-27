// test_cnn_grad.m — Verify CNN gradient against numpy reference
// Conv(1,8,3) → ReLU → Flatten → Linear(5408,10), cross-entropy loss
// Saves weights, input, labels, and gradients to /tmp/cnn_*.bin

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

#ifndef DEVICE
  #define DEVICE "metal"
#endif

int main(void) {
    srand(42);
    MNISTData data = mnist_load("data");
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));
    u32 BS = 32;

    // ── Init weights (same as test_cnn_small.m) ──────────────────
    u32 cw_n = 8*1*3*3;
    f32 *cw = malloc(cw_n * sizeof(f32));
    for (u32 i = 0; i < cw_n; i++) cw[i] = 0.1f * ((f32)rand()/(f32)RAND_MAX * 2 - 1);
    Term conv_w = thvm_tensor(ctx, cw, (Shape){.dims={8,1,3,3},.rank=4});

    f32 *cb = calloc(8, sizeof(f32));
    Term conv_b = thvm_tensor(ctx, cb, SHAPE(8));

    u32 flat_f = 8 * 26 * 26;
    u32 lw_n = flat_f * 10;
    f32 *lw = malloc(lw_n * sizeof(f32));
    f32 bound = 1.0f / sqrtf((f32)flat_f);
    for (u32 i = 0; i < lw_n; i++) lw[i] = bound * ((f32)rand()/(f32)RAND_MAX * 2 - 1);
    Term lin_w = thvm_tensor(ctx, lw, SHAPE(flat_f, 10));

    f32 *lb = calloc(10, sizeof(f32));
    Term lin_b = thvm_tensor(ctx, lb, SHAPE(10));

    // ── Save init weights ────────────────────────────────────────
    { FILE *f=fopen("/tmp/cnn_conv_w.bin","wb"); fwrite(cw,4,cw_n,f); fclose(f); }
    { FILE *f=fopen("/tmp/cnn_conv_b.bin","wb"); fwrite(cb,4,8,f); fclose(f); }
    { FILE *f=fopen("/tmp/cnn_lin_w.bin","wb"); fwrite(lw,4,lw_n,f); fclose(f); }
    { FILE *f=fopen("/tmp/cnn_lin_b.bin","wb"); fwrite(lb,4,10,f); fclose(f); }

    free(cw); free(cb); free(lw); free(lb);

    // ── Mark requires grad ───────────────────────────────────────
    thvm_set_requires_grad(ctx, conv_w);
    thvm_set_requires_grad(ctx, conv_b);
    thvm_set_requires_grad(ctx, lin_w);
    thvm_set_requires_grad(ctx, lin_b);

    // ── Input: first BS images from MNIST ────────────────────────
    // Save X as (BS, 1, 28, 28) floats
    { FILE *f=fopen("/tmp/cnn_X.bin","wb"); fwrite(data.train_images,4,BS*1*28*28,f); fclose(f); }

    // Save Y as (BS, 10) one-hot
    { FILE *f=fopen("/tmp/cnn_Y.bin","wb");
      f32 *yoh=calloc(BS*10,4);
      for(u32 i=0;i<BS;i++) yoh[i*10+data.train_labels[i]]=1.0f;
      fwrite(yoh,4,BS*10,f); free(yoh); fclose(f); }

    Term x = thvm_tensor(ctx, data.train_images,
        (Shape){.dims={BS, 1, 28, 28}, .rank=4});

    // ── Forward pass ─────────────────────────────────────────────
    u32 padding[] = {0,0,0,0}, stride[] = {1,1};
    Term h = thvm_conv2d(ctx, x, conv_w, conv_b, 1, stride, padding);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    h = thvm_reshape(ctx, h, SHAPE(BS, flat_f));
    Term logits = thvm_op(ctx, UOP_ADD,
        thvm_op(ctx, UOP_MM, h, lin_w),
        thvm_expand(ctx, thvm_reshape(ctx, lin_b, SHAPE(1, 10)), SHAPE(BS, 10)));

    // ── Cross-entropy loss ───────────────────────────────────────
    Term loss = cross_entropy_loss(ctx, logits, data.train_labels, BS, 10);
    thvm_reduce(ctx, loss);
    printf("loss=%.6f\n", thvm_to_host(ctx, loss)[0]);

    // ── Gradients ────────────────────────────────────────────────
    Term params[] = { conv_w, conv_b, lin_w, lin_b };
    const char *gnames[] = {"g_conv_w", "g_conv_b", "g_lin_w", "g_lin_b"};
    u32 gsizes[] = { 8*1*3*3, 8, flat_f*10, 10 };

    for (u32 p = 0; p < 4; p++) {
        Term g = thvm_reduce(ctx, thvm_grad(ctx, loss, params[p]));
        if (term_tag(g) != TAG_TEN) {
            printf("%s: not TEN (tag=%u)\n", gnames[p], term_tag(g));
            continue;
        }
        f32 *gv = thvm_to_host(ctx, g);
        char path[128];
        snprintf(path, 128, "/tmp/cnn_%s.bin", gnames[p]);
        FILE *fp = fopen(path, "wb");
        fwrite(gv, 4, gsizes[p], fp);
        fclose(fp);
        printf("%s: first4=[%.8e,%.8e,%.8e,%.8e]\n",
               gnames[p], gv[0], gv[1], gv[2], gv[3]);
    }

    thvm_free(ctx);
    mnist_free(&data);
    printf("DONE\n");
    return 0;
}
