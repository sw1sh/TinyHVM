// test_2conv_grad.m — 2-layer CNN grad check vs numpy
// Conv(1,4,3)→ReLU→Conv(4,8,3)→ReLU→Flatten→Linear(8*24*24,10)
// Sum-of-logits loss (avoids CE diamonds for isolated conv testing)

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

int main(void) {
    MNISTData data = mnist_load("data");
    u32 BS = 4;  // small batch for tractable computation
    srand(42);
    TinyHVM *ctx = thvm_init(thvm_device("metal"));

    // Conv1: (1,4,3,3) — 36 params
    u32 cw1_n = 4*1*3*3;
    f32 *cw1d = malloc(cw1_n*4);
    for(u32 i=0;i<cw1_n;i++) cw1d[i] = 0.1f*((f32)rand()/(f32)RAND_MAX*2-1);
    Term cw1 = thvm_tensor(ctx, cw1d, (Shape){.dims={4,1,3,3},.rank=4});
    f32 *cb1d = calloc(4, 4);
    Term cb1 = thvm_tensor(ctx, cb1d, SHAPE(4));

    // Conv2: (8,4,3,3) — 288 params
    u32 cw2_n = 8*4*3*3;
    f32 *cw2d = malloc(cw2_n*4);
    for(u32 i=0;i<cw2_n;i++) cw2d[i] = 0.05f*((f32)rand()/(f32)RAND_MAX*2-1);
    Term cw2 = thvm_tensor(ctx, cw2d, (Shape){.dims={8,4,3,3},.rank=4});
    f32 *cb2d = calloc(8, 4);
    Term cb2 = thvm_tensor(ctx, cb2d, SHAPE(8));

    // Linear: (8*24*24, 10) = (4608, 10)
    u32 flat_f = 8*24*24;
    f32 *lwd = malloc(flat_f*10*4);
    f32 bound = 1.0f/sqrtf((f32)flat_f);
    for(u32 i=0;i<flat_f*10;i++) lwd[i] = bound*((f32)rand()/(f32)RAND_MAX*2-1);
    Term lw = thvm_tensor(ctx, lwd, SHAPE(flat_f,10));
    f32 *lbd = calloc(10, 4);
    Term lb = thvm_tensor(ctx, lbd, SHAPE(10));

    #define NP 6
    Term params[NP] = {cw1, cb1, cw2, cb2, lw, lb};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx, params[i]);

    // Save weights
    { FILE *f=fopen("/tmp/c2_cw1.bin","wb"); fwrite(cw1d,4,cw1_n,f); fclose(f); }
    { FILE *f=fopen("/tmp/c2_cb1.bin","wb"); fwrite(cb1d,4,4,f); fclose(f); }
    { FILE *f=fopen("/tmp/c2_cw2.bin","wb"); fwrite(cw2d,4,cw2_n,f); fclose(f); }
    { FILE *f=fopen("/tmp/c2_cb2.bin","wb"); fwrite(cb2d,4,8,f); fclose(f); }
    { FILE *f=fopen("/tmp/c2_lw.bin","wb"); fwrite(lwd,4,flat_f*10,f); fclose(f); }
    { FILE *f=fopen("/tmp/c2_lb.bin","wb"); fwrite(lbd,4,10,f); fclose(f); }

    // Input: first BS images
    Term X = thvm_tensor(ctx, data.train_images,
        (Shape){.dims={BS,1,28,28},.rank=4});
    { FILE *f=fopen("/tmp/c2_X.bin","wb"); fwrite(data.train_images,4,BS*784,f); fclose(f); }
    { FILE *f=fopen("/tmp/c2_Y.bin","wb");
      f32 *oh=calloc(BS*10,4);
      for(u32 i=0;i<BS;i++) oh[i*10+data.train_labels[i]]=1;
      fwrite(oh,4,BS*10,f); free(oh); fclose(f); }

    // Forward: Conv1→ReLU→Conv2→ReLU→Flatten→Linear
    u32 pad0[]={0,0,0,0}, str1[]={1,1};
    Term h1 = thvm_conv2d(ctx, X, cw1, cb1, 1, str1, pad0);
    h1 = thvm_op(ctx, UOP_RELU, h1, term_era());
    // h1: [BS, 4, 26, 26]
    Term h2 = thvm_conv2d(ctx, h1, cw2, cb2, 1, str1, pad0);
    h2 = thvm_op(ctx, UOP_RELU, h2, term_era());
    // h2: [BS, 8, 24, 24]
    Term flat = thvm_reshape(ctx, h2, SHAPE(BS, flat_f));
    Term logits = thvm_op(ctx, UOP_ADD,
        thvm_op(ctx, UOP_MM, flat, lw),
        thvm_expand(ctx, thvm_reshape(ctx, lb, SHAPE(1,10)), SHAPE(BS,10)));

    // Simple loss: sum of all logits (avoids softmax diamonds)
    Term loss = thvm_op(ctx, UOP_SUM,
        thvm_reshape(ctx, logits, SHAPE(BS*10)), term_era());
    thvm_reduce(ctx, loss);
    printf("loss=%.6f\n", thvm_to_host(ctx, loss)[0]);

    // Gradients
    const char *gn[NP] = {"g_cw1","g_cb1","g_cw2","g_cb2","g_lw","g_lb"};
    u32 gsz[NP] = {cw1_n, 4, cw2_n, 8, flat_f*10, 10};
    for(u32 p=0;p<NP;p++){
        Term g = thvm_reduce(ctx, thvm_grad(ctx, loss, params[p]));
        if(term_tag(g)!=TAG_TEN){printf("%s: not TEN (tag=%u)\n",gn[p],term_tag(g));continue;}
        f32 *gv = thvm_to_host(ctx, g);
        char path[128]; snprintf(path,128,"/tmp/c2_%s.bin",gn[p]);
        FILE *fp=fopen(path,"wb"); fwrite(gv,4,gsz[p],fp); fclose(fp);
        printf("%s: first4=[%.6e,%.6e,%.6e,%.6e]\n",gn[p],gv[0],gv[1],gv[2],gv[3]);
    }

    free(cw1d);free(cb1d);free(cw2d);free(cb2d);free(lwd);free(lbd);
    thvm_free(ctx); mnist_free(&data);
    return 0;
}
