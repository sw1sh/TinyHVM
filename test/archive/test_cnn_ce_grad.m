// test_cnn_ce_grad.m — CNN + cross-entropy loss gradient check
// Conv(1,8,3)→ReLU→Flatten→Linear(8*26*26,10), CE loss
// CE has internal diamonds (exp, logits reused in softmax)

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
    u32 BS = 4;
    srand(42);
    TinyHVM *ctx = thvm_init("metal");

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

    { FILE *f=fopen("/tmp/ce_cw.bin","wb"); fwrite(cwd,4,cw_n,f); fclose(f); }
    { FILE *f=fopen("/tmp/ce_lw.bin","wb"); fwrite(lwd,4,flat_f*10,f); fclose(f); }

    Term X = thvm_tensor(ctx, data.train_images, (Shape){.dims={BS,1,28,28},.rank=4});
    { FILE *f=fopen("/tmp/ce_X.bin","wb"); fwrite(data.train_images,4,BS*784,f); fclose(f); }
    { FILE *f=fopen("/tmp/ce_Y.bin","wb");
      f32 *oh=calloc(BS*10,4);
      for(u32 i=0;i<BS;i++) oh[i*10+data.train_labels[i]]=1;
      fwrite(oh,4,BS*10,f); free(oh); fclose(f); }

    u32 pad0[]={0,0,0,0}, str1[]={1,1};
    Term h = thvm_conv2d(ctx, X, cw, cb, 1, str1, pad0);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    h = thvm_reshape(ctx, h, SHAPE(BS, flat_f));
    Term logits = thvm_op(ctx, UOP_ADD,
        thvm_mm(ctx, h, lw),
        thvm_expand(ctx, thvm_reshape(ctx, lb, SHAPE(1,10)), SHAPE(BS,10)));

    // Cross-entropy loss (has softmax diamonds)
    Term loss = cross_entropy_loss(ctx, logits, data.train_labels, BS, 10);
    thvm_reduce(ctx, loss);
    printf("loss=%.6f\n", thvm_to_host(ctx, loss)[0]);

    const char *gn[NP] = {"g_cw","g_cb","g_lw","g_lb"};
    u32 gsz[NP] = {cw_n, 8, flat_f*10, 10};
    for(u32 p=0;p<NP;p++){
        Term g = thvm_reduce(ctx, thvm_grad(ctx, loss, params[p]));
        if(term_tag(g)!=TAG_TEN){printf("%s: not TEN (tag=%u)\n",gn[p],term_tag(g));continue;}
        f32 *gv = thvm_to_host(ctx, g);
        char path[128]; snprintf(path,128,"/tmp/ce_%s.bin",gn[p]);
        FILE *fp=fopen(path,"wb"); fwrite(gv,4,gsz[p],fp); fclose(fp);
        printf("%s: first4=[%.6e,%.6e,%.6e,%.6e]\n",gn[p],gv[0],gv[1],gv[2],gv[3]);
    }

    free(cwd);free(cbd);free(lwd);free(lbd);
    thvm_free(ctx); mnist_free(&data);
    return 0;
}
