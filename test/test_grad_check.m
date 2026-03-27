// test_grad_check.m — Compare TinyHVM gradients against numpy
// Skip MLP: h=relu(X@W1+B1), out=h@W2+B2+expand(sum(flatten(h)))
// Saves grads as bin, then python compares.

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
    u32 BS = 128, H = 128;
    srand(12345); // fixed seed for reproducible init
    TinyHVM *ctx = thvm_init(thvm_device("metal"));

    f32 *w1d = malloc(784*H*4);
    for (u32 i=0;i<784*H;i++) w1d[i] = 0.001f*((f32)rand()/(f32)RAND_MAX*2-1);
    f32 *b1d = calloc(H, 4);
    f32 *w2d = malloc(H*10*4);
    for (u32 i=0;i<H*10;i++) w2d[i] = 0.001f*((f32)rand()/(f32)RAND_MAX*2-1);
    f32 *b2d = calloc(10, 4);

    // Save init weights
    { FILE *f=fopen("/tmp/gc_W1.bin","wb"); fwrite(w1d,4,784*H,f); fclose(f); }
    { FILE *f=fopen("/tmp/gc_W2.bin","wb"); fwrite(w2d,4,H*10,f); fclose(f); }

    Term W1=thvm_tensor(ctx,w1d,SHAPE(784,H));
    Term B1=thvm_tensor(ctx,b1d,SHAPE(1,H));
    Term W2=thvm_tensor(ctx,w2d,SHAPE(H,10));
    Term B2=thvm_tensor(ctx,b2d,SHAPE(1,10));
    thvm_set_requires_grad(ctx,W1); thvm_set_requires_grad(ctx,B1);
    thvm_set_requires_grad(ctx,W2); thvm_set_requires_grad(ctx,B2);

    Term X=thvm_tensor(ctx,data.train_images,SHAPE(BS,784));
    f32 *oh=calloc(BS*10,4);
    for(u32 i=0;i<BS;i++) oh[i*10+data.train_labels[i]]=1.0f;
    Term Y=thvm_tensor(ctx,oh,SHAPE(BS,10)); free(oh);

    // Forward: skip MLP
    Term z1=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,X,W1),thvm_expand(ctx,B1,SHAPE(BS,H)));
    Term h=thvm_op(ctx,UOP_RELU,z1,term_era());
    Term hs=thvm_reshape(ctx,thvm_op(ctx,UOP_SUM,thvm_reshape(ctx,h,SHAPE(BS*H)),term_era()),SHAPE(1,1));
    Term out=thvm_op(ctx,UOP_ADD,
        thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,W2),thvm_expand(ctx,B2,SHAPE(BS,10))),
        thvm_expand(ctx,hs,SHAPE(BS,10)));

    // Cross-entropy
    Term loss=cross_entropy_loss(ctx,out,data.train_labels,BS,10);

    // GRAD — reduce each directly, clearing DUP state between walks
    Term params[]={W1,B1,W2,B2};
    Term grads[4];
    for(u32 i=0;i<4;i++) {
        // Clear all DUP grad slots before each walk
        for(u32 j=0;j<ctx->tensor_count;j++)
            if(ctx->tensors[j].dup_loc)
                heap_set(ctx, ctx->tensors[j].dup_loc + 1, term_era());
        grads[i]=thvm_reduce(ctx,thvm_grad(ctx,loss,params[i]));
    }

    printf("loss=%.6f\n",thvm_to_host(ctx,loss)[0]);

    // Save grads
    const char *gn[]={"gW1","gB1","gW2","gB2"};
    u32 gsz[]={784*H, H, H*10, 10};
    for(u32 p=0;p<4;p++){
        Term g=grads[p];
        if(term_tag(g)!=TAG_TEN) g=thvm_reduce(ctx,g);
        if(term_tag(g)!=TAG_TEN){printf("%s: not TEN\n",gn[p]);continue;}
        f32 *gv=thvm_to_host(ctx,g);
        char path[128]; snprintf(path,128,"/tmp/gc_%s.bin",gn[p]);
        FILE *fp=fopen(path,"wb"); fwrite(gv,4,gsz[p],fp); fclose(fp);
        printf("%s: first4=[%.6f,%.6f,%.6f,%.6f]\n",gn[p],gv[0],gv[1],gv[2],gv[3]);
    }

    // Save input data for numpy
    { FILE *f=fopen("/tmp/gc_X.bin","wb"); fwrite(data.train_images,4,BS*784,f); fclose(f); }
    { FILE *f=fopen("/tmp/gc_Y.bin","wb");
      f32 *y2=calloc(BS*10,4);
      for(u32 i=0;i<BS;i++) y2[i*10+data.train_labels[i]]=1;
      fwrite(y2,4,BS*10,f); free(y2); fclose(f); }

    free(w1d);free(b1d);free(w2d);free(b2d);
    thvm_free(ctx); mnist_free(&data);
    return 0;
}
