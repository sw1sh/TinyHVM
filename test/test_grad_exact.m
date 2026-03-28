// test_grad_exact.m — Verify EXACT ZERO gradient diff against numpy
// Simple MLP: out = relu(X@W1+B1) @ W2 + B2, cross-entropy loss
// NO skip connection, NO diamond. Expect exact match.

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
    u32 BS = 32, H = 64;
    srand(12345);
    TinyHVM *ctx = thvm_init("metal");

    f32 *w1d = malloc(784*H*4);
    for (u32 i=0;i<784*H;i++) w1d[i] = 0.001f*((f32)rand()/(f32)RAND_MAX*2-1);
    f32 *b1d = calloc(H, 4);
    f32 *w2d = malloc(H*10*4);
    for (u32 i=0;i<H*10;i++) w2d[i] = 0.001f*((f32)rand()/(f32)RAND_MAX*2-1);
    f32 *b2d = calloc(10, 4);

    { FILE *f=fopen("/tmp/gc_W1.bin","wb"); fwrite(w1d,4,784*H,f); fclose(f); }
    { FILE *f=fopen("/tmp/gc_W2.bin","wb"); fwrite(w2d,4,H*10,f); fclose(f); }

    Term W1=thvm_tensor(ctx,w1d,SHAPE(784,H));
    Term B1=thvm_tensor(ctx,b1d,SHAPE(1,H));
    Term W2=thvm_tensor(ctx,w2d,SHAPE(H,10));
    Term B2=thvm_tensor(ctx,b2d,SHAPE(1,10));
    thvm_set_requires_grad(ctx,W1); thvm_set_requires_grad(ctx,B1);
    thvm_set_requires_grad(ctx,W2); thvm_set_requires_grad(ctx,B2);

    Term X=thvm_tensor(ctx,data.train_images,SHAPE(BS,784));

    // Simple forward: NO skip, NO diamond
    Term z1=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,X,W1),thvm_expand(ctx,B1,SHAPE(BS,H)));
    Term h=thvm_op(ctx,UOP_RELU,z1,term_era());
    Term out=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,W2),thvm_expand(ctx,B2,SHAPE(BS,10)));

    Term loss=cross_entropy_loss(ctx,out,data.train_labels,BS,10);
    thvm_reduce(ctx, loss);

    printf("loss=%.6f\n",thvm_to_host(ctx,loss)[0]);

    #define NP 4
    Term params[]={W1,B1,W2,B2};
    const char *gn[]={"gW1","gB1","gW2","gB2"};
    u32 gsz[]={784*H, H, H*10, 10};

    // Use thvm_grad_multi (single GRAD term for all params) to avoid
    // heap corruption from sequential thvm_grad + thvm_reduce calls.
    Term gs[NP];
    for(u32 p=0;p<NP;p++){
        f32*z=calloc(gsz[p],4);
        gs[p]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[p])].view.shape);
        free(z);
    }
    // Trace full provenance tree from loss
    { Term lr = thvm_reduce(ctx, loss);
      u32 lid = (u32)term_val(lr);
      fprintf(stderr, "PROVENANCE from loss=%u:\n", lid);
      u32 q[64]; u32 qn=0; q[qn++]=lid;
      u32 visited[64]={0}; u32 vn=0;
      while(qn>0 && vn<40) {
          u32 id=q[--qn];
          int skip=0; for(u32 i=0;i<vn;i++) if(visited[i]==id){skip=1;break;}
          if(skip) continue; visited[vn++]=id;
          TensorMeta *tm=&ctx->tensors[id];
          fprintf(stderr,"  [%u] cop=%u(%s) src=[%u,%u] rg=%d buf=%u\n",
              id,tm->creator_op,tm->creator_op<UOP_COUNT?uop_names[tm->creator_op]:"?",
              tm->src_ids[0],tm->src_ids[1],tm->requires_grad,tm->buf_id);
          if(tm->creator_op && tm->src_ids[0] && qn<62) q[qn++]=tm->src_ids[0];
          if(tm->creator_op && tm->src_ids[1] && qn<62) q[qn++]=tm->src_ids[1];
      }
    }

    Term gt=thvm_grad_multi(ctx,loss,params,gs,NP);
    thvm_reduce(ctx,thvm_app(ctx,gt,term_era()));

    // Force GPU flush before reading gradients
    if (ctx->backends[1] && ctx->backends[1]->end_batch) ctx->backends[1]->end_batch();

    for(u32 p=0;p<NP;p++){
        u32 gid = (u32)term_val(gs[p]);
        f32 *gv=thvm_to_host(ctx,gs[p]);
        if(!gv){printf("%s: NULL\n",gn[p]);continue;}
        char path[128]; snprintf(path,128,"/tmp/gc_%s.bin",gn[p]);
        FILE *fp=fopen(path,"wb"); fwrite(gv,4,gsz[p],fp); fclose(fp);
        f32 mx=0; for(u32 i=0;i<gsz[p];i++){f32 a=fabsf(gv[i]);if(a>mx)mx=a;}
        printf("%s: first4=[%.8e,%.8e,%.8e,%.8e] maxabs=%.8e\n",gn[p],gv[0],gv[1],gv[2],gv[3],mx);
    }

    { FILE *f=fopen("/tmp/gc_X.bin","wb"); fwrite(data.train_images,4,BS*784,f); fclose(f); }
    { FILE *f=fopen("/tmp/gc_Y.bin","wb");
      f32 *y2=calloc(BS*10,4);
      for(u32 i=0;i<BS;i++) y2[i*10+data.train_labels[i]]=1;
      fwrite(y2,4,BS*10,f); free(y2); fclose(f); }

    // Also save B1, B2 init for numpy
    { FILE *f=fopen("/tmp/gc_B1.bin","wb"); fwrite(b1d,4,H,f); fclose(f); }
    { FILE *f=fopen("/tmp/gc_B2.bin","wb"); fwrite(b2d,4,10,f); fclose(f); }

    free(w1d);free(b1d);free(w2d);free(b2d);
    thvm_free(ctx); mnist_free(&data);
    return 0;
}
