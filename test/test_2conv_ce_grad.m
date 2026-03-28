// test_2conv_ce_grad.m — 2-layer CNN + CE loss gradient check
// Conv(1,4,3)→ReLU→Conv(4,8,3)→ReLU→Flatten→Linear(8*24*24,10), CE loss

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
    u32 BS = 4, C1 = 4, C2 = 8;
    srand(42);
    TinyHVM *ctx = thvm_init("metal");

    u32 cw1_n=C1*1*3*3; f32 *cw1d=malloc(cw1_n*4);
    for(u32 i=0;i<cw1_n;i++) cw1d[i]=0.1f*((f32)rand()/(f32)RAND_MAX*2-1);
    Term cw1=thvm_tensor(ctx,cw1d,(Shape){.dims={C1,1,3,3},.rank=4});
    f32 *cb1d=calloc(C1,4); Term cb1=thvm_tensor(ctx,cb1d,SHAPE(C1));

    u32 cw2_n=C2*C1*3*3; f32 *cw2d=malloc(cw2_n*4);
    for(u32 i=0;i<cw2_n;i++) cw2d[i]=0.05f*((f32)rand()/(f32)RAND_MAX*2-1);
    Term cw2=thvm_tensor(ctx,cw2d,(Shape){.dims={C2,C1,3,3},.rank=4});
    f32 *cb2d=calloc(C2,4); Term cb2=thvm_tensor(ctx,cb2d,SHAPE(C2));

    u32 flat_f=C2*24*24;
    f32 *lwd=malloc(flat_f*10*4); f32 bound=1.0f/sqrtf((f32)flat_f);
    for(u32 i=0;i<flat_f*10;i++) lwd[i]=bound*((f32)rand()/(f32)RAND_MAX*2-1);
    Term lw=thvm_tensor(ctx,lwd,SHAPE(flat_f,10));
    f32 *lbd=calloc(10,4); Term lb=thvm_tensor(ctx,lbd,SHAPE(10));

    #define NP 6
    Term params[NP]={cw1,cb1,cw2,cb2,lw,lb};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,params[i]);

    {FILE*f=fopen("/tmp/c2ce_cw1.bin","wb");fwrite(cw1d,4,cw1_n,f);fclose(f);}
    {FILE*f=fopen("/tmp/c2ce_cw2.bin","wb");fwrite(cw2d,4,cw2_n,f);fclose(f);}
    {FILE*f=fopen("/tmp/c2ce_lw.bin","wb");fwrite(lwd,4,flat_f*10,f);fclose(f);}

    Term X=thvm_tensor(ctx,data.train_images,(Shape){.dims={BS,1,28,28},.rank=4});
    {FILE*f=fopen("/tmp/c2ce_X.bin","wb");fwrite(data.train_images,4,BS*784,f);fclose(f);}
    {FILE*f=fopen("/tmp/c2ce_Y.bin","wb");
     f32*oh=calloc(BS*10,4);for(u32 i=0;i<BS;i++)oh[i*10+data.train_labels[i]]=1;
     fwrite(oh,4,BS*10,f);free(oh);fclose(f);}

    u32 p0[]={0,0,0,0},s1[]={1,1};
    Term h1=thvm_conv2d(ctx,X,cw1,cb1,1,s1,p0);
    h1=thvm_op(ctx,UOP_RELU,h1,term_era());
    Term h2=thvm_conv2d(ctx,h1,cw2,cb2,1,s1,p0);
    h2=thvm_op(ctx,UOP_RELU,h2,term_era());
    Term flat=thvm_reshape(ctx,h2,SHAPE(BS,flat_f));
    Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,flat,lw),
        thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
    Term loss=cross_entropy_loss(ctx,logits,data.train_labels,BS,10);
    thvm_reduce(ctx,loss);
    printf("loss=%.6f\n",thvm_to_host(ctx,loss)[0]);

    const char*gn[NP]={"g_cw1","g_cb1","g_cw2","g_cb2","g_lw","g_lb"};
    u32 gsz[NP]={cw1_n,C1,cw2_n,C2,flat_f*10,10};
    for(u32 p=0;p<NP;p++){
        Term g=thvm_reduce(ctx,thvm_grad(ctx,loss,params[p]));
        if(term_tag(g)!=TAG_TEN){printf("%s: not TEN\n",gn[p]);continue;}
        f32*gv=thvm_to_host(ctx,g);
        char path[128];snprintf(path,128,"/tmp/c2ce_%s.bin",gn[p]);
        FILE*fp=fopen(path,"wb");fwrite(gv,4,gsz[p],fp);fclose(fp);
        printf("%s: [%.4e,%.4e,%.4e,%.4e]\n",gn[p],gv[0],gv[1],gv[2],gv[3]);
    }

    free(cw1d);free(cb1d);free(cw2d);free(cb2d);free(lwd);free(lbd);
    thvm_free(ctx);mnist_free(&data);
    return 0;
}
