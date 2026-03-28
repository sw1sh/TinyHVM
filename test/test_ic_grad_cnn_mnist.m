// test_ic_grad_cnn_mnist.m — 2-conv+2pool+linear+CE gradient on MNIST
// Conv(8,1,3,3)→ReLU→Pool(2)→Conv(16,8,3,3)→ReLU→Pool(2)→FC(400,10)→CE
// IC-native gradient (thvm_grad_multi), finite-difference check for ALL 6 params
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include "../src/nn/datasets.c"
#include "train_helpers.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static f32 run_fwd(TinyHVM *ctx, f32 *xd, u32 BS, f32 **wd, u8 *labels) {
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,1,28,28},.rank=4});
    Term cw1=thvm_tensor(ctx,wd[0],(Shape){.dims={8,1,3,3},.rank=4});
    Term cb1=thvm_tensor(ctx,wd[1],SHAPE(8));
    Term cw2=thvm_tensor(ctx,wd[2],(Shape){.dims={16,8,3,3},.rank=4});
    Term cb2=thvm_tensor(ctx,wd[3],SHAPE(16));
    Term lw=thvm_tensor(ctx,wd[4],SHAPE(400,10));
    Term lb=thvm_tensor(ctx,wd[5],SHAPE(10));
    u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
    Term h=thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_reshape(ctx,h,SHAPE(BS,400));
    Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
        thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
    return thvm_to_host(ctx,cross_entropy_loss(ctx,logits,labels,BS,10))[0];
}

int main(void) {
    srand(42);
    MNISTData data=mnist_load("data");
    TinyHVM *ctx=thvm_init(thvm_device("metal"));
    u32 BS=4;

    u32 sizes[]={72,8,1152,16,4000,10};
    const char *names[]={"cw1","cb1","cw2","cb2","lw","lb"};
    Shape shapes[]={{.dims={8,1,3,3},.rank=4},{.dims={8},.rank=1},
                    {.dims={16,8,3,3},.rank=4},{.dims={16},.rank=1},
                    {.dims={400,10},.rank=2},{.dims={10},.rank=1}};
    f32 *wd[6];
    f32 bounds[]={1.0f/3,0,1.0f/sqrtf(72),0,1.0f/sqrtf(400),0};
    for(int i=0;i<6;i++){wd[i]=malloc(sizes[i]*4);
        if(!bounds[i]){memset(wd[i],0,sizes[i]*4);continue;}
        for(u32 j=0;j<sizes[i];j++) wd[i][j]=bounds[i]*((f32)rand()/(f32)RAND_MAX*2-1);}

    f32 *xd=malloc(BS*784*4); memcpy(xd,data.train_images,BS*784*4);
    u8 *labels=data.train_labels;

    // Analytic gradient
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,1,28,28},.rank=4});
    thvm_set_requires_grad(ctx,x);
    Term params[6];
    for(int i=0;i<6;i++){params[i]=thvm_tensor(ctx,wd[i],shapes[i]);
        thvm_set_requires_grad(ctx,params[i]);}
    u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
    Term h=thvm_conv2d(ctx,x,params[0],params[1],1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_conv2d(ctx,h,params[2],params[3],1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_reshape(ctx,h,SHAPE(BS,400));
    Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,params[4]),
        thvm_expand(ctx,thvm_reshape(ctx,params[5],SHAPE(1,10)),SHAPE(BS,10)));
    Term loss=cross_entropy_loss(ctx,logits,labels,BS,10);

    Term gs[6];
    for(int i=0;i<6;i++){f32*z=calloc(sizes[i],4);
        gs[i]=thvm_tensor(ctx,z,shapes[i]);free(z);}
    thvm_reduce(ctx,thvm_app(ctx,thvm_grad_multi(ctx,loss,params,gs,6),term_era()));
    printf("loss=%.6f\n",thvm_to_host(ctx,loss)[0]);

    // Copy ALL gradients before numerical loop
    f32 *ag[6];
    for(int i=0;i<6;i++){ag[i]=malloc(sizes[i]*4);
        f32*g=thvm_to_host(ctx,gs[i]);
        if(g)memcpy(ag[i],g,sizes[i]*4);else memset(ag[i],0,sizes[i]*4);}

    // Numerical gradient — 3 samples per param
    f32 eps=1e-3f, atol=1e-3f, rtol=0.05f;
    int all_ok=1;
    for(int p=0;p<6;p++){
        f32 max_d=0,max_g=0;
        u32 nc=sizes[p]<6?sizes[p]:3;
        u32 stride=sizes[p]/nc;
        for(u32 j=0;j<nc;j++){
            u32 idx=j*stride; f32 saved=wd[p][idx];
            wd[p][idx]=saved+eps; thvm_reset(ctx,0);
            f32 lp=run_fwd(ctx,xd,BS,wd,labels);
            wd[p][idx]=saved-eps; thvm_reset(ctx,0);
            f32 lm=run_fwd(ctx,xd,BS,wd,labels);
            wd[p][idx]=saved;
            f32 ng=(lp-lm)/(2*eps),d=fabsf(ag[p][idx]-ng);
            if(d>max_d)max_d=d; if(fabsf(ng)>max_g)max_g=fabsf(ng);
            if(p<2) printf("    [%u] a=%.6f n=%.6f d=%.2e\n",idx,ag[p][idx],ng,d);
        }
        f32 rel=max_g>atol?max_d/max_g:0;
        int ok=(max_d<atol)||(rel<rtol);
        printf("  %s: abs=%.2e rel=%.2e %s\n",names[p],max_d,rel,ok?"OK":"FAIL");
        if(!ok) all_ok=0;
    }
    printf("\n%s\n",all_ok?"ALL PASS":"FAIL");
    for(int i=0;i<6;i++){free(wd[i]);free(ag[i]);}
    free(xd);thvm_free(ctx);
    return all_ok?0:1;
}
