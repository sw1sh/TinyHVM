// test_ic_grad_2conv_pool.m — Conv1→ReLU→Pool→Conv2→ReLU→Flatten→Linear + CE loss
// IC-native gradient, finite-difference check for ALL 6 params
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

static f32 fwd_loss(TinyHVM *ctx, f32 *xd, u32 BS,
    f32 *cw1d, f32 *cb1d, f32 *cw2d, f32 *cb2d, f32 *lwd, f32 *lbd, u8 *labels) {
    thvm_reset(ctx, 0);
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,1,28,28},.rank=4});
    Term cw1=thvm_tensor(ctx,cw1d,(Shape){.dims={8,1,3,3},.rank=4});
    Term cb1=thvm_tensor(ctx,cb1d,SHAPE(8));
    Term cw2=thvm_tensor(ctx,cw2d,(Shape){.dims={16,8,3,3},.rank=4});
    Term cb2=thvm_tensor(ctx,cb2d,SHAPE(16));
    u32 flat_f=16*5*5;
    Term lw=thvm_tensor(ctx,lwd,SHAPE(flat_f,10));
    Term lb=thvm_tensor(ctx,lbd,SHAPE(10));
    u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
    Term h=thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_reshape(ctx,h,SHAPE(BS,flat_f));
    Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
        thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
    Term loss=cross_entropy_loss(ctx,logits,labels,BS,10);
    return thvm_to_host(ctx,loss)[0];
}

int main(void) {
    srand(42);
    MNISTData data=mnist_load("data");
    TinyHVM *ctx=thvm_init(thvm_device("metal"));
    u32 BS=4;

    u32 sizes[]={72,8,1152,16,400*10,10};
    const char *names[]={"cw1","cb1","cw2","cb2","lw","lb"};
    f32 *wd[6];
    f32 bounds[]={1.0f/3,0,1.0f/sqrtf(72),0,1.0f/sqrtf(400),0};
    for(int p=0;p<6;p++){wd[p]=malloc(sizes[p]*4);
        if(bounds[p]==0){memset(wd[p],0,sizes[p]*4);continue;}
        for(u32 i=0;i<sizes[p];i++) wd[p][i]=bounds[p]*((f32)rand()/(f32)RAND_MAX*2-1);}

    f32 *xd=malloc(BS*784*4); memcpy(xd,data.train_images,BS*784*4);
    u8 *labels=data.train_labels;

    // Forward + IC gradient
    thvm_reset(ctx,0);
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,1,28,28},.rank=4});
    thvm_set_requires_grad(ctx,x);
    Shape shapes[]={{.dims={8,1,3,3},.rank=4},{.dims={8},.rank=1},
                    {.dims={16,8,3,3},.rank=4},{.dims={16},.rank=1},
                    {.dims={400,10},.rank=2},{.dims={10},.rank=1}};
    Term params[6];
    for(int i=0;i<6;i++){params[i]=thvm_tensor(ctx,wd[i],shapes[i]);
        thvm_set_requires_grad(ctx,params[i]);}

    u32 flat_f=16*5*5;
    u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
    Term h=thvm_conv2d(ctx,x,params[0],params[1],1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_conv2d(ctx,h,params[2],params[3],1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_reshape(ctx,h,SHAPE(BS,flat_f));
    Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,params[4]),
        thvm_expand(ctx,thvm_reshape(ctx,params[5],SHAPE(1,10)),SHAPE(BS,10)));
    Term loss=cross_entropy_loss(ctx,logits,labels,BS,10);

    Term gs[6];
    for(int i=0;i<6;i++){f32*z=calloc(sizes[i],4);
        gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
    Term gt=thvm_grad_multi(ctx,loss,params,gs,6);
    thvm_reduce(ctx,thvm_app(ctx,gt,term_era()));

    f32 lv=thvm_to_host(ctx,loss)[0];
    printf("loss=%.6f\n",lv);

    // Read ALL gradients BEFORE numerical loop
    f32 *ag[6];
    for(int i=0;i<6;i++){ag[i]=malloc(sizes[i]*4);
        f32*g=thvm_to_host(ctx,gs[i]);
        if(g) memcpy(ag[i],g,sizes[i]*4); else memset(ag[i],0,sizes[i]*4);}

    // Numerical gradient — sample 3 elements per param
    f32 eps=1e-3f;
    int all_ok=1;
    for(int p=0;p<6;p++){
        f32 max_d=0;
        u32 n_check=sizes[p]<6?sizes[p]:3;
        u32 stride=sizes[p]/n_check;
        for(u32 j=0;j<n_check;j++){
            u32 idx=j*stride; f32 saved=wd[p][idx];
            wd[p][idx]=saved+eps;
            f32 lp=fwd_loss(ctx,xd,BS,wd[0],wd[1],wd[2],wd[3],wd[4],wd[5],labels);
            wd[p][idx]=saved-eps;
            f32 lm=fwd_loss(ctx,xd,BS,wd[0],wd[1],wd[2],wd[3],wd[4],wd[5],labels);
            wd[p][idx]=saved;
            f32 ng=(lp-lm)/(2*eps),d=fabsf(ag[p][idx]-ng);
            if(d>max_d) max_d=d;
        }
        // Use relative threshold for large gradients
        f32 max_grad = 0;
        for (u32 j2=0;j2<n_check;j2++) { u32 i2=j2*stride;
            if (fabsf(ag[p][i2])>max_grad) max_grad=fabsf(ag[p][i2]); }
        f32 thr = (max_grad > 1.0f) ? max_grad * 1e-3f : 0.01f;
        int ok=max_d<thr;
        printf("  %s: max_abs=%.2e max_grad=%.2e rel=%.2e %s\n",
            names[p],max_d,max_grad,max_grad>0?max_d/max_grad:0,ok?"OK":"FAIL");
        if(!ok) all_ok=0;
    }
    printf("\n%s\n",all_ok?"ALL PASS":"FAIL");
    for(int i=0;i<6;i++){free(wd[i]);free(ag[i]);}
    free(xd); thvm_free(ctx);
    return all_ok?0:1;
}
