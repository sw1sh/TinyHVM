// test_defer_debug.m — Debug deferred dispatch: 2-step conv+pool training
// Compare weight values after step 0 SGD update
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

int main(void) {
    srand(42);
    MNISTData data = mnist_load("data");
    TinyHVM *ctx = thvm_init("metal");
    u32 BS=64;

    Term cw1=({u32 n=8*1*3*3;f32*d=malloc(n*4);f32 b=1.0f/3;
        for(u32 i=0;i<n;i++)d[i]=b*((f32)rand()/(f32)RAND_MAX*2-1);
        Term t=thvm_tensor(ctx,d,(Shape){.dims={8,1,3,3},.rank=4});free(d);t;});
    Term cb1=({f32*z=calloc(8,4);Term t=thvm_tensor(ctx,z,SHAPE(8));free(z);t;});
    u32 flat_f=8*13*13;
    Term lw=({u32 n=flat_f*10;f32*d=malloc(n*4);f32 b=1.0f/sqrtf((f32)flat_f);
        for(u32 i=0;i<n;i++)d[i]=b*((f32)rand()/(f32)RAND_MAX*2-1);
        Term t=thvm_tensor(ctx,d,SHAPE(flat_f,10));free(d);t;});
    Term lb=({f32*z=calloc(10,4);Term t=thvm_tensor(ctx,z,SHAPE(10));free(z);t;});

    #define NP 4
    Term params[NP]={cw1,cb1,lw,lb};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,params[i]);
    Term train_data=thvm_tensor(ctx,data.train_images,(Shape){.dims={data.n_train,1,28,28},.rank=4});
    u32 n_weights=ctx->tensor_count;

    // Step 0
    {
        Term x=thvm_shrink(ctx,train_data,(u32[]){0,BS,0,1,0,28,0,28},4);
        thvm_set_requires_grad(ctx,x);
        u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
        Term h=thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        h=thvm_maxpool2d(ctx,h,k2,s2);
        h=thvm_reshape(ctx,h,SHAPE(BS,flat_f));
        Term logits=thvm_op(ctx,UOP_ADD,thvm_mm(ctx,h,lw),
            thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
        Term loss=cross_entropy_loss(ctx,logits,data.train_labels,BS,10);

        u32 psz[]={72,8,flat_f*10,10};
        Term gs[NP];
        for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
            gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
        Term grad_term=thvm_grad_multi(ctx,loss,params,gs,NP);
        f32 lr=0.001f; Term lr_t=thvm_tensor(ctx,&lr,SHAPE(1));
        Term sgd=term_era();
        for(int i=NP-1;i>=0;i--)
            sgd=thvm_app(ctx,thvm_assign(ctx,params[i],thvm_op(ctx,UOP_SUB,params[i],
                thvm_op(ctx,UOP_MUL,lr_t,gs[i]))),sgd);
        thvm_reduce(ctx,thvm_app(ctx,grad_term,sgd));

        f32 loss_val=thvm_to_host(ctx,loss)[0];
        printf("step 0: loss=%.4f\n",loss_val);

        // Print first 4 values of conv weight after SGD update
        f32 *w=thvm_to_host(ctx,cw1);
        printf("cw1 after SGD: [%.8f,%.8f,%.8f,%.8f]\n",w[0],w[1],w[2],w[3]);

        for(u32 i=0;i<NP;i++){u32 pid=(u32)term_val(params[i]);
            if(ctx->tensors[pid].host_ptr){free(ctx->tensors[pid].host_ptr);ctx->tensors[pid].host_ptr=NULL;}}
        thvm_reset(ctx,n_weights);
    }

    // Step 1 forward only — check loss
    {
        Term x=thvm_shrink(ctx,train_data,(u32[]){BS,2*BS,0,1,0,28,0,28},4);
        u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
        Term h=thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        h=thvm_maxpool2d(ctx,h,k2,s2);
        h=thvm_reshape(ctx,h,SHAPE(BS,flat_f));
        Term logits=thvm_op(ctx,UOP_ADD,thvm_mm(ctx,h,lw),
            thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
        Term loss=cross_entropy_loss(ctx,logits,&data.train_labels[BS],BS,10);
        printf("step 1: loss=%.4f\n",thvm_to_host(ctx,loss)[0]);
    }

    thvm_free(ctx);
    return 0;
}
