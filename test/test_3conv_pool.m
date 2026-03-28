// test_3conv_pool.m — 3-conv CNN with MaxPool
// Conv(1,8,3,p=0)→ReLU→MaxPool(2,2) → [BS,8,13,13]
// Conv(8,16,3,p=0)→ReLU→MaxPool(2,2) → [BS,16,5,5]
// Conv(16,32,3,p=1)→ReLU            → [BS,32,5,5]
// Flatten → Linear(32*5*5, 10)

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
#include <time.h>

static Term make_w(TinyHVM *c, Shape s, u32 fi) {
    u32 n=1; for(u32 i=0;i<s.rank;i++) n*=s.dims[i];
    f32 *d=malloc(n*4); f32 b=1.0f/sqrtf((f32)fi);
    for(u32 i=0;i<n;i++) d[i]=b*((f32)rand()/(f32)RAND_MAX*2-1);
    Term t=thvm_tensor(c,d,s); free(d); return t;
}
static Term make_z(TinyHVM *c, u32 n) { f32*z=calloc(n,4); Term t=thvm_tensor(c,z,SHAPE(n)); free(z); return t; }

int main(void) {
    srand(42);
    MNISTData data = mnist_load("data");
    TinyHVM *ctx = thvm_init(thvm_device("metal"));
    u32 BS = 64;

    // Layer 1: Conv(1,8,3,p=0)
    Term cw1=make_w(ctx,(Shape){.dims={8,1,3,3},.rank=4},9);
    Term cb1=make_z(ctx,8);
    // Layer 2: Conv(8,16,3,p=0)
    Term cw2=make_w(ctx,(Shape){.dims={16,8,3,3},.rank=4},72);
    Term cb2=make_z(ctx,16);
    // Layer 3: Conv(16,32,3,p=1) — padding=1 keeps spatial at 5x5
    Term cw3=make_w(ctx,(Shape){.dims={32,16,3,3},.rank=4},144);
    Term cb3=make_z(ctx,32);
    // Linear: 32*5*5 → 10
    u32 flat_f=32*5*5;
    Term lw=make_w(ctx,SHAPE(flat_f,10),flat_f);
    Term lb=make_z(ctx,10);

    #define NP 8
    Term params[NP]={cw1,cb1,cw2,cb2,cw3,cb3,lw,lb};
    for(u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx,params[i]);
    Term train_data=thvm_tensor(ctx,data.train_images,(Shape){.dims={data.n_train,1,28,28},.rank=4});
    u32 n_weights=ctx->tensor_count;
    u32 n_steps=100;

    struct timespec t_start; clock_gettime(CLOCK_MONOTONIC,&t_start);
    for(u32 step=0;step<n_steps;step++){
      @autoreleasepool {
        u32 bi=step%(data.n_train/BS);
        Term x=thvm_shrink(ctx,train_data,(u32[]){bi*BS,(bi+1)*BS,0,1,0,28,0,28},4);
        thvm_set_requires_grad(ctx,x);

        u32 p0[]={0,0,0,0}, p1[]={1,1,1,1};
        u32 s1[]={1,1}, k2[]={2,2}, s2[]={2,2};

        // Conv1 → ReLU → MaxPool  [BS,8,13,13]
        Term h=thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        h=thvm_maxpool2d(ctx,h,k2,s2);

        // Conv2 → ReLU → MaxPool  [BS,16,5,5]
        h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        h=thvm_maxpool2d(ctx,h,k2,s2);

        // Conv3 → ReLU  [BS,32,5,5]  (padding=1 keeps 5x5)
        h=thvm_conv2d(ctx,h,cw3,cb3,1,s1,p1);
        h=thvm_op(ctx,UOP_RELU,h,term_era());

        // Flatten → Linear
        h=thvm_reshape(ctx,h,SHAPE(BS,flat_f));
        Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
            thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(BS,10)));
        Term loss=cross_entropy_loss(ctx,logits,&data.train_labels[bi*BS],BS,10);

        // Grad slots + one reduce
        Term gs[NP];
        u32 psz[]={8*9,8,16*8*9,16,32*16*9,32,flat_f*10,10};
        for(int i=0;i<NP;i++){f32*z=calloc(psz[i],4);
            gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);free(z);}
        Term grad_term=thvm_grad_multi(ctx,loss,params,gs,NP);
        f32 lr=0.001f; Term lr_t=thvm_tensor(ctx,&lr,SHAPE(1));
        Term sgd=term_era();
        for(int i=NP-1;i>=0;i--)
            sgd=thvm_app(ctx,thvm_assign(ctx,params[i],thvm_op(ctx,UOP_SUB,params[i],
                thvm_op(ctx,UOP_MUL,lr_t,gs[i]))),sgd);

        // On step 0: dump weights BEFORE the reduce (pre-SGD) and dump X, Y
        if(step==0){
            const char*wnames[NP]={"gc3_cw1","gc3_cb1","gc3_cw2","gc3_cb2","gc3_cw3","gc3_cb3","gc3_lw","gc3_lb"};
            u32 wsz[NP]={8*1*3*3,8,16*8*3*3,16,32*16*3*3,32,flat_f*10,10};
            for(u32 i=0;i<NP;i++){
                f32*wv=thvm_to_host(ctx,params[i]);
                char path[128];snprintf(path,128,"/tmp/%s.bin",wnames[i]);
                FILE*fp=fopen(path,"wb");fwrite(wv,4,wsz[i],fp);fclose(fp);
            }
            // Dump input batch X and one-hot labels Y
            {FILE*fp=fopen("/tmp/gc3_X.bin","wb");
             fwrite(data.train_images,4,BS*1*28*28,fp);fclose(fp);}
            {f32*oh=calloc(BS*10,4);
             for(u32 i=0;i<BS;i++)oh[i*10+data.train_labels[i]]=1.0f;
             FILE*fp=fopen("/tmp/gc3_Y.bin","wb");fwrite(oh,4,BS*10,fp);fclose(fp);free(oh);}
        }

        thvm_reduce(ctx,thvm_app(ctx,grad_term,sgd));

        f32 lv=thvm_to_host(ctx,loss)[0];
        extern u32 total_dispatches;
        extern void print_dispatch_breakdown(void);
        if(step<3||step%20==0||step==n_steps-1)
            printf("step %3u: loss=%.4f dispatches=%u\n",step,lv,total_dispatches);
        if(step==1) print_dispatch_breakdown();

        // On step 0: dump gradients AFTER reduce (gs[] are now populated)
        if(step==0){
            printf("loss=%.6f\n",lv);
            const char*gnames[NP]={"gc3_g_cw1","gc3_g_cb1","gc3_g_cw2","gc3_g_cb2","gc3_g_cw3","gc3_g_cb3","gc3_g_lw","gc3_g_lb"};
            u32 wsz[NP]={8*1*3*3,8,16*8*3*3,16,32*16*3*3,32,flat_f*10,10};
            for(u32 i=0;i<NP;i++){
                f32*gv=thvm_to_host(ctx,gs[i]);
                char path[128];snprintf(path,128,"/tmp/%s.bin",gnames[i]);
                FILE*fp=fopen(path,"wb");fwrite(gv,4,wsz[i],fp);fclose(fp);
                printf("%s: [%.4e,%.4e,%.4e,%.4e]\n",gnames[i],gv[0],gv[1],gv[2],gv[3]);
            }
        }

        total_dispatches=0;
        for(u32 i=0;i<NP;i++){u32 pid=(u32)term_val(params[i]);
            if(ctx->tensors[pid].host_ptr){free(ctx->tensors[pid].host_ptr);ctx->tensors[pid].host_ptr=NULL;}}
        thvm_reset(ctx,n_weights);
      }
    }
    struct timespec t_end; clock_gettime(CLOCK_MONOTONIC,&t_end);
    f32 total_s=(f32)(t_end.tv_sec-t_start.tv_sec)+(f32)(t_end.tv_nsec-t_start.tv_nsec)/1e9f;
    printf("\n%.0fms/step avg\n",total_s*1000/n_steps);

    // Eval
    Term test_data=thvm_tensor(ctx,data.test_images,(Shape){.dims={data.n_test,1,28,28},.rank=4});
    u32 ek=ctx->tensor_count;
    u32 correct=0, tbs=64, tb=data.n_test/tbs;
    for(u32 b=0;b<tb;b++){
        Term x=thvm_shrink(ctx,test_data,(u32[]){b*tbs,(b+1)*tbs,0,1,0,28,0,28},4);
        u32 p0[]={0,0,0,0}, p1[]={1,1,1,1};
        u32 s1[]={1,1}, k2[]={2,2}, s2[]={2,2};
        Term h=thvm_conv2d(ctx,x,cw1,cb1,1,s1,p0);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        h=thvm_maxpool2d(ctx,h,k2,s2);
        h=thvm_conv2d(ctx,h,cw2,cb2,1,s1,p0);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        h=thvm_maxpool2d(ctx,h,k2,s2);
        h=thvm_conv2d(ctx,h,cw3,cb3,1,s1,p1);
        h=thvm_op(ctx,UOP_RELU,h,term_era());
        h=thvm_reshape(ctx,h,SHAPE(tbs,flat_f));
        Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
            thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,10)),SHAPE(tbs,10)));
        f32 acc=thvm_eval_accuracy(ctx,logits,&data.test_labels[b*tbs],tbs,10);
        correct+=(u32)(acc*(f32)tbs/100.0f);
        thvm_reset(ctx,ek);
    }
    printf("Test accuracy: %.1f%% (%u/%u)\n",100.0f*(f32)correct/(f32)(tb*tbs),correct,tb*tbs);
    thvm_free(ctx); mnist_free(&data);
    return 0;
}
