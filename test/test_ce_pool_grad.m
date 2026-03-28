// test_ce_pool_grad.m — Conv+ReLU+Pool+Linear+CE gradient check
// IC-native (thvm_grad_multi), finite-difference verification
// Usage: ./test_ce_pool_grad [BS] [IH]
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>
#include <string.h>

// Forward pass — builds graph, reduces, returns loss scalar
static f32 run_fwd(TinyHVM *ctx, f32 *xd, f32 *w1d, f32 *b1d, f32 *lwd, f32 *lbd,
                    u8 *labels, u32 BS, u32 IH, u32 IW, u32 flat_f, u32 NC) {
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,1,IH,IW},.rank=4});
    Term w1=thvm_tensor(ctx,w1d,(Shape){.dims={2,1,3,3},.rank=4});
    Term b1=thvm_tensor(ctx,b1d,SHAPE(2));
    u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
    Term h=thvm_conv2d(ctx,x,w1,b1,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_reshape(ctx,h,SHAPE(BS,flat_f));
    Term lw=thvm_tensor(ctx,lwd,SHAPE(flat_f,NC));
    Term lb=thvm_tensor(ctx,lbd,SHAPE(NC));
    Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
        thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,NC)),SHAPE(BS,NC)));
    return thvm_to_host(ctx,cross_entropy_loss(ctx,logits,labels,BS,NC))[0];
}

int main(int argc, char **argv) {
    srand(42);
    TinyHVM *ctx=thvm_init(thvm_device("metal"));
    u32 BS=(argc>1)?(u32)atoi(argv[1]):2;
    u32 IH=(argc>2)?(u32)atoi(argv[2]):6, IW=IH;
    u32 OH=(IH-2)/2, OW=(IW-2)/2, flat_f=2*OH*OW, NC=4;

    f32 *xd=malloc(BS*IH*IW*4);
    for(u32 i=0;i<BS*IH*IW;i++) xd[i]=0.1f*(f32)(i%(IH*IW)+1);
    f32 w1d[18]; for(int i=0;i<18;i++) w1d[i]=0.1f*((f32)rand()/(f32)RAND_MAX*2-1);
    f32 b1d[2]={0};
    f32 *lwd=malloc(flat_f*NC*4);
    for(u32 i=0;i<flat_f*NC;i++) lwd[i]=0.1f*((f32)rand()/(f32)RAND_MAX*2-1)/sqrtf((f32)flat_f);
    f32 lbd[4]={0};
    u8 *labels=malloc(BS); for(u32 i=0;i<BS;i++) labels[i]=(u8)(i%NC);

    // Analytic gradient via IC
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,1,IH,IW},.rank=4});
    thvm_set_requires_grad(ctx,x);
    Term w1=thvm_tensor(ctx,w1d,(Shape){.dims={2,1,3,3},.rank=4});
    thvm_set_requires_grad(ctx,w1);
    Term b1=thvm_tensor(ctx,b1d,SHAPE(2));
    Term lw=thvm_tensor(ctx,lwd,SHAPE(flat_f,NC));
    thvm_set_requires_grad(ctx,lw);
    Term lb=thvm_tensor(ctx,lbd,SHAPE(NC));
    u32 p0[]={0,0,0,0},s1[]={1,1},k2[]={2,2},s2[]={2,2};
    Term h=thvm_conv2d(ctx,x,w1,b1,1,s1,p0);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    h=thvm_maxpool2d(ctx,h,k2,s2);
    h=thvm_reshape(ctx,h,SHAPE(BS,flat_f));
    Term logits=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,lw),
        thvm_expand(ctx,thvm_reshape(ctx,lb,SHAPE(1,NC)),SHAPE(BS,NC)));
    Term loss=cross_entropy_loss(ctx,logits,labels,BS,NC);

    Term params[2]={w1,lw};
    Term gs[2]; f32 z1[18]={0}; f32*z2=calloc(flat_f*NC,4);
    gs[0]=thvm_tensor(ctx,z1,(Shape){.dims={2,1,3,3},.rank=4});
    gs[1]=thvm_tensor(ctx,z2,SHAPE(flat_f,NC)); free(z2);
    thvm_reduce(ctx,thvm_app(ctx,thvm_grad_multi(ctx,loss,params,gs,2),term_era()));
    printf("loss=%.6f\n",thvm_to_host(ctx,loss)[0]);

    // Copy gradients BEFORE numerical loop
    f32 gw1[18]; {f32*g=thvm_to_host(ctx,gs[0]); if(g)memcpy(gw1,g,72);else memset(gw1,0,72);}
    f32 *glw=malloc(flat_f*NC*4);
    {f32*g=thvm_to_host(ctx,gs[1]); if(g)memcpy(glw,g,flat_f*NC*4);else memset(glw,0,flat_f*NC*4);}

    // Numerical gradient
    f32 eps=1e-3f;
    f32 max_d1=0,max_g1=0;
    for(int j=0;j<5;j++){
        int idx=j*3; f32 saved=w1d[idx];
        w1d[idx]=saved+eps; thvm_reset(ctx,0);
        f32 lp=run_fwd(ctx,xd,w1d,b1d,lwd,lbd,labels,BS,IH,IW,flat_f,NC);
        w1d[idx]=saved-eps; thvm_reset(ctx,0);
        f32 lm=run_fwd(ctx,xd,w1d,b1d,lwd,lbd,labels,BS,IH,IW,flat_f,NC);
        w1d[idx]=saved;
        f32 ng=(lp-lm)/(2*eps),d=fabsf(gw1[idx]-ng);
        if(d>max_d1)max_d1=d; if(fabsf(ng)>max_g1)max_g1=fabsf(ng);
        printf("  w1[%d]: a=%.6f n=%.6f d=%.2e\n",idx,gw1[idx],ng,d);
    }
    // Standard gradient check: pass if abs < atol OR rel < rtol
    f32 atol=1e-3f, rtol=0.05f;
    f32 rel1=max_g1>atol?max_d1/max_g1:0;
    int ok1=(max_d1<atol)||(rel1<rtol);
    printf("w1: abs=%.2e rel=%.2e %s\n",max_d1,rel1,ok1?"OK":"FAIL");

    f32 max_d2=0,max_g2=0;
    for(int j=0;j<3;j++){
        int idx=j*(int)(flat_f*NC/3); f32 saved=lwd[idx];
        lwd[idx]=saved+eps; thvm_reset(ctx,0);
        f32 lp=run_fwd(ctx,xd,w1d,b1d,lwd,lbd,labels,BS,IH,IW,flat_f,NC);
        lwd[idx]=saved-eps; thvm_reset(ctx,0);
        f32 lm=run_fwd(ctx,xd,w1d,b1d,lwd,lbd,labels,BS,IH,IW,flat_f,NC);
        lwd[idx]=saved;
        f32 ng=(lp-lm)/(2*eps),d=fabsf(glw[idx]-ng);
        if(d>max_d2)max_d2=d; if(fabsf(ng)>max_g2)max_g2=fabsf(ng);
    }
    f32 rel2=max_g2>atol?max_d2/max_g2:0;
    int ok2=(max_d2<atol)||(rel2<rtol);
    printf("lw: abs=%.2e rel=%.2e %s\n",max_d2,rel2,ok2?"OK":"FAIL");

    int ok=ok1&&ok2;
    printf("%s\n",ok?"ALL PASS":"FAIL");
    free(xd);free(lwd);free(glw);free(labels);thvm_free(ctx);
    return ok?0:1;
}
