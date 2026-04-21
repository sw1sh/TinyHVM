// debug_grad.m — conv+BN+linear with CE loss, numerical check
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
#include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static Term relu_fn(TinyHVM *c, Term x) { return thvm_op(c, UOP_RELU, x, term_era()); }

static f32 compute_loss(f32 *c1w_d, f32 *xd, u8 *labels, u32 BS) {
    TinyHVM *c = thvm_init("metal"); c->no_fuse = 1;
    Term W=thvm_tensor(c,c1w_d,(Shape){.dims={4,1,3,3},.rank=4});
    Term B=thvm_tensor(c,(f32[]){0,0,0,0},SHAPE(4));
    Term G=thvm_tensor(c,(f32[]){1,1,1,1},SHAPE(4));
    Term Be=thvm_tensor(c,(f32[]){0,0,0,0},SHAPE(4));
    Term Rm=thvm_tensor(c,(f32[]){0,0,0,0},SHAPE(4));
    Term Rv=thvm_tensor(c,(f32[]){1,1,1,1},SHAPE(4));
    f32 lwd[4*26*26*10]; srand(999); for(u32 i=0;i<4*26*26*10;i++) lwd[i]=0.01f*((f32)rand()/(f32)RAND_MAX-0.5f);
    Term Lw=thvm_tensor(c,lwd,SHAPE(4*26*26,10));
    Term Lb=thvm_tensor(c,(f32[10]){0},SHAPE(10));
    Layer m[]={{.type=LAYER_CONV2D,.conv={W,B,1,4,3}},{.type=LAYER_FN,.fn=relu_fn},
        {.type=LAYER_BN,.bn={G,Be,Rm,Rv,4,26,26}},{.type=LAYER_FLATTEN,.flat={4*26*26}},
        {.type=LAYER_LINEAR,.lin={Lw,Lb,4*26*26,10}}};
    Term X=thvm_tensor(c,xd,(Shape){.dims={BS,1,28,28},.rank=4});
    Term lo=thvm_reduce(c,cross_entropy_loss(c,thvm_sequential(c,X,m,5,BS,1),labels,BS,10));
    f32 lv = thvm_to_host(c,lo)[0];
    thvm_free(c);
    return lv;
}

int main(void) {
    u32 BS=2;
    f32 *xd=malloc(BS*784*sizeof(f32)); srand(42);
    for(u32 i=0;i<BS*784;i++) xd[i]=(f32)rand()/(f32)RAND_MAX;
    u8 labels[]={3,7};
    f32 c1w[36]; srand(123);
    for(u32 i=0;i<36;i++) c1w[i]=0.1f*((f32)rand()/(f32)RAND_MAX-0.5f);

    // SP gradient
    TinyHVM *c = thvm_init("metal"); c->no_fuse = 1;
    Term W=thvm_tensor(c,c1w,(Shape){.dims={4,1,3,3},.rank=4});
    Term B2=thvm_tensor(c,(f32[]){0,0,0,0},SHAPE(4));
    Term G=thvm_tensor(c,(f32[]){1,1,1,1},SHAPE(4));
    Term Be=thvm_tensor(c,(f32[]){0,0,0,0},SHAPE(4));
    Term Rm=thvm_tensor(c,(f32[]){0,0,0,0},SHAPE(4));
    Term Rv=thvm_tensor(c,(f32[]){1,1,1,1},SHAPE(4));
    thvm_set_requires_grad(c,W);
    f32 lwd[4*26*26*10]; srand(999); for(u32 i=0;i<4*26*26*10;i++) lwd[i]=0.01f*((f32)rand()/(f32)RAND_MAX-0.5f);
    Term Lw=thvm_tensor(c,lwd,SHAPE(4*26*26,10));
    Term Lb=thvm_tensor(c,(f32[10]){0},SHAPE(10));
    Layer m[]={{.type=LAYER_CONV2D,.conv={W,B2,1,4,3}},{.type=LAYER_FN,.fn=relu_fn},
        {.type=LAYER_BN,.bn={G,Be,Rm,Rv,4,26,26}},{.type=LAYER_FLATTEN,.flat={4*26*26}},
        {.type=LAYER_LINEAR,.lin={Lw,Lb,4*26*26,10}}};
    Term X=thvm_tensor(c,xd,(Shape){.dims={BS,1,28,28},.rank=4});
    thvm_set_requires_grad(c,X);
    Term lo=thvm_reduce(c,cross_entropy_loss(c,thvm_sequential(c,X,m,5,BS,1),labels,BS,10));
    Term p[]={W}; Term g[1];
    thvm_backward(c,lo,p,g,1);
    f32 gd[36]={0};
    if(term_tag(g[0])==TAG_TEN) memcpy(gd,thvm_to_host(c,g[0]),36*sizeof(f32));
    thvm_free(c);

    // Numerical + comparison
    printf("idx     numerical       SP            ratio\n");
    f32 ep = 1e-4f;
    for (u32 idx=0; idx<36; idx+=7) {
        f32 wp[36], wm[36];
        memcpy(wp,c1w,36*sizeof(f32)); wp[idx]+=ep;
        memcpy(wm,c1w,36*sizeof(f32)); wm[idx]-=ep;
        f32 lp = compute_loss(wp, xd, labels, BS);
        f32 lm = compute_loss(wm, xd, labels, BS);
        f32 num = (lp-lm)/(2*ep);
        f32 ratio = fabsf(num)>1e-8f ? gd[idx]/num : 0;
        printf("%-4u %12.6f %12.6f %12.4f\n", idx, num, gd[idx], ratio);
    }
    free(xd);
    return 0;
}
