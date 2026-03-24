#include "../src/tinyhvm.c"
#include "../src/cpu.c"
#include "../src/metal.m"
#include <math.h>
int main() {
    srand(42);
    TinyHVM *ctx = thvm_init(thvm_device("metal"));
    int BS=4, IN=2, H=3, OUT=2;
    int nw1=IN*H; int nw2=H*OUT;
    f32 *w1d=malloc(nw1*4), *b1d=calloc(H,4), *w2d=malloc(nw2*4), *b2d=calloc(OUT,4);
    f32 *xd=malloc(BS*IN*4), *td=malloc(BS*OUT*4);
    for(int i=0;i<nw1;i++) w1d[i]=((f32)rand()/RAND_MAX-0.5f)*0.2f;
    for(int i=0;i<nw2;i++) w2d[i]=((f32)rand()/RAND_MAX-0.5f)*0.2f;
    for(int i=0;i<BS*IN;i++) xd[i]=((f32)rand()/RAND_MAX-0.5f)*0.2f;
    for(int i=0;i<BS*OUT;i++) td[i]=((f32)rand()/RAND_MAX)*0.5f;

    // Forward + analytical grad for w1
    thvm_reset(ctx,0);
    Term w1=thvm_tensor(ctx,w1d,shape_of((u32[]){IN,H},2));
    Term b1=thvm_tensor(ctx,b1d,shape_of((u32[]){1,H},2));
    Term w2=thvm_tensor(ctx,w2d,shape_of((u32[]){H,OUT},2));
    Term b2=thvm_tensor(ctx,b2d,shape_of((u32[]){1,OUT},2));
    Term x=thvm_tensor(ctx,xd,shape_of((u32[]){BS,IN},2));
    Term t=thvm_tensor(ctx,td,shape_of((u32[]){BS,OUT},2));
    thvm_set_requires_grad(ctx,w1);
    Term h=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,x,w1),b1);
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    Term o=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,w2),b2);
    Term d=thvm_op(ctx,UOP_SUB,o,t);
    Term sq=thvm_op(ctx,UOP_MUL,d,d);
    u32 ax[]={0,1}; Term l=thvm_sum_axes(ctx,sq,ax,2);
    f32 inv=1.f/(BS*OUT); l=thvm_op(ctx,UOP_MUL,l,thvm_tensor(ctx,&inv,SHAPE(1)));
    l=thvm_reshape(ctx,l,SHAPE(1)); l=thvm_reduce(ctx,l);

    Term gw1=thvm_reduce(ctx,thvm_grad(ctx,l,w1));
    f32 *gd=thvm_to_host(ctx,gw1); f32 analytic=gd[0]; free(gd);

    // Numerical: perturb w1[0]
    f32 eps=1e-4f; f32 lp,lm;
    f32 *wp=malloc(nw1*4);

    for(int i=0;i<nw1;i++) wp[i]=w1d[i]; wp[0]+=eps;
    thvm_reset(ctx,0);
    Term _w1=thvm_tensor(ctx,wp,shape_of((u32[]){IN,H},2));
    thvm_tensor(ctx,b1d,shape_of((u32[]){1,H},2));
    thvm_tensor(ctx,w2d,shape_of((u32[]){H,OUT},2));
    thvm_tensor(ctx,b2d,shape_of((u32[]){1,OUT},2));
    x=thvm_tensor(ctx,xd,shape_of((u32[]){BS,IN},2));
    t=thvm_tensor(ctx,td,shape_of((u32[]){BS,OUT},2));
    h=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,x,_w1),thvm_tensor(ctx,b1d,shape_of((u32[]){1,H},2)));
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    o=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,thvm_tensor(ctx,w2d,shape_of((u32[]){H,OUT},2))),thvm_tensor(ctx,b2d,shape_of((u32[]){1,OUT},2)));
    d=thvm_op(ctx,UOP_SUB,o,t); sq=thvm_op(ctx,UOP_MUL,d,d);
    Term _l=thvm_sum_axes(ctx,sq,ax,2);
    _l=thvm_op(ctx,UOP_MUL,_l,thvm_tensor(ctx,&inv,SHAPE(1)));
    _l=thvm_reshape(ctx,_l,SHAPE(1)); _l=thvm_reduce(ctx,_l);
    f32 *rv=thvm_to_host(ctx,_l); lp=rv[0]; free(rv);

    for(int i=0;i<nw1;i++) wp[i]=w1d[i]; wp[0]-=eps;
    thvm_reset(ctx,0);
    _w1=thvm_tensor(ctx,wp,shape_of((u32[]){IN,H},2));
    x=thvm_tensor(ctx,xd,shape_of((u32[]){BS,IN},2));
    t=thvm_tensor(ctx,td,shape_of((u32[]){BS,OUT},2));
    h=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,x,_w1),thvm_tensor(ctx,b1d,shape_of((u32[]){1,H},2)));
    h=thvm_op(ctx,UOP_RELU,h,term_era());
    o=thvm_op(ctx,UOP_ADD,thvm_op(ctx,UOP_MM,h,thvm_tensor(ctx,w2d,shape_of((u32[]){H,OUT},2))),thvm_tensor(ctx,b2d,shape_of((u32[]){1,OUT},2)));
    d=thvm_op(ctx,UOP_SUB,o,t); sq=thvm_op(ctx,UOP_MUL,d,d);
    _l=thvm_sum_axes(ctx,sq,ax,2);
    _l=thvm_op(ctx,UOP_MUL,_l,thvm_tensor(ctx,&inv,SHAPE(1)));
    _l=thvm_reshape(ctx,_l,SHAPE(1)); _l=thvm_reduce(ctx,_l);
    rv=thvm_to_host(ctx,_l); lm=rv[0]; free(rv);

    f32 numerical=(lp-lm)/(2*eps);
    printf("w1[0,0]: analytic=%.6f numerical=%.6f |diff|=%.6f %s\n",
           analytic, numerical, fabsf(analytic-numerical),
           fabsf(analytic-numerical)<1e-3f ? "PASS" : "FAIL");
    return fabsf(analytic-numerical)<1e-3f ? 0 : 1;
}
