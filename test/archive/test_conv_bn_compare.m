// Compare intermediate values at each backward step
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static void test(const char *dev) {
    srand(42);
    TinyHVM *ctx = thvm_init(dev);
    u32 BS=4,C=2,H=6,W=6;
    f32 xd[4*2*6*6];u32 n=sizeof(xd)/4;
    for(u32 i=0;i<n;i++)xd[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,C,H,W},.rank=4});
    thvm_set_requires_grad(ctx,x);
    f32 gd[]={1,1};Term gamma=thvm_tensor(ctx,gd,SHAPE(C));
    f32 bd[]={0,0};Term beta=thvm_tensor(ctx,bd,SHAPE(C));
    f32 rmd[]={0,0};Term rm=thvm_tensor(ctx,rmd,SHAPE(C));
    f32 rvd[]={1,1};Term rv=thvm_tensor(ctx,rvd,SHAPE(C));
    thvm_set_requires_grad(ctx,gamma);thvm_set_requires_grad(ctx,beta);
    BNResult bn=batchnorm_forward(ctx,x,gamma,beta,rm,rv,BS,C,H,W,1);
    // Just sum the BN output — but also track x gradient
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term loss=thvm_sum_axes(ctx,bn.output,(u32[]){0,1,2,3},4);
    loss=thvm_reshape(ctx,loss,SHAPE(1));
    // Compute grad of x through BN
    f32 gxd[4*2*6*6]={0};
    Term gx=thvm_tensor(ctx,gxd,(Shape){.dims={BS,C,H,W},.rank=4});
    Term grad=thvm_grad_multi(ctx,loss,(Term[]){x},(Term[]){gx},1);
    thvm_reduce(ctx,thvm_app(ctx,grad,bn.assigns));
    f32*dx=thvm_to_host(ctx,gx);
    double dnorm=0;for(u32 i=0;i<n;i++)dnorm+=dx[i]*dx[i];
    printf("%-6s: dx_norm=%.6f dx[0..7]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
        dev,sqrt(dnorm),dx[0],dx[1],dx[2],dx[3],dx[4],dx[5],dx[6],dx[7]);
    thvm_free(ctx);
}
int main(void){test("cpu");test("metal");return 0;}
