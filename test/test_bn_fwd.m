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
    u32 BS=4,C=4,H=6,W=6,n=BS*C*H*W;
    f32 *xd=malloc(n*4);for(u32 i=0;i<n;i++)xd[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={BS,C,H,W},.rank=4});free(xd);
    f32 gd[4]={1,1,1,1};Term gamma=thvm_tensor(ctx,gd,SHAPE(C));
    f32 bd[4]={0};Term beta=thvm_tensor(ctx,bd,SHAPE(C));
    f32 rmd[4]={0};Term rmean=thvm_tensor(ctx,rmd,SHAPE(C));
    f32 rvd[4]={1,1,1,1};Term rvar=thvm_tensor(ctx,rvd,SHAPE(C));
    BNResult bn=batchnorm_forward(ctx,x,gamma,beta,rmean,rvar,BS,C,H,W,1);
    thvm_reduce(ctx,bn.output);
    f32 *out=thvm_to_host(ctx,bn.output);
    double norm=0;for(u32 i=0;i<n;i++)norm+=out[i]*out[i];
    printf("%-6s: norm=%.6f first=[%.6f,%.6f,%.6f,%.6f]\n",
        dev,sqrt(norm),out[0],out[1],out[2],out[3]);
    thvm_free(ctx);
}
int main(void){test("cpu");test("metal");return 0;}
