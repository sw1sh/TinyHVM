#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>
int main(void) {
    srand(42);
    TinyHVM *ctx = thvm_init("metal");
    f32 xd[72]; for(int i=0;i<72;i++) xd[i]=0.1f*(f32)(i+1);
    f32 wd[18]; for(int i=0;i<18;i++) wd[i]=0.1f*((f32)rand()/(f32)RAND_MAX*2-1);
    f32 bd[2]={0};
    u32 p0[]={0,0,0,0}, s1[]={1,1};
    // With fuser
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={2,1,6,6},.rank=4});
    Term w=thvm_tensor(ctx,wd,(Shape){.dims={2,1,3,3},.rank=4});
    Term b=thvm_tensor(ctx,bd,SHAPE(2));
    f32*r1=thvm_to_host(ctx,thvm_conv2d(ctx,x,w,b,1,s1,p0));
    printf("fused: [%.6f,%.6f,%.6f,%.6f]\n",r1[0],r1[1],r1[2],r1[3]);
    // Without fuser
    thvm_reset(ctx,0);
    ctx->no_fuse=1;
    x=thvm_tensor(ctx,xd,(Shape){.dims={2,1,6,6},.rank=4});
    w=thvm_tensor(ctx,wd,(Shape){.dims={2,1,3,3},.rank=4});
    b=thvm_tensor(ctx,bd,SHAPE(2));
    f32*r2=thvm_to_host(ctx,thvm_conv2d(ctx,x,w,b,1,s1,p0));
    printf("nofuse: [%.6f,%.6f,%.6f,%.6f]\n",r2[0],r2[1],r2[2],r2[3]);
    f32 d=0; for(int i=0;i<32;i++){f32 dd=fabsf(r1[i]-r2[i]);if(dd>d)d=dd;}
    printf("max_diff=%.2e %s\n",d,d<1e-5?"MATCH":"DIFFER");
    return 0;
}
