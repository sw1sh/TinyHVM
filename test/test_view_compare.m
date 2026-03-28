// Quick test: compare fuse_walk_inner views vs TensorMeta views for conv inner product
#include "src/tinyhvm.c"
#include "src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "src/backend/metal/_.m"
#endif
#include "src/nn/_.c"
#include <stdio.h>

int main(void) {
    srand(42);
    TinyHVM *ctx = thvm_init("metal");
    u32 BS=2;
    f32 xd[2*1*6*6]; for(int i=0;i<72;i++) xd[i]=0.1f*(f32)(i+1);
    f32 wd[2*1*3*3]; for(int i=0;i<18;i++) wd[i]=0.1f*((f32)rand()/(f32)RAND_MAX*2-1);
    f32 bd[2]={0};

    Term x = thvm_tensor(ctx, xd, (Shape){.dims={BS,1,6,6},.rank=4});
    Term w = thvm_tensor(ctx, wd, (Shape){.dims={2,1,3,3},.rank=4});
    Term b = thvm_tensor(ctx, bd, SHAPE(2));

    u32 p0[]={0,0,0,0}, s1[]={1,1};
    Term h = thvm_conv2d(ctx, x, w, b, 1, s1, p0);
    f32 lv = thvm_to_host(ctx, h)[0];
    printf("conv output[0] = %.6f\n", lv);

    // Now check: the conv forward created SUM(MUL(x_perm, w_rs)).
    // The fuser handled it. Let me check if the same result comes from
    // manually materializing the deferred MUL chain.

    // Create the same graph but DON'T let the fuser catch it:
    // Force no_fuse to see what the materialize path produces
    ctx->no_fuse = 1;
    thvm_reset(ctx, 0);
    x = thvm_tensor(ctx, xd, (Shape){.dims={BS,1,6,6},.rank=4});
    w = thvm_tensor(ctx, wd, (Shape){.dims={2,1,3,3},.rank=4});
    b = thvm_tensor(ctx, bd, SHAPE(2));
    h = thvm_conv2d(ctx, x, w, b, 1, s1, p0);
    ctx->no_fuse = 0;
    f32 lv2 = thvm_to_host(ctx, h)[0];
    printf("conv output[0] (no_fuse) = %.6f\n", lv2);
    printf("diff = %.2e\n", fabsf(lv - lv2));
    printf("%s\n", fabsf(lv-lv2) < 1e-5f ? "MATCH" : "DIFFER");
    return 0;
}
