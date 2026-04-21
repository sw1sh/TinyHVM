// Single conv gradient test: does dX (input gradient) work?
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    srand(42);
    TinyHVM *ctx = thvm_init("metal");
    u32 BS=4;
    f32 inp_d[4*2*8*8];
    for(u32 i=0;i<sizeof(inp_d)/4;i++) inp_d[i]=(f32)rand()/RAND_MAX*2-1;
    Term inp = thvm_tensor(ctx, inp_d, (Shape){.dims={BS,2,8,8},.rank=4});
    thvm_set_requires_grad(ctx, inp);

    f32 w_d[3*2*3*3];
    for(u32 i=0;i<sizeof(w_d)/4;i++) w_d[i]=(f32)rand()/RAND_MAX*2-1;
    Term w = thvm_tensor(ctx, w_d, (Shape){.dims={3,2,3,3},.rank=4});
    thvm_set_requires_grad(ctx, w);

    u32 p0[]={0,0,0,0}, s1[]={1,1};
    Term h = thvm_conv2d(ctx, inp, w, term_era(), 1, s1, p0);
    Term loss = thvm_sum_axes(ctx, h, (u32[]){0,1,2,3}, 4);

    // Grad for both inp and w
    Term params[] = {inp, w};
    u32 psz[] = {4*2*8*8, 3*2*3*3};
    Term gs[2];
    for(int i=0;i<2;i++){
        f32*z=calloc(psz[i],4);
        gs[i]=thvm_tensor(ctx,z,ctx->tensors[(u32)term_val(params[i])].view.shape);
        free(z);
    }
    Term gt = thvm_grad_multi(ctx, loss, params, gs, 2);
    thvm_reduce(ctx, gt);

    for(int i=0;i<2;i++){
        f32*g=thvm_to_host(ctx,gs[i]);
        f32 s2=0;for(u32 j=0;j<psz[i];j++)s2+=g[j]*g[j];
        printf("  grad[%s] L2=%.6f %s\n", i?"w":"inp", sqrtf(s2), sqrtf(s2)>1e-10?"OK":"ZERO!");
    }

    thvm_free(ctx);
    return 0;
}
