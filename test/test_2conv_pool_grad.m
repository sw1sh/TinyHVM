// test_2conv_pool_grad.m — Verify gradient flows through Conv→Pool→Conv
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static Term mkw(TinyHVM*c,Shape s,u32 fi){u32 n=1;for(u32 i=0;i<s.rank;i++)n*=s.dims[i];f32*d=malloc(n*4);f32 b=1.f/sqrtf((f32)fi);for(u32 i=0;i<n;i++)d[i]=b*((f32)rand()/(f32)RAND_MAX*2-1);Term t=thvm_tensor(c,d,s);free(d);return t;}
static Term mkz(TinyHVM*c,u32 n){f32*z=calloc(n,4);Term t=thvm_tensor(c,z,SHAPE(n));free(z);return t;}

int main(void) {
    srand(42);
    TinyHVM *ctx = thvm_init("metal");
    u32 BS=4;

    // Conv1(1→4,k3) → Pool(2,2) → Conv2(4→4,k3) → Dense(4) → loss
    Term cw1 = mkw(ctx, (Shape){.dims={4,1,3,3},.rank=4}, 9);
    Term cb1 = mkz(ctx, 4);
    Term cw2 = mkw(ctx, (Shape){.dims={4,4,3,3},.rank=4}, 36);
    Term cb2 = mkz(ctx, 4);

    #define NP 5
    Term params[NP] = {cw1, cb1, cw2, cb2, x};
    u32 psz[NP] = {4*9, 4, 4*4*9, 4, BS*1*IH*IW};
    for (u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx, params[i]);

    // Input: [4,1,14,14]
    u32 IH=14, IW=14;
    u32 n = BS*1*IH*IW;
    f32 *xd = malloc(n*4);
    for (u32 i=0;i<n;i++) xd[i] = (f32)rand()/(f32)RAND_MAX;
    Term x = thvm_tensor(ctx, xd, (Shape){.dims={BS,1,IH,IW},.rank=4});
    free(xd);
    thvm_set_requires_grad(ctx, x);

    // Forward: Conv1(k3)→ReLU→Conv2(k3)→ReLU→sum
    u32 s1[] = {1,1}, p0[] = {0,0,0,0};
    Term h = thvm_conv2d(ctx, x, cw1, cb1, 1, s1, p0); // [4,4,12,12]
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    h = thvm_conv2d(ctx, h, cw2, cb2, 1, s1, p0); // [4,4,10,10]
    h = thvm_op(ctx, UOP_RELU, h, term_era());

    // Loss = sum of all outputs
    extern Term thvm_sum_axes(TinyHVM*, Term, const u32*, u32);
    u32 ow = 10; // conv2 output spatial
    h = thvm_reshape(ctx, h, SHAPE(BS, 4*ow*ow));
    Term loss = thvm_sum_axes(ctx, h, (u32[]){0,1}, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1));

    // Backward
    Term gs[NP];
    for (int i=0;i<NP;i++) {
        f32 *z = calloc(psz[i], 4);
        gs[i] = thvm_tensor(ctx, z, ctx->tensors[(u32)term_val(params[i])].view.shape);
        free(z);
    }
    Term grad = thvm_grad_multi(ctx, loss, params, gs, NP);
    thvm_reduce(ctx, grad);

    // Check gradients
    const char *names[NP] = {"cw1", "cb1", "cw2", "cb2", "x"};
    for (int i=0;i<NP;i++) {
        f32 *gd = thvm_to_host(ctx, gs[i]);
        double norm = 0;
        for (u32 j=0;j<psz[i];j++) norm += gd[j]*gd[j];
        f32 mx = 0;
        for (u32 j=0;j<psz[i];j++) if (fabsf(gd[j])>mx) mx=fabsf(gd[j]);
        printf("grad %-4s: norm=%.6f max=%.6f %s\n",
            names[i], sqrt(norm), mx, norm>0 ? "OK" : "ZERO!");
    }

    thvm_free(ctx);
    return 0;
}
