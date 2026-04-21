// test_2conv_cpu_vs_metal.m — Compare conv backward on CPU vs Metal
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

static void run_test(const char *dev) {
    srand(42); // same seed for reproducibility
    TinyHVM *ctx = thvm_init(dev);
    u32 BS=4, IH=14, IW=14;

    Term cw1 = mkw(ctx, (Shape){.dims={4,1,3,3},.rank=4}, 9);
    Term cb1 = mkz(ctx, 4);
    Term cw2 = mkw(ctx, (Shape){.dims={4,4,3,3},.rank=4}, 36);
    Term cb2 = mkz(ctx, 4);

    u32 n = BS*1*IH*IW;
    f32 *xd = malloc(n*4);
    for (u32 i=0;i<n;i++) xd[i] = (f32)rand()/(f32)RAND_MAX;
    Term x = thvm_tensor(ctx, xd, (Shape){.dims={BS,1,IH,IW},.rank=4});
    free(xd);
    thvm_set_requires_grad(ctx, x);

    #define NP 4
    Term params[NP] = {cw1, cb1, cw2, cb2};
    u32 psz[NP] = {4*9, 4, 4*4*9, 4};
    for (u32 i=0;i<NP;i++) thvm_set_requires_grad(ctx, params[i]);

    u32 s1[] = {1,1}, p0[] = {0,0,0,0};
    Term h = thvm_conv2d(ctx, x, cw1, cb1, 1, s1, p0);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    h = thvm_conv2d(ctx, h, cw2, cb2, 1, s1, p0);
    h = thvm_op(ctx, UOP_RELU, h, term_era());

    extern Term thvm_sum_axes(TinyHVM*, Term, const u32*, u32);
    h = thvm_reshape(ctx, h, SHAPE(BS, 4*10*10));
    Term loss = thvm_sum_axes(ctx, h, (u32[]){0,1}, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1));

    Term gs[NP];
    for (int i=0;i<NP;i++) {
        f32 *z = calloc(psz[i], 4);
        gs[i] = thvm_tensor(ctx, z, ctx->tensors[(u32)term_val(params[i])].view.shape);
        free(z);
    }
    Term grad = thvm_grad_multi(ctx, loss, params, gs, NP);
    thvm_reduce(ctx, grad);

    const char *names[NP] = {"cw1", "cb1", "cw2", "cb2"};
    printf("%-6s: ", dev);
    for (int i=0;i<NP;i++) {
        f32 *gd = thvm_to_host(ctx, gs[i]);
        double norm = 0;
        for (u32 j=0;j<psz[i];j++) norm += gd[j]*gd[j];
        printf("%s=%.2f ", names[i], sqrt(norm));
    }
    printf("\n");


    thvm_free(ctx);
    #undef NP
}

int main(void) {
    run_test("cpu");
    run_test("metal");
    return 0;
}
