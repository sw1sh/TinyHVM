// test_single_conv_dx.m — Single conv dx test with proper sizes
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

    u32 BS=4, Cin=1, H=14, W=14, Cout=4, KH=3, KW=3;
    u32 xn = BS*Cin*H*W;
    u32 wn = Cout*Cin*KH*KW;

    f32 *xd = malloc(xn*4);
    for(u32 i=0;i<xn;i++) xd[i]=(f32)rand()/(f32)RAND_MAX;
    Term x = thvm_tensor(ctx, xd, (Shape){.dims={BS,Cin,H,W},.rank=4});
    free(xd);
    thvm_set_requires_grad(ctx, x);

    f32 *wd = malloc(wn*4);
    for(u32 i=0;i<wn;i++) wd[i]=(f32)rand()/(f32)RAND_MAX * 2 - 1;
    Term w = thvm_tensor(ctx, wd, (Shape){.dims={Cout,Cin,KH,KW},.rank=4});
    free(wd);
    thvm_set_requires_grad(ctx, w);

    // Two convs: conv1 → ReLU → conv2 → sum
    u32 Cout2=4, KH2=3, KW2=3;
    u32 wn2 = Cout2*Cout*KH2*KW2;
    f32 *wd2 = malloc(wn2*4);
    for(u32 i=0;i<wn2;i++) wd2[i]=(f32)rand()/(f32)RAND_MAX * 2 - 1;
    Term w2 = thvm_tensor(ctx, wd2, (Shape){.dims={Cout2,Cout,KH2,KW2},.rank=4});
    free(wd2);
    thvm_set_requires_grad(ctx, w2);

    // Add biases
    f32 *bd1 = calloc(Cout, 4); Term b1 = thvm_tensor(ctx, bd1, SHAPE(Cout)); free(bd1);
    f32 *bd2 = calloc(Cout2, 4); Term b2 = thvm_tensor(ctx, bd2, SHAPE(Cout2)); free(bd2);
    thvm_set_requires_grad(ctx, b1);
    thvm_set_requires_grad(ctx, b2);

    // BN params
    f32 *bn_gd = malloc(Cout*4); for(u32 i=0;i<Cout;i++) bn_gd[i]=1.f;
    Term bn_g = thvm_tensor(ctx, bn_gd, SHAPE(Cout)); free(bn_gd);
    f32 *bn_bd = calloc(Cout,4); Term bn_b = thvm_tensor(ctx, bn_bd, SHAPE(Cout)); free(bn_bd);
    f32 *bn_rm = calloc(Cout,4); Term bn_rmean = thvm_tensor(ctx, bn_rm, SHAPE(Cout)); free(bn_rm);
    f32 *bn_rv = malloc(Cout*4); for(u32 i=0;i<Cout;i++) bn_rv[i]=1.f;
    Term bn_rvar = thvm_tensor(ctx, bn_rv, SHAPE(Cout)); free(bn_rv);

    Term h = thvm_conv2d(ctx, x, w, b1, 1, (u32[]){1,1}, (u32[]){0,0,0,0});
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    // Add BN + Pool
    u32 conv1_OY = H-KH+1; // 12
    BNResult bn = batchnorm_forward(ctx, h, bn_g, bn_b, bn_rmean, bn_rvar, BS, Cout, conv1_OY, conv1_OY, 1);
    h = bn.output;
    h = thvm_maxpool2d(ctx, h, (u32[]){2,2}, (u32[]){2,2}); // Pool: 12→6
    h = thvm_conv2d(ctx, h, w2, b2, 1, (u32[]){1,1}, (u32[]){0,0,0,0});
    h = thvm_op(ctx, UOP_RELU, h, term_era());

    // Loss = reshape + sum
    extern Term thvm_sum_axes(TinyHVM*, Term, const u32*, u32);
    u32 OY = (H-KH+1)/2-KH2+1; // conv1: 12→pool: 6→conv2: 4
    h = thvm_reshape(ctx, h, SHAPE(BS, Cout2*OY*OY));
    Term loss = thvm_sum_axes(ctx, h, (u32[]){0,1}, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1));

    // Backward for both w and w2
    f32 *gwd = calloc(wn, 4);
    Term gw = thvm_tensor(ctx, gwd, (Shape){.dims={Cout,Cin,KH,KW},.rank=4});
    free(gwd);
    f32 *gwd2 = calloc(wn2, 4);
    Term gw2 = thvm_tensor(ctx, gwd2, (Shape){.dims={Cout2,Cout,KH2,KW2},.rank=4});
    free(gwd2);
    // All params
    f32 *gb1d = calloc(Cout, 4); Term gb1 = thvm_tensor(ctx, gb1d, SHAPE(Cout)); free(gb1d);
    f32 *gb2d = calloc(Cout2, 4); Term gb2 = thvm_tensor(ctx, gb2d, SHAPE(Cout2)); free(gb2d);
    Term params[] = {w, b1, w2, b2};
    Term grads_arr[] = {gw, gb1, gw2, gb2};
    Term grad2 = thvm_grad_multi(ctx, loss, params, grads_arr, 4);
    thvm_reduce(ctx, thvm_app(ctx, grad2, bn.assigns));

    const char *names[] = {"w1", "b1", "w2", "b2"};
    Term grads[] = {gw, gb1, gw2, gb2};
    (void)grads_arr;
    u32 psz[] = {wn, Cout, wn2, Cout2};
    printf("%-6s: ", dev);
    for (int i=0;i<4;i++) {
        f32 *gd = thvm_to_host(ctx, grads[i]);
        double norm=0; for(u32 j=0;j<psz[i];j++) norm+=gd[j]*gd[j];
        printf("%s=%.2f ", names[i], sqrt(norm));
    }
    printf("\n");

    thvm_free(ctx);
}

int main(void) {
    test("cpu");
    test("metal");
    return 0;
}
