// test_2conv_grad.m — Minimal test: do gradients flow through 2 convs?
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

    // Random input [4,1,8,8]
    u32 n_inp = BS*1*8*8;
    f32 *inp_d = malloc(n_inp*4);
    for (u32 i=0;i<n_inp;i++) inp_d[i] = (f32)rand()/RAND_MAX*2-1;
    Term inp = thvm_tensor(ctx, inp_d, (Shape){.dims={BS,1,8,8},.rank=4});
    thvm_set_requires_grad(ctx, inp);

    // Conv1: 1->2, k=3 -> 8->6
    f32 w1_d[2*1*3*3], b1_d[2]; memset(b1_d,0,sizeof(b1_d));
    for (u32 i=0;i<18;i++) w1_d[i] = (f32)rand()/RAND_MAX*2-1;
    Term w1 = thvm_tensor(ctx, w1_d, (Shape){.dims={2,1,3,3},.rank=4});
    Term b1 = thvm_tensor(ctx, b1_d, SHAPE(2));
    thvm_set_requires_grad(ctx, w1);
    thvm_set_requires_grad(ctx, b1);

    // Conv2: 2->2, k=3 -> 6->4
    f32 w2_d[2*2*3*3], b2_d[2]; memset(b2_d,0,sizeof(b2_d));
    for (u32 i=0;i<36;i++) w2_d[i] = (f32)rand()/RAND_MAX*2-1;
    Term w2 = thvm_tensor(ctx, w2_d, (Shape){.dims={2,2,3,3},.rank=4});
    Term b2 = thvm_tensor(ctx, b2_d, SHAPE(2));
    thvm_set_requires_grad(ctx, w2);
    thvm_set_requires_grad(ctx, b2);

    // Forward: conv1 -> relu -> conv2 -> sum (as loss)
    u32 p0[]={0,0,0,0}, s1[]={1,1};
    Term h = thvm_conv2d(ctx, inp, w1, b1, 1, s1, p0);
    h = thvm_op(ctx, UOP_RELU, h, term_era());
    h = thvm_conv2d(ctx, h, w2, b2, 1, s1, p0);
    Term loss = thvm_sum_axes(ctx, h, (u32[]){0,1,2,3}, 4);

    // Backward
    Term params[] = {w1, b1, w2, b2};
    u32 psz[] = {18, 2, 36, 2};
    u32 NP = 4;
    Term gs[4];
    for (u32 i=0;i<NP;i++) {
        f32 *z = calloc(psz[i], 4);
        gs[i] = thvm_tensor(ctx, z, ctx->tensors[(u32)term_val(params[i])].view.shape);
        free(z);
    }
    Term grad_term = thvm_grad_multi(ctx, loss, params, gs, NP);
    thvm_reduce(ctx, grad_term);

    // Check gradients
    const char *names[] = {"w1","b1","w2","b2"};
    for (u32 i=0;i<NP;i++) {
        f32 *g = thvm_to_host(ctx, gs[i]);
        f32 s2 = 0;
        for (u32 j=0;j<psz[i];j++) s2 += g[j]*g[j];
        printf("  grad[%s] L2=%.6f (n=%u) %s\n", names[i], sqrtf(s2), psz[i],
            sqrtf(s2) > 1e-10 ? "OK" : "ZERO!");
    }

    free(inp_d);
    thvm_free(ctx);
    return 0;
}
