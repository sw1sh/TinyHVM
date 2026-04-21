#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>
int main(void){
    TinyHVM*ctx=thvm_init("cpu");
    // 1x1 conv: no pooling needed. BS=2,Cin=1,H=3,W=3,Cout=2,K=1
    f32 x[]={1,2,3,4,5,6,7,8,9, -1,-2,-3,-4,-5,-6,-7,-8,-9}; // [2,1,3,3]
    f32 w[]={1, -1}; // [2,1,1,1]
    f32 b[]={0, 0};
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={2,1,3,3},.rank=4});thvm_set_requires_grad(ctx,tx);
    Term tw=thvm_tensor(ctx,w,(Shape){.dims={2,1,1,1},.rank=4});thvm_set_requires_grad(ctx,tw);
    Term tb=thvm_tensor(ctx,b,SHAPE(2));thvm_set_requires_grad(ctx,tb);
    Term h=thvm_conv2d(ctx,tx,tw,tb,1,(u32[]){1,1},(u32[]){0,0,0,0});
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term loss=thvm_sum_axes(ctx,h,(u32[]){0,1,2,3},4);
    loss=thvm_reshape(ctx,loss,SHAPE(1));
    f32 gw0[2]={0}; Term gw=thvm_tensor(ctx,gw0,(Shape){.dims={2,1,1,1},.rank=4});
    Term grad=thvm_grad_multi(ctx,loss,(Term[]){tw},(Term[]){gw},1);
    thvm_eval(ctx,grad);
    f32*dw=thvm_to_host(ctx,gw);
    printf("dw: [%.4f, %.4f]\n", dw[0], dw[1]);
    // Expected: conv_out[b,co,h,w] = w[co]*x[b,h,w]
    // w[0]=1: out[0,:,:,:] = x[0,:,:,:] = sum=45, out[1,:,:,:] = x[1,:,:,:] = sum=-45
    // w[1]=-1: out[0,:,:,:] = -x[0,:,:,:] = sum=-45, out[1,:,:,:] = -x[1,:,:,:] = sum=45
    // loss = sum of all conv_out = 0
    // d(loss)/d(w[0]) = sum of x (all samples, all positions) = 45 + (-45) = 0
    // d(loss)/d(w[1]) = sum of -x = -(45 + (-45)) = 0
    // Hmm, both zero. Let me use a different example.
    printf("(zeros expected: symmetric input)\n");
    thvm_free(ctx);

    // Non-symmetric test
    ctx=thvm_init("cpu");
    f32 x2[]={1,2,3,4,5,6,7,8,9, 1,1,1,1,1,1,1,1,1}; // [2,1,3,3]
    f32 w2[]={1, 2}; // [2,1,1,1]
    f32 b2[]={0, 0};
    tx=thvm_tensor(ctx,x2,(Shape){.dims={2,1,3,3},.rank=4});thvm_set_requires_grad(ctx,tx);
    tw=thvm_tensor(ctx,w2,(Shape){.dims={2,1,1,1},.rank=4});thvm_set_requires_grad(ctx,tw);
    tb=thvm_tensor(ctx,b2,SHAPE(2));thvm_set_requires_grad(ctx,tb);
    h=thvm_conv2d(ctx,tx,tw,tb,1,(u32[]){1,1},(u32[]){0,0,0,0});
    loss=thvm_sum_axes(ctx,h,(u32[]){0,1,2,3},4);
    loss=thvm_reshape(ctx,loss,SHAPE(1));
    f32 gw2d[2]={0}; gw=thvm_tensor(ctx,gw2d,(Shape){.dims={2,1,1,1},.rank=4});
    grad=thvm_grad_multi(ctx,loss,(Term[]){tw},(Term[]){gw},1);
    thvm_eval(ctx,grad);
    dw=thvm_to_host(ctx,gw);
    printf("dw: [%.4f, %.4f]\n", dw[0], dw[1]);
    // conv_out[b,co,h,w] = w[co]*x[b,0,h,w]
    // loss = sum(w[0]*x) + sum(w[1]*x) = w[0]*54 + w[1]*54 = 1*54 + 2*54 = 162
    // d(loss)/d(w[0]) = sum(x[all]) = 45+9 = 54
    // d(loss)/d(w[1]) = sum(x[all]) = 54
    printf("(expect [54.0000, 54.0000])\n");
    thvm_free(ctx);
    return 0;
}
