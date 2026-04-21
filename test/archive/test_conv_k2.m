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
    // K=2 conv: BS=1,Cin=1,H=3,W=3,Cout=1,K=2
    f32 x[]={1,2,3,4,5,6,7,8,9}; // [1,1,3,3]
    f32 w[]={1,1,1,1}; // [1,1,2,2] = all ones
    f32 b[]={0};
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={1,1,3,3},.rank=4});thvm_set_requires_grad(ctx,tx);
    Term tw=thvm_tensor(ctx,w,(Shape){.dims={1,1,2,2},.rank=4});thvm_set_requires_grad(ctx,tw);
    Term tb=thvm_tensor(ctx,b,SHAPE(1));thvm_set_requires_grad(ctx,tb);
    Term h=thvm_conv2d(ctx,tx,tw,tb,1,(u32[]){1,1},(u32[]){0,0,0,0});
    // No relu, no bias
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term loss=thvm_sum_axes(ctx,h,(u32[]){0,1,2,3},4);
    loss=thvm_reshape(ctx,loss,SHAPE(1));
    f32 gw0[4]={0}; Term gw=thvm_tensor(ctx,gw0,(Shape){.dims={1,1,2,2},.rank=4});
    Term grad=thvm_grad_multi(ctx,loss,(Term[]){tw},(Term[]){gw},1);
    thvm_eval(ctx,grad);
    f32*dw=thvm_to_host(ctx,gw);
    printf("dw: [%.2f, %.2f, %.2f, %.2f]\n", dw[0], dw[1], dw[2], dw[3]);
    // conv with w=all_ones: conv_out[0,0,oy,ox] = sum(x[oy:oy+2,ox:ox+2])
    // out = [[12,16],[24,28]], loss = 80
    // d(loss)/d(w[kh,kw]) = sum_over(oy,ox) x[oy+kh, ox+kw]
    // dw[0,0] = x[0,0]+x[0,1]+x[1,0]+x[1,1] = 1+2+4+5 = 12
    // dw[0,1] = x[0,1]+x[0,2]+x[1,1]+x[1,2] = 2+3+5+6 = 16
    // dw[1,0] = x[1,0]+x[1,1]+x[2,0]+x[2,1] = 4+5+7+8 = 24
    // dw[1,1] = x[1,1]+x[1,2]+x[2,1]+x[2,2] = 5+6+8+9 = 28
    printf("(expect [12.00, 16.00, 24.00, 28.00])\n");
    thvm_free(ctx);
    return 0;
}
