#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
int main(void){
    TinyHVM*ctx=thvm_init("cpu");
    f32 x[]={1,1,1,1,1,1,1,1,1}; // all ones [1,1,3,3]
    f32 w[]={1,0,0,0}; // [1,1,2,2]
    f32 b[]={0};
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={1,1,3,3},.rank=4});thvm_set_requires_grad(ctx,tx);
    Term tw=thvm_tensor(ctx,w,(Shape){.dims={1,1,2,2},.rank=4});thvm_set_requires_grad(ctx,tw);
    Term tb=thvm_tensor(ctx,b,SHAPE(1));thvm_set_requires_grad(ctx,tb);
    Term h=thvm_conv2d(ctx,tx,tw,tb,1,(u32[]){1,1},(u32[]){0,0,0,0});
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term loss=thvm_sum_axes(ctx,h,(u32[]){0,1,2,3},4);loss=thvm_reshape(ctx,loss,SHAPE(1));
    f32 gw0[4]={0}; Term gw=thvm_tensor(ctx,gw0,(Shape){.dims={1,1,2,2},.rank=4});
    Term grad=thvm_grad_multi(ctx,loss,(Term[]){tw},(Term[]){gw},1);
    thvm_eval(ctx,grad);
    f32*dw=thvm_to_host(ctx,gw);
    printf("dw: [%.2f, %.2f, %.2f, %.2f]\n", dw[0],dw[1],dw[2],dw[3]);
    // x=ones, loss=sum(conv(ones,w))
    // conv_out = [[w00+w01+w10+w11,...],[...]] = [[1,1],[1,1]] (since w=[1,0,0,0] → out=x[oy,ox]*1)
    // Actually w=[1,0,0,0]: conv_out[oy,ox] = x[oy,ox]*1+x[oy,ox+1]*0+x[oy+1,ox]*0+x[oy+1,ox+1]*0 = x[oy,ox]
    // loss = sum(conv_out) = 4 (2x2 output, all 1s)
    // d(loss)/d(w[kh,kw]) = sum_oy,ox x[oy+kh,ox+kw] = num_output_positions = 4
    // All dw should be 4.0
    printf("(expect all 4.0)\n");
    thvm_free(ctx);
    return 0;
}
