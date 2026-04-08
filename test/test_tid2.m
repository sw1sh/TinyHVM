#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
int main(void){
    srand(42);TinyHVM*ctx=thvm_init("cpu");
    f32 x[]={1,2,3,4,5,6,7,8,9};
    f32 w[]={1,1,1,1}; f32 b[]={0};
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={1,1,3,3},.rank=4});thvm_set_requires_grad(ctx,tx);
    Term tw=thvm_tensor(ctx,w,(Shape){.dims={1,1,2,2},.rank=4});thvm_set_requires_grad(ctx,tw);
    Term tb=thvm_tensor(ctx,b,SHAPE(1));thvm_set_requires_grad(ctx,tb);
    Term h=thvm_conv2d(ctx,tx,tw,tb,1,(u32[]){1,1},(u32[]){0,0,0,0});
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term loss=thvm_sum_axes(ctx,h,(u32[]){0,1,2,3},4);loss=thvm_reshape(ctx,loss,SHAPE(1));
    f32 gw0[4]={0}; Term gw=thvm_tensor(ctx,gw0,(Shape){.dims={1,1,2,2},.rank=4});
    Term grad=thvm_grad_multi(ctx,loss,(Term[]){tw},(Term[]){gw},1);
    // Before eval: dump tensor table
    printf("Pre-eval tensors:\n");
    for(u32 i=1;i<ctx->tensor_count && i<=15;i++){
        TensorMeta*m=&ctx->tensors[i];
        printf("  tid=%u buf=%u cop=%u sh=[",i,m->buf_id,m->creator_op);
        for(u32 d=0;d<m->view.shape.rank;d++) printf("%u,",m->view.shape.dims[d]);
        printf("] s=[");
        for(u32 d=0;d<m->view.shape.rank;d++) printf("%d,",m->view.strides[d]);
        printf("]\n");
    }
    thvm_eval(ctx,grad);
    f32*dw=thvm_to_host(ctx,gw);
    printf("dw: [%.2f, %.2f, %.2f, %.2f] (expect [12,16,24,28])\n", dw[0],dw[1],dw[2],dw[3]);
    thvm_free(ctx);return 0;
}
