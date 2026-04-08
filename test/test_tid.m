#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
int main(void){
    TinyHVM*ctx=thvm_init("cpu");
    f32 x[]={1,2,3,4,5,6,7,8,9};
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={1,1,3,3},.rank=4});
    printf("tx tid=%u\n",(u32)term_val(tx));
    Term pooled=thvm_pool(ctx,tx,(u32[]){2,2},(u32[]){1,1},2);
    printf("pooled tag=%u tid=%u\n",term_tag(pooled), term_tag(pooled)==TAG_TEN?(u32)term_val(pooled):0);
    Term x_rs=thvm_reshape(ctx,pooled,shape_of((u32[]){1,1,1,1,2,2,2,2},8));
    printf("x_rs tag=%u val=%llu\n",term_tag(x_rs),term_val(x_rs));
    Term x_exp=thvm_expand(ctx,x_rs,shape_of((u32[]){1,1,1,1,2,2,2,2},8));
    printf("x_exp tag=%u val=%llu\n",term_tag(x_exp),term_val(x_exp));
    Term x_perm=thvm_permute(ctx,x_exp,(u32[]){0,1,3,4,5,2,6,7},8);
    printf("x_perm tag=%u val=%llu\n",term_tag(x_perm),term_val(x_perm));
    TensorMeta*m4=&ctx->tensors[4];
    printf("tid=4 shape=[");for(u32 d=0;d<m4->view.shape.rank;d++)printf("%u,",m4->view.shape.dims[d]);
    printf("] s=[");for(u32 d=0;d<m4->view.shape.rank;d++)printf("%d,",m4->view.strides[d]);printf("]\n");
    thvm_free(ctx);return 0;
}
