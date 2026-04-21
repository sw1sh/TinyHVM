#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
int main(void){
    TinyHVM*ctx=thvm_init("cpu");
    f32 x[4*1*14*14]; for(int i=0;i<4*14*14;i++)x[i]=1.f;
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={4,1,14,14},.rank=4});
    Term pooled=thvm_pool(ctx,tx,(u32[]){3,3},(u32[]){1,1},2);
    printf("pooled tag=%u\n",term_tag(pooled));
    if(term_tag(pooled)==TAG_TEN) {
        u32 tid=(u32)term_val(pooled);
        TensorMeta*m=&ctx->tensors[tid];
        printf("shape=[");for(u32 d=0;d<m->view.shape.rank;d++)printf("%u,",m->view.shape.dims[d]);
        printf("] s=[");for(u32 d=0;d<m->view.shape.rank;d++)printf("%d,",m->view.strides[d]);
        printf("] off=%d numel=%u\n", m->view.offset, m->view.numel);
    }
    thvm_free(ctx);return 0;
}
