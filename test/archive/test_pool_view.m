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
    Term pooled=thvm_pool(ctx,tx,(u32[]){2,2},(u32[]){1,1},2);
    // pooled should be TAG_TEN (view alias)
    printf("pooled tag=%u\n", term_tag(pooled));
    if (term_tag(pooled)==10) { // TAG_TEN
        u32 tid=(u32)term_val(pooled);
        TensorMeta*m=&ctx->tensors[tid];
        printf("view: rank=%u numel=%u offset=%d contiguous=%d\n",
            m->view.shape.rank, m->view.numel, m->view.offset, m->view.contiguous);
        printf("shape: ");for(u32 d=0;d<m->view.shape.rank;d++)printf("%u ",m->view.shape.dims[d]);printf("\n");
        printf("strides: ");for(u32 d=0;d<m->view.shape.rank;d++)printf("%d ",m->view.strides[d]);printf("\n");
        printf("buf_id=%u src=%u\n", m->buf_id, m->src_ids[0]);
    }
    thvm_free(ctx);
    return 0;
}
