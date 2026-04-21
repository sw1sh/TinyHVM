#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
int main(void){
    TinyHVM*ctx=thvm_init("cpu");
    f32 x[4*14*14]; for(int i=0;i<4*14*14;i++)x[i]=1.f;
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={4,1,14,14},.rank=4});
    Term pooled=thvm_pool(ctx,tx,(u32[]){3,3},(u32[]){1,1},2);
    // pool is TAG_TEN with strides [196,196,14,1,14,1]
    Term x_rs=thvm_reshape(ctx,pooled,shape_of((u32[]){4,1,1,1,12,12,3,3},8));
    // Check x_rs ShapeTracker
    if(term_tag(x_rs)==TAG_TOP) {
        const ShapeTracker *st = st_get_tracker(term_val(x_rs));
        if(st) {
            printf("n_views=%u\n", st->n_views);
            for(int v=0;v<st->n_views;v++) {
                printf("  view[%d] sh=[",v);
                for(u32 d=0;d<st->views[v].shape.rank;d++) printf("%u,",st->views[v].shape.dims[d]);
                printf("] s=[");
                for(u32 d=0;d<st->views[v].shape.rank;d++) printf("%d,",st->views[v].strides[d]);
                printf("]\n");
            }
        }
    } else {
        printf("TAG_TEN\n");
        u32 tid=(u32)term_val(x_rs);
        View*v=&ctx->tensors[tid].view;
        printf("  sh=[");for(u32 d=0;d<v->shape.rank;d++)printf("%u,",v->shape.dims[d]);
        printf("] s=[");for(u32 d=0;d<v->shape.rank;d++)printf("%d,",v->strides[d]);printf("]\n");
    }
    thvm_free(ctx);return 0;
}
