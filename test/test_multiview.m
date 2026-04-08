#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
int main(void){
    TinyHVM*ctx=thvm_init("cpu");
    // Pool stride view [1,1,2,2,2,2] strides [9,9,3,1,3,1] → reshape to [1,1,1,1,2,2,2,2]
    // This is the non-mergeable reshape that previously used the fallback
    f32 x[]={1,2,3,4,5,6,7,8,9};
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={1,1,3,3},.rank=4});
    Term pooled=thvm_pool(ctx,tx,(u32[]){2,2},(u32[]){1,1},2);
    Term x_rs=thvm_reshape(ctx,pooled,shape_of((u32[]){1,1,1,1,2,2,2,2},8));
    // Check ShapeTracker
    if(term_tag(x_rs)==TAG_TOP) {
        const ShapeTracker *st = st_get_tracker(term_val(x_rs));
        if(st) printf("n_views=%u\n", st->n_views);
        else printf("no tracker\n");
    } else printf("TAG_TEN (eager)\n");
    thvm_free(ctx);return 0;
}
