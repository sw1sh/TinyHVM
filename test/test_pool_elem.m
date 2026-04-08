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
    // Should be [1,1,2,2,2,2] = 16 elements
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    // Flatten to 1D to read all values
    Term flat=thvm_reshape(ctx,pooled,SHAPE(16));
    f32 z[16]={0}; Term dst=thvm_tensor(ctx,z,SHAPE(16));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=flat;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32*r=thvm_to_host(ctx,dst);
    printf("pool: ");for(int i=0;i<16;i++)printf("%.0f ",r[i]);printf("\n");
    printf("expect: 1 2 4 5 2 3 5 6 4 5 7 8 5 6 8 9\n");
    thvm_free(ctx);
    return 0;
}
