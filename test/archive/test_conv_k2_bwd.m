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
    f32 w[]={1,1,1,1};
    f32 b[]={0};
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={1,1,3,3},.rank=4});
    Term tw=thvm_tensor(ctx,w,(Shape){.dims={1,1,2,2},.rank=4});
    Term tb=thvm_tensor(ctx,b,SHAPE(1));
    // Reproduce EXACTLY what the backward chain does:
    // Step 1: gy starts as scalar 1 with shape [1]
    f32 one=1; Term gy=thvm_tensor(ctx,&one,SHAPE(1));
    // Step 2: GRAD_RESHAPE [1] → [1,1,1,1]: reshape(gy, [1,1,1,1])
    gy=thvm_reshape(ctx,gy,SHAPE(1,1,1,1));
    // Step 3: GRAD_SUM all dims: expand([1,1,1,1] → [1,1,2,2])
    gy=thvm_expand(ctx,gy,(Shape){.dims={1,1,2,2},.rank=4});
    // Step 4: GRAD_ADD: sum_to_shape(gy, [1,1,2,2], [1,1,2,2]) = gy (no-op)
    // Step 5: GRAD_RESHAPE [1,1,2,2] → [1,1,1,2,2,1,1,1]
    gy=thvm_reshape(ctx,gy,shape_of((u32[]){1,1,1,2,2,1,1,1},8));
    // Step 6: GRAD_SUM axes=[5,6,7]: expand → [1,1,1,2,2,1,2,2]
    gy=thvm_expand(ctx,gy,shape_of((u32[]){1,1,1,2,2,1,2,2},8));
    // Now gy should be all ones with shape [1,1,1,2,2,1,2,2]
    // Flatten to check values
    Term gy_flat=thvm_reshape(ctx,gy,SHAPE(16));
    f32 z[16]={0}; Term dst=thvm_tensor(ctx,z,SHAPE(16));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=gy_flat;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32*r=thvm_to_host(ctx,dst);
    printf("gy values: ");for(int i=0;i<16;i++)printf("%.1f ",r[i]);printf("\n");
    printf("(expect all 1.0)\n");
    thvm_free(ctx);
    return 0;
}
