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
    f32 w[]={1,0,0,1}; // weight
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={1,1,3,3},.rank=4});
    Term tw=thvm_tensor(ctx,w,(Shape){.dims={1,1,2,2},.rank=4});
    // Manual conv: pool → reshape → expand → permute → MUL → SUM
    Term pooled=thvm_pool(ctx,tx,(u32[]){2,2},(u32[]){1,1},2);
    // pooled: [1,1,2,2,2,2], strides=[9,9,3,1,3,1]
    // reshape to [1,1,1,1,2,2,2,2] for broadcasting with weight
    Term x_rs=thvm_reshape(ctx,pooled,shape_of((u32[]){1,1,1,1,2,2,2,2},8));
    // expand to [1,1,1,1,2,2,2,2] (no expansion needed since cout=cin_g=1)
    // permute to [1,1,1,2,2,1,2,2] = [BS,groups,rcout,OY,OX,cin_g,KH,KW]
    Term x_perm=thvm_permute(ctx,x_rs,(u32[]){0,1,3,4,5,2,6,7},8);
    // weight: [1,1,1,1,1,1,2,2]
    Term w_rs=thvm_reshape(ctx,tw,shape_of((u32[]){1,1,1,1,1,1,2,2},8));
    // MUL + SUM(axes 5,6,7)
    Term prod=thvm_op(ctx,UOP_MUL,x_perm,w_rs);
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term summed=thvm_sum_axes(ctx,prod,(u32[]){5,6,7},3);
    Term out=thvm_reshape(ctx,summed,shape_of((u32[]){1,1,2,2},4));
    // Eval
    f32 z[4]={0}; Term dst=thvm_tensor(ctx,z,(Shape){.dims={1,1,2,2},.rank=4});
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=out;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32*r=thvm_to_host(ctx,dst);
    printf("conv: [%.1f, %.1f, %.1f, %.1f] (expect [6, 8, 12, 14])\n", r[0],r[1],r[2],r[3]);
    thvm_free(ctx);
    return 0;
}
