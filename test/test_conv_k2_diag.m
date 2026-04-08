#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
int main(void){
    TinyHVM*ctx=thvm_init("cpu");
    // K=2 conv gradient, manual decomposition to check each step
    f32 x[]={1,2,3,4,5,6,7,8,9}; // [1,1,3,3]
    f32 w[]={1,1,1,1}; // [1,1,2,2]
    f32 b[]={0};
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={1,1,3,3},.rank=4});
    Term tw=thvm_tensor(ctx,w,(Shape){.dims={1,1,2,2},.rank=4});
    Term tb=thvm_tensor(ctx,b,SHAPE(1));
    // Forward: pool → reshape → expand → permute → MUL → SUM → reshape
    Term pooled=thvm_pool(ctx,tx,(u32[]){2,2},(u32[]){1,1},2);
    // pooled: [1,1,2,2,2,2] with strides [9,9,3,1,3,1]
    Term x_rs=thvm_reshape(ctx,pooled,shape_of((u32[]){1,1,1,1,2,2,2,2},8));
    Term x_exp=thvm_expand(ctx,x_rs,shape_of((u32[]){1,1,1,1,2,2,2,2},8));
    Term x_perm=thvm_permute(ctx,x_exp,(u32[]){0,1,3,4,5,2,6,7},8);
    // x_perm: [1,1,1,2,2,1,2,2]
    // Weight gradient: MUL(ones, x_perm) summed over axes [3,4]
    // gy = all ones with shape [1,1,1,2,2,1,2,2]
    u32 gy_n = 1*1*1*2*2*1*2*2; // 16
    f32 gy_d[16]; for(u32 i=0;i<16;i++) gy_d[i]=1.0f;
    Term gy=thvm_tensor(ctx,gy_d,(Shape){.dims={1,1,1,2,2,1,2,2},.rank=8});
    Term prod=thvm_op(ctx,UOP_MUL,gy,x_perm);
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term summed=thvm_sum_axes(ctx,prod,(u32[]){3,4},2);
    Term out=thvm_reshape(ctx,summed,SHAPE(4));
    // Eval
    f32 z[4]={0}; Term dst=thvm_tensor(ctx,z,SHAPE(4));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=out;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32*r=thvm_to_host(ctx,dst);
    printf("grad_w: [%.2f, %.2f, %.2f, %.2f] (expect [12, 16, 24, 28])\n", r[0],r[1],r[2],r[3]);
    thvm_free(ctx);
    return 0;
}
