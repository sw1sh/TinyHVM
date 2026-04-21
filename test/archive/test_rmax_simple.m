#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
int main(void){
    TinyHVM*ctx=thvm_init("cpu");
    f32 a[]={1,2,3,4}; f32 b[]={2,3,4,5};
    Term ta=thvm_tensor(ctx,a,SHAPE(2,2));
    Term tb=thvm_tensor(ctx,b,SHAPE(2,2));
    Term prod=thvm_op(ctx,UOP_MUL,ta,tb);
    Term reshaped=thvm_reshape(ctx,prod,SHAPE(4));
    f32 z[4]={0}; Term dst=thvm_tensor(ctx,z,SHAPE(4));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=reshaped;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32*r=thvm_to_host(ctx,dst);
    printf("[%.1f,%.1f,%.1f,%.1f] (expect [2,6,12,20])\n",r[0],r[1],r[2],r[3]);
    thvm_free(ctx);return 0;
}
