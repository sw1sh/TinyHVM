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
    f32 w[]={1,1,1,1}; f32 b[]={0};
    Term tx=thvm_tensor(ctx,x,(Shape){.dims={1,1,3,3},.rank=4});thvm_set_requires_grad(ctx,tx);
    Term tw=thvm_tensor(ctx,w,(Shape){.dims={1,1,2,2},.rank=4});thvm_set_requires_grad(ctx,tw);
    Term tb=thvm_tensor(ctx,b,SHAPE(1));thvm_set_requires_grad(ctx,tb);
    Term h=thvm_conv2d(ctx,tx,tw,tb,1,(u32[]){1,1},(u32[]){0,0,0,0});
    extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
    Term loss=thvm_sum_axes(ctx,h,(u32[]){0,1,2,3},4);loss=thvm_reshape(ctx,loss,SHAPE(1));
    f32 gw0[4]={0}; Term gw=thvm_tensor(ctx,gw0,(Shape){.dims={1,1,2,2},.rank=4});
    Term grad=thvm_grad_multi(ctx,loss,(Term[]){tw},(Term[]){gw},1);
    // Phase 1: reduce
    Term r = thvm_reduce(ctx, grad);
    // Now dump heap: find MUL TAG_TOPs and check their arguments
    printf("After Phase 1 (pre-scheduling):\n");
    for(u64 hp=1;hp<ctx->heap_pos;hp++){
        Term t=ctx->heap[hp];
        if(term_tag(t)==TAG_TOP && term_ext(t)==UOP_MUL){
            u64 loc=term_val(t);
            Term a=ctx->heap[loc], b2=ctx->heap[loc+1];
            printf("MUL@%llu: a=tag%u/%llu b=tag%u/%llu\n",hp,term_tag(a),term_val(a),term_tag(b2),term_val(b2));
        }
    }
    thvm_free(ctx);return 0;
}
