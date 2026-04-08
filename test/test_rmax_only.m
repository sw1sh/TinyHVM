#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
int main(void){
    TinyHVM*ctx=thvm_init("cpu");
    float xd[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    Term x = thvm_tensor(ctx, xd, SHAPE(1,1,4,4));
    Term x_rs = thvm_reshape(ctx, x, SHAPE(1,1,2,2,2,2));
    Term x_pm = thvm_permute(ctx, x_rs, (u32[]){0,1,2,4,3,5}, 6);
    Term mx = thvm_rmax_axes(ctx, x_pm, (u32[]){4,5}, 2);
    float gyd[4]={1,1,1,1};
    Term gy = thvm_tensor(ctx, gyd, SHAPE(1,1,2,2));
    Term mx_rs = thvm_reshape(ctx, mx, SHAPE(1,1,2,2,1,1));
    Term mx_bc = thvm_expand(ctx, mx_rs, SHAPE(1,1,2,2,2,2));
    f32 one = 1.f;
    Term mask = thvm_op(ctx, UOP_SUB, thvm_tensor(ctx, &one, SHAPE(1)),
                    thvm_op(ctx, UOP_CMP, mx_bc, x_pm));
    Term gy_bc = thvm_expand(ctx, thvm_reshape(ctx, gy, SHAPE(1,1,2,2,1,1)), SHAPE(1,1,2,2,2,2));
    Term grad_in_6d = thvm_op(ctx, UOP_MUL, gy_bc, mask);
    Term grad_pm = thvm_permute(ctx, grad_in_6d, (u32[]){0,1,2,4,3,5}, 6);
    Term grad_x = thvm_reshape(ctx, grad_pm, SHAPE(1,1,4,4));
    // Eval
    f32 z[16]={0}; Term dst=thvm_tensor(ctx,z,SHAPE(1,1,4,4));
    u64 loc=heap_alloc(ctx,2);ctx->heap[loc]=dst;ctx->heap[loc+1]=grad_x;
    u64 ac=heap_alloc(ctx,1);ctx->heap[ac]=term_new(TAG_TOP,UOP_ASSIGN,loc);
    thvm_eval(ctx,term_era());
    f32*r=thvm_to_host(ctx,dst);
    printf("grad: ");for(int i=0;i<16;i++)printf("%.1f ",r[i]);printf("\n");
    int has_nz=0;for(int i=0;i<16;i++)if(r[i]>1e-6f||r[i]<-1e-6f)has_nz=1;
    printf(has_nz?"PASS\n":"FAIL: all zeros\n");
    thvm_free(ctx);return 0;
}
