#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"

int main() {
    TinyHVM ctx = {0};
    f32 data[72]; for(int i=0;i<72;i++) data[i]=(f32)i;
    Term x = thvm_tensor(&ctx, data, SHAPE(2,1,6,6));
    TensorMeta *mx = &ctx.tensors[(u32)term_val(x)];
    printf("Input shape: [%u,%u,%u,%u] numel=%u\n",
           mx->view.shape.dims[0], mx->view.shape.dims[1],
           mx->view.shape.dims[2], mx->view.shape.dims[3], mx->view.numel);

    u32 k[] = {3,3}, s[] = {1,1};
    Term pooled = thvm_pool(&ctx, x, k, s, 2);
    Term pr = thvm_reduce(&ctx, pooled);
    TensorMeta *mp = &ctx.tensors[(u32)term_val(pr)];
    printf("Pool shape: [");
    for(u32 i=0;i<mp->view.shape.rank;i++) printf("%u%s", mp->view.shape.dims[i], i<mp->view.shape.rank-1?",":"");
    printf("] numel=%u\n", mp->view.numel);

    u32 tgt[] = {2, 1, 1, 1, 4, 4, 3, 3};
    u32 tn = 1; for(int i=0;i<8;i++) tn *= tgt[i];
    printf("Target numel=%u, pool numel=%u, match=%d\n", tn, mp->view.numel, tn==mp->view.numel);

    Term rs = thvm_reshape(&ctx, pr, shape_of(tgt, 8));
    rs = thvm_reduce(&ctx, rs);
    printf("Reshape OK!\n");
    (void)rs;
    return 0;
}
