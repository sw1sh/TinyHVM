// test_fuse_assign_only.m — isolated test: FUSE on SEQ(ASSIGN, x)
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>

int main() {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    TinyHVM *ctx = thvm_init("cpu");

    f32 xd[] = {1.0f, 2.0f, 3.0f};
    Term x = thvm_tensor(ctx, xd, (Shape){.dims={3}, .rank=1});
    f32 yd[] = {2.0f, 2.0f, 2.0f};
    Term y = thvm_tensor(ctx, yd, (Shape){.dims={3}, .rank=1});

    Term mul = thvm_op(ctx, UOP_MUL, x, y);
    Term assign = thvm_assign(ctx, x, mul);
    Term seq = thvm_seq(ctx, assign, x);
    fprintf(stderr, "mul_loc=%llu assign_loc=%llu seq_loc=%llu\n",
            (unsigned long long)term_val(mul), (unsigned long long)term_val(assign),
            (unsigned long long)term_val(seq));

    printf("seq: tag=%u ext=%u val=%llu\n", term_tag(seq), term_ext(seq), (unsigned long long)term_val(seq));

    fprintf(stderr, "before reduce: heap[7]=0x%016llx\n", (unsigned long long)ctx->heap[7]);
    // Watch heap[7] by making heap_set print when loc=7
    ctx->heap[7] |= 0x0100000000000000ULL; // set a sentinel bit (unused TAG bit)
    Term phase1 = thvm_reduce(ctx, seq);
    fprintf(stderr, "phase1: tag=%u ext=%u\n", term_tag(phase1), term_ext(phase1));
    Term r = thvm_eval(ctx, phase1);

    printf("result: tag=%u ext=%u val=%llu raw=0x%016llx\n",
           term_tag(r), term_ext(r), (unsigned long long)term_val(r), (unsigned long long)r);

    u32 dt = DTYPE_F32; Shape sh = SHAPE(1);
    f32 *v = thvm_to_host_raw(ctx, x, &dt, &sh);
    if (v) printf("x = [%.1f, %.1f, %.1f] (expect [2,4,6])\n", v[0], v[1], v[2]);
    else printf("x: to_host failed\n");

    thvm_free(ctx);
    return 0;
}
