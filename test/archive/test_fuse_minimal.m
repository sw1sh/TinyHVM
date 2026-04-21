#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
int main() {
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1.0f, 2.0f, 3.0f};
    Term x = thvm_tensor(ctx, xd, (Shape){.dims={3}, .rank=1});
    f32 yd[] = {2.0f, 2.0f, 2.0f};
    Term y = thvm_tensor(ctx, yd, (Shape){.dims={3}, .rank=1});
    Term mul = thvm_op(ctx, UOP_MUL, x, y);

    // Test 1: FUSE on TEN (should pass through)
    {
        u64 fl = heap_alloc(ctx, 1);
        heap_set(ctx, fl, x);
        Term fuse = term_new(TAG_TOP, UOP_FUSE, fl);
        Term r = thvm_reduce(ctx, fuse);
        printf("FUSE(TEN): tag=%u ext=%u (expect TEN=10)\n", term_tag(r), term_ext(r));
    }

    // Test 2: FUSE on MUL (should pass through structural KERNEL to TEN)
    {
        u64 fl = heap_alloc(ctx, 1);
        heap_set(ctx, fl, mul);
        Term fuse = term_new(TAG_TOP, UOP_FUSE, fl);
        Term r = thvm_reduce(ctx, fuse);
        printf("FUSE(MUL): tag=%u ext=%u val=%llu raw=0x%016llx\n",
               term_tag(r), term_ext(r), (unsigned long long)term_val(r), (unsigned long long)r);
    }

    // Test 3: FUSE on SEQ(TEN, TEN)
    {
        Term seq = thvm_seq(ctx, x, y);
        u64 fl = heap_alloc(ctx, 1);
        heap_set(ctx, fl, seq);
        Term fuse = term_new(TAG_TOP, UOP_FUSE, fl);
        Term r = thvm_reduce(ctx, fuse);
        printf("FUSE(SEQ(TEN,TEN)): tag=%u ext=%u val=%llu raw=0x%016llx\n",
               term_tag(r), term_ext(r), (unsigned long long)term_val(r), (unsigned long long)r);
    }

    // Test 4: FUSE on SEQ(ASSIGN(x, MUL(x,y)), x) — the loop pattern
    {
        Term mul2 = thvm_op(ctx, UOP_MUL, x, y);
        Term assign = thvm_assign(ctx, x, mul2);
        Term seq = thvm_seq(ctx, assign, x);
        u64 fl = heap_alloc(ctx, 1);
        heap_set(ctx, fl, seq);
        Term fuse = term_new(TAG_TOP, UOP_FUSE, fl);
        Term r = thvm_reduce(ctx, fuse);
        printf("FUSE(SEQ(ASSIGN,TEN)): tag=%u ext=%u val=%llu raw=0x%016llx\n",
               term_tag(r), term_ext(r), (unsigned long long)term_val(r), (unsigned long long)r);
        u32 dt = DTYPE_F32; Shape sh = SHAPE(1);
        f32 *v = thvm_to_host_raw(ctx, x, &dt, &sh);
        if (v) printf("  x = [%.1f, %.1f, %.1f] (expect [2,4,6])\n", v[0], v[1], v[2]);
    }

    thvm_free(ctx);
    return 0;
}
