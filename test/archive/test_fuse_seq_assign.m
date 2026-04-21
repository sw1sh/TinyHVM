// test_fuse_seq_assign.m — incremental tests toward the loop
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
    Term w = thvm_tensor(ctx, xd, (Shape){.dims={3}, .rank=1});
    f32 sd[] = {2.0f, 2.0f, 2.0f};
    Term s = thvm_tensor(ctx, sd, (Shape){.dims={3}, .rank=1});

    // Test 1: thvm_eval on simple MUL (no loop)
    {
        Term mul = thvm_op(ctx, UOP_MUL, w, s);
        Term r = thvm_eval(ctx, mul);
        printf("test1 eval(MUL): tag=%u ext=%u raw=0x%016llx\n",
               term_tag(r), term_ext(r), (unsigned long long)r);
        u32 dt = DTYPE_F32; Shape sh = SHAPE(1);
        f32 *v = thvm_to_host_raw(ctx, r, &dt, &sh);
        if (v) printf("  result = [%.1f, %.1f, %.1f]\n", v[0], v[1], v[2]);
    }

    // Test 2: direct FUSE on SEQ(ASSIGN(w, w*s), w) — no DUP, direct tensor refs
    {
        TinyHVM *ctx2 = thvm_init("cpu");
        f32 wd2[] = {1.0f, 2.0f, 3.0f};
        Term w2 = thvm_tensor(ctx2, wd2, (Shape){.dims={3}, .rank=1});
        f32 sd2[] = {2.0f, 2.0f, 2.0f};
        Term s2 = thvm_tensor(ctx2, sd2, (Shape){.dims={3}, .rank=1});

        // Use w2 directly (no DUP) — linear use is wrong but tests the mechanism
        Term mul2 = thvm_op(ctx2, UOP_MUL, w2, s2);
        Term assign = thvm_assign(ctx2, w2, mul2);
        Term seq = thvm_seq(ctx2, assign, w2);

        // Use thvm_eval which suppresses quiesce during FUSE pass
        fprintf(stderr, "--- test2 start ---\n");
        Term r = thvm_eval(ctx2, seq);
        printf("test2 FUSE(SEQ(ASSIGN,w)): tag=%u ext=%u raw=0x%016llx\n",
               term_tag(r), term_ext(r), (unsigned long long)r);
        u32 dt = DTYPE_F32; Shape sh = SHAPE(1);
        f32 *v = thvm_to_host_raw(ctx2, w2, &dt, &sh);
        if (v) printf("  w = [%.1f, %.1f, %.1f] (expect [2,4,6])\n", v[0], v[1], v[2]);
        thvm_free(ctx2);
    }

    // Test 3: Two chained SEQ(ASSIGN, SEQ(ASSIGN, w))
    {
        TinyHVM *ctx3 = thvm_init("cpu");
        f32 wd3[] = {1.0f, 2.0f, 3.0f};
        Term w3 = thvm_tensor(ctx3, wd3, (Shape){.dims={3}, .rank=1});
        f32 sd3[] = {2.0f, 2.0f, 2.0f};
        Term s3 = thvm_tensor(ctx3, sd3, (Shape){.dims={3}, .rank=1});

        // w*2 → assign → w*2 again → assign → return w
        Term w3a, w3b, w3c, w3d, w3e;
        { Term t1, t2, t3;
          thvm_dup(ctx3, thvm_fresh_label(ctx3), w3, &w3a, &t1);
          thvm_dup(ctx3, thvm_fresh_label(ctx3), t1, &w3b, &t2);
          thvm_dup(ctx3, thvm_fresh_label(ctx3), t2, &w3c, &t3);
          thvm_dup(ctx3, thvm_fresh_label(ctx3), t3, &w3d, &w3e);
        }
        Term s3a, s3b;
        thvm_dup(ctx3, thvm_fresh_label(ctx3), s3, &s3a, &s3b);

        Term mul1 = thvm_op(ctx3, UOP_MUL, w3a, s3a);
        Term assign1 = thvm_assign(ctx3, w3b, mul1);
        Term mul2 = thvm_op(ctx3, UOP_MUL, w3c, s3b);
        Term assign2 = thvm_assign(ctx3, w3d, mul2);
        Term seq = thvm_seq(ctx3, assign1, thvm_seq(ctx3, assign2, w3e));

        Term r = thvm_eval(ctx3, seq);
        printf("test3 eval(SEQ(ASSIGN,SEQ(ASSIGN,w))): tag=%u ext=%u raw=0x%016llx\n",
               term_tag(r), term_ext(r), (unsigned long long)r);
        u32 dt = DTYPE_F32; Shape sh = SHAPE(1);
        f32 *v = thvm_to_host_raw(ctx3, w3, &dt, &sh);
        if (v) printf("  w = [%.1f, %.1f, %.1f] (expect [4,8,12])\n", v[0], v[1], v[2]);
        thvm_free(ctx3);
    }

    thvm_free(ctx);
    return 0;
}
