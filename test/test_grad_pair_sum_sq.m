// test_grad_pair_sum_sq.m — minimal SUM(MUL(t,t)) case for the pair
// bundle adapter. Isolates whether end-to-end readback of thvm_to_host
// on a TAG_GB term works through SUM+MUL backward structures.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    system("rm -rf graphs/grad_pair_sum_sq && mkdir -p graphs/grad_pair_sum_sq");
    setenv("THVM_GRAPH", "1", 1);
    setenv("THVM_GRAPH_DIR", "graphs/grad_pair_sum_sq", 1);
    /* DON'T set STOP_AFTER_SWEEP — we need codegen to materialize TOPs. */

    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3, 4};
    Term a = thvm_tensor(ctx, ad, SHAPE(4));
    thvm_set_requires_grad(ctx, a);

    // Need a second copy of a for MUL(a, a).
    Term a_fwd, a_grad;
    thvm_dup(ctx, thvm_fresh_label(ctx), a, &a_fwd, &a_grad);
    Term a_mul0, a_mul1;
    thvm_dup(ctx, thvm_fresh_label(ctx), a_fwd, &a_mul0, &a_mul1);

    Term sq = thvm_op(ctx, UOP_MUL, a_mul0, a_mul1);
    Term loss = thvm_sum_axes(ctx, sq, (u32[]){0}, 1);

    Term params[] = {a_grad};
    Term bundle = thvm_eval(ctx, thvm_grad_pair_bundle(ctx, loss, params, 1));

    u32 count = thvm_grad_bundle_count(ctx, bundle);
    fprintf(stderr, "bundle_count=%u\n", count);

    Term g = thvm_grad_bundle_get(ctx, bundle, 0);
    fprintf(stderr, "g: tag=%u ext=%u val=%llu\n",
            term_tag(g), term_ext(g), (unsigned long long)term_val(g));

    u32 dtype = DTYPE_F32;
    Shape sh = SHAPE(4);
    f32 *gd = thvm_to_host_raw(ctx, g, &dtype, &sh);
    if (!gd) {
        fprintf(stderr, "gd not readable\n");
        thvm_free(ctx);
        return 1;
    }
    fprintf(stderr, "gd = [%.4f, %.4f, %.4f, %.4f]\n",
            gd[0], gd[1], gd[2], gd[3]);
    thvm_free(ctx);
    return 0;
}
