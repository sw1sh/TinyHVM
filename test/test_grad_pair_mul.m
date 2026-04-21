// test_grad_pair_mul.m — Leibniz via TAG_GF/TAG_GB
//
//     (x, dx) = GRAD(MUL(t1, t2))   with target = t1
//
// Expected post-reduce:
//     CTR { c0: MUL(t1, t2)              forward
//           c1: MUL(1,t2) + MUL(t1,0) }   backward = d(t1·t2)/dt1 = t2
//
// (The MUL(_,0) arm may collapse via arithmetic simplification later.)

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    system("rm -rf wl/examples/thvm_graphs/grad_pair_mul && mkdir -p wl/examples/thvm_graphs/grad_pair_mul");
    setenv("THVM_GRAPH", "1", 1);
    setenv("THVM_GRAPH_DIR", "wl/examples/thvm_graphs/grad_pair_mul", 1);
    setenv("THVM_GRAPH_STOP_AFTER_SWEEP", "1", 1);

    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_MUL, a, b);

    Term fwd, bwd;
    thvm_grad_pair(ctx, (u32)term_val(a), y, &fwd, &bwd);
    Term root = thvm_ctr(ctx, (Term[]){fwd, bwd}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    printf("grad_pair_mul smoke test done\n");
    return 0;
}
