// test_grad_pair.m — smoke test for the new TAG_GF / TAG_GB primitive.
//
// Constructs:
//     (x, dx) = GRAD(ADD(t1, t2))
//     root    = CTR#2 { x, dx }
// and dumps the Phase-0 pre-reduce graph so we can see the shape:
//     ADD ──(y)──> GRAD_cell ──fwd──> CTR.c0
//                          └──bwd──> CTR.c1
// The cell has a single heap slot containing ADD; the two aux projections
// (TAG_GF, TAG_GB) sit as CTR children pointing at the cell.
//
// No interaction rules for TAG_GF / TAG_GB yet — that comes next. This
// test just validates the construction + renderer scaffold.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    system("rm -rf wl/examples/thvm_graphs/grad_pair && mkdir -p wl/examples/thvm_graphs/grad_pair");
    setenv("THVM_GRAPH", "1", 1);
    setenv("THVM_GRAPH_DIR", "wl/examples/thvm_graphs/grad_pair", 1);
    setenv("THVM_GRAPH_STOP_AFTER_SWEEP", "1", 1);

    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_ADD, a, b);

    Term fwd, bwd;
    thvm_grad_pair(ctx, thvm_fresh_label(ctx), y, &fwd, &bwd);
    Term root = thvm_ctr(ctx, (Term[]){fwd, bwd}, 2);
    thvm_eval(ctx, root);

    thvm_free(ctx);
    printf("grad_pair smoke test done\n");
    return 0;
}
