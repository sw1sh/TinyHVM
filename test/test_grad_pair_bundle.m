// test_grad_pair_bundle.m — bundle adapter over thvm_grad_pair.
//
//     bundle = thvm_grad_pair_bundle(ctx, ADD(t1, t2), [t1, t2], 2)
//     assert thvm_grad_bundle_count(bundle) == 2
//
// Each slot is the backward via the GF/GB rule for that target.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    system("rm -rf wl/examples/thvm_graphs/grad_pair_bundle && mkdir -p wl/examples/thvm_graphs/grad_pair_bundle");
    setenv("THVM_GRAPH", "1", 1);
    setenv("THVM_GRAPH_DIR", "wl/examples/thvm_graphs/grad_pair_bundle", 1);
    setenv("THVM_GRAPH_STOP_AFTER_SWEEP", "1", 1);

    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3}, bd[] = {4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);
    Term y = thvm_op(ctx, UOP_ADD, a, b);

    Term params[] = {a, b};
    Term bundle = thvm_grad_pair_bundle(ctx, y, params, 2);
    thvm_eval(ctx, bundle);

    thvm_free(ctx);
    printf("grad_pair_bundle smoke done\n");
    return 0;
}
