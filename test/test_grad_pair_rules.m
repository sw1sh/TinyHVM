// test_grad_pair_rules.m — per-uop topology for the new TAG_GF/TAG_GB rules.
//
// Each sub-test builds:
//     (fwd, bwd) = GRAD(<uop>(t1, t2))    target = t1
//     root       = CTR { fwd, bwd }
// then dumps thvm_N_*.dot under wl/examples/thvm_graphs/grad_pair_rules/<uop>/.
//
// No numeric checks — just confirms each rule produces a graph and doesn't
// hang.  Look at the phase-1 / phase-2 .dot files to verify the chain
// rule's shape.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <stdlib.h>

#define ROOT "wl/examples/thvm_graphs/grad_pair_rules"

static void setup(const char *name) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "rm -rf %s/%s && mkdir -p %s/%s", ROOT, name, ROOT, name);
    int r = system(cmd); (void)r;
    char dir[256];
    snprintf(dir, sizeof(dir), "%s/%s", ROOT, name);
    setenv("THVM_GRAPH", "1", 1);
    setenv("THVM_GRAPH_DIR", dir, 1);
    setenv("THVM_GRAPH_STOP_AFTER_SWEEP", "1", 1);
}

static void run_binary(const char *name, u32 uop) {
    setup(name);
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3}, bd[] = {4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, uop, a, b);
    Term fwd, bwd;
    thvm_grad_pair(ctx, (u32)term_val(a), y, &fwd, &bwd);
    thvm_eval(ctx, thvm_ctr(ctx, (Term[]){fwd, bwd}, 2));
    thvm_free(ctx);
    printf("%-8s done\n", name);
}

static void run_unary(const char *name, u32 uop) {
    setup(name);
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, uop, a, term_era());
    Term fwd, bwd;
    thvm_grad_pair(ctx, (u32)term_val(a), y, &fwd, &bwd);
    thvm_eval(ctx, thvm_ctr(ctx, (Term[]){fwd, bwd}, 2));
    thvm_free(ctx);
    printf("%-8s done\n", name);
}

int main(void) {
    run_binary("add", UOP_ADD);
    run_binary("sub", UOP_SUB);
    run_binary("mul", UOP_MUL);
    run_binary("div", UOP_DIV);
    run_binary("max", UOP_MAX);
    run_binary("cmp", UOP_CMP);
    run_unary ("neg", UOP_NEG);
    run_unary ("exp", UOP_EXP);
    run_unary ("log", UOP_LOG);
    run_unary ("sqrt", UOP_SQRT);
    run_unary ("relu", UOP_RELU);
    return 0;
}
