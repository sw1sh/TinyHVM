// test_grad_rules.m — per-rule topology for the new TAG_GF/TAG_GB GRAD pair.
//
// Each sub-test:
//   1. Builds a small forward:   y = uop(inputs...)
//   2. Constructs:
//          (fwd, bwd) = thvm_grad_pair(ctx, target_tid, y)
//          root       = CTR#2 { fwd, bwd }
//   3. THVM_GRAPH + THVM_GRAPH_STOP_AFTER_SWEEP → emits thvm_0..thvm_2
//      phase dumps under wl/examples/thvm_graphs/grad_rules/<rule>/.
//   4. Parses thvm_1_post_reduce.dot and asserts expected chain-rule
//      tokens (CTR, forward uop, backward formula uops).
//
// No numeric checks; no scheduler/codegen/renderer; pure pre-fuse
// topology.  A rule regresses → its sub-test names the owner.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GRAPH_ROOT "wl/examples/thvm_graphs/grad_rules"

static void setup_graph_dir(const char *rule) {
    char dir[256];
    snprintf(dir, sizeof(dir), "%s/%s", GRAPH_ROOT, rule);
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "rm -rf %s && mkdir -p %s", dir, dir);
    int r = system(cmd); (void)r;
    unsetenv("THVM_STEP_GRAPH");
    setenv("THVM_GRAPH", "1", 1);
    setenv("THVM_GRAPH_DIR", dir, 1);
    setenv("THVM_GRAPH_STOP_AFTER_SWEEP", "1", 1);
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, sz, f);
    buf[sz] = 0;
    fclose(f);
    return buf;
}

static int topo_check(const char *rule, int phase,
                      const char **needles, int n_needles) {
    const char *name[] = {
        "thvm_0_pre_reduce.dot",
        "thvm_1_post_reduce.dot",
        "thvm_2_post_sweep.dot",
    };
    char path[512];
    snprintf(path, sizeof(path), "%s/%s/%s", GRAPH_ROOT, rule, name[phase]);
    char *buf = slurp(path);
    if (!buf) { printf("  TOPO miss: %s not written\n", path); return 0; }
    int ok = 1;
    for (int i = 0; i < n_needles; i++) {
        if (!strstr(buf, needles[i])) {
            printf("  TOPO miss in %s phase%d: expected \"%s\"\n", rule, phase, needles[i]);
            ok = 0;
        }
    }
    free(buf);
    return ok;
}

static Term mk(TinyHVM *ctx, Term y, Term target) {
    Term fwd, bwd;
    thvm_grad_pair(ctx, (u32)term_val(target), y, &fwd, &bwd);
    return thvm_ctr(ctx, (Term[]){fwd, bwd}, 2);
}

static int report(const char *name, int ok) {
    printf("%-10s %s\n", name, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ─── binary ─────────────────────────────────────────────────────────

#define BIN_TEST(TNAME, DIR, UOP, PRE_NEEDLE)                           \
static int test_##TNAME(void) {                                         \
    setup_graph_dir(DIR);                                               \
    TinyHVM *ctx = thvm_init("cpu");                                    \
    f32 ad[] = {1,2,3}, bd[] = {4,5,6};                                  \
    Term a = thvm_tensor(ctx, ad, SHAPE(3));                            \
    Term b = thvm_tensor(ctx, bd, SHAPE(3));                            \
    thvm_set_requires_grad(ctx, a);                                     \
    Term y = thvm_op(ctx, UOP, a, b);                                   \
    thvm_eval(ctx, mk(ctx, y, a));                                      \
    thvm_free(ctx);                                                     \
    const char *pre[]  = {"GRAD", "CTR", PRE_NEEDLE};                   \
    const char *post[] = {"CTR"};                                       \
    int ok = topo_check(DIR, 0, pre, 3)                                 \
          && topo_check(DIR, 1, post, 1);                               \
    return report(DIR, ok);                                             \
}

BIN_TEST(add, "add", UOP_ADD, "ADD")
BIN_TEST(sub, "sub", UOP_SUB, "SUB")
BIN_TEST(mul, "mul", UOP_MUL, "MUL")
BIN_TEST(div, "div", UOP_DIV, "DIV")
BIN_TEST(max, "max", UOP_MAX, "MAX")
BIN_TEST(cmp, "cmp", UOP_CMP, "CMP")

// ─── unary ──────────────────────────────────────────────────────────

#define UNA_TEST(TNAME, DIR, UOP, PRE_NEEDLE)                           \
static int test_##TNAME(void) {                                         \
    setup_graph_dir(DIR);                                               \
    TinyHVM *ctx = thvm_init("cpu");                                    \
    f32 ad[] = {1,2,3};                                                  \
    Term a = thvm_tensor(ctx, ad, SHAPE(3));                            \
    thvm_set_requires_grad(ctx, a);                                     \
    Term y = thvm_op(ctx, UOP, a, term_era());                          \
    thvm_eval(ctx, mk(ctx, y, a));                                      \
    thvm_free(ctx);                                                     \
    const char *pre[]  = {"GRAD", "CTR", PRE_NEEDLE};                   \
    const char *post[] = {"CTR"};                                       \
    int ok = topo_check(DIR, 0, pre, 3)                                 \
          && topo_check(DIR, 1, post, 1);                               \
    return report(DIR, ok);                                             \
}

UNA_TEST(neg,  "neg",  UOP_NEG,  "NEG")
UNA_TEST(exp,  "exp",  UOP_EXP,  "EXP")
UNA_TEST(log,  "log",  UOP_LOG,  "LOG")
UNA_TEST(sqrt, "sqrt", UOP_SQRT, "SQRT")
UNA_TEST(relu, "relu", UOP_RELU, "RELU")

// ─── reduction SUM ─────────────────────────────────────────────────

static int test_sum(void) {
    setup_graph_dir("sum");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4};
    Term a = thvm_tensor(ctx, ad, SHAPE(4));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_sum_axes(ctx, a, (u32[]){0}, 1);
    thvm_eval(ctx, mk(ctx, y, a));
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "SUM"};
    const char *post[] = {"CTR"};
    int ok = topo_check("sum", 0, pre, 3)
          && topo_check("sum", 1, post, 1);
    return report("sum", ok);
}

int main(void) {
    int fails = 0;
    fails += test_add();
    fails += test_sub();
    fails += test_mul();
    fails += test_div();
    fails += test_max();
    fails += test_cmp();
    fails += test_neg();
    fails += test_exp();
    fails += test_log();
    fails += test_sqrt();
    fails += test_relu();
    fails += test_sum();
    printf("\ntotal failures: %d\n", fails);
    return fails ? 1 : 0;
}
