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
    thvm_grad_pair_target(ctx, target, y, &fwd, &bwd);
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

// ─── view ops ──────────────────────────────────────────────────────

static int test_reshape(void) {
    setup_graph_dir("reshape");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(6));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_reshape(ctx, a, SHAPE(2, 3));
    thvm_eval(ctx, mk(ctx, y, a));
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "RESHAPE"};
    const char *post[] = {"CTR"};
    int ok = topo_check("reshape", 0, pre, 3)
          && topo_check("reshape", 1, post, 1);
    return report("reshape", ok);
}

static int test_expand(void) {
    setup_graph_dir("expand");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_expand(ctx, thvm_reshape(ctx, a, SHAPE(1, 3)), SHAPE(4, 3));
    thvm_eval(ctx, mk(ctx, y, a));
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "EXPAND"};
    const char *post[] = {"CTR"};
    int ok = topo_check("expand", 0, pre, 3)
          && topo_check("expand", 1, post, 1);
    return report("expand", ok);
}

static int test_rmax(void) {
    setup_graph_dir("rmax");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 5, 3, 2};
    Term a = thvm_tensor(ctx, ad, SHAPE(4));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_rmax_axes(ctx, a, (u32[]){0}, 1);
    thvm_eval(ctx, mk(ctx, y, a));
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "RMAX"};
    const char *post[] = {"CTR"};
    int ok = topo_check("rmax", 0, pre, 3)
          && topo_check("rmax", 1, post, 1);
    return report("rmax", ok);
}

static int test_shrink(void) {
    setup_graph_dir("shrink");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(6));
    thvm_set_requires_grad(ctx, a);
    /* shrink[1:5] */
    u32 pairs[2] = {1, 5};
    Term y = thvm_shrink(ctx, a, pairs, 1);
    thvm_eval(ctx, mk(ctx, y, a));
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "SHRINK"};
    const char *post[] = {"CTR"};
    int ok = topo_check("shrink", 0, pre, 3)
          && topo_check("shrink", 1, post, 1);
    return report("shrink", ok);
}

static int test_pad(void) {
    setup_graph_dir("pad");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4};
    Term a = thvm_tensor(ctx, ad, SHAPE(4));
    thvm_set_requires_grad(ctx, a);
    /* pad(1, 1): add 1 element on each side */
    u32 pairs[2] = {1, 1};
    Term y = thvm_pad(ctx, a, pairs, 1);
    thvm_eval(ctx, mk(ctx, y, a));
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "PAD"};
    const char *post[] = {"CTR"};
    int ok = topo_check("pad", 0, pre, 3)
          && topo_check("pad", 1, post, 1);
    return report("pad", ok);
}

// ─── edge cases ────────────────────────────────────────────────────

// Second derivative:  d²/da²(a·a) = d/da(2a) = 2
// First pair produces bwd1 = Leibniz; wrap bwd1 in a second pair to
// capture the derivative of the derivative.
// Multi-target bundle: gradient wrt TWO parameters at once.
//   bundle = grad_pair_bundle(MUL(a, b), [a, b])
// Expected bundle[0] = ∂(a·b)/∂a = b, bundle[1] = ∂(a·b)/∂b = a.
// Chained compute: GRAD output feeds a forward MUL.
//     y        = MUL(a, a)               // forward
//     (_, g)   = GRAD(y, a)               // g = 2a
//     z        = MUL(g, g)                // z = 4a²
// Produces a graph where TAG_GB auxes appear as operands of downstream
// compute TOPs — checks that the renderer + arity lookups handle this.
static int test_chained_compute(void) {
    setup_graph_dir("chained_compute");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term a0, a1;
    thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
    Term y = thvm_op(ctx, UOP_MUL, a0, a1);

    Term fwd, g;
    thvm_grad_pair_target(ctx, a, y, &fwd, &g);
    thvm_spawn_detached_era(ctx, fwd);

    /* DUP g so it can be used twice in MUL(g, g) */
    Term g0, g1;
    thvm_dup(ctx, thvm_fresh_label(ctx), g, &g0, &g1);
    Term z = thvm_op(ctx, UOP_MUL, g0, g1);

    thvm_eval(ctx, z);
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "MUL"};
    int ok = topo_check("chained_compute", 0, pre, 2);
    return report("chained", ok);
}

static int test_bundle_multitarget(void) {
    setup_graph_dir("bundle_multitarget");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3}, bd[] = {4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    Term y = thvm_op(ctx, UOP_MUL, a, b);
    Term params[] = {a, b};
    Term bundle = thvm_grad_pair_bundle(ctx, y, params, 2);
    thvm_eval(ctx, bundle);
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "MUL"};
    const char *post[] = {"CTR"};
    int ok = topo_check("bundle_multitarget", 0, pre, 3)
          && topo_check("bundle_multitarget", 1, post, 1);
    return report("bundle2", ok);
}

static int test_second_derivative(void) {
    setup_graph_dir("second_derivative");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term a0, a1;
    thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
    Term y = thvm_op(ctx, UOP_MUL, a0, a1);

    Term fwd1, bwd1;
    thvm_grad_pair_target(ctx, a, y, &fwd1, &bwd1);
    thvm_spawn_detached_era(ctx, fwd1);

    Term fwd2, bwd2;
    thvm_grad_pair_target(ctx, a, bwd1, &fwd2, &bwd2);

    Term root = thvm_ctr(ctx, (Term[]){fwd2, bwd2}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "MUL"};
    const char *post[] = {"CTR"};
    int ok = topo_check("second_derivative", 0, pre, 3)
          && topo_check("second_derivative", 1, post, 1);
    return report("d2da2", ok);
}

// No-match target sentinel: GRAD(ADD(a, b), target = ~0u).
// No leaf will compare equal → every sub-GRAD on a TEN returns NUM(0)
// → combined backward reduces to NUM(0). Exercises the "target never
// hits a leaf" path that e.g. GRAD wrt a tensor not in the forward
// graph would produce.
//
// NB: the raw u32-target API lets callers pass anything. Real targets
// should be TEN tensor ids; passing `(u32)term_val(TOP)` collides
// with tensor ids that happen to numerically match a heap slot (both
// live in the same 20-bit label field). Use ~0u or the tid of a
// TAG_TEN, never a TOP's val.
static int test_target_nomatch(void) {
    setup_graph_dir("target_nomatch");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3}, bd[] = {4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    Term y = thvm_op(ctx, UOP_ADD, a, b);
    Term fwd, bwd;
    thvm_grad_pair(ctx, ~0u, y, &fwd, &bwd);
    Term root = thvm_ctr(ctx, (Term[]){fwd, bwd}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "ADD"};
    const char *post[] = {"CTR"};
    int ok = topo_check("target_nomatch", 0, pre, 3)
          && topo_check("target_nomatch", 1, post, 1);
    return report("nomatch", ok);
}

static int test_assign(void) {
    setup_graph_dir("assign");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3}, bd[] = {4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_assign(ctx, a, b);
    thvm_eval(ctx, mk(ctx, y, a));
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "ASSIGN"};
    const char *post[] = {"CTR"};
    int ok = topo_check("assign", 0, pre, 3)
          && topo_check("assign", 1, post, 1);
    return report("assign", ok);
}

static int test_mm(void) {
    setup_graph_dir("mm");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4,5,6};
    f32 bd[] = {1,2,3,4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(2, 3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3, 2));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_mm(ctx, a, b);
    thvm_eval(ctx, mk(ctx, y, a));
    thvm_free(ctx);
    // thvm_mm decomposes into RESHAPE/EXPAND/MUL/SUM primitives.
    const char *pre[]  = {"GRAD", "CTR", "MUL"};
    const char *post[] = {"CTR"};
    int ok = topo_check("mm", 0, pre, 3)
          && topo_check("mm", 1, post, 1);
    return report("mm", ok);
}

static int test_permute(void) {
    setup_graph_dir("permute");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(2, 3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_permute(ctx, a, (u32[]){1, 0}, 2);
    thvm_eval(ctx, mk(ctx, y, a));
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "PERMUTE"};
    const char *post[] = {"CTR"};
    int ok = topo_check("permute", 0, pre, 3)
          && topo_check("permute", 1, post, 1);
    return report("permute", ok);
}

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

// y = t1 * t1, d/dt1 = 2*t1 via Leibniz.  Both operands of MUL are the
// same tensor, so the GRAD pair fires twice on the shared target.
static int test_self_mul(void) {
    setup_graph_dir("self_mul");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term a0, a1;
    thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
    Term y = thvm_op(ctx, UOP_MUL, a0, a1);
    thvm_eval(ctx, mk(ctx, y, a));
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "MUL"};
    const char *post[] = {"CTR", "ADD", "MUL"};
    int ok = topo_check("self_mul", 0, pre, 3)
          && topo_check("self_mul", 1, post, 3);
    return report("self_mul", ok);
}

// y = t1 + t1*t2, d/dt1 = 1 + t2.  Target appears both at a top-level
// ADD operand and nested inside MUL — walks exercise the per-leaf
// tid match in the general chain-rule composition.
static int test_nested_reuse(void) {
    setup_graph_dir("nested_reuse");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term a0, a1;
    thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
    Term y = thvm_op(ctx, UOP_ADD, a0, thvm_op(ctx, UOP_MUL, a1, b));
    thvm_eval(ctx, mk(ctx, y, a));
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "ADD", "MUL"};
    const char *post[] = {"CTR", "ADD", "MUL"};
    int ok = topo_check("nested_reuse", 0, pre, 4)
          && topo_check("nested_reuse", 1, post, 3);
    return report("nested_reuse", ok);
}

// y = reshape(t1+t2, [2,3]), target = t2.  Exercises the RESHAPE
// rule's composition when target is NOT the reshape's direct operand
// but a sibling leaf of the ADD underneath.
static int test_reshape_deep_target(void) {
    setup_graph_dir("reshape_deep_target");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4,5,6}, bd[] = {6,5,4,3,2,1};
    Term a = thvm_tensor(ctx, ad, SHAPE(6));
    Term b = thvm_tensor(ctx, bd, SHAPE(6));
    thvm_set_requires_grad(ctx, b);
    Term y = thvm_reshape(ctx, thvm_op(ctx, UOP_ADD, a, b), SHAPE(2, 3));
    thvm_eval(ctx, mk(ctx, y, b));
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "RESHAPE", "ADD"};
    const char *post[] = {"CTR", "RESHAPE", "ADD"};
    int ok = topo_check("reshape_deep_target", 0, pre, 4)
          && topo_check("reshape_deep_target", 1, post, 3);
    return report("reshape_deep_target", ok);
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
    fails += test_reshape();
    fails += test_expand();
    fails += test_permute();
    fails += test_mm();
    fails += test_rmax();
    fails += test_shrink();
    fails += test_pad();
    fails += test_assign();
    fails += test_bundle_multitarget();
    fails += test_chained_compute();
    fails += test_second_derivative();
    fails += test_target_nomatch();
    fails += test_self_mul();
    fails += test_nested_reuse();
    fails += test_reshape_deep_target();
    printf("\ntotal failures: %d\n", fails);
    return fails ? 1 : 0;
}
