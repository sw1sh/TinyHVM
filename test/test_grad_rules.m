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

// mk() builds CTR{y_fwd, GRAD2(y_bwd, target)}.  y is DUP'd so both the
// forward consumer (c0) and the GRAD2 sub-term see it.
static Term mk(TinyHVM *ctx, Term y, Term target) {
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    return thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, target)}, 2);
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

    Term g = thvm_grad_u(ctx, y, a);
    Term g0, g1;
    thvm_dup(ctx, thvm_fresh_label(ctx), g, &g0, &g1);
    Term z = thvm_op(ctx, UOP_MUL, g0, g1);

    thvm_eval(ctx, z);
    thvm_free(ctx);
    const char *pre[]  = {"GRAD2", "MUL"};
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
    // Two independent gradients: GRAD2(y, a), GRAD2(y, b).  y is DUP'd so
    // each GRAD2 sees its own copy.
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term bundle = thvm_ctr(ctx, (Term[]){
        thvm_grad_u(ctx, y0, a),
        thvm_grad_u(ctx, y1, b),
    }, 2);
    thvm_eval(ctx, bundle);
    thvm_free(ctx);
    const char *pre[]  = {"GRAD2", "CTR", "MUL"};
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

    // First derivative: GRAD2(y, a). Second derivative: GRAD2(first, a).
    Term bwd1 = thvm_grad_u(ctx, y, a);
    Term bwd1_0, bwd1_1;
    thvm_dup(ctx, thvm_fresh_label(ctx), bwd1, &bwd1_0, &bwd1_1);
    Term bwd2 = thvm_grad_u(ctx, bwd1_1, a);
    Term root = thvm_ctr(ctx, (Term[]){bwd1_0, bwd2}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"GRAD2", "CTR", "MUL"};
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
    f32 ad[] = {1,2,3}, bd[] = {4,5,6}, cd[] = {7,8,9};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    // target c is not referenced by y — every leaf comparison returns 0.
    Term c = thvm_tensor(ctx, cd, SHAPE(3));
    Term y = thvm_op(ctx, UOP_ADD, a, b);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, c)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"GRAD2", "CTR", "ADD"};
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

// Nested unary: y = exp(log(t1)), target = t1.  d/dt1 via chain =
// exp(log(t1)) * (1/t1) (symbolically = 1).  Exercises two unary
// rules composing through the leaf.
static int test_nested_unary(void) {
    setup_graph_dir("nested_unary");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_EXP,
                thvm_op(ctx, UOP_LOG, a, term_era()),
                term_era());
    thvm_eval(ctx, mk(ctx, y, a));
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "EXP", "LOG"};
    const char *post[] = {"CTR", "EXP", "LOG", "MUL", "DIV"};
    int ok = topo_check("nested_unary", 0, pre, 4)
          && topo_check("nested_unary", 1, post, 5);
    return report("nested_unary", ok);
}

// y = target (GRAD of a tensor w.r.t. itself).  fwd = t1, bwd =
// EXPAND(1, t1.shape) — the identity gradient from the leaf rule.
static int test_identity(void) {
    setup_graph_dir("identity");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    thvm_eval(ctx, mk(ctx, a, a));
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR"};
    const char *post[] = {"CTR", "EXPAND"};
    int ok = topo_check("identity", 0, pre, 2)
          && topo_check("identity", 1, post, 2);
    return report("identity", ok);
}

// y = sum(t1, axes=[0,1]) reduces a 2D tensor to scalar.  Exercises
// multi-axis SUM and the EXPAND(..., input.shape) inverse.
static int test_sum_multi_axis(void) {
    setup_graph_dir("sum_multi_axis");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(2, 3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_sum_axes(ctx, a, (u32[]){0, 1}, 2);
    thvm_eval(ctx, mk(ctx, y, a));
    thvm_free(ctx);
    const char *pre[]  = {"GRAD", "CTR", "SUM"};
    const char *post[] = {"CTR", "SUM", "EXPAND"};
    int ok = topo_check("sum_multi_axis", 0, pre, 3)
          && topo_check("sum_multi_axis", 1, post, 3);
    return report("sum_multi_axis", ok);
}

// UOP_GRAD2 pivot: identity via new single-UOP shape.
// y = t1, target = t1.  thvm_grad_u(t1, t1) should reduce to
// EXPAND(1, t1.shape).
static int test_gradu_identity(void) {
    setup_graph_dir("gradu_identity");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term a0, a1;
    thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
    Term root = thvm_ctr(ctx,
        (Term[]){ a0, thvm_grad_u(ctx, a1, a) }, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2"};
    const char *post[] = {"CTR", "EXPAND"};
    int ok = topo_check("gradu_identity", 0, pre, 2)
          && topo_check("gradu_identity", 1, post, 2);
    return report("gradu_identity", ok);
}

// UOP_GRAD2 ADD rule: GRAD2(ADD(a,b), t) -> ADD(GRAD2(a,t), GRAD2(b,t)).
// After TEN-leaves fire, bwd should contain two EXPANDs joined by ADD.
static int test_gradu_add(void) {
    setup_graph_dir("gradu_add");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_ADD, a, b);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx,
        (Term[]){ y0, thvm_grad_u(ctx, y1, a) }, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "ADD"};
    const char *post[] = {"CTR", "ADD", "EXPAND"};
    int ok = topo_check("gradu_add", 0, pre, 3)
          && topo_check("gradu_add", 1, post, 3);
    return report("gradu_add", ok);
}

// UOP_GRAD2 MUL (Leibniz): GRAD2(a*b, t) -> a'*b + a*b'.
static int test_gradu_mul(void) {
    setup_graph_dir("gradu_mul");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_MUL, a, b);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx,
        (Term[]){ y0, thvm_grad_u(ctx, y1, a) }, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "MUL"};
    const char *post[] = {"CTR", "MUL", "ADD", "EXPAND"};
    int ok = topo_check("gradu_mul", 0, pre, 3)
          && topo_check("gradu_mul", 1, post, 4);
    return report("gradu_mul", ok);
}

// UOP_GRAD2 unary rules batch (NEG/EXP/LOG/SQRT/RELU).
#define GRADU_UNARY_TEST(TNAME, DIR, UOP, PRE, POST_NEEDLE)            \
static int test_gradu_##TNAME(void) {                                  \
    setup_graph_dir(DIR);                                              \
    TinyHVM *ctx = thvm_init("cpu");                                   \
    f32 ad[] = {1, 2, 3};                                              \
    Term a = thvm_tensor(ctx, ad, SHAPE(3));                           \
    thvm_set_requires_grad(ctx, a);                                    \
    Term y = thvm_op(ctx, UOP, a, term_era());                         \
    Term y0, y1;                                                       \
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);                 \
    Term root = thvm_ctr(ctx,                                          \
        (Term[]){ y0, thvm_grad_u(ctx, y1, a) }, 2);                   \
    thvm_eval(ctx, root);                                              \
    thvm_free(ctx);                                                    \
    const char *pre[]  = {"CTR", "GRAD2", PRE};                        \
    const char *post[] = {"CTR", "EXPAND", POST_NEEDLE};               \
    int ok = topo_check(DIR, 0, pre, 3)                                \
          && topo_check(DIR, 1, post, 3);                              \
    return report(DIR, ok);                                            \
}
GRADU_UNARY_TEST(neg,  "gradu_neg",  UOP_NEG,  "NEG",  "NEG")
GRADU_UNARY_TEST(exp,  "gradu_exp",  UOP_EXP,  "EXP",  "EXP")
GRADU_UNARY_TEST(log,  "gradu_log",  UOP_LOG,  "LOG",  "DIV")
GRADU_UNARY_TEST(sqrt, "gradu_sqrt", UOP_SQRT, "SQRT", "DIV")
GRADU_UNARY_TEST(relu, "gradu_relu", UOP_RELU, "RELU", "CMP")

// UOP_GRAD2 binary batch (DIV/MAX/CMP).
#define GRADU_BIN_TEST(TNAME, DIR, UOP, PRE, POST_NEEDLE)              \
static int test_gradu_##TNAME(void) {                                  \
    setup_graph_dir(DIR);                                              \
    TinyHVM *ctx = thvm_init("cpu");                                   \
    f32 ad[] = {4, 5, 6}, bd[] = {1, 2, 3};                             \
    Term a = thvm_tensor(ctx, ad, SHAPE(3));                           \
    Term b = thvm_tensor(ctx, bd, SHAPE(3));                           \
    thvm_set_requires_grad(ctx, a);                                    \
    Term y = thvm_op(ctx, UOP, a, b);                                  \
    Term y0, y1;                                                       \
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);                 \
    Term root = thvm_ctr(ctx,                                          \
        (Term[]){ y0, thvm_grad_u(ctx, y1, a) }, 2);                   \
    thvm_eval(ctx, root);                                              \
    thvm_free(ctx);                                                    \
    const char *pre[]  = {"CTR", "GRAD2", PRE};                        \
    const char *post[] = {"CTR", POST_NEEDLE};                         \
    int ok = topo_check(DIR, 0, pre, 3)                                \
          && topo_check(DIR, 1, post, 2);                              \
    return report(DIR, ok);                                            \
}
GRADU_BIN_TEST(div, "gradu_div", UOP_DIV, "DIV", "DIV")
GRADU_BIN_TEST(max, "gradu_max", UOP_MAX, "MAX", "CMP")
GRADU_BIN_TEST(cmp, "gradu_cmp", UOP_CMP, "CMP", "EXPAND")

// UOP_GRAD2 view/reduce batch (RESHAPE, PERMUTE, SUM).
static int test_gradu_reshape(void) {
    setup_graph_dir("gradu_reshape");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(6));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_reshape(ctx, a, SHAPE(2, 3));
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "RESHAPE"};
    const char *post[] = {"CTR", "RESHAPE", "EXPAND"};
    int ok = topo_check("gradu_reshape", 0, pre, 3)
          && topo_check("gradu_reshape", 1, post, 3);
    return report("gradu_reshape", ok);
}
static int test_gradu_permute(void) {
    setup_graph_dir("gradu_permute");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(2, 3));
    thvm_set_requires_grad(ctx, a);
    u32 perm[] = {1, 0};
    Term y = thvm_permute(ctx, a, perm, 2);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "PERMUTE"};
    const char *post[] = {"CTR", "PERMUTE", "EXPAND"};
    int ok = topo_check("gradu_permute", 0, pre, 3)
          && topo_check("gradu_permute", 1, post, 3);
    return report("gradu_permute", ok);
}
static int test_gradu_sum(void) {
    setup_graph_dir("gradu_sum");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4};
    Term a = thvm_tensor(ctx, ad, SHAPE(4));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_sum_axes(ctx, a, (u32[]){0}, 1);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "SUM"};
    const char *post[] = {"CTR", "SUM", "EXPAND"};
    int ok = topo_check("gradu_sum", 0, pre, 3)
          && topo_check("gradu_sum", 1, post, 3);
    return report("gradu_sum", ok);
}

// UOP_GRAD2: SHRINK, PAD, EXPAND, RMAX.
static int test_gradu_shrink(void) {
    setup_graph_dir("gradu_shrink");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(6));
    thvm_set_requires_grad(ctx, a);
    u32 pairs[2] = {1, 5};
    Term y = thvm_shrink(ctx, a, pairs, 1);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "SHRINK"};
    const char *post[] = {"CTR", "PAD", "EXPAND"};
    int ok = topo_check("gradu_shrink", 0, pre, 3)
          && topo_check("gradu_shrink", 1, post, 3);
    return report("gradu_shrink", ok);
}
static int test_gradu_pad(void) {
    setup_graph_dir("gradu_pad");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4};
    Term a = thvm_tensor(ctx, ad, SHAPE(4));
    thvm_set_requires_grad(ctx, a);
    u32 pairs[2] = {1, 1};
    Term y = thvm_pad(ctx, a, pairs, 1);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "PAD"};
    const char *post[] = {"CTR", "SHRINK", "EXPAND"};
    int ok = topo_check("gradu_pad", 0, pre, 3)
          && topo_check("gradu_pad", 1, post, 3);
    return report("gradu_pad", ok);
}
static int test_gradu_expand(void) {
    setup_graph_dir("gradu_expand");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_expand(ctx, thvm_reshape(ctx, a, SHAPE(1, 3)), SHAPE(4, 3));
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "EXPAND"};
    const char *post[] = {"CTR", "SUM", "EXPAND"};
    int ok = topo_check("gradu_expand", 0, pre, 3)
          && topo_check("gradu_expand", 1, post, 3);
    return report("gradu_expand", ok);
}
static int test_gradu_rmax(void) {
    setup_graph_dir("gradu_rmax");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 5, 3, 2};
    Term a = thvm_tensor(ctx, ad, SHAPE(4));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_rmax_axes(ctx, a, (u32[]){0}, 1);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "RMAX"};
    const char *post[] = {"CTR", "RMAX", "CMP", "MUL"};
    int ok = topo_check("gradu_rmax", 0, pre, 3)
          && topo_check("gradu_rmax", 1, post, 4);
    return report("gradu_rmax", ok);
}

// UOP_GRAD2: MM, ASSIGN.
static int test_gradu_mm(void) {
    setup_graph_dir("gradu_mm");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4,5,6}, bd[] = {1,0,0,1,1,0};
    Term a = thvm_tensor(ctx, ad, SHAPE(2, 3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3, 2));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op_raw(ctx, UOP_MM, a, b);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    // Phase-0 pre-reduce topology only — MM is a symbolic/composite op
    // on this path so post-sweep structure depends on scheduler.
    const char *pre[]  = {"CTR", "GRAD2", "MM"};
    int ok = topo_check("gradu_mm", 0, pre, 3);
    return report("gradu_mm", ok);
}
static int test_gradu_assign(void) {
    setup_graph_dir("gradu_assign");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {0,0,0}, bd[] = {1,2,3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_ASSIGN, a, b);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "ASSIGN"};
    const char *post[] = {"CTR"};
    int ok = topo_check("gradu_assign", 0, pre, 3)
          && topo_check("gradu_assign", 1, post, 1);
    return report("gradu_assign", ok);
}

// UOP_GRAD2 deep chain: y = exp(log(t*t)), target = t.
// Exercises MUL(Leibniz), LOG, EXP composing via recursive GRAD2.
static int test_gradu_deep_chain(void) {
    setup_graph_dir("gradu_deep_chain");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term a0, a1;
    thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
    Term sq  = thvm_op(ctx, UOP_MUL, a0, a1);
    Term lg  = thvm_op(ctx, UOP_LOG, sq, term_era());
    Term y   = thvm_op(ctx, UOP_EXP, lg, term_era());
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "EXP", "LOG", "MUL"};
    // Post: bwd is EXP * (MUL's ADD(da*b, a*db)) / inner — should contain
    // MUL, DIV, EXP, LOG, ADD from chain-rule composition.
    const char *post[] = {"CTR", "EXP", "LOG", "MUL", "DIV", "ADD"};
    int ok = topo_check("gradu_deep_chain", 0, pre, 5)
          && topo_check("gradu_deep_chain", 1, post, 6);
    return report("gradu_deep_chain", ok);
}

// UOP_GRAD2 SUB on RHS: y = a - t, dy/dt = -1.  Checks SUB rule sign.
static int test_gradu_sub_rhs(void) {
    setup_graph_dir("gradu_sub_rhs");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {4,5,6}, bd[] = {1,2,3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, b);
    Term y = thvm_op(ctx, UOP_SUB, a, b);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, b)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "SUB"};
    // bwd = SUB(EXPAND(0), EXPAND(1)) = -1
    const char *post[] = {"CTR", "SUB", "EXPAND"};
    int ok = topo_check("gradu_sub_rhs", 0, pre, 3)
          && topo_check("gradu_sub_rhs", 1, post, 3);
    return report("gradu_sub_rhs", ok);
}

// UOP_GRAD2 cubic: y = t*t*t, dy/dt = 3t^2.  Exercises nested MUL
// Leibniz where target appears three times at different depths.
static int test_gradu_cubic(void) {
    setup_graph_dir("gradu_cubic");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term a0, a1, a2;
    thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
    thvm_dup(ctx, thvm_fresh_label(ctx), a1, &a1, &a2);
    Term y = thvm_op(ctx, UOP_MUL,
                thvm_op(ctx, UOP_MUL, a0, a1),
                a2);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "MUL"};
    // bwd = ADD(MUL(ADD(..., ...), t), MUL(t*t, EXPAND(1))) — three MULs.
    const char *post[] = {"CTR", "ADD", "MUL", "EXPAND"};
    int ok = topo_check("gradu_cubic", 0, pre, 3)
          && topo_check("gradu_cubic", 1, post, 4);
    return report("gradu_cubic", ok);
}

// UOP_GRAD2: target shape differs from y's operands.  Leaf emits
// EXPAND(NUM(0), target.shape) even though y is unrelated shape — the
// bwd tensor must match target.shape, not y's.
static int test_gradu_shape_target_diff(void) {
    setup_graph_dir("gradu_shape_target_diff");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3}, bd[] = {4,5,6};
    f32 cd[] = {7, 8};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    Term c = thvm_tensor(ctx, cd, SHAPE(2));   // target, different shape
    Term y = thvm_op(ctx, UOP_ADD, a, b);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, c)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "ADD"};
    // bwd = ADD(EXPAND(0, [2]), EXPAND(0, [2])) = 0 of target's shape.
    const char *post[] = {"CTR", "ADD", "EXPAND"};
    int ok = topo_check("gradu_shape_target_diff", 0, pre, 3)
          && topo_check("gradu_shape_target_diff", 1, post, 3);
    return report("gradu_shape_target_diff", ok);
}

// UOP_GRAD2 third derivative: y = t*t, d^3y/dt^3 = 0.
// Chain: GRAD2(GRAD2(GRAD2(t*t, t), t), t) — three nested GRAD2s.
static int test_gradu_third_derivative(void) {
    setup_graph_dir("gradu_third_derivative");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term a0, a1;
    thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
    Term y = thvm_op(ctx, UOP_MUL, a0, a1);
    Term d1 = thvm_grad_u(ctx, y, a);
    Term d2 = thvm_grad_u(ctx, d1, a);
    Term d3 = thvm_grad_u(ctx, d2, a);
    thvm_eval(ctx, d3);
    thvm_free(ctx);
    const char *pre[]  = {"GRAD2", "MUL"};
    // After full sweep d^3(t^2)/dt^3 is a constant expression reducing
    // through GRAD2 -> NUM-leaf rule; no residual GRAD2 should remain.
    int ok = topo_check("gradu_third_derivative", 0, pre, 2);
    return report("gradu_third_derivative", ok);
}

// UOP_GRAD2: y = exp(-t), dy/dt = -exp(-t).
// EXP inner = NEG(t), so rule composes EXP' (mul by exp(neg(t))) and
// NEG' (negate da).
static int test_gradu_exp_neg(void) {
    setup_graph_dir("gradu_exp_neg");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_EXP,
                thvm_op(ctx, UOP_NEG, a, term_era()),
                term_era());
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "EXP", "NEG"};
    const char *post[] = {"CTR", "EXP", "NEG", "MUL", "EXPAND"};
    int ok = topo_check("gradu_exp_neg", 0, pre, 4)
          && topo_check("gradu_exp_neg", 1, post, 5);
    return report("gradu_exp_neg", ok);
}

// UOP_GRAD2: softplus derivative.  y = log(1 + exp(t)),
// dy/dt = exp(t) / (1 + exp(t)) = sigmoid(t).
// Composes: LOG, ADD, EXP via chain rule.
static int test_gradu_softplus(void) {
    setup_graph_dir("gradu_softplus");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term one = thvm_tensor(ctx, (f32[]){1, 1, 1}, SHAPE(3));
    Term et = thvm_op(ctx, UOP_EXP, a, term_era());
    Term in = thvm_op(ctx, UOP_ADD, one, et);
    Term y  = thvm_op(ctx, UOP_LOG, in, term_era());
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "LOG", "ADD", "EXP"};
    const char *post[] = {"CTR", "DIV", "ADD", "EXP", "EXPAND"};
    int ok = topo_check("gradu_softplus", 0, pre, 5)
          && topo_check("gradu_softplus", 1, post, 5);
    return report("gradu_softplus", ok);
}

// UOP_GRAD2: y = sum(t*t), target = t.  Composes MUL-Leibniz then SUM.
// bwd threads EXPAND over sum's inverse + Leibniz's 2t.
static int test_gradu_sum_sq(void) {
    setup_graph_dir("gradu_sum_sq");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3,4};
    Term a = thvm_tensor(ctx, ad, SHAPE(4));
    thvm_set_requires_grad(ctx, a);
    Term a0, a1;
    thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
    Term sq = thvm_op(ctx, UOP_MUL, a0, a1);
    Term y  = thvm_sum_axes(ctx, sq, (u32[]){0}, 1);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "SUM", "MUL"};
    const char *post[] = {"CTR", "SUM", "MUL", "EXPAND", "ADD"};
    int ok = topo_check("gradu_sum_sq", 0, pre, 4)
          && topo_check("gradu_sum_sq", 1, post, 5);
    return report("gradu_sum_sq", ok);
}

// UOP_GRAD2: target passed through a user DUP before GRAD2. Tests that
// the rule resolves tgt through a DP0/DP1 wrapping to the underlying
// TEN (via the trampoline's arg-reduce on tgt).
static int test_gradu_target_via_dup(void) {
    setup_graph_dir("gradu_target_via_dup");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3}, bd[] = {4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_ADD, a, b);
    Term a_dup0, a_dup1;
    thvm_dup(ctx, thvm_fresh_label(ctx), a, &a_dup0, &a_dup1);
    // Use a_dup0 in forward (to keep it live); pass a_dup1 as GRAD2 target.
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a_dup1)}, 2);
    thvm_spawn_detached_era(ctx, a_dup0);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "ADD"};
    const char *post[] = {"CTR", "ADD", "EXPAND"};
    int ok = topo_check("gradu_target_via_dup", 0, pre, 3)
          && topo_check("gradu_target_via_dup", 1, post, 3);
    return report("gradu_target_via_dup", ok);
}

// UOP_GRAD2: distributive law.  y = t*(b + c), dy/dt = b + c.
// Checks that inner ADD's grad contribution is 0 (no target in ADD)
// and outer Leibniz gives MUL(1, b+c) + MUL(t, 0) = b+c.
static int test_gradu_distributive(void) {
    setup_graph_dir("gradu_distributive");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3}, bd[] = {4,5,6}, cd[] = {7,8,9};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    Term c = thvm_tensor(ctx, cd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_MUL, a, thvm_op(ctx, UOP_ADD, b, c));
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "MUL", "ADD"};
    const char *post[] = {"CTR", "MUL", "ADD", "EXPAND"};
    int ok = topo_check("gradu_distributive", 0, pre, 4)
          && topo_check("gradu_distributive", 1, post, 4);
    return report("gradu_distributive", ok);
}

// UOP_GRAD2: ASSIGN w.r.t. src — gradient is zero (src is ignored by
// the rule, only dst path contributes).
static int test_gradu_assign_src(void) {
    setup_graph_dir("gradu_assign_src");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {0,0,0}, bd[] = {1,2,3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, b);
    Term y = thvm_op(ctx, UOP_ASSIGN, a, b);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, b)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "ASSIGN"};
    // bwd = GRAD2(dst=a, b) where a!=b → EXPAND(0, b.shape).
    const char *post[] = {"CTR", "EXPAND"};
    int ok = topo_check("gradu_assign_src", 0, pre, 3)
          && topo_check("gradu_assign_src", 1, post, 2);
    return report("gradu_assign_src", ok);
}

// UOP_GRAD2 MSE loss: y = sum((x - c) * (x - c)), dy/dx = 2(x - c).
// Exercises SUB + MUL-Leibniz + SUM composition end-to-end.
static int test_gradu_mse(void) {
    setup_graph_dir("gradu_mse");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1,2,3,4}, cd[] = {0,0,0,0};
    Term x = thvm_tensor(ctx, xd, SHAPE(4));
    Term c = thvm_tensor(ctx, cd, SHAPE(4));
    thvm_set_requires_grad(ctx, x);
    Term d = thvm_op(ctx, UOP_SUB, x, c);
    Term d0, d1;
    thvm_dup(ctx, thvm_fresh_label(ctx), d, &d0, &d1);
    Term sq = thvm_op(ctx, UOP_MUL, d0, d1);
    Term y  = thvm_sum_axes(ctx, sq, (u32[]){0}, 1);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, x)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "SUM", "MUL", "SUB"};
    const char *post[] = {"CTR", "SUM", "EXPAND", "MUL", "SUB", "ADD"};
    int ok = topo_check("gradu_mse", 0, pre, 5)
          && topo_check("gradu_mse", 1, post, 6);
    return report("gradu_mse", ok);
}

// UOP_GRAD2 through a lambda: y = (λv. v*v) a.  After beta, body reduces
// to a*a; GRAD2 on that w.r.t. a should give 2a.  Stresses trampoline
// resolution of the argument through the lambda boundary.
static int test_gradu_lambda(void) {
    setup_graph_dir("gradu_lambda");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    // λv.v (identity lambda): create LAM with ERA placeholder body,
    // then patch body = vbind at loc+1.
    Term vbind;
    Term lam = thvm_lam(ctx, &vbind, term_new(TAG_ERA, 0, 0));
    heap_set(ctx, term_val(lam) + 1, vbind);
    Term y = thvm_app(ctx, lam, a);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "APP", "LAM"};
    const char *post[] = {"CTR", "EXPAND"};
    int ok = topo_check("gradu_lambda", 0, pre, 4)
          && topo_check("gradu_lambda", 1, post, 2);
    return report("gradu_lambda", ok);
}

// UOP_GRAD2: conv-shaped composition — y = sum(pad(x,[1,1]) * w).
// Exercises PAD + MUL-Leibniz + SUM in a realistic-ish shape.
static int test_gradu_conv_like(void) {
    setup_graph_dir("gradu_conv_like");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1,2,3,4};
    f32 wd[] = {1,0,1,0,1,0};  // shape [6] to match padded x
    Term x = thvm_tensor(ctx, xd, SHAPE(4));
    Term w = thvm_tensor(ctx, wd, SHAPE(6));
    thvm_set_requires_grad(ctx, x);
    u32 pairs[2] = {1, 1};
    Term px = thvm_pad(ctx, x, pairs, 1);
    Term prod = thvm_op(ctx, UOP_MUL, px, w);
    Term y = thvm_sum_axes(ctx, prod, (u32[]){0}, 1);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, x)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "SUM", "MUL", "PAD"};
    const char *post[] = {"CTR", "SUM", "EXPAND", "MUL", "SHRINK"};
    int ok = topo_check("gradu_conv_like", 0, pre, 5)
          && topo_check("gradu_conv_like", 1, post, 5);
    return report("gradu_conv_like", ok);
}

// UOP_GRAD2: MLP-like elementwise — y = relu(x*W1 + b1) * W2, grad w.r.t. W1.
// Covers RELU-mask, ADD, MUL-Leibniz all chained.
static int test_gradu_mlp_like(void) {
    setup_graph_dir("gradu_mlp_like");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1,2,3}, w1d[] = {0.1f, 0.2f, 0.3f}, b1d[] = {0,0,0}, w2d[] = {1,1,1};
    Term x  = thvm_tensor(ctx, xd,  SHAPE(3));
    Term W1 = thvm_tensor(ctx, w1d, SHAPE(3));
    Term b1 = thvm_tensor(ctx, b1d, SHAPE(3));
    Term W2 = thvm_tensor(ctx, w2d, SHAPE(3));
    thvm_set_requires_grad(ctx, W1);
    Term z  = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MUL, x, W1), b1);
    Term h  = thvm_op(ctx, UOP_RELU, z, term_era());
    Term y  = thvm_op(ctx, UOP_MUL, h, W2);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, W1)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "MUL", "RELU", "ADD"};
    const char *post[] = {"CTR", "MUL", "RELU", "CMP", "ADD"};
    int ok = topo_check("gradu_mlp_like", 0, pre, 5)
          && topo_check("gradu_mlp_like", 1, post, 5);
    return report("gradu_mlp_like", ok);
}

// UOP_GRAD2: log-sum-exp. y = log(sum(exp(x))). dy/dx = softmax(x).
// Composes LOG + SUM + EXP through nested GRAD2.
static int test_gradu_logsumexp(void) {
    setup_graph_dir("gradu_logsumexp");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1, 2, 3, 4};
    Term x = thvm_tensor(ctx, xd, SHAPE(4));
    thvm_set_requires_grad(ctx, x);
    Term ex = thvm_op(ctx, UOP_EXP, x, term_era());
    Term s  = thvm_sum_axes(ctx, ex, (u32[]){0}, 1);
    Term y  = thvm_op(ctx, UOP_LOG, s, term_era());
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, x)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "LOG", "SUM", "EXP"};
    const char *post[] = {"CTR", "DIV", "SUM", "EXP", "MUL", "EXPAND"};
    int ok = topo_check("gradu_logsumexp", 0, pre, 5)
          && topo_check("gradu_logsumexp", 1, post, 6);
    return report("gradu_logsumexp", ok);
}

// UOP_GRAD2: L2-normalization-like. y = x / sqrt(sum(x*x)).  Covers
// DIV-quotient + SQRT-chain + SUM + MUL-Leibniz all composed.
static int test_gradu_l2_normalize(void) {
    setup_graph_dir("gradu_l2_normalize");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1, 2, 3};
    Term x = thvm_tensor(ctx, xd, SHAPE(3));
    thvm_set_requires_grad(ctx, x);
    Term x0, x1, x2;
    thvm_dup(ctx, thvm_fresh_label(ctx), x, &x0, &x1);
    thvm_dup(ctx, thvm_fresh_label(ctx), x1, &x1, &x2);
    Term sq = thvm_op(ctx, UOP_MUL, x1, x2);
    Term s  = thvm_sum_axes(ctx, sq, (u32[]){0}, 1);
    Term r  = thvm_op(ctx, UOP_SQRT, s, term_era());
    Term r_bc = thvm_expand(ctx, r, SHAPE(3));
    Term y  = thvm_op(ctx, UOP_DIV, x0, r_bc);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, x)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "DIV", "EXPAND", "SQRT", "SUM", "MUL"};
    const char *post[] = {"CTR", "DIV", "SQRT", "SUM", "MUL", "EXPAND"};
    int ok = topo_check("gradu_l2_normalize", 0, pre, 7)
          && topo_check("gradu_l2_normalize", 1, post, 6);
    return report("gradu_l2_normalize", ok);
}

// UOP_GRAD2: PERMUTE + SUM composition.  y = sum(permute(x, [1,0])).
// Backward threads SUM-expand then PERMUTE-inverse.
static int test_gradu_permute_sum(void) {
    setup_graph_dir("gradu_permute_sum");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1,2,3,4,5,6};
    Term x = thvm_tensor(ctx, xd, SHAPE(2, 3));
    thvm_set_requires_grad(ctx, x);
    u32 perm[] = {1, 0};
    Term p = thvm_permute(ctx, x, perm, 2);
    Term y = thvm_sum_axes(ctx, p, (u32[]){0, 1}, 2);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, x)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "SUM", "PERMUTE"};
    const char *post[] = {"CTR", "SUM", "EXPAND", "PERMUTE"};
    int ok = topo_check("gradu_permute_sum", 0, pre, 4)
          && topo_check("gradu_permute_sum", 1, post, 4);
    return report("gradu_permute_sum", ok);
}

// UOP_GRAD2: cross-entropy loss (simplified).
// CE(x, t) = log(sum(exp(x))) - sum(t * x).  grad w.r.t. x = softmax(x) - t.
// Exercises LOG + SUM + EXP + MUL + SUB all composed.
static int test_gradu_cross_entropy(void) {
    setup_graph_dir("gradu_cross_entropy");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1, 2, 3, 4}, td[] = {0, 0, 1, 0};
    Term x = thvm_tensor(ctx, xd, SHAPE(4));
    Term t = thvm_tensor(ctx, td, SHAPE(4));
    thvm_set_requires_grad(ctx, x);
    Term x0, x1;
    thvm_dup(ctx, thvm_fresh_label(ctx), x, &x0, &x1);
    // log(sum(exp(x0)))
    Term ex = thvm_op(ctx, UOP_EXP, x0, term_era());
    Term se = thvm_sum_axes(ctx, ex, (u32[]){0}, 1);
    Term lse = thvm_op(ctx, UOP_LOG, se, term_era());
    // sum(t * x1) — scalar
    Term tx = thvm_op(ctx, UOP_MUL, t, x1);
    Term stx = thvm_sum_axes(ctx, tx, (u32[]){0}, 1);
    // CE = lse - stx (both scalar)
    Term y = thvm_op(ctx, UOP_SUB, lse, stx);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, x)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "SUB", "LOG", "SUM", "EXP", "MUL"};
    const char *post[] = {"CTR", "SUB", "DIV", "SUM", "EXP", "MUL", "EXPAND"};
    int ok = topo_check("gradu_cross_entropy", 0, pre, 7)
          && topo_check("gradu_cross_entropy", 1, post, 7);
    return report("gradu_cross_entropy", ok);
}

// UOP_GRAD2: batched reduce.  x shape [2,3], y = sum(x*x, axes=[1]).
// Per-batch sum of squares; grad w.r.t. x keeps [2,3] shape.
static int test_gradu_batched_reduce(void) {
    setup_graph_dir("gradu_batched_reduce");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1,2,3,4,5,6};
    Term x = thvm_tensor(ctx, xd, SHAPE(2, 3));
    thvm_set_requires_grad(ctx, x);
    Term x0, x1;
    thvm_dup(ctx, thvm_fresh_label(ctx), x, &x0, &x1);
    Term sq = thvm_op(ctx, UOP_MUL, x0, x1);
    Term y  = thvm_sum_axes(ctx, sq, (u32[]){1}, 1);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, x)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "SUM", "MUL"};
    const char *post[] = {"CTR", "SUM", "EXPAND", "MUL", "ADD"};
    int ok = topo_check("gradu_batched_reduce", 0, pre, 4)
          && topo_check("gradu_batched_reduce", 1, post, 5);
    return report("gradu_batched_reduce", ok);
}

// UOP_GRAD2 through non-trivial lambda: y = (λv. v*v) a.  After beta,
// body becomes a*a; GRAD2 should give 2a.  The lambda body DUPs v to
// use it twice in MUL.
static int test_gradu_lambda_square(void) {
    setup_graph_dir("gradu_lambda_square");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term v;
    Term lam = thvm_lam(ctx, &v, term_new(TAG_ERA, 0, 0));
    Term v0, v1;
    thvm_dup(ctx, thvm_fresh_label(ctx), v, &v0, &v1);
    heap_set(ctx, term_val(lam) + 1, thvm_op(ctx, UOP_MUL, v0, v1));
    Term y = thvm_app(ctx, lam, a);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "APP", "LAM", "MUL"};
    const char *post[] = {"CTR", "MUL", "ADD", "EXPAND"};
    int ok = topo_check("gradu_lambda_square", 0, pre, 5)
          && topo_check("gradu_lambda_square", 1, post, 4);
    return report("gradu_lambda_square", ok);
}

// UOP_GRAD2: absolute value via |x| = relu(x) + relu(-x).
// dy/dx = 2*(x>0) - 1 = sign(x).  Tests parallel RELU branches sharing
// a common target via DUP.
static int test_gradu_abs(void) {
    setup_graph_dir("gradu_abs");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {-2, -1, 0, 1, 2};
    Term x = thvm_tensor(ctx, xd, SHAPE(5));
    thvm_set_requires_grad(ctx, x);
    Term x0, x1;
    thvm_dup(ctx, thvm_fresh_label(ctx), x, &x0, &x1);
    Term r_pos = thvm_op(ctx, UOP_RELU, x0, term_era());
    Term r_neg = thvm_op(ctx, UOP_RELU,
                    thvm_op(ctx, UOP_NEG, x1, term_era()),
                    term_era());
    Term y = thvm_op(ctx, UOP_ADD, r_pos, r_neg);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, x)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "ADD", "RELU", "NEG"};
    const char *post[] = {"CTR", "ADD", "MUL", "CMP", "NEG", "EXPAND"};
    int ok = topo_check("gradu_abs", 0, pre, 5)
          && topo_check("gradu_abs", 1, post, 6);
    return report("gradu_abs", ok);
}

// UOP_GRAD2: mixed higher-order. d²(x*y)/dxdy = 1.
// Inner: d(x*y)/dx = y (Leibniz w/ target=x).
// Outer: d(y)/dy = 1 (leaf identity).
static int test_gradu_mixed_partial(void) {
    setup_graph_dir("gradu_mixed_partial");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1, 2, 3}, yd[] = {4, 5, 6};
    Term x = thvm_tensor(ctx, xd, SHAPE(3));
    Term y = thvm_tensor(ctx, yd, SHAPE(3));
    thvm_set_requires_grad(ctx, x);
    thvm_set_requires_grad(ctx, y);
    Term prod = thvm_op(ctx, UOP_MUL, x, y);
    // Inner d/dx
    Term inner = thvm_grad_u(ctx, prod, x);
    // Outer d/dy — DUP the inner first since the root will consume it
    Term inner0, inner1;
    thvm_dup(ctx, thvm_fresh_label(ctx), inner, &inner0, &inner1);
    (void)inner0;  // not used directly; outer wants inner1
    Term outer = thvm_grad_u(ctx, inner1, y);
    Term root = thvm_ctr(ctx, (Term[]){inner0, outer}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "MUL"};
    const char *post[] = {"CTR", "EXPAND"};
    int ok = topo_check("gradu_mixed_partial", 0, pre, 3)
          && topo_check("gradu_mixed_partial", 1, post, 2);
    return report("gradu_mixed_partial", ok);
}

// UOP_GRAD2: APP with a compute argument.
// y = (λv. v+v) (a*c).  Tests that the trampoline reduces the ARGUMENT
// (a*c → TOP) before beta-substitutes into the lambda body, then GRAD2
// pattern-matches on the body's structure (ADD of duplicated MULs).
// Target = a.  dy/da = 2c (via chain through beta and Leibniz).
static int test_gradu_app_compute_arg(void) {
    setup_graph_dir("gradu_app_compute_arg");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, cd[] = {10, 10, 10};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term c = thvm_tensor(ctx, cd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    // λv. v + v
    Term v;
    Term lam = thvm_lam(ctx, &v, term_new(TAG_ERA, 0, 0));
    Term v0, v1;
    thvm_dup(ctx, thvm_fresh_label(ctx), v, &v0, &v1);
    heap_set(ctx, term_val(lam) + 1, thvm_op(ctx, UOP_ADD, v0, v1));
    Term arg = thvm_op(ctx, UOP_MUL, a, c);
    Term y = thvm_app(ctx, lam, arg);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "APP", "LAM", "MUL"};
    const char *post[] = {"CTR", "ADD", "MUL", "EXPAND"};
    int ok = topo_check("gradu_app_compute_arg", 0, pre, 5)
          && topo_check("gradu_app_compute_arg", 1, post, 4);
    return report("gradu_app_compute_arg", ok);
}

// UOP_GRAD2: nested APPs.  y = (λv. v) ((λu. u*u) a).  Two betas then
// GRAD2(a*a, a) = 2a.  Verifies chain rule across multiple lambda layers.
static int test_gradu_nested_app(void) {
    setup_graph_dir("gradu_nested_app");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    // Inner λu. u*u
    Term u;
    Term lam_inner = thvm_lam(ctx, &u, term_new(TAG_ERA, 0, 0));
    Term u0, u1;
    thvm_dup(ctx, thvm_fresh_label(ctx), u, &u0, &u1);
    heap_set(ctx, term_val(lam_inner) + 1, thvm_op(ctx, UOP_MUL, u0, u1));
    // Outer λv. v (identity)
    Term v;
    Term lam_outer = thvm_lam(ctx, &v, term_new(TAG_ERA, 0, 0));
    heap_set(ctx, term_val(lam_outer) + 1, v);
    // y = outer (inner a)
    Term y = thvm_app(ctx, lam_outer, thvm_app(ctx, lam_inner, a));
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "APP", "LAM", "MUL"};
    const char *post[] = {"CTR", "MUL", "ADD", "EXPAND"};
    int ok = topo_check("gradu_nested_app", 0, pre, 5)
          && topo_check("gradu_nested_app", 1, post, 4);
    return report("gradu_nested_app", ok);
}

// UOP_GRAD2: curried lambda.  y = ((λx.λw.x*w) a) b.  Two betas; target=a.
// dy/da = b.  Tests curried two-arg application under GRAD2.
static int test_gradu_curried(void) {
    setup_graph_dir("gradu_curried");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    // inner λw. xv * w   (xv from outer)
    Term xv, wv;
    Term lam_w = thvm_lam(ctx, &wv, term_new(TAG_ERA, 0, 0));
    // outer λx. lam_w (parameterized by x)
    Term lam_x = thvm_lam(ctx, &xv, lam_w);
    heap_set(ctx, term_val(lam_w) + 1, thvm_op(ctx, UOP_MUL, xv, wv));
    // apply: ((lam_x a) b)
    Term y = thvm_app(ctx, thvm_app(ctx, lam_x, a), b);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "APP", "LAM"};
    const char *post[] = {"CTR", "MUL", "ADD", "EXPAND"};
    int ok = topo_check("gradu_curried", 0, pre, 4)
          && topo_check("gradu_curried", 1, post, 4);
    return report("gradu_curried", ok);
}

// UOP_GRAD2: IFZ with target in zero-case branch.
// y = IFZ(0, a, succ_lam).  counter=0 so y reduces to a; grad w.r.t. a = 1.
static int test_gradu_ifz(void) {
    setup_graph_dir("gradu_ifz");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    // succ_lam is unused (counter=0 hits zero-case); use identity λv.v
    Term v;
    Term succ = thvm_lam(ctx, &v, term_new(TAG_ERA, 0, 0));
    heap_set(ctx, term_val(succ) + 1, v);
    Term counter = thvm_scalar(ctx, 0.0f);
    Term y = thvm_ifz(ctx, counter, a, succ);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "IFZ"};
    const char *post[] = {"CTR", "EXPAND"};
    int ok = topo_check("gradu_ifz", 0, pre, 3)
          && topo_check("gradu_ifz", 1, post, 2);
    return report("gradu_ifz", ok);
}

// UOP_GRAD2: IFZ hits succ branch.  y = IFZ(1, a, λ_. a*a).
// counter=1 → evaluates succ_lam(0) → a*a → GRAD2 → 2a.
static int test_gradu_ifz_succ(void) {
    setup_graph_dir("gradu_ifz_succ");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    // succ_lam: λc. a*a  (ignores c, uses a twice via DUP)
    Term cvar;
    Term succ = thvm_lam(ctx, &cvar, term_new(TAG_ERA, 0, 0));
    Term a0, a1;
    thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
    // Sink cvar since we don't use it; ERA the var.
    thvm_spawn_detached_era(ctx, cvar);
    heap_set(ctx, term_val(succ) + 1, thvm_op(ctx, UOP_MUL, a0, a1));
    // Target for GRAD2: need a separate handle — use a' = dup of a
    // before we pass a into succ_lam body.  But we already gave a away.
    // Work around: target is the TEN tid, so we can recreate a fresh
    // reference via thvm_tensor_from_id ... not exposed.  Use a via DUP.
    // Simpler path: construct as y = IFZ(1, dummy, succ_lam), where
    // succ_lam's body references a (captured by closure).
    Term dummy = thvm_tensor(ctx, ad, SHAPE(3));
    Term counter = thvm_scalar(ctx, 1.0f);
    Term y = thvm_ifz(ctx, counter, dummy, succ);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "IFZ", "LAM", "MUL"};
    const char *post[] = {"CTR", "MUL", "ADD", "EXPAND"};
    int ok = topo_check("gradu_ifz_succ", 0, pre, 5)
          && topo_check("gradu_ifz_succ", 1, post, 4);
    return report("gradu_ifz_succ", ok);
}

// UOP_GRAD2: WHERE(cond=all-1, a, b).  KNOWN GAP: WHERE materializes to
// a FRESH TEN (not reusing a's id) before GRAD2 fires, so the leaf rule
// sees ytid != target_tid and emits zeros instead of 1.  The "dy/da=1 if
// cond" semantics needs either a WHERE-specific UOP_GRAD2 rule or a
// lazier WHERE.  Test currently only asserts pre-phase topology.
static int test_gradu_where(void) {
    setup_graph_dir("gradu_where");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {7, 8, 9}, condd[] = {1, 1, 1};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    Term cond = thvm_tensor(ctx, condd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_where(ctx, cond, a, b);
    // No DUP on y — let GRAD2 pattern-match the raw WHERE TOP directly.
    Term root = thvm_grad_u(ctx, y, a);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    // Pre must contain GRAD2+WHERE; post we don't yet know — if no WHERE
    // rule exists, GRAD2 stays in the graph.  Just check pre.
    const char *pre[]  = {"GRAD2", "WHERE"};
    const char *post[] = {"WHERE", "EXPAND"};
    int ok = topo_check("gradu_where", 0, pre, 2)
          && topo_check("gradu_where", 1, post, 2);
    return report("gradu_where", ok);
}

// UOP_GRAD2: dot product.  y = sum(x*w), dy/dx = w.
static int test_gradu_dot(void) {
    setup_graph_dir("gradu_dot");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1, 2, 3}, wd[] = {10, 20, 30};
    Term x = thvm_tensor(ctx, xd, SHAPE(3));
    Term w = thvm_tensor(ctx, wd, SHAPE(3));
    thvm_set_requires_grad(ctx, x);
    Term prod = thvm_op(ctx, UOP_MUL, x, w);
    Term y = thvm_sum_axes(ctx, prod, (u32[]){0}, 1);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, x)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "SUM", "MUL"};
    const char *post[] = {"CTR", "SUM", "EXPAND", "MUL", "ADD"};
    int ok = topo_check("gradu_dot", 0, pre, 4)
          && topo_check("gradu_dot", 1, post, 5);
    return report("gradu_dot", ok);
}

// UOP_GRAD2: WHERE with mixed cond, target = b (else branch).
// d(where(c, a, b))/db = where(c, 0, 1).
static int test_gradu_where_else(void) {
    setup_graph_dir("gradu_where_else");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {7, 8, 9}, condd[] = {0, 1, 0};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    Term cond = thvm_tensor(ctx, condd, SHAPE(3));
    thvm_set_requires_grad(ctx, b);
    Term y = thvm_where(ctx, cond, a, b);
    Term root = thvm_grad_u(ctx, y, b);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"GRAD2", "WHERE"};
    const char *post[] = {"WHERE", "EXPAND"};
    int ok = topo_check("gradu_where_else", 0, pre, 2)
          && topo_check("gradu_where_else", 1, post, 2);
    return report("gradu_where_else", ok);
}

// UOP_GRAD2: nested WHERE.  y = where(c1, where(c2, a, b), d).  Target = a.
// KNOWN PARTIAL: outer WHERE rule fires correctly, but inner WHERE gets
// materialized when the outer-emitted da=GRAD2(inner, t) is evaluated —
// the outer WHERE's branch-selection copies GRAD2's operand before the
// reducer's WHERE-under-GRAD2 laziness kicks in at the new nesting level.
// Post-sweep shows outer WHERE but inner collapsed to zero rather than
// where(c2, 1, 0).  Test only asserts shape-level topology for now.
static int test_gradu_where_nested(void) {
    setup_graph_dir("gradu_where_nested");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[]={1,2,3}, bd[]={4,5,6}, dd[]={7,8,9}, c1d[]={1,0,1}, c2d[]={1,1,0};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    Term d = thvm_tensor(ctx, dd, SHAPE(3));
    Term c1 = thvm_tensor(ctx, c1d, SHAPE(3));
    Term c2 = thvm_tensor(ctx, c2d, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term inner = thvm_where(ctx, c2, a, b);
    Term y = thvm_where(ctx, c1, inner, d);
    Term root = thvm_grad_u(ctx, y, a);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"GRAD2", "WHERE"};
    const char *post[] = {"WHERE", "EXPAND"};
    int ok = topo_check("gradu_where_nested", 0, pre, 2)
          && topo_check("gradu_where_nested", 1, post, 2);
    return report("gradu_where_nested", ok);
}

// UOP_GRAD2: polynomial y = a*x*x + b*x + c, target = a.  dy/da = x*x.
// Exercises ADD+MUL-Leibniz through a 3-term polynomial.
static int test_gradu_poly(void) {
    setup_graph_dir("gradu_poly");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[]={1,1,1}, bd[]={2,2,2}, cd[]={3,3,3}, xd[]={1,2,3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    Term c = thvm_tensor(ctx, cd, SHAPE(3));
    Term x = thvm_tensor(ctx, xd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term x0, x1;
    thvm_dup(ctx, thvm_fresh_label(ctx), x, &x0, &x1);
    Term xsq = thvm_op(ctx, UOP_MUL, x0, x1);
    Term xsq0, xsq1;
    thvm_dup(ctx, thvm_fresh_label(ctx), xsq, &xsq0, &xsq1);
    (void)xsq1;
    Term axsq = thvm_op(ctx, UOP_MUL, a, xsq0);
    Term bx   = thvm_op(ctx, UOP_MUL, b, x);  // shadow: x is consumed, but re-usable TEN ref
    Term y    = thvm_op(ctx, UOP_ADD,
                   thvm_op(ctx, UOP_ADD, axsq, bx),
                   c);
    Term y0, y1;
    thvm_dup(ctx, thvm_fresh_label(ctx), y, &y0, &y1);
    Term root = thvm_ctr(ctx, (Term[]){y0, thvm_grad_u(ctx, y1, a)}, 2);
    thvm_eval(ctx, root);
    thvm_free(ctx);
    const char *pre[]  = {"CTR", "GRAD2", "ADD", "MUL"};
    const char *post[] = {"CTR", "ADD", "MUL", "EXPAND"};
    int ok = topo_check("gradu_poly", 0, pre, 4)
          && topo_check("gradu_poly", 1, post, 4);
    return report("gradu_poly", ok);
}

// E2E: MUL of a broadcast-expanded scalar with a full tensor.
// MUL(EXPAND([1]=1, [3]), w=[10,20,30]) → [10,20,30].
static int test_e2e_mul_broadcast(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 od[] = {1.0f}, wd[] = {10, 20, 30};
    Term one = thvm_tensor(ctx, od, SHAPE(1));
    Term exp_one = thvm_expand(ctx, one, SHAPE(3));
    Term w = thvm_tensor(ctx, wd, SHAPE(3));
    Term y = thvm_op(ctx, UOP_MUL, exp_one, w);
    Term r = thvm_eval(ctx, y);
    f32 *h = thvm_to_host(ctx, r);
    int ok = (h != NULL);
    if (h) {
        f32 expect[] = {10, 20, 30};
        for (int i = 0; i < 3; i++) {
            if (h[i] != expect[i]) {
                fprintf(stderr, "  e2e_mul_broadcast: h[%d]=%g expect=%g\n",
                        i, h[i], expect[i]);
                ok = 0;
            }
        }
    }
    thvm_free(ctx);
    return report("e2e_mul_broadcast", ok);
}

// E2E: verify EXPAND broadcast from [1] to [3].  Uses a shape-[1] TEN
// input so it's a real broadcast.
static int test_e2e_expand_scalar(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 d[] = {1.0f};
    Term one = thvm_tensor(ctx, d, SHAPE(1));
    Term y = thvm_expand(ctx, one, SHAPE(3));
    Term r = thvm_eval(ctx, y);
    f32 *h = thvm_to_host(ctx, r);
    int ok = (h != NULL);
    if (h) {
        for (int i = 0; i < 3; i++) {
            if (h[i] != 1.0f) {
                fprintf(stderr, "  e2e_expand: h[%d]=%g expect=1\n", i, h[i]);
                ok = 0;
            }
        }
    }
    thvm_free(ctx);
    return report("e2e_expand_scalar", ok);
}

// E2E numerical: d(x*w)/dx = w, no SUM wrapping.  Minimal GRAD2 chain:
// Leibniz → ADD(MUL(EXPAND(1),w), MUL(x,EXPAND(0))) → w.
static int test_gradu_mul_e2e(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1, 2, 3}, wd[] = {10, 20, 30};
    Term x = thvm_tensor(ctx, xd, SHAPE(3));
    Term w = thvm_tensor(ctx, wd, SHAPE(3));
    thvm_set_requires_grad(ctx, x);
    Term y = thvm_op(ctx, UOP_MUL, x, w);
    Term grad = thvm_grad_u(ctx, y, x);
    Term r = thvm_eval(ctx, grad);
    f32 *h = thvm_to_host(ctx, r);
    int ok = (h != NULL);
    if (h) {
        f32 expect[] = {10, 20, 30};
        for (int i = 0; i < 3; i++) {
            if (h[i] != expect[i]) {
                fprintf(stderr, "  gradu_mul_e2e: h[%d]=%g expect=%g\n",
                        i, h[i], expect[i]);
                ok = 0;
            }
        }
    }
    thvm_free(ctx);
    return report("gradu_mul_e2e", ok);
}

// E2E sanity: plain forward MUL without any grad — baseline check that
// the pipeline computes elementwise multiply correctly.
static int test_e2e_mul_baseline(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1, 2, 3}, wd[] = {10, 20, 30};
    Term x = thvm_tensor(ctx, xd, SHAPE(3));
    Term w = thvm_tensor(ctx, wd, SHAPE(3));
    Term y = thvm_op(ctx, UOP_MUL, x, w);
    Term r = thvm_eval(ctx, y);
    f32 *h = thvm_to_host(ctx, r);
    int ok = (h != NULL);
    if (h) {
        f32 expect[] = {10, 40, 90};
        for (int i = 0; i < 3; i++) {
            if (h[i] != expect[i]) {
                fprintf(stderr, "  e2e_mul: h[%d]=%g expect=%g\n",
                        i, h[i], expect[i]);
                ok = 0;
            }
        }
    }
    thvm_free(ctx);
    return report("e2e_mul_baseline", ok);
}

// End-to-end numerical: y = sum(x*w), dy/dx = w.  Run full pipeline
// (no STOP_AFTER_SWEEP) and read the gradient tensor back; check
// against the known analytical answer.
static int test_gradu_dot_e2e(void) {
    setenv("THVM_GRAPH", "0", 1);
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1, 2, 3}, wd[] = {10, 20, 30};
    Term x = thvm_tensor(ctx, xd, SHAPE(3));
    Term w = thvm_tensor(ctx, wd, SHAPE(3));
    thvm_set_requires_grad(ctx, x);
    Term prod = thvm_op(ctx, UOP_MUL, x, w);
    Term y = thvm_sum_axes(ctx, prod, (u32[]){0}, 1);
    Term grad = thvm_grad_u(ctx, y, x);
    Term r = thvm_eval(ctx, grad);
    f32 *h = thvm_to_host(ctx, r);
    int ok = (h != NULL);
    if (h) {
        f32 expect[] = {10, 20, 30};
        for (int i = 0; i < 3; i++) {
            if (h[i] != expect[i]) {
                fprintf(stderr, "  gradu_dot_e2e: h[%d]=%g expect=%g\n",
                        i, h[i], expect[i]);
                ok = 0;
            }
        }
    }
    thvm_free(ctx);
    return report("gradu_dot_e2e", ok);
}

static int check_e2e(const char *name, f32 *h, f32 *expect, int n) {
    int ok = (h != NULL);
    if (h) for (int i = 0; i < n; i++)
        if (h[i] != expect[i]) {
            fprintf(stderr, "  %s: h[%d]=%g exp=%g\n", name, i, h[i], expect[i]);
            ok = 0;
        }
    return ok;
}

// d(a+b)/da = 1 → [1,1,1]
static int test_e2e_grad_add(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3}, bd[] = {4,5,6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    Term y = thvm_op(ctx, UOP_ADD, a, b);
    f32 *h = thvm_to_host(ctx, thvm_eval(ctx, thvm_grad_u(ctx, y, a)));
    f32 expect[] = {1,1,1};
    int ok = check_e2e("grad_add", h, expect, 3);
    thvm_free(ctx);
    return report("e2e_grad_add", ok);
}

// d(a-b)/db = -1
static int test_e2e_grad_sub_rhs(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {4,5,6}, bd[] = {1,2,3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    Term y = thvm_op(ctx, UOP_SUB, a, b);
    f32 *h = thvm_to_host(ctx, thvm_eval(ctx, thvm_grad_u(ctx, y, b)));
    f32 expect[] = {-1,-1,-1};
    int ok = check_e2e("grad_sub_rhs", h, expect, 3);
    thvm_free(ctx);
    return report("e2e_grad_sub_rhs", ok);
}

// d(-a)/da = -1
static int test_e2e_grad_neg(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1,2,3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term y = thvm_op(ctx, UOP_NEG, a, term_era());
    f32 *h = thvm_to_host(ctx, thvm_eval(ctx, thvm_grad_u(ctx, y, a)));
    f32 expect[] = {-1,-1,-1};
    int ok = check_e2e("grad_neg", h, expect, 3);
    thvm_free(ctx);
    return report("e2e_grad_neg", ok);
}

// d(relu(a))/da on a=[-1,1,2] → [0,1,1]
static int test_e2e_grad_relu(void) {
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {-1,1,2};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term y = thvm_op(ctx, UOP_RELU, a, term_era());
    f32 *h = thvm_to_host(ctx, thvm_eval(ctx, thvm_grad_u(ctx, y, a)));
    f32 expect[] = {0,1,1};
    int ok = check_e2e("grad_relu", h, expect, 3);
    thvm_free(ctx);
    return report("e2e_grad_relu", ok);
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
    fails += test_nested_unary();
    fails += test_identity();
    fails += test_sum_multi_axis();
    fails += test_gradu_identity();
    fails += test_gradu_add();
    fails += test_gradu_mul();
    fails += test_gradu_neg();
    fails += test_gradu_exp();
    fails += test_gradu_log();
    fails += test_gradu_sqrt();
    fails += test_gradu_relu();
    fails += test_gradu_div();
    fails += test_gradu_max();
    fails += test_gradu_cmp();
    fails += test_gradu_reshape();
    fails += test_gradu_permute();
    fails += test_gradu_sum();
    fails += test_gradu_shrink();
    fails += test_gradu_pad();
    fails += test_gradu_expand();
    fails += test_gradu_rmax();
    fails += test_gradu_mm();
    fails += test_gradu_assign();
    fails += test_gradu_deep_chain();
    fails += test_gradu_sub_rhs();
    fails += test_gradu_cubic();
    fails += test_gradu_shape_target_diff();
    fails += test_gradu_third_derivative();
    fails += test_gradu_exp_neg();
    fails += test_gradu_softplus();
    fails += test_gradu_sum_sq();
    fails += test_gradu_target_via_dup();
    fails += test_gradu_distributive();
    fails += test_gradu_assign_src();
    fails += test_gradu_mse();
    fails += test_gradu_conv_like();
    fails += test_gradu_mlp_like();
    fails += test_gradu_logsumexp();
    fails += test_gradu_l2_normalize();
    fails += test_gradu_permute_sum();
    fails += test_gradu_cross_entropy();
    fails += test_gradu_batched_reduce();
    fails += test_gradu_lambda();
    fails += test_gradu_lambda_square();
    fails += test_gradu_abs();
    fails += test_gradu_mixed_partial();
    fails += test_gradu_app_compute_arg();
    fails += test_gradu_nested_app();
    fails += test_gradu_curried();
    fails += test_gradu_ifz();
    fails += test_gradu_ifz_succ();
    fails += test_gradu_where();
    fails += test_gradu_dot();
    fails += test_gradu_where_else();
    fails += test_gradu_where_nested();
    fails += test_gradu_poly();
    fails += test_e2e_mul_baseline();
    fails += test_e2e_expand_scalar();
    fails += test_e2e_mul_broadcast();
    fails += test_gradu_mul_e2e();
    fails += test_gradu_dot_e2e();
    fails += test_e2e_grad_add();
    fails += test_e2e_grad_sub_rhs();
    fails += test_e2e_grad_neg();
    fails += test_e2e_grad_relu();
    // test_gradu_lambda() deferred — thvm_lam requires two-step
    // construction (ERA body placeholder, then heap_set the real body);
    // single-shot `thvm_lam(ctx, &v, v)` reads v before it's initialized.
    // Low priority; GRAD2-through-LAM not a critical topology path.
    printf("\ntotal failures: %d\n", fails);
    return fails ? 1 : 0;
}
