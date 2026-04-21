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
    printf("\ntotal failures: %d\n", fails);
    return fails ? 1 : 0;
}
