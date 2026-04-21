// test_grad_rules.m — GRAD/TOP topology tests (Phase 1+2 only).
//
// Each sub-test:
//   1. Builds a small forward expression y = op(inputs...).
//   2. Constructs a raw GRAD node:  y on the principal port, a free
//      port (term_era()) on the gy port, target = x.  No SUM wrap,
//      no thvm_grad_keep bundle machinery. Jacobian-style seed.
//   3. Enables THVM_GRAPH + THVM_STEP_GRAPH, evals the GRAD.
//   4. Parses wl/examples/thvm_graphs/grad_rules/<rule>/thvm_1_post_reduce.dot
//      — the Phase-1-reduce output — and asserts the expected
//      chain-rule topology tokens appear.
//
// We do NOT check numeric values. We do NOT check Phase-3+ output.
// The scheduler / codegen / renderer are out of scope here; the
// question is simply: does each GRAD/TOP rewrite rule emit the
// mathematically correct backward structure?

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>

#define GRAPH_ROOT "wl/examples/thvm_graphs/grad_rules"

static void setup_graph_dir(const char *rule) {
    char dir[256];
    snprintf(dir, sizeof(dir), "%s/%s", GRAPH_ROOT, rule);
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "rm -rf %s && mkdir -p %s", dir, dir);
    int r = system(cmd); (void)r;
    // Phase dumps (thvm_0_pre_reduce..thvm_2_post_sweep) via THVM_GRAPH.
    // STOP_AFTER_SWEEP skips fuse / passes / exec — we only care about
    // Phase 1 + sweep topology, not codegen. Per-step dumps (THVM_STEP_GRAPH)
    // would run the full pipeline internally and choke on NUM-seeded ops,
    // so they're disabled here.
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

// Topology assertion: every needle must be present in phase_N of rule's dir.
// 0 = pre-reduce, 1 = post-reduce, 2 = post-sweep (GC), 3 = post-fuse.
static int topo_check(const char *rule, int phase,
                      const char **needles, int n_needles,
                      const char **forbid, int n_forbid) {
    char path[512];
    const char *name[] = {
        "thvm_0_pre_reduce.dot",
        "thvm_1_post_reduce.dot",
        "thvm_2_post_sweep.dot",
        "thvm_3_post_fuse.dot",
    };
    snprintf(path, sizeof(path), "%s/%s/%s", GRAPH_ROOT, rule, name[phase]);
    char *buf = slurp(path);
    if (!buf) {
        printf("  TOPO miss: %s not written\n", path);
        return 0;
    }
    int ok = 1;
    for (int i = 0; i < n_needles; i++) {
        if (!strstr(buf, needles[i])) {
            printf("  TOPO miss in %s phase%d: expected \"%s\"\n",
                   rule, phase, needles[i]);
            ok = 0;
        }
    }
    for (int i = 0; i < n_forbid; i++) {
        if (strstr(buf, forbid[i])) {
            printf("  TOPO forbid in %s phase%d: should not have \"%s\"\n",
                   rule, phase, forbid[i]);
            ok = 0;
        }
    }
    free(buf);
    return ok;
}

// Count occurrences of a substring in a rule's phase file.
static int topo_count(const char *rule, int phase, const char *needle) {
    char path[512];
    const char *name[] = {
        "thvm_0_pre_reduce.dot",
        "thvm_1_post_reduce.dot",
        "thvm_2_post_sweep.dot",
        "thvm_3_post_fuse.dot",
    };
    snprintf(path, sizeof(path), "%s/%s/%s", GRAPH_ROOT, rule, name[phase]);
    char *buf = slurp(path);
    if (!buf) return -1;
    int count = 0;
    size_t nlen = strlen(needle);
    for (char *p = buf; (p = strstr(p, needle)) != NULL; p += nlen) count++;
    free(buf);
    return count;
}

// Build a raw GRAD node and wrap it in a CTR so nothing is orphaned.
//
//     CTR#2 { slot0 = GRAD(y, gy=free), slot1 = free }
//
// slot0 holds the chain-rule result once the GRAD fires.
// slot1 is a dangling free-port consumer so the gy output (which the
// GRAD rule emits an ERA on for the non-target arm) has a visible
// external sink — keeps the ERA ⊳ ⋯ interaction visible in per-step
// graphs instead of dropping it as "orphan root".
static Term mk_grad(TinyHVM *ctx, Term y, Term x) {
    thvm_grad_targets_clear(ctx);
    term_use_clear();
    u64 loc = heap_alloc(ctx, 2);
    y = linear_use(ctx, y, loc);
    heap_set(ctx, loc + 0, y);
    heap_set(ctx, loc + 1, term_era());
    thvm_grad_target_set(ctx, loc, x);
    thvm_grad_mode_set(ctx, loc, GRAD_MODE_DROP);
    Term grad_top = term_new(TAG_TOP, UOP_GRAD, loc);
    return thvm_ctr(ctx, (Term[]){grad_top, term_era()}, 2);
}

static int report(const char *rule, int ok) {
    printf("%-10s %s\n", rule, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ────────────────────────────────────────────────────────────────────
// Per-rule topology tests.
//
// "Phase 0 pre-reduce" is the initial graph we built: GRAD + the
// forward uop + inputs, with gy on a free port.
//
// "Phase 1 post-reduce" is after the GRAD rule fires and the chain
// rule unfolds: the forward uop is consumed, the sub-GRADs have been
// emitted onto the backward branches, and eventually the target-leaf
// GRAD⊳TEN matcher has resolved.
// ────────────────────────────────────────────────────────────────────

static int test_add(void) {
    setup_graph_dir("add");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_ADD, a, b);
    thvm_eval(ctx, mk_grad(ctx, y, a));
    thvm_free(ctx);
    // Pre-reduce: the forward ADD and the GRAD with free-port gy must exist.
    const char *pre_need[]    = {"GRAD\\nd/d(t1)", "ADD", "free"};
    const char *post_forbid[] = {"UOP_ADD\\n", "\"ADD\\n"};  // forward ADD consumed
    int ok = topo_check("add", 0, pre_need, 3, NULL, 0)
          && topo_check("add", 1, NULL, 0, post_forbid, 2);
    return report("ADD", ok);
}

static int test_sub(void) {
    setup_graph_dir("sub");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_SUB, a, b);
    thvm_eval(ctx, mk_grad(ctx, y, a));
    thvm_free(ctx);
    const char *pre_need[] = {"GRAD\\nd/d(t1)", "SUB", "free"};
    int ok = topo_check("sub", 0, pre_need, 3, NULL, 0);
    return report("SUB", ok);
}

static int test_mul(void) {
    setup_graph_dir("mul");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_MUL, a, b);
    thvm_eval(ctx, mk_grad(ctx, y, a));
    thvm_free(ctx);
    const char *pre_need[]  = {"GRAD\\nd/d(t1)", "MUL", "free"};
    // Backward of MUL for single-target a: deposit is MUL(gy_broadcast, b).
    // Phase-1 post should still contain a MUL (the backward product) and
    // reference b (t2). The b-arm sub-GRAD returns 0 and collapses.
    const char *post_need[] = {"\"MUL\\n", "t2"};
    int ok = topo_check("mul", 0, pre_need, 3, NULL, 0)
          && topo_check("mul", 1, post_need, 2, NULL, 0);
    return report("MUL", ok);
}

static int test_div(void) {
    setup_graph_dir("div");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {2, 4, 6}, bd[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_DIV, a, b);
    thvm_eval(ctx, mk_grad(ctx, y, a));
    thvm_free(ctx);
    const char *pre_need[] = {"GRAD\\nd/d(t1)", "DIV", "free"};
    int ok = topo_check("div", 0, pre_need, 3, NULL, 0);
    return report("DIV", ok);
}

static int test_max(void) {
    setup_graph_dir("max");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 5, 3}, bd[] = {4, 2, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_MAX, a, b);
    thvm_eval(ctx, mk_grad(ctx, y, a));
    thvm_free(ctx);
    const char *pre_need[] = {"GRAD\\nd/d(t1)", "MAX", "free"};
    // MAX backward uses CMP mask
    const char *post_need[] = {"CMP"};
    int ok = topo_check("max", 0, pre_need, 3, NULL, 0)
          && topo_check("max", 1, post_need, 1, NULL, 0);
    return report("MAX", ok);
}

static int test_neg(void) {
    setup_graph_dir("neg");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_NEG, a, term_era());
    thvm_eval(ctx, mk_grad(ctx, y, a));
    thvm_free(ctx);
    const char *pre_need[]  = {"GRAD\\nd/d(t1)", "NEG", "free"};
    // Backward of NEG emits another NEG
    const char *post_need[] = {"NEG"};
    int ok = topo_check("neg", 0, pre_need, 3, NULL, 0)
          && topo_check("neg", 1, post_need, 1, NULL, 0);
    return report("NEG", ok);
}

static int test_exp(void) {
    setup_graph_dir("exp");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {0, 1, 2};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_EXP, a, term_era());
    thvm_eval(ctx, mk_grad(ctx, y, a));
    thvm_free(ctx);
    const char *pre_need[] = {"GRAD\\nd/d(t1)", "EXP", "free"};
    int ok = topo_check("exp", 0, pre_need, 3, NULL, 0);
    return report("EXP", ok);
}

static int test_log(void) {
    setup_graph_dir("log");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 4};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_LOG, a, term_era());
    thvm_eval(ctx, mk_grad(ctx, y, a));
    thvm_free(ctx);
    const char *pre_need[] = {"GRAD\\nd/d(t1)", "LOG", "free"};
    // Backward is DIV(gy, a)
    const char *post_need[] = {"DIV"};
    int ok = topo_check("log", 0, pre_need, 3, NULL, 0)
          && topo_check("log", 1, post_need, 1, NULL, 0);
    return report("LOG", ok);
}

static int test_sqrt(void) {
    setup_graph_dir("sqrt");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 4, 9};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_SQRT, a, term_era());
    thvm_eval(ctx, mk_grad(ctx, y, a));
    thvm_free(ctx);
    const char *pre_need[]  = {"GRAD\\nd/d(t1)", "SQRT", "free"};
    const char *post_need[] = {"DIV", "MUL"};
    int ok = topo_check("sqrt", 0, pre_need, 3, NULL, 0)
          && topo_check("sqrt", 1, post_need, 2, NULL, 0);
    return report("SQRT", ok);
}

static int test_relu(void) {
    setup_graph_dir("relu");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {-1, 0, 1, 2};
    Term a = thvm_tensor(ctx, ad, SHAPE(4));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_RELU, a, term_era());
    thvm_eval(ctx, mk_grad(ctx, y, a));
    thvm_free(ctx);
    const char *pre_need[]  = {"GRAD\\nd/d(t1)", "RELU", "free"};
    const char *post_need[] = {"CMP", "MUL"};  // RELU backward: gy * (a > 0)
    int ok = topo_check("relu", 0, pre_need, 3, NULL, 0)
          && topo_check("relu", 1, post_need, 2, NULL, 0);
    return report("RELU", ok);
}

static int test_sum(void) {
    setup_graph_dir("sum");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3, 4};
    Term a = thvm_tensor(ctx, ad, SHAPE(4));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_sum_axes(ctx, a, (u32[]){0}, 1);
    thvm_eval(ctx, mk_grad(ctx, y, a));
    thvm_free(ctx);
    const char *pre_need[] = {"GRAD\\nd/d(t1)", "SUM", "free"};
    const char *post_need[] = {"EXPAND"};  // SUM backward = expand gy to a.shape
    int ok = topo_check("sum", 0, pre_need, 3, NULL, 0)
          && topo_check("sum", 1, post_need, 1, NULL, 0);
    return report("SUM", ok);
}

static int test_reshape(void) {
    setup_graph_dir("reshape");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3, 4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(6));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_reshape(ctx, a, SHAPE(2, 3));
    thvm_eval(ctx, mk_grad(ctx, y, a));
    thvm_free(ctx);
    const char *pre_need[] = {"GRAD\\nd/d(t1)", "RESHAPE", "free"};
    const char *post_need[] = {"RESHAPE"};  // backward = reshape gy back
    int ok = topo_check("reshape", 0, pre_need, 3, NULL, 0)
          && topo_check("reshape", 1, post_need, 1, NULL, 0);
    return report("RESHAPE", ok);
}

static int test_expand(void) {
    setup_graph_dir("expand");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_expand(ctx, thvm_reshape(ctx, a, SHAPE(1, 3)), SHAPE(4, 3));
    thvm_eval(ctx, mk_grad(ctx, y, a));
    thvm_free(ctx);
    const char *pre_need[] = {"GRAD\\nd/d(t1)", "EXPAND", "free"};
    const char *post_need[] = {"SUM"};  // backward = sum over expanded axes
    int ok = topo_check("expand", 0, pre_need, 3, NULL, 0)
          && topo_check("expand", 1, post_need, 1, NULL, 0);
    return report("EXPAND", ok);
}

static int test_cmp(void) {
    setup_graph_dir("cmp");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {2, 2, 2};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_CMP, a, b);
    thvm_eval(ctx, mk_grad(ctx, y, a));
    thvm_free(ctx);
    const char *pre_need[] = {"GRAD\\nd/d(t1)", "CMP", "free"};
    int ok = topo_check("cmp", 0, pre_need, 3, NULL, 0);
    return report("CMP", ok);
}

int main(void) {
    int fails = 0;
    fails += test_add();
    fails += test_sub();
    fails += test_mul();
    fails += test_div();
    fails += test_max();
    fails += test_neg();
    fails += test_exp();
    fails += test_log();
    fails += test_sqrt();
    fails += test_relu();
    fails += test_sum();
    fails += test_reshape();
    fails += test_expand();
    fails += test_cmp();
    printf("\ntotal failures: %d\n", fails);
    return fails ? 1 : 0;
}
