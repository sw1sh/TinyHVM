// test_grad_rules.m — per-rule GRAD/TOP correctness + step-graph dumps.
//
// Each rule exercised in isolation:
//   1. Build small forward with requires_grad leaves.
//   2. Wrap in sum() to get a scalar loss (so the seed is 1.0).
//   3. Set THVM_STEP_GRAPH + THVM_STEP_GRAPH_DIR = wl/examples/thvm_graphs/
//      grad_rules/<rule>/ before eval so the before→after step graphs are
//      emitted per-rule.
//   4. thvm_grad_keep / thvm_grad_multi_keep, eval, read bundle.
//   5. Assert per-element against the analytical gradient.
//   6. Assert step-graph topology: at least one .dot per rule, and the
//      before-state contains the expected GRAD + TOP uop tokens.
//
// When a rule is broken, the relevant sub-test names the owner.

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

#define TOL 1e-4f
#define GRAPH_ROOT "wl/examples/thvm_graphs/grad_rules"

static void setup_graph_dir(const char *rule) {
    char dir[256];
    snprintf(dir, sizeof(dir), "%s/%s", GRAPH_ROOT, rule);
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "rm -rf %s && mkdir -p %s", dir, dir);
    int r = system(cmd); (void)r;
    setenv("THVM_STEP_GRAPH", "1", 1);
    setenv("THVM_STEP_GRAPH_DIR", dir, 1);
}

// Count step_*.dot files written by the dumper.
static int count_step_dots(const char *rule) {
    char dir[256];
    snprintf(dir, sizeof(dir), "%s/%s", GRAPH_ROOT, rule);
    DIR *d = opendir(dir);
    if (!d) return 0;
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (strncmp(e->d_name, "step_", 5) == 0 &&
            strstr(e->d_name, ".dot")) n++;
    }
    closedir(d);
    return n;
}

// Assert a substring appears in the first (before) .dot file for a rule.
// This is how we cover graph topology cheaply: check that the initial
// state before the GRAD redex fires has the structure we expect.
static int assert_in_step0(const char *rule, const char **needles, int n_needles) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s/step_000_*.dot", GRAPH_ROOT, rule);
    // Resolve glob via shell.
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "ls %s/%s/step_000_*.dot 2>/dev/null | head -1",
             GRAPH_ROOT, rule);
    FILE *p = popen(cmd, "r");
    if (!p) return 0;
    if (!fgets(path, sizeof(path), p)) { pclose(p); return 0; }
    pclose(p);
    path[strcspn(path, "\n")] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    if (!buf) { fclose(f); return 0; }
    fread(buf, 1, sz, f);
    buf[sz] = 0;
    fclose(f);
    int ok = 1;
    for (int i = 0; i < n_needles; i++) {
        if (!strstr(buf, needles[i])) { ok = 0; break; }
    }
    free(buf);
    return ok;
}

static int almost_eq(const f32 *got, const f32 *exp, u32 n, f32 tol) {
    for (u32 i = 0; i < n; i++) {
        f32 d = got[i] - exp[i];
        if (d < 0) d = -d;
        if (d > tol) return 0;
    }
    return 1;
}

static int report(const char *name, const f32 *got, const f32 *exp, u32 n) {
    int ok = almost_eq(got, exp, n, TOL);
    printf("%-12s %s", name, ok ? "PASS" : "FAIL");
    if (!ok) {
        printf("\n  got: ");
        for (u32 i = 0; i < n; i++) printf("%.4f ", got[i]);
        printf("\n  exp: ");
        for (u32 i = 0; i < n; i++) printf("%.4f ", exp[i]);
    }
    printf("\n");
    return ok ? 0 : 1;
}

static int report_topo(const char *rule, int n_steps, int topo_ok) {
    int ok = n_steps > 0 && topo_ok;
    printf("%-12s topology %s (%d step dots)\n",
           rule, ok ? "PASS" : "FAIL", n_steps);
    return ok ? 0 : 1;
}

// Helper: run grad_keep on a single-param loss, copy bundle slot 0.
static int read_grad_1(TinyHVM *ctx, Term loss, Term p, f32 *out, u32 n) {
    Term bundle = thvm_eval(ctx, thvm_grad_keep(ctx, loss, p));
    if (thvm_grad_bundle_count(ctx, bundle) != 1) return -1;
    Term g = thvm_grad_bundle_get(ctx, bundle, 0);
    f32 *d = thvm_to_host(ctx, g);
    if (!d) return -1;
    memcpy(out, d, n * sizeof(f32));
    return 0;
}

static Term sum_all(TinyHVM *ctx, Term y, u32 rank) {
    u32 axes[8]; for (u32 i = 0; i < rank; i++) axes[i] = i;
    return thvm_sum_axes(ctx, y, axes, rank);
}

// ─── binary elementwise ─────────────────────────────────────────────

static int test_add(void) {
    setup_graph_dir("add");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_ADD, a, b);
    Term loss = sum_all(ctx, y, 1);
    f32 got[3];
    int r = read_grad_1(ctx, loss, a, got, 3);
    int fails = 0;
    if (r < 0) { printf("ADD read FAIL\n"); fails++; }
    else {
        f32 exp[3] = {1, 1, 1};
        fails += report("ADD", got, exp, 3);
    }
    const char *needles[] = {"GRAD", "SUM", "ADD"};
    fails += report_topo("ADD", count_step_dots("add"),
                         assert_in_step0("add", needles, 3));
    thvm_free(ctx);
    return fails;
}

static int test_sub(void) {
    setup_graph_dir("sub");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);
    Term y = thvm_op(ctx, UOP_SUB, a, b);
    Term loss = sum_all(ctx, y, 1);
    Term params[] = {a, b};
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));
    int fails = 0;
    if (thvm_grad_bundle_count(ctx, bundle) != 2) {
        printf("SUB count FAIL\n"); fails++;
    } else {
        f32 *ga = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 0));
        f32 *gb = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 1));
        f32 ea[3] = {1,1,1}, eb[3] = {-1,-1,-1};
        fails += report("SUB.a", ga, ea, 3);
        fails += report("SUB.b", gb, eb, 3);
    }
    const char *needles[] = {"GRAD", "SUM", "SUB"};
    fails += report_topo("SUB", count_step_dots("sub"),
                         assert_in_step0("sub", needles, 3));
    thvm_free(ctx);
    return fails;
}

static int test_mul(void) {
    setup_graph_dir("mul");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);
    Term y = thvm_op(ctx, UOP_MUL, a, b);
    Term loss = sum_all(ctx, y, 1);
    Term params[] = {a, b};
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));
    int fails = 0;
    f32 *ga = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 0));
    f32 *gb = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 1));
    fails += report("MUL.a", ga, bd, 3);
    fails += report("MUL.b", gb, ad, 3);
    const char *needles[] = {"GRAD", "SUM", "MUL"};
    fails += report_topo("MUL", count_step_dots("mul"),
                         assert_in_step0("mul", needles, 3));
    thvm_free(ctx);
    return fails;
}

static int test_div(void) {
    setup_graph_dir("div");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {2, 4, 6}, bd[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);
    Term y = thvm_op(ctx, UOP_DIV, a, b);
    Term loss = sum_all(ctx, y, 1);
    Term params[] = {a, b};
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));
    int fails = 0;
    f32 *ga = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 0));
    f32 *gb = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 1));
    f32 ea[3], eb[3];
    for (int i = 0; i < 3; i++) { ea[i] = 1.0f/bd[i]; eb[i] = -ad[i]/(bd[i]*bd[i]); }
    fails += report("DIV.a", ga, ea, 3);
    fails += report("DIV.b", gb, eb, 3);
    const char *needles[] = {"GRAD", "SUM", "DIV"};
    fails += report_topo("DIV", count_step_dots("div"),
                         assert_in_step0("div", needles, 3));
    thvm_free(ctx);
    return fails;
}

static int test_max(void) {
    setup_graph_dir("max");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 5, 3}, bd[] = {4, 2, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    thvm_set_requires_grad(ctx, b);
    Term y = thvm_op(ctx, UOP_MAX, a, b);
    Term loss = sum_all(ctx, y, 1);
    Term params[] = {a, b};
    Term bundle = thvm_eval(ctx, thvm_grad_multi_keep(ctx, loss, params, 2));
    int fails = 0;
    f32 *ga = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 0));
    f32 *gb = thvm_to_host(ctx, thvm_grad_bundle_get(ctx, bundle, 1));
    f32 ea[3] = {0, 1, 0}, eb[3] = {1, 0, 1};
    fails += report("MAX.a", ga, ea, 3);
    fails += report("MAX.b", gb, eb, 3);
    const char *needles[] = {"GRAD", "SUM", "MAX"};
    fails += report_topo("MAX", count_step_dots("max"),
                         assert_in_step0("max", needles, 3));
    thvm_free(ctx);
    return fails;
}

// ─── unary elementwise ──────────────────────────────────────────────

static int test_neg(void) {
    setup_graph_dir("neg");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_NEG, a, term_era());
    Term loss = sum_all(ctx, y, 1);
    f32 got[3]; int r = read_grad_1(ctx, loss, a, got, 3);
    int fails = 0;
    if (r < 0) fails++; else {
        f32 exp[3] = {-1, -1, -1};
        fails += report("NEG", got, exp, 3);
    }
    const char *needles[] = {"GRAD", "SUM", "NEG"};
    fails += report_topo("NEG", count_step_dots("neg"),
                         assert_in_step0("neg", needles, 3));
    thvm_free(ctx); return fails;
}

static int test_exp(void) {
    setup_graph_dir("exp");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {0, 1, 2};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_EXP, a, term_era());
    Term loss = sum_all(ctx, y, 1);
    f32 got[3]; int r = read_grad_1(ctx, loss, a, got, 3);
    int fails = 0;
    if (r < 0) fails++; else {
        f32 exp[3] = {expf(0), expf(1), expf(2)};
        fails += report("EXP", got, exp, 3);
    }
    const char *needles[] = {"GRAD", "SUM", "EXP"};
    fails += report_topo("EXP", count_step_dots("exp"),
                         assert_in_step0("exp", needles, 3));
    thvm_free(ctx); return fails;
}

static int test_log(void) {
    setup_graph_dir("log");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 4};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_LOG, a, term_era());
    Term loss = sum_all(ctx, y, 1);
    f32 got[3]; int r = read_grad_1(ctx, loss, a, got, 3);
    int fails = 0;
    if (r < 0) fails++; else {
        f32 exp[3] = {1.0f, 0.5f, 0.25f};
        fails += report("LOG", got, exp, 3);
    }
    const char *needles[] = {"GRAD", "SUM", "LOG"};
    fails += report_topo("LOG", count_step_dots("log"),
                         assert_in_step0("log", needles, 3));
    thvm_free(ctx); return fails;
}

static int test_sqrt(void) {
    setup_graph_dir("sqrt");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 4, 9};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_SQRT, a, term_era());
    Term loss = sum_all(ctx, y, 1);
    f32 got[3]; int r = read_grad_1(ctx, loss, a, got, 3);
    int fails = 0;
    if (r < 0) fails++; else {
        f32 exp[3] = {0.5f, 0.25f, 1.0f/6.0f};
        fails += report("SQRT", got, exp, 3);
    }
    const char *needles[] = {"GRAD", "SUM", "SQRT"};
    fails += report_topo("SQRT", count_step_dots("sqrt"),
                         assert_in_step0("sqrt", needles, 3));
    thvm_free(ctx); return fails;
}

static int test_relu(void) {
    setup_graph_dir("relu");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {-1, 0, 1, 2};
    Term a = thvm_tensor(ctx, ad, SHAPE(4));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_RELU, a, term_era());
    Term loss = sum_all(ctx, y, 1);
    f32 got[4]; int r = read_grad_1(ctx, loss, a, got, 4);
    int fails = 0;
    if (r < 0) fails++; else {
        f32 exp[4] = {0, 0, 1, 1};
        fails += report("RELU", got, exp, 4);
    }
    const char *needles[] = {"GRAD", "SUM", "RELU"};
    fails += report_topo("RELU", count_step_dots("relu"),
                         assert_in_step0("relu", needles, 3));
    thvm_free(ctx); return fails;
}

// ─── reductions ─────────────────────────────────────────────────────

static int test_sum(void) {
    setup_graph_dir("sum");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3, 4};
    Term a = thvm_tensor(ctx, ad, SHAPE(4));
    thvm_set_requires_grad(ctx, a);
    Term loss = thvm_sum_axes(ctx, a, (u32[]){0}, 1);
    f32 got[4]; int r = read_grad_1(ctx, loss, a, got, 4);
    int fails = 0;
    if (r < 0) fails++; else {
        f32 exp[4] = {1, 1, 1, 1};
        fails += report("SUM", got, exp, 4);
    }
    const char *needles[] = {"GRAD", "SUM"};
    fails += report_topo("SUM", count_step_dots("sum"),
                         assert_in_step0("sum", needles, 2));
    thvm_free(ctx); return fails;
}

// ─── movement ops ──────────────────────────────────────────────────

static int test_reshape(void) {
    setup_graph_dir("reshape");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3, 4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(6));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_reshape(ctx, a, SHAPE(2, 3));
    Term loss = sum_all(ctx, y, 2);
    f32 got[6]; int r = read_grad_1(ctx, loss, a, got, 6);
    int fails = 0;
    if (r < 0) fails++; else {
        f32 exp[6] = {1, 1, 1, 1, 1, 1};
        fails += report("RESHAPE", got, exp, 6);
    }
    const char *needles[] = {"GRAD", "SUM", "RESHAPE"};
    fails += report_topo("RESHAPE", count_step_dots("reshape"),
                         assert_in_step0("reshape", needles, 3));
    thvm_free(ctx); return fails;
}

static int test_expand(void) {
    setup_graph_dir("expand");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_expand(ctx, thvm_reshape(ctx, a, SHAPE(1, 3)), SHAPE(4, 3));
    Term loss = sum_all(ctx, y, 2);
    f32 got[3]; int r = read_grad_1(ctx, loss, a, got, 3);
    int fails = 0;
    if (r < 0) fails++; else {
        f32 exp[3] = {4, 4, 4};
        fails += report("EXPAND", got, exp, 3);
    }
    const char *needles[] = {"GRAD", "SUM", "EXPAND"};
    fails += report_topo("EXPAND", count_step_dots("expand"),
                         assert_in_step0("expand", needles, 3));
    thvm_free(ctx); return fails;
}

// ─── non-differentiable ───────────────────────────────────────────

static int test_cmp(void) {
    setup_graph_dir("cmp");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {2, 2, 2};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    thvm_set_requires_grad(ctx, a);
    Term y = thvm_op(ctx, UOP_CMP, a, b);
    Term loss = sum_all(ctx, y, 1);
    f32 got[3]; int r = read_grad_1(ctx, loss, a, got, 3);
    int fails = 0;
    if (r < 0) fails++; else {
        f32 exp[3] = {0, 0, 0};
        fails += report("CMP", got, exp, 3);
    }
    const char *needles[] = {"GRAD", "SUM", "CMP"};
    fails += report_topo("CMP", count_step_dots("cmp"),
                         assert_in_step0("cmp", needles, 3));
    thvm_free(ctx); return fails;
}

int main(void) {
    int fails = 0;
    // binary
    fails += test_add();
    fails += test_sub();
    fails += test_mul();
    fails += test_div();
    fails += test_max();
    // unary
    fails += test_neg();
    fails += test_exp();
    fails += test_log();
    fails += test_sqrt();
    fails += test_relu();
    // reductions
    fails += test_sum();
    // movement
    fails += test_reshape();
    fails += test_expand();
    // non-diff
    fails += test_cmp();

    printf("\ntotal failures: %d\n", fails);
    return fails ? 1 : 0;
}
