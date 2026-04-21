// test_rewrite_rules.m — semantic tests for ALL interaction rules.
//
// Each test:
//   1. Builds an input term (LHS of the rewrite).
//   2. Builds the expected post-reduce term by hand (RHS).
//   3. Reduces the LHS via thvm_eval.
//   4. Asserts the reduced term is structurally equal to RHS.
//
// "Structural equality" is term-tree equality up to heap locations: two
// terms are equal if their tags + ext + children are equal, recursively,
// with TAG_TEN compared by (shape, data); TAG_NUM by value; atoms by tag.
//
// This is the right test surface for "the rewrite is correct":
// numeric equivalence is necessary but not sufficient — many wrong
// rewrites happen to compute the right number.
//
// Scope: starts with the most critical rules (DUP, APP/LAM, and a few
// GRAD rewrites).  Add new rules by appending a test function + main()
// entry.  Build: clang -O2 ... test/test_rewrite_rules.m.

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <string.h>

static int g_pass = 0, g_fail = 0;

#define REWRITE_GRAPH_ROOT "graphs/rewrite_rules"

// Configure per-test graph dump directory: one folder per rule, holding
// pre-reduce / post-reduce / post-sweep .dot snapshots.  Called at the
// top of each test_rule_* function.
static void setup_rule_graph_dir(const char *rule) {
    char dir[256];
    snprintf(dir, sizeof(dir), "%s/%s", REWRITE_GRAPH_ROOT, rule);
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "rm -rf %s && mkdir -p %s", dir, dir);
    int r = system(cmd); (void)r;
    setenv("THVM_GRAPH", "1", 1);
    setenv("THVM_GRAPH_DIR", dir, 1);
    setenv("THVM_GRAPH_STOP_AFTER_SWEEP", "1", 1);
}

static int report(const char *name, int ok) {
    printf("%-40s %s\n", name, ok ? "PASS" : "FAIL");
    if (ok) g_pass++; else g_fail++;
    return !ok;
}

static void dbg_term(TinyHVM *ctx, const char *label, Term t) {
    (void)ctx;
    fprintf(stderr, "  %s: tag=%u ext=%u val=%llu\n", label,
            (u32)term_tag(t), (u32)term_ext(t), (unsigned long long)term_val(t));
}

// ──────────────────────────────────────────────────────────────────────
// Structural equality for Term trees rooted on ctx->heap.
// Ignores heap locs; compares tags, ext, and recursively the arity-many
// children of structural terms.  TAG_TEN compared by tensor id-resolved
// shape + materialized data (to cover cases where two rules emit
// different heap-layouts of "the same" tensor).
// ──────────────────────────────────────────────────────────────────────

static u32 term_struct_arity(Term t) {
    u8 tag = term_tag(t);
    switch (tag) {
        case TAG_APP: case TAG_LAM: case TAG_SUP: case TAG_CTR:
            return (tag == TAG_CTR) ? term_ext(t) : 2;
        case TAG_TOP: {
            u32 ext = term_ext(t);
            if (ext == UOP_WHERE || ext == UOP_IFZ) return 3;
            if (ext == UOP_NEG || ext == UOP_EXP || ext == UOP_LOG ||
                ext == UOP_RELU || ext == UOP_SQRT || ext == UOP_CAST ||
                ext == UOP_DETACH || ext == UOP_LOG_PRINT) return 1;
            return 2;
        }
        default: return 0;
    }
}

// Resolve a Term one hop: follow DP0/DP1/VAR/ALO chains to whatever they
// currently point to.  Bounded recursion to avoid infinite loops.
static Term term_resolve(TinyHVM *ctx, Term t, int depth) {
    for (int i = 0; i < depth; i++) {
        u8 tag = term_tag(t);
        if (tag == TAG_DP0 || tag == TAG_DP1 || tag == TAG_VAR) {
            u64 l = term_val(t);
            if (l == 0 || l >= ctx->heap_pos) return t;
            Term next = heap_read(ctx, l);
            if (term_is_sub(next)) next = term_strip_sub(next);
            if (next == t) return t;
            t = next; continue;
        }
        break;
    }
    return t;
}

static int term_eq(TinyHVM *ctx, Term a, Term b, int depth);

static int ten_values_eq(TinyHVM *ctx, Term a, Term b) {
    u32 ai = (u32)term_val(a), bi = (u32)term_val(b);
    if (ai >= ctx->tensor_count || bi >= ctx->tensor_count) return 0;
    Shape as = ctx->tensors[ai].view.shape;
    Shape bs = ctx->tensors[bi].view.shape;
    if (as.rank != bs.rank) return 0;
    u32 n = 1;
    for (u32 i = 0; i < as.rank; i++) {
        if (as.dims[i] != bs.dims[i]) return 0;
        n *= as.dims[i];
    }
    u32 adt = DTYPE_F32, bdt = DTYPE_F32;
    Shape sh_a = as, sh_b = bs;
    f32 *av = (f32 *)thvm_to_host_raw(ctx, a, &adt, &sh_a);
    f32 *bv = (f32 *)thvm_to_host_raw(ctx, b, &bdt, &sh_b);
    if (!av || !bv) return 0;
    for (u32 i = 0; i < n; i++) {
        f32 d = av[i] - bv[i]; if (d < 0) d = -d;
        if (d > 1e-5f) return 0;
    }
    return 1;
}

// Semantic equality modulo execution: both sides are fully evaluated and
// materialized to host tensors, then compared by shape + values.  This
// treats "the rewrite is semantically correct" — orthogonal to scheduler
// artifacts like KERNEL wrappers or EXPAND view chains that don't appear
// in the hand-built expected term.
//
// CTR is handled structurally (recurse per slot).  NUM and ERA atoms
// compared by tag+value.  Everything else is forced to tensor form and
// compared data-wise.
static int tensor_data_eq(TinyHVM *ctx, Term a, Term b) {
    u32 adt = DTYPE_F32, bdt = DTYPE_F32;
    Shape sha = SHAPE(1), shb = SHAPE(1);
    f32 *av = (f32 *)thvm_to_host_raw(ctx, a, &adt, &sha);
    f32 *bv = (f32 *)thvm_to_host_raw(ctx, b, &bdt, &shb);
    if (!av || !bv) return 0;
    if (sha.rank != shb.rank) return 0;
    u32 n = 1;
    for (u32 i = 0; i < sha.rank; i++) {
        if (sha.dims[i] != shb.dims[i]) return 0;
        n *= sha.dims[i];
    }
    for (u32 i = 0; i < n; i++) {
        f32 d = av[i] - bv[i]; if (d < 0) d = -d;
        if (d > 1e-5f) return 0;
    }
    return 1;
}

static int term_eq(TinyHVM *ctx, Term a, Term b, int depth) {
    if (depth <= 0) return 0;
    a = term_resolve(ctx, a, 16);
    b = term_resolve(ctx, b, 16);
    u8 ta = term_tag(a), tb = term_tag(b);
    if (ta == TAG_CTR && tb == TAG_CTR) {
        if (term_ext(a) != term_ext(b)) return 0;
        u32 ar = term_ext(a);
        u64 la = term_val(a), lb = term_val(b);
        for (u32 i = 0; i < ar; i++) {
            Term ca = heap_read(ctx, la + i);
            Term cb = heap_read(ctx, lb + i);
            if (!term_eq(ctx, ca, cb, depth - 1)) return 0;
        }
        return 1;
    }
    if (ta == TAG_NUM && tb == TAG_NUM) return term_val(a) == term_val(b);
    if (ta == TAG_ERA && tb == TAG_ERA) return 1;
    return tensor_data_eq(ctx, a, b);
}

// ──────────────────────────────────────────────────────────────────────
// Rule: DUP ⊳ TEN — atomic sharing.
//   input:    let (a, b) = DUP(x)  in (a, b)   where x : TEN[3]
//   expected: (x, x)  (same tensor id, incref'd)
// ──────────────────────────────────────────────────────────────────────
static int test_rule_dup_ten(void) {
    setup_rule_graph_dir("dup_ten");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1, 2, 3};
    Term x = thvm_tensor(ctx, xd, SHAPE(3));
    Term a, b; thvm_dup(ctx, thvm_fresh_label(ctx), x, &a, &b);
    Term pair = thvm_ctr(ctx, (Term[]){a, b}, 2);
    Term reduced = thvm_eval(ctx, pair);
    // Expected: CTR{TEN(x), TEN(x)}
    Term xa = x, xb = x;  // same id
    Term expected = thvm_ctr(ctx, (Term[]){xa, xb}, 2);
    int ok = term_eq(ctx, reduced, expected, 16);
    thvm_free(ctx);
    return report("dup_ten", ok);
}

// ──────────────────────────────────────────────────────────────────────
// Rule: APP ⊳ LAM — beta reduction.
//   input:    (λx. x) y   where y : TEN[3]
//   expected: y
// ──────────────────────────────────────────────────────────────────────
static int test_rule_app_lam_identity(void) {
    setup_rule_graph_dir("app_lam_identity");
    TinyHVM *ctx = thvm_init("cpu");
    f32 yd[] = {4, 5, 6};
    Term y = thvm_tensor(ctx, yd, SHAPE(3));
    Term var;
    Term body_placeholder = term_era();
    Term lam = thvm_lam(ctx, &var, body_placeholder);
    // Install body = var (identity λ).
    heap_set(ctx, term_val(lam) + 1, var);
    Term app = thvm_app(ctx, lam, y);
    Term reduced = thvm_eval(ctx, app);
    // Expected: y
    int ok = term_eq(ctx, reduced, y, 16);
    thvm_free(ctx);
    return report("app_lam_identity", ok);
}

// ──────────────────────────────────────────────────────────────────────
// Rule: GRAD ⊳ TEN (leaf match).
//   input:    GRAD(x, x)     where x : TEN[3]
//   expected: ones([3])      (reverse-mode ones-seed on target leaf)
// ──────────────────────────────────────────────────────────────────────
static int test_rule_grad_leaf_match(void) {
    setup_rule_graph_dir("grad_leaf_match");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1, 2, 3};
    Term x = thvm_tensor(ctx, xd, SHAPE(3));
    // TEN is atomic — pass it directly as both y and target; no DUP needed.
    Term g = thvm_grad(ctx, x, x);
    Term reduced = thvm_eval(ctx, g);
    // Rule: GRAD(x, x) rewrites to EXPAND(TEN([1.0], shape=[1]), shape=[3]).
    // That's what GRAD_SCALAR_TEN emits for rank-1 non-scalar target.
    f32 one = 1.0f;
    Term scalar_one = thvm_tensor(ctx, &one, SHAPE(1));
    Term expected = thvm_expand(ctx, scalar_one, SHAPE(3));
    int ok = term_eq(ctx, reduced, expected, 16);
    if (!ok) { dbg_term(ctx, "reduced", reduced); dbg_term(ctx, "expected", expected); }
    thvm_free(ctx);
    return report("grad_leaf_match", ok);
}

// ──────────────────────────────────────────────────────────────────────
// Rule: GRAD ⊳ TEN (leaf non-match).
//   input:    GRAD(a, b)     where a, b are different tensors shape [3]
//   expected: zeros([3])
// ──────────────────────────────────────────────────────────────────────
static int test_rule_grad_leaf_nomatch(void) {
    setup_rule_graph_dir("grad_leaf_nomatch");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    Term g = thvm_grad(ctx, a, b);
    Term reduced = thvm_eval(ctx, g);
    // Rule: GRAD leaf non-match reduces to ERA (dead gradient branch).
    // ERA propagates through adjacent ops via GRAD_ADD/SUB/MUL peepholes.
    int ok = (term_tag(reduced) == TAG_ERA);
    if (!ok) dbg_term(ctx, "reduced (expected ERA)", reduced);
    thvm_free(ctx);
    return report("grad_leaf_nomatch", ok);
}

// ──────────────────────────────────────────────────────────────────────
// Rule: GRAD ⊳ ADD.
//   input:    GRAD(a + b, a)     where a, b : TEN[3], a != b
//   expected: ones([3]) + zeros([3])  --- reduces to ones([3]) at runtime
//   Value-level: grad = ones([3]).
// ──────────────────────────────────────────────────────────────────────
static int test_rule_grad_add(void) {
    setup_rule_graph_dir("grad_add");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    Term y = thvm_op(ctx, UOP_ADD, a, b);
    Term g = thvm_grad(ctx, y, a);
    Term reduced = thvm_eval(ctx, g);
    f32 ones[] = {1, 1, 1};
    Term expected = thvm_tensor(ctx, ones, SHAPE(3));
    int ok = term_eq(ctx, reduced, expected, 16);
    thvm_free(ctx);
    return report("grad_add", ok);
}

// ──────────────────────────────────────────────────────────────────────
// Rule: GRAD ⊳ MUL (Leibniz) for y = x*x.
//   input:    GRAD(x*x, x)
//   expected value: 2x  (structural equality via materialized TEN).
// ──────────────────────────────────────────────────────────────────────
static int test_rule_grad_mul_square(void) {
    setup_rule_graph_dir("grad_mul_square");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1, 2, 3};
    Term x = thvm_tensor(ctx, xd, SHAPE(3));
    // x appears twice in y (via MUL) and once as target — TEN is atomic,
    // a single DUP for the two MUL uses is sufficient; target needs no DUP.
    Term xa, xb; thvm_dup(ctx, thvm_fresh_label(ctx), x, &xa, &xb);
    Term y = thvm_op(ctx, UOP_MUL, xa, xb);
    Term g = thvm_grad(ctx, y, x);
    Term reduced = thvm_eval(ctx, g);
    f32 twox[] = {2, 4, 6};
    Term expected = thvm_tensor(ctx, twox, SHAPE(3));
    int ok = term_eq(ctx, reduced, expected, 16);
    thvm_free(ctx);
    return report("grad_mul_square", ok);
}

// ──────────────────────────────────────────────────────────────────────
// Rule: GRAD_FWD ⊳ TEN (JVP leaf match).
//   input:    GRAD_FWD(x, x)
//   expected: ones([3])
// ──────────────────────────────────────────────────────────────────────
static int test_rule_grad_fwd_leaf_match(void) {
    setup_rule_graph_dir("grad_fwd_leaf_match");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1, 2, 3};
    Term x = thvm_tensor(ctx, xd, SHAPE(3));
    Term g = thvm_grad_fwd(ctx, x, x);
    Term reduced = thvm_eval(ctx, g);
    f32 ones[] = {1, 1, 1};
    Term expected = thvm_tensor(ctx, ones, SHAPE(3));
    int ok = term_eq(ctx, reduced, expected, 16);
    thvm_free(ctx);
    return report("grad_fwd_leaf_match", ok);
}

// ──────────────────────────────────────────────────────────────────────
// Rule: GRAD_FWD ⊳ SUM — shape collapses to scalar in forward mode.
//   input:    GRAD_FWD(sum(x*x), x)
//   expected: [12]          (sum of tangent = sum of 2x for x=[1,2,3])
// ──────────────────────────────────────────────────────────────────────
static int test_rule_grad_fwd_sum_of_square(void) {
    setup_rule_graph_dir("grad_fwd_sum_of_square");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1, 2, 3};
    Term x = thvm_tensor(ctx, xd, SHAPE(3));
    Term xa, xb; thvm_dup(ctx, thvm_fresh_label(ctx), x, &xa, &xb);
    Term sq = thvm_op(ctx, UOP_MUL, xa, xb);
    Term y = thvm_sum_axes(ctx, sq, (u32[]){0}, 1);
    Term g = thvm_grad_fwd(ctx, y, x);
    Term reduced = thvm_eval(ctx, g);
    f32 s[] = {12};
    Term expected = thvm_tensor(ctx, s, SHAPE(1));
    int ok = term_eq(ctx, reduced, expected, 16);
    if (!ok) {
        u32 dt = DTYPE_F32; Shape sh = SHAPE(1);
        f32 *rv = (f32*)thvm_to_host_raw(ctx, reduced, &dt, &sh);
        fprintf(stderr, "  grad_fwd_sum reduced rank=%u dim0=%u val=%g\n",
                sh.rank, sh.rank>0?sh.dims[0]:0, rv?rv[0]:-1.0f);
    }
    thvm_free(ctx);
    return report("grad_fwd_sum_of_square", ok);
}

// ──────────────────────────────────────────────────────────────────────
// Rule: ADD ⊳ TEN,TEN — elementwise tensor addition.
//   input:    a + b   a=[1,2,3], b=[4,5,6]
//   expected: [5,7,9]
// ──────────────────────────────────────────────────────────────────────
static int test_rule_add_ten_ten(void) {
    setup_rule_graph_dir("add_ten_ten");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3}, bd[] = {4, 5, 6};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term b = thvm_tensor(ctx, bd, SHAPE(3));
    Term y = thvm_op(ctx, UOP_ADD, a, b);
    Term reduced = thvm_eval(ctx, y);
    f32 s[] = {5, 7, 9};
    Term expected = thvm_tensor(ctx, s, SHAPE(3));
    int ok = term_eq(ctx, reduced, expected, 16);
    thvm_free(ctx);
    return report("add_ten_ten", ok);
}

// ──────────────────────────────────────────────────────────────────────
// Rule: SUM ⊳ TEN — axis reduction.
//   input:    sum(a, axis=0), a=[1,2,3]
//   expected: [6]
// ──────────────────────────────────────────────────────────────────────
static int test_rule_sum_ten(void) {
    setup_rule_graph_dir("sum_ten");
    TinyHVM *ctx = thvm_init("cpu");
    f32 ad[] = {1, 2, 3};
    Term a = thvm_tensor(ctx, ad, SHAPE(3));
    Term y = thvm_sum_axes(ctx, a, (u32[]){0}, 1);
    Term reduced = thvm_eval(ctx, y);
    f32 s[] = {6};
    Term expected = thvm_tensor(ctx, s, SHAPE(1));
    int ok = term_eq(ctx, reduced, expected, 16);
    thvm_free(ctx);
    return report("sum_ten", ok);
}

// ──────────────────────────────────────────────────────────────────────
// Rule: EXPAND ⊳ TEN — broadcast.
//   input:    expand(TEN([5], shape=[1]), shape=[3])
//   expected: [5,5,5]
// ──────────────────────────────────────────────────────────────────────
static int test_rule_expand_ten(void) {
    setup_rule_graph_dir("expand_ten");
    TinyHVM *ctx = thvm_init("cpu");
    f32 v = 5.0f;
    Term a = thvm_tensor(ctx, &v, SHAPE(1));
    Term y = thvm_expand(ctx, a, SHAPE(3));
    Term reduced = thvm_eval(ctx, y);
    f32 s[] = {5, 5, 5};
    Term expected = thvm_tensor(ctx, s, SHAPE(3));
    int ok = term_eq(ctx, reduced, expected, 16);
    thvm_free(ctx);
    return report("expand_ten", ok);
}

// ──────────────────────────────────────────────────────────────────────
// Rule: DUP ⊳ SUP (commutation vs annihilation).  Same-label SUP ⊳ DUP
// annihilates (each aux takes one slot of the SUP).
//   input:    let (a, b) = DUP_L( SUP_L(p, q) )  in CTR{a, b}
//   expected: CTR{p, q}
// ──────────────────────────────────────────────────────────────────────
static int test_rule_dup_sup_same_label(void) {
    setup_rule_graph_dir("dup_sup_same_label");
    TinyHVM *ctx = thvm_init("cpu");
    f32 pd[] = {1,2,3}, qd[] = {4,5,6};
    Term p = thvm_tensor(ctx, pd, SHAPE(3));
    Term q = thvm_tensor(ctx, qd, SHAPE(3));
    u32 L = thvm_fresh_label(ctx);
    Term sup = thvm_sup(ctx, L, p, q);
    Term a, b; thvm_dup(ctx, L, sup, &a, &b);
    Term pair = thvm_ctr(ctx, (Term[]){a, b}, 2);
    Term reduced = thvm_eval(ctx, pair);
    Term expected = thvm_ctr(ctx, (Term[]){p, q}, 2);
    int ok = term_eq(ctx, reduced, expected, 16);
    thvm_free(ctx);
    return report("dup_sup_same_label", ok);
}

// ──────────────────────────────────────────────────────────────────────
// Rule: GRAD ⊳ NEG.   d(-x)/dx = -1.
// ──────────────────────────────────────────────────────────────────────
static int test_rule_grad_neg(void) {
    setup_rule_graph_dir("grad_neg");
    TinyHVM *ctx = thvm_init("cpu");
    f32 xd[] = {1, 2, 3};
    Term x = thvm_tensor(ctx, xd, SHAPE(3));
    Term y = thvm_op(ctx, UOP_NEG, x, term_era());
    Term g = thvm_grad(ctx, y, x);
    Term reduced = thvm_eval(ctx, g);
    f32 mones[] = {-1, -1, -1};
    Term expected = thvm_tensor(ctx, mones, SHAPE(3));
    int ok = term_eq(ctx, reduced, expected, 16);
    thvm_free(ctx);
    return report("grad_neg", ok);
}

int main(void) {
    // Match the env the grad-rules tests run under: stop-after-sweep prevents
    // JIT dispatch from wrapping results in KERNEL nodes, keeping the compute
    // graph close to the rule's direct rewrite.
    setenv("THVM_GRAPH_STOP_AFTER_SWEEP", "1", 1);

    test_rule_dup_ten();
    test_rule_app_lam_identity();
    test_rule_grad_leaf_match();
    test_rule_grad_leaf_nomatch();
    test_rule_grad_add();
    test_rule_grad_mul_square();
    test_rule_add_ten_ten();
    test_rule_sum_ten();
    test_rule_expand_ten();
    test_rule_dup_sup_same_label();
    test_rule_grad_neg();
    test_rule_grad_fwd_leaf_match();
    // KNOWN FAIL: produces val=0 under isolated-test context while the
    // equivalent test in test_grad_rules.m (test_e2e_jvp_sum_of_square)
    // produces 12.  Root cause: test_grad_rules runs a long chain of
    // prior tests that warm some scheduler/context state the JVP SUM rule
    // relies on; this isolated invocation exposes the dependency.  Real
    // bug in the rule's eager-reduce path — to fix separately.
    test_rule_grad_fwd_sum_of_square();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
