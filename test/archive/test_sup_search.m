// test_sup_search.m — Tests for type enumeration + unordered SUP machinery
// Tests: EQL, AND, OR, MAT, ANY, collapse_ordered, USP, UDP

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#include "../src/backend/metal/_.m"

#define DEVICE "cpu"

#include <stdio.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s (line %d)\n", msg, __LINE__); tests_failed++; } \
    else { tests_passed++; } \
} while(0)

#define ASSERT_EQ(a, b, msg) ASSERT((a) == (b), msg)

// ── EQL tests ────────────────────────────────────────────────────────────────

static void test_eql_basic(void) {
    printf("test_eql_basic:\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // EQL(3, 3) → NUM(1)
    Term eq1 = thvm_eql(ctx, term_num_u32(3), term_num_u32(3));
    Term r1 = thvm_reduce(ctx, eq1);
    ASSERT_EQ(term_tag(r1), TAG_NUM, "eql(3,3) is NUM");
    ASSERT_EQ(term_as_u32(r1), 1, "eql(3,3) = 1");

    // EQL(3, 4) → NUM(0)
    Term eq2 = thvm_eql(ctx, term_num_u32(3), term_num_u32(4));
    Term r2 = thvm_reduce(ctx, eq2);
    ASSERT_EQ(term_tag(r2), TAG_NUM, "eql(3,4) is NUM");
    ASSERT_EQ(term_as_u32(r2), 0, "eql(3,4) = 0");

    // EQL(ERA, anything) → ERA
    Term eq3 = thvm_eql(ctx, term_era(), term_num_u32(5));
    Term r3 = thvm_reduce(ctx, eq3);
    ASSERT_EQ(term_tag(r3), TAG_ERA, "eql(ERA,5) = ERA");

    thvm_free(ctx);
}

static void test_eql_any(void) {
    printf("test_eql_any:\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // EQL(ANY, 42) → 1
    Term eq1 = thvm_eql(ctx, thvm_any(), term_num_u32(42));
    Term r1 = thvm_reduce(ctx, eq1);
    ASSERT_EQ(term_tag(r1), TAG_NUM, "eql(ANY,42) is NUM");
    ASSERT_EQ(term_as_u32(r1), 1, "eql(ANY,42) = 1");

    // EQL(7, ANY) → 1
    Term eq2 = thvm_eql(ctx, term_num_u32(7), thvm_any());
    Term r2 = thvm_reduce(ctx, eq2);
    ASSERT_EQ(term_as_u32(r2), 1, "eql(7,ANY) = 1");

    thvm_free(ctx);
}

static void test_eql_sup(void) {
    printf("test_eql_sup:\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // EQL(&L{1,2}, 1) → &L{EQL(1,1), EQL(2,1)} → &L{1, 0}
    u32 lab = thvm_fresh_label(ctx);
    Term sup = thvm_sup(ctx, lab, term_num_u32(1), term_num_u32(2));
    Term eq = thvm_eql(ctx, sup, term_num_u32(1));
    CollapseResult cr = thvm_collapse(ctx, eq);
    ASSERT_EQ(cr.count, 2, "eql-sup collapses to 2 leaves");
    // One should be 1, other 0
    u32 sum = 0;
    for (u32 i = 0; i < cr.count; i++) {
        ASSERT_EQ(term_tag(cr.terms[i]), TAG_NUM, "eql-sup leaf is NUM");
        sum += term_as_u32(cr.terms[i]);
    }
    ASSERT_EQ(sum, 1, "eql-sup: exactly one match");
    thvm_collapse_free(&cr);

    thvm_free(ctx);
}

// ── AND/OR tests ─────────────────────────────────────────────────────────────

static void test_and_or_basic(void) {
    printf("test_and_or_basic:\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // AND(1, 0) → 0  (short-circuit: nonzero passes through to b)
    Term a1 = thvm_and(ctx, term_num_u32(1), term_num_u32(0));
    Term r1 = thvm_reduce(ctx, a1);
    ASSERT_EQ(term_tag(r1), TAG_NUM, "and(1,0) is NUM");
    ASSERT_EQ(term_as_u32(r1), 0, "and(1,0) = 0");

    // AND(0, 99) → 0  (short-circuit: zero → 0)
    Term a2 = thvm_and(ctx, term_num_u32(0), term_num_u32(99));
    Term r2 = thvm_reduce(ctx, a2);
    ASSERT_EQ(term_as_u32(r2), 0, "and(0,99) = 0");

    // AND(1, 1) → 1
    Term a3 = thvm_and(ctx, term_num_u32(1), term_num_u32(1));
    Term r3 = thvm_reduce(ctx, a3);
    ASSERT_EQ(term_as_u32(r3), 1, "and(1,1) = 1");

    // OR(0, 1) → 1
    Term o1 = thvm_or(ctx, term_num_u32(0), term_num_u32(1));
    Term ro1 = thvm_reduce(ctx, o1);
    ASSERT_EQ(term_as_u32(ro1), 1, "or(0,1) = 1");

    // OR(1, 0) → 1  (short-circuit: nonzero → 1)
    Term o2 = thvm_or(ctx, term_num_u32(1), term_num_u32(0));
    Term ro2 = thvm_reduce(ctx, o2);
    ASSERT_EQ(term_as_u32(ro2), 1, "or(1,0) = 1");

    // OR(0, 0) → 0
    Term o3 = thvm_or(ctx, term_num_u32(0), term_num_u32(0));
    Term ro3 = thvm_reduce(ctx, o3);
    ASSERT_EQ(term_as_u32(ro3), 0, "or(0,0) = 0");

    thvm_free(ctx);
}

static void test_and_sup(void) {
    printf("test_and_sup:\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // AND(&L{0,1}, 42) → &L{AND(0,42), AND(1,42)} → &L{0, 42}
    u32 lab = thvm_fresh_label(ctx);
    Term sup = thvm_sup(ctx, lab, term_num_u32(0), term_num_u32(1));
    Term a = thvm_and(ctx, sup, term_num_u32(42));
    CollapseResult cr = thvm_collapse(ctx, a);
    ASSERT_EQ(cr.count, 2, "and-sup collapses to 2");
    // Expect {0, 42}
    u32 has_0 = 0, has_42 = 0;
    for (u32 i = 0; i < cr.count; i++) {
        if (term_tag(cr.terms[i]) == TAG_NUM) {
            if (term_as_u32(cr.terms[i]) == 0) has_0 = 1;
            if (term_as_u32(cr.terms[i]) == 42) has_42 = 1;
        }
    }
    ASSERT(has_0, "and-sup has 0 branch");
    ASSERT(has_42, "and-sup has 42 branch");
    thvm_collapse_free(&cr);

    thvm_free(ctx);
}

// ── MAT tests ────────────────────────────────────────────────────────────────

static void test_mat_basic(void) {
    printf("test_mat_basic:\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // MAT(match_tag=3, handler=NUM(100), fallback=id) applied to NUM(3) → NUM(100)
    Term handler = term_num_u32(100);
    Term var;
    Term fallback = thvm_lam(ctx, &var, term_era());  // placeholder body
    heap_set(ctx, term_val(fallback) + 1, var);  // patch body = var (identity lambda)
    Term mat = thvm_mat(ctx, 3, handler, fallback);
    Term applied = thvm_app(ctx, mat, term_num_u32(3));
    Term r1 = thvm_reduce(ctx, applied);
    ASSERT_EQ(term_tag(r1), TAG_NUM, "mat(3,h,m) applied to 3 is NUM");
    ASSERT_EQ(term_as_u32(r1), 100, "mat match → handler");

    // MAT(match_tag=3, handler=NUM(100), fallback=id) applied to NUM(5) → APP(id, 5) → 5
    Term handler2 = term_num_u32(100);
    Term var2;
    Term fallback2 = thvm_lam(ctx, &var2, term_era());
    heap_set(ctx, term_val(fallback2) + 1, var2);
    Term mat2 = thvm_mat(ctx, 3, handler2, fallback2);
    Term applied2 = thvm_app(ctx, mat2, term_num_u32(5));
    Term r2 = thvm_reduce(ctx, applied2);
    ASSERT_EQ(term_tag(r2), TAG_NUM, "mat(3,h,m) applied to 5 is NUM");
    ASSERT_EQ(term_as_u32(r2), 5, "mat mismatch → fallback(5) = 5");

    thvm_free(ctx);
}

static void test_mat_sup(void) {
    printf("test_mat_sup:\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // MAT(3, NUM(100), id) applied to &L{3, 5}
    // → &L{MAT₀(3), MAT₁(5)} → &L{100, 5}
    Term handler = term_num_u32(100);
    Term var;
    Term fallback = thvm_lam(ctx, &var, var);
    Term mat = thvm_mat(ctx, 3, handler, fallback);

    u32 lab = thvm_fresh_label(ctx);
    Term sup_arg = thvm_sup(ctx, lab, term_num_u32(3), term_num_u32(5));
    Term applied = thvm_app(ctx, mat, sup_arg);

    CollapseResult cr = thvm_collapse(ctx, applied);
    ASSERT_EQ(cr.count, 2, "mat-sup collapses to 2");
    u32 has_100 = 0, has_5 = 0;
    for (u32 i = 0; i < cr.count; i++) {
        if (term_tag(cr.terms[i]) == TAG_NUM) {
            if (term_as_u32(cr.terms[i]) == 100) has_100 = 1;
            if (term_as_u32(cr.terms[i]) == 5) has_5 = 1;
        }
    }
    ASSERT(has_100, "mat-sup: match branch = 100");
    ASSERT(has_5, "mat-sup: mismatch branch = 5");
    thvm_collapse_free(&cr);

    thvm_free(ctx);
}

// ── Collapse ordered tests ───────────────────────────────────────────────────

static void test_collapse_ordered(void) {
    printf("test_collapse_ordered:\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // Build: &L{INC(INC(NUM(1))), NUM(2)}
    // INC wrapping gives NUM(1) lower key (higher priority)
    // So collapse_ordered should return NUM(1) first
    u32 lab = thvm_fresh_label(ctx);
    Term wrapped = thvm_inc(ctx, thvm_inc(ctx, term_num_u32(1)));
    Term sup = thvm_sup(ctx, lab, wrapped, term_num_u32(2));

    CollapseResult cr = thvm_collapse_ordered(ctx, sup, 10);
    ASSERT_EQ(cr.count, 2, "ordered collapse: 2 results");
    // First result should be the INC-wrapped one (lower key = higher priority)
    ASSERT_EQ(term_tag(cr.terms[0]), TAG_NUM, "first is NUM");
    ASSERT_EQ(term_as_u32(cr.terms[0]), 1, "INC-wrapped comes first");
    ASSERT_EQ(term_as_u32(cr.terms[1]), 2, "unwrapped comes second");
    thvm_collapse_free(&cr);

    thvm_free(ctx);
}

static void test_collapse_ordered_limit(void) {
    printf("test_collapse_ordered_limit:\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // Build 4 branches, limit to 2
    u32 l0 = thvm_fresh_label(ctx);
    u32 l1 = thvm_fresh_label(ctx);
    Term sup = thvm_sup(ctx, l0,
        thvm_sup(ctx, l1, term_num_u32(10), term_num_u32(20)),
        thvm_sup(ctx, l1, term_num_u32(30), term_num_u32(40)));

    CollapseResult cr = thvm_collapse_ordered(ctx, sup, 2);
    ASSERT_EQ(cr.count, 2, "ordered collapse limited to 2");
    thvm_collapse_free(&cr);

    thvm_free(ctx);
}

// ── USP/UDP tests ────────────────────────────────────────────────────────────

static void test_usp_basic(void) {
    printf("test_usp_basic:\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // USP collapses like SUP
    u32 lab = thvm_fresh_label(ctx);
    Term usp = thvm_usp(ctx, lab, term_num_u32(10), term_num_u32(20));
    CollapseResult cr = thvm_collapse(ctx, usp);
    ASSERT_EQ(cr.count, 2, "usp collapse: 2 branches");
    u32 sum = 0;
    for (u32 i = 0; i < cr.count; i++) sum += term_as_u32(cr.terms[i]);
    ASSERT_EQ(sum, 30, "usp branches: 10 + 20");
    thvm_collapse_free(&cr);

    thvm_free(ctx);
}

static void test_udp_usp_same_label(void) {
    printf("test_udp_usp_same_label:\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // UDP_L(USP_L(a, b)) → a  (consume one branch)
    u32 lab = thvm_fresh_label(ctx);
    Term usp = thvm_usp(ctx, lab, term_num_u32(42), term_num_u32(99));
    Term udp = thvm_udp(ctx, lab, usp);
    Term r = thvm_reduce(ctx, udp);
    ASSERT_EQ(term_tag(r), TAG_NUM, "udp-usp same label: NUM");
    ASSERT_EQ(term_as_u32(r), 42, "udp-usp same label: gets first branch");

    thvm_free(ctx);
}

static void test_udp_atom(void) {
    printf("test_udp_atom:\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // UDP on an atom → returns the atom
    u32 lab = thvm_fresh_label(ctx);
    Term udp = thvm_udp(ctx, lab, term_num_u32(77));
    Term r = thvm_reduce(ctx, udp);
    ASSERT_EQ(term_tag(r), TAG_NUM, "udp-atom: NUM");
    ASSERT_EQ(term_as_u32(r), 77, "udp-atom: value");

    thvm_free(ctx);
}

// ── Integration: synthesis pattern ───────────────────────────────────────────

static void test_synthesis_pattern(void) {
    printf("test_synthesis_pattern (SUP search + EQL filter):\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // Superpose 3 candidates: {10, 20, 30}
    // Filter: which one equals 20?
    // Expected: collapse to get {0, 1, 0} from EQL, then AND to filter
    u32 l0 = thvm_fresh_label(ctx);
    u32 l1 = thvm_fresh_label(ctx);
    Term candidates = thvm_sup(ctx, l0,
        term_num_u32(10),
        thvm_sup(ctx, l1, term_num_u32(20), term_num_u32(30)));

    // Check: candidates === 20
    Term check = thvm_eql(ctx, candidates, term_num_u32(20));
    CollapseResult cr = thvm_collapse(ctx, check);
    ASSERT_EQ(cr.count, 3, "synthesis: 3 branches");

    // Count matches
    u32 matches = 0;
    for (u32 i = 0; i < cr.count; i++) {
        if (term_tag(cr.terms[i]) == TAG_NUM && term_as_u32(cr.terms[i]) == 1)
            matches++;
    }
    ASSERT_EQ(matches, 1, "synthesis: exactly one candidate matches 20");
    thvm_collapse_free(&cr);

    thvm_free(ctx);
}

static void test_synthesis_and_chain(void) {
    printf("test_synthesis_and_chain (multi-constraint filter):\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // Search for x where x > 0 AND x < 3 (encoded as x != 0 AND x != 3)
    // Candidates: {0, 1, 2, 3}
    u32 l0 = thvm_fresh_label(ctx);
    u32 l1 = thvm_fresh_label(ctx);
    u32 l2 = thvm_fresh_label(ctx);
    Term candidates = thvm_sup(ctx, l0,
        thvm_sup(ctx, l1, term_num_u32(0), term_num_u32(1)),
        thvm_sup(ctx, l2, term_num_u32(2), term_num_u32(3)));

    // Constraint: NOT(x == 0) AND NOT(x == 3)
    // Since we don't have NOT, we use: (x != 0) implemented as OP2(sub, EQL(x,0), 1)
    // Simpler: use EQL to check x==target for a known-good value
    // Actually let's just verify the EQL+AND pipeline works with two constraints:
    // c1 = (x !== 0) = 1 - EQL(x, 0)
    // c2 = (x !== 3) = 1 - EQL(x, 3)
    // valid = AND(c1, c2)
    Term c1 = thvm_op2(ctx, 1, term_num_u32(1),  // 1 - EQL(x, 0)
                thvm_eql(ctx, candidates, term_num_u32(0)));
    Term c2 = thvm_op2(ctx, 1, term_num_u32(1),  // 1 - EQL(x, 3)
                thvm_eql(ctx, candidates, term_num_u32(3)));
    Term valid = thvm_and(ctx, c1, c2);

    CollapseResult cr = thvm_collapse(ctx, valid);
    ASSERT_EQ(cr.count, 4, "and-chain: 4 branches");

    // Branches where both constraints pass should be 1, others 0
    u32 passing = 0;
    for (u32 i = 0; i < cr.count; i++) {
        if (term_tag(cr.terms[i]) == TAG_NUM && term_as_u32(cr.terms[i]) == 1)
            passing++;
    }
    ASSERT_EQ(passing, 2, "and-chain: 2 candidates pass (1 and 2)");
    thvm_collapse_free(&cr);

    thvm_free(ctx);
}

// ── DUP through new nodes ────────────────────────────────────────────────────

static void test_dup_new_nodes(void) {
    printf("test_dup_new_nodes:\n");
    TinyHVM *ctx = thvm_init(DEVICE);

    // DUP through EQL: should produce two copies
    u32 lab = thvm_fresh_label(ctx);
    Term eql = thvm_eql(ctx, term_num_u32(5), term_num_u32(5));
    Term dp0, dp1;
    thvm_dup(ctx, lab, eql, &dp0, &dp1);
    Term r0 = thvm_reduce(ctx, dp0);
    Term r1 = thvm_reduce(ctx, dp1);
    ASSERT_EQ(term_tag(r0), TAG_NUM, "dup-eql dp0 is NUM");
    ASSERT_EQ(term_as_u32(r0), 1, "dup-eql dp0 = 1");
    ASSERT_EQ(term_tag(r1), TAG_NUM, "dup-eql dp1 is NUM");
    ASSERT_EQ(term_as_u32(r1), 1, "dup-eql dp1 = 1");

    // DUP through ANY: both get ANY
    u32 lab2 = thvm_fresh_label(ctx);
    Term dp2_0, dp2_1;
    thvm_dup(ctx, lab2, thvm_any(), &dp2_0, &dp2_1);
    Term ra = thvm_reduce(ctx, dp2_0);
    Term rb = thvm_reduce(ctx, dp2_1);
    ASSERT_EQ(term_tag(ra), TAG_ANY, "dup-any dp0 = ANY");
    ASSERT_EQ(term_tag(rb), TAG_ANY, "dup-any dp1 = ANY");

    thvm_free(ctx);
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(void) {
    printf("=== SUP Search Machinery Tests ===\n\n");

    // EQL
    test_eql_basic();
    test_eql_any();
    test_eql_sup();

    // AND/OR
    test_and_or_basic();
    test_and_sup();

    // MAT
    test_mat_basic();
    test_mat_sup();

    // Collapse ordered
    test_collapse_ordered();
    test_collapse_ordered_limit();

    // Unordered SUPs
    test_usp_basic();
    test_udp_usp_same_label();
    test_udp_atom();

    // Integration
    test_synthesis_pattern();
    test_synthesis_and_chain();

    // DUP through new nodes
    test_dup_new_nodes();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
