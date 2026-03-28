// rewrite/_.c — Declarative pattern-based rewrite rules
//
// Each rule: (match_uop, optional child constraints, rewrite function).
// Rules are tried in order. First match wins.
// A rule returns t unchanged if it doesn't apply (try next rule).

static int rewrite_active = 0; // reentrancy guard

static int is_view_op(u32 uop) {
    return uop == UOP_RESHAPE || uop == UOP_PERMUTE || uop == UOP_EXPAND ||
           uop == UOP_SHRINK || uop == UOP_PAD;
}

#define UOP_ANY 255

typedef struct {
    u32  match_uop;       // UOP to match (or UOP_ANY for wildcard)
    u32  child_uop;       // optional: arg0 must be TAG_TOP with this UOP
    Term (*rewrite)(TinyHVM *ctx, Term t, u64 loc, Term a, Term b);
} RewriteRule;

// ============================================================
// Rule: SUM/RMAX whose child is a lazy elementwise chain
// → dispatch fused reduce+elementwise kernel via fuse_or_reduce
// ============================================================
static Term rule_sum_fuse(TinyHVM *ctx, Term t, u64 loc, Term a, Term b) {
    (void)b; (void)loc;
    if (term_tag(a) == TAG_TOP) {
        u32 child_uop = term_ext(a);
        if (!is_elementwise(child_uop) && !is_view_op(child_uop)) return t;
    } else if (term_tag(a) == TAG_TEN) {
        // Deferred child: materialize it (fuses the elementwise chain),
        // then let the SUM dispatch normally on the materialized result.
        u32 tid = (u32)term_val(a);
        if (tid < ctx->tensor_count && ctx->tensors[tid].buf_id == 0 &&
            ctx->tensors[tid].creator_op) {
            tensor_materialize(ctx, tid);
        }
        return t; // SUM will dispatch normally on the now-materialized input
    } else {
        return t;
    }
    rewrite_active = 1;
    u32 fid = fuse_or_reduce(ctx, t);
    rewrite_active = 0;
    if (fid != ~0u) {
        return term_ten(fid, DTYPE_F32);
    }
    return t;
}


// ============================================================
// Rule: elementwise chain (2+ ops) → fuse into one kernel
// ============================================================
static Term rule_elementwise_fuse(TinyHVM *ctx, Term t, u64 loc, Term a, Term b) {
    (void)loc;
    int chain = 0;
    if (term_tag(a) == TAG_TOP &&
        (is_elementwise(term_ext(a)) || is_view_op(term_ext(a))))
        chain = 1;
    if (term_tag(b) != TAG_ERA && term_tag(b) == TAG_TOP &&
        (is_elementwise(term_ext(b)) || is_view_op(term_ext(b))))
        chain = 1;
    if (!chain) return t;
    rewrite_active = 1;
    u32 fid = fuse_or_reduce(ctx, t);
    rewrite_active = 0;
    if (fid != ~0u) return term_ten(fid, DTYPE_F32);
    return t;
}

// ============================================================
// Rule table
// ============================================================
// Rule: RESHAPE(SUM/RMAX(ew_chain)) → fuse_or_reduce handles this pattern
static Term rule_reshape_reduce_fuse(TinyHVM *ctx, Term t, u64 loc, Term a, Term b) {
    (void)b; (void)loc;
    if (term_tag(a) != TAG_TOP) return t;
    u32 inner_uop = term_ext(a);
    if (inner_uop != UOP_SUM && inner_uop != UOP_RMAX) return t;
    // Check if the SUM/RMAX's child is elementwise/view
    u64 sum_loc = term_val(a);
    Term sum_input = heap_read(ctx, sum_loc);
    if (term_tag(sum_input) != TAG_TOP) return t;
    if (!is_elementwise(term_ext(sum_input)) && !is_view_op(term_ext(sum_input))) return t;
    rewrite_active = 1;
    u32 fid = fuse_or_reduce(ctx, t);
    rewrite_active = 0;
    if (fid != ~0u) return term_ten(fid, DTYPE_F32);
    return t;
}

static RewriteRule rewrite_rules[] = {
    // RESHAPE(SUM/RMAX(ew_chain)) — must be before SUM/RMAX rules
    { UOP_RESHAPE, UOP_ANY, rule_reshape_reduce_fuse },
    // Reduce + elementwise fusion (SUM(ew_chain), RMAX(ew_chain))
    { UOP_SUM,  UOP_ANY, rule_sum_fuse },
    { UOP_RMAX, UOP_ANY, rule_sum_fuse },
    // Elementwise chain fusion
    { UOP_ADD,  UOP_ANY, rule_elementwise_fuse },
    { UOP_SUB,  UOP_ANY, rule_elementwise_fuse },
    { UOP_MUL,  UOP_ANY, rule_elementwise_fuse },
    { UOP_DIV,  UOP_ANY, rule_elementwise_fuse },
    { UOP_NEG,  UOP_ANY, rule_elementwise_fuse },
    { UOP_RELU, UOP_ANY, rule_elementwise_fuse },
    { UOP_EXP,  UOP_ANY, rule_elementwise_fuse },
    { UOP_LOG,  UOP_ANY, rule_elementwise_fuse },
    { UOP_SQRT, UOP_ANY, rule_elementwise_fuse },
    { UOP_MAX,  UOP_ANY, rule_elementwise_fuse },
    { UOP_CMP,  UOP_ANY, rule_elementwise_fuse },
};
#define N_REWRITE_RULES (sizeof(rewrite_rules) / sizeof(rewrite_rules[0]))

// Try all matching rewrite rules. Returns rewritten term, or t if none matched.
static Term rewrite_apply(TinyHVM *ctx, Term t) {
    if (rewrite_active) return t;
    if (term_tag(t) != TAG_TOP) return t;
    u32 uop = term_ext(t);
    u64 loc = term_val(t);
    Term a = heap_read(ctx, loc);
    Term b = heap_read(ctx, loc + 1);

    for (u32 i = 0; i < N_REWRITE_RULES; i++) {
        RewriteRule *r = &rewrite_rules[i];
        if (r->match_uop != UOP_ANY && r->match_uop != uop) continue;
        if (r->child_uop != UOP_ANY && term_tag(a) == TAG_TOP &&
            term_ext(a) != r->child_uop) continue;
        Term result = r->rewrite(ctx, t, loc, a, b);
        if (result != t) return result; // rule matched and rewrote
    }
    return t; // no rule matched
}
