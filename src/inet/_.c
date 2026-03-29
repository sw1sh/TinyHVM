// inet/_.c — Interaction combinator term constructors

Term thvm_lam(TinyHVM *ctx, Term *var_out, Term body) {
    u64 loc = heap_alloc(ctx, 2);
    Term var = term_new(TAG_VAR, 0, loc);
    heap_set(ctx, loc,     term_set_sub(var));
    heap_set(ctx, loc + 1, body);
    if (var_out) *var_out = var;
    return term_new(TAG_LAM, 0, loc);
}

Term thvm_app(TinyHVM *ctx, Term fun, Term arg) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     fun);
    heap_set(ctx, loc + 1, arg);
    return term_new(TAG_APP, 0, loc);
}

Term thvm_sup(TinyHVM *ctx, u32 label, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     a);
    heap_set(ctx, loc + 1, b);
    return term_new(TAG_SUP, label, loc);
}

// DUP: split a term into two projections (DP0, DP1) with a label.
// heap[loc] = value. DP0/DP1 reduce the value, then fire DUP interaction rules.
// Label determines annihilation (same label SUP) vs commutation (different label).
void thvm_dup(TinyHVM *ctx, u32 label, Term z, Term *out0, Term *out1) {
    u64 loc = heap_alloc(ctx, 1);
    heap_set(ctx, loc, z);
    *out0 = term_new(TAG_DP0, label, loc);
    *out1 = term_new(TAG_DP1, label, loc);
}

// Allocate a fresh label (monotonic counter). Only call at search-space construction
// time — interaction rules propagate existing labels, never create fresh ones.
u32 thvm_fresh_label(TinyHVM *ctx) {
    return ctx->next_sup_label++;
}

u32 thvm_define(TinyHVM *ctx, Term body) {
    assert(ctx->def_count < 256);
    u32 name = ctx->def_count++;
    ctx->defs[name] = body;
    return name;
}

Term thvm_ref(TinyHVM *ctx, u32 name) {
    (void)ctx;
    return term_new(TAG_REF, name, 0);
}

Term thvm_where(TinyHVM *ctx, Term cond, Term then_t, Term else_t) {
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     cond);
    heap_set(ctx, loc + 1, then_t);
    heap_set(ctx, loc + 2, else_t);
    return term_new(TAG_TOP, UOP_WHERE, loc);
}

Term thvm_assign(TinyHVM *ctx, Term dst, Term src) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     dst);
    heap_set(ctx, loc + 1, src);
    return term_new(TAG_TOP, UOP_ASSIGN, loc);
}

Term thvm_ifz(TinyHVM *ctx, Term counter, Term zero_case, Term succ_lam) {
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     counter);
    heap_set(ctx, loc + 1, zero_case);
    heap_set(ctx, loc + 2, succ_lam);
    return term_new(TAG_TOP, UOP_IFZ, loc);
}

Term thvm_log_print(TinyHVM *ctx, Term tensor) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     tensor);
    heap_set(ctx, loc + 1, term_era());
    return term_new(TAG_TOP, UOP_LOG_PRINT, loc);
}

// BRI: bridge (dual of lambda — contra-variant binding for ICC types)
Term thvm_bri(TinyHVM *ctx, Term *var_out, Term body) {
    u64 loc = heap_alloc(ctx, 2);
    Term var = term_new(TAG_VAR, 0, loc);
    heap_set(ctx, loc,     term_set_sub(var));
    heap_set(ctx, loc + 1, body);
    if (var_out) *var_out = var;
    return term_new(TAG_BRI, 0, loc);
}

// ANN: annotation {term : type}
Term thvm_ann(TinyHVM *ctx, Term term, Term type) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     term);
    heap_set(ctx, loc + 1, type);
    return term_new(TAG_ANN, 0, loc);
}

// DSU: dynamic superposition — label is an expression reduced at interaction time
Term thvm_dsu(TinyHVM *ctx, Term label_expr, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     label_expr);
    heap_set(ctx, loc + 1, a);
    heap_set(ctx, loc + 2, b);
    return term_new(TAG_DSU, 0, loc);
}

// DDU: dynamic dup — label is an expression, reduces then clones val
Term thvm_ddu(TinyHVM *ctx, Term label_expr, Term val, Term bod) {
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     label_expr);
    heap_set(ctx, loc + 1, val);
    heap_set(ctx, loc + 2, bod);
    return term_new(TAG_DDU, 0, loc);
}

// INC: priority wrapper — transparent to reduce, lower priority in collapse
Term thvm_inc(TinyHVM *ctx, Term term) {
    u64 loc = heap_alloc(ctx, 1);
    heap_set(ctx, loc, term);
    return term_new(TAG_INC, 0, loc);
}

// Collapse: DFS that extracts all SUP branches into a flat array
static void collapse_dfs(TinyHVM *ctx, Term t, CollapseResult *cr) {
    Term reduced = thvm_reduce(ctx, t);
    if (term_tag(reduced) == TAG_SUP) {
        u64 loc = term_val(reduced);
        collapse_dfs(ctx, heap_read(ctx, loc + 0), cr);
        collapse_dfs(ctx, heap_read(ctx, loc + 1), cr);
    } else {
        if (cr->count >= cr->cap) {
            cr->cap = cr->cap ? cr->cap * 2 : 16;
            cr->terms = realloc(cr->terms, cr->cap * sizeof(Term));
        }
        cr->terms[cr->count++] = reduced;
    }
}

CollapseResult thvm_collapse(TinyHVM *ctx, Term t) {
    CollapseResult cr = {NULL, 0, 0};
    collapse_dfs(ctx, t, &cr);
    return cr;
}

void thvm_collapse_free(CollapseResult *cr) {
    if (cr->terms) { free(cr->terms); cr->terms = NULL; }
    cr->count = cr->cap = 0;
}

// OP2: binary integer operation on TAG_NUM values.
// opr: 0=add, 1=sub, 2=mul, 3=div, 4=eq, 5=mod
Term thvm_op2(TinyHVM *ctx, u32 opr, Term x, Term y) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     x);
    heap_set(ctx, loc + 1, y);
    return term_new(TAG_OP2, opr, loc);
}
