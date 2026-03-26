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

Term thvm_sup(TinyHVM *ctx, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     a);
    heap_set(ctx, loc + 1, b);
    return term_new(TAG_SUP, 0, loc);
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
