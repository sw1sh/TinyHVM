// schedule/_.c — Three-phase eval: reduce → schedule → reduce

// Debug: dump TAG_TOP stats on the heap after first reduce
static void sched_dump_heap(TinyHVM *ctx) {
    u32 counts[UOP_COUNT] = {0};
    u32 n_ten = 0, n_era = 0, n_app = 0, n_top = 0, n_other = 0;
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term t = ctx->heap[h];
        u8 tag = term_tag(t);
        if (tag == TAG_TEN) n_ten++;
        else if (tag == TAG_ERA) n_era++;
        else if (tag == TAG_APP) n_app++;
        else if (tag == TAG_TOP) { n_top++; u32 uop = term_ext(t); if (uop < UOP_COUNT) counts[uop]++; }
        else n_other++;
    }
    fprintf(stderr, "HEAP[%llu]: TEN=%u ERA=%u APP=%u TOP=%u other=%u\n",
        ctx->heap_pos, n_ten, n_era, n_app, n_top, n_other);
    fprintf(stderr, "  TOPs: ");
    for (u32 u = 0; u < UOP_COUNT; u++)
        if (counts[u]) fprintf(stderr, "%s=%u ", uop_names[u], counts[u]);
    fprintf(stderr, "\n");
}

Term thvm_eval(TinyHVM *ctx, Term t) {
    // Phase 1: Pure IC reduce — everything lazy
    t = thvm_reduce(ctx, t);

    if (getenv("THVM_SCHED_DIAG")) {
        fprintf(stderr, "EVAL phase 1 done: t tag=%u\n", term_tag(t));
        sched_dump_heap(ctx);
    }

    // Phase 2: TODO — scheduler walks TAG_TOP graph, builds UOP_FUSING,
    // creates new reducible graph with ASSIGN → UOP_FUSING connections.
    // For now: just return. Caller must handle unreduced state.

    return t;
}

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_eval(ctx, t);
}
