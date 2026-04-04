// schedule/_.c — Heap-based scheduler

// Find ASSIGN TAG_TOPs reachable from root.
static u32 sched_find_assigns(TinyHVM *ctx, Term t, u64 *locs, u32 max) {
    u32 n = 0;
    u64 stk[512]; u32 sp = 0;
    stk[sp++] = (u64)t;
    while (sp > 0 && n < max) {
        Term cur = (Term)stk[--sp];
        u8 tag = term_tag(cur);
        if (tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM) continue;
        if (tag == TAG_TOP) {
            u64 loc = term_val(cur);
            if (term_ext(cur) == UOP_ASSIGN) { locs[n++] = loc; continue; }
            if (sp < 510) { stk[sp++] = (u64)heap_read(ctx, loc); stk[sp++] = (u64)heap_read(ctx, loc+1); }
            continue;
        }
        if (tag == TAG_APP || tag == TAG_SUP) {
            u64 loc = term_val(cur);
            if (sp < 510) { stk[sp++] = (u64)heap_read(ctx, loc); stk[sp++] = (u64)heap_read(ctx, loc+1); }
        }
    }
    return n;
}

Term thvm_eval(TinyHVM *ctx, Term t) {
    // Phase 1: Reduce — compute TAG_TOPs stay WNF, GRAD fires
    t = thvm_reduce(ctx, t);

    // Phase 2: Find ASSIGNs, dispatch their src trees
    u64 assign_locs[128]; u32 na = sched_find_assigns(ctx, t, assign_locs, 128);
    if (getenv("THVM_SCHED_DIAG"))
        fprintf(stderr, "SCHED: %u assigns\n", na);

    for (u32 i = 0; i < na; i++) {
        Term src = heap_read(ctx, assign_locs[i] + 1);
        if (term_tag(src) != TAG_TOP) continue;
        // fuse_or_reduce walks TAG_TOP tree: ew ops → FusedOps, non-ew → lazy leaves.
        // Lazy leaves get thvm_reduce'd internally. But with lazy TAG_TOPs as WNF,
        // thvm_reduce returns them unchanged. So fuse_or_reduce falls back to reduce_id
        // which also returns unchanged.
        //
        // The fix: temporarily make compute TAG_TOPs NOT WNF so the trampoline
        // processes them normally during fuse_or_reduce's lazy leaf resolution.
        // We do this by setting a flag that the trampoline checks.
        ctx->dispatch_mode = 1;
        u32 fid = fuse_or_reduce(ctx, src);
        if (fid != ~0u) {
            heap_set(ctx, assign_locs[i] + 1, term_ten(fid, DTYPE_F32));
        } else {
            Term r = thvm_reduce(ctx, src);
            if (term_tag(r) == TAG_TEN)
                heap_set(ctx, assign_locs[i] + 1, r);
        }
        ctx->dispatch_mode = 0;
    }

    // Phase 3: Fire ASSIGNs
    ctx->dispatch_mode = 1;
    for (u32 i = 0; i < na; i++) {
        Term src = heap_read(ctx, assign_locs[i] + 1);
        Term dst = heap_read(ctx, assign_locs[i]);
        if (term_tag(src) == TAG_TEN && term_tag(dst) == TAG_TEN) {
            Term assign = term_new(TAG_TOP, UOP_ASSIGN, assign_locs[i]);
            thvm_reduce(ctx, assign);
        }
    }
    ctx->dispatch_mode = 0;

    return term_era();
}

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_eval(ctx, t);
}
