// schedule/_.c — Three-phase eval: reduce → schedule(rewrite) → reduce

int fuse_no_lazy_resolve = 0;

static void sched_dump_heap(TinyHVM *ctx) {
    u32 counts[UOP_COUNT] = {0};
    u32 n_ten = 0, n_top = 0;
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term t = ctx->heap[h];
        if (term_tag(t) == TAG_TEN) n_ten++;
        else if (term_tag(t) == TAG_TOP) { n_top++; u32 uop = term_ext(t); if (uop < UOP_COUNT) counts[uop]++; }
    }
    fprintf(stderr, "HEAP[%llu]: TEN=%u TOP=%u | ", ctx->heap_pos, n_ten, n_top);
    for (u32 u = 0; u < UOP_COUNT; u++) if (counts[u]) fprintf(stderr, "%s=%u ", uop_names[u], counts[u]);
    fprintf(stderr, "\n");
}

// Schedule: walk ALL TAG_TOPs on heap, dispatch fused kernels.
// Multi-pass: each pass dispatches kernels whose leaves are ready (TAG_TEN).
// Inner kernels dispatch first → their TAG_TEN results become leaves for outer kernels.
// GRAD is included — it's just another TAG_TOP that fires when its args are ready.
static u32 sched_all(TinyHVM *ctx) {
    u32 total = 0;
    fuse_no_lazy_resolve = 1;
    for (u32 pass = 0; pass < 50; pass++) {
        u32 progress = 0;
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term ht = ctx->heap[h];
            if (term_tag(ht) != TAG_TOP) continue;
            u32 uop = term_ext(ht);
            // Skip ASSIGN (fired after all kernels dispatch)
            if (uop == UOP_ASSIGN) continue;
            // GRAD: reduce it (fires GRAD handler which creates backward TAG_TOPs)
            if (uop == UOP_GRAD) {
                // Only fire if arg0 (loss) is now TAG_TEN (forward dispatched)
                u64 loc = term_val(ht);
                Term a0 = heap_read(ctx, loc);
                if (term_tag(a0) != TAG_TEN) continue; // not ready yet
                Term r = thvm_reduce(ctx, ht);
                ctx->heap[h] = r;
                progress++;
                continue;
            }
            // Compute TAG_TOPs: try fuse_or_reduce
            u32 fid = fuse_or_reduce(ctx, ht);
            if (fid != ~0u) {
                ctx->heap[h] = term_ten(fid, DTYPE_F32);
                progress++;
            }
        }
        total += progress;
        if (getenv("THVM_SCHED_DIAG") && progress > 0)
            fprintf(stderr, "  pass %u: %u dispatched\n", pass, progress);
        if (progress == 0) break;
    }
    fuse_no_lazy_resolve = 0;
    return total;
}

Term thvm_eval(TinyHVM *ctx, Term t) {
    // Phase 1: Pure IC reduce — everything lazy
    t = thvm_reduce(ctx, t);
    if (getenv("THVM_SCHED_DIAG")) { fprintf(stderr, "phase1: "); sched_dump_heap(ctx); }

    // Phase 2: Schedule ALL kernels (forward + GRAD + backward)
    u32 n = sched_all(ctx);
    if (getenv("THVM_SCHED_DIAG")) { fprintf(stderr, "phase2(%u total): ", n); sched_dump_heap(ctx); }

    // Phase 3: Fire ASSIGNs
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_ASSIGN) {
            Term r = thvm_reduce(ctx, ht);
            ctx->heap[h] = r;
        }
    }
    if (getenv("THVM_SCHED_DIAG")) { fprintf(stderr, "phase3: "); sched_dump_heap(ctx); }

    return term_era();
}

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_eval(ctx, t);
}
