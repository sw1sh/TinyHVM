// schedule/_.c — Three-phase eval: reduce → schedule(dispatch) → ASSIGNs
//
// NO flags (defer_all, dispatch_mode, no_fuse) — pure graph walk + dispatch.
// View ops handled by fuse_walk_inner (EXPAND/PERMUTE/RESHAPE composition).
// SHRINK/PAD are lazy leaf boundaries in fuse_walk_inner.

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

// Schedule: dispatch fused kernels via forward heap scan.
// Each pass: try fuse_or_reduce on every SUM/RMAX/RESHAPE/ew TAG_TOP.
// fuse_walk_inner handles view composition (EXPAND/PERMUTE/RESHAPE).
// SHRINK/PAD are lazy leaf boundaries → multi-pass resolves layer by layer.
// GRAD fires via thvm_reduce when arg0 is TAG_TEN.
static u32 sched_all(TinyHVM *ctx) {
    u32 total = 0;
    fuse_no_lazy_resolve = 1;
    ctx->no_grad_alloc = 1;

    for (u32 pass = 0; pass < 50; pass++) {
        u32 progress = 0;

        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term ht = ctx->heap[h];
            if (term_tag(ht) != TAG_TOP) continue;
            u32 uop = term_ext(ht);
            if (uop == UOP_ASSIGN) continue;

            // GRAD: fire when arg0 ready
            if (uop == UOP_GRAD) {
                u64 loc = term_val(ht);
                Term a0 = heap_read(ctx, loc);
                if (term_tag(a0) != TAG_TEN) continue;
                Term r = thvm_reduce(ctx, ht);
                ctx->heap[h] = r;
                progress++;
                continue;
            }

            // Skip view ops — handled by fuse_walk_inner inside fuse_or_reduce.
            // SHRINK/PAD become lazy leaf boundaries (resolved when input is TAG_TEN).
            if (is_view_op(uop)) continue;

            // All compute ops (reduce + ew): dispatch via fuse_or_reduce
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
    ctx->no_grad_alloc = 0;
    return total;
}

Term thvm_eval(TinyHVM *ctx, Term t) {
    // Phase 1: Pure IC reduce — everything lazy.
    // no_fuse=1: ALL compute+view TAG_TOPs are WNF (no TensorMeta, no dispatch).
    ctx->no_fuse = 1;
    t = thvm_reduce(ctx, t);
    ctx->no_fuse = 0;
    if (getenv("THVM_SCHED_DIAG")) { fprintf(stderr, "phase1: "); sched_dump_heap(ctx); }

    // Phase 2: Schedule ALL kernels (forward + GRAD + backward)
    u32 n = sched_all(ctx);
    if (getenv("THVM_SCHED_DIAG")) { fprintf(stderr, "phase2(%u total): ", n); sched_dump_heap(ctx); }

    // Phase 3: Fire ASSIGNs via thvm_reduce (view ops fire via interact handler)
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
