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

// Schedule: dispatch fused kernels in dependency order.
// Each pass has two sub-passes:
//   1. Reduces (SUM/RMAX/RESHAPE) + GRADs — absorbs ew chains into fused kernels
//   2. Standalone ew ops — dispatches remaining ew between reduces (RELU, bias ADD)
// This interleaving ensures: inner reduce → ew (relu/bias) → outer reduce → ...
// View ops resolved by trampoline (non-WNF) during ASSIGN phase.
static u32 sched_all(TinyHVM *ctx) {
    u32 total = 0;
    fuse_no_lazy_resolve = 1;
    ctx->no_grad_alloc = 1; // suppress requires_grad intermediate buffers (memory planner TODO)

    for (u32 pass = 0; pass < 50; pass++) {
        u32 progress = 0;

        // Unified scan: process each TAG_TOP based on type.
        // Forward scan means inner ops (lower positions) resolve first,
        // making their results available as TAG_TEN for outer ops.
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term ht = ctx->heap[h];
            if (term_tag(ht) != TAG_TOP) continue;
            u32 uop = term_ext(ht);
            if (uop == UOP_ASSIGN) continue;

            // GRAD: fire when arg0 (loss) is TAG_TEN
            if (uop == UOP_GRAD) {
                u64 loc = term_val(ht);
                Term a0 = heap_read(ctx, loc);
                if (term_tag(a0) != TAG_TEN) continue;
                Term r = thvm_reduce(ctx, ht);
                ctx->heap[h] = r;
                progress++;
                continue;
            }

            // Reduce patterns: SUM, RMAX, RESHAPE(SUM/RMAX)
            if (uop == UOP_SUM || uop == UOP_RMAX || uop == UOP_RESHAPE) {
                u32 fid = fuse_or_reduce(ctx, ht);
                if (fid != ~0u) {
                    Term result = term_ten(fid, DTYPE_F32);
                    ctx->heap[h] = result;
                    progress++;
                }
                continue;
            }

            // View ops: resolve via thvm_reduce (non-WNF trampoline).
            // EXPAND/PERMUTE create view aliases (no allocation). SHRINK creates sub-views.
            // RESHAPE may contiguify (allocation) — suppress during scheduling.
            if (is_view_op(uop)) {
                // Pre-resolve DP0/DP1 in args (shape params might be DUP-shared)
                u64 vloc = term_val(ht);
                for (u32 ai = 0; ai < 2; ai++) {
                    Term va = heap_read(ctx, vloc + ai);
                    if (term_tag(va) == TAG_DP0 || term_tag(va) == TAG_DP1) {
                        Term shared = heap_read(ctx, term_val(va));
                        if (term_tag(shared) == TAG_TEN || term_tag(shared) == TAG_NUM)
                            heap_set(ctx, vloc + ai, shared);
                    }
                }
                // Suppress contiguify during scheduler view resolution
                ctx->no_fuse = 1; // reuse flag to signal "no materialize"
                Term r = thvm_reduce(ctx, ht);
                ctx->no_fuse = 0;
                if (r != ht && term_tag(r) == TAG_TEN) {
                    ctx->heap[h] = r;
                    progress++;
                }
                continue;
            }

            // Ew ops: dispatch via fuse_or_reduce
            if (is_elementwise(uop)) {
                u32 fid = fuse_or_reduce(ctx, ht);
                if (fid != ~0u) {
                    Term result = term_ten(fid, DTYPE_F32);
                    ctx->heap[h] = result;
                    progress++;
                }
                continue;
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
    // Suppress contiguify: view ops fire (non-WNF) but must not allocate GPU buffers.
    ctx->no_fuse = 1;
    t = thvm_reduce(ctx, t);
    ctx->no_fuse = 0;
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
