// schedule/_.c — Three-phase eval: reduce → schedule(rewrite) → reduce
//
// Phase 1: thvm_reduce — pure IC. GRAD fires, compute ops stay TAG_TOP.
// Phase 2: sched_all — pure rewrite. TAG_TOPs → dispatched TAG_TENs on heap.
//          (Currently dispatches directly via fuse_or_reduce.
//           TODO: rewrite to UOP_FUSING specs, dispatch in phase 3.)
// Phase 3: thvm_reduce — ASSIGNs fire. View ops resolve via interact handler.
//
// NO flags. Compute ops are WNF because interact handler returns t.

int fuse_no_lazy_resolve = 0;

// Global kernel table: scheduler writes, UOP_FUSING handler reads.
KernelEntry sched_kernels[SCHED_MAX_KERNELS];
u32 sched_kernel_count = 0;

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

// Phase 2: Walk TAG_TOPs on heap, dispatch fused kernels.
// Currently dispatches directly via fuse_or_reduce.
// TODO: replace with pure rewrite (fuse_build_kernel → UOP_FUSING on heap).
static u32 sched_all(TinyHVM *ctx) {
    u32 total = 0;
    fuse_no_lazy_resolve = 1;

    for (u32 pass = 0; pass < 50; pass++) {
        u32 progress = 0;

        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term ht = ctx->heap[h];
            if (term_tag(ht) != TAG_TOP) continue;
            u32 uop = term_ext(ht);
            if (uop == UOP_ASSIGN) continue;

            // GRAD: fire when arg0 ready (pure IC — creates backward TAG_TOPs)
            if (uop == UOP_GRAD) {
                u64 loc = term_val(ht);
                Term a0 = heap_read(ctx, loc);
                if (term_tag(a0) != TAG_TEN) continue;
                Term r = thvm_reduce(ctx, ht);
                ctx->heap[h] = r;
                progress++;
                continue;
            }

            // View ops: pool RESHAPEs resolved when input is TAG_TEN.
            // Other views composed by fuse_walk_inner (no dispatch needed).
            if (is_view_op(uop)) {
                u64 vloc = term_val(ht);
                Term vinput = heap_read(ctx, vloc);
                if (term_tag(vinput) == TAG_TEN) {
                    u32 src_id = (u32)term_val(vinput);
                    const View *sv = st_get(vloc);
                    if (sv && sv->numel != ctx->tensors[src_id].view.numel) {
                        u32 vid = tensor_view_of(ctx, src_id, *sv);
                        ctx->tensors[vid].creator_op = UOP_RESHAPE;
                        ctx->tensors[vid].src_ids[0] = src_id;
                        ctx->heap[h] = term_ten(vid, DTYPE_F32);
                        progress++;
                    }
                }
                continue;
            }

            // Compute ops: build kernel spec → write UOP_FUSING to heap.
            // Pure rewrite — no dispatch, no buf_alloc, no TensorMeta.
            {
                KernelEntry ke;
                if (fuse_build_kernel(ctx, ht, &ke)) {
                    u32 kid = sched_kernel_count++;
                    sched_kernels[kid] = ke;
                    // Allocate heap space for UOP_FUSING node
                    u64 floc = ctx->heap_pos; ctx->heap_pos += 2;
                    heap_set(ctx, floc, term_era());              // arg0 (triggers immediate fire)
                    heap_set(ctx, floc + 1, term_new(TAG_NUM, 0, kid));  // arg1 = kernel index
                    ctx->heap[h] = term_new(TAG_TOP, UOP_FUSING, floc);
                    // Store output shape so later fuse_build_kernel sees this as a ready leaf
                    View out_v = view_create(ke.out_shape);
                    st_set(floc, &out_v);
                    progress++;
                }
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

// Dispatch all UOP_FUSING on the heap → TAG_TEN
static void sched_dispatch_fusing(TinyHVM *ctx) {
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_FUSING) {
            Term r = thvm_reduce(ctx, ht);
            ctx->heap[h] = r;
        }
    }
}

Term thvm_eval(TinyHVM *ctx, Term t) {
    sched_kernel_count = 0;

    // Phase 1: Pure IC reduce — GRAD fires, compute+view ops stay TAG_TOP (WNF)
    t = thvm_reduce(ctx, t);
    if (getenv("THVM_SCHED_DIAG")) { fprintf(stderr, "phase1: "); sched_dump_heap(ctx); }

    // Phase 2+3 loop: schedule (rewrite) → dispatch (reduce) → repeat.
    // Each iteration: schedule inner kernels → dispatch them → next level.
    for (u32 iter = 0; iter < 50; iter++) {
        u32 n = sched_all(ctx);
        if (n == 0) break;
        if (getenv("THVM_SCHED_DIAG"))
            fprintf(stderr, "  sched iter %u: %u kernels\n", iter, n);
        sched_dispatch_fusing(ctx);
    }
    if (getenv("THVM_SCHED_DIAG")) { fprintf(stderr, "after sched: "); sched_dump_heap(ctx); }

    // Final: fire ASSIGNs
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_ASSIGN) {
            Term r = thvm_reduce(ctx, ht);
            ctx->heap[h] = r;
        }
    }
    if (getenv("THVM_SCHED_DIAG")) { fprintf(stderr, "after assign: "); sched_dump_heap(ctx); }

    return term_era();
}

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_eval(ctx, t);
}
