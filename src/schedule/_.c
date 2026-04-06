// schedule/_.c — Three-phase eval: reduce → schedule(rewrite) → reduce
//
// Phase 1: thvm_reduce — pure IC. GRAD fires, compute ops stay TAG_TOP.
// Phase 2: sched_all — pure rewrite. TAG_TOPs → UOP_FUSING specs on heap.
//          Each pass: scan heap, convert schedulable ops to UOP_FUSING.
//          Multi-consumer propagation: same term at multiple positions → same FUSING.
// Phase 3: ASSIGN loop → thvm_reduce(assign) → trampoline fires FUSING chain
//          bottom-up → GPU dispatch → TAG_TEN → ASSIGN copies gradient.
//
// UOP_FUSING interact handler deduplicates by kid: same kid fired only once.
//
// NO flags. Compute ops are WNF because interact handler returns t.

int fuse_no_lazy_resolve = 0;

// Global kernel table: scheduler writes, UOP_FUSING handler reads.
KernelEntry sched_kernels[SCHED_MAX_KERNELS];
u32 sched_kernel_count = 0;

// kid_results: TAG_TEN result for each dispatched kid (ERA = not yet dispatched).
// Shared with UOP_FUSING handler in tensor_ops.c.
Term kid_results[SCHED_MAX_KERNELS];

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



// Walk through view ops and DP refs to find the innermost compute/leaf term.
static Term sched_unwrap_views(TinyHVM *ctx, Term t) {
    for (int d = 0; d < 20; d++) {
        if (term_tag(t) == TAG_DP0 || term_tag(t) == TAG_DP1)
            t = heap_read(ctx, term_val(t));
        if (term_tag(t) != TAG_TOP) break;
        if (!is_view_op(term_ext(t))) break;
        t = heap_read(ctx, term_val(t)); // walk through view op's input
    }
    return t;
}

// Schedule a single TAG_TOP as a FUSING kernel. Returns the FUSING term, or ERA on failure.
static Term sched_one(TinyHVM *ctx, Term ht, u64 h) {
    KernelEntry ke; ke.fail_code = 0;
    if (!fuse_build_kernel(ctx, ht, &ke)) {
        if (getenv("THVM_SCHED_DIAG")) {
            u64 _l = term_val(ht);
            Term _a0 = heap_read(ctx, _l);
            fprintf(stderr, "  fuse_fail: %s@%llu arg0=tag%u/%s fc=%d\n",
                uop_names[term_ext(ht)], h, term_tag(_a0),
                term_tag(_a0)==TAG_TOP ? uop_names[term_ext(_a0)] : "", ke.fail_code);
        }
        return term_era();
    }
    ke.original_term = ht;
    u32 kid = sched_kernel_count++;
    sched_kernels[kid] = ke;
    u64 floc = ctx->heap_pos; ctx->heap_pos += 2;
    heap_set(ctx, floc, term_era());
    heap_set(ctx, floc + 1, term_new(TAG_NUM, 0, kid));
    Term ft = term_new(TAG_TOP, UOP_FUSING, floc);
    View out_v = view_create(ke.out_shape);
    st_set(floc, &out_v);
    // Replace at h and propagate
    ctx->heap[h] = ft;
    for (u64 ph = 1; ph < ctx->heap_pos; ph++)
        if (ph != h && ctx->heap[ph] == ht) ctx->heap[ph] = ft;
    // Also replace absorbed ew ops (children that were fused into this kernel)
    extern Term fuse_absorbed[];
    extern u32 fuse_n_absorbed;
    for (u32 ai = 0; ai < fuse_n_absorbed; ai++) {
        Term at = fuse_absorbed[ai];
        for (u64 ph = 1; ph < ctx->heap_pos; ph++)
            if (ctx->heap[ph] == at) ctx->heap[ph] = ft;
    }
    return ft;
}

static u32 sched_all(TinyHVM *ctx) {
    u32 total = 0;
    fuse_no_lazy_resolve = 1;

    for (u32 pass = 0; pass < 50; pass++) {
        u32 progress = 0;
        u32 era_progress = 0;

        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term ht = ctx->heap[h];
            if (term_tag(ht) != TAG_TOP) continue;
            u32 uop = term_ext(ht);
            if (uop == UOP_ASSIGN || uop == UOP_GRAD || uop == UOP_FUSING) continue;

            // View ops: resolve to TAG_TEN alias when input is TAG_TEN.
            if (is_view_op(uop)) {
                u64 vloc = term_val(ht);
                Term vinput = heap_read(ctx, vloc);
                if (term_tag(vinput) == TAG_DP0 || term_tag(vinput) == TAG_DP1)
                    vinput = heap_read(ctx, term_val(vinput));
                if (term_tag(vinput) == TAG_TEN) {
                    u32 src_id = (u32)term_val(vinput);
                    const View *sv = st_get(vloc);
                    if (sv) {
                        u32 vid = tensor_view_of(ctx, src_id, *sv);
                        ctx->tensors[vid].creator_op = (sv->numel != ctx->tensors[src_id].view.numel) ? UOP_RESHAPE : uop;
                        ctx->tensors[vid].src_ids[0] = src_id;
                        Term result = term_ten(vid, DTYPE_F32);
                        ctx->heap[h] = result;
                        progress++;
                        for (u64 ph = 1; ph < ctx->heap_pos; ph++)
                            if (ph != h && ctx->heap[ph] == ht) ctx->heap[ph] = result;
                    }
                }
                continue;
            }

            // ERA propagation for compute ops with ERA inputs.
            {
                u64 eloc = term_val(ht);
                Term ea = heap_read(ctx, eloc);
                u8 ea_tag = term_tag(ea);
                u32 n_args = (uop >= UOP_ADD && uop <= UOP_SUB) || uop == UOP_MM ? 2 :
                             (uop == UOP_SUM || uop == UOP_RMAX) ? 2 :
                             (uop >= UOP_RESHAPE && uop <= UOP_PAD) ? 2 : 1;
                if (ea_tag == TAG_ERA) {
                    if (n_args == 1) {
                        ctx->heap[h] = term_era(); era_progress++; continue;
                    }
                    Term eb = heap_read(ctx, eloc + 1);
                    u8 eb_tag = term_tag(eb);
                    if (uop == UOP_ADD) {
                        if (eb_tag == TAG_TEN || eb_tag == TAG_ERA || eb_tag == TAG_NUM) {
                            ctx->heap[h] = eb; era_progress++; continue;
                        }
                    } else {
                        ctx->heap[h] = term_era(); era_progress++; continue;
                    }
                } else if (n_args == 2) {
                    Term eb = heap_read(ctx, eloc + 1);
                    if (term_tag(eb) == TAG_ERA) {
                        if (uop == UOP_ADD) {
                            if (ea_tag == TAG_TEN || ea_tag == TAG_NUM) {
                                ctx->heap[h] = ea; era_progress++; continue;
                            }
                        } else {
                            ctx->heap[h] = term_era(); era_progress++; continue;
                        }
                    }
                }
            }
        }
        total += era_progress;
        if (era_progress == 0) break;
    }

    // Forward scan: schedule each compute op as a kernel.
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) != TAG_TOP) continue;
        u32 uop = term_ext(ht);
        if (is_view_op(uop) || uop == UOP_ASSIGN || uop == UOP_GRAD || uop == UOP_FUSING) continue;
        sched_one(ctx, ht, h);
    }

    fuse_no_lazy_resolve = 0;
    return total;
}

Term thvm_eval(TinyHVM *ctx, Term t) {
    sched_kernel_count = 0;
    for (u32 i = 0; i < SCHED_MAX_KERNELS; i++) kid_results[i] = term_era();

    // Phase 1: Pure IC reduce — GRAD fires, compute/movement ops stay TAG_TOP (WNF).
    // Binary backward (BG) puts one branch on the heap as a pending GRAD term.
    // Scan and reduce pending GRADs until no more appear.
    t = thvm_reduce(ctx, t);
    for (u32 _gi = 0; _gi < 100; _gi++) {
        int found = 0;
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term ht = ctx->heap[h];
            if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_GRAD) {
                ctx->heap[h] = thvm_reduce(ctx, ht);
                found = 1;
            }
        }
        if (!found) break;
    }
    if (getenv("THVM_SCHED_DIAG")) { fprintf(stderr, "phase1: "); sched_dump_heap(ctx); }

    // Phase 2: schedule compute ops → FUSING, resolve view ops → TAG_TEN aliases.
    sched_all(ctx);
    if (getenv("THVM_SCHED_DIAG")) { fprintf(stderr, "after sched: "); sched_dump_heap(ctx); }

    { Backend *be = ctx_default_backend(ctx);
      if (be && be->end_batch) be->end_batch(); }

    // Phase 3: Fire ASSIGNs. ASSIGN handler resolves WNF view chains
    // inline by dispatching inner FUSINGs and creating view aliases.
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
