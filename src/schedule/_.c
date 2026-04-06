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
            if (uop == UOP_ASSIGN) continue;
            if (uop == UOP_GRAD)   continue;
            if (uop == UOP_FUSING) continue; // already scheduled; arg0=ERA is intentional
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

            // Compute op: build kernel spec → write UOP_FUSING to heap (pure rewrite).
            {
                KernelEntry ke; ke.fail_code = 0;
                if (!fuse_build_kernel(ctx, ht, &ke)) {
                    if (getenv("THVM_SCHED_DIAG")) {
                        u64 _l = term_val(ht);
                        Term _a0 = heap_read(ctx, _l);
                        fprintf(stderr, "  fuse_fail p%u: %s@%llu arg0=tag%u/%s fc=%d\n",
                            pass, uop_names[uop], h, term_tag(_a0),
                            term_tag(_a0)==TAG_TOP ? uop_names[term_ext(_a0)] : "", ke.fail_code);
                    }
                } else {
                    ke.original_term = ht;
                    u32 kid = sched_kernel_count++;
                    sched_kernels[kid] = ke;
                    u64 floc = ctx->heap_pos; ctx->heap_pos += 2;
                    heap_set(ctx, floc, term_era());
                    heap_set(ctx, floc + 1, term_new(TAG_NUM, 0, kid));
                    ctx->heap[h] = term_new(TAG_TOP, UOP_FUSING, floc);
                    View out_v = view_create(ke.out_shape);
                    st_set(floc, &out_v);
                    progress++;
                    // Propagate: all copies of this term → same FUSING.
                    // Ensures arg slots and DUP nodes see FUSING consistently.
                    { Term ft = ctx->heap[h];
                      for (u64 ph = 1; ph < ctx->heap_pos; ph++)
                          if (ph != h && ctx->heap[ph] == ht) ctx->heap[ph] = ft; }
                }
            }
        }

        total += progress + era_progress;
        if (getenv("THVM_SCHED_DIAG") && (progress > 0 || era_progress > 0))
            fprintf(stderr, "  pass %u: %u kernels %u era\n", pass, progress, era_progress);
        if (progress == 0) break;
    }

    fuse_no_lazy_resolve = 0;
    return total;
}

Term thvm_eval(TinyHVM *ctx, Term t) {
    sched_kernel_count = 0;
    for (u32 i = 0; i < SCHED_MAX_KERNELS; i++) kid_results[i] = term_era();

    // Phase 1: Pure IC reduce — GRAD fires, compute ops stay TAG_TOP (WNF)
    t = thvm_reduce(ctx, t);
    if (getenv("THVM_SCHED_DIAG")) { fprintf(stderr, "phase1: "); sched_dump_heap(ctx); }

    // Phase 2: Pure rewrite — convert all TAG_TOP compute ops to UOP_FUSING specs.
    // FUSING leaves are allowed: phase 3 resolves them bottom-up recursively.
    sched_all(ctx);
    if (getenv("THVM_SCHED_DIAG")) { fprintf(stderr, "after sched: "); sched_dump_heap(ctx); }

    // Flush GPU before phase 3 so forward buffers are committed.
    // pool_reset with keep=tensor_count recycles old buffers without losing live ones.
    { Backend *be = ctx_default_backend(ctx);
      if (be && be->end_batch) be->end_batch(); }

    // Phase 3: Fire ASSIGNs — trampoline reduces src chain, fires FUSING → GPU → TAG_TEN.
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
