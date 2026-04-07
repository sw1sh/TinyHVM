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

// Absorbed terms: ew ops fused into reduce kernels (pass 1). Pass 2 skips these.
#define ABSORBED_MAX 4096
static Term sched_absorbed_terms[ABSORBED_MAX];
static u32  sched_n_absorbed = 0;
static int sched_is_absorbed(Term t) {
    for (u32 i = 0; i < sched_n_absorbed; i++)
        if (sched_absorbed_terms[i] == t) return 1;
    return 0;
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
        // Fallback for reduce ops: create a single-input reduce kernel.
        // The input becomes a lazy leaf, dispatched separately.
        // IMPORTANT: clear stale absorbed entries from the failed walk.
        extern u32 fuse_n_absorbed;
        fuse_n_absorbed = 0;
        u32 top_uop = term_ext(ht);
        if (ke.fail_code == 6 && (top_uop == UOP_SUM || top_uop == UOP_RMAX ||
            (top_uop == UOP_RESHAPE && ke.fail_code == 6))) {
            // Unwrap RESHAPE(SUM) if needed
            Term cur = ht;
            Term reshape_term = term_era();
            if (top_uop == UOP_RESHAPE) {
                Term inner = heap_read(ctx, term_val(cur));
                if (term_tag(inner) == TAG_DP0 || term_tag(inner) == TAG_DP1)
                    inner = heap_read(ctx, term_val(inner));
                if (term_tag(inner) == TAG_TOP &&
                    (term_ext(inner) == UOP_SUM || term_ext(inner) == UOP_RMAX)) {
                    reshape_term = cur; cur = inner;
                } else goto fuse_give_up;
            }
            u32 cur_uop = term_ext(cur);
            u64 sum_loc = term_val(cur);
            Term sum_input = heap_read(ctx, sum_loc);
            if (term_tag(sum_input) == TAG_DP0 || term_tag(sum_input) == TAG_DP1)
                sum_input = heap_read(ctx, term_val(sum_input));
            // Get input shape from shape table
            const View *input_v = NULL;
            if (term_tag(sum_input) == TAG_TOP)
                input_v = st_get(term_val(sum_input));
            else if (term_tag(sum_input) == TAG_TEN)
                input_v = &ctx->tensors[(u32)term_val(sum_input)].view;
            if (!input_v) goto fuse_give_up;
            // Build minimal kernel: one leaf (input), no ops, with reduce
            memset(&ke, 0, sizeof(ke));
            ke.n_ops = 0;
            ke.n_leaves = 1;
            ke.leaf_views[0] = *input_v;
            ke.leaf_terms[0] = sum_input;
            if (term_tag(sum_input) == TAG_TEN)
                ke.leaf_ids[0] = (u32)term_val(sum_input);
            else
                ke.leaf_ids[0] = (u32)(term_val(sum_input) | 0x80000000u);
            ke.full_shape = input_v->shape;
            // Reduce spec from axes tensor
            ke.has_reduce = cur_uop;
            Term sum_axes = heap_read(ctx, sum_loc + 1);
            if (term_tag(sum_axes) == TAG_DP0 || term_tag(sum_axes) == TAG_DP1)
                sum_axes = heap_read(ctx, term_val(sum_axes));
            View out_view = *input_v;
            if (term_tag(sum_axes) == TAG_TEN) {
                u32 ax_id = (u32)term_val(sum_axes);
                TensorMeta *axt = &ctx->tensors[ax_id];
                u32 n_axes = axt->view.numel;
                f32 axes_f[MAX_DIM];
                META_READ(axt->backend, axt->buf_id, axes_f, n_axes * sizeof(f32));
                for (u32 ai = 0; ai < n_axes; ai++) {
                    int ax = (int)axes_f[ai];
                    if (ax >= 0 && ax < (int)input_v->shape.rank) {
                        ke.reduce.is_reduce[ax] = 1;
                        out_view.shape.dims[ax] = 1;
                    }
                }
            } else {
                // No explicit axes: reduce last non-1 dim
                for (int d = (int)input_v->shape.rank - 1; d >= 0; d--)
                    if (input_v->shape.dims[d] > 1) {
                        ke.reduce.is_reduce[d] = 1;
                        out_view.shape.dims[d] = 1; break;
                    }
            }
            ke.reduce.reduce_type = cur_uop;
            out_view.numel = 1;
            for (u32 d = 0; d < out_view.shape.rank; d++) out_view.numel *= out_view.shape.dims[d];
            ke.out_shape = out_view.shape;
            ke.reshape_term = reshape_term;
            ke.sum_term = cur;
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "  fuse_fallback: %s@%llu input_tag=%u\n",
                    uop_names[cur_uop], h, term_tag(sum_input));
            goto fuse_ok;
        }
        // Fallback for elementwise ops: create single-op kernel with lazy leaves.
        if (is_elementwise(top_uop)) {
            u64 eloc = term_val(ht);
            Term ea = heap_read(ctx, eloc);
            if (term_tag(ea) == TAG_DP0 || term_tag(ea) == TAG_DP1)
                ea = heap_read(ctx, term_val(ea));
            const View *va = NULL;
            if (term_tag(ea) == TAG_TOP) va = st_get(term_val(ea));
            else if (term_tag(ea) == TAG_TEN) va = &ctx->tensors[(u32)term_val(ea)].view;
            if (!va) goto fuse_give_up;
            memset(&ke, 0, sizeof(ke));
            ke.n_leaves = 1;
            ke.leaf_views[0] = *va;
            ke.leaf_terms[0] = ea;
            ke.leaf_ids[0] = (term_tag(ea) == TAG_TEN) ? (u32)term_val(ea) :
                             (u32)(term_val(ea) | 0x80000000u);
            int binary = is_binary(top_uop);
            if (binary) {
                Term eb = heap_read(ctx, eloc + 1);
                if (term_tag(eb) == TAG_DP0 || term_tag(eb) == TAG_DP1)
                    eb = heap_read(ctx, term_val(eb));
                const View *vb = NULL;
                if (term_tag(eb) == TAG_TOP) vb = st_get(term_val(eb));
                else if (term_tag(eb) == TAG_TEN) vb = &ctx->tensors[(u32)term_val(eb)].view;
                else if (term_tag(eb) == TAG_NUM) {
                    f32 val = term_as_f32(eb);
                    eb = thvm_tensor(ctx, &val, (Shape){.dims={1},.rank=1});
                    vb = &ctx->tensors[(u32)term_val(eb)].view;
                }
                if (!vb) goto fuse_give_up;
                ke.n_leaves = 2;
                ke.leaf_views[1] = *vb;
                ke.leaf_terms[1] = eb;
                ke.leaf_ids[1] = (term_tag(eb) == TAG_TEN) ? (u32)term_val(eb) :
                                 (u32)(term_val(eb) | 0x80000000u);
            }
            ke.n_ops = 1;
            ke.ops[0] = (FusedOp){.uop = top_uop, .arg_a = 0, .arg_b = binary ? 1 : 0};
            // Output shape = broadcast of leaves
            View av_bc, bv_bc; u32 bc_shape[MAX_DIM], bc_ndim;
            if (binary && ke.n_leaves == 2) {
                if (view_broadcast(&ke.leaf_views[0], &ke.leaf_views[1], &av_bc, &bv_bc, bc_shape, &bc_ndim))
                    ke.full_shape = shape_of(bc_shape, bc_ndim);
                else
                    ke.full_shape = va->shape;
            } else {
                ke.full_shape = va->shape;
            }
            ke.out_shape = ke.full_shape;
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "  fuse_ew_fallback: %s@%llu n_leaves=%u\n",
                    uop_names[top_uop], h, ke.n_leaves);
            goto fuse_ok;
        }
        fuse_give_up:
        return term_era();
    }
    fuse_ok:
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
    // Record absorbed ew ops (pass 2 skips them via sched_is_absorbed).
    extern Term fuse_absorbed[];
    extern u32 fuse_n_absorbed;
    for (u32 ai = 0; ai < fuse_n_absorbed; ai++)
        if (sched_n_absorbed < ABSORBED_MAX)
            sched_absorbed_terms[sched_n_absorbed++] = fuse_absorbed[ai];
    return ft;
}

static u32 sched_all(TinyHVM *ctx) {
    u32 total = 0;
    fuse_no_lazy_resolve = 1;
    sched_n_absorbed = 0;

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

    // Two-pass: reduces first (absorb ew children), then remaining ew.
    // Pass 1: reduce roots (SUM, RMAX, RESHAPE(SUM/RMAX))
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) != TAG_TOP) continue;
        u32 uop = term_ext(ht);
        int is_reduce_root = (uop == UOP_SUM || uop == UOP_RMAX);
        if (!is_reduce_root && uop == UOP_RESHAPE) {
            Term inner = sched_unwrap_views(ctx, ht);
            is_reduce_root = (term_tag(inner) == TAG_TOP &&
                (term_ext(inner) == UOP_SUM || term_ext(inner) == UOP_RMAX));
        }
        if (!is_reduce_root) continue;
        sched_one(ctx, ht, h);
    }
    // Pass 2: remaining compute ops (skip absorbed — they're fused into reduce kernels)
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) != TAG_TOP) continue;
        u32 uop = term_ext(ht);
        if (is_view_op(uop) || uop == UOP_ASSIGN || uop == UOP_GRAD || uop == UOP_FUSING) continue;
        if (sched_is_absorbed(ht)) continue;
        sched_one(ctx, ht, h);
    }

    // Pass 3: schedule absorbed ops that are needed by lazy leaves.
    // RESHAPE boundaries create lazy leaves that may reference ops absorbed
    // by another kernel. Scan all FUSING kernels' lazy leaves and schedule
    // any remaining TAG_TOP ops independently (ignoring absorption).
    for (u32 ki = 0; ki < sched_kernel_count; ki++) {
        KernelEntry *kk = &sched_kernels[ki];
        for (u32 li = 0; li < kk->n_leaves; li++) {
            if (!(kk->leaf_ids[li] == 0 || (kk->leaf_ids[li] & 0x80000000u))) continue;
            // Walk through view chain to find inner term
            Term lt = kk->leaf_terms[li];
            Term inner = lt;
            while (term_tag(inner) == TAG_TOP && is_view_op(term_ext(inner))) {
                Term nx = heap_read(ctx, term_val(inner));
                if (term_tag(nx) == TAG_DP0 || term_tag(nx) == TAG_DP1)
                    nx = heap_read(ctx, term_val(nx));
                inner = nx;
            }
            // If inner is still an unscheduled TAG_TOP, schedule it now
            if (term_tag(inner) == TAG_TOP && term_ext(inner) != UOP_FUSING &&
                !is_view_op(term_ext(inner)) && term_ext(inner) != UOP_ASSIGN &&
                term_ext(inner) != UOP_GRAD) {
                // Find its heap position
                for (u64 h = 1; h < ctx->heap_pos; h++) {
                    if (ctx->heap[h] == inner) {
                        sched_one(ctx, inner, h);
                        break;
                    }
                }
            }
        }
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
