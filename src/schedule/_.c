// Forward declarations (defined in debug/graph.c + debug/dump.c, included after this file)
static void thvm_heap_dot(TinyHVM *ctx, const char *path);
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root);

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
int _assign_dispatch_enabled = 0;

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

// Absorbed terms: alias to shared registry in fuse/_.c
#define sched_absorbed_terms _sched_absorbed
#define sched_n_absorbed _sched_n_absorbed
#define sched_is_absorbed _sched_is_absorbed
#define ABSORBED_MAX SCHED_ABSORBED_MAX

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
                    goto fuse_give_up; // incompatible shapes — can't create ew kernel
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
    // Record absorbed ew ops AND absorbed reduce/reshape terms.
    // The fuse_walk_inner checks sched_is_absorbed before absorbing a reduce,
    // preventing double-absorption through DUP.
    extern Term fuse_absorbed[];
    extern u32 fuse_n_absorbed;
    for (u32 ai = 0; ai < fuse_n_absorbed; ai++)
        if (sched_n_absorbed < ABSORBED_MAX)
            sched_absorbed_terms[sched_n_absorbed++] = fuse_absorbed[ai];
    // Record absorbed reduce + reshape terms
    if (term_tag(ke.sum_term) == TAG_TOP)
        if (sched_n_absorbed < ABSORBED_MAX)
            sched_absorbed_terms[sched_n_absorbed++] = ke.sum_term;
    if (term_tag(ke.reshape_term) == TAG_TOP)
        if (sched_n_absorbed < ABSORBED_MAX)
            sched_absorbed_terms[sched_n_absorbed++] = ke.reshape_term;
    // Note: we do NOT replace DUP shared locations on heap — that crashes the
    // IC reducer. Instead, _sched_is_absorbed prevents double absorption.
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

    // Pass 1: reduces without ew consumers
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
        // Check for ew consumer
        int has_ew_consumer = 0;
        for (u64 ch = 1; ch < ctx->heap_pos && !has_ew_consumer; ch++) {
            Term ct = ctx->heap[ch];
            if (term_tag(ct)!=TAG_TOP || !is_elementwise(term_ext(ct))) continue;
            u64 cloc = term_val(ct);
            for (u32 ai = 0; ai < 2 && !has_ew_consumer; ai++) {
                Term ca = heap_read(ctx, cloc + ai);
                if (term_tag(ca)==TAG_DP0||term_tag(ca)==TAG_DP1) ca = heap_read(ctx, term_val(ca));
                if (ca == ht) { has_ew_consumer = 1; break; }
                while (term_tag(ca)==TAG_TOP && is_view_op(term_ext(ca))) {
                    Term nx = heap_read(ctx, term_val(ca));
                    if (term_tag(nx)==TAG_DP0||term_tag(nx)==TAG_DP1) nx = heap_read(ctx, term_val(nx));
                    if (nx == ht) { has_ew_consumer = 1; break; }
                    ca = nx;
                }
            }
        }
        if (has_ew_consumer) {
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "  skip_reduce_for_ew: %s@%llu\n",
                        uop < UOP_COUNT ? uop_names[uop] : "?", (unsigned long long)h);
            continue;
        }
        sched_one(ctx, ht, h);
    }
    // Pass 2: ew ops ONLY — their walks absorb unscheduled reduces.
    // CRITICAL: do NOT schedule SUM/RMAX here. They must remain TAG_TOP
    // so that ew walks can find and absorb them via _fuse_can_absorb_reduce.
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) != TAG_TOP) continue;
        u32 uop = term_ext(ht);
        if (!is_elementwise(uop)) continue;
        if (sched_is_absorbed(ht)) continue;
        sched_one(ctx, ht, h);
    }
    // Pass 3: remaining reduces (not absorbed)
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
        if (sched_is_absorbed(ht)) continue;
        sched_one(ctx, ht, h);
    }
    // Pass 3: anything remaining
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

    // CSE: deduplicate kernels that compute the same thing.
    // DUP in the IC graph creates identical reduce/ew terms on separate branches.
    // Match by: ops structure + leaf equivalence (concrete IDs or lazy→FUSING kid).
    // Run multiple rounds: deduping a leaf kernel may make dependent kernels match.
    {
        u32 n_deduped = 0;
        for (u32 _cse_round = 0; _cse_round < 10; _cse_round++) {
        u32 round_deduped = 0;
        // Helper: resolve a leaf to a canonical key for comparison.
        // Concrete leaf → tensor ID. Lazy FUSING leaf → kernel kid. Lazy TAG_TOP → heap loc.
        #define LEAF_KEY(ke, li) ({ \
            u32 _lid = (ke)->leaf_ids[li]; u32 _key = _lid; \
            if (_lid == 0 || LEAF_IS_LAZY(_lid)) { \
                Term _lt = (ke)->leaf_terms[li]; \
                for (int _rd = 0; _rd < 20; _rd++) { \
                    if (term_tag(_lt)==TAG_DP0||term_tag(_lt)==TAG_DP1) \
                        { _lt = heap_read(ctx, term_val(_lt)); continue; } \
                    if (term_tag(_lt)==TAG_TOP && is_view_op(term_ext(_lt))) \
                        { Term _nx = heap_read(ctx, term_val(_lt)); _lt = _nx; continue; } \
                    break; \
                } \
                if (term_tag(_lt)==TAG_TOP && term_ext(_lt)==UOP_FUSING) { \
                    u64 _fl = term_val(_lt); Term _kt = heap_read(ctx, _fl+1); \
                    if (term_tag(_kt)==TAG_NUM) { \
                        u32 _kid = (u32)term_val(_kt); \
                        while (_kid < sched_kernel_count && canonical[_kid] != _kid) _kid = canonical[_kid]; \
                        _key = 0xC0000000u|_kid; \
                    } \
                } else _key = (u32)(term_val(_lt) | 0x80000000u); \
            } \
            _key; })

        // Dedup map: canonical[ki] = canonical kid for ki (follows dedup chain)
        u32 canonical[SCHED_MAX_KERNELS];
        for (u32 ki = 0; ki < sched_kernel_count; ki++) canonical[ki] = ki;

        // Find ka's FUSING term on heap (rebuild each CSE round)
        Term fusing_terms[SCHED_MAX_KERNELS];
        for (u32 ki = 0; ki < sched_kernel_count; ki++) {
            fusing_terms[ki] = term_era();
            for (u64 h = 1; h < ctx->heap_pos; h++) {
                Term ht = ctx->heap[h];
                if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_FUSING) {
                    u64 floc = term_val(ht);
                    Term kid_t = heap_read(ctx, floc + 1);
                    if (term_tag(kid_t) == TAG_NUM && (u32)term_val(kid_t) == ki) {
                        fusing_terms[ki] = ht; break;
                    }
                }
            }
        }

        for (u32 ki = 0; ki < sched_kernel_count; ki++) {
            KernelEntry *ka = &sched_kernels[ki];
            if (ka->n_ops == 0 && ka->n_leaves == 0) continue;
            for (u32 kj = ki + 1; kj < sched_kernel_count; kj++) {
                KernelEntry *kb = &sched_kernels[kj];
                if (kb->n_ops == 0 && kb->n_leaves == 0) continue;
                if (ka->n_ops != kb->n_ops || ka->n_leaves != kb->n_leaves) continue;
                if (ka->has_reduce != kb->has_reduce) continue;
                if (ka->full_shape.rank != kb->full_shape.rank) continue;
                int same = 1;
                for (u32 d = 0; d < ka->full_shape.rank && same; d++)
                    if (ka->full_shape.dims[d] != kb->full_shape.dims[d]) same = 0;
                for (u32 d = 0; d < ka->out_shape.rank && same; d++)
                    if (ka->out_shape.dims[d] != kb->out_shape.dims[d]) same = 0;
                if (!same) continue;
                for (u32 i = 0; i < ka->n_ops && same; i++)
                    if (ka->ops[i].uop != kb->ops[i].uop ||
                        ka->ops[i].arg_a != kb->ops[i].arg_a ||
                        ka->ops[i].arg_b != kb->ops[i].arg_b) same = 0;
                if (!same) continue;
                // Compare leaves via canonical keys
                for (u32 i = 0; i < ka->n_leaves && same; i++) {
                    u32 key_a = LEAF_KEY(ka, i);
                    u32 key_b = LEAF_KEY(kb, i);
                    if (key_a != key_b) same = 0;
                }
                if (!same) continue;
                if (ka->has_reduce) {
                    for (u32 d = 0; d < MAX_DIM && same; d++)
                        if (ka->reduce.is_reduce[d] != kb->reduce.is_reduce[d]) same = 0;
                    if (!same) continue;
                }
                // Duplicate! Redirect kb's FUSING to ka's on the heap.
                Term ka_ft = fusing_terms[ki];
                if (term_tag(ka_ft) == TAG_ERA) continue;
                for (u64 h = 1; h < ctx->heap_pos; h++) {
                    Term ht = ctx->heap[h];
                    if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_FUSING) {
                        u64 floc = term_val(ht);
                        Term kid_t = heap_read(ctx, floc + 1);
                        if (term_tag(kid_t) == TAG_NUM && (u32)term_val(kid_t) == kj)
                            ctx->heap[h] = ka_ft;
                    }
                }
                kb->n_ops = 0; kb->n_leaves = 0; kb->has_reduce = 0;
                canonical[kj] = ki; // record dedup mapping
                n_deduped++; round_deduped++;
            }
        }
        if (round_deduped == 0) break;
        } // end CSE rounds
        #undef LEAF_KEY
        if (n_deduped && getenv("THVM_SCHED_DIAG"))
            fprintf(stderr, "CSE: deduped %u kernels\n", n_deduped);
    }

    // Post-reduce ew merge: fold single-consumer ew kernels into their consumer.
    // If ew kernel E feeds exactly 1 kernel K (via FUSING leaf), and K is a reduce
    // kernel, merge E's ops as post-reduce ops into K.
    {
        u32 n_merged = 0;
        // For each FUSING kernel, find which kid it references (build consumer map)
        for (u32 ei = 0; ei < sched_kernel_count; ei++) {
            KernelEntry *ek = &sched_kernels[ei];
            if (ek->n_ops == 0 && ek->n_leaves == 0) continue; // dead
            if (ek->has_reduce) continue; // only merge ew into reduce
            if (ek->n_ops == 0) continue; // passthrough, skip

            // Find E's FUSING term on heap → its kid
            u32 ek_kid = ei;

            // Count how many live kernels consume E (reference E's kid as a FUSING leaf)
            u32 n_consumers = 0;
            u32 consumer_kid = 0xFFFFFFFFu;
            u32 consumer_leaf_idx = 0;
            for (u32 ci = 0; ci < sched_kernel_count; ci++) {
                if (ci == ei) continue;
                KernelEntry *ck = &sched_kernels[ci];
                if (ck->n_ops == 0 && ck->n_leaves == 0) continue;
                for (u32 li = 0; li < ck->n_leaves; li++) {
                    u32 lid = ck->leaf_ids[li];
                    if (lid != 0 && !LEAF_IS_LAZY(lid)) continue;
                    // Resolve lazy leaf to FUSING kid
                    Term lt = ck->leaf_terms[li];
                    while (term_tag(lt)==TAG_TOP && is_view_op(term_ext(lt))) {
                        Term nx = heap_read(ctx, term_val(lt));
                        if (term_tag(nx)==TAG_DP0||term_tag(nx)==TAG_DP1) nx=heap_read(ctx,term_val(nx));
                        lt = nx;
                    }
                    if (term_tag(lt)==TAG_TOP && term_ext(lt)==UOP_FUSING) {
                        u64 fl = term_val(lt);
                        Term kt = heap_read(ctx, fl+1);
                        if (term_tag(kt)==TAG_NUM && (u32)term_val(kt)==ek_kid) {
                            n_consumers++;
                            consumer_kid = ci;
                            consumer_leaf_idx = li;
                        }
                    }
                }
            }

            // Only merge if single consumer
            if (n_consumers != 1) continue;
            KernelEntry *ck = &sched_kernels[consumer_kid];
            // For reduce consumers: merge as post-reduce ops
            // For ew consumers: not supported yet (would need prepend)
            if (!ck->has_reduce) continue;
            if (ck->reduce.post_reduce_start) continue; // already has post-reduce
            // Check: can we fit E's ops + leaves into the consumer?
            if (ck->n_ops + ek->n_ops > FUSE_MAX_OPS) continue;
            if (ck->n_leaves + ek->n_leaves > FUSE_MAX_LEAVES) continue;

            // Merge: E's ops become post-reduce ops in the consumer.
            // The reduce result index in the consumer is: ck->n_leaves + ck->n_ops - 1
            // E's ops reference E's leaves and E's prior ops.
            // After merge: E's leaf indices shift by ck->n_leaves, E's op refs shift similarly.
            u32 pre_n_leaves = ck->n_leaves;
            u32 pre_n_ops = ck->n_ops;

            // Add E's non-FUSING leaves to the consumer
            // E's leaf that references the consumer itself (the reduce result) should be
            // remapped to the reduce result index (pre_n_leaves + pre_n_ops - 1).
            u32 reduce_result_idx = pre_n_leaves + pre_n_ops - 1;
            // But if pre_n_ops == 0, reduce result is the passthrough from leaf[0].
            if (pre_n_ops == 0) reduce_result_idx = 0;

            // Map E's leaf indices to merged indices
            u32 ek_leaf_remap[FUSE_MAX_LEAVES];
            u32 n_post_leaves = 0;
            for (u32 li = 0; li < ek->n_leaves; li++) {
                u32 lid = ek->leaf_ids[li];
                // Check if this leaf references the consumer's output (the reduce result)
                int is_reduce_ref = 0;
                if (lid == 0 || LEAF_IS_LAZY(lid)) {
                    Term lt = ek->leaf_terms[li];
                    while (term_tag(lt)==TAG_TOP && is_view_op(term_ext(lt))) {
                        Term nx = heap_read(ctx, term_val(lt));
                        if (term_tag(nx)==TAG_DP0||term_tag(nx)==TAG_DP1) nx=heap_read(ctx,term_val(nx));
                        lt = nx;
                    }
                    if (term_tag(lt)==TAG_TOP && term_ext(lt)==UOP_FUSING) {
                        u64 fl = term_val(lt);
                        Term kt = heap_read(ctx, fl+1);
                        if (term_tag(kt)==TAG_NUM && (u32)term_val(kt)==consumer_kid)
                            is_reduce_ref = 1;
                    }
                }
                if (is_reduce_ref) {
                    // This leaf reads the reduce result → remap to reduce_result_idx
                    ek_leaf_remap[li] = reduce_result_idx;
                } else {
                    // Add as post-reduce leaf
                    u32 new_li = ck->n_leaves;
                    if (new_li >= FUSE_MAX_LEAVES) goto skip_merge;
                    ck->leaf_ids[new_li] = ek->leaf_ids[li];
                    ck->leaf_views[new_li] = ek->leaf_views[li];
                    ck->leaf_terms[new_li] = ek->leaf_terms[li];
                    ck->leaf_sts[new_li] = ek->leaf_sts[li];
                    ek_leaf_remap[li] = new_li;
                    ck->n_leaves++;
                    n_post_leaves++;
                }
            }

            // Record post-reduce start
            ck->reduce.post_reduce_start = pre_n_ops;
            ck->reduce.n_post_leaves = n_post_leaves;

            // Append E's ops, remapping references
            for (u32 oi = 0; oi < ek->n_ops; oi++) {
                u32 new_oi = ck->n_ops;
                FusedOp op = ek->ops[oi];
                // Remap arg_a
                if (op.arg_a < ek->n_leaves) op.arg_a = ek_leaf_remap[op.arg_a];
                else op.arg_a = op.arg_a - ek->n_leaves + pre_n_leaves + pre_n_ops;
                // Remap arg_b
                if (op.arg_b < ek->n_leaves) op.arg_b = ek_leaf_remap[op.arg_b];
                else op.arg_b = op.arg_b - ek->n_leaves + pre_n_leaves + pre_n_ops;
                ck->ops[new_oi] = op;
                ck->n_ops++;
            }

            // Kill E
            ek->n_ops = 0; ek->n_leaves = 0; ek->has_reduce = 0;
            n_merged++;
            continue;
            skip_merge:;
        }
        if (n_merged && getenv("THVM_SCHED_DIAG"))
            fprintf(stderr, "POST_REDUCE_MERGE: merged %u ew kernels\n", n_merged);
    }

    fuse_no_lazy_resolve = 0;
    return total;
}

// === DELETED: 500+ lines of merge passes (4-9), memory analysis, dead kernel redirect ===
// These were post-hoc merge attempts that created broken kernels.
// Replaced by: nothing. The simple 3-pass scheduling is sufficient.
// Kernel count matches tinygrad without merging (the fuser already
// absorbs ew ops into reduce kernels during the walk).



Term thvm_eval(TinyHVM *ctx, Term t) {
    sched_kernel_count = 0;
    for (u32 i = 0; i < SCHED_MAX_KERNELS; i++) kid_results[i] = term_era();

    // Phase 0 graph: before reduce — full graph from root term t
    if (getenv("THVM_GRAPH")) thvm_heap_dot_root(ctx, "/tmp/thvm_0_pre_reduce.dot", t);

    // Phase 1: Pure IC reduce — GRAD fires, compute/movement ops stay TAG_TOP (WNF).
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
    // DUP nodes persist for atoms (TEN/NUM) to maintain single-output invariant.
    // DUP⊳atom fires during reduction (both ports get the value) but the DUP
    // node stays on the heap as a structural sharing marker.
    if (getenv("THVM_SCHED_DIAG")) { fprintf(stderr, "phase1: "); sched_dump_heap(ctx); }
    // Phase 1 graph: after reduce — pure TAG_TOPs + combinators
    if (getenv("THVM_GRAPH")) thvm_heap_dot_root(ctx, "/tmp/thvm_1_post_reduce.dot", t);

    // Phase 2: schedule compute ops → FUSING, resolve view ops → TAG_TEN aliases.
    sched_all(ctx);
    // Phase 2 graph: after scheduling — FUSING kernels
    if (getenv("THVM_GRAPH")) thvm_heap_dot(ctx, "/tmp/thvm_2_post_sched.dot");
    if (getenv("THVM_SCHED_DIAG")) {
        u32 n_reduce=0, n_ew=0, n_fallback=0;
        u32 max_ops=0, max_leaves=0;
        for (u32 ki=0; ki<sched_kernel_count; ki++) {
            if (sched_kernels[ki].has_reduce) n_reduce++;
            else if (sched_kernels[ki].n_ops > 0) n_ew++;
            else n_fallback++;
            if (sched_kernels[ki].n_ops > max_ops) max_ops = sched_kernels[ki].n_ops;
            if (sched_kernels[ki].n_leaves > max_leaves) max_leaves = sched_kernels[ki].n_leaves;
        }
        fprintf(stderr, "sched_kernel_count=%u (reduce=%u ew=%u passthru=%u max_ops=%u max_leaves=%u)\n",
                sched_kernel_count, n_reduce, n_ew, n_fallback, max_ops, max_leaves);
        fprintf(stderr, "after sched: "); sched_dump_heap(ctx);
    }
    if (getenv("THVM_KERN_DUMP")) {
        u64 total_flops = 0, total_mem_r = 0, total_mem_w = 0;
        u32 n_live = 0;
        for (u32 ki = 0; ki < sched_kernel_count; ki++) {
            KernelEntry *k = &sched_kernels[ki];
            if (k->n_ops == 0 && k->n_leaves == 0) continue; // dead
            n_live++;
            // Compute FLOPs: full_numel * n_ops (ew ops per element) + reduce accumulation
            u64 full_numel = 1;
            for (u32 d = 0; d < k->full_shape.rank; d++) full_numel *= k->full_shape.dims[d];
            u64 out_numel = 1;
            for (u32 d = 0; d < k->out_shape.rank; d++) out_numel *= k->out_shape.dims[d];
            // ew FLOPs: each element of full_shape does n_ops operations
            // reduce FLOPs: each element of full_shape does 1 accumulate
            u64 ew_flops = full_numel * (k->n_ops > 0 ? k->n_ops : 0);
            u64 reduce_flops = k->has_reduce ? full_numel : 0;
            u64 flops = ew_flops + reduce_flops;
            if (flops == 0) flops = full_numel; // passthrough
            // Memory: reads = sum of leaf numels, write = out_numel
            u64 mem_r = 0;
            for (u32 li = 0; li < k->n_leaves; li++) {
                u64 ln = 1;
                for (u32 d = 0; d < k->leaf_views[li].shape.rank; d++)
                    ln *= k->leaf_views[li].shape.dims[d];
                mem_r += ln * 4;
            }
            u64 mem_w = out_numel * 4;
            total_flops += flops;
            total_mem_r += mem_r;
            total_mem_w += mem_w;
            // Format: tinygrad-style
            const char *ktype = k->has_reduce ? "r" : "E";
            fprintf(stderr, "*** K %3u  %s_", ki, ktype);
            // Shape encoding (tinygrad style: bold=full, colored=out for reduces)
            for (u32 d = 0; d < k->full_shape.rank; d++)
                fprintf(stderr, "%u%s", k->full_shape.dims[d], d+1<k->full_shape.rank?"_":"");
            if (k->has_reduce) {
                fprintf(stderr, " → ");
                for (u32 d = 0; d < k->out_shape.rank; d++)
                    fprintf(stderr, "%u%s", k->out_shape.dims[d], d+1<k->out_shape.rank?"_":"");
            }
            // Stats
            fprintf(stderr, "  arg %2u  ops %2u  flops %8llu  mem_r %7.2fKB  mem_w %7.2fKB  lv=[",
                    k->n_leaves, k->n_ops, flops,
                    (double)mem_r/1024.0, (double)mem_w/1024.0);
            for (u32 li = 0; li < k->n_leaves; li++)
                fprintf(stderr, "%u%s", k->leaf_ids[li], li+1<k->n_leaves?",":"");
            fprintf(stderr, "]\n");
        }
        fprintf(stderr, "--- TOTAL: %u live kernels  flops=%llu (%.2f MFLOP)  "
                "mem_r=%.2fMB  mem_w=%.2fMB  mem_total=%.2fMB\n",
                n_live, total_flops, (double)total_flops/1e6,
                (double)total_mem_r/1e6, (double)total_mem_w/1e6,
                (double)(total_mem_r+total_mem_w)/1e6);
    }

    { Backend *be = ctx_default_backend(ctx);
      if (be && be->end_batch) be->end_batch(); }

    // Phase 3: Fire ASSIGNs (enable dispatch, then reduce each).
    _assign_dispatch_enabled = 1;
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_ASSIGN) {
            Term r = thvm_reduce(ctx, ht);
            ctx->heap[h] = r;
        }
    }
    _assign_dispatch_enabled = 0;
    // Phase 3 graph: after dispatch — materialized tensors + remaining combinators
    if (getenv("THVM_GRAPH")) thvm_heap_dot(ctx, "/tmp/thvm_3_post_dispatch.dot");

    if (getenv("THVM_MEM_DIAG")) { // Memory analysis
        u32 alive[SCHED_MAX_KERNELS], n_alive = 0;
        for (u32 ki = 0; ki < sched_kernel_count; ki++) {
            KernelEntry *k = &sched_kernels[ki];
            if (k->n_ops == 0 && k->n_leaves == 0) continue;
            alive[n_alive++] = ki;
        }
        #define MEM_MAX_BUFS 2048
        typedef struct { u32 id; u64 size; u32 birth; u32 death; int is_output; } MemBuf;
        MemBuf bufs[MEM_MAX_BUFS]; u32 n_bufs = 0;
        #define MEM_FIND_OR_ADD(bid, sz, b_ai) do { \
            u32 _bi = 0xFFFFFFFFu; \
            for (u32 _b = 0; _b < n_bufs; _b++) \
                if (bufs[_b].id == (bid)) { _bi = _b; break; } \
            if (_bi == 0xFFFFFFFFu && n_bufs < MEM_MAX_BUFS) { \
                _bi = n_bufs++; \
                bufs[_bi] = (MemBuf){.id=(bid), .size=(sz), .birth=(b_ai), .death=(b_ai), .is_output=0}; \
            } \
            if (_bi != 0xFFFFFFFFu) { \
                if ((b_ai) < bufs[_bi].birth) bufs[_bi].birth = (b_ai); \
                if ((b_ai) > bufs[_bi].death) bufs[_bi].death = (b_ai); \
                if ((sz) > bufs[_bi].size) bufs[_bi].size = (sz); \
            } \
        } while(0)
        for (u32 ai = 0; ai < n_alive; ai++) {
            KernelEntry *k = &sched_kernels[alive[ai]];
            for (u32 li = 0; li < k->n_leaves; li++) {
                u32 lid = k->leaf_ids[li];
                if (lid == 0 || LEAF_IS_LAZY(lid)) {
                    Term lt = k->leaf_terms[li], ft = lt;
                    while (term_tag(ft)==TAG_TOP && is_view_op(term_ext(ft))) {
                        Term nx = heap_read(ctx, term_val(ft));
                        if (term_tag(nx)==TAG_DP0||term_tag(nx)==TAG_DP1) nx=heap_read(ctx,term_val(nx));
                        ft = nx;
                    }
                    if (term_tag(ft)==TAG_TOP && term_ext(ft)==UOP_FUSING) {
                        u64 floc = term_val(ft);
                        Term kid_t = heap_read(ctx, floc+1);
                        if (term_tag(kid_t)==TAG_NUM) {
                            u32 pki = (u32)term_val(kid_t);
                            u32 oid = 0x40000000u|pki;
                            KernelEntry *p = &sched_kernels[pki];
                            u64 osz = 1;
                            for (u32 d=0;d<p->out_shape.rank;d++) osz*=p->out_shape.dims[d];
                            MEM_FIND_OR_ADD(oid, osz*4, ai);
                        }
                    }
                    continue;
                }
                u64 sz = (lid < ctx->tensor_count)
                    ? (u64)ctx->tensors[lid].view.numel * 4
                    : (u64)k->leaf_views[li].numel * 4;
                MEM_FIND_OR_ADD(lid, sz, ai);
            }
            u32 oid = 0x40000000u|alive[ai];
            u64 osz = 1;
            // Use reshape target shape if set (actual materialized output is smaller)
            if (term_tag(k->reshape_term) != TAG_ERA) {
                const View *rv = st_get(term_val(k->reshape_term));
                if (rv) { for (u32 d=0;d<rv->shape.rank;d++) osz*=rv->shape.dims[d]; }
                else { for (u32 d=0;d<k->out_shape.rank;d++) osz*=k->out_shape.dims[d]; }
            } else {
                for (u32 d=0;d<k->out_shape.rank;d++) osz*=k->out_shape.dims[d];
            }
            MEM_FIND_OR_ADD(oid, osz*4, ai);
            for (u32 b=0;b<n_bufs;b++) if(bufs[b].id==oid){bufs[b].is_output=1;break;}
        }
        // Dead-chain propagation: if a kernel's output feeds ONLY into dead kernels,
        // kill the kernel too. Propagate until stable.
        u32 n_fused_away = 0;
        for (u32 _dc = 0; _dc < 20; _dc++) {
            u32 n_dc = 0;
            // Rebuild alive list
            n_alive = 0;
            for (u32 ki = 0; ki < sched_kernel_count; ki++) {
                KernelEntry *k = &sched_kernels[ki];
                if (k->n_ops == 0 && k->n_leaves == 0) continue;
                alive[n_alive++] = ki;
            }
            for (u32 ai = 0; ai < n_alive; ai++) {
                u32 ki = alive[ai];
                // Check: does any OTHER alive kernel consume this kernel's output?
                int has_alive_consumer = 0;
                for (u32 aj = 0; aj < n_alive && !has_alive_consumer; aj++) {
                    if (aj == ai) continue;
                    KernelEntry *ck = &sched_kernels[alive[aj]];
                    for (u32 li = 0; li < ck->n_leaves; li++) {
                        u32 lid = ck->leaf_ids[li];
                        // Concrete leaf: check if tensor was derived from ki's output
                        if (lid != 0 && !LEAF_IS_LAZY(lid) && lid < ctx->tensor_count) {
                            TensorMeta *lm = &ctx->tensors[lid];
                            if (lm->src_ids[0]) {
                                // Trace back: src tensor numel matches ki's out_shape?
                                KernelEntry *pk = &sched_kernels[ki];
                                u32 pk_numel = 1;
                                for (u32 d=0;d<pk->out_shape.rank;d++) pk_numel*=pk->out_shape.dims[d];
                                TensorMeta *sm = &ctx->tensors[lm->src_ids[0]];
                                if (sm->view.numel == pk_numel) { has_alive_consumer = 1; break; }
                            }
                        }
                        // FUSING leaf: check kid
                        if (lid == 0 || LEAF_IS_LAZY(lid)) {
                            Term lt = ck->leaf_terms[li], ft = lt;
                            while (term_tag(ft)==TAG_TOP && is_view_op(term_ext(ft))) {
                                Term nx=heap_read(ctx,term_val(ft));
                                if(term_tag(nx)==TAG_DP0||term_tag(nx)==TAG_DP1) nx=heap_read(ctx,term_val(nx));
                                ft=nx;
                            }
                            if (term_tag(ft)==TAG_TOP && term_ext(ft)==UOP_FUSING) {
                                u64 fl=term_val(ft); Term kt=heap_read(ctx,fl+1);
                                if (term_tag(kt)==TAG_NUM && (u32)term_val(kt)==ki)
                                    { has_alive_consumer = 1; break; }
                            }
                        }
                    }
                }
                if (!has_alive_consumer) {
                    sched_kernels[ki].n_ops = 0;
                    sched_kernels[ki].n_leaves = 0;
                    sched_kernels[ki].has_reduce = 0; // analysis-only kill
                    n_dc++; n_fused_away++;
                }
            }
            if (n_dc == 0) break;
        }
        // Rebuild alive list after dead-chain propagation
        n_alive = 0;
        for (u32 ki = 0; ki < sched_kernel_count; ki++) {
            KernelEntry *k = &sched_kernels[ki];
            if (k->n_ops == 0 && k->n_leaves == 0) continue;
            alive[n_alive++] = ki;
        }
        // Rebuild buf registry with only alive kernels
        n_bufs = 0;
        for (u32 ai = 0; ai < n_alive; ai++) {
            KernelEntry *k = &sched_kernels[alive[ai]];
            for (u32 li = 0; li < k->n_leaves; li++) {
                u32 lid = k->leaf_ids[li];
                if (lid == 0 || LEAF_IS_LAZY(lid)) {
                    Term lt=k->leaf_terms[li],ft=lt;
                    while(term_tag(ft)==TAG_TOP&&is_view_op(term_ext(ft))){Term nx=heap_read(ctx,term_val(ft));if(term_tag(nx)==TAG_DP0||term_tag(nx)==TAG_DP1)nx=heap_read(ctx,term_val(nx));ft=nx;}
                    if(term_tag(ft)==TAG_TOP&&term_ext(ft)==UOP_FUSING){u64 fl=term_val(ft);Term kt=heap_read(ctx,fl+1);if(term_tag(kt)==TAG_NUM){u32 pki=(u32)term_val(kt);u32 oid=0x40000000u|pki;KernelEntry*p=&sched_kernels[pki];u64 osz=1;for(u32 d=0;d<p->out_shape.rank;d++)osz*=p->out_shape.dims[d];MEM_FIND_OR_ADD(oid,osz*4,ai);}}
                    continue;
                }
                u64 sz;
                if (lid < ctx->tensor_count) {
                    TensorMeta *lm = &ctx->tensors[lid];
                    // Contiguified view tensors: use source size (tinygrad reads through
                    // strided view, never contiguifies). Trace src_ids chain to root buffer.
                    u32 root = lid;
                    while (root < ctx->tensor_count && ctx->tensors[root].src_ids[0] &&
                           ctx->tensors[root].creator_op &&
                           ctx->tensors[root].src_ids[0] < ctx->tensor_count) {
                        root = ctx->tensors[root].src_ids[0];
                    }
                    if (root != lid) {
                        sz = (u64)ctx->tensors[root].view.numel * 4;
                    } else {
                        sz = (u64)lm->view.numel * 4;
                    }
                } else {
                    sz = (u64)k->leaf_views[li].numel * 4;
                }
                MEM_FIND_OR_ADD(lid,sz,ai);
            }
            u32 oid=0x40000000u|alive[ai]; u64 osz=1;
            if(term_tag(k->reshape_term)!=TAG_ERA){const View*rv=st_get(term_val(k->reshape_term));if(rv){for(u32 d=0;d<rv->shape.rank;d++)osz*=rv->shape.dims[d];}else{for(u32 d=0;d<k->out_shape.rank;d++)osz*=k->out_shape.dims[d];}}
            else{for(u32 d=0;d<k->out_shape.rank;d++)osz*=k->out_shape.dims[d];}
            MEM_FIND_OR_ADD(oid,osz*4,ai);
            for(u32 b=0;b<n_bufs;b++)if(bufs[b].id==oid){bufs[b].is_output=1;break;}
        }
        // Output dedup: if two output bufs have the same size and one's lifetime
        // is contained within the other's, the later reuses the earlier's buffer.
        // (Forward output reused in backward — tinygrad does this automatically.)
        for (u32 b1 = 0; b1 < n_bufs; b1++) {
            if (!bufs[b1].is_output || bufs[b1].size == 0) continue;
            for (u32 b2 = b1 + 1; b2 < n_bufs; b2++) {
                if (!bufs[b2].is_output || bufs[b2].size == 0) continue;
                if (bufs[b1].size != bufs[b2].size) continue;
                // Same size outputs: merge lifetimes, zero out the later one
                if (bufs[b2].birth >= bufs[b1].birth) {
                    if (bufs[b2].death > bufs[b1].death) bufs[b1].death = bufs[b2].death;
                    bufs[b2].size = 0;
                    n_fused_away++;
                }
            }
        }
        u32 peak_live=0; u64 peak_bytes=0;
        for (u32 ai=0;ai<n_alive;ai++) {
            u32 live=0; u64 lb=0;
            for (u32 b=0;b<n_bufs;b++)
                if(bufs[b].birth<=ai&&bufs[b].death>=ai){live++;lb+=bufs[b].size;}
            if(live>peak_live)peak_live=live;
            if(lb>peak_bytes)peak_bytes=lb;
        }
        u32 slot_count=0; u32 slot_end[MEM_MAX_BUFS]; u64 slot_size[MEM_MAX_BUFS]; u64 reuse_saved=0;
        for (u32 b=0;b<n_bufs;b++) {
            int found=-1;
            for (u32 s=0;s<slot_count;s++)
                if(slot_end[s]<bufs[b].birth && slot_size[s]>=bufs[b].size){found=(int)s;break;}
            if(found>=0){slot_end[found]=bufs[b].death;if(bufs[b].size>slot_size[found])slot_size[found]=bufs[b].size;reuse_saved+=bufs[b].size;}
            else if(slot_count<MEM_MAX_BUFS){slot_end[slot_count]=bufs[b].death;slot_size[slot_count]=bufs[b].size;slot_count++;}
        }
        u64 total_bytes=0; for(u32 b=0;b<n_bufs;b++) total_bytes+=bufs[b].size;
        u64 slot_bytes=0; for(u32 s=0;s<slot_count;s++) slot_bytes+=slot_size[s];
        u32 n_in=0,n_out=0; for(u32 b=0;b<n_bufs;b++){if(bufs[b].is_output)n_out++;else n_in++;}
        fprintf(stderr,"MEM_ANALYSIS: alive=%u bufs=%u (in=%u out=%u fused=%u) peak_live=%u\n",
                n_alive,n_bufs,n_in,n_out,n_fused_away,peak_live);
        fprintf(stderr,"  no_reuse=%.2fMB peak=%.2fMB\n",(double)total_bytes/1e6,(double)peak_bytes/1e6);
        // Dump large buffers
        for (u32 b=0;b<n_bufs;b++) {
            if (bufs[b].size >= 4096) {
                u32 bid = bufs[b].id;
                const char *kind = bufs[b].is_output ? "OUT" : "IN";
                if (bid < ctx->tensor_count) {
                    TensorMeta *tm = &ctx->tensors[bid];
                    fprintf(stderr,"  BUF %s id=%u sz=%.2fMB shape=[", kind, bid, (double)bufs[b].size/1e6);
                    for (u32 d=0;d<tm->view.shape.rank;d++) fprintf(stderr,"%u,",tm->view.shape.dims[d]);
                    fprintf(stderr,"] live=%u-%u\n", bufs[b].birth, bufs[b].death);
                } else {
                    fprintf(stderr,"  BUF %s id=0x%x sz=%.2fMB live=%u-%u\n", kind, bid, (double)bufs[b].size/1e6, bufs[b].birth, bufs[b].death);
                }
            }
        }
        fprintf(stderr,"  with_reuse: slots=%u need=%.2fMB saved=%.2fMB (%.0f%%)\n",
                slot_count,(double)slot_bytes/1e6,(double)reuse_saved/1e6,
                total_bytes>0?100.0*reuse_saved/total_bytes:0.0);
    } // end THVM_MEM_DIAG

    return term_era();
}

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_eval(ctx, t);
}
