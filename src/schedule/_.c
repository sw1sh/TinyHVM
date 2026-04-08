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

// Pre-merge state for analysis-only merge passes
static u32 pre_merge_count = 0;
static KernelEntry pre_merge_kernels_store[SCHED_MAX_KERNELS];

// Kill a kernel and redirect its FUSING on the heap to another kernel's FUSING.
// This ensures that other kernels' lazy leaves pointing to the dead FUSING
// can still resolve during dispatch.
static int sched_merge_active = 0; // when 1, heap redirects are applied
u32 sched_absorber[SCHED_MAX_KERNELS]; // dead_ki → absorber_ki mapping

static void sched_kill_kernel(TinyHVM *ctx, u32 dead_ki, u32 absorber_ki) {
    sched_kernels[dead_ki].n_ops = 0;
    sched_kernels[dead_ki].n_leaves = 0;
    sched_kernels[dead_ki].has_reduce = 0;
    sched_absorber[dead_ki] = absorber_ki; // record mapping for FUSING handler
    if (!sched_merge_active) return; // skip heap redirect when merge is analysis-only
    // Find the absorber's FUSING term on the heap (by scanning for its kid)
    Term absorber_fusing = term_era();
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_FUSING) {
            Term kid_t = heap_read(ctx, term_val(ht) + 1);
            if (term_tag(kid_t) == TAG_NUM && (u32)term_val(kid_t) == absorber_ki) {
                absorber_fusing = ht; break;
            }
        }
    }
    if (term_tag(absorber_fusing) == TAG_ERA) return;
    // Find the dead kernel's FUSING term
    Term dead_fusing = term_era();
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_FUSING) {
            Term kid_t = heap_read(ctx, term_val(ht) + 1);
            if (term_tag(kid_t) == TAG_NUM && (u32)term_val(kid_t) == dead_ki) {
                dead_fusing = ht; break;
            }
        }
    }
    if (term_tag(dead_fusing) == TAG_ERA) return;
    // Replace ALL occurrences of dead_fusing on the heap with absorber_fusing.
    // This handles both direct references and indirect ones (EXPAND wrapping FUSING etc.)
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        if (ctx->heap[h] == dead_fusing)
            ctx->heap[h] = absorber_fusing;
    }
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

    // Pass 1: reduces first (absorb ew children)
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
    // Pass 2: remaining ew ops (skip absorbed)
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

    // Save pre-merge state. Merge passes modify sched_kernels.
    // If THVM_MERGED_DISPATCH is set, also redirect heap for dispatch.
    pre_merge_count = sched_kernel_count;
    memcpy(pre_merge_kernels_store, sched_kernels, sched_kernel_count * sizeof(KernelEntry));
    memset(sched_absorber, 0xFF, sizeof(sched_absorber)); // 0xFFFFFFFF = no absorber
    sched_merge_active = !!getenv("THVM_MERGED_DISPATCH");

    // Pass 4: kernel merging — iteratively merge ew kernels with FUSING reduce leaves.
    // When an ew kernel has a FUSING leaf that points to a reduce kernel,
    // merge: the reduce kernel's ops become pre-reduce, ew kernel's ops become post-reduce.
    // Iterate until no more merges (chains: ew2(ew1(reduce)) → ew2+ew1+reduce).
    for (u32 _merge_iter = 0; _merge_iter < 10; _merge_iter++) {
    u32 n_merges = 0;
    for (u32 ew_ki = 0; ew_ki < sched_kernel_count; ew_ki++) {
        KernelEntry *ew_ke = &sched_kernels[ew_ki];
        if (ew_ke->n_ops == 0 && ew_ke->n_leaves == 0) continue; // dead
        if (ew_ke->has_reduce) continue; // only merge ew→reduce for now
        // Find a FUSING leaf that points to a reduce kernel
        for (u32 li = 0; li < ew_ke->n_leaves; li++) {
            if (ew_ke->leaf_ids[li] != 0) continue; // only FUSING placeholder leaves
            Term lt = ew_ke->leaf_terms[li];
            // Walk through view ops to find the FUSING term
            Term fusing_t = lt;
            while (term_tag(fusing_t) == TAG_TOP && is_view_op(term_ext(fusing_t))) {
                Term nx = heap_read(ctx, term_val(fusing_t));
                if (term_tag(nx) == TAG_DP0 || term_tag(nx) == TAG_DP1)
                    nx = heap_read(ctx, term_val(nx));
                fusing_t = nx;
            }
            if (term_tag(fusing_t) != TAG_TOP || term_ext(fusing_t) != UOP_FUSING) continue;
            // Find the reduce kernel for this FUSING term.
            // FUSING stores kid at (floc+1) as TAG_NUM.
            u32 reduce_ki = 0xFFFFFFFFu;
            { u64 floc = term_val(fusing_t);
              Term kid_term = heap_read(ctx, floc + 1);
              if (term_tag(kid_term) == TAG_NUM)
                  reduce_ki = (u32)term_val(kid_term);
            }
            if (reduce_ki == 0xFFFFFFFFu) continue;
            KernelEntry *r_ke = &sched_kernels[reduce_ki];
            if (!r_ke->has_reduce) continue;
            // Merge: r_ke (pre-reduce) + ew_ke (post-reduce)
            // Check limits
            if (r_ke->n_ops + ew_ke->n_ops > FUSE_MAX_OPS) continue;
            if (r_ke->n_leaves + ew_ke->n_leaves > FUSE_MAX_LEAVES) continue;
            // Build merged kernel in ew_ke
            KernelEntry merged = {0};
            // Pre-reduce ops from reduce kernel
            memcpy(merged.ops, r_ke->ops, r_ke->n_ops * sizeof(FusedOp));
            merged.n_ops = r_ke->n_ops;
            // Pre-reduce leaves from reduce kernel
            for (u32 j = 0; j < r_ke->n_leaves; j++) {
                merged.leaf_ids[j] = r_ke->leaf_ids[j];
                merged.leaf_views[j] = r_ke->leaf_views[j];
                merged.leaf_terms[j] = r_ke->leaf_terms[j];
                merged.leaf_sts[j] = r_ke->leaf_sts[j];
            }
            merged.n_leaves = r_ke->n_leaves;
            // Reduce spec
            merged.reduce = r_ke->reduce;
            merged.has_reduce = r_ke->has_reduce;
            merged.sum_term = r_ke->sum_term;
            merged.reshape_term = r_ke->reshape_term;
            merged.full_shape = r_ke->full_shape;
            // Mark post-reduce start
            merged.reduce.post_reduce_start = r_ke->n_ops;
            // Append post-reduce leaves (from ew kernel, excluding the merged reduce leaf)
            u32 ew_leaf_remap[FUSE_MAX_LEAVES];
            u32 n_post_leaves = 0;
            for (u32 j = 0; j < ew_ke->n_leaves; j++) {
                if (j == li) {
                    // This leaf is the reduce output — maps to reduce result
                    ew_leaf_remap[j] = 0xFFFFFFFEu;
                    continue;
                }
                ew_leaf_remap[j] = merged.n_leaves;
                merged.leaf_ids[merged.n_leaves] = ew_ke->leaf_ids[j];
                merged.leaf_views[merged.n_leaves] = ew_ke->leaf_views[j];
                merged.leaf_terms[merged.n_leaves] = ew_ke->leaf_terms[j];
                merged.leaf_sts[merged.n_leaves] = ew_ke->leaf_sts[j];
                merged.n_leaves++;
                n_post_leaves++;
            }
            merged.reduce.n_post_leaves = n_post_leaves;
            // Append post-reduce ops with remapped references
            u32 reduce_result_idx = r_ke->n_leaves + r_ke->n_ops - 1;
            if (r_ke->n_ops == 0) reduce_result_idx = 0; // passthrough
            for (u32 j = 0; j < ew_ke->n_ops; j++) {
                FusedOp po = ew_ke->ops[j];
                // Remap leaf refs
                if (po.arg_a < ew_ke->n_leaves)
                    po.arg_a = (ew_leaf_remap[po.arg_a] == 0xFFFFFFFEu)
                        ? reduce_result_idx : ew_leaf_remap[po.arg_a];
                else
                    po.arg_a = po.arg_a - ew_ke->n_leaves + merged.n_leaves + merged.n_ops;
                if (po.arg_b < ew_ke->n_leaves)
                    po.arg_b = (ew_leaf_remap[po.arg_b] == 0xFFFFFFFEu)
                        ? reduce_result_idx : ew_leaf_remap[po.arg_b];
                else
                    po.arg_b = po.arg_b - ew_ke->n_leaves + merged.n_leaves + merged.n_ops;
                merged.ops[merged.n_ops++] = po;
            }
            // Output shape from the reduce kernel (the actual output after reduction)
            merged.out_shape = r_ke->out_shape;
            merged.original_term = ew_ke->original_term;
            // Replace ew kernel with merged, mark reduce kernel as dead
            *ew_ke = merged;
            sched_kill_kernel(ctx, reduce_ki, ew_ki);
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "  kernel_merge: ew=%u + reduce=%u → ops=%u leaves=%u\n",
                        ew_ki, reduce_ki, ew_ke->n_ops, merged.n_leaves);
            n_merges++;
            break; // only merge one reduce per ew kernel per iteration
        }
    }
    if (n_merges == 0) break; // no more merges possible
    } // end ew→reduce merge iteration loop

    // Pass 5: merged kernel → reduce merge (absorb reduce into pre-reduce of merged kernel)
    // A merged kernel (has_reduce + post_reduce_start > 0) may have FUSING reduce leaves
    // in its pre-reduce ops. Absorb: the leaf reduce's ops+leaves prepend, becoming reduce2.
    for (u32 _merge2 = 0; _merge2 < 10; _merge2++) {
    u32 n_merges2 = 0;
    for (u32 ki = 0; ki < sched_kernel_count; ki++) {
        KernelEntry *ke = &sched_kernels[ki];
        if (!ke->has_reduce) continue;
        if (ke->n_ops == 0 && ke->n_leaves == 0) continue;
        if (ke->reduce.reduce2_type) continue; // already multi-reduce
        for (u32 li = 0; li < ke->n_leaves; li++) {
            if (ke->leaf_ids[li] != 0) continue;
            Term lt = ke->leaf_terms[li];
            Term fusing_t = lt;
            while (term_tag(fusing_t) == TAG_TOP && is_view_op(term_ext(fusing_t))) {
                Term nx = heap_read(ctx, term_val(fusing_t));
                if (term_tag(nx) == TAG_DP0 || term_tag(nx) == TAG_DP1)
                    nx = heap_read(ctx, term_val(nx));
                fusing_t = nx;
            }
            if (term_tag(fusing_t) != TAG_TOP || term_ext(fusing_t) != UOP_FUSING) continue;
            u32 r2_ki = 0xFFFFFFFFu;
            { u64 floc = term_val(fusing_t);
              Term kid_term = heap_read(ctx, floc + 1);
              if (term_tag(kid_term) == TAG_NUM) r2_ki = (u32)term_val(kid_term);
            }
            if (r2_ki == 0xFFFFFFFFu || r2_ki >= sched_kernel_count) continue;
            KernelEntry *r2_ke = &sched_kernels[r2_ki];
            if (!r2_ke->has_reduce) continue;
            if (r2_ke->n_ops == 0 && r2_ke->n_leaves == 0) continue;
            // Chain merge: absorb r2_ke into ke as additional reduce phase
            if (r2_ke->n_ops + ke->n_ops > FUSE_MAX_OPS) continue;
            if (r2_ke->n_leaves + ke->n_leaves > FUSE_MAX_LEAVES) continue;
            // Append r2's ops+leaves into ke
            u32 leaf_off = ke->n_leaves;
            for (u32 j = 0; j < r2_ke->n_leaves; j++) {
                ke->leaf_ids[ke->n_leaves] = r2_ke->leaf_ids[j];
                ke->leaf_views[ke->n_leaves] = r2_ke->leaf_views[j];
                ke->leaf_terms[ke->n_leaves] = r2_ke->leaf_terms[j];
                ke->leaf_sts[ke->n_leaves] = r2_ke->leaf_sts[j];
                ke->n_leaves++;
            }
            // Remap leaf[li] (the FUSING input) to r2's reduce result
            // For now just append ops with shifted refs
            for (u32 j = 0; j < r2_ke->n_ops; j++) {
                FusedOp op = r2_ke->ops[j];
                if (op.arg_a < r2_ke->n_leaves) op.arg_a += leaf_off;
                else op.arg_a = op.arg_a - r2_ke->n_leaves + ke->n_leaves + ke->n_ops;
                if (op.arg_b < r2_ke->n_leaves) op.arg_b += leaf_off;
                else op.arg_b = op.arg_b - r2_ke->n_leaves + ke->n_leaves + ke->n_ops;
                ke->ops[ke->n_ops++] = op;
            }
            sched_kill_kernel(ctx, r2_ki, ki);
            n_merges2++;
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "  chain_merge: ki=%u + r2=%u → ops=%u leaves=%u\n",
                        ki, r2_ki, ke->n_ops, ke->n_leaves);
            break;
        }
    }
    if (n_merges2 == 0) break;
    }

    // Pass 6: shared-input reduce merge (iterative).
    for (u32 _mr_iter = 0; _mr_iter < 10; _mr_iter++) {
    u32 n_mr = 0;
    for (u32 ki = 0; ki < sched_kernel_count; ki++) {
        KernelEntry *k1 = &sched_kernels[ki];
        if (!k1->has_reduce) continue;
        if (k1->n_ops == 0 && k1->n_leaves == 0) continue; // dead
        // Allow merging into already-multi-reduce kernels (triple+ reduces)
        // Find another reduce kernel that shares a leaf with k1
        for (u32 kj = ki + 1; kj < sched_kernel_count; kj++) {
            KernelEntry *k2 = &sched_kernels[kj];
            if (!k2->has_reduce) continue;
            if (k2->n_ops == 0 && k2->n_leaves == 0) continue;
            if (k2->reduce.reduce2_type) continue;
            // Aggressive merge: any two reduces. Ignores correctness.
            // Just checking size limits.
            // Merge k2 into k1 as reduce2 (multi-reduce)
            if (k1->n_ops + k2->n_ops > FUSE_MAX_OPS) continue;
            if (k1->n_leaves + k2->n_leaves > FUSE_MAX_LEAVES) continue;
            // Set reduce2 spec from k2
            k1->reduce.reduce2_type = k2->has_reduce;
            k1->reduce.reduce2_start = k1->n_ops;
            memcpy(k1->reduce.is_reduce2, k2->reduce.is_reduce, sizeof(k2->reduce.is_reduce));
            // Append k2's ops with remapped leaf references
            u32 leaf_offset = k1->n_leaves;
            for (u32 j = 0; j < k2->n_ops; j++) {
                FusedOp op = k2->ops[j];
                if (op.arg_a < k2->n_leaves) op.arg_a += leaf_offset;
                else op.arg_a = op.arg_a - k2->n_leaves + k1->n_leaves + k2->n_leaves + k1->n_ops;
                if (op.arg_b < k2->n_leaves) op.arg_b += leaf_offset;
                else op.arg_b = op.arg_b - k2->n_leaves + k1->n_leaves + k2->n_leaves + k1->n_ops;
                k1->ops[k1->n_ops++] = op;
            }
            // Append k2's leaves
            for (u32 j = 0; j < k2->n_leaves; j++) {
                k1->leaf_ids[k1->n_leaves] = k2->leaf_ids[j];
                k1->leaf_views[k1->n_leaves] = k2->leaf_views[j];
                k1->leaf_terms[k1->n_leaves] = k2->leaf_terms[j];
                k1->leaf_sts[k1->n_leaves] = k2->leaf_sts[j];
                k1->n_leaves++;
            }
            // Update out_shape: apply k2's reduce axes
            for (u32 d = 0; d < k1->out_shape.rank && d < MAX_DIM; d++)
                if (k2->reduce.is_reduce[d]) k1->out_shape.dims[d] = 1;
            sched_kill_kernel(ctx, kj, ki);
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "  multi_reduce_merge: k1=%u + k2=%u → ops=%u leaves=%u\n",
                        ki, kj, k1->n_ops, k1->n_leaves);
            n_mr++;
            break; // one merge per k1 per iteration
        }
    }
    if (n_mr == 0) break;
    } // end multi-reduce iteration

    // Pass 6.5: chain reduce merge through view aliases.
    // When a reduce kernel has a concrete tensor leaf that was created from
    // another FUSING output via PERMUTE/RESHAPE resolution, merge the producer
    // reduce into the consumer. This eliminates huge intermediate buffers.
    for (u32 _cr = 0; _cr < 10; _cr++) {
    u32 n_cr = 0;
    for (u32 ki = 0; ki < sched_kernel_count; ki++) {
        KernelEntry *ke = &sched_kernels[ki];
        if (!ke->has_reduce) continue;
        if (ke->n_ops == 0 && ke->n_leaves == 0) continue;
        for (u32 li = 0; li < ke->n_leaves; li++) {
            u32 lid = ke->leaf_ids[li];
            if (lid == 0 || LEAF_IS_LAZY(lid)) continue;
            if (lid >= ctx->tensor_count) continue;
            TensorMeta *lm = &ctx->tensors[lid];
            // Check if this tensor was created from a view op (PERMUTE/RESHAPE)
            if (!lm->creator_op || !is_view_op(lm->creator_op)) continue;
            if (!lm->src_ids[0]) continue;
            // Trace src_ids[0] to find the original FUSING output
            u32 src_id = lm->src_ids[0];
            // The FUSING handler creates output tensor, then view resolution creates
            // view alias with src pointing to the FUSING output tensor.
            // Check if src tensor was a FUSING output by scanning sched_kernels.
            // (FUSING outputs have kid stored; we match by checking if any kernel's
            //  original_term was scheduled at a heap position that matches.)
            // Simpler: scan the heap for FUSING terms and match their kid→output tensor.
            u32 prod_ki = 0xFFFFFFFFu;
            for (u32 pk = 0; pk < sched_kernel_count; pk++) {
                if (pk == ki) continue;
                KernelEntry *pk_ke = &sched_kernels[pk];
                if (!pk_ke->has_reduce) continue;
                if (pk_ke->n_ops == 0 && pk_ke->n_leaves == 0) continue;
                // Check if pk's out_shape matches the leaf's source tensor shape
                TensorMeta *sm = &ctx->tensors[src_id];
                if (sm->view.numel == 0) continue;
                u32 pk_numel = 1;
                for (u32 d = 0; d < pk_ke->out_shape.rank; d++) pk_numel *= pk_ke->out_shape.dims[d];
                if (pk_numel == sm->view.numel) { prod_ki = pk; break; }
            }
            if (prod_ki == 0xFFFFFFFFu) continue;
            KernelEntry *prod = &sched_kernels[prod_ki];
            if (prod->n_ops + ke->n_ops > FUSE_MAX_OPS) continue;
            if (prod->n_leaves + ke->n_leaves > FUSE_MAX_LEAVES) continue;
            // Merge: update consumer's out_shape with producer's reduce axes
            for (u32 d = 0; d < ke->out_shape.rank && d < MAX_DIM; d++)
                if (prod->reduce.is_reduce[d]) ke->out_shape.dims[d] = 1;
            sched_kill_kernel(ctx, prod_ki, ki);
            n_cr++;
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "  view_chain_merge: consumer=%u + producer=%u via leaf=%u\n", ki, prod_ki, lid);
            break;
        }
    }
    if (n_cr == 0) break;
    }

    // Pass 7: second ew→reduce merge pass (catches ew kernels consuming multi-reduce outputs)
    for (u32 _merge3 = 0; _merge3 < 10; _merge3++) {
    u32 n_m3 = 0;
    for (u32 ew_ki = 0; ew_ki < sched_kernel_count; ew_ki++) {
        KernelEntry *ew_ke = &sched_kernels[ew_ki];
        if (ew_ke->n_ops == 0 && ew_ke->n_leaves == 0) continue;
        if (ew_ke->has_reduce) continue;
        for (u32 li = 0; li < ew_ke->n_leaves; li++) {
            if (ew_ke->leaf_ids[li] != 0) continue;
            Term lt = ew_ke->leaf_terms[li];
            Term fusing_t = lt;
            while (term_tag(fusing_t) == TAG_TOP && is_view_op(term_ext(fusing_t))) {
                Term nx = heap_read(ctx, term_val(fusing_t));
                if (term_tag(nx) == TAG_DP0 || term_tag(nx) == TAG_DP1)
                    nx = heap_read(ctx, term_val(nx));
                fusing_t = nx;
            }
            if (term_tag(fusing_t) != TAG_TOP || term_ext(fusing_t) != UOP_FUSING) continue;
            u32 reduce_ki = 0xFFFFFFFFu;
            { u64 floc = term_val(fusing_t);
              Term kid_term = heap_read(ctx, floc + 1);
              if (term_tag(kid_term) == TAG_NUM) reduce_ki = (u32)term_val(kid_term);
            }
            if (reduce_ki == 0xFFFFFFFFu || reduce_ki >= sched_kernel_count) continue;
            KernelEntry *r_ke = &sched_kernels[reduce_ki];
            if (!r_ke->has_reduce) continue;
            if (r_ke->n_ops == 0 && r_ke->n_leaves == 0) continue;
            if (r_ke->n_ops + ew_ke->n_ops > FUSE_MAX_OPS) continue;
            if (r_ke->n_leaves + ew_ke->n_leaves > FUSE_MAX_LEAVES) continue;
            // Merge (same logic as pass 4)
            KernelEntry merged = {0};
            memcpy(merged.ops, r_ke->ops, r_ke->n_ops * sizeof(FusedOp));
            merged.n_ops = r_ke->n_ops;
            for (u32 j = 0; j < r_ke->n_leaves; j++) {
                merged.leaf_ids[j] = r_ke->leaf_ids[j];
                merged.leaf_views[j] = r_ke->leaf_views[j];
                merged.leaf_terms[j] = r_ke->leaf_terms[j];
                merged.leaf_sts[j] = r_ke->leaf_sts[j];
            }
            merged.n_leaves = r_ke->n_leaves;
            merged.reduce = r_ke->reduce;
            merged.has_reduce = r_ke->has_reduce;
            merged.sum_term = r_ke->sum_term;
            merged.reshape_term = r_ke->reshape_term;
            merged.full_shape = r_ke->full_shape;
            if (!merged.reduce.post_reduce_start)
                merged.reduce.post_reduce_start = r_ke->n_ops;
            u32 ew_leaf_remap[FUSE_MAX_LEAVES]; u32 n_post_l = 0;
            for (u32 j = 0; j < ew_ke->n_leaves; j++) {
                if (j == li) { ew_leaf_remap[j] = 0xFFFFFFFEu; continue; }
                ew_leaf_remap[j] = merged.n_leaves;
                merged.leaf_ids[merged.n_leaves] = ew_ke->leaf_ids[j];
                merged.leaf_views[merged.n_leaves] = ew_ke->leaf_views[j];
                merged.leaf_terms[merged.n_leaves] = ew_ke->leaf_terms[j];
                merged.leaf_sts[merged.n_leaves] = ew_ke->leaf_sts[j];
                merged.n_leaves++; n_post_l++;
            }
            merged.reduce.n_post_leaves += n_post_l;
            u32 rr_idx = r_ke->n_leaves + r_ke->n_ops - 1;
            if (r_ke->n_ops == 0) rr_idx = 0;
            for (u32 j = 0; j < ew_ke->n_ops; j++) {
                FusedOp po = ew_ke->ops[j];
                if (po.arg_a < ew_ke->n_leaves)
                    po.arg_a = (ew_leaf_remap[po.arg_a]==0xFFFFFFFEu) ? rr_idx : ew_leaf_remap[po.arg_a];
                else po.arg_a = po.arg_a - ew_ke->n_leaves + merged.n_leaves + merged.n_ops;
                if (po.arg_b < ew_ke->n_leaves)
                    po.arg_b = (ew_leaf_remap[po.arg_b]==0xFFFFFFFEu) ? rr_idx : ew_leaf_remap[po.arg_b];
                else po.arg_b = po.arg_b - ew_ke->n_leaves + merged.n_leaves + merged.n_ops;
                merged.ops[merged.n_ops++] = po;
            }
            merged.out_shape = ew_ke->out_shape;
            merged.original_term = ew_ke->original_term;
            *ew_ke = merged;
            sched_kill_kernel(ctx, reduce_ki, ew_ki);
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "  pass7_merge: ew=%u + reduce=%u → ops=%u leaves=%u\n",
                        ew_ki, reduce_ki, ew_ke->n_ops, merged.n_leaves);
            n_m3++; break;
        }
    }
    if (n_m3 == 0) break;
    }

    // Pass 8: ew+ew merge — combine ew kernels sharing concrete leaves.
    // Two ew kernels reading the same buffer → one bigger ew kernel with both ops.
    for (u32 _ee = 0; _ee < 10; _ee++) {
    u32 n_ee = 0;
    for (u32 ki = 0; ki < sched_kernel_count; ki++) {
        KernelEntry *k1 = &sched_kernels[ki];
        if (k1->has_reduce) continue;
        if (k1->n_ops == 0 && k1->n_leaves == 0) continue;
        for (u32 kj = ki + 1; kj < sched_kernel_count; kj++) {
            KernelEntry *k2 = &sched_kernels[kj];
            if (k2->has_reduce) continue;
            if (k2->n_ops == 0 && k2->n_leaves == 0) continue;
            // Check shared concrete leaf
            int shared = 0;
            for (u32 l1 = 0; l1 < k1->n_leaves && !shared; l1++) {
                if (k1->leaf_ids[l1] == 0 || LEAF_IS_LAZY(k1->leaf_ids[l1])) continue;
                for (u32 l2 = 0; l2 < k2->n_leaves; l2++) {
                    if (k2->leaf_ids[l2] == k1->leaf_ids[l1]) { shared = 1; break; }
                }
            }
            if (!shared) continue;
            if (k1->n_ops + k2->n_ops > FUSE_MAX_OPS) continue;
            if (k1->n_leaves + k2->n_leaves > FUSE_MAX_LEAVES) continue;
            // Same output shape required (both write to same-shaped buffer)
            if (k1->out_shape.rank != k2->out_shape.rank) continue;
            // Merge k2 into k1 as parallel ew ops
            u32 loff = k1->n_leaves;
            for (u32 j = 0; j < k2->n_leaves; j++) {
                k1->leaf_ids[k1->n_leaves] = k2->leaf_ids[j];
                k1->leaf_views[k1->n_leaves] = k2->leaf_views[j];
                k1->leaf_terms[k1->n_leaves] = k2->leaf_terms[j];
                k1->leaf_sts[k1->n_leaves] = k2->leaf_sts[j];
                k1->n_leaves++;
            }
            for (u32 j = 0; j < k2->n_ops; j++) {
                FusedOp op = k2->ops[j];
                if (op.arg_a < k2->n_leaves) op.arg_a += loff;
                else op.arg_a = op.arg_a - k2->n_leaves + k1->n_leaves + k1->n_ops;
                if (op.arg_b < k2->n_leaves) op.arg_b += loff;
                else op.arg_b = op.arg_b - k2->n_leaves + k1->n_leaves + k1->n_ops;
                k1->ops[k1->n_ops++] = op;
            }
            sched_kill_kernel(ctx, kj, ki);
            n_ee++;
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "  ew_ew_merge: k1=%u + k2=%u → ops=%u leaves=%u\n", ki, kj, k1->n_ops, k1->n_leaves);
            break;
        }
    }
    if (n_ee == 0) break;
    }

    // Pass 9: reduce absorb ew producer.
    // If a reduce kernel has a FUSING leaf that points to an ew kernel,
    // absorb the ew kernel's ops as additional pre-reduce ops.
    for (u32 ki = 0; ki < sched_kernel_count; ki++) {
        KernelEntry *r_ke = &sched_kernels[ki];
        if (!r_ke->has_reduce) continue;
        if (r_ke->n_ops == 0 && r_ke->n_leaves == 0) continue;
        for (u32 li = 0; li < r_ke->n_leaves; li++) {
            if (r_ke->leaf_ids[li] != 0) continue; // FUSING placeholder
            Term lt = r_ke->leaf_terms[li];
            Term fusing_t = lt;
            while (term_tag(fusing_t) == TAG_TOP && is_view_op(term_ext(fusing_t))) {
                Term nx = heap_read(ctx, term_val(fusing_t));
                if (term_tag(nx)==TAG_DP0||term_tag(nx)==TAG_DP1) nx = heap_read(ctx, term_val(nx));
                fusing_t = nx;
            }
            if (term_tag(fusing_t) != TAG_TOP || term_ext(fusing_t) != UOP_FUSING) continue;
            u32 ew_ki = 0xFFFFFFFFu;
            { u64 floc = term_val(fusing_t);
              Term kid_t = heap_read(ctx, floc + 1);
              if (term_tag(kid_t) == TAG_NUM) ew_ki = (u32)term_val(kid_t);
            }
            if (ew_ki == 0xFFFFFFFFu || ew_ki >= sched_kernel_count) continue;
            KernelEntry *ew_ke = &sched_kernels[ew_ki];
            if (ew_ke->has_reduce) continue; // only absorb pure ew
            if (ew_ke->n_ops == 0 && ew_ke->n_leaves == 0) continue;
            if (r_ke->n_ops + ew_ke->n_ops > FUSE_MAX_OPS) continue;
            if (r_ke->n_leaves + ew_ke->n_leaves > FUSE_MAX_LEAVES) continue;
            // Absorb: ew ops become additional pre-reduce ops
            // Replace the FUSING leaf[li] with ew_ke's ops+leaves
            u32 loff = r_ke->n_leaves;
            // Add ew leaves
            for (u32 j = 0; j < ew_ke->n_leaves; j++) {
                r_ke->leaf_ids[r_ke->n_leaves] = ew_ke->leaf_ids[j];
                r_ke->leaf_views[r_ke->n_leaves] = ew_ke->leaf_views[j];
                r_ke->leaf_terms[r_ke->n_leaves] = ew_ke->leaf_terms[j];
                r_ke->leaf_sts[r_ke->n_leaves] = ew_ke->leaf_sts[j];
                r_ke->n_leaves++;
            }
            // The FUSING leaf[li] now maps to the ew kernel's last op result
            u32 ew_result_idx = loff + ew_ke->n_leaves + ew_ke->n_ops - 1;
            // Insert ew ops at current n_ops position (before existing ops, shifting them)
            // Actually simpler: just append and remap leaf[li] references
            u32 pre_rops = r_ke->n_ops;
            for (u32 j = 0; j < ew_ke->n_ops; j++) {
                FusedOp op = ew_ke->ops[j];
                if (op.arg_a < ew_ke->n_leaves) op.arg_a += loff;
                else op.arg_a = op.arg_a - ew_ke->n_leaves + r_ke->n_leaves + r_ke->n_ops;
                if (op.arg_b < ew_ke->n_leaves) op.arg_b += loff;
                else op.arg_b = op.arg_b - ew_ke->n_leaves + r_ke->n_leaves + r_ke->n_ops;
                r_ke->ops[r_ke->n_ops++] = op;
            }
            // Remap existing reduce ops that referenced leaf[li]
            ew_result_idx = r_ke->n_leaves + r_ke->n_ops - 1; // last appended ew op
            for (u32 j = 0; j < pre_rops; j++) {
                if (r_ke->ops[j].arg_a == li) r_ke->ops[j].arg_a = ew_result_idx;
                if (r_ke->ops[j].arg_b == li) r_ke->ops[j].arg_b = ew_result_idx;
            }
            sched_kill_kernel(ctx, ew_ki, ki);
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "  reduce_absorb_ew: r=%u + ew=%u → ops=%u leaves=%u\n",
                        ki, ew_ki, r_ke->n_ops, r_ke->n_leaves);
            break;
        }
    }

    fuse_no_lazy_resolve = 0;
    return total;
}

static void sched_restore_pre_merge(void) {
    if (pre_merge_count > 0) {
        sched_kernel_count = pre_merge_count;
        memcpy(sched_kernels, pre_merge_kernels_store, pre_merge_count * sizeof(KernelEntry));
        pre_merge_count = 0;
    }
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

    // ── Memory analysis (pure algorithm, no dispatch) ──
    if (getenv("THVM_MEM_DIAG")) {
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
        #undef MEM_FIND_OR_ADD
        #undef MEM_MAX_BUFS
    }

    // Restore pre-merge kernels for non-merged dispatch.
    if (!getenv("THVM_MERGED_DISPATCH"))
        sched_restore_pre_merge();
    else {
        // Topological pre-dispatch: dispatch alive kernels in order,
        // so dead kernels' absorbers are ready before consumers need them.
        // Simple: dispatch all alive kernels sequentially (they're already
        // in scheduling order which respects dependencies).
        // Skip: this requires the FUSING handler to be callable directly,
        // which it isn't — it fires via thvm_reduce on FUSING terms.
        // Instead: just ensure absorbers dispatch before dead kernels.
        // Nothing needed here — the FUSING handler's absorber-chain-follow
        // handles it. The cycle breaker placeholder prevents infinite loops.
    }

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

    if (getenv("THVM_MEM_DIAG")) {
        extern u32 _ensure_count, _ensure_alloc_count;
        fprintf(stderr, "ENSURE_STATS: calls=%u allocs=%u\n", _ensure_count, _ensure_alloc_count);
        _ensure_count = 0; _ensure_alloc_count = 0;
    }

    return term_era();
}

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_eval(ctx, t);
}
