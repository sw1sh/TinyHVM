// Forward declarations (defined in debug/dump.c, included after this file)
static void thvm_heap_dot(TinyHVM *ctx, const char *path);
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root);
static void thvm_step_graph_eval_begin(TinyHVM *ctx, Term root);
static void thvm_step_graph_after_interaction(TinyHVM *ctx, Term before, Term root);
static void thvm_step_graph_finalize(TinyHVM *ctx);
static void thvm_step_graph_set_before_grad_y(Term y);
static void thvm_step_graph_set_before_era_payload(Term payload);
static void thvm_step_graph_set_before_top_era(int had_era);

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
static u64 phase1_root_slot = 0;

static u32 phase1_top_arity(u32 ext) {
    if (ext == UOP_FUSING) return 0;
    if (ext == UOP_WHERE || ext == UOP_IFZ) return 3;
    if (ext == UOP_GRAD) return 2;
    if (!is_binary(ext) && is_elementwise(ext)) return 1;
    return 2;
}

static u32 phase1_term_arity(Term t);
static int phase1_has_parent_ref(TinyHVM *ctx, Term target);

static int phase1_top_has_era_arg(TinyHVM *ctx, Term t, u64 *slot_out, Term *term_out) {
    if (term_tag(t) != TAG_TOP) return 0;
    u64 loc = term_val(t);
    u32 arity = phase1_top_arity(term_ext(t));
    for (u32 i = 0; i < arity; i++) {
        Term child = heap_read(ctx, loc + i);
        if (term_tag(child) == TAG_ERA && term_val(child) != 0) {
            if (slot_out) *slot_out = loc + i;
            if (term_out) *term_out = child;
            return 1;
        }
    }
    return 0;
}

static int phase1_top_has_add_zero_arg(TinyHVM *ctx, Term t, u64 *slot_out, Term *term_out) {
    if (term_tag(t) != TAG_TOP || term_ext(t) != UOP_ADD) return 0;
    u64 loc = term_val(t);
    for (u32 i = 0; i < 2; i++) {
        Term child = heap_read(ctx, loc + i);
        if (term_tag(child) == TAG_NUM && term_as_f32(child) == 0.0f) {
            if (slot_out) *slot_out = loc + i;
            if (term_out) *term_out = child;
            return 1;
        }
    }
    return 0;
}

static int phase1_term_maybe_active(TinyHVM *ctx, Term t) {
    u8 tag = term_tag(t);
    if (tag == TAG_TOP) {
        return term_ext(t) == UOP_GRAD ||
               phase1_top_has_era_arg(ctx, t, NULL, NULL) ||
               phase1_top_has_add_zero_arg(ctx, t, NULL, NULL);
    }
    if (tag == TAG_ERA) return term_val(t) != 0 && !phase1_has_parent_ref(ctx, t);
    if (tag == TAG_TEN || tag == TAG_NUM || tag == TAG_LAM || tag == TAG_SUP ||
        tag == TAG_BRI || tag == TAG_MAT || tag == TAG_ANY || tag == TAG_USP)
        return 0;
    return 1;
}

static int phase1_term_first_reachable_occurrence(TinyHVM *ctx, u64 h, Term t, const u8 *reach) {
    for (u64 i = 1; i < h; i++) {
        if (reach && !reach[i]) continue;
        if (ctx->heap[i] == t) return 0;
    }
    return 1;
}

static u32 phase1_term_arity(Term t) {
    u8 tag = term_tag(t);
    u32 ext = term_ext(t);
    switch (tag) {
        case TAG_TOP:
            if (ext == UOP_FUSING) return 0;
            if (ext == UOP_WHERE || ext == UOP_IFZ) return 3;
            if (ext == UOP_GRAD) return 2;
            if (!is_binary(ext) && is_elementwise(ext)) return 1;
            return 2;
        case TAG_APP:
        case TAG_LAM:
        case TAG_BRI:
        case TAG_SUP:
        case TAG_USP:
        case TAG_OP2:
        case TAG_EQL:
        case TAG_AND:
        case TAG_OR:
        case TAG_MAT:
        case TAG_ANN:
            return 2;
        case TAG_DSU:
        case TAG_DDU:
            return 3;
        case TAG_DP0:
        case TAG_DP1:
        case TAG_UDP:
        case TAG_ERA:
        case TAG_VAR:
        case TAG_INC:
            return 1;
        default:
            return 0;
    }
}

static int phase1_has_parent_ref(TinyHVM *ctx, Term target) {
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term p = ctx->heap[h];
        u32 ar = phase1_term_arity(p);
        u64 loc = term_val(p);
        if (ar == 0 || loc == 0 || loc + ar > ctx->heap_pos) continue;
        for (u32 i = 0; i < ar; i++) {
            if (heap_read(ctx, loc + i) == target) return 1;
        }
    }
    return 0;
}

static void phase1_mark_reachable_slots(TinyHVM *ctx, Term root, u8 *reach) {
    if (!reach || ctx->heap_pos == 0) return;
    memset(reach, 0, (size_t)ctx->heap_pos);
    u8 *seen_slot = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    u8 *seen_dup  = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    u64 work_cap = ctx->heap_pos ? (ctx->heap_pos * 8) : 0;
    Term *work = work_cap ? (Term *)malloc(sizeof(Term) * (size_t)work_cap) : NULL;
    u64 wp = 0;
    #define P1_PUSH(_tt) do { \
        if (work && wp < work_cap) work[wp++] = (_tt); \
    } while (0)

    if (!(term_tag(root) == TAG_ERA && term_val(root) == 0)) P1_PUSH(root);
    if (phase1_root_slot > 0 && phase1_root_slot < ctx->heap_pos)
        P1_PUSH(heap_read(ctx, phase1_root_slot));
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) == TAG_ERA && term_val(ht) != 0) {
            reach[h] = 1;
            P1_PUSH(ht);
        }
    }

    while (work && wp > 0) {
        Term tt = work[--wp];
        u8 tg = term_tag(tt);
        u64 tv = term_val(tt);

        if (tg == TAG_DP0 || tg == TAG_DP1) {
            u64 dl = tv;
            if (dl == 0 || dl >= ctx->heap_pos || (seen_dup && seen_dup[dl])) continue;
            if (seen_dup) seen_dup[dl] = 1;
            reach[dl] = 1;
            P1_PUSH(heap_read(ctx, dl));
            continue;
        }

        if (tg == TAG_ERA) {
            if (tv == 0 || tv >= ctx->heap_pos) continue;
            if (!seen_slot || !seen_slot[tv]) {
                if (seen_slot) seen_slot[tv] = 1;
                reach[tv] = 1;
                P1_PUSH(heap_read(ctx, tv));
            }
            continue;
        }

        if (tg == TAG_CTR) {
            if (tv == 0 || tv >= ctx->heap_pos) continue;
            if (!seen_slot || !seen_slot[tv]) {
                if (seen_slot) seen_slot[tv] = 1;
                reach[tv] = 1;
                P1_PUSH(heap_read(ctx, tv));
            }
            Term nt = heap_read(ctx, tv);
            u32 n = (term_tag(nt) == TAG_NUM) ? (u32)term_val(nt) : 0;
            if (n > 64) n = 64;
            for (u32 i = 0; i < 2 * n; i++) {
                u64 p = tv + 1 + i;
                if (p < ctx->heap_pos && (!seen_slot || !seen_slot[p])) {
                    if (seen_slot) seen_slot[p] = 1;
                    reach[p] = 1;
                    P1_PUSH(heap_read(ctx, p));
                }
            }
            continue;
        }

        u32 ar = phase1_term_arity(tt);
        for (u32 i = 0; i < ar; i++) {
            u64 p = tv + i;
            if (p < ctx->heap_pos && (!seen_slot || !seen_slot[p])) {
                if (seen_slot) seen_slot[p] = 1;
                reach[p] = 1;
                P1_PUSH(heap_read(ctx, p));
            }
        }
    }

    free(work);
    free(seen_dup);
    free(seen_slot);
    #undef P1_PUSH
}

static int phase1_term_is_whnf_atom(Term t) {
    u8 tag = term_tag(t);
    return tag == TAG_TEN || tag == TAG_NUM || tag == TAG_LAM || tag == TAG_SUP ||
           tag == TAG_BRI || tag == TAG_MAT || tag == TAG_ANY || tag == TAG_USP;
}

static int phase1_grad_y_ready(Term t) {
    u8 tag = term_tag(t);
    return tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM ||
           tag == TAG_SUP || tag == TAG_TOP;
}

static int phase1_top_arg0_ready(Term t) {
    u8 tag = term_tag(t);
    return tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM || tag == TAG_SUP;
}

static int phase1_dp_can_fire_on(Term t) {
    switch (term_tag(t)) {
        case TAG_SUP:
        case TAG_LAM:
        case TAG_BRI:
        case TAG_ANN:
        case TAG_TEN:
        case TAG_ERA:
        case TAG_NUM:
        case TAG_ANY:
        case TAG_CTR:
        case TAG_TOP:
        case TAG_USP:
        case TAG_OP2:
        case TAG_APP:
        case TAG_EQL:
        case TAG_AND:
        case TAG_OR:
        case TAG_MAT:
            return 1;
        default:
            return 0;
    }
}

static int thvm_phase1_predict_next_redex(TinyHVM *ctx, Term t, Term *out_before, Term *out_whnf) {
    u8 tag = term_tag(t);
    if (phase1_term_is_whnf_atom(t)) {
        if (out_whnf) *out_whnf = t;
        return 0;
    }

    if (tag == TAG_ERA) {
        if (term_val(t) == 0) {
            if (out_whnf) *out_whnf = t;
            return 0;
        }
        if (out_before) *out_before = t;
        return 1;
    }

    if (tag == TAG_TOP) {
        u32 uop = term_ext(t);
        u64 loc = term_val(t);
        if (uop == UOP_GRAD) {
            Term y = heap_read(ctx, loc + 0);
            Term wy = y;
            if (thvm_phase1_predict_next_redex(ctx, y, out_before, &wy))
                return 1;
            if (phase1_grad_y_ready(wy)) {
                if (out_before) *out_before = t;
                return 1;
            }
            if (out_whnf) *out_whnf = t;
            return 0;
        }

        if (phase1_top_has_era_arg(ctx, t, NULL, NULL) ||
            phase1_top_has_add_zero_arg(ctx, t, NULL, NULL)) {
            if (out_before) *out_before = t;
            return 1;
        }
        if (out_whnf) *out_whnf = t;
        return 0;
    }

    if (tag == TAG_DP0 || tag == TAG_DP1) {
        u64 loc = term_val(t);
        if (loc == 0 || loc >= ctx->heap_pos) {
            if (out_whnf) *out_whnf = t;
            return 0;
        }
        Term val = heap_read(ctx, loc);
        Term wv = val;
        if (thvm_phase1_predict_next_redex(ctx, val, out_before, &wv))
            return 1;
        if (phase1_dp_can_fire_on(wv)) {
            if (out_before) *out_before = t;
            return 1;
        }
        if (out_whnf) *out_whnf = t;
        return 0;
    }

    if (out_whnf) *out_whnf = t;
    return 0;
}

static int thvm_phase1_find_next_actual(TinyHVM *ctx, Term root,
                                        u64 *out_source_slot, Term *out_before) {
    Term before = 0;
    Term whnf = root;
    if (thvm_phase1_predict_next_redex(ctx, root, &before, &whnf)) {
        if (out_source_slot) *out_source_slot = phase1_root_slot;
        if (out_before) *out_before = before;
        return 1;
    }

    u8 *reach = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    phase1_mark_reachable_slots(ctx, root, reach);
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        if (h == phase1_root_slot) continue;
        if (reach && !reach[h]) continue;
        Term ht = ctx->heap[h];
        if (!phase1_term_maybe_active(ctx, ht)) continue;
        if ((term_tag(ht) == TAG_DP0 || term_tag(ht) == TAG_DP1) &&
            !phase1_term_first_reachable_occurrence(ctx, h, ht, reach)) {
            continue;
        }
        if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_GRAD &&
            !phase1_term_first_reachable_occurrence(ctx, h, ht, reach)) {
            continue;
        }
        Term w = ht;
        if (thvm_phase1_predict_next_redex(ctx, ht, &before, &w)) {
            if (out_source_slot) *out_source_slot = h;
            if (out_before) *out_before = before;
            free(reach);
            return 1;
        }
    }
    free(reach);
    return 0;
}

static int thvm_phase1_fire_one(TinyHVM *ctx, Term in, Term *out_term, Term *out_before) {
    if (term_tag(in) == TAG_TOP && term_ext(in) == UOP_GRAD) {
        u64 gl = term_val(in);
        if (gl + 1 < ctx->heap_pos) thvm_step_graph_set_before_grad_y(heap_read(ctx, gl + 0));
        else thvm_step_graph_set_before_grad_y(term_era());
    } else {
        thvm_step_graph_set_before_grad_y(term_era());
    }
    if (term_tag(in) == TAG_TOP && term_ext(in) != UOP_GRAD)
        thvm_step_graph_set_before_top_era(phase1_top_has_era_arg(ctx, in, NULL, NULL));
    else
        thvm_step_graph_set_before_top_era(0);
    if (term_tag(in) == TAG_ERA) {
        u64 el = term_val(in);
        Term payload = (el > 0 && el < ctx->heap_pos) ? heap_read(ctx, el) : term_era();
        thvm_step_graph_set_before_era_payload(payload);
    } else {
        thvm_step_graph_set_before_era_payload(term_era());
    }

    struct InteractionTrace tr = {0};
    struct InteractionTrace *saved_buf = ctx->trace_buf;
    u32 saved_count = ctx->trace_count;
    u32 saved_cap = ctx->trace_cap;
    u8 saved_en = ctx->trace_enabled;

    ctx->trace_buf = &tr;
    ctx->trace_count = 0;
    ctx->trace_cap = 1;
    ctx->trace_enabled = 1;

    u64 itrs_before = ctx->itrs;
    Term r = thvm_reduce_steps(ctx, in, 1);
    int traced = (ctx->steps_taken > 0) && (ctx->trace_count > 0);
    int fired = traced || (ctx->itrs != itrs_before);
    Term before = in;
    if (traced) {
        before = term_new((u8)tr.before_tag, tr.before_ext, tr.before_loc);
    }

    ctx->trace_buf = saved_buf;
    ctx->trace_count = saved_count;
    ctx->trace_cap = saved_cap;
    ctx->trace_enabled = saved_en;

    if (out_term) *out_term = r;
    if (out_before) *out_before = before;
    return fired;
}

static Term thvm_phase1_seed_root_grad(TinyHVM *ctx, Term t) {
    // Keep one mirrored heap cell for the current phase-1 root term.
    // Dumping code is heap-driven; without this, newly returned roots can disappear.
    if (phase1_root_slot == 0 || phase1_root_slot >= ctx->heap_pos)
        phase1_root_slot = heap_alloc(ctx, 1);
    heap_set(ctx, phase1_root_slot, t);
    return t;
}

static Term thvm_phase1_structural_nf(TinyHVM *ctx, Term t) {
    u8 *reach = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    for (u32 guard = 0; guard < 100000; guard++) {
        Term tr = t;
        Term before = t;
        if (phase1_term_maybe_active(ctx, t) &&
            thvm_phase1_fire_one(ctx, t, &tr, &before)) {
            t = tr;
            if (phase1_root_slot > 0 && phase1_root_slot < ctx->heap_pos)
                heap_set(ctx, phase1_root_slot, t);
            thvm_step_graph_after_interaction(ctx, before, t);
            continue;
        }

        // Otherwise fire one interaction from phase-1 heap agents.
        // No priority classes here: with correct local rules, order should
        // not change structural validity.
        int fired = 0;
        phase1_mark_reachable_slots(ctx, t, reach);
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            if (h == phase1_root_slot) continue;
            if (reach && !reach[h]) continue;
            Term ht = ctx->heap[h];
            if (!phase1_term_maybe_active(ctx, ht)) continue;
            if ((term_tag(ht) == TAG_DP0 || term_tag(ht) == TAG_DP1) &&
                !phase1_term_first_reachable_occurrence(ctx, h, ht, reach)) {
                continue;
            }
            if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_GRAD &&
                !phase1_term_first_reachable_occurrence(ctx, h, ht, reach)) {
                continue;
            }
            Term hr = ht;
            Term before_h = ht;
            if (thvm_phase1_fire_one(ctx, ht, &hr, &before_h)) {
                if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_GRAD) {
                    for (u64 i = 1; i < ctx->heap_pos; i++) {
                        if (ctx->heap[i] == ht) ctx->heap[i] = hr;
                    }
                } else {
                    ctx->heap[h] = hr;
                }
                if (phase1_root_slot > 0 && phase1_root_slot < ctx->heap_pos)
                    heap_set(ctx, phase1_root_slot, t);
                thvm_step_graph_after_interaction(ctx, before_h, t);
                fired = 1;
                break;
            }
        }
        if (!fired) break;
    }
    free(reach);
    return t;
}

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

    if (getenv("THVM_STEP_GRAPH")) {
        phase1_root_slot = 0;
        t = thvm_phase1_seed_root_grad(ctx, t);
        thvm_step_graph_eval_begin(ctx, t);
        t = thvm_phase1_structural_nf(ctx, t);
        thvm_step_graph_finalize(ctx);
        phase1_root_slot = 0;
        return t;
    }

    if (getenv("THVM_GRAPH")) thvm_heap_dot_root(ctx, "/tmp/thvm_0_pre_reduce.dot", t);
    t = thvm_reduce(ctx, t);
    if (getenv("THVM_GRAPH")) thvm_heap_dot_root(ctx, "/tmp/thvm_1_post_reduce.dot", t);

    sched_all(ctx);
    if (getenv("THVM_GRAPH")) thvm_heap_dot(ctx, "/tmp/thvm_2_post_sched.dot");

    t = thvm_reduce(ctx, t);
    if (getenv("THVM_GRAPH")) thvm_heap_dot(ctx, "/tmp/thvm_3_post_dispatch.dot");
    return t;
}

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_eval(ctx, t);
}
