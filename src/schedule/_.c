// Forward declarations (defined in debug/dump.c, included after this file)
static void thvm_heap_dot(TinyHVM *ctx, const char *path);
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root);
static void thvm_heap_dot_set_sched_kernels(int enabled);
static void thvm_step_graph_eval_begin(TinyHVM *ctx, Term root);
static void thvm_step_graph_after_interaction(TinyHVM *ctx, Term before, Term root);
static void thvm_step_graph_finalize(TinyHVM *ctx);
static void thvm_step_graph_set_before_grad_y(Term y);
static void thvm_step_graph_set_before_era_payload(Term payload);
static void thvm_step_graph_set_before_top_era(int had_era);
static void thvm_step_graph_set_before_top_add_zero(int had_add_zero);

// schedule/_.c — Unified IC eval: reduce is always pure phase 1.
//
// thvm_reduce(t) — pure IC reduction. Stops at WNF:
//   GRAD fires, combinators fire, compute ops stay TAG_TOP (WNF).
//
// thvm_eval(t) — injects scheduling signals after reduce reaches WNF:
//   1. thvm_reduce(t)                  — phase 1: pure IC to WNF
//   2. thvm_reduce(UOP_FUSE(t))        — phase 2: fusion as interaction
//   3. thvm_reduce(UOP_SCHED(t))       — phase 3: plan + dispatch as interaction
//
// The choice: call thvm_reduce for phase 1 only, call thvm_eval for full pipeline.
// UOP_KERNEL interact handler deduplicates by kid: same kid fired only once.

int fuse_no_lazy_resolve = 0;
int _assign_dispatch_enabled = 0;

// Global kernel table: scheduler writes, UOP_KERNEL handler reads.
KernelEntry sched_kernels[SCHED_MAX_KERNELS];
u32 sched_kernel_count = 0;

// kid_results: TAG_TEN result for each dispatched kid (ERA = not yet dispatched).
// Shared with UOP_KERNEL handler in tensor_ops.c.
Term kid_results[SCHED_MAX_KERNELS];
static u64 phase1_root_slot = 0;

typedef struct {
    Backend *backend;
    u32      buf_id;
    u64      bytes;
    u8       detached;
} SchedSlot;

static u16 sched_output_remaining_uses[SCHED_MAX_KERNELS];
static u8  sched_output_pinned[SCHED_MAX_KERNELS];
static u32 sched_exec_order[SCHED_MAX_KERNELS];
static u32 sched_exec_count = 0;
static SchedSlot sched_slots[SCHED_MAX_KERNELS + 1];
static u32 sched_slot_count = 0;

static void sched_backend_release_buf(Backend *backend, u32 buf_id) {
    if (!backend || buf_id == 0) return;
    if (backend->buf_decref) backend->buf_decref(buf_id);
    else if (backend->buf_free) backend->buf_free(buf_id);
}

static void sched_planner_release_detached_slots(void) {
    for (u32 i = 1; i <= sched_slot_count; i++) {
        SchedSlot *slot = &sched_slots[i];
        if (!slot->detached || slot->buf_id == 0) continue;
        sched_backend_release_buf(slot->backend, slot->buf_id);
        slot->buf_id = 0;
        slot->detached = 0;
    }
}

static void sched_planner_reset_state(void) {
    sched_planner_release_detached_slots();
    memset(sched_output_remaining_uses, 0, sizeof(sched_output_remaining_uses));
    memset(sched_output_pinned, 0, sizeof(sched_output_pinned));
    memset(sched_exec_order, 0, sizeof(sched_exec_order));
    memset(sched_slots, 0, sizeof(sched_slots));
    sched_exec_count = 0;
    sched_slot_count = 0;
}

static u32 sched_slot_bind_output(TinyHVM *ctx, u32 slot_id, u32 raw_tid, u32 out_tid) {
    if (slot_id == 0 || slot_id > sched_slot_count || raw_tid == 0) return 0;
    SchedSlot *slot = &sched_slots[slot_id];
    if (!slot->backend) slot->backend = ctx_default_backend(ctx);
    if (!slot->backend || slot->bytes == 0) return 0;
    if (slot->buf_id == 0) {
        slot->buf_id = slot->backend->buf_alloc(slot->bytes);
        if (slot->buf_id == 0) return 0;
    }
    slot->detached = 0;
    TensorMeta *raw_m = &ctx->tensors[raw_tid];
    raw_m->backend = slot->backend;
    raw_m->buf_id = slot->buf_id;
    raw_m->planned_slot = slot_id;
    if (!out_tid || out_tid == raw_tid) return slot->buf_id;
    TensorMeta *out_m = &ctx->tensors[out_tid];
    out_m->backend = raw_m->backend;
    out_m->buf_id = slot->buf_id;
    out_m->planned_slot = slot_id;
    if (out_m->backend && out_m->backend->buf_incref)
        out_m->backend->buf_incref(slot->buf_id);
    return slot->buf_id;
}

static void sched_release_output_kid(TinyHVM *ctx, u32 kid) {
    if (kid >= sched_kernel_count) return;
    KernelEntry *ke = &sched_kernels[kid];
    u32 slot_id = ke->output_slot;
    u32 raw_tid = ke->raw_output_tid ? ke->raw_output_tid : ke->output_tid;
    if (raw_tid == 0 || raw_tid >= ctx->tensor_count) return;
    TensorMeta *raw_m = &ctx->tensors[raw_tid];
    if (raw_m->buf_id == 0 || !raw_m->backend) return;

    if (ke->output_tid && ke->output_tid != raw_tid && ke->output_tid < ctx->tensor_count) {
        TensorMeta *out_m = &ctx->tensors[ke->output_tid];
        if (out_m->buf_id && out_m->backend && out_m->backend->buf_decref)
            out_m->backend->buf_decref(out_m->buf_id);
        out_m->buf_id = 0;
    }

    if (slot_id > 0 && slot_id <= sched_slot_count) {
        SchedSlot *slot = &sched_slots[slot_id];
        slot->backend = raw_m->backend;
        slot->buf_id = raw_m->buf_id;
        slot->bytes = (u64)raw_m->view.numel * dtype_size(raw_m->dtype);
        slot->detached = 1;
    } else {
        sched_backend_release_buf(raw_m->backend, raw_m->buf_id);
    }

    raw_m->buf_id = 0;
}

static const char *thvm_graph_dir(void) {
    const char *dir = getenv("THVM_GRAPH_DIR");
    return (dir && dir[0]) ? dir : "/tmp";
}

static void thvm_graph_dump_path(char *buf, size_t nbuf, const char *name) {
    snprintf(buf, nbuf, "%s/%s", thvm_graph_dir(), name);
}

static u32 phase1_top_arity(u32 ext) {
    if (ext == UOP_KERNEL) return 0;
    if (ext == UOP_WHERE || ext == UOP_IFZ) return 3;
    if (ext == UOP_GRAD) return 2;
    if (ext == UOP_LOG_PRINT) return 1;
    if (ext == UOP_DETACH) return 1;
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
        if (term_tag(child) == TAG_ERA) {
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

static int phase1_top_direct_uop(u32 uop) {
    return uop == UOP_ASSIGN || uop == UOP_GRAD || uop == UOP_IFZ ||
           uop == UOP_LOG_PRINT || uop == UOP_TODEVICE || uop == UOP_DETACH ||
           uop == UOP_WHERE ||
           uop == UOP_KERNEL;
}

static int phase1_term_maybe_active(TinyHVM *ctx, Term t) {
    u8 tag = term_tag(t);
    if (tag == TAG_TOP) {
        return phase1_top_direct_uop(term_ext(t)) ||
               phase1_top_has_era_arg(ctx, t, NULL, NULL) ||
               phase1_top_has_add_zero_arg(ctx, t, NULL, NULL);
    }
    if (tag == TAG_ERA) return term_val(t) != 0 && !phase1_has_parent_ref(ctx, t);
    if (tag == TAG_TEN || tag == TAG_NUM || tag == TAG_LAM || tag == TAG_SUP ||
        tag == TAG_BRI || tag == TAG_MAT || tag == TAG_ANY || tag == TAG_USP)
        return 0;
    return 1;
}

static int phase1_term_needs_global_cleanup(TinyHVM *ctx, Term t) {
    u8 tag = term_tag(t);
    if (tag == TAG_ERA) return term_val(t) != 0 && !phase1_has_parent_ref(ctx, t);
    if (tag != TAG_TOP) return 0;
    return phase1_top_has_era_arg(ctx, t, NULL, NULL) ||
           phase1_top_has_add_zero_arg(ctx, t, NULL, NULL);
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
            return phase1_top_arity(ext);
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
        case TAG_CTR:
            return ext;
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

static int phase1_log_print_arg_ready(Term t) {
    return phase1_top_arg0_ready(t);
}

static int phase1_app_mat_arg_ready(Term t) {
    u8 tag = term_tag(t);
    return tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM ||
           tag == TAG_SUP || tag == TAG_CTR || tag == TAG_ANY;
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

    if (tag == TAG_REF) {
        if (out_before) *out_before = t;
        return 1;
    }

    if (tag == TAG_CTR) {
        u32 ar = term_ext(t);
        u64 loc = term_val(t);
        if (ar == 0 || loc == 0 || loc >= ctx->heap_pos) {
            if (out_whnf) *out_whnf = t;
            return 0;
        }
        Term head = heap_read(ctx, loc + 0);
        Term whead = head;
        if (thvm_phase1_predict_next_redex(ctx, head, out_before, &whead))
            return 1;
        if (ar == 1 ||
            term_tag(whead) == TAG_NUM ||
            term_tag(whead) == TAG_ERA ||
            term_tag(whead) == TAG_TEN) {
            if (out_before) *out_before = t;
            return 1;
        }
        if (out_whnf) *out_whnf = t;
        return 0;
    }

    if (tag == TAG_APP) {
        u64 loc = term_val(t);
        if (loc == 0 || loc + 1 >= ctx->heap_pos) {
            if (out_whnf) *out_whnf = t;
            return 0;
        }
        Term fun = heap_read(ctx, loc + 0);
        Term wfun = fun;
        if (thvm_phase1_predict_next_redex(ctx, fun, out_before, &wfun))
            return 1;
        if (term_tag(wfun) == TAG_MAT) {
            Term arg = heap_read(ctx, loc + 1);
            Term warg = arg;
            if (thvm_phase1_predict_next_redex(ctx, arg, out_before, &warg))
                return 1;
            if (phase1_app_mat_arg_ready(warg)) {
                if (out_before) *out_before = t;
                return 1;
            }
            if (out_whnf) *out_whnf = t;
            return 0;
        }
        switch (term_tag(wfun)) {
            case TAG_SUP:
            case TAG_BRI:
            case TAG_LAM:
            case TAG_TEN:
            case TAG_NUM:
            case TAG_ERA:
            case TAG_USP:
                if (out_before) *out_before = t;
                return 1;
            default:
                break;
        }
        if (out_whnf) *out_whnf = t;
        return 0;
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

        if (uop == UOP_LOG_PRINT) {
            Term a0 = heap_read(ctx, loc + 0);
            Term wa0 = a0;
            if (thvm_phase1_predict_next_redex(ctx, a0, out_before, &wa0))
                return 1;
            if (phase1_log_print_arg_ready(wa0)) {
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

static void thvm_phase1_capture_step_before_meta(TinyHVM *ctx, Term before) {
    if (term_tag(before) == TAG_TOP && term_ext(before) == UOP_GRAD) {
        u64 gl = term_val(before);
        if (gl + 1 < ctx->heap_pos) thvm_step_graph_set_before_grad_y(heap_read(ctx, gl + 0));
        else thvm_step_graph_set_before_grad_y(term_era());
    } else {
        thvm_step_graph_set_before_grad_y(term_era());
    }
    if (term_tag(before) == TAG_TOP && term_ext(before) != UOP_GRAD)
        thvm_step_graph_set_before_top_era(phase1_top_has_era_arg(ctx, before, NULL, NULL));
    else
        thvm_step_graph_set_before_top_era(0);
    if (term_tag(before) == TAG_TOP && term_ext(before) != UOP_GRAD)
        thvm_step_graph_set_before_top_add_zero(phase1_top_has_add_zero_arg(ctx, before, NULL, NULL));
    else
        thvm_step_graph_set_before_top_add_zero(0);
    if (term_tag(before) == TAG_ERA) {
        u64 el = term_val(before);
        Term payload = (el > 0 && el < ctx->heap_pos) ? heap_read(ctx, el) : term_era();
        thvm_step_graph_set_before_era_payload(payload);
    } else {
        thvm_step_graph_set_before_era_payload(term_era());
    }
}

static int thvm_phase1_fire_one(TinyHVM *ctx, Term in, Term *out_term, Term *out_before) {
    Term predicted_before = in;
    thvm_phase1_predict_next_redex(ctx, in, &predicted_before, NULL);
    thvm_phase1_capture_step_before_meta(ctx, predicted_before);

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
    size_t reach_cap = (size_t)ctx->heap_pos;
    u8 *reach = (u8 *)calloc(reach_cap ? reach_cap : 1, 1);
    for (u32 guard = 0; guard < 100000; guard++) {
        int fired = 0;
        if ((size_t)ctx->heap_pos > reach_cap) {
            size_t new_cap = (size_t)ctx->heap_pos;
            u8 *new_reach = (u8 *)realloc(reach, new_cap);
            if (!new_reach) break;
            memset(new_reach + reach_cap, 0, new_cap - reach_cap);
            reach = new_reach;
            reach_cap = new_cap;
        }
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
        phase1_mark_reachable_slots(ctx, t, reach);
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            if (h == phase1_root_slot) continue;
            Term ht = ctx->heap[h];
            int reachable = !(reach && !reach[h]);
            if (!reachable && !phase1_term_needs_global_cleanup(ctx, ht)) continue;
            if (!phase1_term_maybe_active(ctx, ht)) continue;
            if ((term_tag(ht) == TAG_DP0 || term_tag(ht) == TAG_DP1) &&
                reachable &&
                !phase1_term_first_reachable_occurrence(ctx, h, ht, reach)) {
                continue;
            }
            if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_GRAD &&
                reachable &&
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
        if (term_tag(t) == TAG_DP0 || term_tag(t) == TAG_DP1) {
            t = heap_read(ctx, term_val(t));
            continue;
        }
        if (term_tag(t) == TAG_VAR) {
            Term sub = heap_read(ctx, term_val(t));
            if (!term_is_sub(sub)) {
                t = sub;
                continue;
            }
        }
        if (term_tag(t) != TAG_TOP || !is_view_op(term_ext(t))) break;
        t = heap_read(ctx, term_val(t));
    }
    return t;
}

static int sched_is_kernelizable_uop(u32 uop) {
    return is_elementwise(uop) || uop == UOP_CAST || uop == UOP_SUM || uop == UOP_RMAX;
}

static int sched_is_kernelizable_term(Term t) {
    return term_tag(t) == TAG_TOP && sched_is_kernelizable_uop(term_ext(t));
}

static const View *sched_term_view(TinyHVM *ctx, Term t) {
    if (term_tag(t) == TAG_TOP) return st_get(term_val(t));
    if (term_tag(t) == TAG_TEN) return tensor_view_get(&ctx->tensors[(u32)term_val(t)]);
    if (term_tag(t) == TAG_VAR) {
        u64 loc = term_val(t);
        Term sub = heap_read(ctx, loc);
        if (!term_is_sub(sub)) return sched_term_view(ctx, sub);
        return st_get(loc);
    }
    return NULL;
}

static const ShapeTracker *sched_term_tracker(TinyHVM *ctx, Term t) {
    if (term_tag(t) == TAG_TOP) return st_get_tracker(term_val(t));
    if (term_tag(t) == TAG_TEN) return tensor_st_get(&ctx->tensors[(u32)term_val(t)]);
    if (term_tag(t) == TAG_VAR) {
        u64 loc = term_val(t);
        Term sub = heap_read(ctx, loc);
        if (!term_is_sub(sub)) return sched_term_tracker(ctx, sub);
        return st_get_tracker(loc);
    }
    return NULL;
}

static u32 sched_term_dtype(TinyHVM *ctx, Term t) {
    switch (term_tag(t)) {
        case TAG_TEN:
            return ctx->tensors[(u32)term_val(t)].dtype;
        case TAG_NUM:
            return term_ext(t) == NUM_U32 ? DTYPE_U32 : DTYPE_F32;
        case TAG_TOP: {
            u32 uop = term_ext(t);
            u64 loc = term_val(t);
            if (uop == UOP_CAST) {
                Term b = heap_read(ctx, loc + 1);
                if (term_tag(b) == TAG_TEN) {
                    u32 bid = (u32)term_val(b);
                    if (bid < ctx->tensor_count) {
                        u32 raw[MAX_DIM];
                        if (tensor_meta_read_u32(ctx, bid, raw, MAX_DIM) == 1 && raw[0] < DTYPE_COUNT)
                            return raw[0];
                    }
                }
            }
            if (is_view_op(uop) || sched_is_kernelizable_uop(uop) || uop == UOP_CAST) {
                Term a = heap_read(ctx, loc + 0);
                u32 dt = sched_term_dtype(ctx, a);
                if (dt < DTYPE_COUNT) return dt;
            }
            return DTYPE_F32;
        }
        default:
            return DTYPE_F32;
    }
}

typedef enum {
    SCHED_PARENT_NONE = 0,
    SCHED_PARENT_COMPUTE = 1,
    SCHED_PARENT_EXTERNAL = 2,
} SchedParentClass;

typedef struct {
    Term compute_term;
    Term root_term;
    Term external_root;
    Term forced_root;
    u16  consumer_count;
    u16  external_count;
    u32  kid;
    u32  raw_output_tid;
    u32  output_tid;
    u8   is_boundary;
    u8   force_boundary;
} SchedBoundary;

static SchedBoundary sched_boundaries[SCHED_MAX_KERNELS];
static u32 sched_boundary_count = 0;
static u64 sched_boundary_locs[SCHED_MAX_KERNELS];
static u32 sched_boundary_output_tids[SCHED_MAX_KERNELS];
static u32 sched_boundary_kids[SCHED_MAX_KERNELS];
static u64 sched_kernel_locs[SCHED_MAX_KERNELS];

static SchedBoundary *sched_boundary_find(Term compute_term, int create) {
    if (!sched_is_kernelizable_term(compute_term)) return NULL;
    u64 loc = term_val(compute_term);
    for (u32 i = 0; i < sched_boundary_count; i++) {
        if (term_val(sched_boundaries[i].compute_term) == loc) return &sched_boundaries[i];
    }
    if (!create || sched_boundary_count >= SCHED_MAX_KERNELS) return NULL;
    SchedBoundary *b = &sched_boundaries[sched_boundary_count++];
    memset(b, 0, sizeof(*b));
    b->compute_term = compute_term;
    b->root_term = compute_term;
    b->external_root = term_era();
    b->forced_root = term_era();
    return b;
}

static int sched_is_reduce_term(Term t) {
    return term_tag(t) == TAG_TOP &&
           (term_ext(t) == UOP_SUM || term_ext(t) == UOP_RMAX);
}

static void sched_boundary_force_root(SchedBoundary *b, Term root_hint) {
    if (term_tag(root_hint) == TAG_ERA) return;
    if (term_tag(b->forced_root) == TAG_ERA) {
        b->forced_root = root_hint;
    } else if (b->forced_root != root_hint) {
        b->forced_root = b->compute_term;
    }
}

static void sched_boundary_note_consumer(TinyHVM *ctx, Term child, SchedParentClass parent_class) {
    if (parent_class == SCHED_PARENT_NONE) return;
    Term inner = sched_unwrap_views(ctx, child);
    if (!sched_is_kernelizable_term(inner)) return;
    SchedBoundary *b = sched_boundary_find(inner, 1);
    if (!b) return;
    if (parent_class == SCHED_PARENT_EXTERNAL) {
        b->external_count++;
        if (term_tag(b->external_root) == TAG_ERA) b->external_root = child;
        else if (b->external_root != child) b->external_root = inner;
    } else if (parent_class == SCHED_PARENT_COMPUTE) {
        b->consumer_count++;
        if (child != inner && term_tag(child) == TAG_TOP && is_view_op(term_ext(child))) {
            b->force_boundary = 1;
            sched_boundary_force_root(b, child);
        }
        if (sched_is_reduce_term(inner)) {
            b->force_boundary = 1;
            sched_boundary_force_root(b, child);
        }
    }
}

static SchedParentClass sched_parent_class(Term parent) {
    u8 tag = term_tag(parent);
    if (tag == TAG_DP0 || tag == TAG_DP1 || tag == TAG_ERA) return SCHED_PARENT_NONE;
    if (tag == TAG_TOP) {
        u32 uop = term_ext(parent);
        if (is_view_op(uop)) return SCHED_PARENT_NONE;
        if (sched_is_kernelizable_uop(uop)) return SCHED_PARENT_COMPUTE;
    }
    return SCHED_PARENT_EXTERNAL;
}

static void sched_collect_boundaries(TinyHVM *ctx, Term root) {
    sched_boundary_count = 0;
    memset(sched_boundaries, 0, sizeof(sched_boundaries));

    sched_boundary_note_consumer(ctx, root, SCHED_PARENT_EXTERNAL);

    u8 *seen_slot = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    u8 *seen_dup  = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    u64 work_cap = ctx->heap_pos ? (ctx->heap_pos * 8) : 0;
    typedef struct {
        Term term;
        SchedParentClass inherited_class;
    } SchedWork;
    SchedWork *work = work_cap ? (SchedWork *)malloc(sizeof(SchedWork) * (size_t)work_cap) : NULL;
    u64 wp = 0;
    #define SCHED_PUSH(_tt, _pc) do { \
        if (work && wp < work_cap) { \
            work[wp].term = (_tt); \
            work[wp].inherited_class = (_pc); \
            wp++; \
        } \
    } while (0)
    SCHED_PUSH(root, SCHED_PARENT_EXTERNAL);

    while (work && wp > 0) {
        SchedWork item = work[--wp];
        Term tt = item.term;
        u8 tg = term_tag(tt);
        u64 tv = term_val(tt);

        if (tg == TAG_DP0 || tg == TAG_DP1) {
            if (tv == 0 || tv >= ctx->heap_pos || (seen_dup && seen_dup[tv])) continue;
            if (seen_dup) seen_dup[tv] = 1;
            SCHED_PUSH(heap_read(ctx, tv), item.inherited_class);
            continue;
        }
        if (tg == TAG_VAR) {
            if (tv == 0 || tv >= ctx->heap_pos) continue;
            Term sub = heap_read(ctx, tv);
            if (!term_is_sub(sub)) SCHED_PUSH(sub, item.inherited_class);
            continue;
        }
        if (tg == TAG_ERA) {
            if (tv == 0 || tv >= ctx->heap_pos) continue;
            if (!seen_slot || !seen_slot[tv]) {
                if (seen_slot) seen_slot[tv] = 1;
                SCHED_PUSH(heap_read(ctx, tv), SCHED_PARENT_NONE);
            }
            continue;
        }

        u32 ar = phase1_term_arity(tt);
        SchedParentClass pclass = sched_parent_class(tt);
        if (pclass == SCHED_PARENT_NONE && tg == TAG_TOP && is_view_op(term_ext(tt)))
            pclass = item.inherited_class;
        for (u32 i = 0; i < ar; i++) {
            u64 p = tv + i;
            if (p >= ctx->heap_pos) continue;
            Term child = heap_read(ctx, p);
            sched_boundary_note_consumer(ctx, child, pclass);
            if (!seen_slot || !seen_slot[p]) {
                if (seen_slot) seen_slot[p] = 1;
                SCHED_PUSH(child, pclass);
            }
        }
    }

    free(work);
    free(seen_dup);
    free(seen_slot);
    #undef SCHED_PUSH
}

static u32 sched_select_boundaries(void) {
    u32 selected = 0;
    for (u32 i = 0; i < sched_boundary_count; i++) {
        SchedBoundary *b = &sched_boundaries[i];
        b->is_boundary = b->force_boundary || (b->external_count > 0) || (b->consumer_count > 1);
        if (!b->is_boundary) continue;
        if (b->force_boundary && term_tag(b->forced_root) != TAG_ERA) {
            b->root_term = b->forced_root;
        } else if (b->external_count == 1 && b->consumer_count == 0 &&
            term_tag(b->external_root) == TAG_TOP) {
            b->root_term = b->external_root;
        } else {
            b->root_term = b->compute_term;
        }
        b->kid = selected;
        sched_boundary_locs[selected] = term_val(b->compute_term);
        sched_boundary_kids[selected] = selected;
        selected++;
    }
    return selected;
}

static void sched_prepare_boundary_output(TinyHVM *ctx, SchedBoundary *b) {
    const View *raw_v = sched_term_view(ctx, b->compute_term);
    if (!raw_v) raw_v = sched_term_view(ctx, b->root_term);
    if (!raw_v) return;

    u32 raw_tid = tensor_create_unbacked(ctx, raw_v->shape, sched_term_dtype(ctx, b->compute_term));
    TensorMeta *raw_m = &ctx->tensors[raw_tid];
    raw_m->creator_op = UOP_KERNEL;
    raw_m->fusing_loc = term_val(b->compute_term);

    b->raw_output_tid = raw_tid;
    b->output_tid = raw_tid;

    const View *root_v = sched_term_view(ctx, b->root_term);
    const ShapeTracker *root_st = sched_term_tracker(ctx, b->root_term);
    if (root_v && (!shape_eq(root_v->shape, raw_v->shape) || (root_st && root_st->n_views > 1))) {
        u32 out_tid = tensor_view_of(ctx, raw_tid, *root_v);
        TensorMeta *out_m = &ctx->tensors[out_tid];
        out_m->creator_op = UOP_KERNEL;
        out_m->src_ids[0] = raw_tid;
        out_m->fusing_loc = term_val(b->compute_term);
        if (root_st && root_st->n_views > 0) out_m->st = *root_st;
        b->output_tid = out_tid;
    }
    sched_boundary_output_tids[b->kid] = b->output_tid;
}

static void sched_exec_order_visit(u32 kid, u8 *seen) {
    if (kid >= sched_kernel_count || seen[kid]) return;
    seen[kid] = 1;
    KernelEntry *ke = &sched_kernels[kid];
    for (u32 di = 0; di < ke->n_deps; di++)
        sched_exec_order_visit(ke->dep_kids[di], seen);
    if (sched_exec_count < SCHED_MAX_KERNELS)
        sched_exec_order[sched_exec_count++] = kid;
}

static void sched_plan_output_liveness(TinyHVM *ctx) {
    memset(sched_output_remaining_uses, 0, sizeof(sched_output_remaining_uses));
    memset(sched_output_pinned, 0, sizeof(sched_output_pinned));
    memset(sched_exec_order, 0, sizeof(sched_exec_order));
    memset(sched_slots, 0, sizeof(sched_slots));
    sched_exec_count = 0;
    sched_slot_count = 0;

    for (u32 i = 0; i < sched_boundary_count; i++) {
        SchedBoundary *b = &sched_boundaries[i];
        if (!b->is_boundary) continue;
        if (b->kid < SCHED_MAX_KERNELS && b->external_count > 0)
            sched_output_pinned[b->kid] = 1;
    }

    u8 seen[SCHED_MAX_KERNELS];
    memset(seen, 0, sizeof(seen));
    for (u32 i = 0; i < sched_boundary_count; i++) {
        SchedBoundary *b = &sched_boundaries[i];
        if (!b->is_boundary || b->kid >= sched_kernel_count) continue;
        if (b->external_count > 0)
            sched_exec_order_visit(b->kid, seen);
    }
    for (u32 kid = 0; kid < sched_kernel_count; kid++) {
        if (!seen[kid])
            sched_exec_order_visit(kid, seen);
    }

    u32 last_use[SCHED_MAX_KERNELS];
    for (u32 pos = 0; pos < sched_exec_count; pos++) {
        u32 kid = sched_exec_order[pos];
        last_use[kid] = pos;
    }
    for (u32 kid = 0; kid < sched_kernel_count; kid++) {
        KernelEntry *ke = &sched_kernels[kid];
        for (u32 di = 0; di < ke->n_deps; di++) {
            u32 dep_kid = ke->dep_kids[di];
            if (dep_kid < SCHED_MAX_KERNELS) {
                sched_output_remaining_uses[dep_kid]++;
                for (u32 pos = 0; pos < sched_exec_count; pos++) {
                    if (sched_exec_order[pos] == kid && pos > last_use[dep_kid]) {
                        last_use[dep_kid] = pos;
                        break;
                    }
                }
            }
        }
    }

    u32 slot_end[SCHED_MAX_KERNELS + 1];
    u8 slot_pinned[SCHED_MAX_KERNELS + 1];
    memset(slot_end, 0, sizeof(slot_end));
    memset(slot_pinned, 0, sizeof(slot_pinned));

    for (u32 pos = 0; pos < sched_exec_count; pos++) {
        u32 kid = sched_exec_order[pos];
        KernelEntry *ke = &sched_kernels[kid];
        u32 raw_tid = ke->raw_output_tid ? ke->raw_output_tid : ke->output_tid;
        if (raw_tid == 0 || raw_tid >= ctx->tensor_count) continue;

        TensorMeta *raw_m = &ctx->tensors[raw_tid];
        Backend *backend = raw_m->backend ? raw_m->backend : ctx_default_backend(ctx);
        u64 bytes = (u64)raw_m->view.numel * dtype_size(raw_m->dtype);
        u32 best_slot = 0;
        u64 best_bytes = UINT64_MAX;

        if (!sched_output_pinned[kid]) {
            for (u32 sid = 1; sid <= sched_slot_count; sid++) {
                if (slot_pinned[sid]) continue;
                if (slot_end[sid] >= pos) continue;
                if (sched_slots[sid].backend != backend) continue;
                if (sched_slots[sid].bytes < bytes) continue;
                if (sched_slots[sid].bytes < best_bytes) {
                    best_slot = sid;
                    best_bytes = sched_slots[sid].bytes;
                }
            }
        }

        if (best_slot == 0) {
            if (sched_slot_count + 1 > SCHED_MAX_KERNELS) continue;
            best_slot = ++sched_slot_count;
            sched_slots[best_slot].backend = backend;
            sched_slots[best_slot].bytes = bytes;
            sched_slots[best_slot].buf_id = 0;
            sched_slots[best_slot].detached = 0;
            slot_pinned[best_slot] = sched_output_pinned[kid];
        }

        slot_end[best_slot] = sched_output_pinned[kid] ? UINT32_MAX : last_use[kid];
        ke->output_slot = best_slot;
        raw_m->planned_slot = best_slot;
        if (ke->output_tid && ke->output_tid != raw_tid && ke->output_tid < ctx->tensor_count)
            ctx->tensors[ke->output_tid].planned_slot = best_slot;
    }
}

static void sched_replace_term_everywhere(TinyHVM *ctx, Term original, Term rewritten) {
    for (u64 h = 1; h < ctx->heap_pos; h++)
        if (ctx->heap[h] == original) ctx->heap[h] = rewritten;
}

static Term sched_install_kernel(TinyHVM *ctx, SchedBoundary *b, KernelEntry *ke) {
    u64 floc = ctx->heap_pos;
    ctx->heap_pos += 2;
    heap_set(ctx, floc, term_era());
    heap_set(ctx, floc + 1, term_num_u32(b->kid));
    Term ft = term_new(TAG_TOP, UOP_KERNEL, floc);
    sched_kernel_locs[b->kid] = floc;

    const ShapeTracker *out_st = tensor_st_get(&ctx->tensors[b->output_tid]);
    if (out_st && out_st->n_views > 0) st_set_tracker(floc, out_st);
    else {
        const View *out_v = tensor_view_get(&ctx->tensors[b->output_tid]);
        View fallback = out_v ? *out_v : view_create(ke->out_shape);
        st_set(floc, &fallback);
    }

    // Wrap kernel in SEQ chains for ASSIGN dependencies.
    // SEQ(ASSIGN, KERNEL) forces ASSIGN to fire before KERNEL dispatches.
    extern Term fuse_assign_deps[];
    extern u32  fuse_n_assign_deps;
    for (u32 i = 0; i < fuse_n_assign_deps; i++) {
        ft = thvm_seq(ctx, fuse_assign_deps[i], ft);
        // Copy shape metadata to SEQ's heap loc so downstream can find it
        const View *sv = st_get(floc);
        if (sv) st_set(term_val(ft), sv);
    }

    thvm_sched_rewrite_remember(ctx, b->root_term, ft);
    sched_replace_term_everywhere(ctx, b->root_term, ft);
    return ft;
}

static Term thvm_sched_dispatch_kernel(TinyHVM *ctx, u32 kid) {
    if (kid >= sched_kernel_count) return term_era();
    if (term_tag(kid_results[kid]) != TAG_ERA) return kid_results[kid];

    KernelEntry *ke = &sched_kernels[kid];
    for (u32 di = 0; di < ke->n_deps; di++) {
        Term dep = thvm_sched_dispatch_kernel(ctx, ke->dep_kids[di]);
        if (term_tag(dep) == TAG_ERA) return dep;
    }

    u32 raw_tid = ke->raw_output_tid ? ke->raw_output_tid : ke->output_tid;
    if (raw_tid == 0) return term_era();
    ENSURE(ctx, raw_tid);
    TensorMeta *md = &ctx->tensors[raw_tid];

    if (md->buf_id == 0) {
        if (sched_slot_bind_output(ctx, ke->output_slot, raw_tid, ke->output_tid) == 0)
            return term_era();
    } else if (ke->output_tid && ke->output_tid != raw_tid &&
               ke->output_tid < ctx->tensor_count &&
               ctx->tensors[ke->output_tid].buf_id == 0) {
        if (sched_slot_bind_output(ctx, ke->output_slot, raw_tid, ke->output_tid) == 0)
            return term_era();
    }

    u32 bufs[FUSE_MAX_LEAVES];
    const View *views[FUSE_MAX_LEAVES];
    for (u32 i = 0; i < ke->n_leaves; i++) {
        if (ke->leaf_kinds[i] == KERNEL_LEAF_TENSOR) {
            u32 lid = ke->leaf_ids[i];
            if (lid == 0) return term_era();
            ENSURE(ctx, lid);
            if (ctx->tensors[lid].buf_id == 0) return term_era();
            bufs[i] = ctx->tensors[lid].buf_id;
            views[i] = &ke->leaf_views[i];
            continue;
        }
        if (ke->leaf_kinds[i] == KERNEL_LEAF_NUM) {
            f32 val = ke->leaf_nums[i];
            Term scalar = thvm_scalar_typed(ctx, val, ke->leaf_dtypes[i]);
            u32 lid = (u32)term_val(scalar);
            bufs[i] = ctx->tensors[lid].buf_id;
            views[i] = &ke->leaf_views[i];
            continue;
        }
        return term_era();
    }

    const ShapeTracker *st_ptrs[FUSE_MAX_LEAVES];
    int has_multiview = 0;
    for (u32 i = 0; i < ke->n_leaves; i++) {
        st_ptrs[i] = &ke->leaf_sts[i];
        if (ke->leaf_sts[i].n_views >= 2) has_multiview = 1;
    }

    md->backend->dispatch_kernel_rs(
        md->buf_id, bufs, views,
        has_multiview ? st_ptrs : NULL, ke->n_leaves,
        ke->ops, ke->n_ops, &ke->full_shape,
        ke->has_reduce ? &ke->reduce : NULL,
        NULL, NULL, 0, ke->leaf_dtypes, md->dtype);
    ctx->itrs++;
    md->creator_op = UOP_KERNEL;
    md->fusing_loc = sched_kernel_locs[kid];

    if (ke->output_tid && ke->output_tid != raw_tid) {
        TensorMeta *out_m = &ctx->tensors[ke->output_tid];
        out_m->creator_op = UOP_KERNEL;
        out_m->src_ids[0] = raw_tid;
        out_m->fusing_loc = sched_kernel_locs[kid];
    }

    Term result = term_ten(ke->output_tid ? ke->output_tid : raw_tid, ctx->tensors[ke->output_tid ? ke->output_tid : raw_tid].dtype);
    kid_results[kid] = result;

    for (u32 di = 0; di < ke->n_deps; di++) {
        u32 dep_kid = ke->dep_kids[di];
        if (dep_kid >= SCHED_MAX_KERNELS || sched_output_remaining_uses[dep_kid] == 0) continue;
        sched_output_remaining_uses[dep_kid]--;
        if (sched_output_remaining_uses[dep_kid] == 0 && !sched_output_pinned[dep_kid])
            sched_release_output_kid(ctx, dep_kid);
    }
    return result;
}

u32 sched_all(TinyHVM *ctx, Term root) {
    sched_planner_reset_state();
    if (ctx->sched_rewrites)
        memset(ctx->sched_rewrites, 0, SCHED_REWRITE_CAP * sizeof(SchedRewriteEntry));
    memset(sched_kernel_locs, 0, sizeof(sched_kernel_locs));
    memset(sched_boundary_output_tids, 0, sizeof(sched_boundary_output_tids));
    memset(sched_boundary_kids, 0, sizeof(sched_boundary_kids));
    _sched_n_absorbed = 0;
    sched_kernel_count = 0;
    for (u32 i = 0; i < SCHED_MAX_KERNELS; i++) kid_results[i] = term_era();

    sched_collect_boundaries(ctx, root);
    u32 selected = sched_select_boundaries();
    if (selected == 0) return 0;

    for (u32 i = 0; i < sched_boundary_count; i++) {
        SchedBoundary *b = &sched_boundaries[i];
        if (!b->is_boundary) continue;
        sched_prepare_boundary_output(ctx, b);
    }

    fuse_no_lazy_resolve = 1;
    for (u32 i = 0; i < sched_boundary_count; i++) {
        SchedBoundary *b = &sched_boundaries[i];
        if (!b->is_boundary) continue;
        KernelEntry ke;
        memset(&ke, 0, sizeof(ke));
        fuse_set_schedule_boundaries(sched_boundary_locs, sched_boundary_output_tids,
                                     sched_boundary_kids, selected, term_val(b->compute_term));
        if (!fuse_build_kernel(ctx, b->root_term, &ke)) {
            fuse_clear_schedule_boundaries();
            fuse_no_lazy_resolve = 0;
            if (getenv("THVM_SCHED_DIAG")) {
                fprintf(stderr, "sched_build_fail: root=%s@%llu fc=%d\n",
                        term_ext(b->root_term) < UOP_COUNT ? uop_names[term_ext(b->root_term)] : "?",
                        (unsigned long long)term_val(b->root_term), ke.fail_code);
            }
            return 0;
        }
        ke.original_term = b->root_term;
        ke.raw_output_tid = b->raw_output_tid;
        ke.output_tid = b->output_tid;
        sched_kernels[b->kid] = ke;
    }
    fuse_clear_schedule_boundaries();
    fuse_no_lazy_resolve = 0;
    sched_kernel_count = selected;
    sched_plan_output_liveness(ctx);

    for (u32 i = 0; i < sched_boundary_count; i++) {
        SchedBoundary *b = &sched_boundaries[i];
        if (!b->is_boundary) continue;
        sched_install_kernel(ctx, b, &sched_kernels[b->kid]);
    }
    return selected;
}

// === DELETED: 500+ lines of merge passes (4-9), memory analysis, dead kernel redirect ===
// These were post-hoc merge attempts that created broken kernels.
// Replaced by: nothing. The simple 3-pass scheduling is sufficient.
// Kernel count matches tinygrad without merging (the fuser already
// absorbs ew ops into reduce kernels during the walk).



Term thvm_eval(TinyHVM *ctx, Term t) {
    int outermost = (ctx->eval_depth++ == 0);
    int run_step_graph = outermost && getenv("THVM_STEP_GRAPH") && !ctx->step_graph_consumed;
    int run_coarse_graph = outermost && getenv("THVM_GRAPH") && !ctx->coarse_graph_consumed;
    if (run_step_graph) {
        ctx->step_graph_consumed = 1;
        phase1_root_slot = 0;
        t = thvm_phase1_seed_root_grad(ctx, t);
        thvm_step_graph_eval_begin(ctx, t);
        t = thvm_phase1_structural_nf(ctx, t);
        t = thvm_reduce(ctx, t);
        if (phase1_root_slot > 0 && phase1_root_slot < ctx->heap_pos)
            heap_set(ctx, phase1_root_slot, t);
        thvm_step_graph_finalize(ctx);
        phase1_root_slot = 0;
        ctx->eval_depth--;
        return t;
    }

    if (run_coarse_graph) {
        ctx->coarse_graph_consumed = 1;
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", thvm_graph_dir());
        system(cmd);
        t = thvm_phase1_seed_root_grad(ctx, t);
        char path[512];
        thvm_graph_dump_path(path, sizeof(path), "thvm_0_pre_reduce.dot");
        thvm_heap_dot_set_sched_kernels(0);
        thvm_heap_dot_root(ctx, path, t);
        if (phase1_root_slot > 0 && phase1_root_slot < ctx->heap_pos)
            heap_set(ctx, phase1_root_slot, term_era());
    }
    _assign_dispatch_enabled = 0;
    t = run_coarse_graph ? thvm_phase1_structural_nf(ctx, t)
                         : thvm_reduce(ctx, t);
    if (run_coarse_graph) {
        t = thvm_phase1_seed_root_grad(ctx, t);
        char path[512];
        thvm_graph_dump_path(path, sizeof(path), "thvm_1_post_reduce.dot");
        thvm_heap_dot_set_sched_kernels(0);
        thvm_heap_dot_root(ctx, path, t);
        if (phase1_root_slot > 0 && phase1_root_slot < ctx->heap_pos)
            heap_set(ctx, phase1_root_slot, term_era());
    }

    // Phase 2: wrap in UOP_FUSE and reduce — fires fusion as interaction
    {
        u64 fuse_loc = heap_alloc(ctx, 1);
        heap_set(ctx, fuse_loc, t);
        Term fuse_term = term_new(TAG_TOP, UOP_FUSE, fuse_loc);
        t = thvm_reduce(ctx, fuse_term);
    }
    if (run_coarse_graph) {
        t = thvm_phase1_seed_root_grad(ctx, t);
        char path[512];
        thvm_graph_dump_path(path, sizeof(path), "thvm_2_post_sched.dot");
        thvm_heap_dot_set_sched_kernels(1);
        thvm_heap_dot_root(ctx, path, t);
        if (phase1_root_slot > 0 && phase1_root_slot < ctx->heap_pos)
            heap_set(ctx, phase1_root_slot, term_era());
    }

    // Phase 3: wrap in UOP_SCHED and reduce — fires planning + dispatch
    {
        u64 sched_loc = heap_alloc(ctx, 1);
        heap_set(ctx, sched_loc, t);
        Term sched_term = term_new(TAG_TOP, UOP_SCHED, sched_loc);
        t = thvm_reduce(ctx, sched_term);
        _assign_dispatch_enabled = 0;
    }
    if (run_coarse_graph) {
        t = thvm_phase1_seed_root_grad(ctx, t);
        char path[512];
        thvm_graph_dump_path(path, sizeof(path), "thvm_3_post_dispatch.dot");
        thvm_heap_dot_set_sched_kernels(0);
        thvm_heap_dot_root(ctx, path, t);
        if (phase1_root_slot > 0 && phase1_root_slot < ctx->heap_pos)
            heap_set(ctx, phase1_root_slot, term_era());
        phase1_root_slot = 0;
    }
    sched_planner_release_detached_slots();
    ctx->eval_depth--;
    return t;
}

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_eval(ctx, t);
}
