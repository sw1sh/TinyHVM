// Forward declarations (defined in debug/dump.c, included after this file)
static void thvm_heap_dot(TinyHVM *ctx, const char *path);
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root);
static void thvm_heap_dot_set_sched_kernels(int enabled);
static void thvm_step_graph_eval_begin(TinyHVM *ctx, Term root);
static void thvm_step_graph_name_for_interaction(TinyHVM *ctx, u64 source_slot, Term before,
                                                 char *buf, size_t nbuf);
static void thvm_step_graph_after_interaction(TinyHVM *ctx, u64 source_slot, Term before,
                                              const char *current_name, Term root);
static void thvm_step_graph_finalize(TinyHVM *ctx);
static void thvm_step_graph_set_root(Term root);
static void thvm_step_graph_set_before_grad_y(Term y);
static void thvm_step_graph_set_before_era_payload(Term payload);
static void thvm_step_graph_set_before_top_era(int had_era);
static void thvm_step_graph_set_before_top_add_zero(int had_add_zero);
static void thvm_step_graph_set_before_top_partner(Term partner, u64 slot);

// Per-step metadata captured by thvm_reduce_step_collect for the dumper /
// trace consumers. `before` is the redex term as it existed pre-fire;
// `source_slot` is the heap slot it occupied (0 when fired on the root).
typedef struct {
    Term before;
    u64  source_slot;
    u8   kind;
    char name[160];
} StepMeta;

static int thvm_reduce_step_collect(TinyHVM *ctx, Term *traced_io,
                                    StepMeta *meta,
                                    u8 **reach_io, size_t *reach_cap_io);

static void thvm_step_meta_apply_kind_name(u8 kind, char *buf, size_t nbuf) {
    if (!buf || nbuf == 0 || kind != THVM_STEP_KIND_SWEEP || !buf[0]) return;
    if (strncmp(buf, "SWEEP_", 6) == 0) return;
    char tmp[160];
    snprintf(tmp, sizeof(tmp), "SWEEP_%s", buf);
    snprintf(buf, nbuf, "%s", tmp);
}

// schedule/_.c — Unified IC eval: reduce + fuse in two passes.
//
// thvm_reduce(t) — pure IC reduction. Stops at WNF:
//   GRAD fires, combinators fire, compute ops stay TAG_TOP (WNF).
//
// thvm_eval(t) — reduce then fuse:
//   1. thvm_reduce(t)             — pure IC to WNF
//   2. thvm_reduce(UOP_FUSE(t))   — local fusion + dispatch + ASSIGN
//
// UOP_FUSE distributes through SEQ/CTR/ASSIGN and rewrites fuseable structure
// into explicit heap-visible UOP_KERNEL nodes. KERNEL remains lazy until a
// consuming reduction path reaches it.

int fuse_no_lazy_resolve = 0;
int _assign_dispatch_enabled = 0;

// Global kernel table: scheduler writes, UOP_KERNEL handler reads.
KernelEntry sched_kernels[SCHED_MAX_KERNELS];
u32 sched_kernel_count = 0;

// kid_results: TAG_TEN result for each dispatched kid (ERA = not yet dispatched).
// Shared with UOP_KERNEL handler in tensor_ops.c.
Term kid_results[SCHED_MAX_KERNELS];

// Buffer epoch invalidation: ASSIGN bumps epoch, kernel cache checks epoch match.
u32 buf_epoch[MAX_BUF_EPOCHS];  // per-buffer generation counter
// Per-kernel: input buffer IDs and their epochs at dispatch time.
u32 kid_input_bufs[SCHED_MAX_KERNELS][KERNEL_MAX_INPUTS];
u32 kid_input_epochs[SCHED_MAX_KERNELS][KERNEL_MAX_INPUTS];
u32 kid_n_inputs[SCHED_MAX_KERNELS];

static u64 step_root_slot = 0;

typedef struct {
    u32 tid;
    u64 bytes;
    void *data;
} StepGraphTensorSnap;

static StepGraphTensorSnap *thvm_step_graph_snapshot_tensors(TinyHVM *ctx, u32 *out_count) {
    if (out_count) *out_count = 0;
    if (!ctx || !ctx->tensors || ctx->tensor_count == 0) return NULL;
    StepGraphTensorSnap *snaps = (StepGraphTensorSnap *)calloc(ctx->tensor_count, sizeof(StepGraphTensorSnap));
    if (!snaps) return NULL;
    u32 n = 0;
    for (u32 tid = 0; tid < ctx->tensor_count; tid++) {
        TensorMeta *tm = &ctx->tensors[tid];
        if (!tm->backend || !tm->backend->buf_read || !tm->backend->buf_write) continue;
        if (tm->buf_id == 0 || tm->refcount == 0) continue;
        const View *v = tensor_view_get(tm);
        u64 bytes = (u64)(v ? v->numel : 0u) * (u64)dtype_size(tm->dtype);
        if (bytes == 0) continue;
        void *buf = malloc((size_t)bytes);
        if (!buf) continue;
        tm->backend->buf_read(tm->buf_id, buf, bytes);
        snaps[n++] = (StepGraphTensorSnap){ .tid = tid, .bytes = bytes, .data = buf };
    }
    if (out_count) *out_count = n;
    return snaps;
}

static void thvm_step_graph_restore_tensors(TinyHVM *ctx, StepGraphTensorSnap *snaps, u32 count) {
    if (!ctx || !snaps) return;
    for (u32 i = 0; i < count; i++) {
        StepGraphTensorSnap *s = &snaps[i];
        if (!s->data || s->tid >= ctx->tensor_count) continue;
        TensorMeta *tm = &ctx->tensors[s->tid];
        if (tm->backend && tm->backend->buf_write && tm->buf_id != 0 && tm->refcount != 0)
            tm->backend->buf_write(tm->buf_id, s->data, s->bytes);
        free(s->data);
    }
    free(snaps);
}

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
    return (dir && dir[0]) ? dir : "graphs";
}

static void thvm_graph_dump_path(char *buf, size_t nbuf, const char *name) {
    snprintf(buf, nbuf, "%s/%s", thvm_graph_dir(), name);
}

static void thvm_graph_term_summary(Term t, char *buf, size_t nbuf) {
    if (!buf || nbuf == 0) return;
    switch (term_tag(t)) {
        case TAG_TOP: {
            u32 uop = term_ext(t);
            snprintf(buf, nbuf, "TOP/%s@%llu",
                     uop < UOP_COUNT ? uop_names[uop] : "?",
                     (unsigned long long)term_val(t));
            return;
        }
        case TAG_TEN:
            snprintf(buf, nbuf, "TEN/t%u", (u32)term_val(t));
            return;
        case TAG_NUM:
            if (term_ext(t) == NUM_U32)
                snprintf(buf, nbuf, "NUM/u32(%u)", term_as_u32(t));
            else
                snprintf(buf, nbuf, "NUM/f32(%.6g)", term_as_f32(t));
            return;
        case TAG_CTR:
            snprintf(buf, nbuf, "CTR/%u@%llu",
                     term_ext(t), (unsigned long long)term_val(t));
            return;
        case TAG_ERA:
            snprintf(buf, nbuf, "ERA@%llu", (unsigned long long)term_val(t));
            return;
        default:
            snprintf(buf, nbuf, "tag=%u ext=%u val=%llu",
                     term_tag(t), term_ext(t), (unsigned long long)term_val(t));
            return;
    }
}

static u32 step_top_arity(u32 ext) {
    return thvm_uop_storage_arity(ext);
}

static u32 step_term_arity(Term t);
static int step_has_parent_ref(TinyHVM *ctx, Term target);

static u64 step_find_child_slot_in_term(TinyHVM *ctx, Term parent, Term target, u32 depth) {
    if (depth > 64) return 0;
    u32 ar = step_term_arity(parent);
    u64 loc = term_val(parent);
    if (ar == 0 || loc == 0 || loc + ar > ctx->heap_pos) return 0;
    for (u32 i = 0; i < ar; i++) {
        if (heap_read(ctx, loc + i) == target) return loc + i;
    }
    for (u32 i = 0; i < ar; i++) {
        Term child = heap_read(ctx, loc + i);
        switch (term_tag(child)) {
            case TAG_VAR:
            case TAG_DP0:
            case TAG_DP1:
            case TAG_UDP:
            case TAG_ERA:
            case TAG_REF:
            case TAG_ALO:
            case TAG_TEN:
            case TAG_NUM:
            case TAG_ANY:
                continue;
            default:
                break;
        }
        u64 hit = step_find_child_slot_in_term(ctx, child, target, depth + 1);
        if (hit != 0) return hit;
    }
    return 0;
}

static u64 thvm_step_redex_source_slot(TinyHVM *ctx, u64 container_slot, Term container, Term before) {
    if (container == before) return container_slot;
    u64 hit = step_find_child_slot_in_term(ctx, container, before, 0);
    return hit ? hit : container_slot;
}

static int step_term_is_active_era_like(TinyHVM *ctx, Term t, Term *era_out) {
    if (term_tag(t) == TAG_ERA && term_val(t) != 0) {
        if (era_out) *era_out = t;
        return 1;
    }
    if (term_tag(t) == TAG_VAR) {
        u64 loc = term_val(t);
        if (loc < ctx->heap_pos) {
            Term sub = heap_read(ctx, loc);
            if (term_tag(sub) == TAG_ERA && term_val(sub) != 0) {
                if (era_out) *era_out = sub;
                return 1;
            }
        }
    }
    return 0;
}

static int step_top_has_era_arg(TinyHVM *ctx, Term t, u64 *slot_out, Term *term_out) {
    if (term_tag(t) != TAG_TOP) return 0;
    u64 loc = term_val(t);
    u32 arity = step_top_arity(term_ext(t));
    for (u32 i = 0; i < arity; i++) {
        Term child = heap_read(ctx, loc + i);
        Term era_child = 0;
        if (step_term_is_active_era_like(ctx, child, &era_child)) {
            if (slot_out) *slot_out = loc + i;
            if (term_out) *term_out = era_child;
            return 1;
        }
    }
    return 0;
}

static int step_top_has_add_zero_arg(TinyHVM *ctx, Term t, u64 *slot_out, Term *term_out) {
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

static int step_top_direct_uop(u32 uop) {
    return uop == UOP_ASSIGN || uop == UOP_GRAD || uop == UOP_IFZ ||
           uop == UOP_LOG_PRINT || uop == UOP_TODEVICE || uop == UOP_DETACH ||
           uop == UOP_WHERE || uop == UOP_FUSE ||
           uop == UOP_KERNEL;
}

static int step_top_direct_uop_ctx(TinyHVM *ctx, u32 uop) {
    (void)ctx;
    return step_top_direct_uop(uop);
}

static int step_fuse_payload_is_terminal_passthru(Term t) {
    switch (term_tag(t)) {
        case TAG_TEN:
        case TAG_ERA:
        case TAG_NUM:
            return 1;
        default:
            return 0;
    }
}

static int step_top_is_hidden_trace_passthru(TinyHVM *ctx, Term t) {
    if (term_tag(t) != TAG_TOP) return 0;
    u32 uop = term_ext(t);
    if (uop != UOP_FUSE) return 0;
    u64 loc = term_val(t);
    if (loc == 0 || loc >= ctx->heap_pos) return 0;
    return step_fuse_payload_is_terminal_passthru(heap_read(ctx, loc));
}

static int step_term_maybe_active(TinyHVM *ctx, Term t) {
    u8 tag = term_tag(t);
    if (tag == TAG_TOP) {
        if (step_top_is_hidden_trace_passthru(ctx, t))
            return 0;
        return step_top_direct_uop_ctx(ctx, term_ext(t)) ||
               step_top_has_era_arg(ctx, t, NULL, NULL) ||
               step_top_has_add_zero_arg(ctx, t, NULL, NULL);
    }
    if (tag == TAG_ERA) return term_val(t) != 0 && !step_has_parent_ref(ctx, t);
    if (tag == TAG_TEN || tag == TAG_NUM || tag == TAG_LAM || tag == TAG_SUP ||
        tag == TAG_BRI || tag == TAG_MAT || tag == TAG_ANY || tag == TAG_USP)
        return 0;
    return 1;
}

static int step_term_needs_global_cleanup(TinyHVM *ctx, Term t) {
    u8 tag = term_tag(t);
    if (tag == TAG_ERA) return term_val(t) != 0 && !step_has_parent_ref(ctx, t);
    if (tag != TAG_TOP) return 0;
    return step_top_has_era_arg(ctx, t, NULL, NULL) ||
           step_top_has_add_zero_arg(ctx, t, NULL, NULL);
}

static int step_term_first_reachable_occurrence(TinyHVM *ctx, u64 h, Term t, const u8 *reach) {
    for (u64 i = 1; i < h; i++) {
        if (reach && !reach[i]) continue;
        if (ctx->heap[i] == t) return 0;
    }
    return 1;
}

static u32 step_term_arity(Term t) {
    u8 tag = term_tag(t);
    u32 ext = term_ext(t);
    switch (tag) {
        case TAG_TOP:
            return step_top_arity(ext);
        case TAG_APP:
        case TAG_LAM:
        case TAG_BRI:
        case TAG_SEQ:
        case TAG_SUP:
        case TAG_USP:
        case TAG_OP2:
        case TAG_EQL:
        case TAG_AND:
        case TAG_OR:
        case TAG_MAT:
        case TAG_ANN:
            return 2;
        case TAG_ALO:
            return 0;
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

static int step_has_parent_ref(TinyHVM *ctx, Term target) {
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term p = ctx->heap[h];
        u32 ar = step_term_arity(p);
        u64 loc = term_val(p);
        if (ar == 0 || loc == 0 || loc + ar > ctx->heap_pos) continue;
        for (u32 i = 0; i < ar; i++) {
            if (heap_read(ctx, loc + i) == target) return 1;
        }
    }
    return 0;
}

static int step_slot_is_local_assign_target(TinyHVM *ctx, u64 slot) {
    (void)ctx; (void)slot;
    return 0;
}

static void step_mark_reachable_slots(TinyHVM *ctx, Term root, u8 *reach) {
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
    if (step_root_slot > 0 && step_root_slot < ctx->heap_pos)
        P1_PUSH(heap_read(ctx, step_root_slot));

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

        // Match normal reduction reachability: lambda / matcher bodies stay
        // latent until ordinary APP-driven evaluation enters them.
        if (tg == TAG_LAM || tg == TAG_BRI || tg == TAG_MAT || tg == TAG_ANN)
            continue;

        u32 ar = step_term_arity(tt);
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

static int step_term_is_whnf_atom(Term t) {
    u8 tag = term_tag(t);
    return tag == TAG_TEN || tag == TAG_NUM || tag == TAG_LAM || tag == TAG_SUP ||
           tag == TAG_BRI || tag == TAG_MAT || tag == TAG_ANY || tag == TAG_USP;
}

static int step_grad_y_ready(Term t) {
    u8 tag = term_tag(t);
    return tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM ||
           tag == TAG_SUP || tag == TAG_TOP;
}

static int step_top_arg0_ready(Term t) {
    u8 tag = term_tag(t);
    return tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM || tag == TAG_SUP;
}

static int step_fuse_payload_top_ready(TinyHVM *ctx, Term t) {
    if (!ctx || term_tag(t) != TAG_TOP) return 0;
    u32 uop = term_ext(t);
    if (uop == UOP_ASSIGN || uop == UOP_KERNEL || uop == UOP_FUSE)
        return 1;
    if (!(is_binary(uop) || is_elementwise(uop) ||
          uop == UOP_SUM || uop == UOP_RMAX || is_view_op(uop)))
        return 0;
    u64 loc = term_val(t);
    if (loc == 0 || loc >= ctx->heap_pos) return 0;
    if (!thvm_kernel_local_child_ready(ctx, heap_read(ctx, loc + 0)))
        return 0;
    if (!is_binary(uop) && is_elementwise(uop))
        return 1;
    if (loc + 1 >= ctx->heap_pos) return 0;
    return thvm_kernel_local_child_ready(ctx, heap_read(ctx, loc + 1));
}

static int step_fuse_payload_ready(TinyHVM *ctx, Term t) {
    if (step_fuse_payload_is_terminal_passthru(t))
        return 0;
    switch (term_tag(t)) {
        case TAG_SEQ:
        case TAG_CTR:
            return 1;
        case TAG_TOP:
            return step_fuse_payload_top_ready(ctx, t);
        default:
            return 0;
    }
}

static int step_trace_root_is_terminal(TinyHVM *ctx, Term t) {
    return step_term_is_whnf_atom(t) || step_top_is_hidden_trace_passthru(ctx, t);
}

static int step_top_frame_arg0_ready(Term t, u32 uop) {
    u8 tag = term_tag(t);
    return tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM || tag == TAG_SUP ||
           (tag == TAG_TOP && uop == UOP_GRAD) ||
           (tag == TAG_VAR && uop == UOP_DETACH) ||
           (uop == UOP_FUSE) ||
           (uop == UOP_KERNEL &&
            tag != TAG_DP0 && tag != TAG_DP1);
}

static int step_top_frame_arg1_ready(Term t, u32 uop) {
    u8 tag = term_tag(t);
    int ok = tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM ||
             tag == TAG_LAM || tag == TAG_SUP;
    if (uop == UOP_KERNEL) {
        ok = ok || tag == TAG_SEQ || tag == TAG_CTR || tag == TAG_ANY ||
             (tag == TAG_TOP && term_ext(t) != UOP_FUSE);
    }
    return ok;
}

static int step_top_frame_arg2_ready(Term t, u32 uop) {
    u8 tag = term_tag(t);
    int ok = (uop == UOP_KERNEL)
           ? (tag == TAG_NUM)
           : (tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM ||
              tag == TAG_CTR || tag == TAG_ANY || tag == TAG_LAM || tag == TAG_SUP);
    return ok;
}

static int step_log_print_arg_ready(Term t) {
    return step_top_arg0_ready(t);
}

static int step_app_mat_arg_ready(Term t) {
    u8 tag = term_tag(t);
    return tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM ||
           tag == TAG_SUP || tag == TAG_CTR || tag == TAG_ANY;
}

static int step_dp_can_fire_on(Term t) {
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

static int thvm_step_predict_next_redex(TinyHVM *ctx, Term t, Term *out_before, Term *out_whnf) {
    u8 tag = term_tag(t);
    if (step_term_is_whnf_atom(t)) {
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

    if (tag == TAG_ALO) {
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
        if (thvm_step_predict_next_redex(ctx, head, out_before, &whead))
            return 1;
        if (ar == 1 || term_tag(whead) == TAG_ERA) {
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
        if (thvm_step_predict_next_redex(ctx, fun, out_before, &wfun))
            return 1;
        if (term_tag(wfun) == TAG_MAT) {
            Term arg = heap_read(ctx, loc + 1);
            Term warg = arg;
            if (thvm_step_predict_next_redex(ctx, arg, out_before, &warg))
                return 1;
            if (step_app_mat_arg_ready(warg)) {
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
        // Phase-1 traces stop at the visible KERNEL DAG; settled monolithic
        // kernels only become reducible again once phase-2 enables dispatch.
        if (uop == UOP_KERNEL && thvm_kernel_is_monolithic(ctx, t) &&
            !ctx->dispatch_enabled) {
            if (out_whnf) *out_whnf = t;
            return 0;
        }
        if (uop == UOP_GRAD) {
            Term y = heap_read(ctx, loc + 0);
            Term wy = y;
            if (thvm_step_predict_next_redex(ctx, y, out_before, &wy))
                return 1;
            if (step_grad_y_ready(wy)) {
                if (out_before) *out_before = t;
                return 1;
            }
            if (out_whnf) *out_whnf = t;
            return 0;
        }

        if (uop == UOP_LOG_PRINT) {
            Term a0 = heap_read(ctx, loc + 0);
            Term wa0 = a0;
            if (thvm_step_predict_next_redex(ctx, a0, out_before, &wa0))
                return 1;
            if (step_log_print_arg_ready(wa0)) {
                if (out_before) *out_before = t;
                return 1;
            }
            if (out_whnf) *out_whnf = t;
            return 0;
        }

        // IFZ: fires when counter (arg0) is ready
        if (uop == UOP_IFZ) {
            Term a0 = heap_read(ctx, loc + 0);
            Term wa0 = a0;
            if (thvm_step_predict_next_redex(ctx, a0, out_before, &wa0))
                return 1;
            if (step_top_arg0_ready(wa0)) {
                if (out_before) *out_before = t;
                return 1;
            }
            if (out_whnf) *out_whnf = t;
            return 0;
        }

        if (step_top_direct_uop_ctx(ctx, uop)) {
            Term a0 = heap_read(ctx, loc + 0);
            Term wa0 = a0;
            if (step_top_arity(uop) > 0 &&
                thvm_step_predict_next_redex(ctx, a0, out_before, &wa0))
                return 1;
            if (step_top_has_era_arg(ctx, t, NULL, NULL) ||
                step_top_has_add_zero_arg(ctx, t, NULL, NULL)) {
                if (out_before) *out_before = t;
                return 1;
            }
            int arg0_ready = step_top_frame_arg0_ready(wa0, uop);
            if (uop == UOP_FUSE)
                arg0_ready = step_fuse_payload_ready(ctx, wa0);
            if (uop == UOP_KERNEL)
                arg0_ready = thvm_kernel_local_child_ready(ctx, wa0);
            if (!arg0_ready) {
                if (out_whnf) *out_whnf = t;
                return 0;
            }
            if (step_top_arity(uop) == 1) {
                if (out_before) *out_before = t;
                return 1;
            }

            Term a1 = heap_read(ctx, loc + 1);
            Term wa1 = a1;
            if (thvm_step_predict_next_redex(ctx, a1, out_before, &wa1))
                return 1;
            int arg1_ready = step_top_frame_arg1_ready(wa1, uop);
            if (uop == UOP_KERNEL)
                arg1_ready = thvm_kernel_local_child_ready(ctx, wa1);
            if (!arg1_ready) {
                if (out_whnf) *out_whnf = t;
                return 0;
            }
            if (step_top_arity(uop) == 2) {
                if (out_before) *out_before = t;
                return 1;
            }

            Term a2 = heap_read(ctx, loc + 2);
            Term wa2 = a2;
            if (thvm_step_predict_next_redex(ctx, a2, out_before, &wa2))
                return 1;
            if (!step_top_frame_arg2_ready(wa2, uop)) {
                if (out_whnf) *out_whnf = t;
                return 0;
            }
            if (out_before) *out_before = t;
            return 1;
        }

        // Ordinary compute/view TOP nodes are WNF in the reducer unless an
        // immediate ERA / ADD(0, x) rewrite applies on the current heap state.
        if (step_top_has_era_arg(ctx, t, NULL, NULL) ||
            step_top_has_add_zero_arg(ctx, t, NULL, NULL)) {
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
        if (thvm_step_predict_next_redex(ctx, val, out_before, &wv))
            return 1;
        if (step_dp_can_fire_on(wv)) {
            if (out_before) *out_before = t;
            return 1;
        }
        if (out_whnf) *out_whnf = t;
        return 0;
    }

    // TAG_VAR: substituted VARs are reducible (resolve to the substituted value)
    if (tag == TAG_VAR) {
        u64 loc = term_val(t);
        if (loc < ctx->heap_pos) {
            Term sub = heap_read(ctx, loc);
            if (term_tag(sub) == TAG_ERA && term_val(sub) != 0) {
                if (out_whnf) *out_whnf = sub;
                return 0;
            }
            if (!term_is_sub(sub)) {
                // Substituted — follow through to resolved value
                if (out_before) *out_before = t;
                return 1;
            }
        }
        if (out_whnf) *out_whnf = t;
        return 0;
    }

    if (out_whnf) *out_whnf = t;
    return 0;
}

static int thvm_step_find_next_actual(TinyHVM *ctx, Term root,
                                      u64 *out_source_slot, Term *out_before,
                                      u8 *out_kind) {
    Term before = 0;
    Term whnf = root;
    if (thvm_step_predict_next_redex(ctx, root, &before, &whnf)) {
        if (out_source_slot) *out_source_slot = step_root_slot;
        if (out_before) *out_before = before;
        if (out_kind) *out_kind = THVM_STEP_KIND_ROOT;
        return 1;
    }
    // The collector still gives the mirrored root one real step even when the
    // predictor classifies FUSE(TEN/ERA/NUM) as a hidden terminal passthrough.
    // Keep next-step discovery aligned with that behavior so the handoff from
    // mainline root work to detached sweep cleanup doesn't skip the final root
    // interaction in filenames/highlights.
    if (step_top_is_hidden_trace_passthru(ctx, root)) {
        if (out_source_slot) *out_source_slot = step_root_slot;
        if (out_before) *out_before = root;
        if (out_kind) *out_kind = THVM_STEP_KIND_ROOT;
        return 1;
    }

    u8 *reach = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    step_mark_reachable_slots(ctx, root, reach);
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        if (h == step_root_slot) continue;
        if (step_slot_is_local_assign_target(ctx, h)) continue;
        Term ht = ctx->heap[h];
        int reachable = !(reach && !reach[h]);
        if (!reachable && !step_term_needs_global_cleanup(ctx, ht)) continue;
        if (!step_term_maybe_active(ctx, ht)) continue;
        if ((term_tag(ht) == TAG_DP0 || term_tag(ht) == TAG_DP1) &&
            reachable &&
            !step_term_first_reachable_occurrence(ctx, h, ht, reach)) {
            continue;
        }
        if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_GRAD &&
            reachable &&
            !step_term_first_reachable_occurrence(ctx, h, ht, reach)) {
            continue;
        }
        Term w = ht;
        if (thvm_step_predict_next_redex(ctx, ht, &before, &w)) {
            if (out_source_slot) *out_source_slot = h;
            if (out_before) *out_before = before;
            if (out_kind) *out_kind = reachable ? THVM_STEP_KIND_HEAP
                                                : THVM_STEP_KIND_SWEEP;
            free(reach);
            return 1;
        }
    }
    free(reach);
    return 0;
}

static void thvm_step_capture_step_before_meta(TinyHVM *ctx, Term before) {
    if (term_tag(before) == TAG_TOP && term_ext(before) == UOP_GRAD) {
        u64 gl = term_val(before);
        if (gl + 1 < ctx->heap_pos) thvm_step_graph_set_before_grad_y(heap_read(ctx, gl + 0));
        else thvm_step_graph_set_before_grad_y(term_era());
    } else {
        thvm_step_graph_set_before_grad_y(term_era());
    }
    if (term_tag(before) == TAG_TOP && term_ext(before) != UOP_GRAD)
        thvm_step_graph_set_before_top_era(step_top_has_era_arg(ctx, before, NULL, NULL));
    else
        thvm_step_graph_set_before_top_era(0);
    if (term_tag(before) == TAG_TOP && term_ext(before) != UOP_GRAD)
        thvm_step_graph_set_before_top_add_zero(step_top_has_add_zero_arg(ctx, before, NULL, NULL));
    else
        thvm_step_graph_set_before_top_add_zero(0);
    if (term_tag(before) == TAG_TOP && term_ext(before) != UOP_GRAD) {
        u64 loc = term_val(before);
        Term partner = term_era();
        u64 partner_slot = 0;
        if (loc > 0 && loc < ctx->heap_pos) {
            u32 uop = term_ext(before);
            if (uop == UOP_FUSE) {
                partner_slot = loc;
                partner = heap_read(ctx, loc);
            } else if (uop == UOP_KERNEL && loc + 1 < ctx->heap_pos) {
                Term a0 = heap_read(ctx, loc + 0);
                Term a1 = heap_read(ctx, loc + 1);
                if (term_tag(a0) == TAG_TOP && term_ext(a0) == UOP_KERNEL) {
                    partner_slot = loc + 0;
                    partner = a0;
                } else if (term_tag(a1) == TAG_TOP && term_ext(a1) == UOP_KERNEL) {
                    partner_slot = loc + 1;
                    partner = a1;
                } else {
                    partner_slot = loc + 0;
                    partner = a0;
                }
            }
        }
        thvm_step_graph_set_before_top_partner(partner, partner_slot);
    } else {
        thvm_step_graph_set_before_top_partner(term_era(), 0);
    }
    if (term_tag(before) == TAG_ERA) {
        u64 el = term_val(before);
        Term payload = (el > 0 && el < ctx->heap_pos) ? thvm_era_payload(ctx, heap_read(ctx, el)) : term_era();
        thvm_step_graph_set_before_era_payload(payload);
    } else {
        thvm_step_graph_set_before_era_payload(term_era());
    }
}


static Term thvm_step_seed_root_grad(TinyHVM *ctx, Term t) {
    // Keep one mirrored heap cell for the current phase-1 root term.
    // Dumping code is heap-driven; without this, newly returned roots can disappear.
    if (step_root_slot == 0 || step_root_slot >= ctx->heap_pos)
        step_root_slot = heap_alloc(ctx, 1);
    heap_set(ctx, step_root_slot, t);
    return t;
}

static void thvm_step_graph_scrub_detached_eras(TinyHVM *ctx) {
    u32 removed = 0;
    u32 first_tag = 0;
    u64 first_loc = 0;
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) != TAG_ERA || term_val(ht) == 0) continue;
        if (step_has_parent_ref(ctx, ht)) continue;
        u64 loc = term_val(ht);
        if (loc > 0 && loc < ctx->heap_pos) {
            if (removed == 0) {
                first_tag = term_tag(heap_read(ctx, loc));
                first_loc = loc;
            }
            removed++;
            heap_set(ctx, loc, term_era());
        }
        ctx->heap[h] = term_era();
    }
    // #region agent log
    do {
        static u32 scrub_dbg_count = 0;
        if (removed == 0 || scrub_dbg_count >= 12) break;
        scrub_dbg_count++;
        char _dbg[192];
        snprintf(_dbg, sizeof(_dbg),
                 "{\"removed\":%u,\"first_tag\":%u,\"first_loc\":%llu}",
                 removed, first_tag, (unsigned long long)first_loc);
        thvm_agent_debug_log("pre-fix", "H10", "src/schedule/_.c:792",
                             "step_graph_scrub_detached_eras", _dbg);
    } while (0);
    // #endregion
}


static int thvm_eval_mixed_dispatch_enabled(void) {
    const char *env = getenv("THVM_EVAL_MIXED_DISPATCH");
    return env && env[0] && strcmp(env, "0") != 0;
}

static Term thvm_eval_collect_fixed_point(TinyHVM *ctx, Term t, int dispatch_enabled) {
    u8 *reach = NULL;
    size_t reach_cap = 0;
    step_root_slot = 0;
    Term traced = thvm_step_seed_root_grad(ctx, t);
    ctx->dispatch_enabled = dispatch_enabled;
    while (thvm_reduce_step_collect(ctx, &traced, NULL, &reach, &reach_cap)) {}
    ctx->dispatch_enabled = 0;
    free(reach);
    if (step_root_slot > 0 && step_root_slot < ctx->heap_pos)
        heap_set(ctx, step_root_slot, term_era());
    step_root_slot = 0;
    return traced;
}

static Term thvm_eval_fuse_fixed_point(TinyHVM *ctx, Term t, int mixed_dispatch) {
    u64 fuse_loc = heap_alloc(ctx, 1);
    heap_set(ctx, fuse_loc, t);
    return thvm_eval_collect_fixed_point(ctx, term_new(TAG_TOP, UOP_FUSE, fuse_loc),
                                         mixed_dispatch);
}

static Term thvm_eval_exec_fixed_point(TinyHVM *ctx, Term t) {
    return thvm_eval_collect_fixed_point(ctx, t, 1);
}

static Term thvm_eval_reduce_fused(TinyHVM *ctx, Term t) {
    int mixed_dispatch = thvm_eval_mixed_dispatch_enabled();
    return thvm_eval_fuse_fixed_point(ctx, t, mixed_dispatch);
}

// One step-by-step IC reduction: fires a single visible interaction at the
// next redex (root if active, otherwise the first reachable heap agent).
// Mutates *traced in place. If meta != NULL, populates with the pre-redex
// term and the heap slot it occupied (0 = root). reach/reach_cap are scratch
// buffers reused across calls; pass &reach/&cap once and the function grows
// them as the heap expands. Returns 1 if an interaction fired, 0 at NF.
//
// This is THE step primitive: thvm_reduce_collect loops it without meta;
// thvm_trace_step_graph_session loops it with meta to drive the dumper.
static int thvm_reduce_step_collect(TinyHVM *ctx, Term *traced_io,
                                    StepMeta *meta,
                                    u8 **reach_io, size_t *reach_cap_io) {
    if ((size_t)ctx->heap_pos > *reach_cap_io) {
        size_t new_cap = (size_t)ctx->heap_pos;
        u8 *nr = (u8 *)realloc(*reach_io, new_cap);
        if (!nr) return 0;
        memset(nr + *reach_cap_io, 0, new_cap - *reach_cap_io);
        *reach_io = nr;
        *reach_cap_io = new_cap;
    }
    Term traced = *traced_io;

    // ── Root-driven attempt: reduce the visible root for one step. ──
    {
        Term container_before = traced;
        Term predicted_before = traced;
        u64 predicted_source_slot = step_root_slot;
        char predicted_name[160] = {0};
        thvm_step_predict_next_redex(ctx, traced, &predicted_before, NULL);
        thvm_step_capture_step_before_meta(ctx, predicted_before);
        predicted_source_slot = thvm_step_redex_source_slot(ctx, step_root_slot, traced, predicted_before);
        thvm_step_graph_name_for_interaction(ctx, predicted_source_slot, predicted_before,
                                             predicted_name, sizeof(predicted_name));

        // Only use whole-root reduction when the next visible redex is the
        // root itself. If the actual next step lives inside the heap, driving
        // reduction from the root can also collapse administrative parents
        // (for example APP(NUM, x) or FUSE(CTR(...))) in the same budget-1
        // call, which makes one snapshot contain multiple interactions.
        if (predicted_source_slot == step_root_slot) {
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
            Term result = thvm_reduce_steps(ctx, traced, 1);
            int did_fire = (ctx->steps_taken > 0 && ctx->trace_count > 0) ||
                           (ctx->itrs != itrs_before);

            Term before = predicted_before;
            if (ctx->trace_count > 0) {
                Term traced_before = term_new((u8)tr.before_tag, tr.before_ext, tr.before_loc);
                before = traced_before;
                if (traced_before != predicted_before) {
                    predicted_source_slot = thvm_step_redex_source_slot(ctx, step_root_slot,
                                                                        container_before, traced_before);
                    thvm_step_graph_set_before_grad_y(term_era());
                    thvm_step_graph_set_before_era_payload(term_era());
                    thvm_step_graph_set_before_top_era(0);
                    thvm_step_graph_set_before_top_add_zero(0);
                    thvm_step_graph_set_before_top_partner(term_era(), 0);
                    thvm_step_graph_name_for_interaction(ctx, predicted_source_slot, traced_before,
                                                         predicted_name, sizeof(predicted_name));
                }
            }

            ctx->trace_buf = saved_buf;
            ctx->trace_count = saved_count;
            ctx->trace_cap = saved_cap;
            ctx->trace_enabled = saved_en;

            if (did_fire) {
                traced = result;
                if (step_root_slot > 0 && step_root_slot < ctx->heap_pos)
                    heap_set(ctx, step_root_slot, traced);
                *traced_io = traced;
                if (meta) {
                    meta->before = before;
                    meta->source_slot = predicted_source_slot;
                    meta->kind = THVM_STEP_KIND_ROOT;
                    snprintf(meta->name, sizeof(meta->name), "%s", predicted_name);
                }
                return 1;
            }
        }
    }
    int root_terminal = step_trace_root_is_terminal(ctx, traced);

    // ── Heap sweep: fire on the first reachable heap redex, then detached cleanup. ──
    u8 *reach = *reach_io;
    step_mark_reachable_slots(ctx, traced, reach);
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        if (h == step_root_slot) continue;
        if (step_slot_is_local_assign_target(ctx, h)) continue;
        Term ht = ctx->heap[h];
        int reachable = !(reach && !reach[h]);
        if (!reachable && !step_term_needs_global_cleanup(ctx, ht)) continue;
        if (!step_term_maybe_active(ctx, ht)) continue;
        if ((term_tag(ht) == TAG_DP0 || term_tag(ht) == TAG_DP1) &&
            reachable &&
            !step_term_first_reachable_occurrence(ctx, h, ht, reach)) continue;
        if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_GRAD &&
            reachable &&
            !step_term_first_reachable_occurrence(ctx, h, ht, reach)) continue;

        Term container_before = ht;
        Term predicted_before = ht;
        u64 predicted_source_slot = h;
        u8 predicted_kind = reachable ? THVM_STEP_KIND_HEAP
                                      : THVM_STEP_KIND_SWEEP;
        char predicted_name[160] = {0};
        thvm_step_predict_next_redex(ctx, ht, &predicted_before, NULL);
        thvm_step_capture_step_before_meta(ctx, predicted_before);
        predicted_source_slot = thvm_step_redex_source_slot(ctx, h, ht, predicted_before);
        thvm_step_graph_name_for_interaction(ctx, predicted_source_slot, predicted_before,
                                             predicted_name, sizeof(predicted_name));
        if (predicted_kind == THVM_STEP_KIND_SWEEP || root_terminal)
            thvm_step_meta_apply_kind_name(THVM_STEP_KIND_SWEEP,
                                           predicted_name, sizeof(predicted_name));

        u64 target_slot = (predicted_source_slot > 0 && predicted_source_slot < ctx->heap_pos &&
                           heap_read(ctx, predicted_source_slot) == predicted_before)
                        ? predicted_source_slot
                        : h;
        Term target_before = (target_slot > 0 && target_slot < ctx->heap_pos)
                           ? heap_read(ctx, target_slot)
                           : ht;
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
        Term hr = thvm_reduce_steps(ctx, target_before, 1);
        int did_fire = (ctx->steps_taken > 0 && ctx->trace_count > 0) ||
                       (ctx->itrs != itrs_before);
        Term before_h = predicted_before;
        if (ctx->trace_count > 0) {
            Term tb = term_new((u8)tr.before_tag, tr.before_ext, tr.before_loc);
            before_h = tb;
            if (tb != predicted_before) {
                predicted_source_slot = thvm_step_redex_source_slot(ctx, h, container_before, tb);
                thvm_step_graph_set_before_grad_y(term_era());
                thvm_step_graph_set_before_era_payload(term_era());
                thvm_step_graph_set_before_top_era(0);
                thvm_step_graph_set_before_top_add_zero(0);
                thvm_step_graph_set_before_top_partner(term_era(), 0);
                thvm_step_graph_name_for_interaction(ctx, predicted_source_slot, tb,
                                                     predicted_name, sizeof(predicted_name));
                if (predicted_kind == THVM_STEP_KIND_SWEEP || root_terminal)
                    thvm_step_meta_apply_kind_name(THVM_STEP_KIND_SWEEP,
                                                   predicted_name, sizeof(predicted_name));
            }
        }
        ctx->trace_buf = saved_buf;
        ctx->trace_count = saved_count;
        ctx->trace_cap = saved_cap;
        ctx->trace_enabled = saved_en;

        if (!did_fire) continue;

        if (term_tag(target_before) == TAG_TOP && term_ext(target_before) == UOP_GRAD) {
            for (u64 i = 1; i < ctx->heap_pos; i++)
                if (ctx->heap[i] == target_before) ctx->heap[i] = hr;
        } else {
            ctx->heap[target_slot] = hr;
        }
        if (step_root_slot > 0 && step_root_slot < ctx->heap_pos)
            heap_set(ctx, step_root_slot, traced);
        *traced_io = traced;
        if (meta) {
            meta->before = before_h;
            meta->source_slot = predicted_source_slot;
            meta->kind = (predicted_kind == THVM_STEP_KIND_SWEEP || root_terminal)
                       ? THVM_STEP_KIND_SWEEP
                       : THVM_STEP_KIND_HEAP;
            snprintf(meta->name, sizeof(meta->name), "%s", predicted_name);
        }
        return 1;
    }
    return 0;
}

// Deep-clone the input so the trace's heap mutations don't corrupt the
// caller's view of the original term. FUSE-wraps the clone so compute ops
// fire (ADD/MUL etc. are WNF without it), seeds the root mirror slot, and
// returns the wrapped term plus a reach buffer the loop will reuse.
// step_root_slot is set as a side effect.
static Term thvm_step_session_init(TinyHVM *ctx, Term root,
                                   u8 **reach_out, size_t *reach_cap_out) {
    step_root_slot = 0;
    Term cloned = term_clone(ctx, root);
    u64 fuse_loc = heap_alloc(ctx, 1);
    heap_set(ctx, fuse_loc, cloned);
    Term traced = term_new(TAG_TOP, UOP_FUSE, fuse_loc);
    traced = thvm_step_seed_root_grad(ctx, traced);
    *reach_cap_out = (size_t)ctx->heap_pos;
    *reach_out = (u8 *)calloc(*reach_cap_out ? *reach_cap_out : 1, 1);
    return traced;
}

static Term thvm_trace_step_graph_session(TinyHVM *ctx, Term traced) {
    u8 *reach = NULL;
    size_t reach_cap = 0;
    traced = thvm_step_session_init(ctx, traced, &reach, &reach_cap);
    thvm_step_graph_eval_begin(ctx, traced);

    for (u32 guard = 0; guard < 100000; guard++) {
        StepMeta meta;
        if (!thvm_reduce_step_collect(ctx, &traced, &meta, &reach, &reach_cap))
            break;
        thvm_step_graph_after_interaction(ctx, meta.source_slot, meta.before, meta.name, traced);
    }
    free(reach);

    // Finalize BEFORE phase-2 quiesce so the final snapshot reflects the
    // step-by-step IC normal form (e.g. the KERNEL DAG), not the dispatched
    // tensor produced by phase-2.
    thvm_step_graph_set_root(traced);
    thvm_step_graph_finalize(ctx);
    traced = reduce_net_quiesce(ctx, traced);
    step_root_slot = 0;
    sched_planner_release_detached_slots();
    return traced;
}

// Public: drive thvm_reduce_step_collect to NF (or `cap` steps), recording
// each post-step term in `out_terms`. Same harness the step-graph dumper
// uses, minus the meta — for callers that just want the state sequence.
// Does NOT enable dispatch, so the final term is the IC normal form
// (e.g. the KERNEL DAG), not a dispatched TEN.
u32 thvm_reduce_collect(TinyHVM *ctx, Term root, Term *out_terms, u32 cap) {
    u32 n = 0;
    u8 *reach = NULL;
    size_t reach_cap = 0;
    Term traced = thvm_step_session_init(ctx, root, &reach, &reach_cap);
    // Freeze each snapshot with a deep clone so WL can walk it later without
    // seeing heap mutations from subsequent interactions.
    if (n < cap) out_terms[n++] = term_clone(ctx, traced);
    while (n < cap &&
           thvm_reduce_step_collect(ctx, &traced, NULL, &reach, &reach_cap)) {
        out_terms[n++] = term_clone(ctx, traced);
    }
    free(reach);
    step_root_slot = 0;
    return n;
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



// Walk through view ops, monolithic local KERNELs, and DP refs to find the
// innermost schedulable compute/leaf term.
static Term sched_unwrap_views(TinyHVM *ctx, Term t) {
    for (int d = 0; d < 20; d++) {
        // DUP is a hard boundary — do not unwrap through DP0/DP1.
        if (term_tag(t) == TAG_DP0 || term_tag(t) == TAG_DP1) break;
        if (term_tag(t) == TAG_VAR) {
            Term sub = heap_read(ctx, term_val(t));
            if (!term_is_sub(sub)) {
                t = sub;
                continue;
            }
        }
        if (term_tag(t) == TAG_TOP && term_ext(t) == UOP_KERNEL &&
            thvm_kernel_is_monolithic(ctx, t)) {
            t = thvm_kernel_monolithic_payload(ctx, t);
            continue;
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
u64 sched_kernel_locs[SCHED_MAX_KERNELS];

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

        // DUP is a hard fusion boundary. Note the shared child as a forced
        // boundary (it must materialize), then continue traversal so we still
        // discover structure below it — but the boundary prevents fusing through.
        if (tg == TAG_DP0 || tg == TAG_DP1) {
            if (tv == 0 || tv >= ctx->heap_pos || (seen_dup && seen_dup[tv])) continue;
            if (seen_dup) seen_dup[tv] = 1;
            Term shared = heap_read(ctx, tv);
            sched_boundary_note_consumer(ctx, shared, SCHED_PARENT_EXTERNAL);
            SCHED_PUSH(shared, SCHED_PARENT_EXTERNAL);
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

        u32 ar = step_term_arity(tt);
        SchedParentClass pclass = sched_parent_class(tt);
        if (pclass == SCHED_PARENT_NONE && tg == TAG_TOP && is_view_op(term_ext(tt)))
            pclass = item.inherited_class;
        if (tg == TAG_TOP && term_ext(tt) == UOP_KERNEL && thvm_kernel_is_monolithic(ctx, tt)) {
            // Monolithic public kernels are just wrappers around the real
            // compute payload. Do not count the payload as a second consumer of
            // itself, or boundary selection collapses KERNEL@loc back to
            // payload@loc and rewrites miss the actual parent slot.
            SCHED_PUSH(thvm_kernel_monolithic_payload(ctx, tt), item.inherited_class);
            continue;
        }
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

static Term thvm_build_exec_trigger(TinyHVM *ctx, u32 kid, Term dep_chain, u32 flags);

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
    // Encode dependencies as a right-nested CTR chain: CTR(dep0, CTR(dep1, ERA)).
    // Global passes install executable triggers, not structural UOP_KERNEL
    // shells — phase 3 consumes UOP_EXEC nodes directly.
    extern Term fuse_assign_deps[];
    extern u32  fuse_n_assign_deps;

    Term dep_chain = term_era();
    for (u32 di = 0; di < ke->n_deps; di++) {
        u32 dep_kid = ke->dep_kids[di];
        if (dep_kid < SCHED_MAX_KERNELS && sched_kernel_locs[dep_kid]) {
            Term dep_kernel = term_new(TAG_TOP, UOP_EXEC, sched_kernel_locs[dep_kid]);
            u64 ctr_loc = heap_alloc(ctx, 2);
            heap_set(ctx, ctr_loc + 0, dep_kernel);
            heap_set(ctx, ctr_loc + 1, dep_chain);
            dep_chain = term_new(TAG_CTR, 0, ctr_loc);
        }
    }
    for (u32 i = 0; i < fuse_n_assign_deps; i++) {
        u64 ctr_loc = heap_alloc(ctx, 2);
        heap_set(ctx, ctr_loc + 0, fuse_assign_deps[i]);
        heap_set(ctx, ctr_loc + 1, dep_chain);
        dep_chain = term_new(TAG_CTR, 0, ctr_loc);
    }

    Term ft = thvm_build_exec_trigger(ctx, b->kid, dep_chain, 0);
    u64 floc = term_val(ft);
    sched_kernel_locs[b->kid] = floc;

    const ShapeTracker *out_st = tensor_st_get(&ctx->tensors[b->output_tid]);
    if (out_st && out_st->n_views > 0) st_set_tracker(floc, out_st);
    else {
        const View *out_v = tensor_view_get(&ctx->tensors[b->output_tid]);
        View fallback = out_v ? *out_v : view_create(ke->out_shape);
        st_set(floc, &fallback);
    }

    // Legacy SEQ wrapping kept for backward compatibility until
    // nested EXEC dependency chains are fully normalized.
    if (term_tag(dep_chain) != TAG_ERA) {
        // Walk the CTR chain and wrap as SEQ for now.
        Term walk = dep_chain;
        while (term_tag(walk) == TAG_CTR) {
            Term dep = heap_read(ctx, term_val(walk));
            ft = thvm_seq(ctx, dep, ft);
            walk = heap_read(ctx, term_val(walk) + 1);
        }
    }

    thvm_sched_rewrite_remember(ctx, b->root_term, ft);
    sched_replace_term_everywhere(ctx, b->root_term, ft);
    return ft;
}

// Build a UOP_EXEC trigger node with CTR-encoded deps.
// Heap layout: [NUM(kid), dep_chain, NUM(flags)]
static Term thvm_build_exec_trigger(TinyHVM *ctx, u32 kid, Term dep_chain, u32 flags) {
    u64 eloc = heap_alloc(ctx, 3);
    heap_set(ctx, eloc + 0, term_num_u32(kid));
    heap_set(ctx, eloc + 1, dep_chain);
    heap_set(ctx, eloc + 2, term_num_u32(flags));
    return term_new(TAG_TOP, UOP_EXEC, eloc);
}

static Term thvm_sched_dispatch_kernel(TinyHVM *ctx, u32 kid) {
    if (kid >= sched_kernel_count) return term_era();
    // Cache check with epoch invalidation is in UOP_KERNEL/UOP_EXEC handler.
    KernelEntry *ke = &sched_kernels[kid];

    u32 raw_tid = ke->raw_output_tid ? ke->raw_output_tid : ke->output_tid;
    if (raw_tid == 0) return term_era();
    ENSURE(ctx, raw_tid);
    TensorMeta *md = &ctx->tensors[raw_tid];

    if (md->buf_id == 0) {
        if (ke->output_slot) {
            if (sched_slot_bind_output(ctx, ke->output_slot, raw_tid, ke->output_tid) == 0)
                return term_era();
        } else {
            Backend *be = md->backend ? md->backend : ctx_default_backend(ctx);
            if (!be) return term_era();
            u64 bytes = (u64)md->view.numel * dtype_size(md->dtype);
            md->backend = be;
            md->buf_id = be->buf_alloc(bytes);
            if (md->buf_id == 0) return term_era();
            if (ke->output_tid && ke->output_tid != raw_tid) {
                TensorMeta *out_m = &ctx->tensors[ke->output_tid];
                out_m->backend = be;
                out_m->buf_id = md->buf_id;
                if (out_m->backend && out_m->backend->buf_incref)
                    out_m->backend->buf_incref(out_m->buf_id);
            }
        }
    } else if (ke->output_tid && ke->output_tid != raw_tid &&
               ke->output_tid < ctx->tensor_count &&
               ctx->tensors[ke->output_tid].buf_id == 0) {
        if (ke->output_slot) {
            if (sched_slot_bind_output(ctx, ke->output_slot, raw_tid, ke->output_tid) == 0)
                return term_era();
        } else {
            TensorMeta *out_m = &ctx->tensors[ke->output_tid];
            out_m->backend = md->backend;
            out_m->buf_id = md->buf_id;
            if (out_m->backend && out_m->backend->buf_incref)
                out_m->backend->buf_incref(out_m->buf_id);
        }
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

    UOpKernel uk;
    u64 lower_sig = 0;
    int lowered = getenv("THVM_NO_LOWER") ? 0 : thvm_lower_kernel_uop(ctx, ke, &uk, &lower_sig);
    if (getenv("THVM_LOWER_DIAG")) {
        if (lowered) {
            fprintf(stderr,
                    "LOWER_KERNEL_BUILT kid=%u cache_sig=0x%016llx lower_sig=0x%016llx lower_heap=%llu ops=%u rewrites=%u\n",
                    kid,
                    (unsigned long long)ke->normalized_sig,
                    (unsigned long long)lower_sig,
                    (unsigned long long)ctx->lower_ctx.heap_pos,
                    uk.n_ops,
                    ctx->lower_ctx.rewrite_count);
            if (getenv("THVM_LOWER_TRACE"))
                thvm_lower_dump_uop_kernel(&uk);
        } else {
            fprintf(stderr,
                    "LOWER_KERNEL_FALLBACK kid=%u cache_sig=0x%016llx reason=builder\n",
                    kid, (unsigned long long)ke->normalized_sig);
        }
    }

    if (lowered)
        thvm_lower_dump_graphs(ctx, ke, &uk, kid, sched_kernel_locs[kid],
                               ke->normalized_sig, lower_sig,
                               md->backend, md->buf_id, bufs, ke->n_leaves);

    if (lowered && md->backend->dispatch_uop_kernel) {
        md->backend->dispatch_uop_kernel(md->buf_id, bufs, ke->n_leaves, &uk, ke->normalized_sig);
    } else {
        md->backend->dispatch_kernel_rs(
            md->buf_id, bufs, views,
            has_multiview ? st_ptrs : NULL, ke->n_leaves,
            ke->ops, ke->n_ops, &ke->full_shape,
            ke->has_reduce ? &ke->reduce : NULL,
            NULL, NULL, 0, ke->leaf_dtypes, md->dtype);
    }
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
    memset(kid_n_inputs, 0, sizeof(kid_n_inputs));
    // Note: buf_epoch is NOT reset — it persists across scheduling passes
    // so ASSIGN writes from previous iterations are visible.

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

static Term thvm_sched_global_pass(TinyHVM *ctx, Term root) {
    if (!ctx) return root;
    Term sched_root = root;
    if (term_tag(root) == TAG_TOP && term_ext(root) == UOP_KERNEL &&
        thvm_kernel_is_monolithic(ctx, root))
        sched_root = thvm_kernel_monolithic_payload(ctx, root);
    if (sched_all(ctx, sched_root) == 0) return root;
    Term rewritten = thvm_sched_rewrite_get(ctx, sched_root);
    return term_tag(rewritten) != TAG_ERA ? rewritten : root;
}

// === DELETED: 500+ lines of merge passes (4-9), memory analysis, dead kernel redirect ===
// These were post-hoc merge attempts that created broken kernels.
// Replaced by: nothing. The simple 3-pass scheduling is sufficient.
// Kernel count matches tinygrad without merging (the fuser already
// absorbs ew ops into reduce kernels during the walk).

// ── Global compiler passes (kernel DAG redesign layer 2) ──────────
// Run registered named passes on the post-fusion coarse IC.
// Each pass takes the root term and returns a (possibly rewritten) root.
static Term thvm_run_global_passes(TinyHVM *ctx, Term t) {
    FILE *dump = NULL;
    if (getenv("THVM_GRAPH")) {
        char path[512];
        char mkdir_cmd[1024];
        thvm_graph_dump_path(path, sizeof(path), "thvm_global_passes.txt");
        snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p \"%s\"", thvm_graph_dir());
        system(mkdir_cmd);
        dump = fopen(path, "w");
        if (dump) {
            char summary[128];
            thvm_graph_term_summary(t, summary, sizeof(summary));
            fprintf(dump, "registered_passes=%u\n", ctx->n_compiler_passes);
            fprintf(dump, "mixed_dispatch=%d\n", thvm_eval_mixed_dispatch_enabled());
            fprintf(dump, "input=%s\n", summary);
        }
    }
    if (ctx->n_compiler_passes == 0) {
        if (dump) {
            char summary[128];
            thvm_graph_term_summary(t, summary, sizeof(summary));
            fprintf(dump, "status=no passes registered\n");
            fprintf(dump, "output=%s\n", summary);
            fclose(dump);
        }
        return t;
    }
    for (u32 i = 0; i < ctx->n_compiler_passes; i++) {
        if (ctx->compiler_passes[i]) {
            Term before = t;
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "GLOBAL_PASS[%u]: %s\n", i,
                        ctx->pass_names[i] ? ctx->pass_names[i] : "(unnamed)");
            t = ctx->compiler_passes[i](ctx, t);
            if (dump) {
                char before_summary[128], after_summary[128];
                thvm_graph_term_summary(before, before_summary, sizeof(before_summary));
                thvm_graph_term_summary(t, after_summary, sizeof(after_summary));
                fprintf(dump, "pass[%u].name=%s\n", i,
                        ctx->pass_names[i] ? ctx->pass_names[i] : "(unnamed)");
                fprintf(dump, "pass[%u].before=%s\n", i, before_summary);
                fprintf(dump, "pass[%u].after=%s\n", i, after_summary);
            }
        }
    }
    if (dump) {
        char summary[128];
        thvm_graph_term_summary(t, summary, sizeof(summary));
        fprintf(dump, "output=%s\n", summary);
        fclose(dump);
    }
    return t;
}

void thvm_register_pass(TinyHVM *ctx, ThvmCompilerPass pass, const char *name) {
    if (ctx->n_compiler_passes >= 16) return;
    u32 idx = ctx->n_compiler_passes++;
    ctx->compiler_passes[idx] = pass;
    ctx->pass_names[idx] = name;
}

Term thvm_eval(TinyHVM *ctx, Term t) {
    int outermost = (ctx->eval_depth++ == 0);
    int run_step_graph = outermost && getenv("THVM_STEP_GRAPH") && !ctx->step_graph_consumed;
    int run_coarse_graph = outermost && getenv("THVM_GRAPH") && !ctx->coarse_graph_consumed;
    if (run_step_graph) {
        ctx->step_graph_consumed = 1;
        u32 snap_count = 0;
        StepGraphTensorSnap *snaps = thvm_step_graph_snapshot_tensors(ctx, &snap_count);
        // session_init clones; we just hand off the original term.
        Term traced = t;
        // #region agent log
        do {
            static u32 step_trace_start_dbg_count = 0;
            if (step_trace_start_dbg_count >= 8) break;
            step_trace_start_dbg_count++;
            char _dbg[256];
            snprintf(_dbg, sizeof(_dbg),
                     "{\"root_tag\":%u,\"root_ext\":%u,\"root_val\":%llu,"
                     "\"traced_tag\":%u,\"traced_ext\":%u,\"traced_val\":%llu,"
                     "\"snap_count\":%u,\"heap_pos\":%llu}",
                     (u32)term_tag(t), (u32)term_ext(t), (unsigned long long)term_val(t),
                     (u32)term_tag(traced), (u32)term_ext(traced), (unsigned long long)term_val(traced),
                     snap_count, (unsigned long long)ctx->heap_pos);
            thvm_agent_debug_log("pre-fix", "H2", "src/schedule/_.c:1525",
                                 "step_graph_trace_start", _dbg);
        } while (0);
        // #endregion
        traced = thvm_trace_step_graph_session(ctx, traced);
        thvm_step_graph_restore_tensors(ctx, snaps, snap_count);
        // Keep step-graph tracing path semantically aligned with normal eval:
        // after emitting graphs, run the standard pipeline (without tracing)
        // to settle any residual administrative structure.
        u8 saved_settled_replay = ctx->step_graph_settled_replay;
        ctx->step_graph_settled_replay = 1;
        Term settled = thvm_eval(ctx, t);
        ctx->step_graph_settled_replay = saved_settled_replay;
        // #region agent log
        do {
            static u32 step_trace_settled_dbg_count = 0;
            if (step_trace_settled_dbg_count >= 8) break;
            step_trace_settled_dbg_count++;
            char _dbg[256];
            snprintf(_dbg, sizeof(_dbg),
                     "{\"returned_tag\":%u,\"returned_ext\":%u,\"returned_val\":%llu,"
                     "\"traced_tag\":%u,\"traced_ext\":%u,\"traced_val\":%llu}",
                     (u32)term_tag(settled), (u32)term_ext(settled), (unsigned long long)term_val(settled),
                     (u32)term_tag(traced), (u32)term_ext(traced), (unsigned long long)term_val(traced));
            thvm_agent_debug_log("pre-fix", "H2", "src/schedule/_.c:1557",
                                 "step_graph_settled_result", _dbg);
        } while (0);
        // #endregion
        ctx->eval_depth--;
        return settled;
    }

    if (run_coarse_graph) {
        ctx->coarse_graph_consumed = 1;
        u32 snap_count = 0;
        StepGraphTensorSnap *snaps = thvm_step_graph_snapshot_tensors(ctx, &snap_count);
        Term traced = term_clone(ctx, t);
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", thvm_graph_dir());
        system(cmd);
        traced = thvm_step_seed_root_grad(ctx, traced);
        char path[512];
        thvm_graph_dump_path(path, sizeof(path), "thvm_0_pre_reduce.dot");
        thvm_heap_dot_set_sched_kernels(0);
        thvm_heap_dot_root(ctx, path, traced);
        if (step_root_slot > 0 && step_root_slot < ctx->heap_pos)
            heap_set(ctx, step_root_slot, term_era());

        // Phase 1: pure IC reduction — combinators fire, compute ops are WNF.
        traced = thvm_reduce(ctx, traced);
        if (getenv("THVM_SCHED_DIAG"))
            fprintf(stderr, "PHASE1_RESULT: tag=%u ext=%u val=%llu\n",
                    term_tag(traced), term_ext(traced), (unsigned long long)term_val(traced));
        traced = thvm_step_seed_root_grad(ctx, traced);
        thvm_graph_dump_path(path, sizeof(path), "thvm_1_post_reduce.dot");
        thvm_heap_dot_set_sched_kernels(0);
        thvm_heap_dot_root(ctx, path, traced);
        if (step_root_slot > 0 && step_root_slot < ctx->heap_pos)
            heap_set(ctx, step_root_slot, term_era());

        // Phase 2: wrap in UOP_FUSE and reduce. By default this produces the
        // post-local coarse graph only; THVM_EVAL_MIXED_DISPATCH=1 preserves
        // the legacy mixed path that also dispatches during this pass.
        traced = thvm_eval_fuse_fixed_point(ctx, traced, thvm_eval_mixed_dispatch_enabled());
        traced = thvm_step_seed_root_grad(ctx, traced);
        thvm_graph_dump_path(path, sizeof(path), "thvm_2_post_dispatch.dot");
        thvm_heap_dot_set_sched_kernels(0);
        thvm_heap_dot_root(ctx, path, traced);
        if (step_root_slot > 0 && step_root_slot < ctx->heap_pos)
            heap_set(ctx, step_root_slot, term_era());
        step_root_slot = 0;

        // Phase 2.5: run global compiler passes.
        traced = thvm_run_global_passes(ctx, traced);
        traced = thvm_step_seed_root_grad(ctx, traced);
        thvm_graph_dump_path(path, sizeof(path), "thvm_3_post_passes.dot");
        thvm_heap_dot_set_sched_kernels(0);
        thvm_heap_dot_root(ctx, path, traced);
        if (step_root_slot > 0 && step_root_slot < ctx->heap_pos)
            heap_set(ctx, step_root_slot, term_era());
        step_root_slot = 0;

        // Phase 3: second reduce for exec triggers.
        if (ctx->n_compiler_passes > 0) {
            traced = thvm_eval_exec_fixed_point(ctx, traced);
        }
        traced = thvm_step_seed_root_grad(ctx, traced);
        thvm_graph_dump_path(path, sizeof(path), "thvm_4_post_exec.dot");
        thvm_heap_dot_set_sched_kernels(0);
        thvm_heap_dot_root(ctx, path, traced);
        if (step_root_slot > 0 && step_root_slot < ctx->heap_pos)
            heap_set(ctx, step_root_slot, term_era());
        step_root_slot = 0;
        thvm_step_graph_restore_tensors(ctx, snaps, snap_count);
        sched_planner_release_detached_slots();
        // Like step-graph tracing, the coarse-graph path only exists to emit
        // diagnostics. After dumping the explicit coarse snapshots, finish with
        // the ordinary eval path on the untouched original program so the
        // caller receives the real final result.
        Term settled = thvm_eval(ctx, t);
        ctx->eval_depth--;
        return settled;
    }

    // Phase 1+2: reduce to WNF then fuse (creates UOP_KERNEL nodes + dispatches)
    t = thvm_eval_reduce_fused(ctx, t);
    // Phase 2.5: run global compiler passes (may install UOP_EXEC triggers)
    t = thvm_run_global_passes(ctx, t);
    // Phase 3: second reduce — fires any UOP_EXEC triggers from passes
    if (ctx->n_compiler_passes > 0) {
        t = thvm_eval_exec_fixed_point(ctx, t);
    }
    sched_planner_release_detached_slots();
    ctx->eval_depth--;
    return t;
}

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_eval(ctx, t);
}
