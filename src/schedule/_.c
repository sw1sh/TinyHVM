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

static const char *thvm_graph_dir(void) {
    const char *dir = getenv("THVM_GRAPH_DIR");
    return (dir && dir[0]) ? dir : "/tmp";
}

static void thvm_graph_dump_path(char *buf, size_t nbuf, const char *name) {
    snprintf(buf, nbuf, "%s/%s", thvm_graph_dir(), name);
}

static u32 phase1_top_arity(u32 ext) {
    if (ext == UOP_FUSING) return 0;
    if (ext == UOP_WHERE || ext == UOP_IFZ) return 3;
    if (ext == UOP_GRAD) return 2;
    if (ext == UOP_LOG_PRINT) return 1;
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
        if (term_tag(t) != TAG_TOP || !is_view_op(term_ext(t))) break;
        t = heap_read(ctx, term_val(t));
    }
    return t;
}

static int sched_is_kernelizable_uop(u32 uop) {
    return is_elementwise(uop) || uop == UOP_SUM || uop == UOP_RMAX;
}

static int sched_is_kernelizable_term(Term t) {
    return term_tag(t) == TAG_TOP && sched_is_kernelizable_uop(term_ext(t));
}

static const View *sched_term_view(TinyHVM *ctx, Term t) {
    if (term_tag(t) == TAG_TOP) return st_get(term_val(t));
    if (term_tag(t) == TAG_TEN) return tensor_view_get(&ctx->tensors[(u32)term_val(t)]);
    return NULL;
}

static const ShapeTracker *sched_term_tracker(TinyHVM *ctx, Term t) {
    if (term_tag(t) == TAG_TOP) return st_get_tracker(term_val(t));
    if (term_tag(t) == TAG_TEN) return tensor_st_get(&ctx->tensors[(u32)term_val(t)]);
    return NULL;
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
    Term *work = work_cap ? (Term *)malloc(sizeof(Term) * (size_t)work_cap) : NULL;
    u64 wp = 0;
    #define SCHED_PUSH(_tt) do { if (work && wp < work_cap) work[wp++] = (_tt); } while (0)
    SCHED_PUSH(root);

    while (work && wp > 0) {
        Term tt = work[--wp];
        u8 tg = term_tag(tt);
        u64 tv = term_val(tt);

        if (tg == TAG_DP0 || tg == TAG_DP1) {
            if (tv == 0 || tv >= ctx->heap_pos || (seen_dup && seen_dup[tv])) continue;
            if (seen_dup) seen_dup[tv] = 1;
            SCHED_PUSH(heap_read(ctx, tv));
            continue;
        }
        if (tg == TAG_ERA) {
            if (tv == 0 || tv >= ctx->heap_pos) continue;
            if (!seen_slot || !seen_slot[tv]) {
                if (seen_slot) seen_slot[tv] = 1;
                SCHED_PUSH(heap_read(ctx, tv));
            }
            continue;
        }

        u32 ar = phase1_term_arity(tt);
        SchedParentClass pclass = sched_parent_class(tt);
        for (u32 i = 0; i < ar; i++) {
            u64 p = tv + i;
            if (p >= ctx->heap_pos) continue;
            Term child = heap_read(ctx, p);
            sched_boundary_note_consumer(ctx, child, pclass);
            if (!seen_slot || !seen_slot[p]) {
                if (seen_slot) seen_slot[p] = 1;
                SCHED_PUSH(child);
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

    u32 raw_tid = tensor_create(ctx, raw_v->shape, DTYPE_F32);
    TensorMeta *raw_m = &ctx->tensors[raw_tid];
    raw_m->creator_op = UOP_FUSING;
    raw_m->fusing_loc = term_val(b->compute_term);

    b->raw_output_tid = raw_tid;
    b->output_tid = raw_tid;

    const View *root_v = sched_term_view(ctx, b->root_term);
    const ShapeTracker *root_st = sched_term_tracker(ctx, b->root_term);
    if (root_v && (!shape_eq(root_v->shape, raw_v->shape) || (root_st && root_st->n_views > 1))) {
        u32 out_tid = tensor_view_of(ctx, raw_tid, *root_v);
        TensorMeta *out_m = &ctx->tensors[out_tid];
        out_m->creator_op = UOP_FUSING;
        out_m->src_ids[0] = raw_tid;
        out_m->fusing_loc = term_val(b->compute_term);
        if (root_st && root_st->n_views > 0) out_m->st = *root_st;
        b->output_tid = out_tid;
    }
    sched_boundary_output_tids[b->kid] = b->output_tid;
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
    Term ft = term_new(TAG_TOP, UOP_FUSING, floc);
    sched_kernel_locs[b->kid] = floc;

    const ShapeTracker *out_st = tensor_st_get(&ctx->tensors[b->output_tid]);
    if (out_st && out_st->n_views > 0) st_set_tracker(floc, out_st);
    else {
        const View *out_v = tensor_view_get(&ctx->tensors[b->output_tid]);
        View fallback = out_v ? *out_v : view_create(ke->out_shape);
        st_set(floc, &fallback);
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

    u32 bufs[FUSE_MAX_LEAVES];
    const View *views[FUSE_MAX_LEAVES];
    for (u32 i = 0; i < ke->n_leaves; i++) {
        if (ke->leaf_kinds[i] == KERNEL_LEAF_TENSOR) {
            u32 lid = ke->leaf_ids[i];
            if (lid == 0) return term_era();
            ENSURE(ctx, lid);
            bufs[i] = ctx->tensors[lid].buf_id;
            views[i] = &ke->leaf_views[i];
            continue;
        }
        if (ke->leaf_kinds[i] == KERNEL_LEAF_NUM) {
            f32 val = ke->leaf_nums[i];
            Term scalar = thvm_tensor(ctx, &val, (Shape){.dims={1}, .rank=1});
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
        NULL, NULL, 0);
    ctx->itrs++;
    md->creator_op = UOP_FUSING;
    md->fusing_loc = sched_kernel_locs[kid];

    if (ke->output_tid && ke->output_tid != raw_tid) {
        TensorMeta *out_m = &ctx->tensors[ke->output_tid];
        out_m->creator_op = UOP_FUSING;
        out_m->src_ids[0] = raw_tid;
        out_m->fusing_loc = sched_kernel_locs[kid];
    }

    Term result = term_ten(ke->output_tid ? ke->output_tid : raw_tid, DTYPE_F32);
    kid_results[kid] = result;
    return result;
}

static u32 sched_all(TinyHVM *ctx, Term root) {
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
    if (getenv("THVM_STEP_GRAPH")) {
        phase1_root_slot = 0;
        t = thvm_phase1_seed_root_grad(ctx, t);
        thvm_step_graph_eval_begin(ctx, t);
        t = thvm_phase1_structural_nf(ctx, t);
        thvm_step_graph_finalize(ctx);
        phase1_root_slot = 0;
        return t;
    }

    if (getenv("THVM_GRAPH")) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", thvm_graph_dir());
        system(cmd);
        t = thvm_phase1_seed_root_grad(ctx, t);
        char path[512];
        thvm_graph_dump_path(path, sizeof(path), "thvm_0_pre_reduce.dot");
        thvm_heap_dot_root(ctx, path, t);
        if (phase1_root_slot > 0 && phase1_root_slot < ctx->heap_pos)
            heap_set(ctx, phase1_root_slot, term_era());
    }
    t = thvm_reduce(ctx, t);
    if (getenv("THVM_GRAPH")) {
        t = thvm_phase1_seed_root_grad(ctx, t);
        char path[512];
        thvm_graph_dump_path(path, sizeof(path), "thvm_1_post_reduce.dot");
        thvm_heap_dot_root(ctx, path, t);
        if (phase1_root_slot > 0 && phase1_root_slot < ctx->heap_pos)
            heap_set(ctx, phase1_root_slot, term_era());
    }

    sched_all(ctx, t);
    {
        Term rt = thvm_sched_rewrite_get(ctx, t);
        if (!(term_tag(rt) == TAG_ERA && term_val(rt) == 0))
            t = rt;
    }
    if (getenv("THVM_GRAPH")) {
        t = thvm_phase1_seed_root_grad(ctx, t);
        char path[512];
        thvm_graph_dump_path(path, sizeof(path), "thvm_2_post_sched.dot");
        thvm_heap_dot_root(ctx, path, t);
        if (phase1_root_slot > 0 && phase1_root_slot < ctx->heap_pos)
            heap_set(ctx, phase1_root_slot, term_era());
    }

    t = thvm_reduce(ctx, t);
    if (getenv("THVM_GRAPH")) {
        t = thvm_phase1_seed_root_grad(ctx, t);
        char path[512];
        thvm_graph_dump_path(path, sizeof(path), "thvm_3_post_dispatch.dot");
        thvm_heap_dot_root(ctx, path, t);
        if (phase1_root_slot > 0 && phase1_root_slot < ctx->heap_pos)
            heap_set(ctx, phase1_root_slot, term_era());
        phase1_root_slot = 0;
    }
    return t;
}

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_eval(ctx, t);
}
