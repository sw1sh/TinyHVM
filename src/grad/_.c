// grad/_.c — Autograd: thvm_grad, thvm_grad_multi
//
// Pure IC term constructors. No computation here — everything
// is driven by thvm_reduce through the UOP_GRAD interaction handler.

#define GRAD_TARGET_CTX_MAX 8
#define GRAD_LOC_MAX 8192
#define GRAD_GROUP_MAX 8192
#define GRAD_GROUP_ENTRIES_MAX 16384

typedef struct {
    Term term;
    u32  tid;
    Term slot;
} GradTargetEntry;

typedef struct {
    u32 start;
    u32 n;
} GradTargetGroup;

typedef struct {
    TinyHVM *ctx;
    u32 n_group;
    u32 n_entry;
    GradTargetGroup groups[GRAD_GROUP_MAX];
    GradTargetEntry entries[GRAD_GROUP_ENTRIES_MAX];
    u8 modes[GRAD_LOC_MAX];
    u32 group_ids[GRAD_LOC_MAX]; // 1-based group id; 0 means none
    Term bundles[GRAD_LOC_MAX];
    u32 n_loc;
    u64 locs[GRAD_LOC_MAX];
    Term xs[GRAD_LOC_MAX];
} GradTargetSet;

static GradTargetSet grad_target_sets[GRAD_TARGET_CTX_MAX];

static void grad_targets_reset(GradTargetSet *s) {
    if (!s) return;
    s->n_group = 0;
    s->n_entry = 0;
    s->n_loc = 0;
    for (u32 i = 0; i < GRAD_LOC_MAX; i++) {
        s->modes[i] = GRAD_MODE_DROP;
        s->group_ids[i] = 0;
        s->bundles[i] = term_era();
        s->xs[i] = thvm_any();
    }
}

static u32 thvm_grad_seed_dtype_hint(TinyHVM *ctx, Term t) {
    for (u32 depth = 0; depth < 32; depth++) {
        while (term_tag(t) == TAG_DP0 || term_tag(t) == TAG_DP1)
            t = heap_read(ctx, term_val(t));

        if (term_tag(t) == TAG_TEN) {
            u32 tid = (u32)term_val(t);
            return (tid < ctx->tensor_count) ? ctx->tensors[tid].dtype : DTYPE_F32;
        }
        if (term_tag(t) == TAG_NUM) return DTYPE_F32;
        if (term_tag(t) != TAG_TOP) break;

        u64 loc = term_val(t);
        if (loc >= ctx->heap_pos) break;

        switch (term_ext(t)) {
            case UOP_WHERE:
            case UOP_IFZ:
                t = heap_read(ctx, loc + 1);
                continue;
            default:
                t = heap_read(ctx, loc + 0);
                continue;
        }
    }
    return DTYPE_F32;
}

static Term thvm_grad_seed_like(TinyHVM *ctx, Term loss) {
    u32 dtype = thvm_grad_seed_dtype_hint(ctx, loss);
    // Backward seeds should be scalar tensors, not TAG_NUM literals. Keep the
    // seed at the loss float dtype when we know it; integer losses still fall
    // back to f32 until compute kernels become genuinely typed.
    if (dtype != DTYPE_F32 && dtype != DTYPE_F16) dtype = DTYPE_F32;
    return thvm_scalar_typed(ctx, 1.0f, dtype);
}

static GradTargetSet *grad_targets_get(TinyHVM *ctx, int create) {
    for (u32 i = 0; i < GRAD_TARGET_CTX_MAX; i++) {
        if (grad_target_sets[i].ctx == ctx) return &grad_target_sets[i];
    }
    if (!create) return NULL;
    for (u32 i = 0; i < GRAD_TARGET_CTX_MAX; i++) {
        if (grad_target_sets[i].ctx == NULL) {
            grad_target_sets[i].ctx = ctx;
            grad_targets_reset(&grad_target_sets[i]);
            return &grad_target_sets[i];
        }
    }
    return NULL;
}

static int grad_loc_index(GradTargetSet *s, u64 grad_loc) {
    if (!s) return -1;
    for (u32 i = 0; i < s->n_loc; i++) {
        if (s->locs[i] == grad_loc) return (int)i;
    }
    return -1;
}

static int grad_loc_ensure(GradTargetSet *s, u64 grad_loc) {
    int idx = grad_loc_index(s, grad_loc);
    if (idx >= 0) return idx;
    if (!s || s->n_loc >= GRAD_LOC_MAX) return -1;
    idx = (int)s->n_loc++;
    s->locs[idx] = grad_loc;
    s->xs[idx] = thvm_any();
    s->modes[idx] = GRAD_MODE_DROP;
    s->group_ids[idx] = 0;
    s->bundles[idx] = term_era();
    return idx;
}

static u32 grad_group_alloc(GradTargetSet *s, Term *params, Term *slots, u32 n_params) {
    if (!s || n_params == 0) return 0;
    assert(n_params <= THVM_GRAD_TARGETS_MAX);
    if (s->n_group >= GRAD_GROUP_MAX) return 0;
    if (s->n_entry + n_params > GRAD_GROUP_ENTRIES_MAX) return 0;
    u32 gid = ++s->n_group; // 1-based
    GradTargetGroup *g = &s->groups[gid - 1];
    g->start = s->n_entry;
    g->n = n_params;
    for (u32 i = 0; i < n_params; i++) {
        Term p = params[i];
        s->entries[s->n_entry++] = (GradTargetEntry){
            .term = p,
            .tid  = term_tag(p) == TAG_TEN ? (u32)term_val(p) : ~0u,
            .slot = slots ? slots[i] : term_era(),
        };
    }
    return gid;
}

static int grad_loc_group(GradTargetSet *s, u64 grad_loc, GradTargetGroup **out_group) {
    if (out_group) *out_group = NULL;
    if (!s) return 0;
    int idx = grad_loc_index(s, grad_loc);
    if (idx < 0) return 0;
    u32 gid = s->group_ids[idx];
    if (gid == 0 || gid > s->n_group) return 0;
    if (out_group) *out_group = &s->groups[gid - 1];
    return 1;
}

static Term thvm_grad_bundle_new(TinyHVM *ctx, u32 n_params) {
    if (n_params == 0) return term_new(TAG_CTR, 0, 0);
    assert(n_params < 256 && "grad bundle arity exceeds TAG_CTR ext range");
    u64 loc = heap_alloc(ctx, n_params);
    for (u32 i = 0; i < n_params; i++)
        heap_set(ctx, loc + i, term_num_f32(0.0f));
    return term_new(TAG_CTR, (u8)n_params, loc);
}

static Term thvm_grad_bundle_whnf(TinyHVM *ctx, Term bundle) {
    if (term_tag(bundle) == TAG_APP) {
        u64 app_loc = term_val(bundle);
        if (app_loc + 1 < ctx->heap_pos) {
            Term fun = heap_read(ctx, app_loc + 0);
            if (term_tag(fun) == TAG_TOP && term_ext(fun) == UOP_GRAD) {
                Term kept = thvm_grad_keep_bundle_get(ctx, term_val(fun));
                if (term_tag(kept) == TAG_CTR) return kept;
            }
        }
    }
    // Keep bundles are fixed-arity carriers. Do not reduce TAG_CTR directly:
    // generic CTR reduction collapses arity-1 containers to their head term,
    // which destroys bundle structure for single-parameter keep mode.
    if (term_tag(bundle) == TAG_CTR) return bundle;
    return thvm_reduce(ctx, bundle);
}

void thvm_grad_targets_clear(TinyHVM *ctx) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s) return;
    grad_targets_reset(s);
}

void thvm_grad_target_set(TinyHVM *ctx, u64 grad_loc, Term x) {
    GradTargetSet *s = grad_targets_get(ctx, 1);
    if (!s) return;
    int idx = grad_loc_ensure(s, grad_loc);
    if (idx < 0) return;
    s->xs[idx] = x;
}

Term thvm_grad_target_get(TinyHVM *ctx, u64 grad_loc) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s) return thvm_any();
    int idx = grad_loc_index(s, grad_loc);
    if (idx >= 0) return s->xs[idx];
    return thvm_any();
}

void thvm_grad_mode_set(TinyHVM *ctx, u64 grad_loc, u32 mode) {
    GradTargetSet *s = grad_targets_get(ctx, 1);
    if (!s) return;
    int idx = grad_loc_ensure(s, grad_loc);
    if (idx < 0) return;
    s->modes[idx] = (u8)mode;
}

void thvm_grad_targets_set_for_loc(TinyHVM *ctx, u64 grad_loc, Term *params, Term *grad_slots, u32 n_params) {
    GradTargetSet *s = grad_targets_get(ctx, 1);
    if (!s) return;
    int idx = grad_loc_ensure(s, grad_loc);
    if (idx < 0) return;
    s->group_ids[idx] = grad_group_alloc(s, params, grad_slots, n_params);
}

void thvm_grad_targets_share(TinyHVM *ctx, u64 dst_loc, u64 src_loc) {
    GradTargetSet *s = grad_targets_get(ctx, 1);
    if (!s) return;
    int src_idx = grad_loc_index(s, src_loc);
    if (src_idx < 0) return;
    int dst_idx = grad_loc_ensure(s, dst_loc);
    if (dst_idx < 0) return;
    s->xs[dst_idx] = s->xs[src_idx];
    s->modes[dst_idx] = s->modes[src_idx];
    s->group_ids[dst_idx] = s->group_ids[src_idx];
    s->bundles[dst_idx] = s->bundles[src_idx];
}

static Term grad_resolve_target_term(TinyHVM *ctx, Term t) {
    for (u32 depth = 0; depth < 32; depth++) {
        u8 tag = term_tag(t);
        if (tag == TAG_DP0 || tag == TAG_DP1) {
            t = heap_read(ctx, term_val(t));
            continue;
        }
        if (tag == TAG_VAR) {
            Term sub = heap_read(ctx, term_val(t));
            if (term_is_sub(sub)) break;
            t = sub;
            continue;
        }
        if (tag == TAG_TOP && term_ext(t) == UOP_DETACH) {
            Term det = thvm_eval(ctx, t);
            if (det == t) break;
            t = det;
            continue;
        }
        break;
    }
    return t;
}

int thvm_grad_targets_find_term_at(TinyHVM *ctx, u64 grad_loc, Term t, u32 *out_index, Term *out_slot) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s) return 0;
    GradTargetGroup *g = NULL;
    if (!grad_loc_group(s, grad_loc, &g)) return 0;
    Term rt = grad_resolve_target_term(ctx, t);
    for (u32 i = 0; i < g->n; i++) {
        GradTargetEntry *e = &s->entries[g->start + i];
        Term pt = grad_resolve_target_term(ctx, e->term);
        if (pt != rt) continue;
        if (out_index) *out_index = i;
        if (out_slot) *out_slot = e->slot;
        return 1;
    }
    return 0;
}

int thvm_grad_targets_find_index_at(TinyHVM *ctx, u64 grad_loc, u32 tid, u32 *out_index, Term *out_slot) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s) return 0;
    GradTargetGroup *g = NULL;
    if (!grad_loc_group(s, grad_loc, &g)) return 0;
    for (u32 i = 0; i < g->n; i++) {
        GradTargetEntry *e = &s->entries[g->start + i];
        if (e->tid != tid) continue;
        if (out_index) *out_index = i;
        if (out_slot) *out_slot = e->slot;
        return 1;
    }
    return 0;
}

int thvm_grad_targets_find_slot_at(TinyHVM *ctx, u64 grad_loc, u32 tid, Term *out_slot) {
    return thvm_grad_targets_find_index_at(ctx, grad_loc, tid, NULL, out_slot);
}

u32 thvm_grad_targets_count_at(TinyHVM *ctx, u64 grad_loc) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    GradTargetGroup *g = NULL;
    return grad_loc_group(s, grad_loc, &g) ? g->n : 0;
}

u32 thvm_grad_targets_get_tid_at(TinyHVM *ctx, u64 grad_loc, u32 index) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    GradTargetGroup *g = NULL;
    if (!grad_loc_group(s, grad_loc, &g) || index >= g->n) return ~0u;
    return s->entries[g->start + index].tid;
}

Term thvm_grad_targets_get_term_at(TinyHVM *ctx, u64 grad_loc, u32 index) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    GradTargetGroup *g = NULL;
    if (!grad_loc_group(s, grad_loc, &g) || index >= g->n) return term_era();
    return s->entries[g->start + index].term;
}

Term thvm_grad_targets_get_slot_at(TinyHVM *ctx, u64 grad_loc, u32 index) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    GradTargetGroup *g = NULL;
    if (!grad_loc_group(s, grad_loc, &g) || index >= g->n) return term_era();
    return s->entries[g->start + index].slot;
}

u32 thvm_grad_mode_get(TinyHVM *ctx, u64 grad_loc) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s) return GRAD_MODE_DROP;
    int idx = grad_loc_index(s, grad_loc);
    if (idx >= 0) return s->modes[idx];
    return GRAD_MODE_DROP;
}

void thvm_grad_keep_bundle_set(TinyHVM *ctx, u64 grad_loc, Term bundle) {
    GradTargetSet *s = grad_targets_get(ctx, 1);
    if (!s) return;
    int idx = grad_loc_ensure(s, grad_loc);
    if (idx < 0) return;
    s->bundles[idx] = bundle;
}

Term thvm_grad_keep_bundle_get(TinyHVM *ctx, u64 grad_loc) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s) return term_era();
    int idx = grad_loc_index(s, grad_loc);
    if (idx < 0) return term_era();
    return s->bundles[idx];
}

void thvm_grad_bundle_accum(TinyHVM *ctx, u64 grad_loc, u32 index, Term grad) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s || index >= thvm_grad_targets_count_at(ctx, grad_loc)) return;
    Term bundle = thvm_grad_keep_bundle_get(ctx, grad_loc);
    if (term_tag(bundle) != TAG_CTR) return;
    u64 loc = term_val(bundle);
    if (loc == 0 || loc + index >= ctx->heap_pos) return;
    Term prev = heap_read(ctx, loc + index);
    if (getenv("THVM_LOOP_DIAG")) {
        fprintf(stderr,
                "BUNDLE_ACCUM loc=%llu idx=%u slot=%llu prev_tag=%u prev_ext=%u grad_tag=%u grad_ext=%u grad_val=%llu\n",
                (unsigned long long)grad_loc,
                index,
                (unsigned long long)(loc + index),
                (u32)term_tag(prev),
                (u32)term_ext(prev),
                (u32)term_tag(grad),
                (u32)term_ext(grad),
                (unsigned long long)term_val(grad));
    }
    // Preserve a standalone copy in the bundle. Shared DP/TOP nodes from the
    // in-flight backward net can collapse when other consumers reduce; cloning
    // decouples each accumulated slot from that shared reduction state.
    Term kept_grad = term_clone(ctx, grad);
    if (term_tag(prev) == TAG_NUM && term_as_f32(prev) == 0.0f) {
        heap_set(ctx, loc + index, kept_grad);
        return;
    }
    if (term_tag(kept_grad) == TAG_NUM && term_as_f32(kept_grad) == 0.0f) return;
    heap_set(ctx, loc + index, thvm_op_raw(ctx, UOP_ADD, prev, kept_grad));
}

Term thvm_grad(TinyHVM *ctx, Term y, Term x) {
    // Single-target mode: no global target table.
    thvm_grad_targets_clear(ctx);
    term_use_clear();
    Term seed = thvm_grad_seed_like(ctx, y);
    u64 loc = heap_alloc(ctx, 2);
    y = linear_use(ctx, y, loc);
    heap_set(ctx, loc, y);
    heap_set(ctx, loc + 1, seed);
    thvm_grad_target_set(ctx, loc, x);
    thvm_grad_mode_set(ctx, loc, GRAD_MODE_DROP);
    return term_new(TAG_TOP, UOP_GRAD, loc);
}

// Multi-target GRAD: a single GRAD with x=ANY pattern.
// Param->slot mapping is kept in an internal target table and consumed
// when GRAD reaches TAG_TEN leaves.
Term thvm_grad_multi(TinyHVM *ctx, Term loss, Term *params, Term *grad_slots, u32 n_params) {
    thvm_grad_targets_clear(ctx);
    term_use_clear();
    Term seed = thvm_grad_seed_like(ctx, loss);
    u64 loc = heap_alloc(ctx, 2);
    loss = linear_use(ctx, loss, loc);
    heap_set(ctx, loc, loss);
    heap_set(ctx, loc + 1, seed);
    thvm_grad_target_set(ctx, loc, thvm_any());
    thvm_grad_mode_set(ctx, loc, grad_slots ? GRAD_MODE_SLOT : GRAD_MODE_DROP);
    thvm_grad_targets_set_for_loc(ctx, loc, params, grad_slots, n_params);
    return term_new(TAG_TOP, UOP_GRAD, loc);
}

Term thvm_grad_keep(TinyHVM *ctx, Term y, Term x) {
    return thvm_grad_multi_keep(ctx, y, &x, 1);
}

Term thvm_grad_multi_keep(TinyHVM *ctx, Term loss, Term *params, u32 n_params) {
    thvm_grad_targets_clear(ctx);
    term_use_clear();
    Term seed = thvm_grad_seed_like(ctx, loss);
    u64 grad_loc = heap_alloc(ctx, 2);
    loss = linear_use(ctx, loss, grad_loc);
    heap_set(ctx, grad_loc + 0, loss);
    heap_set(ctx, grad_loc + 1, seed);
    Term driver = term_new(TAG_TOP, UOP_GRAD, grad_loc);

    Term bundle = thvm_grad_bundle_new(ctx, n_params);
    Term keep_root = thvm_app(ctx, driver, bundle);

    thvm_grad_target_set(ctx, grad_loc, thvm_any());
    thvm_grad_mode_set(ctx, grad_loc, GRAD_MODE_KEEP);
    thvm_grad_targets_set_for_loc(ctx, grad_loc, params, NULL, n_params);
    thvm_grad_keep_bundle_set(ctx, grad_loc, bundle);
    return keep_root;
}

u32 thvm_grad_bundle_count(TinyHVM *ctx, Term bundle) {
    bundle = thvm_grad_bundle_whnf(ctx, bundle);
    if (term_tag(bundle) == TAG_CTR) return (u32)term_ext(bundle);
    if (term_tag(bundle) == TAG_ERA && term_val(bundle) == 0) return 0;
    return 1;
}

Term thvm_grad_bundle_get(TinyHVM *ctx, Term bundle, u32 index) {
    bundle = thvm_grad_bundle_whnf(ctx, bundle);
    if (term_tag(bundle) != TAG_CTR) return index == 0 ? bundle : term_era();
    if (index >= (u32)term_ext(bundle)) return term_era();
    u64 loc = term_val(bundle);
    if (loc == 0 || loc + index >= ctx->heap_pos) return term_era();
    return thvm_eval(ctx, heap_read(ctx, loc + index));
}
