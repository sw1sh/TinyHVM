// grad/_.c — Autograd: thvm_grad, thvm_grad_multi
//
// Pure IC term constructors. No computation here — everything
// is driven by thvm_reduce through the UOP_GRAD interaction handler.

#define GRAD_TARGET_CTX_MAX 8
#define GRAD_TARGETS_MAX 512
#define GRAD_LOC_MAX 8192

typedef struct {
    TinyHVM *ctx;
    u32 n;
    u32 tids[GRAD_TARGETS_MAX];
    Term slots[GRAD_TARGETS_MAX];
    u8 keep_bundle;
    Term bundle;
    u64 bundle_loc;
    u32 n_loc;
    u64 locs[GRAD_LOC_MAX];
    Term xs[GRAD_LOC_MAX];
} GradTargetSet;

static GradTargetSet grad_target_sets[GRAD_TARGET_CTX_MAX];

static void grad_targets_reset(GradTargetSet *s) {
    if (!s) return;
    s->n = 0;
    s->n_loc = 0;
    s->keep_bundle = 0;
    s->bundle = term_era();
    s->bundle_loc = 0;
    for (u32 i = 0; i < GRAD_TARGETS_MAX; i++)
        s->slots[i] = term_era();
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
    return idx;
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

void thvm_grad_targets_set(TinyHVM *ctx, Term *params, Term *grad_slots, u32 n_params) {
    GradTargetSet *s = grad_targets_get(ctx, 1);
    if (!s) return;
    grad_targets_reset(s);
    for (u32 i = 0; i < n_params && s->n < GRAD_TARGETS_MAX; i++) {
        if (term_tag(params[i]) != TAG_TEN) continue;
        s->tids[s->n] = (u32)term_val(params[i]);
        s->slots[s->n] = grad_slots ? grad_slots[i] : term_era();
        s->n++;
    }
}

void thvm_grad_targets_set_keep(TinyHVM *ctx, Term *params, u32 n_params, Term bundle) {
    GradTargetSet *s = grad_targets_get(ctx, 1);
    if (!s) return;
    grad_targets_reset(s);
    s->keep_bundle = 1;
    s->bundle = bundle;
    s->bundle_loc = term_tag(bundle) == TAG_CTR ? term_val(bundle) : 0;
    for (u32 i = 0; i < n_params && s->n < GRAD_TARGETS_MAX; i++) {
        if (term_tag(params[i]) != TAG_TEN) continue;
        s->tids[s->n] = (u32)term_val(params[i]);
        s->slots[s->n] = term_era();
        s->n++;
    }
}

int thvm_grad_targets_find_index(TinyHVM *ctx, u32 tid, u32 *out_index, Term *out_slot) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s) return 0;
    for (u32 i = 0; i < s->n; i++) {
        if (s->tids[i] != tid) continue;
        if (out_index) *out_index = i;
        if (out_slot) *out_slot = s->slots[i];
        return 1;
    }
    return 0;
}

int thvm_grad_targets_find_slot(TinyHVM *ctx, u32 tid, Term *out_slot) {
    return thvm_grad_targets_find_index(ctx, tid, NULL, out_slot);
}

u32 thvm_grad_targets_count(TinyHVM *ctx) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    return s ? s->n : 0;
}

u32 thvm_grad_targets_get_tid(TinyHVM *ctx, u32 index) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s || index >= s->n) return ~0u;
    return s->tids[index];
}

int thvm_grad_targets_has_keep_bundle(TinyHVM *ctx) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    return s ? (int)s->keep_bundle : 0;
}

Term thvm_grad_targets_get_keep_bundle(TinyHVM *ctx) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    return (s && s->keep_bundle) ? s->bundle : term_era();
}

void thvm_grad_bundle_accum(TinyHVM *ctx, u32 index, Term grad) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s || !s->keep_bundle || index >= s->n) return;
    u64 loc = s->bundle_loc;
    if (loc == 0 || loc + index >= ctx->heap_pos) return;
    Term prev = heap_read(ctx, loc + index);
    if (term_tag(prev) == TAG_NUM && term_as_f32(prev) == 0.0f) {
        heap_set(ctx, loc + index, grad);
        return;
    }
    if (term_tag(grad) == TAG_NUM && term_as_f32(grad) == 0.0f) return;
    heap_set(ctx, loc + index, thvm_op_raw(ctx, UOP_ADD, prev, grad));
}

Term thvm_grad(TinyHVM *ctx, Term y, Term x) {
    // Single-target mode: no global target table.
    thvm_grad_targets_clear(ctx);
    term_use_clear();
    Term seed = term_num_f32(1.0f);
    u64 loc = heap_alloc(ctx, 2);
    y = linear_use(ctx, y, loc);
    heap_set(ctx, loc, y);
    heap_set(ctx, loc + 1, seed);
    thvm_grad_target_set(ctx, loc, x);
    return term_new(TAG_TOP, UOP_GRAD, loc);
}

// Multi-target GRAD: a single GRAD with x=ANY pattern.
// Param->slot mapping is kept in an internal target table and consumed
// when GRAD reaches TAG_TEN leaves.
Term thvm_grad_multi(TinyHVM *ctx, Term loss, Term *params, Term *grad_slots, u32 n_params) {
    thvm_grad_targets_set(ctx, params, grad_slots, n_params);
    term_use_clear();
    Term seed = term_num_f32(1.0f);
    u64 loc = heap_alloc(ctx, 2);
    loss = linear_use(ctx, loss, loc);
    heap_set(ctx, loc, loss);
    heap_set(ctx, loc + 1, seed);
    thvm_grad_target_set(ctx, loc, thvm_any());
    return term_new(TAG_TOP, UOP_GRAD, loc);
}

Term thvm_grad_keep(TinyHVM *ctx, Term y, Term x) {
    return thvm_grad_multi_keep(ctx, y, &x, 1);
}

Term thvm_grad_multi_keep(TinyHVM *ctx, Term loss, Term *params, u32 n_params) {
    Term bundle = thvm_grad_bundle_new(ctx, n_params);
    thvm_grad_targets_set_keep(ctx, params, n_params, bundle);
    term_use_clear();
    Term seed = term_num_f32(1.0f);
    u64 loc = heap_alloc(ctx, 2);
    loss = linear_use(ctx, loss, loc);
    heap_set(ctx, loc, loss);
    heap_set(ctx, loc + 1, seed);
    thvm_grad_target_set(ctx, loc, thvm_any());
    Term driver = term_new(TAG_TOP, UOP_GRAD, loc);
    return thvm_app(ctx, driver, bundle);
}

u32 thvm_grad_bundle_count(TinyHVM *ctx, Term bundle) {
    bundle = thvm_grad_bundle_whnf(ctx, bundle);
    if (term_tag(bundle) != TAG_CTR) return 0;
    return (u32)term_ext(bundle);
}

Term thvm_grad_bundle_get(TinyHVM *ctx, Term bundle, u32 index) {
    bundle = thvm_grad_bundle_whnf(ctx, bundle);
    if (term_tag(bundle) != TAG_CTR) return term_era();
    if (index >= (u32)term_ext(bundle)) return term_era();
    u64 loc = term_val(bundle);
    if (loc == 0 || loc + index >= ctx->heap_pos) return term_era();
    return heap_read(ctx, loc + index);
}
