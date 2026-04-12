// grad/_.c — Autograd: thvm_grad, thvm_grad_multi
//
// Pure IC term constructors. No computation here — everything
// is driven by thvm_reduce through the UOP_GRAD interaction handler.

#define GRAD_TARGET_CTX_MAX 8
#define GRAD_TARGETS_MAX 512
#define GRAD_LOC_MAX 8192

typedef struct {
    TinyHVM *ctx;
    u8 default_mode;
    u32 n;
    u32 tids[GRAD_TARGETS_MAX];
    Term slots[GRAD_TARGETS_MAX];
    u8 modes[GRAD_LOC_MAX];
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
    s->default_mode = GRAD_MODE_DROP;
    s->keep_bundle = 0;
    s->bundle = term_era();
    s->bundle_loc = 0;
    for (u32 i = 0; i < GRAD_TARGETS_MAX; i++)
        s->slots[i] = term_era();
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
    s->modes[idx] = s->default_mode;
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
    s->default_mode = grad_slots ? GRAD_MODE_SLOT : GRAD_MODE_DROP;
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
    s->default_mode = GRAD_MODE_KEEP;
    s->keep_bundle = 1;
    s->bundle = bundle;
    s->bundle_loc = term_tag(bundle) == TAG_CTR ? (term_val(bundle) + (term_ext(bundle) > n_params ? 1 : 0)) : 0;
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

u32 thvm_grad_mode_get(TinyHVM *ctx, u64 grad_loc) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s) return GRAD_MODE_DROP;
    int idx = grad_loc_index(s, grad_loc);
    if (idx >= 0) return s->modes[idx];
    return s->default_mode;
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
    Term seed = thvm_grad_seed_like(ctx, y);
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
    Term seed = thvm_grad_seed_like(ctx, loss);
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
    term_use_clear();
    Term seed = thvm_grad_seed_like(ctx, loss);
    u64 grad_loc = heap_alloc(ctx, 2);
    loss = linear_use(ctx, loss, grad_loc);
    heap_set(ctx, grad_loc + 0, loss);
    heap_set(ctx, grad_loc + 1, seed);
    Term driver = term_new(TAG_TOP, UOP_GRAD, grad_loc);

    assert(n_params + 1 < 256 && "keep bundle arity exceeds TAG_CTR ext range");
    u64 bundle_loc = heap_alloc(ctx, n_params + 1);
    heap_set(ctx, bundle_loc + 0, driver);
    for (u32 i = 0; i < n_params; i++)
        heap_set(ctx, bundle_loc + 1 + i, term_num_f32(0.0f));
    Term bundle = term_new(TAG_CTR, (u8)(n_params + 1), bundle_loc);

    thvm_grad_targets_set_keep(ctx, params, n_params, bundle);
    thvm_grad_target_set(ctx, grad_loc, thvm_any());
    return bundle;
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
    return heap_read(ctx, loc + index);
}
