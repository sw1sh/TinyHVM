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
    u32 n_loc;
    u64 locs[GRAD_LOC_MAX];
    Term xs[GRAD_LOC_MAX];
} GradTargetSet;

static GradTargetSet grad_target_sets[GRAD_TARGET_CTX_MAX];

static GradTargetSet *grad_targets_get(TinyHVM *ctx, int create) {
    for (u32 i = 0; i < GRAD_TARGET_CTX_MAX; i++) {
        if (grad_target_sets[i].ctx == ctx) return &grad_target_sets[i];
    }
    if (!create) return NULL;
    for (u32 i = 0; i < GRAD_TARGET_CTX_MAX; i++) {
        if (grad_target_sets[i].ctx == NULL) {
            grad_target_sets[i].ctx = ctx;
            grad_target_sets[i].n = 0;
            return &grad_target_sets[i];
        }
    }
    return NULL;
}

void thvm_grad_targets_clear(TinyHVM *ctx) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s) return;
    s->n = 0;
    s->n_loc = 0;
}

void thvm_grad_target_set(TinyHVM *ctx, u64 grad_loc, Term x) {
    GradTargetSet *s = grad_targets_get(ctx, 1);
    if (!s) return;
    for (u32 i = 0; i < s->n_loc; i++) {
        if (s->locs[i] == grad_loc) {
            s->xs[i] = x;
            return;
        }
    }
    if (s->n_loc >= GRAD_LOC_MAX) return;
    s->locs[s->n_loc] = grad_loc;
    s->xs[s->n_loc] = x;
    s->n_loc++;
}

Term thvm_grad_target_get(TinyHVM *ctx, u64 grad_loc) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s) return thvm_any();
    for (u32 i = 0; i < s->n_loc; i++) {
        if (s->locs[i] == grad_loc) return s->xs[i];
    }
    return thvm_any();
}

void thvm_grad_targets_set(TinyHVM *ctx, Term *params, Term *grad_slots, u32 n_params) {
    GradTargetSet *s = grad_targets_get(ctx, 1);
    if (!s) return;
    s->n = 0;
    for (u32 i = 0; i < n_params && s->n < GRAD_TARGETS_MAX; i++) {
        if (term_tag(params[i]) != TAG_TEN) continue;
        s->tids[s->n] = (u32)term_val(params[i]);
        s->slots[s->n] = grad_slots ? grad_slots[i] : term_era();
        s->n++;
    }
}

int thvm_grad_targets_find_slot(TinyHVM *ctx, u32 tid, Term *out_slot) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s) return 0;
    for (u32 i = 0; i < s->n; i++) {
        if (s->tids[i] == tid) {
            if (out_slot) *out_slot = s->slots[i];
            return 1;
        }
    }
    return 0;
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

Term thvm_grad(TinyHVM *ctx, Term y, Term x) {
    // Single-target mode: no global target table.
    thvm_grad_targets_clear(ctx);
    // GRAD seed is a scalar tensor to keep UOP inputs tensor-typed in the heap graph.
    f32 one = 1.0f;
    Term seed = thvm_tensor(ctx, &one, SHAPE(1));

    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     y);
    heap_set(ctx, loc + 1, seed);
    thvm_grad_target_set(ctx, loc, x);
    return term_new(TAG_TOP, UOP_GRAD, loc);
}

// Multi-target GRAD: a single GRAD with x=ANY pattern.
// Param->slot mapping is kept in an internal target table and consumed
// when GRAD reaches TAG_TEN leaves.
Term thvm_grad_multi(TinyHVM *ctx, Term loss, Term *params, Term *grad_slots, u32 n_params) {
    thvm_grad_targets_set(ctx, params, grad_slots, n_params);
    // Same scalar tensor seed rule for multi-target.
    f32 one = 1.0f;
    Term seed = thvm_tensor(ctx, &one, SHAPE(1));
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc, loss);
    heap_set(ctx, loc + 1, seed);
    thvm_grad_target_set(ctx, loc, thvm_any());
    return term_new(TAG_TOP, UOP_GRAD, loc);
}
