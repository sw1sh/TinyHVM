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
    Term outs[GRAD_TARGETS_MAX];
    u64 out_locs[GRAD_TARGETS_MAX];
    u32 n_detached_roots;
    Term detached_roots[GRAD_TARGETS_MAX];
    u64 detached_root_locs[GRAD_TARGETS_MAX];
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
            for (u32 j = 0; j < GRAD_TARGETS_MAX; j++) {
                grad_target_sets[i].outs[j] = term_era();
                grad_target_sets[i].out_locs[j] = 0;
                grad_target_sets[i].detached_roots[j] = term_era();
                grad_target_sets[i].detached_root_locs[j] = 0;
            }
            grad_target_sets[i].n_detached_roots = 0;
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

void thvm_grad_targets_clear(TinyHVM *ctx) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s) return;
    s->n = 0;
    s->n_loc = 0;
    s->n_detached_roots = 0;
    for (u32 i = 0; i < GRAD_TARGETS_MAX; i++) {
        s->outs[i] = term_era();
        s->out_locs[i] = 0;
        s->detached_roots[i] = term_era();
        s->detached_root_locs[i] = 0;
    }
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
    s->n = 0;
    s->n_loc = 0;
    s->n_detached_roots = 0;
    for (u32 i = 0; i < GRAD_TARGETS_MAX; i++) {
        s->outs[i] = term_era();
        s->out_locs[i] = 0;
        s->detached_roots[i] = term_era();
        s->detached_root_locs[i] = 0;
    }
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

void thvm_grad_output_add(TinyHVM *ctx, u32 tid, Term grad) {
    GradTargetSet *s = grad_targets_get(ctx, 1);
    if (!s) return;
    for (u32 i = 0; i < s->n; i++) {
        if (s->tids[i] != tid) continue;
        Term prev = s->outs[i];
        if (term_tag(prev) == TAG_ERA && term_val(prev) == 0) {
            s->outs[i] = grad;
        } else if (term_tag(grad) == TAG_NUM && term_as_f32(grad) == 0.0f) {
            // Keep the previous nonzero-visible branch as the detached output root.
        } else if (term_tag(prev) == TAG_NUM && term_as_f32(prev) == 0.0f) {
            s->outs[i] = grad;
        } else {
            s->outs[i] = thvm_op_raw(ctx, UOP_ADD, prev, grad);
        }
        if (s->out_locs[i] == 0)
            s->out_locs[i] = heap_alloc(ctx, 1);
        heap_set(ctx, s->out_locs[i], s->outs[i]);
        return;
    }
}

u32 thvm_grad_output_count(TinyHVM *ctx) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s) return 0;
    u32 n = 0;
    for (u32 i = 0; i < s->n; i++) {
        if (!(term_tag(s->outs[i]) == TAG_ERA && term_val(s->outs[i]) == 0))
            n++;
    }
    return n;
}

Term thvm_grad_output_get(TinyHVM *ctx, u32 index) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s) return term_era();
    u32 seen = 0;
    for (u32 i = 0; i < s->n; i++) {
        Term out = s->outs[i];
        if (term_tag(out) == TAG_ERA && term_val(out) == 0) continue;
        if (seen++ == index) return out;
    }
    return term_era();
}

void thvm_grad_detached_root_add(TinyHVM *ctx, Term root) {
    GradTargetSet *s = grad_targets_get(ctx, 1);
    if (!s || s->n_detached_roots >= GRAD_TARGETS_MAX) return;
    u32 i = s->n_detached_roots++;
    s->detached_roots[i] = root;
    if (s->detached_root_locs[i] == 0)
        s->detached_root_locs[i] = heap_alloc(ctx, 1);
    heap_set(ctx, s->detached_root_locs[i], root);
}

u32 thvm_grad_detached_root_count(TinyHVM *ctx) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    return s ? s->n_detached_roots : 0;
}

Term thvm_grad_detached_root_get(TinyHVM *ctx, u32 index) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s || index >= s->n_detached_roots) return term_era();
    u64 loc = s->detached_root_locs[index];
    if (loc > 0 && loc < ctx->heap_pos) return heap_read(ctx, loc);
    return s->detached_roots[index];
}

u64 thvm_grad_detached_root_loc_get(TinyHVM *ctx, u32 index) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s || index >= s->n_detached_roots) return 0;
    return s->detached_root_locs[index];
}

Term thvm_grad(TinyHVM *ctx, Term y, Term x) {
    // Single-target mode: no global target table.
    thvm_grad_targets_clear(ctx);
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
    if (grad_slots == NULL && getenv("THVM_STEP_GRAPH")) {
        thvm_grad_targets_clear(ctx);
        term_use_clear();
        Term seed = term_num_f32(1.0f);
        Term first = term_era();
        u32 n_valid = 0;
        for (u32 i = 0; i < n_params; i++) {
            if (term_tag(params[i]) != TAG_TEN) continue;
            u64 loc = heap_alloc(ctx, 2);
            Term loss_i = linear_use(ctx, loss, loc);
            heap_set(ctx, loc, loss_i);
            heap_set(ctx, loc + 1, seed);
            thvm_grad_target_set(ctx, loc, params[i]);
            Term root = term_new(TAG_TOP, UOP_GRAD, loc);
            if (n_valid++ == 0) first = root;
            else thvm_grad_detached_root_add(ctx, root);
        }
        return first;
    }

    thvm_grad_targets_set(ctx, params, grad_slots, n_params);
    Term seed = term_num_f32(1.0f);
    u64 loc = heap_alloc(ctx, 2);
    loss = linear_use(ctx, loss, loc);
    heap_set(ctx, loc, loss);
    heap_set(ctx, loc + 1, seed);
    thvm_grad_target_set(ctx, loc, thvm_any());
    return term_new(TAG_TOP, UOP_GRAD, loc);
}
