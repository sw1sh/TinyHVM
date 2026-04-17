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
    u64 keep_app_locs[GRAD_LOC_MAX];
    u32 n_loc;
    u64 locs[GRAD_LOC_MAX];
    Term xs[GRAD_LOC_MAX];
} GradTargetSet;

static GradTargetSet grad_target_sets[GRAD_TARGET_CTX_MAX];
static GradTargetSet grad_book_target_sets[GRAD_TARGET_CTX_MAX];

static Term grad_resolve_target_term(TinyHVM *ctx, Term t);

static void grad_targets_reset(GradTargetSet *s) {
    if (!s) return;
    s->n_group = 0;
    s->n_entry = 0;
    s->n_loc = 0;
    for (u32 i = 0; i < GRAD_LOC_MAX; i++) {
        s->modes[i] = GRAD_MODE_DROP;
        s->group_ids[i] = 0;
        s->bundles[i] = term_era();
        s->keep_app_locs[i] = 0;
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

static GradTargetSet *grad_book_targets_get(TinyHVM *ctx, int create) {
    for (u32 i = 0; i < GRAD_TARGET_CTX_MAX; i++) {
        if (grad_book_target_sets[i].ctx == ctx) return &grad_book_target_sets[i];
    }
    if (!create) return NULL;
    for (u32 i = 0; i < GRAD_TARGET_CTX_MAX; i++) {
        if (grad_book_target_sets[i].ctx == NULL) {
            grad_book_target_sets[i].ctx = ctx;
            grad_targets_reset(&grad_book_target_sets[i]);
            return &grad_book_target_sets[i];
        }
    }
    return NULL;
}

static GradTargetSet *grad_targets_domain_get(TinyHVM *ctx, u64 grad_loc, int create) {
    return thvm_grad_is_book_loc(grad_loc)
        ? grad_book_targets_get(ctx, create)
        : grad_targets_get(ctx, create);
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
    s->keep_app_locs[idx] = 0;
    return idx;
}

static u32 grad_group_alloc_entries(GradTargetSet *s, GradTargetEntry *entries, u32 n_entries) {
    if (!s || n_entries == 0) return 0;
    assert(n_entries <= THVM_GRAD_TARGETS_MAX);
    if (s->n_group >= GRAD_GROUP_MAX) return 0;
    if (s->n_entry + n_entries > GRAD_GROUP_ENTRIES_MAX) return 0;
    u32 gid = ++s->n_group; // 1-based
    GradTargetGroup *g = &s->groups[gid - 1];
    g->start = s->n_entry;
    g->n = n_entries;
    for (u32 i = 0; i < n_entries; i++) {
        s->entries[s->n_entry++] = entries[i];
    }
    return gid;
}

static u32 grad_group_alloc(TinyHVM *ctx, GradTargetSet *s, u64 grad_loc, Term *params, Term *slots, u32 n_params) {
    if (!s || n_params == 0) return 0;
    assert(n_params <= THVM_GRAD_TARGETS_MAX);
    GradTargetEntry entries[THVM_GRAD_TARGETS_MAX];
    for (u32 i = 0; i < n_params; i++) {
        Term p = params[i];
        u32 tid = term_tag(p) == TAG_TEN ? (u32)term_val(p) : ~0u;
        if (tid == ~0u && !thvm_grad_is_book_loc(grad_loc)) {
            Term rp = grad_resolve_target_term(ctx, p);
            if (term_tag(rp) == TAG_TEN) {
                tid = (u32)term_val(rp);
            }
        }
        entries[i] = (GradTargetEntry){
            .term = p,
            .tid  = tid,
            .slot = slots ? slots[i] : term_era(),
        };
    }
    return grad_group_alloc_entries(s, entries, n_params);
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
    if (getenv("THVM_LOOP_DIAG")) {
        fprintf(stderr, "BUNDLE_WHNF in tag=%u ext=%u val=%llu\n",
                (u32)term_tag(bundle), (u32)term_ext(bundle),
                (unsigned long long)term_val(bundle));
    }
    // thvm_grad_keep/thvm_grad_multi_keep are often eagerly evaluated by callers.
    // The keep driver then collapses to ERA after depositing into side bundles.
    // Recover the single active keep-bundle from the target table in that case.
    if (term_tag(bundle) == TAG_ERA && term_val(bundle) == 0) {
        GradTargetSet *s = grad_targets_get(ctx, 0);
        if (s) {
            Term only = term_era();
            u32 best_arity = 0;
            for (u32 i = 0; i < s->n_loc; i++) {
                Term b = s->bundles[i];
                if (term_tag(b) != TAG_CTR) continue;
                u32 arity = (u32)term_ext(b);
                if (term_tag(only) != TAG_CTR) {
                    only = b;
                    best_arity = arity;
                    continue;
                }
                // Shared target propagation can replicate identical terms.
                if (only == b) continue;
                // Prefer non-empty bundles if both are present.
                if (arity > best_arity) {
                    only = b;
                    best_arity = arity;
                    continue;
                }
                if (arity < best_arity) continue;
                // Two distinct bundles with the same arity: ambiguous.
                only = term_era();
                break;
            }
            if (term_tag(only) == TAG_CTR) return only;
        }
    }
    if (term_tag(bundle) == TAG_APP) {
        u64 app_loc = term_val(bundle);
        if (app_loc + 1 < ctx->heap_pos) {
            return heap_read(ctx, app_loc + 1);
        }
    }
    // Keep bundles are fixed-arity carriers when arity > 1. Do not reduce
    // TAG_CTR directly: generic CTR reduction collapses arity-1 containers to
    // their head term, which would destroy multi-keep bundle structure.
    if (term_tag(bundle) == TAG_CTR) return bundle;
    Term out = thvm_reduce(ctx, bundle);
    if (getenv("THVM_LOOP_DIAG")) {
        fprintf(stderr, "BUNDLE_WHNF out tag=%u ext=%u val=%llu\n",
                (u32)term_tag(out), (u32)term_ext(out),
                (unsigned long long)term_val(out));
    }
    return out;
}

static Term thvm_grad_force_slot_expr(TinyHVM *ctx, Term t, u32 depth) {
    if (depth > 16) return t;
    if (term_tag(t) == TAG_TEN || term_tag(t) == TAG_NUM || term_tag(t) == TAG_ERA) return t;
    if (term_tag(t) != TAG_TOP) return t;

    u32 uop = term_ext(t);
    u64 loc = term_val(t);
    if (loc >= ctx->heap_pos) return t;

    // Recursively materialize common elementwise slot expressions.
    if (uop == UOP_ADD || uop == UOP_SUB || uop == UOP_MUL || uop == UOP_DIV || uop == UOP_MAX) {
        Term a = thvm_grad_force_slot_expr(ctx, heap_read(ctx, loc + 0), depth + 1);
        Term b = thvm_grad_force_slot_expr(ctx, heap_read(ctx, loc + 1), depth + 1);
        if (term_tag(a) == TAG_TEN && term_tag(b) == TAG_TEN) {
            return thvm_force_tensor_term(ctx, thvm_op(ctx, uop, a, b));
        }
    }
    if (uop == UOP_NEG || uop == UOP_EXP || uop == UOP_LOG || uop == UOP_RELU || uop == UOP_SQRT) {
        Term a = thvm_grad_force_slot_expr(ctx, heap_read(ctx, loc + 0), depth + 1);
        if (term_tag(a) == TAG_TEN) {
            return thvm_force_tensor_term(ctx, thvm_op(ctx, uop, a, term_era()));
        }
    }
    return t;
}

static Term thvm_grad_slot_resolve(TinyHVM *ctx, Term t) {
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
        break;
    }
    return t;
}

static void thvm_grad_slot_copy_into(TinyHVM *ctx, Term dst_t, Term src_t) {
    if (term_tag(dst_t) != TAG_TEN || term_tag(src_t) != TAG_TEN) return;

    u32 dst_id = (u32)term_val(dst_t);
    u32 src_id = (u32)term_val(src_t);
    if (dst_id >= ctx->tensor_count || src_id >= ctx->tensor_count) return;
    if (dst_id == src_id) return;

    TensorMeta *md = &ctx->tensors[dst_id];
    TensorMeta *ms = &ctx->tensors[src_id];
    ENSURE(ctx, src_id);
    ms = &ctx->tensors[src_id];
    md = &ctx->tensors[dst_id];
    if (ms->buf_id == 0 || !md->backend) return;

    if (md->backend->buf_copy && ms->view.contiguous &&
        ms->view.numel == md->view.numel &&
        ms->dtype == md->dtype) {
        md->backend->buf_copy(md->buf_id, ms->buf_id,
            (u64)md->view.numel * dtype_size(md->dtype));
    } else if (md->backend->contiguify && ms->dtype == md->dtype) {
        md->backend->contiguify(md->buf_id, md->view.numel,
                                 ms->buf_id, &ms->view);
    } else {
        u32 src_dtype = ms->dtype;
        void *src_host = thvm_to_host_raw(ctx, src_t, &src_dtype, NULL);
        u64 n = md->view.numel;
        u64 nbytes = n * dtype_size(md->dtype);
        if (!src_host) return;
        if (src_dtype == md->dtype) {
            md->backend->buf_write(md->buf_id, src_host, nbytes);
        } else {
            void *dst_host = malloc((size_t)nbytes);
            for (u32 i = 0; i < n; i++)
                dtype_store_from_f32(dst_host, md->dtype, i,
                                     dtype_load_as_f32(src_host, src_dtype, i));
            md->backend->buf_write(md->buf_id, dst_host, nbytes);
            free(dst_host);
        }
    }

    if (md->host_ptr) {
        free(md->host_ptr);
        md->host_ptr = NULL;
    }
    extern u32 buf_epoch[];
    if (md->buf_id < MAX_BUF_EPOCHS)
        buf_epoch[md->buf_id]++;
}

static void thvm_grad_slot_accum(TinyHVM *ctx, Term slot, Term grad) {
    slot = thvm_grad_slot_resolve(ctx, slot);
    grad = thvm_grad_slot_resolve(ctx, grad);
    if (term_tag(slot) != TAG_TEN) return;
    if (term_tag(grad) == TAG_NUM && term_as_f32(grad) == 0.0f) return;
    if (term_tag(grad) == TAG_ERA && term_val(grad) == 0) return;

    // Keep the visible backward graph mathematical: slot writes happen here,
    // not as APP(ASSIGN(...), ERA) terms inside GRAD/MUL and friends.
    u8 saved_trace_enabled = ctx->trace_enabled;
    ctx->trace_enabled = 0;
    Term accum = thvm_eval(ctx, thvm_op_raw(ctx, UOP_ADD, slot, grad));
    ctx->trace_enabled = saved_trace_enabled;
    if (term_tag(accum) != TAG_TEN) return;
    thvm_grad_slot_copy_into(ctx, slot, accum);
}

void thvm_grad_targets_clear(TinyHVM *ctx) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (s) grad_targets_reset(s);
    GradTargetSet *bs = grad_book_targets_get(ctx, 0);
    if (bs) grad_targets_reset(bs);
}

void thvm_grad_target_set(TinyHVM *ctx, u64 grad_loc, Term x) {
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 1);
    if (!s) return;
    int idx = grad_loc_ensure(s, grad_loc);
    if (idx < 0) return;
    s->xs[idx] = x;
}

Term thvm_grad_target_get(TinyHVM *ctx, u64 grad_loc) {
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 0);
    if (!s) return thvm_any();
    int idx = grad_loc_index(s, grad_loc);
    if (idx >= 0) return s->xs[idx];
    return thvm_any();
}

void thvm_grad_mode_set(TinyHVM *ctx, u64 grad_loc, u32 mode) {
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 1);
    if (!s) return;
    int idx = grad_loc_ensure(s, grad_loc);
    if (idx < 0) return;
    s->modes[idx] = (u8)mode;
}

void thvm_grad_targets_set_for_loc(TinyHVM *ctx, u64 grad_loc, Term *params, Term *grad_slots, u32 n_params) {
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 1);
    if (!s) return;
    int idx = grad_loc_ensure(s, grad_loc);
    if (idx < 0) return;
    s->group_ids[idx] = grad_group_alloc(ctx, s, grad_loc, params, grad_slots, n_params);
}

void thvm_grad_targets_share(TinyHVM *ctx, u64 dst_loc, u64 src_loc) {
    GradTargetSet *src_set = grad_targets_domain_get(ctx, src_loc, 0);
    GradTargetSet *dst_set = grad_targets_domain_get(ctx, dst_loc, 1);
    if (!src_set || !dst_set) return;
    int src_idx = grad_loc_index(src_set, src_loc);
    if (src_idx < 0) return;
    int dst_idx = grad_loc_ensure(dst_set, dst_loc);
    if (dst_idx < 0) return;
    dst_set->xs[dst_idx] = src_set->xs[src_idx];
    dst_set->modes[dst_idx] = src_set->modes[src_idx];
    dst_set->bundles[dst_idx] = src_set->bundles[src_idx];
    dst_set->keep_app_locs[dst_idx] = src_set->keep_app_locs[src_idx];
    GradTargetGroup *g = NULL;
    if (grad_loc_group(src_set, src_loc, &g) && g && g->n > 0) {
        GradTargetEntry entries[THVM_GRAD_TARGETS_MAX];
        assert(g->n <= THVM_GRAD_TARGETS_MAX);
        for (u32 i = 0; i < g->n; i++) {
            entries[i] = src_set->entries[g->start + i];
        }
        dst_set->group_ids[dst_idx] = grad_group_alloc_entries(dst_set, entries, g->n);
    } else {
        dst_set->group_ids[dst_idx] = 0;
    }
}

static Term grad_resolve_target_term(TinyHVM *ctx, Term t) {
    for (u32 depth = 0; depth < 32; depth++) {
        u8 tag = term_tag(t);
        if (tag == TAG_ALO) {
            Term forced = thvm_alo_force(ctx, t);
            if (forced == t) break;
            t = forced;
            continue;
        }
        if (tag == TAG_TOP && term_ext(t) == UOP_ASSIGN) {
            u64 loc = term_val(t);
            if (loc >= ctx->heap_pos) break;
            t = heap_read(ctx, loc + 0);
            continue;
        }
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
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 0);
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
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 0);
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
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 0);
    GradTargetGroup *g = NULL;
    return grad_loc_group(s, grad_loc, &g) ? g->n : 0;
}

u32 thvm_grad_targets_get_tid_at(TinyHVM *ctx, u64 grad_loc, u32 index) {
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 0);
    GradTargetGroup *g = NULL;
    if (!grad_loc_group(s, grad_loc, &g) || index >= g->n) return ~0u;
    return s->entries[g->start + index].tid;
}

Term thvm_grad_targets_get_term_at(TinyHVM *ctx, u64 grad_loc, u32 index) {
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 0);
    GradTargetGroup *g = NULL;
    if (!grad_loc_group(s, grad_loc, &g) || index >= g->n) return term_era();
    return s->entries[g->start + index].term;
}

Term thvm_grad_targets_get_slot_at(TinyHVM *ctx, u64 grad_loc, u32 index) {
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 0);
    GradTargetGroup *g = NULL;
    if (!grad_loc_group(s, grad_loc, &g) || index >= g->n) return term_era();
    return s->entries[g->start + index].slot;
}

u32 thvm_grad_mode_get(TinyHVM *ctx, u64 grad_loc) {
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 0);
    if (!s) return GRAD_MODE_DROP;
    int idx = grad_loc_index(s, grad_loc);
    if (idx >= 0) return s->modes[idx];
    return GRAD_MODE_DROP;
}

void thvm_grad_keep_bundle_set(TinyHVM *ctx, u64 grad_loc, Term bundle) {
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 1);
    if (!s) return;
    int idx = grad_loc_ensure(s, grad_loc);
    if (idx < 0) return;
    s->bundles[idx] = bundle;
}

Term thvm_grad_keep_bundle_get(TinyHVM *ctx, u64 grad_loc) {
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 0);
    if (!s) return term_era();
    int idx = grad_loc_index(s, grad_loc);
    if (idx < 0) return term_era();
    return s->bundles[idx];
}

void thvm_grad_keep_app_loc_set(TinyHVM *ctx, u64 grad_loc, u64 app_loc) {
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 1);
    if (!s) return;
    int idx = grad_loc_ensure(s, grad_loc);
    if (idx < 0) return;
    s->keep_app_locs[idx] = app_loc;
}

u64 thvm_grad_keep_app_loc_get(TinyHVM *ctx, u64 grad_loc) {
    GradTargetSet *s = grad_targets_domain_get(ctx, grad_loc, 0);
    if (!s) return 0;
    int idx = grad_loc_index(s, grad_loc);
    if (idx < 0) return 0;
    return s->keep_app_locs[idx];
}

void thvm_grad_bundle_accum(TinyHVM *ctx, u64 grad_loc, u32 index, Term grad) {
    GradTargetSet *s = grad_targets_get(ctx, 0);
    if (!s || index >= thvm_grad_targets_count_at(ctx, grad_loc)) {
        if (getenv("THVM_LOOP_DIAG")) {
            fprintf(stderr,
                    "BUNDLE_ACCUM_SKIP loc=%llu idx=%u reason=missing_set_or_bad_index have_set=%d count=%u\n",
                    (unsigned long long)grad_loc,
                    index,
                    s != NULL,
                    thvm_grad_targets_count_at(ctx, grad_loc));
        }
        return;
    }
    Term bundle = thvm_grad_keep_bundle_get(ctx, grad_loc);
    Term bundle_alias = bundle;
    u64 app_loc = thvm_grad_keep_app_loc_get(ctx, grad_loc);
    if (app_loc != 0 && thvm_grad_is_book_loc(app_loc) && term_tag(bundle_alias) == TAG_ALO) {
        u64 alo_loc = term_val(bundle_alias);
        if (alo_loc != 0 && alo_loc + 1 < ctx->heap_pos) {
            Term sid_term = heap_read(ctx, alo_loc + 1);
            if (term_tag(sid_term) == TAG_NUM) {
                u64 dyn_app_loc = 0;
                if (thvm_alo_lookup_node(ctx, term_as_u32(sid_term), thvm_grad_unkey_book_loc(app_loc), &dyn_app_loc))
                    app_loc = dyn_app_loc;
            }
        }
    }
    if (term_tag(bundle) == TAG_ALO) {
        bundle = thvm_alo_force(ctx, bundle);
    }
    int bundle_fallback_ok = 0;
    if (term_tag(bundle) == TAG_CTR) {
        u64 bundle_loc = term_val(bundle);
        bundle_fallback_ok = bundle_loc != 0 &&
                             index < (u32)term_ext(bundle) &&
                             bundle_loc + index < ctx->heap_pos;
    }
    u64 loc = 0;
    Term prev = term_era();
    if (app_loc != 0) {
        if (thvm_grad_is_book_loc(app_loc) && bundle_fallback_ok) {
            app_loc = 0;
        }
        if (app_loc + 1 >= ctx->heap_pos) {
            if (bundle_fallback_ok) {
                app_loc = 0;
            } else {
                if (getenv("THVM_LOOP_DIAG")) {
                    fprintf(stderr,
                            "BUNDLE_ACCUM_SKIP loc=%llu idx=%u reason=bad_keep_app keep_app=%llu heap_pos=%llu\n",
                            (unsigned long long)grad_loc,
                            index,
                            (unsigned long long)app_loc,
                            (unsigned long long)ctx->heap_pos);
                }
                return;
            }
        }
        if (app_loc != 0) {
            Term app_arg = heap_read(ctx, app_loc + 1);
            if (term_tag(app_arg) == TAG_ALO) {
                app_arg = thvm_alo_force(ctx, app_arg);
                heap_set(ctx, app_loc + 1, app_arg);
            }
            if (term_tag(app_arg) == TAG_CTR) {
                loc = term_val(app_arg);
                if (loc == 0 || loc + index >= ctx->heap_pos) {
                    if (bundle_fallback_ok) {
                        app_loc = 0;
                        loc = 0;
                    } else {
                        if (getenv("THVM_LOOP_DIAG")) {
                            fprintf(stderr,
                                    "BUNDLE_ACCUM_SKIP loc=%llu idx=%u reason=bad_app_ctr keep_app=%llu arg_tag=%u arg_ext=%u arg_val=%llu heap_pos=%llu\n",
                                    (unsigned long long)grad_loc,
                                    index,
                                    (unsigned long long)app_loc,
                                    (u32)term_tag(app_arg),
                                    (u32)term_ext(app_arg),
                                    (unsigned long long)term_val(app_arg),
                                    (unsigned long long)ctx->heap_pos);
                        }
                        return;
                    }
                }
                if (app_loc != 0)
                    prev = heap_read(ctx, loc + index);
            } else if (index == 0) {
                loc = app_loc;
                prev = app_arg;
            } else if (bundle_fallback_ok) {
                app_loc = 0;
                loc = 0;
            } else {
                if (getenv("THVM_LOOP_DIAG")) {
                    fprintf(stderr,
                            "BUNDLE_ACCUM_SKIP loc=%llu idx=%u reason=non_ctr_app_arg keep_app=%llu arg_tag=%u arg_ext=%u arg_val=%llu\n",
                            (unsigned long long)grad_loc,
                            index,
                            (unsigned long long)app_loc,
                            (u32)term_tag(app_arg),
                            (u32)term_ext(app_arg),
                            (unsigned long long)term_val(app_arg));
                }
                return;
            }
        }
    }
    if (loc == 0 && app_loc == 0 && term_tag(bundle) == TAG_CTR) {
        loc = term_val(bundle);
        if (loc == 0 || loc + index >= ctx->heap_pos) {
            if (getenv("THVM_LOOP_DIAG")) {
                fprintf(stderr,
                        "BUNDLE_ACCUM_SKIP loc=%llu idx=%u reason=bad_bundle_loc keep_tag=%u keep_ext=%u keep_val=%llu heap_pos=%llu\n",
                        (unsigned long long)grad_loc,
                        index,
                        (u32)term_tag(bundle),
                        (u32)term_ext(bundle),
                        (unsigned long long)term_val(bundle),
                        (unsigned long long)ctx->heap_pos);
            }
            return;
        }
        prev = heap_read(ctx, loc + index);
    }
    if (loc == 0) {
        if (getenv("THVM_LOOP_DIAG")) {
            fprintf(stderr,
                    "BUNDLE_ACCUM_SKIP loc=%llu idx=%u reason=non_bundle keep_tag=%u keep_ext=%u keep_val=%llu keep_app=%llu\n",
                    (unsigned long long)grad_loc,
                    index,
                    (u32)term_tag(bundle),
                    (u32)term_ext(bundle),
                    (unsigned long long)term_val(bundle),
                    (unsigned long long)app_loc);
        }
        return;
    }
    if (getenv("THVM_LOOP_DIAG")) {
        fprintf(stderr,
                "BUNDLE_ACCUM loc=%llu idx=%u slot=%llu prev_tag=%u prev_ext=%u grad_tag=%u grad_ext=%u grad_val=%llu\n",
                (unsigned long long)grad_loc,
                index,
                (unsigned long long)(app_loc != 0
                    ? (term_tag(heap_read(ctx, app_loc + 1)) == TAG_CTR ? (loc + index) : (loc + 1))
                    : (loc + index)),
                (u32)term_tag(prev),
                (u32)term_ext(prev),
                (u32)term_tag(grad),
                (u32)term_ext(grad),
                (unsigned long long)term_val(grad));
    }
    u64 dst_slot = app_loc != 0
        ? (term_tag(heap_read(ctx, app_loc + 1)) == TAG_CTR ? (loc + index) : (loc + 1))
        : (loc + index);
    if (term_tag(prev) == TAG_NUM && term_as_f32(prev) == 0.0f) {
        // Keep mode should preserve the backward branch in the bundle even if
        // the reducer continues cleaning up the original path after the
        // GRAD/TEN hit. Clone the branch into the bundle so later erasure on
        // the source path cannot mutate the kept result.
        Term stored = term_clone(ctx, grad);
        if (getenv("THVM_LOOP_DIAG")) {
            fprintf(stderr,
                    "BUNDLE_ACCUM_STORE loc=%llu idx=%u dst_slot=%llu stored_tag=%u stored_ext=%u stored_val=%llu src_tag=%u src_ext=%u src_val=%llu\n",
                    (unsigned long long)grad_loc,
                    index,
                    (unsigned long long)dst_slot,
                    (u32)term_tag(stored),
                    (u32)term_ext(stored),
                    (unsigned long long)term_val(stored),
                    (u32)term_tag(grad),
                    (u32)term_ext(grad),
                    (unsigned long long)term_val(grad));
        }
        heap_set(ctx, dst_slot, stored);
        return;
    }
    if (term_tag(grad) == TAG_NUM && term_as_f32(grad) == 0.0f) return;
    Term stored = term_clone(ctx, grad);
    if (getenv("THVM_LOOP_DIAG")) {
        fprintf(stderr,
                "BUNDLE_ACCUM_ADD loc=%llu idx=%u dst_slot=%llu prev_tag=%u prev_ext=%u prev_val=%llu stored_tag=%u stored_ext=%u stored_val=%llu src_tag=%u src_ext=%u src_val=%llu\n",
                (unsigned long long)grad_loc,
                index,
                (unsigned long long)dst_slot,
                (u32)term_tag(prev),
                (u32)term_ext(prev),
                (unsigned long long)term_val(prev),
                (u32)term_tag(stored),
                (u32)term_ext(stored),
                (unsigned long long)term_val(stored),
                (u32)term_tag(grad),
                (u32)term_ext(grad),
                (unsigned long long)term_val(grad));
    }
    heap_set(ctx, dst_slot, thvm_op_raw(ctx, UOP_ADD, prev, stored));
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

    Term bundle = n_params == 1 ? term_num_f32(0.0f) : thvm_grad_bundle_new(ctx, n_params);
    Term keep_root = thvm_app(ctx, driver, bundle);

    thvm_grad_target_set(ctx, grad_loc, thvm_any());
    thvm_grad_mode_set(ctx, grad_loc, GRAD_MODE_KEEP);
    thvm_grad_targets_set_for_loc(ctx, grad_loc, params, NULL, n_params);
    thvm_grad_keep_app_loc_set(ctx, grad_loc, term_val(keep_root));
    if (n_params != 1)
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
    if (getenv("THVM_LOOP_DIAG")) {
        fprintf(stderr, "BUNDLE_GET whnf tag=%u ext=%u val=%llu idx=%u\n",
                (u32)term_tag(bundle), (u32)term_ext(bundle),
                (unsigned long long)term_val(bundle), index);
    }
    if (term_tag(bundle) != TAG_CTR) return index == 0 ? bundle : term_era();
    {
        u64 bloc = term_val(bundle);
        u32 arity = (u32)term_ext(bundle);
        int need_stage = 0;
        if (bloc != 0 && bloc + arity <= ctx->heap_pos) {
            for (u32 i = 0; i < arity; i++) {
                Term child = heap_read(ctx, bloc + i);
                if (term_tag(child) == TAG_TOP) {
                    need_stage = 1;
                    break;
                }
            }
        }
        if (need_stage) {
            bundle = thvm_eval_exec_fixed_point(ctx, bundle);
            bundle = thvm_grad_bundle_whnf(ctx, bundle);
            if (getenv("THVM_LOOP_DIAG")) {
                fprintf(stderr, "BUNDLE_GET staged tag=%u ext=%u val=%llu\n",
                        (u32)term_tag(bundle), (u32)term_ext(bundle),
                        (unsigned long long)term_val(bundle));
            }
        }
    }
    if (index >= (u32)term_ext(bundle)) return term_era();
    u64 loc = term_val(bundle);
    if (loc == 0 || loc + index >= ctx->heap_pos) return term_era();
    if (getenv("THVM_LOOP_DIAG")) {
        Term raw = heap_read(ctx, loc + index);
        fprintf(stderr, "BUNDLE_GET raw slot=%llu tag=%u ext=%u val=%llu\n",
                (unsigned long long)(loc + index),
                (u32)term_tag(raw), (u32)term_ext(raw),
                (unsigned long long)term_val(raw));
        if (term_tag(raw) == TAG_TOP && term_ext(raw) == UOP_ADD) {
            u64 rloc = term_val(raw);
            if (rloc + 1 < ctx->heap_pos) {
                Term ra = heap_read(ctx, rloc + 0);
                Term rb = heap_read(ctx, rloc + 1);
                fprintf(stderr, "BUNDLE_GET raw_add a=(tag=%u ext=%u val=%llu) b=(tag=%u ext=%u val=%llu)\n",
                        (u32)term_tag(ra), (u32)term_ext(ra), (unsigned long long)term_val(ra),
                        (u32)term_tag(rb), (u32)term_ext(rb), (unsigned long long)term_val(rb));
            }
        }
    }
    Term raw = heap_read(ctx, loc + index);
    Term out = thvm_force_tensor_term(ctx, raw);
    if (term_tag(out) == TAG_ERA && term_val(out) == 0) {
        Term rebuilt = thvm_grad_force_slot_expr(ctx, raw, 0);
        if (!(term_tag(rebuilt) == TAG_ERA && term_val(rebuilt) == 0)) return rebuilt;
    }
    return out;
}
