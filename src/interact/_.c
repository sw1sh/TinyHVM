// interact/_.c — interaction handler dispatch
// Split into sub-files for readability:
//   grad.c        — UOP_GRAD (backward chain rule)
//   tensor_ops.c  — ASSIGN, TODEVICE, WHERE, IFZ, LOG_PRINT, ew/reduce/view ops
//   combinators.c — APP, LAM, REF, SUP, DP0/DP1, OP2, VAR, etc.

// Read small metadata (axes, shapes, pad specs) without GPU flush.
#define META_READ(be, buf_id, out, bytes) \
    ((be)->buf_read_nosync ? \
     (be)->buf_read_nosync((buf_id), (out), (bytes)) : \
     (be)->buf_read((buf_id), (out), (bytes)))

// Forward declarations (defined in rewrite/_.c, included after interact)
static int is_view_op(u32 uop);
static int is_elementwise(u32 uop);

static Term thvm_era_payload(TinyHVM *ctx, Term item) {
    while (term_tag(item) == TAG_ERA) {
        u64 el = term_val(item);
        if (el == 0 || el >= ctx->heap_pos) return term_era();
        item = heap_read(ctx, el);
    }
    return item;
}

static Term thvm_make_active_era(TinyHVM *ctx, Term item) {
    item = thvm_era_payload(ctx, item);
    if (term_tag(item) == TAG_ERA && term_val(item) == 0) return term_era();
    u64 el = heap_alloc(ctx, 1);
    heap_set(ctx, el, item);
    return term_era_at(el);
}

static int thvm_term_is_active_era_like(TinyHVM *ctx, Term item, Term *era_out) {
    if (term_tag(item) == TAG_ERA && term_val(item) != 0) {
        if (era_out) *era_out = item;
        return 1;
    }
    if (term_tag(item) == TAG_VAR) {
        u64 loc = term_val(item);
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

static void thvm_spawn_detached_era(TinyHVM *ctx, Term item) {
    Term era = thvm_make_active_era(ctx, item);
    if (term_tag(era) == TAG_ERA && term_val(era) != 0) {
        u64 slot = heap_alloc(ctx, 1);
        heap_set(ctx, slot, era);
    }
}

static u32 thvm_alo_state_push(TinyHVM *ctx, u32 parent, u8 kind, u64 bind_book, u64 bind_dyn, u32 label_old, u32 label_new) {
    if (!ctx->alo_states) return 0;
    if (ctx->alo_state_count >= ctx->alo_state_cap) {
        u32 new_cap = ctx->alo_state_cap ? (ctx->alo_state_cap << 1) : (1u << 16);
        ctx->alo_states = (AloState *)realloc(ctx->alo_states, (size_t)new_cap * sizeof(AloState));
        memset(ctx->alo_states + ctx->alo_state_cap, 0, (size_t)(new_cap - ctx->alo_state_cap) * sizeof(AloState));
        ctx->alo_state_cap = new_cap;
    }
    u32 id = ctx->alo_state_count++;
    ctx->alo_states[id] = (AloState){
        .parent = parent,
        .bind_book = bind_book,
        .bind_dyn = bind_dyn,
        .label_old = label_old,
        .label_new = label_new,
        .kind = kind
    };
    return id;
}

static int thvm_alo_lookup_bind(TinyHVM *ctx, u32 state_id, u64 bind_book, u64 *out_dyn) {
    for (u32 sid = state_id; sid != 0; sid = ctx->alo_states[sid].parent) {
        AloState *s = &ctx->alo_states[sid];
        if (s->kind == 1 && s->bind_book == bind_book) {
            if (out_dyn) *out_dyn = s->bind_dyn;
            return 1;
        }
    }
    return 0;
}

static u32 thvm_alo_get_or_add_label(TinyHVM *ctx, u32 state_id, u32 old_label, u32 *io_state_id) {
    for (u32 sid = state_id; sid != 0; sid = ctx->alo_states[sid].parent) {
        AloState *s = &ctx->alo_states[sid];
        if (s->kind == 2 && s->label_old == old_label) return s->label_new;
    }
    u32 fresh = thvm_fresh_label(ctx);
    u32 next_state = thvm_alo_state_push(ctx, state_id, 2, 0, 0, old_label, fresh);
    if (io_state_id) *io_state_id = next_state;
    return fresh;
}

static Term thvm_alo_make(TinyHVM *ctx, Term book_term, u32 state_id) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc + 0, book_term);
    heap_set(ctx, loc + 1, term_num_u32(state_id));
    return term_new(TAG_ALO, 0, loc);
}

static Term thvm_alo_suspend_child(TinyHVM *ctx, Term child, u32 state_id) {
    u8 tag = term_tag(child);
    if (tag == TAG_NUM || tag == TAG_TEN || tag == TAG_ERA || tag == TAG_ANY || tag == TAG_REF)
        return child;
    if (tag == TAG_VAR) {
        u64 dyn_loc = 0;
        if (thvm_alo_lookup_bind(ctx, state_id, term_val(child), &dyn_loc))
            return term_new(TAG_VAR, term_ext(child), dyn_loc);
        return child;
    }
    return thvm_alo_make(ctx, child, state_id);
}

static u32 thvm_alo_top_arity(u32 ext) {
    if (ext == UOP_KERNEL) return 0;
    if (ext == UOP_FUSE || ext == UOP_SCHED) return 1;
    if (ext == UOP_FUSE2) return 3;
    if (ext == UOP_WHERE || ext == UOP_IFZ) return 3;
    if (ext == UOP_GRAD) return 2;
    if (ext == UOP_LOG_PRINT || ext == UOP_DETACH) return 1;
    if (!is_binary(ext) && is_elementwise(ext)) return 1;
    return 2;
}

static u32 thvm_alo_book_arity(Term t) {
    switch (term_tag(t)) {
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
        case TAG_DSU:
        case TAG_DDU:
            return 3;
        case TAG_INC:
        case TAG_UDP:
            return 1;
        case TAG_CTR:
            return term_ext(t);
        case TAG_TOP:
            return thvm_alo_top_arity(term_ext(t));
        default:
            return 0;
    }
}

static Term thvm_alo_realize(TinyHVM *ctx, Term book_term, u32 state_id) {
    u8 tag = term_tag(book_term);
    if (tag == TAG_REF || tag == TAG_NUM || tag == TAG_TEN || tag == TAG_ERA || tag == TAG_ANY)
        return book_term;

    if (tag == TAG_VAR) {
        u64 old_loc = term_val(book_term);
        u64 dyn_loc = 0;
        if (thvm_alo_lookup_bind(ctx, state_id, old_loc, &dyn_loc))
            return term_new(TAG_VAR, term_ext(book_term), dyn_loc);
        return book_term;
    }

    if (tag == TAG_DP0 || tag == TAG_DP1) {
        u32 walk_state = state_id;
        u32 new_lab = thvm_alo_get_or_add_label(ctx, walk_state, term_ext(book_term), &walk_state);
        u64 old_loc = term_val(book_term);
        u64 dyn_loc = 0;
        if (!thvm_alo_lookup_bind(ctx, walk_state, old_loc, &dyn_loc)) {
            dyn_loc = heap_alloc(ctx, 1);
            Term child = (old_loc > 0 && old_loc < ctx->book_heap_pos) ? ctx->book_heap[old_loc] : term_era();
            heap_set(ctx, dyn_loc, thvm_alo_suspend_child(ctx, child, walk_state));
            walk_state = thvm_alo_state_push(ctx, walk_state, 1, old_loc, dyn_loc, 0, 0);
        }
        return term_new(tag, new_lab, dyn_loc);
    }

    if (tag == TAG_SUP || tag == TAG_USP || tag == TAG_UDP) {
        u32 walk_state = state_id;
        u32 new_lab = thvm_alo_get_or_add_label(ctx, walk_state, term_ext(book_term), &walk_state);
        u32 ar = thvm_alo_book_arity(book_term);
        if (ar == 0) return term_new(tag, new_lab, term_val(book_term));
        u64 old_loc = term_val(book_term);
        u64 new_loc = heap_alloc(ctx, ar);
        for (u32 i = 0; i < ar; i++) {
            Term child = (old_loc > 0 && old_loc + i < ctx->book_heap_pos) ? ctx->book_heap[old_loc + i] : term_era();
            heap_set(ctx, new_loc + i, thvm_alo_suspend_child(ctx, child, walk_state));
        }
        return term_new(tag, new_lab, new_loc);
    }

    if (tag == TAG_LAM || tag == TAG_BRI) {
        u64 old_loc = term_val(book_term);
        u64 new_loc = heap_alloc(ctx, 2);
        Term var = term_new(TAG_VAR, 0, new_loc);
        heap_set(ctx, new_loc + 0, term_set_sub(var));
        u32 body_state = thvm_alo_state_push(ctx, state_id, 1, old_loc, new_loc, 0, 0);
        Term body = (old_loc > 0 && old_loc + 1 < ctx->book_heap_pos) ? ctx->book_heap[old_loc + 1] : term_era();
        heap_set(ctx, new_loc + 1, thvm_alo_suspend_child(ctx, body, body_state));
        return term_new(tag, term_ext(book_term), new_loc);
    }

    u32 ar = thvm_alo_book_arity(book_term);
    if (ar == 0) return book_term;
    u64 old_loc = term_val(book_term);
    u64 new_loc = heap_alloc(ctx, ar);
    for (u32 i = 0; i < ar; i++) {
        Term child = (old_loc > 0 && old_loc + i < ctx->book_heap_pos) ? ctx->book_heap[old_loc + i] : term_era();
        heap_set(ctx, new_loc + i, thvm_alo_suspend_child(ctx, child, state_id));
    }
    return term_new(tag, term_ext(book_term), new_loc);
}

static Term thvm_alo_force(TinyHVM *ctx, Term alo) {
    u64 alo_loc = term_val(alo);
    if (alo_loc == 0 || alo_loc + 1 >= ctx->heap_pos) return alo;
    Term book_term = heap_read(ctx, alo_loc + 0);
    Term sid_term = heap_read(ctx, alo_loc + 1);
    if (term_tag(sid_term) != TAG_NUM) return book_term;
    u32 state_id = term_as_u32(sid_term);
    Term out = thvm_alo_realize(ctx, book_term, state_id);
    // #region agent log
    do {
        static u32 alo_force_dbg_count = 0;
        if (alo_force_dbg_count >= 12) break;
        alo_force_dbg_count++;
        u32 out_ar = thvm_alo_book_arity(out);
        u32 child0_tag = 255;
        u32 child1_tag = 255;
        u64 child0_val = 0;
        u64 child1_val = 0;
        u64 out_loc = term_val(out);
        if (out_ar > 0 && out_loc > 0 && out_loc < ctx->heap_pos) {
            Term child0 = heap_read(ctx, out_loc + 0);
            child0_tag = term_tag(child0);
            child0_val = term_val(child0);
            if (out_ar > 1 && out_loc + 1 < ctx->heap_pos) {
                Term child1 = heap_read(ctx, out_loc + 1);
                child1_tag = term_tag(child1);
                child1_val = term_val(child1);
            }
        }
        char _dbg[384];
        snprintf(_dbg, sizeof(_dbg),
                 "{\"alo_tag\":%u,\"alo_loc\":%llu,\"book_tag\":%u,\"book_ext\":%u,"
                 "\"book_val\":%llu,\"state_id\":%u,\"out_tag\":%u,\"out_ext\":%u,"
                 "\"out_val\":%llu,\"out_arity\":%u,\"out_child0_tag\":%u,"
                 "\"out_child0_val\":%llu,\"out_child1_tag\":%u,\"out_child1_val\":%llu}",
                 (u32)term_tag(alo), (unsigned long long)alo_loc,
                 (u32)term_tag(book_term), (u32)term_ext(book_term),
                 (unsigned long long)term_val(book_term), state_id,
                 (u32)term_tag(out), (u32)term_ext(out),
                 (unsigned long long)term_val(out), out_ar,
                 child0_tag, (unsigned long long)child0_val,
                 child1_tag, (unsigned long long)child1_val);
        thvm_agent_debug_log("pre-fix", "H4", "src/interact/_.c:220",
                             "alo_force_result", _dbg);
    } while (0);
    // #endregion
    thvm_step_alo_subst_record(ctx, alo_loc, out, book_term);
    // #region agent log
    do {
        static u32 alo_subst_dbg_count = 0;
        if (alo_subst_dbg_count >= 12) break;
        alo_subst_dbg_count++;
        char _dbg[224];
        snprintf(_dbg, sizeof(_dbg),
                 "{\"old_alo_loc\":%llu,\"out_tag\":%u,\"out_val\":%llu}",
                 (unsigned long long)alo_loc,
                 (u32)term_tag(out),
                 (unsigned long long)term_val(out));
        thvm_agent_debug_log("pre-fix", "H11", "src/interact/_.c:264",
                             "alo_subst_recorded", _dbg);
    } while (0);
    // #endregion
    heap_set(ctx, alo_loc + 0, out);
    heap_set(ctx, alo_loc + 1, term_era());
    return out;
}

static Term thvm_interact(TinyHVM *ctx, Term t) {
    // Return result directly — the trampoline handles TAG_TOP results
    // via `next = r; goto enter;` (no need to force-reduce here).
    // Return directly — trampoline handles TAG_TOP via goto enter
    #define RETURN_REDUCED(result) do { return (result); } while(0)
    // One interact() call performs exactly one local rewrite.
    // Continued reduction is the reducer trampoline's job.
    #define INET_RECURSE() do { return t; } while (0)
    // No GRAD_STEP — all GRAD sub-terms are placed on heap iteratively

    u32 tag;
inet_step:
    tag = term_tag(t);

    switch (tag) {
        case TAG_TOP: {
            u32 uop = term_ext(t);
            u64 loc = term_val(t);

            // === UOP_GRAD ===
#include "grad.c"

            // === Tensor ops (ASSIGN, ew, reduce, view, etc.) ===
#include "tensor_ops.c"
        } // end case TAG_TOP

        // === Combinators (APP, LAM, SUP, DP0/DP1, etc.) ===
#include "combinators.c"

        default:
            #undef INET_RECURSE
            return t;
    }
    #undef INET_RECURSE
    return t;
}

// ============================================================
// Trampoline frame tags (used by reduce/_.c)
// ============================================================
#define TAG_TOP1  0x7E  // TAG_TOP arg0 done, entering arg1
#define TAG_TOP2  0x7F  // TAG_TOP arg1 done, entering arg2 (GRAD/WHERE/IFZ)
#define TAG_OP2_1 0x7D  // OP2 arg0 done (non-SUP), entering arg1. EXT=opr, VAL=loc
#define TAG_EQL_1 0x7C  // EQL arg0 done (non-SUP), entering arg1. VAL=loc
#define TAG_MAT_1 0x7B  // APP fun=MAT, entering arg. VAL=app_loc

#define FRAME_CAP 65536
