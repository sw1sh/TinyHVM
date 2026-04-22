// Forward declarations for fusion (defined in fuse/_.c)
static int is_elementwise(u32 uop);
static int is_binary(u32 uop);

// Reduce a term to TAG_TEN and return its tensor ID (or ~0u on failure)

// Does this UOP allocate a fresh buffer (safe to decref inputs)?
// Movement ops share the input buffer → NOT safe to decref.
static inline int uop_allocates_fresh(u32 uop) {
    switch (uop) {
        case UOP_RESHAPE: case UOP_PERMUTE: case UOP_EXPAND:
        case UOP_SHRINK:  case UOP_PAD:
            return 0;  // shares buffer
        case UOP_ASSIGN:
            return 0;  // result IS the dst input
        case UOP_GRAD:
            return 0;  // complex autograd — don't touch
        case UOP_IFZ:
        case UOP_LOG_PRINT:
        case UOP_DETACH:
        case UOP_KERNEL:
        case UOP_FUSE:
            return 0;  // returns sub-terms / passthrough, not a fresh tensor
        default:
            return 1;  // compute ops: ADD, SUB, MUL, MM, SUM, etc.
    }
}

static inline u32 reduce_top_arity(u32 uop) {
    return thvm_uop_storage_arity(uop);
}

static inline int reduce_term_is_era_like(TinyHVM *ctx, Term t, Term *era_out) {
    if (term_tag(t) == TAG_ERA) {
        if (era_out) *era_out = t;
        return 1;
    }
    if (term_tag(t) == TAG_VAR) {
        u64 loc = term_val(t);
        if (loc < ctx->heap_pos) {
            Term sub = heap_read(ctx, loc);
            if (term_tag(sub) == TAG_ERA) {
                if (era_out) *era_out = sub;
                return 1;
            }
        }
    }
    return 0;
}

static inline int reduce_top_has_era_arg(TinyHVM *ctx, Term t) {
    if (term_tag(t) != TAG_TOP) return 0;
    u32 uop = term_ext(t);
    if (uop == UOP_FUSE) return 0;
    u64 loc = term_val(t);
    u32 arity = reduce_top_arity(uop);
    for (u32 i = 0; i < arity; i++) {
        Term child = heap_read(ctx, loc + i);
        if (reduce_term_is_era_like(ctx, child, NULL)) return 1;
    }
    return 0;
}

static inline int reduce_top_has_add_zero_arg(TinyHVM *ctx, Term t) {
    if (term_tag(t) != TAG_TOP || term_ext(t) != UOP_ADD) return 0;
    u64 loc = term_val(t);
    Term a = heap_read(ctx, loc + 0);
    Term b = heap_read(ctx, loc + 1);
    return (term_tag(a) == TAG_NUM && term_as_f32(a) == 0.0f) ||
           (term_tag(b) == TAG_NUM && term_as_f32(b) == 0.0f);
}

/* A FUSE child is "absorbable" when thvm_fuse_public_term can either
 * bind it as a kernel leaf (TEN/KERNEL/atom) OR wrap it in FUSE so it
 * fires next. Compute-like raw TOPs (ADD/MUL/.../view ops) fall into
 * the second bucket via thvm_fuse_wrap_child. Rejecting them here
 * strands the parent FUSE forever when the outer scheduler pipeline
 * is not going to run (e.g. during pure IC step tracing). */
static inline int reduce_fuse_child_absorbable(TinyHVM *ctx, Term child) {
    if (thvm_kernel_local_child_ready(ctx, child)) return 1;
    if (term_tag(child) == TAG_TOP) {
        u32 cu = term_ext(child);
        if (is_binary(cu) || is_elementwise(cu) || cu == UOP_SUM ||
            cu == UOP_RMAX || is_view_op(cu) || cu == UOP_FUSE)
            return 1;
    }
    return 0;
}

static inline int reduce_fuse_payload_top_ready(TinyHVM *ctx, Term t) {
    if (!ctx || term_tag(t) != TAG_TOP) return 0;
    u32 uop = term_ext(t);
    if (uop == UOP_ASSIGN || uop == UOP_KERNEL || uop == UOP_FUSE)
        return 1;
    if (!(is_binary(uop) || is_elementwise(uop) ||
          uop == UOP_SUM || uop == UOP_RMAX || is_view_op(uop)))
        return 0;
    u64 loc = term_val(t);
    if (loc == 0 || loc >= ctx->heap_pos) return 0;
    if (!reduce_fuse_child_absorbable(ctx, heap_read(ctx, loc + 0)))
        return 0;
    if (!is_binary(uop) && is_elementwise(uop))
        return 1;
    if (loc + 1 >= ctx->heap_pos) return 0;
    return reduce_fuse_child_absorbable(ctx, heap_read(ctx, loc + 1));
}

static inline int reduce_fuse_payload_ready(TinyHVM *ctx, Term t) {
    switch (term_tag(t)) {
        case TAG_TEN:
        case TAG_ERA:
        case TAG_NUM:
        case TAG_SEQ:
        case TAG_CTR:
        case TAG_LAM:
        case TAG_BRI:
        case TAG_SUP:
        case TAG_USP:
        case TAG_MAT:
        case TAG_ANN:
        case TAG_ANY:
            return 1;
        case TAG_TOP:
            return reduce_fuse_payload_top_ready(ctx, t);
        // APP / REF / ALO / VAR / DP / etc. are not useful shapes for FUSE
        // to act on — let the reducer collapse them first, then FUSE sees
        // whatever CTR / compute TOP / SEQ they expose.
        default:
            return 0;
    }
}

static inline int reduce_top_direct_uop(u32 uop) {
    return uop == UOP_ASSIGN || uop == UOP_GRAD || uop == UOP_GRAD_FWD || uop == UOP_IFZ ||
           uop == UOP_LOG_PRINT || uop == UOP_TODEVICE || uop == UOP_CAST ||
           uop == UOP_DETACH ||
           uop == UOP_WHERE ||
           uop == UOP_EXEC ||
           uop == UOP_KERNEL ||
           uop == UOP_FUSE;
}

static inline int reduce_top_direct_uop_ctx(TinyHVM *ctx, u32 uop) {
    (void)ctx;
    return reduce_top_direct_uop(uop);
}

static inline u32 reduce_net_term_arity(Term t) {
    u8 tag = term_tag(t);
    u32 ext = term_ext(t);
    switch (tag) {
        case TAG_TOP:
            return reduce_top_arity(ext);
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

static int reduce_net_has_parent_ref(TinyHVM *ctx, Term target) {
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term p = ctx->heap[h];
        u32 ar = reduce_net_term_arity(p);
        u64 loc = term_val(p);
        if (ar == 0 || loc == 0 || loc + ar > ctx->heap_pos) continue;
        for (u32 i = 0; i < ar; i++) {
            if (heap_read(ctx, loc + i) == target) return 1;
        }
    }
    return 0;
}

static int reduce_net_term_is_seq_kernel(TinyHVM *ctx, Term t) {
    if (term_tag(t) != TAG_TOP || term_ext(t) != UOP_KERNEL) return 0;
    u64 loc = term_val(t);
    if (loc == 0 || loc + 2 >= ctx->heap_pos) return 0;
    Term root_uop = heap_read(ctx, loc + 2);
    return term_tag(root_uop) == TAG_NUM && (u32)term_val(root_uop) == UOP_COUNT;
}

static void reduce_net_mark_reachable_slots(TinyHVM *ctx, Term root, u8 *reach) {
    if (!reach || ctx->heap_pos == 0) return;
    memset(reach, 0, (size_t)ctx->heap_pos);
    u8 *seen_slot = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    u8 *seen_dup  = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    u64 work_cap = ctx->heap_pos ? (ctx->heap_pos * 8) : 0;
    Term *work = work_cap ? (Term *)malloc(sizeof(Term) * (size_t)work_cap) : NULL;
    u64 wp = 0;
    #define REDUCE_PUSH(_tt) do { \
        if (work && wp < work_cap) work[wp++] = (_tt); \
    } while (0)

    if (term_tag(root) != TAG_ERA && !(term_tag(root) == TAG_ERA && term_val(root) == 0))
        REDUCE_PUSH(root);

    while (work && wp > 0) {
        Term tt = work[--wp];
        u8 tg = term_tag(tt);
        u64 tv = term_val(tt);

        if (tg == TAG_DP0 || tg == TAG_DP1) {
            u64 dl = tv;
            if (dl == 0 || dl >= ctx->heap_pos || (seen_dup && seen_dup[dl])) continue;
            if (seen_dup) seen_dup[dl] = 1;
            reach[dl] = 1;
            REDUCE_PUSH(heap_read(ctx, dl));
            continue;
        }

        if (tg == TAG_ERA) {
            continue;
        }

        // Global quiescence must respect lazy barriers. Lambda / matcher bodies
        // are only reduced when entered by ordinary APP-driven evaluation, not
        // by the heap-wide reachable scan.
        if (tg == TAG_LAM || tg == TAG_BRI || tg == TAG_MAT || tg == TAG_ANN) {
            continue;
        }

        // SEQ is strict in its left branch. The heap-wide reachable scan must
        // not descend into the continuation before the ordinary SEQ
        // interaction returns it.
        if (tg == TAG_SEQ) {
            u64 p = tv + 0;
            if (p < ctx->heap_pos && (!seen_slot || !seen_slot[p])) {
                if (seen_slot) seen_slot[p] = 1;
                reach[p] = 1;
                REDUCE_PUSH(heap_read(ctx, p));
            }
            continue;
        }

        if (reduce_net_term_is_seq_kernel(ctx, tt)) {
            u64 left_p = tv + 0;
            Term left = term_era();
            if (left_p < ctx->heap_pos && (!seen_slot || !seen_slot[left_p])) {
                if (seen_slot) seen_slot[left_p] = 1;
                reach[left_p] = 1;
                left = heap_read(ctx, left_p);
                REDUCE_PUSH(left);
            } else if (left_p < ctx->heap_pos) {
                left = heap_read(ctx, left_p);
            }
            if (term_tag(left) != TAG_TOP) {
                u64 right_p = tv + 1;
                if (right_p < ctx->heap_pos && (!seen_slot || !seen_slot[right_p])) {
                    if (seen_slot) seen_slot[right_p] = 1;
                    reach[right_p] = 1;
                    REDUCE_PUSH(heap_read(ctx, right_p));
                }
            }
            continue;
        }

        u32 ar = reduce_net_term_arity(tt);
        for (u32 i = 0; i < ar; i++) {
            u64 p = tv + i;
            if (p < ctx->heap_pos && (!seen_slot || !seen_slot[p])) {
                if (seen_slot) seen_slot[p] = 1;
                reach[p] = 1;
                REDUCE_PUSH(heap_read(ctx, p));
            }
        }
    }

    free(work);
    free(seen_dup);
    free(seen_slot);
    #undef REDUCE_PUSH
}

static int reduce_net_term_first_reachable_occurrence(TinyHVM *ctx, u64 h, Term t, const u8 *reach) {
    for (u64 i = 1; i < h; i++) {
        if (reach && !reach[i]) continue;
        if (ctx->heap[i] == t) return 0;
    }
    return 1;
}

static int reduce_net_term_maybe_active(TinyHVM *ctx, Term t) {
    u8 tag = term_tag(t);
    if (tag == TAG_TOP) {
        return reduce_top_direct_uop_ctx(ctx, term_ext(t)) ||
               reduce_top_has_era_arg(ctx, t) ||
               reduce_top_has_add_zero_arg(ctx, t);
    }
    if (tag == TAG_ERA) return term_val(t) != 0 && !reduce_net_has_parent_ref(ctx, t);
    if (tag == TAG_TEN || tag == TAG_NUM || tag == TAG_LAM || tag == TAG_SUP ||
        tag == TAG_BRI || tag == TAG_MAT || tag == TAG_ANY || tag == TAG_USP)
        return 0;
    return 1;
}


// Forward decl — defined in interact/_.c.
static Term thvm_interact(TinyHVM *ctx, Term t);

static int reduce_net_fire_one(TinyHVM *ctx, Term in, Term *out_term) {
    // Fire one IC interaction on `in`.  No argument reduction, no
    // recursion — exactly what quiesce needs for per-slot sweep.
    u64 itrs_before = ctx->itrs;
    Term r = thvm_interact(ctx, in);
    int fired = (r != in) || (ctx->itrs != itrs_before);
    if (out_term) *out_term = r;
    return fired;
}

static Term reduce_net_quiesce(TinyHVM *ctx, Term root) {
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
        Term rr = root;
        if (reduce_net_term_maybe_active(ctx, root)) {
            if (reduce_net_fire_one(ctx, root, &rr)) {
                root = rr;
                continue;
            }
        }

        reduce_net_mark_reachable_slots(ctx, root, reach);
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term ht = ctx->heap[h];
            if (reach && !reach[h]) continue;
            if (!reduce_net_term_maybe_active(ctx, ht)) continue;
            if ((term_tag(ht) == TAG_DP0 || term_tag(ht) == TAG_DP1 ||
                 (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_GRAD)) &&
                !reduce_net_term_first_reachable_occurrence(ctx, h, ht, reach)) {
                continue;
            }
            Term hr = ht;
            if (!reduce_net_fire_one(ctx, ht, &hr)) continue;
            /* Atom-share ERA guard: if firing a cell's term collapses
             * it to ERA while the term itself is a live compute TOP,
             * suppress the write. Atom-shared TOP DUPs cause quiesce
             * to visit the shared structure from multiple arms; partial
             * resolutions can return ERA even though the value is still
             * referenced by other arms. Letting ERA through here
             * propagates up to the root (prior boundary fix) or to a
             * parent's arg slot, silently dropping live computation. */
            if (term_tag(hr) == TAG_ERA && term_tag(ht) == TAG_TOP) {
                u32 htu = term_ext(ht);
                if (htu != UOP_DETACH && htu != UOP_ASSIGN &&
                    htu != UOP_KERNEL && htu != UOP_EXEC) {
                    continue;
                }
            }
            if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_GRAD) {
                for (u64 i = 1; i < ctx->heap_pos; i++) {
                    if (ctx->heap[i] == ht) heap_set(ctx, i, hr);
                }
            } else {
                heap_set(ctx, h, hr);
            }
            fired = 1;
            break;
        }
        if (!fired) break;
    }
    free(reach);
    return root;
}

// Decref input tensors after a TOP fires and produces a fresh output.
// Skip if the OUTPUT is grad-tracked — GRAD walks src_ids backward through
// the entire forward tape, so all intermediate tensors must stay alive.

// ============================================================
// print.c — Debug printer
// ============================================================

static const char *tag_names[] = {
    "APP", "LAM", "VAR", "SUP", "DP0", "DP1", "ERA",
    "NUM", "REF", "OP2", "TEN", "TOP", "CTR",
    "BRI", "ANN", "DSU", "DDU", "INC",
    "EQL", "AND", "OR", "MAT", "ANY", "USP", "UDP", "SEQ"
};

// uop_names now in tinyhvm.h

void thvm_print_term(TinyHVM *ctx, Term t) {
    u32 tag = term_tag(t);
    (void)ctx;
    if (tag < TAG_COUNT) printf("%s", tag_names[tag]);
    else printf("?%u", tag);

    switch (tag) {
        case TAG_NUM:
            if (term_ext(t) == NUM_F32) printf("(%.4f)", term_as_f32(t));
            else printf("(%u)", term_as_u32(t));
            break;
        case TAG_TEN: {
            u32 tid = (u32)term_val(t);
            printf("(id=%u", tid);
            if (ctx && tid < ctx->tensor_count) {
                View *v = &ctx->tensors[tid].view;
                printf(" [");
                for (u32 i = 0; i < v->shape.rank; i++) printf("%s%u", i?",":"", v->shape.dims[i]);
                printf("]");
            }
            printf(")");
            break;
        }
        case TAG_TOP:
            if (term_ext(t) < UOP_COUNT) printf("(%s @%llu)", uop_names[term_ext(t)], (unsigned long long)term_val(t));
            else printf("(uop=%u @%llu)", term_ext(t), (unsigned long long)term_val(t));
            break;
        case TAG_SUP:
            printf("(lab=%u @%llu)", term_ext(t), (unsigned long long)term_val(t));
            break;
        case TAG_BRI:
            printf("(θ @%llu)", (unsigned long long)term_val(t));
            break;
        case TAG_ANN:
            printf("({:} @%llu)", (unsigned long long)term_val(t));
            break;
        case TAG_DSU:
            printf("(&dyn @%llu)", (unsigned long long)term_val(t));
            break;
        case TAG_DDU:
            printf("(!&dyn @%llu)", (unsigned long long)term_val(t));
            break;
        case TAG_INC:
            printf("(↓ @%llu)", (unsigned long long)term_val(t));
            break;
        default:
            if (term_val(t) || term_ext(t))
                printf("(ext=%u, val=%llu)", term_ext(t), (unsigned long long)term_val(t));
            break;
    }
}

// ============================================================
// api.c — High-level tensor API
// ============================================================

// Device registry
extern Backend cpu_backend;
