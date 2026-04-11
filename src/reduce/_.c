// Forward declarations for fusion (defined in fuse/_.c)
static int is_elementwise(u32 uop);

// Reduce a term to TAG_TEN and return its tensor ID (or ~0u on failure)
static u32 reduce_id(TinyHVM *ctx, Term t) {
    t = thvm_reduce(ctx, t);
    return (term_tag(t) == TAG_TEN) ? (u32)term_val(t) : ~0u;
}

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
            return 0;  // returns sub-terms / passthrough, not a fresh tensor
        default:
            return 1;  // compute ops: ADD, SUB, MUL, MM, SUM, etc.
    }
}

static inline u32 reduce_top_arity(u32 uop) {
    if (uop == UOP_FUSING) return 0;
    if (uop == UOP_WHERE || uop == UOP_IFZ) return 3;
    if (uop == UOP_GRAD) return 2;
    if (uop == UOP_LOG_PRINT) return 1;
    if (!is_binary(uop) && is_elementwise(uop)) return 1;
    return 2;
}

static inline int reduce_top_has_era_arg(TinyHVM *ctx, Term t) {
    if (term_tag(t) != TAG_TOP) return 0;
    u64 loc = term_val(t);
    u32 arity = reduce_top_arity(term_ext(t));
    for (u32 i = 0; i < arity; i++) {
        Term child = heap_read(ctx, loc + i);
        if (term_tag(child) == TAG_ERA) return 1;
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

static inline int reduce_top_direct_uop(u32 uop) {
    return uop == UOP_ASSIGN || uop == UOP_GRAD || uop == UOP_IFZ ||
           uop == UOP_LOG_PRINT || uop == UOP_TODEVICE || uop == UOP_WHERE ||
           uop == UOP_FUSING;
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

    if (!(term_tag(root) == TAG_ERA && term_val(root) == 0)) REDUCE_PUSH(root);
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) == TAG_ERA && term_val(ht) != 0) {
            reach[h] = 1;
            REDUCE_PUSH(ht);
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
            REDUCE_PUSH(heap_read(ctx, dl));
            continue;
        }

        if (tg == TAG_ERA) {
            if (tv == 0 || tv >= ctx->heap_pos) continue;
            if (!seen_slot || !seen_slot[tv]) {
                if (seen_slot) seen_slot[tv] = 1;
                reach[tv] = 1;
                REDUCE_PUSH(heap_read(ctx, tv));
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
        return reduce_top_direct_uop(term_ext(t)) ||
               reduce_top_has_era_arg(ctx, t) ||
               reduce_top_has_add_zero_arg(ctx, t);
    }
    if (tag == TAG_ERA) return term_val(t) != 0 && !reduce_net_has_parent_ref(ctx, t);
    if (tag == TAG_TEN || tag == TAG_NUM || tag == TAG_LAM || tag == TAG_SUP ||
        tag == TAG_BRI || tag == TAG_MAT || tag == TAG_ANY || tag == TAG_USP)
        return 0;
    return 1;
}

static int reduce_net_term_needs_global_cleanup(TinyHVM *ctx, Term t) {
    u8 tag = term_tag(t);
    if (tag == TAG_ERA) return term_val(t) != 0 && !reduce_net_has_parent_ref(ctx, t);
    if (tag != TAG_TOP) return 0;
    return reduce_top_has_era_arg(ctx, t) || reduce_top_has_add_zero_arg(ctx, t);
}

static int reduce_net_fire_one(TinyHVM *ctx, Term in, Term *out_term) {
    u64 itrs_before = ctx->itrs;
    Term r = thvm_reduce_steps(ctx, in, 1);
    int fired = (ctx->steps_taken > 0) || (ctx->itrs != itrs_before);
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
            int reachable = !(reach && !reach[h]);
            if (!reachable && !reduce_net_term_needs_global_cleanup(ctx, ht)) continue;
            if (!reduce_net_term_maybe_active(ctx, ht)) continue;
            if ((term_tag(ht) == TAG_DP0 || term_tag(ht) == TAG_DP1 ||
                 (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_GRAD)) &&
                reachable &&
                !reduce_net_term_first_reachable_occurrence(ctx, h, ht, reach)) {
                continue;
            }
            Term hr = ht;
            if (!reduce_net_fire_one(ctx, ht, &hr)) continue;
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
static void top_decref_inputs(TinyHVM *ctx, u64 loc, u32 uop, Term result) {
    if (!uop_allocates_fresh(uop)) return;
    // If the result tensor needs grad, the inputs are part of the tape
    if (term_tag(result) == TAG_TEN) {
        u32 rid = (u32)term_val(result);
        if (ctx->tensors[rid].requires_grad) return;
    }
    u32 arity = (uop == UOP_WHERE) ? 3 : 2;
    for (u32 i = 0; i < arity; i++) {
        Term a = heap_read(ctx, loc + i);
        if (term_tag(a) == TAG_TEN) {
            u32 tid = (u32)term_val(a);
            if (!ctx->tensors[tid].requires_grad)
                tensor_decref(ctx, tid);
        }
    }
}

// Pre-allocated stack pool — each recursion level gets a slice.
// 4096 frames per slice, 64 max nesting depth = 2MB TLS.
#define REDUCE_SLICE 16384   // 16K frames per depth level (128KB)
#define REDUCE_MAX_DEPTH 512  // 512 depth × 128KB = 64MB TLS per thread
static _Thread_local Term reduce_pool[REDUCE_SLICE * REDUCE_MAX_DEPTH];
static _Thread_local int  reduce_depth = 0;

Term thvm_reduce(TinyHVM *ctx, Term root) {
    int depth = reduce_depth++;
    if (depth >= REDUCE_MAX_DEPTH) {
        fprintf(stderr, "REDUCE_OVERFLOW: depth=%d tag=%u ext=%u\n", depth, term_tag(root), term_ext(root));
        assert(0 && "reduce depth overflow");
    }
    Term *stk = &reduce_pool[depth * REDUCE_SLICE];
    int  sp = 0;

    Term next = root;
    Term whnf;
    int budget_exhausted = 0;

    #define PUSH(f_)  do { if (sp >= REDUCE_SLICE) { fprintf(stderr, "SP_OVERFLOW sp=%d depth=%d\n", sp, depth); fflush(stderr); assert(0); } stk[sp++] = (f_); } while(0)
    #define BUDGET_HIT() (ctx->step_budget > 0 && ctx->steps_taken >= ctx->step_budget)
    #define TRACE_STEP(before, result) do { \
        if (ctx->trace_enabled && ctx->trace_count < ctx->trace_cap) { \
            struct InteractionTrace *_tr = &ctx->trace_buf[ctx->trace_count++]; \
            _tr->before_tag = term_tag(before); \
            _tr->before_ext = term_ext(before); \
            _tr->before_loc = term_val(before); \
            _tr->after_tag = term_tag(result); \
            _tr->after_ext = term_ext(result); \
            _tr->after_loc = term_val(result); \
            _tr->rule_id = term_tag(before); \
        } \
        if (ctx->step_budget > 0 && ++ctx->steps_taken >= ctx->step_budget) { \
            budget_exhausted = 1; \
            whnf = (result); \
            goto apply; \
        } \
    } while(0)

  enter: {
    u8 tag = term_tag(next);

    // Already WNF atoms → go directly to apply
    if (tag == TAG_TEN || tag == TAG_NUM ||
        tag == TAG_LAM || tag == TAG_SUP || tag == TAG_BRI ||
        tag == TAG_MAT || tag == TAG_ANY || tag == TAG_USP) { whnf = next; goto apply; }

    // ERA: inert ERA(0) is WNF; active ERA(val!=0) must interact.
    if (tag == TAG_ERA) {
        if (term_val(next) == 0) { whnf = next; goto apply; }
        if (BUDGET_HIT()) { whnf = next; goto apply; }
        Term r = thvm_interact(ctx, next);
        if (r == next) { whnf = next; goto apply; }
        TRACE_STEP(next, r);
        next = r;
        goto enter;
    }

    // TAG_TOP: ALL compute and movement ops are WNF. Nothing fires eagerly.
    // Only ASSIGN, GRAD, IFZ, LOG_PRINT, TODEVICE, WHERE, FUSING fire.
    // Movement ops are resolved by the scheduler in Phase 2.
    if (tag == TAG_TOP) {
        u32 _uop = term_ext(next);
        if (reduce_top_has_era_arg(ctx, next) || reduce_top_has_add_zero_arg(ctx, next)) {
            if (BUDGET_HIT()) { whnf = next; goto apply; }
            Term r = thvm_interact(ctx, next);
            if (r == next) { whnf = next; goto apply; }
            TRACE_STEP(next, r);
            next = r;
            goto enter;
        }
        if (_uop != UOP_ASSIGN && _uop != UOP_GRAD && _uop != UOP_IFZ &&
            _uop != UOP_LOG_PRINT && _uop != UOP_TODEVICE && _uop != UOP_WHERE &&
            _uop != UOP_FUSING) {
            whnf = next; goto apply;
        }
        // Non-compute: reduce args then fire interact
        u64 loc = term_val(next);
        Term a0 = heap_read(ctx, loc + 0);
        PUSH(next);
        next = a0;
        goto enter;
    }

    // Combinator tags: push frame and reduce arg0 for tags that need it.
    if (tag == TAG_CTR) {
        u64 loc = term_val(next);
        if (term_ext(next) == 0 || loc == 0 || loc >= ctx->heap_pos) {
            if (BUDGET_HIT()) { whnf = next; goto apply; }
            Term r = thvm_interact(ctx, next);
            if (r == next) { whnf = next; goto apply; }
            TRACE_STEP(next, r);
            next = r;
            goto enter;
        }
        PUSH(next);
        next = heap_read(ctx, loc + 0);
        goto enter;
    }

    if (tag == TAG_APP || tag == TAG_DP0 || tag == TAG_DP1 ||
        tag == TAG_OP2 || tag == TAG_EQL || tag == TAG_AND || tag == TAG_OR ||
        tag == TAG_DSU || tag == TAG_DDU || tag == TAG_UDP) {
        u64 loc = term_val(next);
        PUSH(next);
        next = heap_read(ctx, loc + 0);
        goto enter;
    }

    {
        if (BUDGET_HIT()) { whnf = next; goto apply; }
        Term r = thvm_interact(ctx, next);
        if (r == next) { whnf = next; goto apply; }
        TRACE_STEP(next, r);
        next = r;
        goto enter;
    }
  }

  apply: {
    while (sp > 0) {
        Term frame = stk[--sp];
        u8   ftag  = term_tag(frame);

        if (ftag == TAG_TOP) {
            // arg0 just finished. whnf = arg0's result.
            u64 loc = term_val(frame);
            if (term_tag(whnf) != TAG_TEN && term_tag(whnf) != TAG_ERA &&
                term_tag(whnf) != TAG_NUM && term_tag(whnf) != TAG_SUP &&
                !(term_tag(whnf) == TAG_TOP && term_ext(frame) == UOP_GRAD)) {
                heap_set(ctx, loc+0, whnf); whnf = frame; continue;
            }
            heap_set(ctx, loc + 0, whnf);  // store arg0 result

            // GRAD: gy stays lazy and target spec is kept outside heap.
            // Fire immediately once y is in WNF.
            if (term_ext(frame) == UOP_GRAD) {
                if (budget_exhausted || BUDGET_HIT()) { whnf = frame; continue; }
                Term r = thvm_interact(ctx, frame);
                if (r == frame) { whnf = frame; continue; }
                TRACE_STEP(frame, r);
                top_decref_inputs(ctx, loc, term_ext(frame), r);
                next = r;
                goto enter;
            }

            // Check arg1: is it already ready?
            Term a1 = heap_read(ctx, loc + 1);
            u8 a1t2 = term_tag(a1);

            // Any WNF in arg1 slot is "ready"
            if (a1t2 == TAG_TEN || a1t2 == TAG_ERA || a1t2 == TAG_NUM ||
                a1t2 == TAG_LAM || a1t2 == TAG_SUP) {
                // Both args ready — fire
                if (budget_exhausted || BUDGET_HIT()) { whnf = frame; continue; }
                Term r = thvm_interact(ctx, frame);
                if (r == frame) { whnf = frame; continue; }
                TRACE_STEP(frame, r);
                top_decref_inputs(ctx, loc, term_ext(frame), r);
                next = r; goto enter;
            }
            // arg1 not ready: push TOP1 sentinel frame
            PUSH(term_new(TAG_TOP1, (u8)term_ext(frame), loc));
            next = a1;
            goto enter;
        }

        if (ftag == TAG_TOP1) {
            // arg1 just finished. whnf = arg1's result (any WNF is valid: TEN, ERA, NUM, LAM, SUP).
            u64 loc = term_val(frame);
            u8  w1t = term_tag(whnf);
            // Accept any WNF as "ready". Also accept TAG_TOP for ASSIGN:
            // ASSIGN handler resolves WNF view chains via dispatch.
            if (w1t != TAG_TEN && w1t != TAG_ERA && w1t != TAG_NUM &&
                w1t != TAG_LAM && w1t != TAG_SUP &&
                !(w1t == TAG_TOP && term_ext(frame) == UOP_ASSIGN)) { whnf = frame; continue; }
            heap_set(ctx, loc + 1, whnf);  // store arg1 result
            // 3-arg ops (WHERE, IFZ): reduce arg2 before firing
            u32 _uop1 = term_ext(frame);
            if (_uop1 == UOP_WHERE || _uop1 == UOP_IFZ) {
                Term a2 = heap_read(ctx, loc + 2);
                u8 a2t = term_tag(a2);
                if (a2t != TAG_TEN && a2t != TAG_ERA && a2t != TAG_NUM &&
                    a2t != TAG_CTR && a2t != TAG_ANY && a2t != TAG_LAM && a2t != TAG_SUP) {
                    PUSH(term_new(TAG_TOP2, (u8)_uop1, loc));
                    next = a2;
                    goto enter;
                }
            }
            Term top_frame = term_new(TAG_TOP, _uop1, loc);
            if (budget_exhausted || BUDGET_HIT()) { whnf = top_frame; continue; }
            Term r = thvm_interact(ctx, top_frame);
            if (r == top_frame) { whnf = top_frame; continue; }
            TRACE_STEP(top_frame, r);
            top_decref_inputs(ctx, loc, _uop1, r);
            next = r; goto enter;
        }

        if (ftag == TAG_TOP2) {
            // arg2 just finished. whnf = arg2's WNF result.
            u64 loc = term_val(frame);
            u8 w2t = term_tag(whnf);
            if (w2t != TAG_TEN && w2t != TAG_ERA && w2t != TAG_NUM &&
                w2t != TAG_LAM && w2t != TAG_SUP && w2t != TAG_CTR && w2t != TAG_ANY) { whnf = frame; continue; }
            heap_set(ctx, loc + 2, whnf);  // store arg2 result
            // All needed args ready — fire interact
            Term top_frame = term_new(TAG_TOP, term_ext(frame), loc);
            if (budget_exhausted || BUDGET_HIT()) { whnf = top_frame; continue; }
            Term r = thvm_interact(ctx, top_frame);
            if (r == top_frame) { whnf = top_frame; continue; }
            TRACE_STEP(top_frame, r);
            top_decref_inputs(ctx, loc, term_ext(frame), r);
            next = r; goto enter;
        }

        // Combinator frames — arg0 reduced, store and fire (or reduce arg1)
        if (ftag == TAG_CTR) {
            u64 floc = term_val(frame);
            heap_set(ctx, floc + 0, whnf);
            if (budget_exhausted || BUDGET_HIT()) { whnf = frame; continue; }
            Term r = thvm_interact(ctx, frame);
            if (r == frame) { whnf = frame; continue; }
            TRACE_STEP(frame, r);
            next = r; goto enter;
        }

        if (ftag == TAG_APP || ftag == TAG_DP0 || ftag == TAG_DP1 ||
            ftag == TAG_DSU || ftag == TAG_DDU || ftag == TAG_UDP ||
            ftag == TAG_AND || ftag == TAG_OR) {
            u64 floc = term_val(frame);
            heap_set(ctx, floc + 0, whnf);
            // APP-MAT: fun=MAT → need to reduce arg before firing
            if (ftag == TAG_APP && term_tag(whnf) == TAG_MAT) {
                Term arg = heap_read(ctx, floc + 1);
                u8 at = term_tag(arg);
                if (at == TAG_TEN || at == TAG_ERA || at == TAG_NUM ||
                    at == TAG_SUP || at == TAG_CTR || at == TAG_ANY) {
                    heap_set(ctx, floc + 1, arg);
                    goto fire_comb;
                }
                PUSH(term_new(TAG_MAT_1, 0, floc));
                next = arg;
                goto enter;
            }
            fire_comb:;
            if (budget_exhausted || BUDGET_HIT()) { whnf = frame; continue; }
            Term r = thvm_interact(ctx, frame);
            if (r == frame) { whnf = frame; continue; }
            TRACE_STEP(frame, r);
            next = r; goto enter;
        }

        // OP2/EQL: arg0 done. If SUP → fire (lazy distribution). Else → reduce arg1.
        if (ftag == TAG_OP2 || ftag == TAG_EQL) {
            u64 floc = term_val(frame);
            heap_set(ctx, floc + 0, whnf);
            if (term_tag(whnf) == TAG_SUP) {
                // SUP distribution: handler uses lazy arg1 for DUP
                if (budget_exhausted || BUDGET_HIT()) { whnf = frame; continue; }
                Term r = thvm_interact(ctx, frame);
                if (r == frame) { whnf = frame; continue; }
                TRACE_STEP(frame, r);
                next = r; goto enter;
            }
            // Non-SUP: reduce arg1 via dedicated frame
            Term a1 = heap_read(ctx, floc + 1);
            u8 a1t = term_tag(a1);
            if (a1t == TAG_TEN || a1t == TAG_ERA || a1t == TAG_NUM ||
                a1t == TAG_SUP || a1t == TAG_CTR || a1t == TAG_ANY) {
                // arg1 already WNF
                if (budget_exhausted || BUDGET_HIT()) { whnf = frame; continue; }
                Term r = thvm_interact(ctx, frame);
                if (r == frame) { whnf = frame; continue; }
                TRACE_STEP(frame, r);
                next = r; goto enter;
            }
            u8 ftag2 = (ftag == TAG_OP2) ? TAG_OP2_1 : TAG_EQL_1;
            PUSH(term_new(ftag2, term_ext(frame), floc));
            next = a1;
            goto enter;
        }

        // OP2_1/EQL_1/MAT_1: second arg done, reconstruct and fire
        if (ftag == TAG_OP2_1 || ftag == TAG_EQL_1 || ftag == TAG_MAT_1) {
            u64 floc = term_val(frame);
            u8 w1t = term_tag(whnf);
            if (w1t != TAG_TEN && w1t != TAG_ERA && w1t != TAG_NUM &&
                w1t != TAG_SUP && w1t != TAG_CTR && w1t != TAG_ANY) { whnf = frame; continue; }
            heap_set(ctx, floc + 1, whnf);
            // Reconstruct original frame
            u8 orig_tag = (ftag == TAG_OP2_1) ? TAG_OP2 :
                          (ftag == TAG_EQL_1) ? TAG_EQL : TAG_APP;
            u32 orig_ext = (ftag == TAG_OP2_1) ? term_ext(frame) : 0;
            Term orig = term_new(orig_tag, orig_ext, floc);
            if (budget_exhausted || BUDGET_HIT()) { whnf = orig; continue; }
            Term r = thvm_interact(ctx, orig);
            if (r == orig) { whnf = orig; continue; }
            TRACE_STEP(orig, r);
            next = r; goto enter;
        }

        whnf = frame; continue;
    }
    // Stack empty — whnf is the result
  }

    reduce_depth--;
    #undef PUSH
    #undef BUDGET_HIT
    #undef TRACE_STEP
    if (depth == 0 && ctx->step_budget == 0)
        whnf = reduce_net_quiesce(ctx, whnf);
    return whnf;
}

Term thvm_reduce_steps(TinyHVM *ctx, Term t, u32 max_steps) {
    ctx->step_budget = max_steps;
    ctx->steps_taken = 0;
    Term result = thvm_reduce(ctx, t);
    ctx->step_budget = 0;
    return result;
}

// ============================================================
// print.c — Debug printer
// ============================================================

static const char *tag_names[] = {
    "APP", "LAM", "VAR", "SUP", "DP0", "DP1", "ERA",
    "NUM", "REF", "OP2", "TEN", "TOP", "CTR",
    "BRI", "ANN", "DSU", "DDU", "INC",
    "EQL", "AND", "OR", "MAT", "ANY", "USP", "UDP"
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
