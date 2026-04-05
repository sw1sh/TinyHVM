// Forward declarations for fusion (defined in fuse/_.c)
static int is_elementwise(u32 uop);
static u32 fuse_or_reduce(TinyHVM *ctx, Term t);

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

    #define PUSH(f_)  do { if (sp >= REDUCE_SLICE) { fprintf(stderr, "SP_OVERFLOW sp=%d depth=%d\n", sp, depth); fflush(stderr); assert(0); } stk[sp++] = (f_); } while(0)
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
            reduce_depth--; return (result); \
        } \
    } while(0)

  enter: {
    u8 tag = term_tag(next);

    // Already WNF atoms → go directly to apply
    if (tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM ||
        tag == TAG_LAM || tag == TAG_SUP || tag == TAG_BRI ||
        tag == TAG_MAT || tag == TAG_ANY || tag == TAG_USP) { whnf = next; goto apply; }

    // TAG_TOP: compute ops are WNF (lazy). Non-compute ops reduce args and fire.
    if (tag == TAG_TOP) {
        u32 _uop = term_ext(next);
        // Compute ops: WNF only if args are also WNF (TAG_TEN, ERA, NUM, lazy TAG_TOP).
        // If any arg needs reduction (GRAD, ASSIGN, APP), process normally.
        if (!ctx->dispatch_mode &&
            _uop != UOP_ASSIGN && _uop != UOP_GRAD && _uop != UOP_IFZ &&
            _uop != UOP_LOG_PRINT && _uop != UOP_TODEVICE && _uop != UOP_WHERE &&
            _uop != UOP_FUSING &&
            _uop != UOP_RESHAPE && _uop != UOP_PERMUTE && _uop != UOP_EXPAND &&
            _uop != UOP_SHRINK && _uop != UOP_PAD) {
            u64 _loc = term_val(next);
            int _lazy = 1;
            // Check if any arg needs reduction (GRAD, APP, ASSIGN, etc.)
            for (u32 _ai = 0; _ai < 2 && _lazy; _ai++) {
                Term _a = heap_read(ctx, _loc + _ai);
                if (term_tag(_a) == TAG_TOP) {
                    u32 _au = term_ext(_a);
                    if (_au == UOP_GRAD || _au == UOP_ASSIGN || _au == UOP_IFZ || _au == UOP_FUSING)
                        _lazy = 0;
                } else if (term_tag(_a) == TAG_APP) _lazy = 0;
            }
            if (_lazy) { whnf = next; goto apply; }
        }
        // Non-compute: reduce args then fire interact (existing behavior)
        Term rw = rewrite_apply(ctx, next);
        if (rw != next) { next = rw; goto enter; }

        u64 loc = term_val(next);
        PUSH(next);
        next = heap_read(ctx, loc + 0);
        goto enter;
    }

    // Combinator tags: push frame and reduce arg0 for tags that need it.
    if (tag == TAG_APP || tag == TAG_DP0 || tag == TAG_DP1 ||
        tag == TAG_OP2 || tag == TAG_EQL || tag == TAG_AND || tag == TAG_OR ||
        tag == TAG_DSU || tag == TAG_DDU || tag == TAG_UDP) {
        u64 loc = term_val(next);
        PUSH(next);
        next = heap_read(ctx, loc + 0);
        goto enter;
    }

    {
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

            // Check arg1: is it already ready?
            Term a1 = heap_read(ctx, loc + 1);
            u8 a1t2 = term_tag(a1);

            // GRAD: skip arg1 (gy stays lazy), reduce arg2 (x) via TAG_TOP2.
            // Chain rule formulas just wrap gy in new lazy ops — no reduction needed.
            if (term_ext(frame) == UOP_GRAD) {
                // arg0 (y) is done. arg1 (gy) stays lazy. Now reduce arg2 (x).
                Term a2 = heap_read(ctx, loc + 2);
                u8 a2t = term_tag(a2);
                if (a2t == TAG_TEN || a2t == TAG_ERA || a2t == TAG_NUM || a2t == TAG_CTR) {
                    // arg2 already WNF — fire immediately
                    Term r = thvm_interact(ctx, frame);
                    if (r == frame) { whnf = frame; continue; }
                    TRACE_STEP(frame, r);
                    top_decref_inputs(ctx, loc, term_ext(frame), r);
                    next = r; goto enter;
                }
                // arg2 not ready: push TAG_TOP2, reduce arg2
                PUSH(term_new(TAG_TOP2, (u8)term_ext(frame), loc));
                next = a2;
                goto enter;
            }

            // Any WNF in arg1 slot is "ready"
            if (a1t2 == TAG_TEN || a1t2 == TAG_ERA || a1t2 == TAG_NUM ||
                a1t2 == TAG_LAM || a1t2 == TAG_SUP) {
                // Both args ready — fire
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
            // Accept any WNF as "ready"
            if (w1t != TAG_TEN && w1t != TAG_ERA && w1t != TAG_NUM &&
                w1t != TAG_LAM && w1t != TAG_SUP) { whnf = frame; continue; }
            heap_set(ctx, loc + 1, whnf);  // store arg1 result
            // 3-arg ops (WHERE, IFZ): reduce arg2 before firing
            u32 _uop1 = term_ext(frame);
            if (_uop1 == UOP_WHERE || _uop1 == UOP_IFZ) {
                Term a2 = heap_read(ctx, loc + 2);
                u8 a2t = term_tag(a2);
                if (a2t != TAG_TEN && a2t != TAG_ERA && a2t != TAG_NUM &&
                    a2t != TAG_CTR && a2t != TAG_LAM && a2t != TAG_SUP) {
                    PUSH(term_new(TAG_TOP2, (u8)_uop1, loc));
                    next = a2;
                    goto enter;
                }
            }
            Term top_frame = term_new(TAG_TOP, _uop1, loc);
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
                w2t != TAG_LAM && w2t != TAG_SUP && w2t != TAG_CTR) { whnf = frame; continue; }
            heap_set(ctx, loc + 2, whnf);  // store arg2 result
            // All needed args ready — fire interact
            Term top_frame = term_new(TAG_TOP, term_ext(frame), loc);
            Term r = thvm_interact(ctx, top_frame);
            if (r == top_frame) { whnf = top_frame; continue; }
            TRACE_STEP(top_frame, r);
            top_decref_inputs(ctx, loc, term_ext(frame), r);
            next = r; goto enter;
        }

        // Combinator frames (APP, DP0, DP1)
        {
            u64 floc = term_val(frame);
            if (ftag == TAG_APP) {
                heap_set(ctx, floc + 0, whnf);
                Term r = thvm_interact(ctx, frame);
                if (r == frame) { whnf = frame; continue; }
                TRACE_STEP(frame, r);
                next = r; goto enter;
            }
            if (ftag == TAG_DP0 || ftag == TAG_DP1) {
                heap_set(ctx, floc + 0, whnf);
                Term r = thvm_interact(ctx, frame);
                if (r == frame) { whnf = frame; continue; }
                TRACE_STEP(frame, r);
                next = r; goto enter;
            }
            whnf = frame; continue;
        }

        // Combinator frames — arg0 reduced, store and fire (or reduce arg1)
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
                    at == TAG_SUP || at == TAG_CTR) {
                    heap_set(ctx, floc + 1, arg);
                    goto fire_comb;
                }
                PUSH(term_new(TAG_MAT_1, 0, floc));
                next = arg;
                goto enter;
            }
            fire_comb:;
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
                Term r = thvm_interact(ctx, frame);
                if (r == frame) { whnf = frame; continue; }
                TRACE_STEP(frame, r);
                next = r; goto enter;
            }
            // Non-SUP: reduce arg1 via dedicated frame
            Term a1 = heap_read(ctx, floc + 1);
            u8 a1t = term_tag(a1);
            if (a1t == TAG_TEN || a1t == TAG_ERA || a1t == TAG_NUM ||
                a1t == TAG_SUP || a1t == TAG_CTR) {
                // arg1 already WNF
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
                w1t != TAG_SUP && w1t != TAG_CTR) { whnf = frame; continue; }
            heap_set(ctx, floc + 1, whnf);
            // Reconstruct original frame
            u8 orig_tag = (ftag == TAG_OP2_1) ? TAG_OP2 :
                          (ftag == TAG_EQL_1) ? TAG_EQL : TAG_APP;
            u32 orig_ext = (ftag == TAG_OP2_1) ? term_ext(frame) : 0;
            Term orig = term_new(orig_tag, orig_ext, floc);
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
    #undef TRACE_STEP
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

