// Forward declarations for fusion (defined in grad/_.c)
static int is_elementwise(u32 uop);
static u32 fuse_or_reduce(TinyHVM *ctx, Term t);

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
// GRAD backward through deep networks (CNN) causes O(chain_length) recursion
// via RETURN_REDUCED in thvm_interact. Keep SLICE small to fit in TLS.
#define REDUCE_SLICE 256
#define REDUCE_MAX_DEPTH 512
static _Thread_local Term reduce_pool[REDUCE_SLICE * REDUCE_MAX_DEPTH];  // 1MB TLS
static _Thread_local int  reduce_depth = 0;

Term thvm_reduce(TinyHVM *ctx, Term root) {
    int depth = reduce_depth++;
    assert(depth < REDUCE_MAX_DEPTH);
    Term *stk = &reduce_pool[depth * REDUCE_SLICE];
    int  sp = 0;

    Term next = root;
    Term whnf;

    #define PUSH(f_)  do { assert(sp < REDUCE_SLICE); stk[sp++] = (f_); } while(0)

  enter: {
    u8 tag = term_tag(next);

    // Already WNF atoms → go directly to apply
    if (tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM ||
        tag == TAG_LAM || tag == TAG_SUP) { whnf = next; goto apply; }

    // TAG_TOP: try declarative rewrite rules first (fusion, etc.).
    // If no rule matches, reduce args depth-first then fire interact.
    if (tag == TAG_TOP) {
        // Rewrite rules: pattern-match and dispatch (fusion, etc.)
        Term rw = rewrite_apply(ctx, next);
        if (rw != next) { next = rw; goto enter; }

        // No rule matched: default depth-first reduction
        u64 loc = term_val(next);
        PUSH(next);
        next = heap_read(ctx, loc + 0);
        goto enter;
    }

    // Combinator tags: dispatch thvm_interact (handles APP/REF/DP0/DP1)
    {
        Term r = thvm_interact(ctx, next);
        if (r == next) { whnf = next; goto apply; }  // combinator WNF or stuck
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
            if (term_tag(whnf) != TAG_TEN && term_tag(whnf) != TAG_ERA) {
                heap_set(ctx, loc+0, whnf); whnf = frame; continue;
            }
            heap_set(ctx, loc + 0, whnf);  // store arg0 result

            // Check arg1: is it already ready?
            Term a1 = heap_read(ctx, loc + 1);
            // Any WNF in arg1 slot is "ready" (TEN for tensors, NUM for axes, ERA for optional)
            u8 a1t2 = term_tag(a1);
            if (a1t2 == TAG_TEN || a1t2 == TAG_ERA || a1t2 == TAG_NUM ||
                a1t2 == TAG_LAM || a1t2 == TAG_SUP) {
                // Both args ready — fire
                Term r = thvm_interact(ctx, frame);
                if (r == frame) { whnf = frame; continue; }
                top_decref_inputs(ctx, loc, term_ext(frame), r);
                next = r; goto enter;
            }
            // arg1 not ready: push TOP1 sentinel frame (val=loc, ext=loc so we can find it),
            // then enter arg1. When arg1 returns, TOP1 handler fires the rule.
            PUSH(term_new(TAG_TOP1, (u8)term_ext(frame), loc));  // sentinel: "waiting for arg1"
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
            Term top_frame = term_new(TAG_TOP, term_ext(frame), loc);
            Term r = thvm_interact(ctx, top_frame);
            if (r == top_frame) { whnf = top_frame; continue; }
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
                next = r; goto enter;
            }
            if (ftag == TAG_DP0 || ftag == TAG_DP1) {
                heap_set(ctx, floc + 0, whnf);
                Term r = thvm_interact(ctx, frame);
                if (r == frame) { whnf = frame; continue; }
                next = r; goto enter;
            }
            whnf = frame; continue;
        }
    }
    // Stack empty — whnf is the result
  }

    reduce_depth--;
    #undef PUSH
    return whnf;
}

// ============================================================
// print.c — Debug printer
// ============================================================

static const char *tag_names[] = {
    "APP", "LAM", "VAR", "SUP", "DP0", "DP1", "ERA",
    "NUM", "REF", "OP2", "TEN", "TOP", "CTR"
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
        default:
            if (term_val(t) || term_ext(t))
                printf("(ext=%u, val=%llu)", term_ext(t), (unsigned long long)term_val(t));
            break;
    }
}

// ============================================================
// api.c — High-level tensor API
// ============================================================

// Device registry — both backends are always linked in.
extern Backend cpu_backend;
#ifdef __APPLE__
extern Backend metal_backend;
#endif

