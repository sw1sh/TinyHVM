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
#define REDUCE_MAX_DEPTH 32  // 32 depth × 128KB = 4MB TLS per thread
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
        tag == TAG_LAM || tag == TAG_SUP || tag == TAG_BRI) { whnf = next; goto apply; }

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
            if (term_tag(whnf) != TAG_TEN && term_tag(whnf) != TAG_ERA && term_tag(whnf) != TAG_NUM) {
                heap_set(ctx, loc+0, whnf); whnf = frame; continue;
            }
            heap_set(ctx, loc + 0, whnf);  // store arg0 result

            // Check arg1: is it already ready?
            Term a1 = heap_read(ctx, loc + 1);
            u8 a1t2 = term_tag(a1);

            // GRAD: fire with lazy gy (arg1) to keep backward chain lazy.
            // Chain rule formulas just wrap gy in new lazy ops — no reduction needed.
            // Base case / deposit explicitly reduce gy inside the GRAD handler.
            if (term_ext(frame) == UOP_GRAD) {
                Term r = thvm_interact(ctx, frame);
                if (r == frame) { whnf = frame; continue; }
                TRACE_STEP(frame, r);
                top_decref_inputs(ctx, loc, term_ext(frame), r);
                next = r; goto enter;
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
    "BRI", "ANN", "DSU", "DDU", "INC"
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

