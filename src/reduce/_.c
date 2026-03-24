Term thvm_reduce(TinyHVM *ctx, Term root) {
    Term *stk = (Term *)malloc(FRAME_CAP * sizeof(Term));
    int   sp  = 0;

    Term next = root;
    Term whnf;

    #define PUSH(f_)  do { assert(sp < FRAME_CAP); stk[sp++] = (f_); } while(0)

  enter: {
    u8 tag = term_tag(next);

    // Already WNF atoms → go directly to apply
    if (tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM ||
        tag == TAG_LAM || tag == TAG_SUP) { whnf = next; goto apply; }

    // Memo hit for TAG_TOP
    if (tag == TAG_TOP) {
        u64 loc = term_val(next);
        if (loc < ctx->reduce_memo_size && ctx->reduce_memo[loc]) {
            whnf = ctx->reduce_memo[loc]; goto apply;
        }
        // Push node as frame, enter arg-slot 0 (strict left operand)
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
            if (term_tag(a1) == TAG_TOP) {
                u64 al = term_val(a1);
                if (al < ctx->reduce_memo_size && ctx->reduce_memo[al])
                    { a1 = ctx->reduce_memo[al]; heap_set(ctx, loc+1, a1); }
            }
            // Any WNF in arg1 slot is "ready" (TEN for tensors, NUM for axes, ERA for optional)
            u8 a1t2 = term_tag(a1);
            if (a1t2 == TAG_TEN || a1t2 == TAG_ERA || a1t2 == TAG_NUM ||
                a1t2 == TAG_LAM || a1t2 == TAG_SUP) {
                // Both args ready — fire
                Term r = thvm_interact(ctx, frame);
                if (r == frame) { whnf = frame; continue; }
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

    free(stk);
    #undef PUSH

    // Prefer memo for root
    if (term_tag(root) == TAG_TOP) {
        u64 loc = term_val(root);
        if (loc < ctx->reduce_memo_size && ctx->reduce_memo[loc])
            return ctx->reduce_memo[loc];
    }
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

