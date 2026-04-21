// wnf/_.c — HVM4-style eval/apply stack machine for IC reduction.
//
// See resources/plans/eval_apply_stack_machine.md.
//
// Stage 1 scope: handle GRAD/GRAD_FWD as eval-apply frames so nested
// GRAD fires inner-first mechanically.  For any tag the machine doesn't
// yet support, fall back to the existing thvm_reduce.
//
// Enter:  push continuation frame, descend into principal.  Atoms break
//         out of enter into apply.
// Apply:  pop frame; dispatch on (frame_tag × whnf_tag); rewrite.
//
// Stack is global per-thread, allocated lazily, grows on overflow.

#define WNF_STACK_INIT_CAP 4096

typedef struct {
    Term *buf;
    u32   cap;
    u32   pos;
} WnfStack;

static WnfStack g_wnf_stack = {0};

static void wnf_stack_push(Term t) {
    if (!g_wnf_stack.buf) {
        g_wnf_stack.buf = (Term *)malloc(sizeof(Term) * WNF_STACK_INIT_CAP);
        g_wnf_stack.cap = WNF_STACK_INIT_CAP;
        g_wnf_stack.pos = 0;
    }
    if (g_wnf_stack.pos + 1 >= g_wnf_stack.cap) {
        g_wnf_stack.cap *= 2;
        g_wnf_stack.buf = (Term *)realloc(g_wnf_stack.buf, sizeof(Term) * g_wnf_stack.cap);
    }
    g_wnf_stack.buf[g_wnf_stack.pos++] = t;
}

// Forward decl of the fallback to the existing reducer.
Term thvm_reduce(TinyHVM *ctx, Term t);

// ──────────────────────────────────────────────────────────────────────
// GRAD apply-phase dispatch.  Called when a GRAD frame is popped with
// whnf being the reduced y.  Produces the rewritten term; caller sets
// next = result and goto enter (if result needs further reduction) or
// whnf = result (if already in WHNF).
// ──────────────────────────────────────────────────────────────────────

// Leaf rule: GRAD(TEN y, TEN target).  y.tid == target.tid → ones, else ERA.
// Uses the same GRAD_SCALAR_TEN pattern as src/interact/grad.c.
static Term wnf_grad_apply_ten(TinyHVM *ctx, Term frame, Term y_whnf, int is_fwd) {
    u64 loc = term_val(frame);
    if (loc == 0 || loc + 1 >= ctx->heap_pos) return term_era();
    Term tgt = heap_read(ctx, loc + 1);
    if (term_tag(tgt) != TAG_TEN) {
        // Target not resolved — can't compute leaf here.  Fall back.
        return thvm_reduce(ctx, term_new(TAG_TOP, is_fwd ? UOP_GRAD_FWD : UOP_GRAD, loc));
    }
    u32 ytid = (u32)term_val(y_whnf);
    u32 ttid = (u32)term_val(tgt);
    if (ytid != ttid) { ctx->itrs++; return term_era(); }
    Shape osh = SHAPE(1);
    if (ytid < ctx->tensor_count) osh = ctx->tensors[ytid].view.shape;
    // Forward mode needs a dense materialized ones tensor (stride-0
    // EXPAND views break MM decomposition).
    if (is_fwd && osh.rank > 0) {
        u32 n = 1; for (u32 i = 0; i < osh.rank; i++) n *= osh.dims[i];
        f32 *buf = (f32 *)malloc(n * sizeof(f32));
        for (u32 i = 0; i < n; i++) buf[i] = 1.0f;
        Term out = thvm_tensor(ctx, buf, osh);
        free(buf);
        ctx->itrs++; return out;
    }
    // Reverse mode: rank-matched [1,1,..] scalar + EXPAND to target shape.
    Shape one_shape = {.rank = osh.rank};
    for (u32 i = 0; i < osh.rank; i++) one_shape.dims[i] = 1;
    if (one_shape.rank == 0) { one_shape.rank = 1; one_shape.dims[0] = 1; }
    f32 v = 1.0f;
    Term scalar = thvm_tensor(ctx, &v, one_shape);
    int is_scalar_shape = (osh.rank == 0) ||
                          (osh.rank == 1 && osh.dims[0] == 1);
    ctx->itrs++;
    return is_scalar_shape ? scalar : thvm_expand(ctx, scalar, osh);
}

// Whnf constants at the frame dispatch level.
static Term wnf_grad_apply_num_or_era(TinyHVM *ctx) {
    ctx->itrs++;
    return term_era();
}

// For compute TOPs as whnf, delegate to the existing grad rule in
// src/interact/grad.c by constructing a GRAD(whnf, target) TOP and
// running thvm_reduce.  Stage 1 transitional: gets the right rewrite
// with existing rule bodies without re-implementing them in the wnf
// machine.  Later stages will inline them.
static Term wnf_grad_apply_compute_top(TinyHVM *ctx, Term frame, Term y_whnf, int is_fwd) {
    u64 loc = term_val(frame);
    if (loc == 0 || loc + 1 >= ctx->heap_pos) return term_era();
    Term tgt = heap_read(ctx, loc + 1);
    // Build fresh GRAD(y_whnf, tgt) and reduce via the existing rule.
    u32 uop = is_fwd ? UOP_GRAD_FWD : UOP_GRAD;
    u64 nl = heap_alloc(ctx, 2);
    heap_set(ctx, nl + 0, y_whnf);
    heap_set(ctx, nl + 1, tgt);
    // Copy shape tracking if present.
    const View *v = st_get(loc);
    if (v) st_set(nl, v);
    return thvm_reduce(ctx, term_new(TAG_TOP, uop, nl));
}

// ──────────────────────────────────────────────────────────────────────
// Entry point: drive term to WHNF via enter/apply.
// ──────────────────────────────────────────────────────────────────────

static inline int wnf_is_atom(u8 tag) {
    return tag == TAG_TEN || tag == TAG_NUM || tag == TAG_ERA ||
           tag == TAG_LAM || tag == TAG_SUP || tag == TAG_ANY;
}

Term thvm_wnf(TinyHVM *ctx, Term term) {
    if (wnf_is_atom(term_tag(term))) return term;

    u32 base = g_wnf_stack.pos;
    Term next = term;
    Term whnf;

enter: {
    u8 tag = term_tag(next);

    // Atoms: transition to apply.
    if (wnf_is_atom(tag)) { whnf = next; goto apply; }

    // GRAD / GRAD_FWD: push as continuation frame, descend into y.
    if (tag == TAG_TOP) {
        u32 ext = term_ext(next);
        if (ext == UOP_GRAD || ext == UOP_GRAD_FWD) {
            u64 loc = term_val(next);
            if (loc == 0 || loc + 1 >= ctx->heap_pos) {
                whnf = next; goto apply;
            }
            wnf_stack_push(next);
            next = heap_read(ctx, loc + 0);
            goto enter;
        }
        // Other TOPs: compute TOPs (ADD/MUL/...) are IC-WHNF, KERNEL/FUSE
        // and ASSIGN need the old reducer.  Fall back.
        whnf = thvm_reduce(ctx, next);
        goto apply;
    }

    // Any other tag: fall back to the existing reducer.  We'll migrate
    // more tags into explicit enter/apply cases in later stages.
    whnf = thvm_reduce(ctx, next);
    goto apply;
}

apply: {
    while (g_wnf_stack.pos > base) {
        Term frame = g_wnf_stack.buf[--g_wnf_stack.pos];
        u8 ftag = term_tag(frame);
        if (ftag != TAG_TOP) {
            // Stage 1 only pushes TAG_TOP frames.  If something else
            // slipped in, panic-degrade: unwind and return whnf.
            break;
        }
        u32 fuop = term_ext(frame);
        if (fuop == UOP_GRAD || fuop == UOP_GRAD_FWD) {
            int is_fwd = (fuop == UOP_GRAD_FWD);
            u8 wtag = term_tag(whnf);
            if (wtag == TAG_TEN) {
                whnf = wnf_grad_apply_ten(ctx, frame, whnf, is_fwd);
                continue;
            }
            if (wtag == TAG_NUM || wtag == TAG_ERA) {
                whnf = wnf_grad_apply_num_or_era(ctx);
                continue;
            }
            if (wtag == TAG_TOP) {
                u32 wext = term_ext(whnf);
                if (wext == UOP_GRAD || wext == UOP_GRAD_FWD) {
                    // Should not happen: by construction, inner GRAD was
                    // driven to WHNF before this frame popped.  If it
                    // still is a GRAD, we're stuck — return original.
                    // Reinstall whnf into frame's y slot and return frame.
                    u64 fl = term_val(frame);
                    heap_set(ctx, fl + 0, whnf);
                    whnf = frame;
                    continue;
                }
                // Compute TOP as whnf: delegate to existing rule.
                next = wnf_grad_apply_compute_top(ctx, frame, whnf, is_fwd);
                // Result may need further reduction (if it's another GRAD
                // chain).  Re-enter.
                goto enter;
            }
            // Other whnf tags (DP/VAR stuck, etc.): reinstall and return.
            u64 fl = term_val(frame);
            heap_set(ctx, fl + 0, whnf);
            whnf = frame;
            continue;
        }
        // Non-GRAD frame: shouldn't happen in stage 1.  Reinstall.
        break;
    }
    return whnf;
}
}

// Cleanup on context free (call from thvm_free if desired).
static void wnf_stack_reset(void) {
    if (g_wnf_stack.buf) {
        free(g_wnf_stack.buf);
        g_wnf_stack.buf = NULL;
        g_wnf_stack.cap = 0;
        g_wnf_stack.pos = 0;
    }
    (void)wnf_stack_reset;
}
