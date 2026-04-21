// wnf/_.c — HVM4-style eval/apply stack machine for IC reduction.
//
// Replacement for the predicate-based trampoline in src/schedule/_.c.
// Plan: resources/plans/eval_apply_stack_machine.md.
//
// Stage 0 (this file): scaffold.  Entry point thvm_wnf(ctx, t).  Handles
// atoms (TEN/NUM/ERA/LAM/SUP) directly.  Everything else falls back to
// the old thvm_reduce for now — we migrate tags one-by-one in subsequent
// stages.
//
// The machine's core loop:
//
//   enter:   push continuation frame, descend into principal.  Atoms
//            break out of enter into apply.
//   apply:   pop frame, dispatch on (frame_tag × whnf_tag) pair, rewrite.
//
// No readiness predicates.  By construction, when apply pops a frame,
// the current whnf is actually WHNF — because enter drove it there.

// Per-context stack.  Allocated lazily on first wnf invocation.
#define WNF_STACK_INIT_CAP 4096

typedef struct {
    Term *buf;
    u32   cap;
    u32   pos;
    u32   base;     // base offset for the current wnf invocation
} WnfStack;

static WnfStack g_wnf_stack = {0};

static void wnf_stack_ensure(u32 need) {
    if (!g_wnf_stack.buf) {
        g_wnf_stack.buf = (Term *)malloc(sizeof(Term) * WNF_STACK_INIT_CAP);
        g_wnf_stack.cap = WNF_STACK_INIT_CAP;
        g_wnf_stack.pos = 0;
        g_wnf_stack.base = 0;
    }
    while (need > g_wnf_stack.cap) {
        g_wnf_stack.cap *= 2;
        g_wnf_stack.buf = (Term *)realloc(g_wnf_stack.buf, sizeof(Term) * g_wnf_stack.cap);
    }
}

// Re-build a term from a partial descent when we bail out (budget, etc.).
// Walks remaining stack frames and re-installs `cur` into each frame's
// principal slot, producing the original (or updated) surface term.
// Stage 0: stub — just returns cur.  Real rebuild comes with stage 3.
static Term wnf_rebuild(TinyHVM *ctx, Term cur, u32 s_pos, u32 base) {
    (void)ctx;
    while (s_pos > base) {
        Term frame = g_wnf_stack.buf[--s_pos];
        (void)frame;
        // TODO stage 3: per-tag rebuild (APP: reinstall fun slot; DP0/DP1:
        // reinstall body slot; OP2: reinstall x slot; etc.).
    }
    return cur;
}

// Forward decl of the fallback to the existing reducer.
// Tags not yet migrated to the stack machine route through this.
Term thvm_reduce(TinyHVM *ctx, Term t);

// Tag classification.  A tag is "whnf-atom" if no rule fires on it as
// a principal — it just sits.  These are the values that pop out of
// enter directly into apply.
static inline int wnf_is_atom(u8 tag) {
    return tag == TAG_TEN || tag == TAG_NUM || tag == TAG_ERA ||
           tag == TAG_LAM || tag == TAG_SUP || tag == TAG_ANY;
}

// Main entry.  Drives `term` to WHNF using the enter/apply machine.
// Stage 0 behavior: for atoms, return immediately.  For anything that
// would push a frame, fall back to thvm_reduce for now.
Term thvm_wnf(TinyHVM *ctx, Term term) {
    // Early exit: already WHNF.
    u8 tag = term_tag(term);
    if (wnf_is_atom(tag)) return term;

    // Stage 0: no frames pushed yet — delegate everything else.
    //
    // Stages 1+ will populate the enter/apply loop with per-tag
    // descent/dispatch logic, eventually handling GRAD, APP/LAM/SUP/DUP,
    // compute TOPs, etc.  The scaffold below shows the intended shape.
    //
    // Enter/apply skeleton (stub):
    //
    //   wnf_stack_ensure(128);
    //   u32 base = g_wnf_stack.pos;
    //   u32 s_pos = base;
    //   Term next = term;
    //   Term whnf;
    //   enter:
    //     switch (term_tag(next)) {
    //       case TAG_APP: {
    //         u64 loc = term_val(next);
    //         g_wnf_stack.buf[s_pos++] = next;   // push continuation
    //         next = heap_read(ctx, loc + 0);     // descend into fun
    //         goto enter;
    //       }
    //       case TAG_TOP: {
    //         u32 uop = term_ext(next);
    //         if (uop == UOP_GRAD || uop == UOP_GRAD_FWD) {
    //             u64 loc = term_val(next);
    //             g_wnf_stack.buf[s_pos++] = next;
    //             next = heap_read(ctx, loc + 0); // descend into y
    //             goto enter;
    //         }
    //         // Compute TOPs: IC-WHNF, dispatch handled elsewhere.
    //         whnf = next; goto apply;
    //       }
    //       ...atoms... whnf = next; goto apply;
    //     }
    //   apply:
    //     while (s_pos > base) {
    //       Term frame = g_wnf_stack.buf[--s_pos];
    //       switch (term_tag(frame)) {
    //         case TAG_APP:
    //           switch (term_tag(whnf)) {
    //             case TAG_LAM: next = wnf_app_lam(ctx, frame, whnf); goto enter;
    //             case TAG_SUP: whnf = wnf_app_sup(ctx, frame, whnf); continue;
    //             ...
    //           }
    //         case TAG_TOP: { // GRAD frame
    //           u32 uop = term_ext(frame);
    //           if (uop == UOP_GRAD || uop == UOP_GRAD_FWD) {
    //             next = wnf_grad_apply(ctx, frame, whnf); goto enter;
    //           }
    //         }
    //       }
    //     }
    //     return whnf;
    //
    // For now, route to old reducer.  Swap in real logic stage by stage.
    return thvm_reduce(ctx, term);
}

// Cleanup on context free.
static void wnf_stack_reset(void) {
    if (g_wnf_stack.buf) {
        free(g_wnf_stack.buf);
        g_wnf_stack.buf = NULL;
        g_wnf_stack.cap = 0;
        g_wnf_stack.pos = 0;
        g_wnf_stack.base = 0;
    }
}
