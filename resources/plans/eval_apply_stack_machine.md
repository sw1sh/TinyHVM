# Refactor: eval/apply stack machine for IC reduction

**Implementation location:** `src/wnf/_.c` (new file, mirrors HVM4's
`clang/wnf/_.c` organization).  Existing reducer (`src/schedule/_.c`,
`src/reduce/_.c`) stays in place during migration and gets retired at
stage 4.

**Docs to update on completion:**
- `resources/heap_and_trampoline.md` — currently describes a
  two-phase ENTER/APPLY loop but only at a surface level; needs a full
  rewrite to describe the HVM4-style stack machine.
- New: `resources/wnf_stack_machine.md` — definitive reference for
  the finished implementation.

## Why

TinyHVM's current reducer (`src/schedule/_.c`) is predicate-driven: it
tries to *guess* ahead of time whether a term is "ready" to interact via
checks like `step_grad_y_ready`, `step_app_mat_arg_ready`,
`step_top_frame_arg0_ready`, etc.  Each agent contributes its own
readiness predicate.  When an agent's rule has more cases than its
predicate acknowledges (or fewer), interactions either fail to fire or
fire prematurely on unreduced operands.  Higher-order GRAD is the
latest symptom: inner GRAD stays unreduced inside outer GRAD's y slot
because the predicate-based trampoline can't cleanly interleave driving
principals with firing rules.

HVM4 (`/Users/swish/src/HVM4/clang/wnf/_.c`) avoids the problem entirely
by using a classic **enter/apply stack machine**:

- **enter**: push current term as a continuation frame, descend into
  its principal port.  Recurse.  Atoms and WHNF-ready tags break out
  of enter into apply.
- **apply**: pop frames one at a time.  Each pop has two tags: the
  frame's tag (what eliminator was waiting) and the WHNF tag (what
  arrived).  Dispatch on the pair.  Rewrites produce new `next` terms
  that re-enter, or updated `whnf` that continues the apply loop.

There are no readiness predicates.  By construction, when you pop a
frame in apply, the WHNF is actually WHNF, because enter drove the
principal all the way down before transitioning.

## Shape

```
wnf(term):
  stack = []
  next = term
  enter:
    switch term_tag(next):
      # Eliminators: push frame, descend
      case APP:  push(APP, loc); next = fun;    goto enter
      case DP0:  push(DP0, loc); next = body;   goto enter
      case OP2:  push(OP2, loc); next = lhs;    goto enter
      case MAT:  push(MAT, loc); next = scrut;  goto enter
      # TinyHVM extension: compute TOPs with "principal" = first slot
      case TOP(GRAD):    push(GRAD, loc); next = y;       goto enter
      case TOP(ADD):     # compute op — no IC rule; treat as WHNF leaf
                         whnf = next; goto apply
      case TOP(MUL/SUM/...):  whnf = next; goto apply
      # Atoms / ready tags
      case TEN, NUM, ERA, LAM, SUP, ...:  whnf = next; goto apply

  apply:
    while stack nonempty:
      frame = pop()
      switch (frame.tag, whnf.tag):
        case (APP, LAM):           next = beta(frame, whnf); goto enter
        case (APP, SUP):           whnf = commute(frame, whnf)
        case (APP, ERA):           whnf = era_app()
        case (DP0, TEN):           whnf = dup_ten(frame, whnf)
        case (DP0, LAM):           whnf = dup_lam(frame, whnf)
        case (DP0, SUP):           next = dup_sup(frame, whnf); goto enter
        # GRAD frame
        case (GRAD, TEN):          whnf = grad_leaf(frame, whnf)
        case (GRAD, ERA):          whnf = grad_era(frame, whnf)
        case (GRAD, TOP(MUL)):     next = grad_mul(frame, whnf);  goto enter
        case (GRAD, TOP(ADD)):     next = grad_add(frame, whnf);  goto enter
        case (GRAD, TOP(GRAD)):    # ← this cleanly never triggers in pure IC
                                   # because inner GRAD must have already
                                   # been applied before we got here.
        ...
```

The `(GRAD, TOP(GRAD))` case can't happen: if a GRAD was on the stack
as a frame, it was pushed *during enter* before its y could be
driven.  By the time apply pops the frame, whnf has already been
driven to a non-GRAD tag via a deeper push+apply cycle — meaning nested
GRADs fire inner→outer mechanically, no predicate needed.

## Migration plan

### Stage 0: add eval/apply primitives alongside existing reducer

Keep `thvm_reduce_step_collect` in place.  Add a new entry point
`thvm_wnf(ctx, term)` that implements enter/apply.  Initially supports
only a subset of tags (say TEN, NUM, ERA, APP, LAM, SUP, DP0, DP1,
TOP(GRAD), TOP(GRAD_FWD)), falling back to the old reducer for
anything else.

**Sanity check**: run `test_rewrite_rules` under both paths, assert
the same term-equality.

### Stage 1: migrate GRAD

Move GRAD dispatch into apply's `(GRAD, X)` branches.  Rule bodies
become pure structural rewrites — no recursive `GRAD_REC` macro,
because recursion happens by pushing new GRAD frames onto the stack
and re-entering.

Delete `step_grad_y_ready` — unnecessary.

**Verify**: all current GRAD rewrite_rules tests pass.  Higher-order
tests (`d2_square_rev_rev`, etc.) start passing.

### Stage 2: migrate compute TOPs

Handle ADD, MUL, SUM, EXPAND, etc. as "compute leaves" in the eval
machine — they're WHNF from IC's perspective (no rule fires on bare
ADD), but they can appear in apply-phase dispatches when driven by a
GRAD or similar upstream frame.

### Stage 3: migrate APP/LAM/SUP/DUP

Replace the inline tag checks in the old predict-next-redex with
frame-based dispatch.  This is the biggest chunk — touches every
core IC combinator.

### Stage 4: delete the old reducer

Remove `thvm_reduce_step_collect`, `thvm_step_predict_next_redex`,
`step_*_ready` predicates.  `thvm_reduce` becomes a thin wrapper
around `thvm_wnf`.  `thvm_eval` retains its pre/post structure
(reduce → fuse → dispatch) but the reduce phase uses the new machine.

### Stage 5: scheduler cleanup

The scheduler proper (fuse + dispatch) should only see compute TOPs
and KERNEL nodes.  Assert this as an invariant.  Any residual GRAD,
APP, etc. reaching phase 2 is a bug in the eval machine.

## Risks

- **Stack size.** HVM4 uses a fixed WNF_STACK.  TinyHVM should allocate
  one per context.  Growth policy: start at 4K entries, double on
  overflow.  Needs plumbing in ctx init/free.
- **Interaction count tracking.** Current code tracks `ctx->itrs` at
  every rule fire site.  Migrate carefully — the stack machine has
  fewer visible fire sites (pop in apply counts as one).
- **Debug trace / step graph.** Current step-trace hooks into
  `thvm_step_predict_next_redex`.  Need new hooks in enter/apply.
  Probably simpler: emit a step event per frame push and per apply
  dispatch.
- **Scheduling integration.** `ctx->dispatch_enabled` currently drives
  KERNEL materialization during reduce.  In the new machine, the
  reduce phase only sees IC agents; dispatch happens in a separate
  post-reduce phase.  Simpler separation, but may need fuse+dispatch
  adjustments.

## Estimated size

- Stage 0 (framework + stub): ~300 lines new, 0 lines deleted.
- Stage 1 (GRAD migration): ~200 lines new, ~400 lines deleted from
  `src/interact/grad.c`.
- Stage 2 (compute TOPs): ~100 lines.
- Stage 3 (core combinators): ~800 lines new, ~1200 lines deleted.
- Stage 4 (old reducer removal): net ~-2000 lines.
- Stage 5 (scheduler cleanup): ~100 lines.

Total: net line reduction (~-1500 lines), significant test-coverage
requirement across every stage.

## What triggers this

Right now: higher-order GRAD (2nd derivative, HVP, mixed partials)
structurally doesn't work.  1st-order VJP and JVP work because the
predicate-based reducer happens to get one level of nesting right.
Deeper and it breaks.

Beyond GRAD, the predicate mess is the root cause of several other
issues visible in `TodoWrite`:

- "proper GRAD⊳APP interaction rule" — APP-of-GRAD's rule doesn't fire
  reliably.
- "nested WHERE inner materializes before inner GRAD fires" — exactly
  the same predicate-ordering issue at a different agent.
- Any future 3-port agent (e.g. TRI, richer PRI-style combinators) will
  need its own predicate hack.

Fixing the underlying structure solves the class of bugs, not just
GRAD.
