# TinyHVM Heap & Trampoline: How Recursive Inet Training Works

> **Status (2026-04-22):** This doc describes the **current**
> predicate-based trampoline (`src/schedule/_.c`, `src/reduce/_.c`).
> That reducer is scheduled for replacement with an HVM4-style
> enter/apply stack machine living in `src/wnf/_.c` — see
> [resources/plans/eval_apply_stack_machine.md](plans/eval_apply_stack_machine.md).
> The predicate-driven design sketched below doesn't cleanly support
> nested IC interactions (higher-order GRAD, GRAD⊳APP, nested WHERE);
> that's the reason for the planned refactor.

> **See also:** [docs/step_trampoline.md](../docs/step_trampoline.md) for the
> step-graph tracer, walker, and WL `TStep`/`TStepTrace`/`TDotGraph` API
> that build on top of the primitives documented here.

## Term Representation

Every term is a 64-bit packed word: `[SUB:1 | TAG:7 | EXT:18 | VAL:38]`

| Tag | What it is | WNF? |
|-----|-----------|------|
| `TAG_TEN` | Tensor value (val=tensor_id) | ✓ |
| `TAG_ERA` | Eraser (garbage collect) | ✓ |
| `TAG_NUM` | 32-bit number | ✓ |
| `TAG_LAM` | Lambda (val→heap loc) | ✓ |
| `TAG_SUP` | Superposition | ✓ |
| `TAG_APP` | Application (val→heap loc) | - |
| `TAG_VAR` | Variable (val→heap loc) | - |
| `TAG_REF` | Definition reference (lab=def_id) | - |
| `TAG_TOP` | Tensor operation (lab=uop, val→heap loc) | - |

## Heap Layout

The heap is a flat `Term[]` array. Compound nodes store their children at consecutive heap slots:

```
LAM at heap[loc]:    [loc] = var_slot (sub-flagged VAR if unbound)
                     [loc+1] = body term

APP at heap[loc]:    [loc] = function term
                     [loc+1] = argument term

TOP at heap[loc]:    [loc] = arg0 (left operand)
                     [loc+1] = arg1 (right operand / shape)
```

## Trampoline (`src/reduce/_.c`)

**Primitive:** `thvm_reduce_steps(ctx, t, max_steps)` is the worker. It sets
`ctx->step_budget = max_steps`, runs the two-phase loop below firing
interactions until budget or WNF, and returns. `thvm_reduce(ctx, t)` is
literally its fixed point — a one-liner that calls `thvm_reduce_steps` with
`max_steps = 0` (unlimited).

Two-phase loop, no recursion for the hot path:

### ENTER phase
Walk the head, pushing eliminators as stack frames:
- `TAG_TOP` → push frame, enter arg0 (strict left operand)
- `TAG_APP/VAR/REF` → dispatch to `thvm_interact`
- WNF (`TEN/ERA/NUM/LAM/SUP`) → jump to APPLY

### APPLY phase
Pop frames, dispatch on frame tag + WHNF result:
- `TOP frame` → arg0 done; enter arg1; when both done → fire the tensor op
- `APP frame + LAM` → beta-reduce: write arg (raw) to var slot, continue with body
- `APP frame + TEN/ERA` → discard TEN, continue with arg (**APP-TEN rule**)

## Key Interaction Rules

### APP-LAM (beta reduction)
```
APP(LAM(x, body), arg)  →  body[x := arg]
```
The arg is written **raw** (unreduced) to the var slot. It only gets reduced later when something encounters the VAR in enter phase and calls `thvm_reduce` on the stored value.

### APP-TEN (sequencing combinator)
```
APP(TEN, continuation)  →  continuation
```
When a tensor value ends up in function position (e.g. after an ASSIGN fires), discard it and return the argument. This enables **sequencing**: `APP(ASSIGN(...), next)` forces the ASSIGN via function-position enter, then APP-TEN hands off to `next`.

### REF (definition unfold + ALO clone)
```
REF(name)  →  term_clone(ctx->defs[name])
```
Each unfold deep-copies the definition body via `term_clone`, allocating fresh heap nodes and rebinding all LAM/VAR pairs. This is the ALO (allocate) mechanism that ensures each recursive step gets fresh nodes.

## Sequencing ASSIGN Operations

The challenge: how to force N side-effect operations (weight updates) in the correct order within a pure interaction net.

### The APP-TEN Pattern

Chain ASSIGNs in function position:
```
APP(aW1, APP(aB1, APP(aW2, APP(aB2, recursive_call))))
```

Reduction trace:
1. Enter outer APP → push frame → enter fun = `aW1` (TAG_TOP)
2. `aW1` is ASSIGN: trampoline enters arg0 (W1→TEN), enters arg1 (SGD→TEN), fires ASSIGN blit → returns TEN
3. Apply: APP frame + TEN → APP-TEN fires → continue with arg = `APP(aB1, ...)`
4. Repeat for aB1, aW2, aB2
5. Finally reach `APP(REF(train), m)` → unfold, clone, recurse

## IFZ — Control Flow via Scalar Tensor

```
IFZ(counter, zero_case, succ_lam)
```

A 3-arg TOP node. `counter` is a scalar `TEN([1])`:
- `counter == 0` → return `zero_case` (ERA = training done)
- `counter > 0`  → create `TEN(counter-1)`, return `APP(succ_lam, TEN(counter-1))`

No O(n) heap allocation — just a single `TEN(N)` scalar at entry, decremented each step.

## Single-step tracing

Every interaction the trampoline fires passes through the `TRACE_STEP(before, result)`
macro ([src/reduce/_.c:361](../src/reduce/_.c#L361)). This increments
`ctx->steps_taken`, writes to an optional `ctx->trace_buf`, and — when
step-graph tracing is active — triggers the per-interaction dump in
`thvm_trace_step_graph_session` ([src/schedule/_.c](../src/schedule/_.c)).

`thvm_interact`'s entry runs the pre-hook so `before`-redex metadata is
read from live heap *before* the interaction mutates it; `TRACE_STEP` fires
the post-hook after.

From Wolfram, `TStep[t]` and `TStepTrace[t]` wrap the budget-1 cycle with
walker-visible-change semantics (skip admin reductions that don't alter the
rendered tree); see [docs/step_trampoline.md](../docs/step_trampoline.md)
for the full API and the C ↔ WL correspondence table.
