# Interaction Reduction: Term, Heap, and the WNF Stack Machine

How TinyHVM actually fires interactions today — end to end, from the
64-bit term encoding up to the two-phase evaluator.  Companion to
[`step_trampoline.md`](step_trampoline.md) (which covers tracing) and
[`eval.md`](eval.md) (which covers the outer pipeline); this doc is
about the reducer itself.

## Layers at a glance

```
thvm_eval                        src/schedule/_.c       outer staged pipeline
 └─ thvm_reduce / thvm_normalize src/wnf + parallel/     WNF + deep-WHNF
     └─ thvm_reduce_budget        src/wnf/_.c            stack-machine loop
         └─ thvm_interact         src/interact/_.c       local rewrite rules
             └─ heap_read/set/take src/heap/*.c           term cells
```

`thvm_reduce_budget` is the single stack-machine driver.  Everything
else either wraps it (`thvm_reduce`, `thvm_normalize`,
`thvm_trace_step_graph_session`) or implements one rewrite rule
(`thvm_interact` and its per-tag subfiles).

## Term encoding

A `Term` is a `u64` packed as:

```
 63     62..56    55..38        37..0
 [SUB]  [TAG:7]   [EXT:18]     [VAL:38]
```

([`src/tinyhvm.h:30-52`](../src/tinyhvm.h#L30))

- **SUB** (1 bit) — "this is a resolved substitution, not a term".
  Readers test with `term_is_sub`, strip with `term_strip_sub`, flag
  with `term_set_sub` ([`src/term/sub.c`](../src/term/sub.c)).  Writers
  use it to mark a heap cell as "the other aux of a DUP already fired;
  this is the stored clone, not a body to reduce".
- **TAG** (7 bits) — the term kind: `TAG_APP`, `TAG_LAM`, `TAG_VAR`,
  `TAG_SUP`, `TAG_DP0`/`TAG_DP1`, `TAG_ERA`, `TAG_NUM`, `TAG_REF`,
  `TAG_OP2`, `TAG_TEN`, `TAG_TOP`, `TAG_CTR`, `TAG_BRI`, `TAG_ANN`,
  `TAG_DSU`/`TAG_DDU`, `TAG_INC`, `TAG_EQL`, `TAG_AND`, `TAG_OR`,
  `TAG_MAT`, `TAG_ANY`, `TAG_USP`/`TAG_UDP`, `TAG_SEQ`, `TAG_ALO`
  ([`src/tinyhvm.h:59-100`](../src/tinyhvm.h#L59)).
- **EXT** (18 bits) — a per-tag label:
  - `TAG_TOP` stores the `UOP_*` opcode here (MUL, SUM, GRAD, FUSE,
    KERNEL, …).
  - `TAG_CTR` stores the constructor arity.
  - `TAG_NUM` stores the `NUM_U32` / `NUM_F32` discriminator.
  - `TAG_SUP`/`TAG_DP0`/`TAG_DP1` store the DUP label.
  - `TAG_REF` stores the def name index.
- **VAL** (38 bits, 256 GB-addressable) — the heap location for
  compound terms, or the inline payload for atoms (tensor id for
  `TAG_TEN`, 32-bit immediate for `TAG_NUM`).

The encoding is deliberately HVM4-compatible; the 7-bit tag leaves
plenty of headroom for new agents.

## Heap

The heap is a single `Term *heap` on the context
([`src/tinyhvm.h:1102`](../src/tinyhvm.h#L1102)), indexed by `u64`
location.  Cells are allocated contiguously; compound terms store
their args at consecutive slots.

### Allocation

```c
u64 heap_alloc(TinyHVM *ctx, u64 w);
```

[`src/heap/alloc.c`](../src/heap/alloc.c) — bump-pointer.  Single
thread: `ctx->heap_pos += w`, return the old pos.  Multi-thread: each
thread owns a pre-reserved bank (`bank_next..bank_end`) and bumps
within it with no contention.  Overflow aborts with a stderr message;
cap is `HEAP_CAP = 1 << 25` terms (256 MB) by default
([`src/tinyhvm.h:409-410`](../src/tinyhvm.h#L409)).

### Read / write

```c
Term heap_read (TinyHVM *ctx, u64 loc);
void heap_set  (TinyHVM *ctx, u64 loc, Term t);
Term heap_take (TinyHVM *ctx, u64 loc);
```

`heap_read`/`heap_set` are plain loads/stores
([`src/heap/read.c`](../src/heap/read.c),
[`src/heap/set.c`](../src/heap/set.c)).
`heap_take` ([`src/heap/take.c`](../src/heap/take.c)) is the atomic
swap-with-zero primitive used by parallel DUP: the first aux to
arrive owns the cell and writes back a `SUB`-flagged clone for the
sibling.  Single-thread fast path is a plain read.

### Compound layout

Per-tag arg counts come from `reduce_net_term_arity`
([`src/reduce/_.c:99`](../src/reduce/_.c#L99)); per-UOP storage arity
from `thvm_uop_storage_arity`
([`src/interact/_.c:76`](../src/interact/_.c#L76)).  A few examples:

- `APP (f x)` → 2 slots: `[f, x]`
- `LAM λx.body` → 2 slots: `[body, bind_var]`
- `SUP {a, b}` → 2 slots: `[a, b]`
- `DP0/DP1 (val)` → 1 slot: `[body]` (shared with sibling)
- `TOP(MUL, a, b)` → 2 slots: `[a, b]`
- `TOP(KERNEL)` → 3 slots: `[left, right_or_meta, NUM(root_uop)]`
- `TOP(GRAD, y, tgt)` → 2 slots: `[y, tgt]`
- `TOP(WHERE/IFZ, c, t, e)` → 3 slots
- `CTR#N{x1..xn}` → N slots

Atoms (`TEN`, `NUM`, `ERA(0)`, `ANY`) carry everything in the word
and use no heap.

### SUB bit: DUP synchronization

The SUB bit turns heap cells into a tiny synchronization primitive.
Per [`src/wnf/_.c:742`](../src/wnf/_.c#L742), a `DP0`/`DP1` aux at
`dup_loc`:

1. Reads `cell = heap[dup_loc]`.
2. If `term_is_sub(cell)` → sibling already fired; strip the SUB and
   return the stored clone.
3. Otherwise push a `WNF_F_DUP` frame, drive the body to WHNF, and on
   return write the sibling's clone back with `term_set_sub`.

The ERA-DUP GC sweep in
[`src/wnf/_.c:1070+`](../src/wnf/_.c#L1070) uses the same flag as a
"first aux is gone; sibling can take the stored value as identity"
signal.  No side table, no refcount — just one bit per cell.

## Normal forms

`wnf_is_atom` ([`src/wnf/_.c:430`](../src/wnf/_.c#L430)) answers
"definitely WHNF, no further reduction possible":

```c
TAG_TEN | TAG_NUM | TAG_LAM | TAG_SUP | TAG_ANY
```

`TAG_ERA` is deliberately excluded: inert `ERA(0)` is WHNF but
**active** `ERA(val != 0)` drives the iterative eraser walker that
commutes ERA through one layer at a time
([`src/wnf/_.c:1070+`](../src/wnf/_.c#L1070)).  That's an in-place
loop, not a stack frame.

Other tags either become WHNF when no rule applies (`BRI`, `MAT`,
`USP`, fully-populated `CTR`) or push a frame and recurse down the
principal port.

## The WNF stack machine

[`src/wnf/_.c`](../src/wnf/_.c) is a classical eval/apply stack
machine (see
[`resources/plans/eval_apply_stack_machine.md`](../resources/plans/eval_apply_stack_machine.md)).
Two labels — `enter` and `apply` — drive everything.

### Signature

```c
Term thvm_reduce       (TinyHVM *ctx, Term term);              // budget = 0 = unbounded
Term thvm_reduce_budget(TinyHVM *ctx, Term term, u32 budget);  // fire ≤ budget rules
```

Both return a Term in WHNF (or a resumable partial term if the budget
ran out).  `thvm_reduce` is just `thvm_reduce_budget(ctx, term, 0)`
([`src/wnf/_.c:595`](../src/wnf/_.c#L595)).

### Frame record

```c
typedef struct {
    u8   kind;                    // WNF_F_APP, WNF_F_VJP, ...
    u8   flags;                   // per-kind bits (e.g. DUP side, is_fwd)
    Term t0, t1, t2, t3;          // per-kind payload (operands, tgt, ...)
} WnfFrame;
```

Stack state is three file-scope globals: `g_wnf_stack_buf` (dynamic),
`g_wnf_stack_cap`, `g_wnf_stack_pos`.  Capacity starts at 4096 frames
and doubles on overflow ([`src/wnf/_.c:256`](../src/wnf/_.c#L256)).
The stack is single-owner per reducer instance; re-entry (e.g. from
`thvm_eval` inside a rule) saves and restores `base = g_wnf_stack_pos`
so nested reductions drain their own frames without disturbing the
outer call.

### Frame kinds (selected)

From [`src/wnf/_.c:27-126`](../src/wnf/_.c#L27):

| Frame | Semantics |
|-------|-----------|
| `WNF_F_APP` | `(f x)`: drove `f` to WHNF; dispatch on `f`'s tag (LAM → β, SUP → commute, MAT → second phase, …). |
| `WNF_F_APP_MAT` | `APP` whose `f` reduced to `MAT`: now drive `arg` to WHNF before matching. |
| `WNF_F_SEQ` | `SEQ(a, b)`: drove `a` to WHNF; discard, continue with `b`. |
| `WNF_F_DUP` | `DP0/DP1`: drove body to WHNF; write sibling's clone back SUB-flagged, return own half. |
| `WNF_F_OP2_X` / `WNF_F_OP2_Y` | `OP2(opr, x, y)`: two-phase — WHNF of `x` first, then `y`, then compute. |
| `WNF_F_EQL_X` / `WNF_F_EQL_Y` | `(a === b)`: similar two-phase. |
| `WNF_F_AND_OR` | `AND`/`OR`: strict on left; short-circuit or return right. |
| `WNF_F_DSU` / `WNF_F_DDU` / `WNF_F_UDP` | Dynamic-label SUP/DUP and unordered DUP aux. |
| `WNF_F_GRAD*` | JVP descent frames — one per rule variant (ADD/SUB, NEG, MUL, DIV, EXP, LOG, SQRT, RELU, SUM, RESHAPE/PERMUTE, SHRINK/PAD, MAX, WHERE, MM_FWD, RMAX, …). Encoded as enter-phase → push frame(s) → descend; apply phase pops and builds the gradient expression. |
| `WNF_F_VJP*` | Reverse-mode VJP descent: initial frame, binary-op phase 1 (grad_a in hand, descend into b), phase 2 (combine). |

Each frame carries the minimum state needed to resume — typically the
parent heap loc, any operands already computed, and the original term
so a budget-exhaust or unhandled WHNF can rebuild the parent cleanly.

### Enter phase

`enter:` ([`src/wnf/_.c:647`](../src/wnf/_.c#L647)) walks down the
principal port.  Concretely, given `next`:

- **Atom** (`wnf_is_atom`) → set `whnf = next`, jump to `apply`.
- **`VAR`** → read the substitution cell.  If flagged → unbound, it's
  WHNF; if `ERA(payload)` → build an active-ERA and apply;
  if it substitutes to a pure `DETACH` TOP → force to TEN and apply;
  otherwise `next = sub`, loop.
- **`ANN`** → strip (transparent); `next = heap[loc]`, loop.
- **`BRI`** → WHNF until applied (like LAM); apply.
- **`CTR`** → peephole: unary `CTR#1{x} → x`; `CTR#N{ERA, xs}`
  shrinks to `CTR#(N-1){xs}`; otherwise WHNF.
- **`REF`** → on first reference, materialize the def book; then
  `thvm_alo_realize` one layer and loop.
- **`ALO`** → force one static→dynamic layer
  (`thvm_alo_force`) and loop.
- **`DP0`/`DP1`** → SUB shortcut as above; pure-compute-TOP
  transparent projection; otherwise push `WNF_F_DUP` and descend into
  the body.
- **`OP2`** → push `WNF_F_OP2_X`, descend into `x`.
- **`APP`** → push `WNF_F_APP`, descend into `f`.
- **`SEQ`** → push `WNF_F_SEQ`, descend into `a`.
- **`EQL`** → push `WNF_F_EQL_X`, descend into `x`.
- **`AND`/`OR`** → push `WNF_F_AND_OR`, descend into left.
- **`DSU`/`DDU`/`UDP`** → push their frame, descend into label/body.
- **`INC`** → transparent; fire, loop into slot 0.
- **`ERA(val != 0)`** → iterative eraser walker (no frame):
  commutes through one layer at a time, spawning detached ERAs for
  the non-continuation children.  Terminates at an atom or DP/VAR
  boundary ([`src/wnf/_.c:1070-1160`](../src/wnf/_.c#L1070)).
- **`TOP`** — the big switch.  Reduce operand slots needed by the
  rule (slot 0 always; slot 1 for `ASSIGN`; slots 1–2 for `WHERE`),
  snapshot the `(outer × inner)` UOP pair for the step-graph hook,
  call `thvm_interact`.  If the rule rewrote or mutated a slot, fire
  the per-interaction hook and loop; otherwise this TOP is WNF
  (opaque rule couldn't match) — apply.

### Apply phase

`apply:` ([`src/wnf/_.c:1233`](../src/wnf/_.c#L1233)) pops the top
frame and dispatches on `(frame.kind, term_tag(whnf))`:

- Frame kinds that only needed the subterm in WHNF (`OP2`, `EQL`,
  `AND_OR`, `DSU`, `DDU`) write the WHNF back to the right slot,
  possibly fire the actual rule, and either install a new `whnf` or
  descend into the next slot (pushing a `*_Y` frame).
- `WNF_F_APP` dispatches on `whnf` — LAM → β-reduce, MAT → transition
  to `WNF_F_APP_MAT`, SUP → commute, compute-TOP → app-of-top rewrite
  via `thvm_interact`, …
- `WNF_F_SEQ` discards `whnf`, sets `whnf = f.t1`, and re-enters.
- `WNF_F_DUP` writes the sibling clone SUB-flagged, returns the
  matching aux.
- `WNF_F_GRAD*` / `WNF_F_VJP*` rebuild gradient expressions from the
  received upstream cotangent or forward tangent, possibly pushing
  more frames to descend further.

If dispatch can't reduce further (unhandled `whnf` shape — e.g.
`OP2(NUM, TOP)`), the apply handler rebuilds the parent term and
leaves it as `whnf`.  The loop continues until `g_wnf_stack_pos ==
base` ([`src/wnf/_.c:1234`](../src/wnf/_.c#L1234)), at which point
`whnf` is the final answer.

### Budget and drain

```c
#define WNF_FIRED() do { \
    ctx->itrs++; \
    if (g_wnf_step_hook) g_wnf_step_hook(ctx); \
    if (g_wnf_budget > 0 && ++g_wnf_budget_fired >= g_wnf_budget) \
        g_wnf_budget_hit = 1; \
    ... \
} while (0)
```

([`src/wnf/_.c:231`](../src/wnf/_.c#L231))

Every rule fire goes through `WNF_FIRED()` (or the annotated
`WNF_FIRED_AT(rule, cursor, tgt)` variant for GRAD / VJP descent).
That increments `ctx->itrs`, invokes the optional per-step hook (used
by the step-graph tracer —
[`docs/step_trampoline.md`](step_trampoline.md)), and trips the
budget flag when exhausted.

Budget-hit paths short-circuit into `wnf_drain_stack`
([`src/wnf/_.c:489`](../src/wnf/_.c#L489)), which walks each
remaining frame and installs the current `whnf` back at the right
slot, rebuilding the parent term into a valid resumable Term.  That
result is a WHNF-shaped expression the caller can pass back to
`thvm_reduce_budget` later to continue from where it left off.

## Interaction dispatch

`thvm_interact` ([`src/interact/_.c:1018`](../src/interact/_.c#L1018))
is a single `switch (term_tag(t))` that fires exactly one local
rewrite and returns.  The rule sub-files are `#include`d directly
into the big switch, which the comments note is a deliberate monolith
for fast inlining:

```c
static Term thvm_interact(TinyHVM *ctx, Term t) {
    thvm_step_graph_on_pre_interaction(ctx, t);
    switch (term_tag(t)) {
        case TAG_TOP: {
            u32 uop = term_ext(t);
            u64 loc = term_val(t);
            #include "grad.c"         // UOP_GRAD / UOP_GRAD_FWD / UOP_GRAD_PIN
            #include "tensor_ops.c"   // UOP_ASSIGN, ew, reduce, view, FUSE, KERNEL, ...
        }
        #include "combinators.c"      // APP, LAM, SUP, DP0/DP1, REF, ...
        default: return t;
    }
}
```

The rules are written as **local rewrites**: read the relevant heap
slots, produce a fresh term (or mutate one slot in place — e.g. FUSE
rewriting a compute-TOP payload into a KERNEL), and return.  Driving
the result back to WHNF is the reducer's job; rules never recurse
into `thvm_reduce` for the same subterm.

Two patterns show up throughout:

- **Pure rewrite** — return a new Term; the reducer sees it's
  different from `next` and loops back to `enter`.
- **Slot mutation** — rewrite `heap[loc + i]` to a new inner term,
  return the *same outer* Term.  The reducer detects
  `_inner_post != _inner_snap` ([`src/wnf/_.c:960`](../src/wnf/_.c#L960))
  and treats the TOP as WHNF-reached for this step; the next enter
  pass will see the mutated slot.  FUSE's KERNEL-building rule uses
  this.

## Deep normalization

`thvm_reduce` only drives the **root** to WHNF.  The args at `root +
i` may still be non-WHNF (e.g. a `CTR` of unreduced kernels).
`thvm_normalize` ([`src/parallel/normalize.c`](../src/parallel/normalize.c))
closes that gap:

```c
Term thvm_normalize(TinyHVM *ctx, Term root);
```

1. Anchor `root` in a fresh heap slot.
2. Push that slot onto a work-stealing deque (`WsDeque` from
   [`src/parallel/wsdeque.c`](../src/parallel/wsdeque.c)).
3. Pop a slot; if unseen, `thvm_reduce` the term there to WHNF, write
   it back, and push each of its arg slots.
4. Repeat until the deque is empty.
5. Return the anchor's final value.

The seen-set (`g_norm_seen`, a flat `u8` array sized by max heap loc)
prevents revisiting shared slots.  Push order is right-to-left by
default (preserving lazy evaluation order); under
`THVM_STEP_GRAPH=1` the push order reverses so the principal port
(slot 0) pops first, matching the IC spec convention used by the
step-graph trace.

This replaces the older `reduce_net_quiesce` whole-heap sweep: only
root-reachable cells are visited, and each visit is a proper
`thvm_reduce` call rather than the ad-hoc single-fire the legacy code
used.  The serial owner model makes a future multi-thread extension
(steal-from-victim, mirroring `thvm_collapse_par` in
[`src/inet/_.c`](../src/inet/_.c)) a drop-in change.

## Kernel-readiness predicates

[`src/reduce/_.c`](../src/reduce/_.c) is now a thin helper file used
by the FUSE / KERNEL rules to answer:

- `reduce_top_has_era_arg(t)` — does any arg slot hold an ERA that
  would commute through this TOP?  Used by the ERA-peephole paths.
- `reduce_top_has_add_zero_arg(t)` — `ADD(x, 0)` / `ADD(0, x)`
  shortcut.
- `reduce_fuse_child_absorbable(child)` — can FUSE bind this child
  directly as a kernel leaf, or wrap it in another FUSE that will
  fire next?  Gates the FUSE rule's local decision.
- `reduce_fuse_payload_top_ready(t)` — is this compute TOP ready to
  be wrapped in a KERNEL?
- `reduce_top_direct_uop(u)` — uops that reduce directly at the WNF
  layer (`ASSIGN`, `GRAD`, `IFZ`, `LOG_PRINT`, `TODEVICE`, `CAST`,
  `DETACH`, `WHERE`, `EXEC`, `KERNEL`, `FUSE`) versus ones the FUSE
  layer handles.
- `reduce_net_term_arity(t)` — per-tag arg count (used by
  `thvm_normalize`'s enqueue loop and the debug printer).

The file is intentionally small (~200 LOC after the
`reduce_net_quiesce` refactor); the actual reducer lives in
[`src/wnf/_.c`](../src/wnf/_.c).

## Outer pipeline

`thvm_eval` ([`src/schedule/_.c`](../src/schedule/_.c)) composes
these into the tensor-aware pipeline described in
[`eval.md`](eval.md):

1. **Structural reduce** — `thvm_reduce` on the bare program.  Rewrites
   APP/LAM/IFZ/GRAD/MAT/CTR/DUP/SEQ; leaves compute TOPs
   (ADD/MUL/SUM/RESHAPE/…) as WNF lazy terms.
2. **FUSE pass** — wrap root in `FUSE(...)` and reduce again.  FUSE
   rewrites `FUSE(binary(a,b)) → KERNEL(FUSE(a), FUSE(b),
   NUM(binary))`, growing a structural KERNEL DAG.
3. **Global passes** — `thvm_run_global_passes` may rewrite the
   settled coarse graph and install `EXEC` triggers.
4. **Second reduce** — fires any `EXEC` triggers; `KERNEL → TEN`
   dispatch happens when reduction demands a materialized value (e.g.
   under ASSIGN, LOG_PRINT, or the final result).

Both the structural reduce and the FUSE reduce are the same WNF
stack machine.  The only difference is the root term (bare program
vs. `FUSE(...)` wrap) and whether `ctx->dispatch_enabled` is set to
allow `KERNEL → TEN` to fire.  `thvm_normalize` is called between
phases to push WHNF into every arg slot so subsequent rules see a
stable graph.

## Files

- [`src/wnf/_.c`](../src/wnf/_.c) — stack-machine trampoline
  (`thvm_reduce`, `thvm_reduce_budget`, frame handlers, step hook).
- [`src/interact/_.c`](../src/interact/_.c) — single-dispatch
  interaction switch; rules live in
  [`grad.c`](../src/interact/grad.c),
  [`tensor_ops.c`](../src/interact/tensor_ops.c),
  [`combinators.c`](../src/interact/combinators.c) (included into
  `thvm_interact`).
- [`src/parallel/normalize.c`](../src/parallel/normalize.c) —
  `thvm_normalize` (deep-WHNF root-reachable walker).
- [`src/parallel/wsdeque.c`](../src/parallel/wsdeque.c) —
  work-stealing deque used by normalize; also by the parallel
  collapse path in [`src/inet/_.c`](../src/inet/_.c).
- [`src/reduce/_.c`](../src/reduce/_.c) — kernel-readiness
  predicates consumed by FUSE / KERNEL rules; term-arity table.
- [`src/heap/`](../src/heap/) — `heap_alloc`, `heap_read`,
  `heap_set`, `heap_take`.
- [`src/term/`](../src/term/) — term word helpers (`term_new`,
  `term_tag`, `term_ext`, `term_val`, SUB bit).
- [`src/tinyhvm.h`](../src/tinyhvm.h) — bit layout, tag / UOP tables,
  `TinyHVM` context.
- [`resources/plans/eval_apply_stack_machine.md`](../resources/plans/eval_apply_stack_machine.md)
  — original design note for the stack machine.
