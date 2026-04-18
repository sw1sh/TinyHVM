# GRAD Interaction Semantics

GRAD is the backward-mode automatic differentiation agent in TinyHVM's interaction net.
It propagates through the compute graph as a first-class IC node, following standard
interaction calculus rules.

## Node Structure

GRAD is a **2-port node**:

```
     y       gy
     |        |
   +----------+
   |   GRAD   |
   | d/d(x)   |
   +----------+
```

- **y** (principal port): the compute expression to differentiate through
- **gy** (auxiliary port): the incoming gradient (internal state, changes as GRAD propagates)
- **x** (metadata label): target pattern — NOT a port, shown only in node label as `d/d(x)`

GRAD does NOT allocate tensors on construction. The seed gradient is a scalar tensor
matching the loss dtype, created by `thvm_grad_seed_like()`.

## Label Matching (Differentiate vs Annihilate)

GRAD works like labeled nodes in HVM. When GRAD reaches a `TAG_TEN` leaf, the outcome
depends on whether the GRAD label matches the tensor:

| GRAD label | TEN leaf | Result |
|------------|----------|--------|
| matches (target tensor) | requires_grad=true | **Differentiate**: return gy (or accumulate) |
| doesn't match | requires_grad=true | **Annihilate**: ERA the gradient |
| any | requires_grad=false | **Annihilate**: ERA the gradient |
| any | has creator_op | **Walk**: continue backward through provenance |

Matching is resolved via the `GradTargetSet` — a table of `(term, tid, slot)` entries
keyed by GRAD heap location. The pattern stored in `x` can be:
- `TAG_TEN` (single target) — match one specific tensor
- `TAG_ANY` (wildcard) — match all requires_grad tensors

## GRAD Modes

Three accumulation strategies, selected at construction:

| Mode | Constructor | Behavior at TAG_TEN leaf |
|------|-------------|--------------------------|
| `GRAD_MODE_DROP` (0) | `thvm_grad()` | Erase both y and gy — no gradient retention |
| `GRAD_MODE_KEEP` (1) | `thvm_grad_multi_keep()` | Accumulate into fixed-arity CTR bundle |
| `GRAD_MODE_SLOT` (2) | `thvm_grad_multi()` | Side-effectful += via ASSIGN into slots |

## Interaction Rules

GRAD fires when its `y` port reaches WNF. Dispatch is by `y`'s tag:

### Constants and Erasure
- `TAG_ERA` (y=0): produce explicit zero gradient → `GRAD_ZERO(gy)`
- `TAG_NUM`: numeric constant → `GRAD_ZERO(gy)`

### Compute Ops (TAG_TOP)

GRAD interacts with lazy compute ops via backward rules:

**Unary gradients** (`UG(da)` macro — single GRAD on input):
- `UOP_NEG`: `UG(-gy)`
- `UOP_EXP`: `UG(gy * exp(at))`
- `UOP_LOG`: `UG(gy / at)`
- `UOP_SQRT`: `UG(gy / (2 * sqrt(at)))`
- `UOP_RELU`: `UG(gy * (at > 0))`
- `UOP_CAST`: `UG(cast(gy, src_dtype))`

**Binary gradients** (`BG(da, db)` macro — split gy via DUP, two GRAD branches):
- `UOP_ADD`: `BG(gy, gy)` — gradient passes through both
- `UOP_SUB`: `BG(gy, -gy)`
- `UOP_MUL`: `BG(gy * bt, gy * at)` — requires DUP split of both gy and operands
- `UOP_DIV`: `BG(gy / bt, -gy * at / (bt * bt))`
- `UOP_MAX`: `BG(gy * (at >= bt), gy * (at < bt))`

**Movement ops** (apply inverse transform to gy):
- `UOP_RESHAPE`: reshape gy back to input shape
- `UOP_PERMUTE`: inverse permute gy
- `UOP_EXPAND`: sum gy along expanded axes
- `UOP_SHRINK`: pad gy back
- `UOP_PAD`: shrink gy back

**Reduction ops**:
- `UOP_SUM`: reshape then expand gy to input shape
- `UOP_RMAX`: masked expansion based on argmax

**Non-differentiable**:
- `UOP_CMP`: `GRAD_ERASE(gy)` — comparison ops have zero gradient

### Superposition (TAG_SUP)
- Split into two GRAD branches with DP0/DP1 labels (DUP fan-out)

### DUP Auxiliary Ports
- **GRAD waits** when `y` is `DP0` or `DP1` — these are auxiliary ports
- GRAD only fires on principal-port targets (TAG_TOP, TAG_TEN, etc.)

## Key Macros

```
GRAD_RETURN(result)   — clear heap slots, restore flags, return to trampoline
GRAD_ZERO(gy)         — produce zero gradient (ERA or NUM(0) depending on mode)
GRAD_ERASE(gy)        — explicit erasure agent for non-differentiable ops
BG(da, db)            — binary gradient: DUP-split gy and operands
UG(da)                — unary gradient: single branch
BG_GY(da, db)         — binary variant needing only gy split (not operands)
GRAD_COMBINE          — sequence two GRAD branches (SLOT or KEEP mode)
```

## Construction API

```c
// Single target, drop mode
Term thvm_grad(ThvmCtx* ctx, Term y, Term x);

// Multi-target with accumulation slots
Term thvm_grad_multi(ThvmCtx* ctx, Term loss,
                     Term params[], Term grad_slots[], u32 n_params);

// Multi-target, keep mode (returns CTR bundle)
Term thvm_grad_multi_keep(ThvmCtx* ctx, Term loss,
                          Term params[], u32 n_params);
```

## Files

- `src/grad/_.c` — constructors, target set management, bundle operations
- `src/interact/grad.c` — interaction rules (GRAD_ENTRY dispatch)
- `src/tinyhvm.h` — GradTargetSet, GradTargetGroup, GradTargetEntry structs

---

# How Sharing Interacts with GRAD: the DUP/ERA Protocol

> This section documents the theory landed 2026-04-19 that finally resolved the
> MM+DUP backward grad-drop (after 20+ attempts). It explains *why* DUP, ERA,
> and GRAD compose correctly for shared subexpressions — the subtle invariants
> that break if you use the wrong rule.

## The Problem Shape

Consider a program that duplicates a compute node and reads both projections:

```
(l1, l2) = DUP(MM(x, W))
loss     = sum(exp(l1)) + sum(exp(l2))
```

Expected `∂loss/∂W = 2 · xᵀ @ exp(logits)` — both arms contribute.

The classical IC `DUP ⊳ TOP` **commutation rule** (creates per-arm copies
with sub-DUPs for every non-atom child) produces correct forward values, but
for backward it creates the **grad-drop bug**: each non-atom child becomes a
fresh sub-DUP whose port tracking has two slots; `DUP_STATE_RETURN` writes
the post-fire value only to the **latest** tracked slot, orphaning the other.
The orphaned arm's backward chain reads ERA and dies silently. `W` gets only
one arm's gradient; the other is lost.

## The Correct Theory (three cooperating rules)

The resolution is that `DUP` should not eagerly fire when its body is pure
compute — it is a **transparent projection**, not an effect. Three rule
changes cooperate:

### Rule 1: DP as transparent projection for pure compute TOP

When a `TAG_DP0` or `TAG_DP1` dispatch reads `val = heap[dup_loc]` and finds
a pure-compute `TAG_TOP` (any uop that is not `UOP_DETACH`, `UOP_ASSIGN`,
`UOP_KERNEL`, `UOP_EXEC`, or `UOP_GRAD`), **do not fire the DUP**. Simply
`RETURN_REDUCED(val)`. Both consumers read `body` through DP transparently
each time. The DUP node stays intact with port tracking intact.

```
DP0(dup_loc) reduces  →  heap_read(dup_loc)  →  val (TOP)
                         return val directly
                         — no port_slot writes
                         — no commutation
                         — no sub-cdup creation
```

This is [`src/interact/combinators.c`](../src/interact/combinators.c) in the
`case TAG_DP0: case TAG_DP1:` dispatch, just after reading `val`.

### Rule 2: Refcount-based DUP cleanup on `port_forget`

When does the shared body get garbage-collected? Answer: when the *last*
consumer drops its DP ref. Implementation is in
[`thvm_dup_port_forget`](../src/tinyhvm.h): after decrementing the port
count for a tracked slot, check if both `port_count[0] == 0` and
`port_count[1] == 0`. If yes, the DUP is unreferenced — clear the body to
`ERA`:

```c
if (e->port_count[0] == 0 && e->port_count[1] == 0) {
    ctx->heap[dup_loc] = term_era();  // last ref dropped, body is garbage
}
```

This is the "shared resource lifetime" side of refcount semantics. ERA on a
*reference* (e.g., `heap_set(cell, ERA)` on a consumer cell holding a DP ref)
only erases that one reference — `port_forget` removes the slot from
tracking. The body lives on as long as any consumer holds a DP. When the
final reference drops, body is ERA'd and the standard `ERA ⊳ TOP` cascade
cleans up its args. **This is NOT deferred garbage collection** — it's
refcounted cleanup at the moment of last-reference drop.

### Rule 3: `MAT-CTR` eager binding

A deadlock arises under rule 1 in programs with multi-target gradient + LAM
destructuring (`thvm_grad_multi_keep` → `MAT(lam, ctr)`):

- The CTR's children are unreduced compute TOPs (the grad formulas).
- The LAM body contains backward chains that reference `VAR`s only
  `MAT-CTR` can substitute.
- Pre-fix, `MAT-CTR` deferred on unreduced compute TOP children, waiting
  for the scheduler to materialize them to TENs first.
- But the scheduler can't build kernels for chains with unbound `VAR`s.
- Deadlock.

Fix: [`src/interact/combinators.c` APP-MAT-CTR rule](../src/interact/combinators.c)
— remove the "defer on non-movement TOP child" guard. The LAM body binds
compute TOPs directly and reduces them on demand. The `VAR`s get their
values, the scheduler can then make progress, kernels dispatch, everything
resolves.

## Why This Composes Correctly

The key invariant: **ERA on a DP reference does not destroy shared structure**.

- `heap_set(cell, ERA)` where `cell` held `DP0(dup)` → `port_forget` drops
  `cell` from DUP's port tracking. Count decrements. If 0/0, body ERA'd via
  rule 2. Otherwise, the DUP still has live consumers and the body stays.

- The existing `ERA ⊳ TAG_DP0/DP1` rule ([combinators.c:373](../src/interact/combinators.c#L373))
  handles the case where an active ERA reduces *against* a DP: it collapses
  the DUP, projecting the body to the other port's tracked slot. That is
  also refcount-correct: one consumer ERAing projects the value to the
  remaining consumer.

- The `ERA ⊳ TAG_TOP` cascade at [combinators.c:422](../src/interact/combinators.c#L422)
  fires only when the TOP is genuinely orphaned (e.g., the DUP body was
  cleared by rule 2 because all consumers released, and then something
  tries to ERA the orphan). At that point the cascade is correct — no
  other consumer references the args.

## GRAD Walks Compose

Under these rules, two backward arms of a `DUP(TOP)` both resolve through
the preserved body transparently. Each arm emits its own backward formula
referencing the shared structure; `GRAD_RETURN` clears only the GRAD
node's own `loc/loc+1`, never `y_loc` (the body's args). So multiple walks
coexist. The grads accumulate at parameters via `BUNDLE_ACCUM_ADD`.

## What Still Commutes

Effectful TOPs (`UOP_ASSIGN`, `UOP_KERNEL`, `UOP_EXEC`, `UOP_DETACH`) and
`UOP_GRAD` itself still take the classical commutation path. Their
semantics require per-arm independence: each arm of a DUP through an
ASSIGN must fire its own write, each KERNEL is an independent dispatch.

Non-TOP compound terms (`TAG_LAM`, `TAG_BRI`, `TAG_SUP`, `TAG_ANN`) also
still commute — they are structural compounds with scope-dependent
substitution that would be unsound to share atomically.

## Side Effect: MNIST Trains Better

Because both arms of every `DUP ⊳ MM` now contribute correctly, the W
parameter actually receives its gradient (pre-fix: W was frozen, only B
trained, 68% ceiling). Metal MNIST goes from 68.3% → 90.2% in 3 epochs
without any hyperparameter change.

## Bounded Recursion Guards (implementation detail)

Under this regime, two recursion paths need explicit caps:

- `fuse_walk_inner` depth-capped at 256
  ([src/fuse/_.c](../src/fuse/_.c)) — shared bodies can appear in walks
  before refcount cleanup collects them.
- `thvm_eval_internal` followup-round loop capped at 1024 iterations
  ([src/schedule/_.c](../src/schedule/_.c)) — converted from tail
  recursion to avoid C stack growth in programs that make genuine
  per-round progress without finalizing (e.g., recursive train loops).

Neither cap fires in healthy programs. They are safety nets.

## Known Limitation

`test_tiny_twoparam_grad_loop` — a recursive SGD loop using
`thvm_grad_multi_keep` inside a `REF`-based recursion — fails the
assertion. No crash, just ASSIGN doesn't fire because `update_w`'s SUB
reduces to ERA somewhere in the loop-state unfolding. Separate
investigation. The per-step tests `test_tiny_twoparam_noloop` and
`test_mm_dup_exp_sum` pass, as does MNIST which exercises a similar
MAT-CTR → SEQ → ASSIGN chain but without the recursive REF unfolding.

## Files Changed (the MM+DUP fix)

- [`src/interact/combinators.c`](../src/interact/combinators.c) — transparent DP, MAT-CTR eager
- [`src/tinyhvm.h`](../src/tinyhvm.h) — refcount cleanup in `thvm_dup_port_forget`
- [`src/fuse/_.c`](../src/fuse/_.c) — fuse_walk DP resolve + depth cap
- [`src/reduce/_.c`](../src/reduce/_.c) — quiesce HEAP-fire ERA guard
- [`src/schedule/_.c`](../src/schedule/_.c) — quiesce ROOT ERA guard + followup-round loop
- [`test/test_mm_dup_exp_sum.m`](../test/test_mm_dup_exp_sum.m) — regression test
