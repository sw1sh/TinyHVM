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

# How Sharing Interacts with GRAD: HVM4-style SUB-bit DUP

> **History.** The earlier "transparent DP + refcount body-clear" protocol
> was replaced 2026-04-20 with an HVM4-style substitution model after the
> refcount side channel proved fragile. The new scheme is local, uses no
> global tracking table, and removes ~250 lines of bookkeeping code.

## The Problem Shape

```
(l1, l2) = DUP(MM(x, W))
loss     = sum(exp(l1)) + sum(exp(l2))
```

Expected `∂loss/∂W = 2 · xᵀ @ exp(logits)` — both arms must contribute.

## Mechanism

Each DUP cell at heap slot `L` holds one of:

- **body** (plain term) — DUP not yet fired.
- **SUB(value)** — either
  (a) the other aux already fired normally and left *our* pre-computed clone
  here (HVM4 `heap_subst_cop` pattern), or
  (b) the other aux's consumer was erased and the DUP collapsed to
  identity — `value` is body itself, this surviving aux takes it as-is.
  Both cases share the same reader: strip the SUB bit and return the stored
  term.
- **ERA** — the DUP has been fully consumed/swept.

The SUB bit is the existing `term_set_sub` flag (bit 63), previously only
used on VAR binder slots. The DUP slot and the VAR slot never share
readers, so reusing the bit is safe.

### DP aux reduce (consumer pulls its DP)

Ported from [HVM4 `wnf/_.c` DP0/DP1 dispatch](../../HVM4/clang/wnf/_.c)
and [`heap_subst_cop`](../../HVM4/clang/heap/subst_cop.c):

```
cell = heap[dup_loc]
if term_is_sub(cell):
    # sibling already fired or collapsed-to-identity; take the value
    return term_strip_sub(cell)
else:
    # fire DUP⊳WHNF(cell). Produce (r0, r1) per the rule for cell's tag
    # (annihilate, commute, atom, TEN-incref, etc.).
    sibling = (my_side == 0) ? r1 : r0
    mine    = (my_side == 0) ? r0 : r1
    heap[dup_loc] = term_set_sub(sibling)    # sibling reads this later
    return mine
```

Implementation: [`src/interact/combinators.c` DP0/DP1
dispatch](../src/interact/combinators.c) + the `DUP_STATE_RETURN` macro.

### ERA⊳DP (garbage-collection-style, local)

When a consumer slot holding `DPx_L` is erased, the DUP cell at `L` needs
to handle "one of my aux consumers just went away":

```
cell = heap[L]
if term_is_sub(cell):
    # sibling already fired and left a clone for us, but we got erased
    # before consuming it. Emit a visible detached ERA on the orphan.
    thvm_spawn_detached_era(ctx, term_strip_sub(cell))
    heap[L] = ERA
else:
    # first aux to drop. Body survives for the sibling to take as identity.
    heap[L] = term_set_sub(cell)
```

This is local: touches only `heap[L]`. No sibling-slot lookup, no global
table. The detached ERA emitted in the SUB case keeps cleanup visible in
the step graph (no silent orphans).

## Why This Composes Correctly

- No `DupPortEntry` / `port_slot` / `port_count`: no global hash table,
  no stale entries from raw writes, no side channel.
- `heap_set` is a plain write — any code path (including scheduler
  rewrites, parallel workers, step-trace) can update slots without
  breaking DUP invariants.
- The DUP cell itself is the meeting point for both auxes, coordinating
  by the SUB bit in its body slot.
- Identity-collapse after one-aux ERA avoids the wasteful clone
  allocation entirely (the surviving aux reads body directly).

## Transparent Projection for Pure Compute TOP

The pre-fire early-return for pure compute TOP bodies is *kept*: when a
DP reduces and reads a non-effectful `TAG_TOP` at the DUP cell, both
auxes can share the same TAG_TOP handle without firing the DUP (the
scheduler materializes the TOP once and both auxes read the resulting
TEN). Effectful uops (`UOP_ASSIGN`, `UOP_KERNEL`, `UOP_EXEC`,
`UOP_DETACH`, `UOP_GRAD`) still go through the full fire + subst_cop
path.

## GRAD Walks Compose

Two backward arms of a `DUP(TOP)` both read body through the cell.
Whichever pulls first either (a) transparently projects for pure compute
TOP (both arms share), or (b) fires the DUP with HVM4 subst_cop (the
other arm reads the SUB-stored clone on its pull). `GRAD_RETURN` still
clears only the GRAD node's own `loc/loc+1`, never `y_loc`, so walks
coexist safely.

## TEN Refcount Notes

`DUP ⊳ TEN` does `tensor_incref(val)` before `DUP_STATE_RETURN(val, val,
val)`. The SUB-bit reader also `incref`s when the stripped value is TEN.
The SUB cell itself holds a reference until the DUP is ERA'd (via the
SUB-seen-on-second-ERA path above), which balances the two increfs against
the eventual releases by ERA⊳TEN on downstream consumers.

## Known Limitation

`test_tiny_twoparam_grad_loop` — a recursive SGD loop using
`thvm_grad_multi_keep` inside `REF`-based recursion — still fails the
assertion. No crash, but `update_w`'s SUB reduces to ERA somewhere in the
loop-state unfolding. The SUB-bit rewrite did not fix this; investigation
in the grad/scheduler path, not the DUP layer, is required.

Non-loop tests pass:
- [`test/test_mm_dup_exp_sum.m`](../test/test_mm_dup_exp_sum.m) — MM+DUP+EXP+SUM
- [`test/test_mm_dup_2arm.m`](../test/test_mm_dup_2arm.m) — MM+DUP+SUM (MNIST pattern)
- [`test/test_dup_chain.m`](../test/test_dup_chain.m) — 5-deep DUP chain with CTR-wrapped ERA arm

## Bounded Recursion Guards (implementation detail)

- `fuse_walk_inner` depth-capped at 256
  ([src/fuse/_.c](../src/fuse/_.c)) — shared bodies can appear in walks.
- `thvm_eval_internal` followup-round loop capped at 1024 iterations
  ([src/schedule/_.c](../src/schedule/_.c)) — safety net for recursive
  train loops.

## Files (SUB-bit rewrite, 2026-04-20)

- [`src/term/sub.c`](../src/term/sub.c) — added `term_strip_sub` helper.
- [`src/interact/combinators.c`](../src/interact/combinators.c) — DP
  dispatch SUB-check early return, `DUP_STATE_RETURN` rewritten to HVM4
  subst_cop shape, ERA⊳DP rewritten to local SUB-or-sweep.
- [`src/tinyhvm.h`](../src/tinyhvm.h) — deleted `DupPortEntry` struct and
  all `thvm_dup_port_*` helpers.
- [`src/heap/set.c`](../src/heap/set.c) — reverted to plain write.
- [`src/ctx/init.c`](../src/ctx/init.c) — removed `dup_ports` alloc/free.
- New tests: [`test/test_dup_chain.m`](../test/test_dup_chain.m),
  [`test/test_mm_dup_2arm.m`](../test/test_mm_dup_2arm.m).
