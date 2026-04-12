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
