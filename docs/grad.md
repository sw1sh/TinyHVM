# GRAD: Gradient Semantics in TinyHVM

TinyHVM differentiates through its interaction-calculus term graph via a single
UOP: **`UOP_GRAD(y, target)`**.  Reducing a `GRAD` term produces a tensor of
`target.shape` holding `∂(Σⱼ y[j])/∂target` — reverse-mode VJP seeded with a
ones-cotangent on `y`.  A second UOP, `UOP_GRAD_FWD`, exists in
[`src/interact/grad.c`](../src/interact/grad.c) for forward-mode (JVP) but is
not yet part of the public API; the tracing hook in
[`src/schedule/_.c`](../src/schedule/_.c) detects both.

The implementation lives in [`src/interact/grad.c`](../src/interact/grad.c)
(rules) and [`src/inet/_.c`](../src/inet/_.c) (`thvm_grad` builder).

## Primitive

```c
Term thvm_grad(TinyHVM *ctx, Term y, Term target);
```

Constructs a `TAG_TOP` term with `UOP_GRAD` at a 2-slot heap cell:
`heap[loc+0] = y`, `heap[loc+1] = target`. The shape of the result is tracked
as `target.shape` in the shape table so downstream wrappers (ADD, NEG, …) can
infer their own output shapes and the scheduler can allocate a `raw_output_tid`.

There is **one** GRAD node kind. No separate forward projection; no DUP-shaped
pair; no side channel. The bundled multi-target helper `thvm_grad_bundle`
is a thin wrapper that emits `n_params` independent `GRAD` terms inside a
`CTR`.

## Contract

`GRAD(y, target)` reduces to a tensor `g` such that

```
g[i] = ∂ (Σⱼ y[j]) / ∂ target[i]
g.shape = target.shape
```

This is **reverse-mode** with an implicit ones-cotangent on `y`. Composing two
`GRAD` nodes (`GRAD(GRAD(f, x), x)`) yields the scalar-loss Hessian-vector
product, not a forward-mode JVP. See *JVP vs VJP* below.

### Why reverse mode with ones-seed

- Most neural-network training loops need `∂loss/∂θ` where `loss` is scalar.
  Seeding the cotangent with ones makes the primitive trivially match that
  shape without a caller-provided cotangent.
- For elementwise pipelines the ones-seed VJP coincides with the straightforward
  Leibniz rule, so elementwise rule writing stays simple (no explicit
  accumulator plumbing).
- For MM and other non-diagonal Jacobians the rule must explicitly emit the
  VJP expression (e.g. `ones(y.shape) @ bᵀ`); this is handled case-by-case.

### JVP vs VJP

| Mode | Seed | Result shape | Useful for |
|------|------|--------------|------------|
| **JVP** (forward-mode) | tangent on input (`target.shape`) | `y.shape` | forward tangent, Hessian-vector products via trick |
| **VJP** (reverse-mode) | cotangent on output (`y.shape`) | `target.shape` | `∂loss/∂θ` (standard training) |

`UOP_GRAD` is VJP with `cotangent = ones(y.shape)`. TinyHVM has no separate
JVP primitive today; higher-order derivatives are expressed as nested
`GRAD`s, which is VJP-of-VJP.

Higher-order note: for `f : ℝⁿ → ℝ`, `GRAD(f, x)` gives `∇f` of shape `n`.
Then `GRAD(sum(∇f), x)` gives a vector whose `i`-th entry is
`Σⱼ ∂²f / ∂xⱼ ∂xᵢ` — the Hessian row-sum, i.e. `H · 1`. To recover a clean
second derivative on a scalar `y = f(x)` where `x` is a single scalar,
nesting the primitives just works: the ones-seed collapses correctly because
`y` is scalar at every level.

## Rules

All rules live in [`src/interact/grad.c`](../src/interact/grad.c) and fire
from the `TAG_TOP` branch of `src/interact/_.c` when the TOP's uop is
`UOP_GRAD`. Each rule reads `y = heap[loc+0]`, dispatches on `y`'s tag, and
returns the gradient term.

### Leaf rules

- `y = TAG_TEN` (literal tensor): if `TEN id == target id` (after stripping
  `DP0`/`DP1`/`VAR` projections), emit `ones(target.shape)`; else
  `zeros(target.shape)`. Built via `GRAD_SCALAR_TEN` helper — a rank-matched
  shape-`[1,1,…]` TEN expanded to `target.shape`.
- `y = TAG_NUM`: constant → `zeros(target.shape)`.

### Elementwise (Leibniz forward-style, valid for diagonal Jacobians)

- `UOP_ADD`: `GRAD(a, t) + GRAD(b, t)`
- `UOP_SUB`: `GRAD(a, t) - GRAD(b, t)`
- `UOP_NEG`: `-GRAD(a, t)`
- `UOP_MUL`: `b * GRAD(a, t) + a * GRAD(b, t)` (Leibniz; requires
  `a.shape = b.shape = target.shape`)
- `UOP_DIV`: `(GRAD(a, t) / b) - a * GRAD(b, t) / b²`
- `UOP_EXP`: `exp(a) * GRAD(a, t)`
- `UOP_LOG`: `GRAD(a, t) / a`
- `UOP_SQRT`: `GRAD(a, t) / (2 √a)`
- `UOP_RELU`: `(a > 0) * GRAD(a, t)`
- `UOP_MAX`: `(a >= b) * GRAD(a, t) + (a < b) * GRAD(b, t)`
- `UOP_CMP`: constant zero (non-differentiable)
- `UOP_WHERE`: `where(cond, GRAD(a, t), GRAD(b, t))` (condition assumed
  constant w.r.t. target)

### Reductions

- `UOP_SUM(a, axes)`: recurse `GRAD(a, t)`, then `expand` to `a.shape`.
- `UOP_RMAX(a, axes)`: mask-expand with argmax indicator; requires three DUP
  copies of `a`.

### Movement ops

- `UOP_RESHAPE(a, shape)`: recurse on `a`; wrap with `reshape(da, a.shape)`
  only when `a.numel == target.numel` (identity target).
- `UOP_PERMUTE(a, axes)`: recurse; apply **inverse** permutation when
  `a.numel == target.numel`.
- `UOP_EXPAND(a, y_shape)`: for leaf operand `a` with `TEN id != target id`,
  short-circuit to `zeros(target.shape)`. Otherwise emit
  `sum_to_shape(ones(y_shape), target.shape)` — the reduction count reflects
  the broadcast factor.
- `UOP_SHRINK(a, pads)`: direct emit `pad(ones(y.shape), complementary_pads)`
  for the leaf-identity case.
- `UOP_PAD(a, pads)`: direct emit `shrink(ones(y.shape), unpad_pads)`.

### Matrix multiply (reverse-mode)

- `UOP_MM(a, b)`: genuine reverse-mode VJP. Detect which operand (after
  projection stripping) matches `target`; emit `ones(y.shape) @ bᵀ` for
  `a`-match or `aᵀ @ ones(y.shape)` for `b`-match. Neither ⇒ zeros. The
  Leibniz rule (`da @ b + a @ db`) was shape-incompatible with the
  `target.shape` contract and has been replaced.

### Pass-through

- `UOP_ASSIGN(dst, src)`: gradient flows through `dst` only.
- `UOP_DETACH`: emits zeros.

### Unimplemented / returns zeros

Any `TAG_TOP` uop not listed above falls through to `zeros(target.shape)`.
The previous design's fallthrough-to-`NUM(0.0)` rule is preserved but now
produces a shape-correct tensor rather than a bare scalar.

## Helper macros (grad.c)

```c
GRAD_SCALAR_TEN(v, shape)   // rank-matched shape-[1,..] TEN of v, expanded
GRAD_ONES_OF(shape)         // same, hardcoded value 1.0
GRAD_TERM_SHAPE(term, out)  // read TEN/TOP shape into `out` Shape lvalue
```

## DUP interaction

GRAD wraps its operands through normal IC DUP sharing. Where the rule needs
the same subterm twice (`a0, a1 = DUP(a)`), it emits an explicit
`thvm_dup(...)`. No special-case DUP handling for GRAD is needed in the
DUP combinator — it falls through via the HVM4-style SUB-bit substitution
(see *DUP + ERA sweeps* below).

## Known limitations

- **MUL-Leibniz shape mismatch.** `UOP_MUL` currently emits the elementwise
  Leibniz rule, which requires both operands to have shape `target.shape`.
  Pipelines where one operand is a `pad(x)` or `expand(x)` and the other is
  `target.shape` fail shape-check: `b * GRAD(a, t)` multiplies `b` (y-shape)
  by a `target.shape` gradient.  The general reverse-mode VJP for MUL —
  `sum_to_shape(b ⊙ upstream_grad_of_a, target.shape)` — needs an explicit
  cotangent-carrying primitive (see last bullet).  Affects
  `test_e2e_conv_like`.
- **In-VM recursive training loop.** `IFZ + REF + APP + ASSIGN + SEQ`
  recursion (for on-device training loops) currently returns NULL on readback.
  Not a grad-rule problem — reducer/scheduler integration work tracked in
  [`recursion.md`](recursion.md).  Affects `test_e2e_recursive_sgd`.
- **No explicit cotangent-carrying primitive.** A GRAD node that takes an
  incoming gradient on its aux port (`GRAD(y, cot)`) would let the rules
  express reverse-mode VJP naturally without assuming ones-seed.  This is the
  cleanest path to fix MUL/conv and is the current direction of the VJP
  refactor — see [`step_graph_ic_goal.md`](step_graph_ic_goal.md) for the
  target IC topology.

## Files

- [`src/interact/grad.c`](../src/interact/grad.c) — rule implementations.
- [`src/interact/_.c`](../src/interact/_.c) — `UOP_GRAD` dispatch under
  `TAG_TOP`, helper macros.
- [`src/inet/_.c`](../src/inet/_.c) — `thvm_grad` builder,
  `thvm_grad_bundle` multi-target wrapper.
- [`src/reduce/_.c`](../src/reduce/_.c) — `UOP_GRAD` added to reduce-trampoline
  direct-uop list and frame acceptance logic.
- [`test/test_grad_rules.m`](../test/test_grad_rules.m) — topology + E2E
  numeric tests.

---

# How Sharing Interacts with GRAD: HVM4-style SUB-bit DUP

TinyHVM's DUP cell uses HVM4-style substitution. This matters for gradient
correctness when the same compute subterm is consumed by multiple backward
arms (e.g. `(l1, l2) = DUP(MM(x, W))` then `loss = sum(exp(l1)) + sum(exp(l2))`;
expected `∂loss/∂W = 2 · xᵀ @ exp(logits)`).

Each DUP cell at heap slot `L` holds one of:

- **body** — DUP not yet fired.
- **SUB(value)** — either the other aux already fired normally and left *our*
  pre-computed clone here (HVM4 `heap_subst_cop` pattern), or the other aux's
  consumer was erased and the DUP collapsed to identity. Same reader: strip
  the SUB bit and return the stored term.
- **ERA** — DUP fully consumed.

### DP aux reduce

```
cell = heap[dup_loc]
if term_is_sub(cell):
    return term_strip_sub(cell)      # sibling already fired / collapsed
else:
    fire DUP⊳WHNF(cell) → (r0, r1)
    heap[dup_loc] = term_set_sub(my_side == 0 ? r1 : r0)
    return my_side == 0 ? r0 : r1
```

### ERA⊳DP (local GC sweep)

```
cell = heap[L]
if term_is_sub(cell):
    thvm_spawn_detached_era(ctx, term_strip_sub(cell))   # orphan clone
    heap[L] = ERA
else:
    heap[L] = term_set_sub(cell)     # first aux to drop; sibling takes as identity
```

No global tracking table, no side channel. `heap_set` is a plain write.

## Transparent projection for pure compute TOP

When a DP aux pulls and reads a non-effectful `TAG_TOP` at the DUP cell, both
auxes can share the same TAG_TOP handle without firing the DUP — the scheduler
materializes the TOP once and both auxes read the resulting TEN. Effectful
uops (`UOP_ASSIGN`, `UOP_KERNEL`, `UOP_EXEC`, `UOP_DETACH`, `UOP_GRAD`) still
go through the full fire + subst_cop path.

## Bounded recursion guards

- `fuse_walk_inner` depth-capped at 256 ([src/fuse/_.c](../src/fuse/_.c)).
- `thvm_eval_internal` followup-round loop capped at 1024 iterations
  ([src/schedule/_.c](../src/schedule/_.c)) — safety net for recursive loops.
