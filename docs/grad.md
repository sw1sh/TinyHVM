# GRAD2: Gradient Semantics in TinyHVM

TinyHVM differentiates through its interaction-calculus term graph via a single
UOP: **`UOP_GRAD2(y, target)`**. Reducing a `GRAD2` term produces a tensor of
`target.shape` holding the scalar-loss gradient `∂(sum y)/∂target` — i.e.
reverse-mode automatic differentiation seeded with a ones-cotangent on `y`.

The previous DUP-shaped `TAG_GF/TAG_GB` pair + `GRAD_ENTRY` macro system
(`thvm_grad`, `thvm_grad_multi`, `thvm_grad_multi_keep`, `BG`/`UG`/`GRAD_RETURN`)
was removed 2026-04. The active implementation lives in
[`src/interact/grad2.c`](../src/interact/grad2.c).

## Primitive

```c
Term thvm_grad_u(TinyHVM *ctx, Term y, Term target);
```

Constructs a `TAG_TOP` term with `UOP_GRAD2` at a 2-slot heap cell:
`heap[loc+0] = y`, `heap[loc+1] = target`. The shape of the result is tracked
as `target.shape` in the shape table so downstream wrappers (ADD, NEG, …) can
infer their own output shapes and the scheduler can allocate a `raw_output_tid`.

There is **one** GRAD node kind. No separate forward projection; no DUP-shaped
pair; no side channel. The bundled multi-target helper `thvm_grad_pair_bundle`
is a thin wrapper that emits `n_params` independent `GRAD2` terms inside a
`CTR`.

## Contract

`GRAD2(y, target)` reduces to a tensor `g` such that

```
g[i] = ∂ (Σⱼ y[j]) / ∂ target[i]
g.shape = target.shape
```

This is **reverse-mode** with an implicit ones-cotangent on `y`. Composing two
`GRAD2` nodes (`GRAD2(GRAD2(f, x), x)`) yields the scalar-loss Hessian-vector
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

`UOP_GRAD2` is VJP with `cotangent = ones(y.shape)`. TinyHVM has no separate
JVP primitive today; higher-order derivatives are expressed as nested
`GRAD2`s, which is VJP-of-VJP.

Higher-order note: for `f : ℝⁿ → ℝ`, `GRAD2(f, x)` gives `∇f` of shape `n`.
Then `GRAD2(sum(∇f), x)` gives a vector whose `i`-th entry is
`Σⱼ ∂²f / ∂xⱼ ∂xᵢ` — the Hessian row-sum, i.e. `H · 1`. To recover a clean
second derivative on a scalar `y = f(x)` where `x` is a single scalar,
nesting the primitives just works: the ones-seed collapses correctly because
`y` is scalar at every level.

## Rules

All rules live in [`src/interact/grad2.c`](../src/interact/grad2.c) and fire
from the `TAG_TOP` branch of `src/interact/_.c` when the TOP's uop is
`UOP_GRAD2`. Each rule reads `y = heap[loc+0]`, dispatches on `y`'s tag, and
returns the gradient term.

### Leaf rules

- `y = TAG_TEN` (literal tensor): if `TEN id == target id` (after stripping
  `DP0`/`DP1`/`VAR` projections), emit `ones(target.shape)`; else
  `zeros(target.shape)`. Built via `GRAD2_SCALAR_TEN` helper — a rank-matched
  shape-`[1,1,…]` TEN expanded to `target.shape`.
- `y = TAG_NUM`: constant → `zeros(target.shape)`.

### Elementwise (Leibniz forward-style, valid for diagonal Jacobians)

- `UOP_ADD`: `GRAD2(a, t) + GRAD2(b, t)`
- `UOP_SUB`: `GRAD2(a, t) - GRAD2(b, t)`
- `UOP_NEG`: `-GRAD2(a, t)`
- `UOP_MUL`: `b * GRAD2(a, t) + a * GRAD2(b, t)` (Leibniz; requires
  `a.shape = b.shape = target.shape`)
- `UOP_DIV`: `(GRAD2(a, t) / b) - a * GRAD2(b, t) / b²`
- `UOP_EXP`: `exp(a) * GRAD2(a, t)`
- `UOP_LOG`: `GRAD2(a, t) / a`
- `UOP_SQRT`: `GRAD2(a, t) / (2 √a)`
- `UOP_RELU`: `(a > 0) * GRAD2(a, t)`
- `UOP_MAX`: `(a >= b) * GRAD2(a, t) + (a < b) * GRAD2(b, t)`
- `UOP_CMP`: constant zero (non-differentiable)
- `UOP_WHERE`: `where(cond, GRAD2(a, t), GRAD2(b, t))` (condition assumed
  constant w.r.t. target)

### Reductions

- `UOP_SUM(a, axes)`: recurse `GRAD2(a, t)`, then `expand` to `a.shape`.
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

## Helper macros (grad2.c)

```c
GRAD2_SCALAR_TEN(v, shape)   // rank-matched shape-[1,..] TEN of v, expanded
GRAD2_ONES_OF(shape)         // same, hardcoded value 1.0
GRAD2_TERM_SHAPE(term, out)  // read TEN/TOP shape into `out` Shape lvalue
```

## DUP interaction

GRAD2 wraps its operands through normal IC DUP sharing. Where the rule needs
the same subterm twice (`a0, a1 = DUP(a)`), it emits an explicit
`thvm_dup(...)`. No special-case DUP handling for GRAD2 is needed in the
DUP combinator — it falls through via the HVM4-style SUB-bit substitution
(see *DUP + ERA sweeps* below).

## Known limitations

- **MUL-Leibniz shape mismatch.** `UOP_MUL` currently emits the elementwise
  Leibniz rule, which requires both operands to have shape `target.shape`.
  Pipelines where one operand is a `pad(x)` or `expand(x)` and the other is
  `target.shape` fail shape-check: `b * GRAD2(a, t)` multiplies `b` (y-shape)
  by a `target.shape` gradient. Fixing this requires a general reverse-mode
  VJP for MUL: `sum_to_shape(b ⊙ upstream_grad_of_a, target.shape)`. Affects
  `test_e2e_conv_like`.
- **In-VM recursive training loop.** `IFZ + REF + APP + ASSIGN + SEQ`
  recursion (for on-device training loops) currently returns NULL on readback.
  Not a grad-rule problem — reducer/scheduler integration work.
  Affects `test_e2e_recursive_sgd`.
- **No explicit cotangent-carrying primitive.** A GRAD node that takes an
  incoming gradient on its aux port (`GRAD(y, cot)`) would let the rules
  express reverse-mode VJP naturally without assuming ones-seed. This is the
  cleanest path to fix MUL/conv; it's the primary design work item for the
  next GRAD refactor.

## Files

- [`src/interact/grad2.c`](../src/interact/grad2.c) — rule implementations.
- [`src/interact/_.c`](../src/interact/_.c) — `UOP_GRAD2` dispatch under
  `TAG_TOP`, helper macros.
- [`src/inet/_.c`](../src/inet/_.c) — `thvm_grad_u` builder,
  `thvm_grad_pair_bundle` multi-target wrapper.
- [`src/reduce/_.c`](../src/reduce/_.c) — `UOP_GRAD2` added to reduce-trampoline
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
uops (`UOP_ASSIGN`, `UOP_KERNEL`, `UOP_EXEC`, `UOP_DETACH`, `UOP_GRAD2`) still
go through the full fire + subst_cop path.

## Bounded recursion guards

- `fuse_walk_inner` depth-capped at 256 ([src/fuse/_.c](../src/fuse/_.c)).
- `thvm_eval_internal` followup-round loop capped at 1024 iterations
  ([src/schedule/_.c](../src/schedule/_.c)) — safety net for recursive loops.
