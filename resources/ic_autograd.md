# IC-Native Autograd: Lazy Reverse-Mode AD in a Graph Reducer

How TinyHVM computes gradients, the theoretical background it draws from, and an honest
assessment of where the theory applies and where it doesn't.

## Theoretical Background

### Linear Logic (Girard 1987)

Linear logic treats values as **resources** used exactly once. Two structural rules that classical
logic takes for granted become explicit:

- **Contraction** (copying) → explicit DUP, guarded by the `!` (bang) exponential modality
- **Weakening** (discarding) → explicit ERA, also guarded by `!`

In interaction net terms: DUP and ERA are the combinators for these. Every copy is visible in the
graph. This is the foundation Lafont's interaction combinators (1997) build on, and what HVM uses.

### Differential Linear Logic (Ehrhard & Regnier 2003)

Added a **codereliction** `d` dual to `!`. Intuitively:

- `!A` = "as many copies of A as you want" (the usual exponential)
- `dA` = "one linear approximation of A" — a directional probe / perturbation

The derivative of a term `!A → B` is a map that takes one linear copy of `A` (a perturbation) and
produces the linear response in `B`. This is a **syntactic** operation on proofs/terms:

```
d(λx.t) · u = λx.(dt · u)     -- derivative distributes under abstraction
d(t u) · v = (dt · v) u + t v  -- Leibniz/product rule, syntactically
```

Key insight: differentiation and the exponential modality (which governs duplication) are **dual
operations** in a precise categorical sense. The codereliction `d` is the "one-shot linear probe"
dual to the "unlimited copies" `!`.

### Differential Interaction Nets (Ehrhard & Regnier 2006)

Reformulated in interaction nets: a DUP node applied to a function-like node produces two copies
plus a "differential" residual. The interaction rules encode the chain rule as graph rewrites.

### What the Theory Gives You: Forward-Mode AD

The differential lambda calculus is inherently **forward-mode**: you propagate a perturbation
(tangent) forward through the term. A tensor `T : !Float^n` differentiates to a tangent pair
`(value, tangent) : Float^n ⊗ Float^n`. A function `f : !Float^n → Float^m` differentiates to
`df : Float^n → Float^n ⊸ Float^m`.

**Backpropagation requires reverse-mode** — propagating cotangents backward. Getting this from the
differential lambda calculus requires a continuation-passing transform or linear negation
`(A ⊸ B)` transposed to `(B^⊥ ⊸ A^⊥)`, related to Girard's geometry of interaction (reversing
token flow through a net). This is substantially more involved than "DUP = derivative."

Relevant work bridging this gap:
- Abadi & Plotkin (2020), "A simple differentiable programming language"
- Brunel, Mazza, Pagani (2020), backpropagation as functor
- Hasegawa (2017), linear logic and geometry of interaction for reverse-mode AD
- Alvarez-Picallo & Lemay (2020), cartesian difference categories for AD

**Nobody has built a practical ML system using differential interaction nets for backprop.**

---

## What TinyHVM Actually Does

TinyHVM's autograd is **standard reverse-mode AD with lazy term representation in an IC heap**.

### The Mechanism

Each TOP node stores **provenance** — which UOp produced it and which source tensor IDs fed into it
(`creator_op`, `src_ids[0]`, `src_ids[1]`). `thvm_grad(ctx, y, x)` creates a lazy `UOP_GRAD`
term. When reduced, the GRAD handler reads `y`'s provenance and applies the corresponding chain
rule — emitting new `UOP_GRAD` terms as lazy TAG_TOP nodes. Those nodes reduce through the same
GRAD handler, working backward until the base case (`y == x`) is reached and `gy` is returned.

No DUP nodes appear in the backward pass. `GRAD3(a, da, x)` creates `TAG_TOP(UOP_GRAD)` nodes,
not `TAG_DP0`/`TAG_DP1`. The gradient computation is a hand-written chain-rule switch on
`creator_op`, not an IC graph rewrite.

### What IS IC-native about it

- **Single reduction engine**: gradients are lazy heap terms that reduce through the same
  `thvm_reduce()` as forward ops — no separate backward engine
- **Demand-driven**: only compute gradients that are actually forced (lazy evaluation)
- **O(1) provenance per tensor**: no tape with O(ops) memory overhead
- **REACHES pruning**: DFS to check if gradient can flow from `y` to `x`, efficiently skipping
  irrelevant branches

These are real engineering wins from "lazy evaluation of reverse-mode AD in a graph reducer."

### What is NOT IC-native about it

- No DUP-TOP interaction rules (DUP meeting a tensor op to produce derivative residuals)
- No ERA-TOP dead-code elimination via IC reduction
- No SUP-TOP distribution (cloning ops across superposition branches)
- No net reversal for reverse-mode — gradients are explicit chain-rule dispatch, not graph polarity
  reversal
- The GRAD handler is an imperative switch statement, not IC interaction rules

---

## Gradient Rules

| Op | Forward | Grad w.r.t. a | Grad w.r.t. b |
|---|---|---|---|
| ADD | `z = a + b` | `gy` | `gy` |
| SUB | `z = a - b` | `gy` | `-gy` |
| MUL | `z = a * b` | `gy * b` | `gy * a` |
| DIV | `z = a / b` | `gy / b` | `-gy * a / b²` |
| MM  | `z = mm(A,B)` | `mm(gy, Bᵀ)` | `mm(Aᵀ, gy)` |
| RELU | `z = relu(a)` | `gy * (a > 0)` | — |
| EXP | `z = exp(a)` | `gy * z` | — |
| LOG | `z = log(a)` | `gy / a` | — |
| SUM | `z = sum(a)` | `expand(gy, a.shape)` | — |
| EXPAND | `z = expand(a, shape)` | `sum_to_shape(gy, z.shape, a.shape)` | — |

**SUM backward invariant**: `gy` always has the **keepdims shape** — reduced axes are set to
1 rather than removed. So `expand(gy, input.shape)` is a direct broadcast, no reshape needed.
The old code incorrectly reshaped `gy` to all-ones before expanding, which failed when `gy`
was already a non-scalar keepdims shape from an outer SUM.

---

## Gradient Seed Shape

`thvm_grad(ctx, y, x)` seeds with `ones` shaped to match `y`. Previously the seed was always
a scalar `[1]` regardless of `y`'s shape. This caused `test_grad_mm` to abort: the MM backward
computes `mm(gy, Bᵀ)` which requires `gy` to be rank-2.

**Current behavior**: seed = `tensor_fill(ctx, y.shape, 1.0f)`.

---

## Fusion and Autograd Interaction

The fused `SUM(MUL(a, b))` kernel is a performance optimization that skips materializing the
intermediate MUL result. **This fusion is only safe when no MUL input requires a gradient.**

Gate: `if (!ma->requires_grad && !mb->requires_grad) { /* fuse */ }`

When either MUL input requires grad, the MUL node must materialize separately. The GRAD handler
for SUM needs to recurse through MUL to apply the chain rule; if MUL was fused away, the SUM's
`src_ids[0]` would point to the MUL's input (e.g. `diff`) instead of its output (`sq = diff²`),
causing the MUL backward to be skipped and losing the `2*diff` factor.

The `recording` flag still exists — it gates **provenance writes** at tensor creation
(`if (ctx->recording) { md->creator_op = ...; }`). What was removed is using `recording`
as the **fusion gate**. Previously the fused `SUM(MUL)` path had `goto skip_fused_mul_sum`
when `ctx->recording` — replaced with `!ma->requires_grad && !mb->requires_grad`. Two tensors
can be on a gradient path (`requires_grad = 1`) even when `recording = 0` (e.g. during
backward itself), so `requires_grad` is the correct fusion gate.

---

## Strided Reduce Correctness

`thvm_sum_axes` passes through the SUM reducer which reads the source buffer. If the source is
a non-contiguous **expand view** (stride=0 in broadcast dims), a flat `buf_read` reads the
wrong data — the backing buffer may be 1 element while the view claims N. Fix: both the
multi-axis and single-axis reduce paths now materialize non-contiguous inputs using view strides
before accumulating.

---

## Honest Comparison: Theory vs. Implementation

| Claim | Reality |
|---|---|
| "Backward = DUP propagating through forward graph" | Metaphor. Code is chain-rule dispatch on `creator_op` |
| "No tape, just graph reduction" | True. Gradients are lazy terms in the same heap. But the computation is standard reverse AD |
| "DUP = differentiation" | True in differential LL (codereliction). Not what happens in TinyHVM's gradient code |
| "Higher-order gradients via GRAD-of-GRAD" | Would work in principle. Closer to the DLL spirit. Untested |
| IC reduction for backprop | Future aspiration, not current reality |

### Where the theory could realistically contribute

**Higher-order differentiation**: if TinyHVM ever needs gradients of gradients (Hessian-vector
products, etc.), the differential lambda calculus gives a principled way to nest derivatives.
`∂(∂f/∂x · u)/∂x · v` is the Hessian applied to `v`, and the syntactic rules handle it
correctly without special-casing.

**True IC-native backprop** (future): implementing real DUP-TOP, ERA-TOP, SUP-TOP interaction
rules so backward becomes a genuine graph rewrite. This would require:

1. DUP-TOP interaction: DUP meeting a tensor op produces the op on both copies + differential
2. Linear type discipline: track which wires are linear vs exponential
3. Net reversal: reverse polarity of derivative subnet for reverse-mode

This remains a research problem, not an engineering task.

---

## Why Not Just Use a Tape?

| | Tape (PyTorch-style) | TinyHVM (lazy reverse AD in IC heap) |
|---|---|---|
| Data structure | Separate ops list | Same graph |
| Forward/backward | Two phases | One reduction |
| Memory | O(ops) tape | O(1) provenance per tensor |
| Higher-order gradients | Tape of tapes (tricky) | GRAD-of-GRAD (untested but natural) |
| Implementation size | Separate backward engine | ~200 lines of GRAD handler |

---

## References

1. Ehrhard & Regnier (2003). "The differential lambda-calculus." *TCS* 309(1-3), 1-41.
2. Ehrhard & Regnier (2006). "Differential interaction nets." *TCS* 364(2), 166-195.
3. Girard (1987). "Linear logic." *TCS* 50(1), 1-101.
4. Lafont (1997). "Interaction combinators." *Information and Computation* 137(1), 69-101.
5. Abadi & Plotkin (2020). "A simple differentiable programming language." *POPL*.
6. Brunel, Mazza, Pagani (2020). "Backpropagation in the simply typed lambda-calculus with linear negation." *POPL*.
7. Hasegawa (2017). "Linear logic, geometry of interaction, and reverse-mode AD."
8. Alvarez-Picallo & Lemay (2020). "Cartesian difference categories." *FoSSaCS*.
