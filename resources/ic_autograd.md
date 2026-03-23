# IC-Native Autograd: Differentiation as Duplication

How interaction nets compute gradients, and why TinyHVM uses this instead of a tape.

## Where This Comes From

The connection between duplication and differentiation comes from **Differential Linear Logic**
by Ehrhard and Regnier (2003). In linear logic every copy is an explicit DUP, and that DUP
operation corresponds exactly to taking a derivative. TinyHVM builds a tensor runtime around this.

**Papers:**
- Ehrhard & Regnier, "The differential lambda-calculus" (2003)
- Ehrhard & Regnier, "Differential interaction nets" (2006)
- Lafont, "Interaction combinators" (1997) — foundation HVM builds on

---

## The Core Idea

Backward pass = DUP node propagating through the forward graph.

Each TOP node in the graph stores **provenance** — which UOp produced it and which source tensor
IDs fed into it (`creator_op`, `src_ids[0]`, `src_ids[1]`). `thvm_grad(ctx, y, x)` creates a
lazy `UOP_GRAD` term. When reduced, the GRAD handler reads `y`'s provenance and applies the
corresponding gradient rule, recursing until it hits `x` (base case: return `gy`).

No tape, no backward loop — gradients are just more lazy TOP nodes that reduce through the same
forward engine.

---

## Gradient Rules (DUP-Op Interactions)

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

The `recording` flag is NOT the correct gate. Two tensors can both be on a gradient path
(`requires_grad = 1`) even when `recording = 0` (e.g., during backward itself). `requires_grad`
is the always-correct signal.

---

## Strided Reduce Correctness

`thvm_sum_axes` passes through the SUM reducer which reads the source buffer. If the source is
a non-contiguous **expand view** (stride=0 in broadcast dims), a flat `buf_read` reads the
wrong data — the backing buffer may be 1 element while the view claims N. Fix: both the
multi-axis and single-axis reduce paths now materialize non-contiguous inputs using view strides
before accumulating.

---

## Why Not Just Use a Tape?

| | Tape (PyTorch-style) | IC-native (TinyHVM) |
|---|---|---|
| Data structure | Separate ops list | Same graph |
| Forward/backward | Two phases | One reduction |
| Higher-order gradients | Tape of tapes (tricky) | Just GRAD the GRAD node |
| Implementation size | Separate backward engine | ~200 lines of GRAD handler |

---

## References

1. Ehrhard & Regnier (2003). "The differential lambda-calculus." *TCS* 309(1-3), 1-41.
2. Ehrhard & Regnier (2006). "Differential interaction nets." *TCS* 364(2), 166-195.
3. Girard (1987). "Linear logic." *TCS* 50(1), 1-101.
4. Lafont (1997). "Interaction combinators." *Information and Computation* 137(1), 69-101.
