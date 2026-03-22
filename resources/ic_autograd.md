# IC-Native Autograd: Differentiation as Duplication

How interaction nets compute gradients, and why TinyHVM uses this instead of a tape.

## Where This Comes From

The connection between duplication and differentiation isn't something we invented. It comes from **Differential Linear Logic** by Thomas Ehrhard and Laurent Regnier (2003). The short version: in linear logic, you can't copy things for free — every copy is an explicit DUP. Ehrhard showed that this DUP operation corresponds exactly to taking a derivative. The math checks out and it's been studied for 20+ years.

What nobody's done is build a tensor runtime around it. That's what TinyHVM is.

**Papers:**
- Ehrhard & Regnier, "The differential lambda-calculus" (2003)
- Ehrhard & Regnier, "Differential interaction nets" (2006)
- Lafont, "Interaction combinators" (1997) — the foundation HVM builds on

---

## The Core Idea

A function's derivative tells you: if I wiggle the input, how does the output wiggle? In interaction nets, the way you "use an input twice" is by duplicating it with a DUP node. The DUP has to pass through every operation the input touches. As it passes through each op, the DUP-op interaction rule naturally computes the chain rule.

So: backward pass = DUP propagating through the forward graph.

---

## Quick Recap: What Are TOP Nodes?

`TOP` = **T**ensor **OP**eration. These are lazy computation nodes. When you write:

```c
Term z = thvm_op(ctx, UOP_MM, x, w);
```

Nothing executes yet. You get a `TAG_TOP` node sitting in the heap saying "I'm a matmul waiting to happen." The arguments `x` and `w` are stored in `HEAP[loc]` and `HEAP[loc+1]`.

Reduction evaluates these lazily — when the reducer hits a TOP whose inputs are realized tensors (TAG_TEN), it dispatches to the GPU backend.

---

## How DUP Produces Gradients

Say we have `z = mm(A, B)` and we want `∂z/∂A` and `∂z/∂B`. We inject a DUP node after `z`:

```
A ──→ [MM] ──→ z ──→ [DUP] ──→ z_fwd  (use the value)
B ──↗                    └──→ z_grad (get the gradient)
```

The DUP propagates backward into the MM node. The DUP-MM interaction rule says:

```
∂z/∂A = MM(grad, Bᵀ)
∂z/∂B = MM(Aᵀ, grad)
```

These are just two more MM operations — new TOP nodes that get reduced by the same forward machinery. There's no special "backward kernel." Backward is just more forward.

### All the Rules

**DUP-ADD**: `z = a + b`
- `∂z/∂a = grad` (just copy)
- `∂z/∂b = grad` (just copy)

DUP through add = copy the gradient to both branches. Trivial.

**DUP-MUL**: `z = a * b`
- `∂z/∂a = grad * b`
- `∂z/∂b = grad * a`

DUP through mul = multiply grad by the other input. Needs saved forward values.

**DUP-MM**: `z = mm(A[M,K], B[K,N])`
- `∂z/∂A = mm(grad[M,N], Bᵀ[N,K])` → `[M,K]` ✓
- `∂z/∂B = mm(Aᵀ[K,M], grad[M,N])` → `[K,N]` ✓

DUP through matmul = two matmuls with transposes. Shapes work out.

**DUP-RELU**: `z = relu(a)`
- `∂z/∂a = grad * (a > 0 ? 1 : 0)`

DUP through relu = mask the gradient where input was negative.

**DUP-SUM**: `z = sum(a)`
- `∂z/∂a = broadcast(grad, shape_of_a)`

DUP through sum = broadcast the scalar gradient back to the original shape.

---

## Why Not Just Use a Tape?

| | Tape (PyTorch-style) | IC-native (TinyHVM) |
|---|---|---|
| Data structure | Separate ops list | Same graph |
| Forward/backward | Two phases | One reduction |
| Shared subexpressions | Recomputed or manually cached | DUP-SUP annihilation handles it |
| Higher-order gradients | Tape of tapes (tricky) | Just DUP the DUP |
| Parallelism | Backward is sequential | Independent gradients reduce in parallel |

The IC approach doesn't need a separate backward pass. The gradients fall out of the same graph reduction. And if two losses share computation, DUP-SUP annihilation avoids recomputing shared parts — you get optimal sharing for free.

---

## What We'll Build in Phase 2

Practically, the reducer gets new cases:

```c
// In thvm_reduce, when DUP meets a realized TOP:
// 1. Look up what forward op produced this tensor
// 2. Apply the corresponding gradient rule
// 3. Return new TOP nodes (lazy backward ops)
```

Each tensor that needs gradients stores its "provenance" — which op and which inputs created it. When DUP reaches it, the provenance tells us which gradient rule to fire.

The backward ops (`mm(grad, Bᵀ)` etc.) are just normal TOP nodes. They reduce via the same GPU dispatch. Simple.

---

## References

1. Ehrhard & Regnier (2003). "The differential lambda-calculus." *TCS* 309(1-3), 1-41.
2. Ehrhard & Regnier (2006). "Differential interaction nets." *TCS* 364(2), 166-195.
3. Girard (1987). "Linear logic." *TCS* 50(1), 1-101.
4. Lafont (1997). "Interaction combinators." *Information and Computation* 137(1), 69-101.
