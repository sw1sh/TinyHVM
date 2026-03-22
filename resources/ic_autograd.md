# IC-Native Autograd: Differentiation as Duplication

How interaction nets naturally compute gradients, and why this is the right foundation for TinyHVM's backward pass.

## Origin

This is **not** a new idea — the theoretical connection between **duplication** and **differentiation** is a known result from **Differential Linear Logic** (DiLL), introduced by Thomas Ehrhard and Laurent Regnier in 2003.

**Key papers:**
- Ehrhard & Regnier, "The differential lambda-calculus" (2003) — [doi:10.1016/S0304-3975(03)00392-X](https://doi.org/10.1016/S0304-3975(03)00392-X)
- Ehrhard & Regnier, "Differential interaction nets" (2006) — extends interaction nets with differentiation
- Ehrhard, "An introduction to differential linear logic: proof-nets, models and antiderivatives" (2017)

**What's new in TinyHVM:** applying this theory to *tensor computation* and building an actual runtime that uses it. Nobody has done this before. The theory says DUP≈derivative; we're engineering it into a working ML system.

---

## The Idea in One Sentence

> In linear logic, a function's **derivative** is its **linear approximation** — and in interaction nets, the **DUP node** extracts exactly the linear part of a computation.

---

## Background: Linear Logic and Duplication

In **linear logic** (Girard, 1987), every resource must be used **exactly once** — no implicit copying or discarding. To use something twice, you must explicitly **duplicate** it with the `!` (bang/exponential) modality.

Interaction nets implement linear logic. In HVM/TinyHVM:
- **Variables are affine** — used at most once
- **DUP nodes** explicitly copy terms
- **ERA nodes** explicitly discard terms

This means: **every non-linear use is visible in the graph as a DUP node**.

---

## The Connection: DUP = ∂/∂x

Consider a function `f(x) = x * x`. In normal math, the derivative is `f'(x) = 2x`.

Now look at what happens when we **duplicate** the input in an interaction net:

```
                       ┌──→ a ──→ [MUL] ──→ y
           x ──→ [DUP]─┤
                       └──→ b ──→ [MUL] ──→ y
```

The DUP creates two copies of `x`. Each copy flows into one argument of MUL. The **interaction** between DUP and MUL must handle the fact that `x` is used twice.

In differential linear logic, the DUP-over-a-function interaction produces TWO things:
1. The **original function applied normally** (the forward pass)
2. The **derivative with respect to the duplicated variable** (the backward pass)

This isn't metaphorical — it's a provable structural property of the calculus.

---

## How It Works Mechanically in TinyHVM

### Setup: TAG_TOP nodes are tensor operations

`TOP` = **T**ensor **OP**eration. When you write:
```c
Term z = thvm_op(ctx, UOP_MM, x, w);  // lazy matmul
```

This creates a `TAG_TOP` node in the heap:
```
HEAP[loc]   = x   (first arg)
HEAP[loc+1] = w   (second arg)
Term z = TOP(uop=UOP_MM, loc)
```

It doesn't compute anything yet — it's a lazy graph node.

### Forward pass (Phase 1 — what we built)

When we reduce `z`, the reducer sees `TAG_TOP`, evaluates args to `TAG_TEN` (realized tensors), dispatches to GPU:
```
TOP(MM, loc) + [x:TEN, w:TEN] → dispatch gpu->op_mm → result:TEN
```

### Backward pass (Phase 2 — the new part)

Now suppose we want gradients. We **DUP** the computation:

```c
// Build: loss = sum(relu(mm(x, w) + b))
Term z   = thvm_op(ctx, UOP_MM, x, w);
Term zb  = thvm_op(ctx, UOP_ADD, z, b);
Term a   = thvm_op(ctx, UOP_RELU, zb, term_era());
Term loss = thvm_op(ctx, UOP_SUM, a, term_era());
```

To get gradients, we **duplicate** the loss through the graph. The DUP propagates backward through each TOP node, and each TOP node has a **DUP-TOP interaction rule** that produces the gradient:

### DUP-TOP Interaction Rules

When a DUP meets a TOP node during reduction, it doesn't just copy the TOP — it **splits into forward + backward**:

#### Rule 1: DUP-ADD

```
DUP ─── ADD(a, b) ──→ (forward: ADD(a, b), backward: (grad_a, grad_b))
```

Addition distributes: `∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1`.
So the DUP just **copies the incoming gradient** to both branches:

```
         ┌──→ grad_a = grad    (copy)
grad ───DUP
         └──→ grad_b = grad    (copy)
```

#### Rule 2: DUP-MUL

```
∂(a*b)/∂a = b,  ∂(a*b)/∂b = a
```

The DUP-MUL interaction produces:
```
         ┌──→ grad_a = MUL(grad, b)    (grad × other_input)
grad ───DUP
         └──→ grad_b = MUL(grad, a)    (grad × other_input)
```

The DUP needs access to the *saved forward inputs* — this is why we record them.

#### Rule 3: DUP-MM (Matmul)

```
z = MM(A, B)  where A:[M,K], B:[K,N], z:[M,N]
∂z/∂A = MM(grad, Bᵀ)     [M,N] × [N,K] → [M,K]
∂z/∂B = MM(Aᵀ, grad)     [K,M] × [M,N] → [K,N]
```

The DUP-MM interaction produces:
```
          ┌──→ grad_A = MM(grad, TRANSPOSE(B))
grad ────DUP
          └──→ grad_B = MM(TRANSPOSE(A), grad)
```

This creates **new TOP nodes** for the backward matmuls — they're just more lazy graph nodes that get reduced the same way.

#### Rule 4: DUP-RELU

```
∂relu(a)/∂a = (a > 0) ? 1 : 0
```

DUP-RELU produces:
```
grad ──→ grad_a = MUL(grad, CMP_GT(a, 0))    (masked passthrough)
```

Again, needs saved forward input `a`.

---

## Why This Is Better Than Tape-Based Autograd

| Aspect | Tape (PyTorch) | IC-Native (TinyHVM) |
|--------|---------------|---------------------|
| **Data structure** | Separate tape list | Same graph |
| **Forward/backward** | Two separate phases | Single reduction |
| **Memory** | Stores all intermediates | Optimal sharing via DUP |
| **Parallelism** | Sequential backward | All independent gradients reduce in parallel |
| **Higher-order derivs** | Manual tape-of-tape | Just DUP again (DUP-DUP = second derivative) |

The IC approach means:
- **No separate backward pass** — gradients are produced as natural byproducts of DUP interactions
- **Optimal sharing** — if two losses share a sub-computation, DUP-SUP annihilation avoids redundant gradient computation
- **Higher-order gradients for free** — DUP a DUP = second derivative, via the same rules

---

## Practical Implementation in TinyHVM

For Phase 2, we implement this as a **hybrid**:

1. **Forward pass**: reduce TOP nodes as before (dispatch to GPU)
2. **Recording**: each TOP reduction saves its inputs in the TensorMeta (needed for backward rules)
3. **Backward trigger**: `thvm_backward(ctx, loss)` injects a DUP at the loss, then reduces
4. **DUP-TOP rules**: implemented as new cases in the reducer

```c
case TAG_DP0:  // or TAG_DP1
case TAG_DUP: {
    // DUP meets a TOP → apply gradient rule
    Term inner = thvm_reduce(ctx, ...);
    if (term_tag(inner) == TAG_TOP) {
        u32 uop = term_ext(inner);
        // Apply the DUP-TOP interaction rule for this uop
        switch (uop) {
            case UOP_ADD: /* copy grad to both branches */ break;
            case UOP_MM:  /* mm(grad, Bt), mm(At, grad) */ break;
            case UOP_RELU: /* mul(grad, cmp(a, 0)) */     break;
        }
    }
}
```

The beauty: backward ops are just **more TOP nodes**. They get reduced by the same forward-pass machinery. There's no separate "backward kernel" — `mm` backward is just two more `mm` forward passes (with transposes).

---

## References

1. Ehrhard, T. & Regnier, L. (2003). "The differential lambda-calculus." *Theoretical Computer Science*, 309(1-3), 1-41.
2. Ehrhard, T. & Regnier, L. (2006). "Differential interaction nets." *Theoretical Computer Science*, 364(2), 166-195.
3. Girard, J.-Y. (1987). "Linear logic." *Theoretical Computer Science*, 50(1), 1-101.
4. Lafont, Y. (1997). "Interaction combinators." *Information and Computation*, 137(1), 69-101.
5. Mazza, D. (2017). "Linear logic and computation: a new point of view." — Survey connecting differential LL to computation.
6. Taelin, V. (2024). "The Interaction Calculus." — HVM4 specification (gist `903f20e0`).
