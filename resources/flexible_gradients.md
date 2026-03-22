# Flexible Gradients: From Tape to Function Transforms

Why TinyHVM should evolve from tape-based autograd (what we have now) toward JAX-style composable function transformations, and how interaction combinators make this natural.

## Three Approaches to Autodiff

### 1. Tape-Based (PyTorch, our current impl)

Record operations as you execute them, walk backward.

```
forward:  record x*x → tape  
backward: walk tape, apply chain rule
```

Limitations:
- Can only differentiate what was recorded
- `grad(grad(f))` needs re-recording on each call
- No static analysis — the tape only exists at runtime
- Memory proportional to computation depth

### 2. Graph Rewriting (tinygrad's direction)

Build a lazy UOp graph, then *rewrite the graph* to add backward ops.

```python
# tinygrad: backward is a graph transformation
loss.backward()  # rewrites the UOp graph to include gradients
```

Tinygrad is moving here — backward operates at the `LazyBuffer` level, not on the tensor API. The gradient computation is another set of UOps that get scheduled and fused just like forward ops. This means:
- Gradients can be kernel-fused with forward ops
- The scheduler treats forward and backward identically
- Still fundamentally reverse-mode only

### 3. Function Transforms (JAX)

`grad` is a function that takes a function and returns a new function.

```python
f  = lambda x: x ** 2
f' = grad(f)          # returns a NEW function: x → 2x  
f'' = grad(grad(f))   # x → 2
```

Key properties:
- **Composable**: `grad`, `vmap`, `jit` are independent transforms
- **Both modes**: `jvp` (forward) and `vjp` (reverse) are first-class
- **Higher-order**: `grad(grad(f))` just works
- **Traced, not taped**: JAX traces the function to build a `jaxpr` IR, then transforms the IR

## How IC Makes This Natural

Here's the insight: in interaction combinators, **functions are nets (graphs of nodes)**. A function `f` is a piece of interaction net with one free port (input) and one free port (output). Applying a transformation to `f` means *rewriting the net*.

### `grad(f)` = DUP + Differentiation Rules

When you DUP a function in IC:

```
f ──→ [DUP] ──→ f₁  (used for forward)
             └──→ f₂  (used for backward)
```

The two copies share the same structure. The "adjoint" copy `f₂` is `f` with each operation replaced by its gradient rule:

```
Original net:     x ─→ [MUL] ─→ [ADD] ─→ y
                        ↑          ↑
                        w          b

Adjoint net:     dy ─→ [ADD_bwd] ─→ [MUL_bwd] ─→ dx
                  (=1 pass)      (= w·dy)   (multiply by weight)
```

In IC terms: `grad` is not a runtime operation. It's a **static transformation** on the interaction net. Given the net for `f`, produce the net for `f'`. The result is itself an IC net that can be further transformed.

### `grad(grad(f))` = DUP(DUP(f))

Since `grad(f)` produces an IC net, `grad(grad(f))` just DUPs it again and applies differentiation rules to the adjoint net. This gives us higher-order derivatives for free — they're just nested DUP operations.

### `vmap(f)` = Parallel Reduction

`vmap` maps a function over a batch dimension. In IC, this is parallel reduction of independent copies:

```
[x₁, x₂, x₃] ──→ [DUP³] ──→ f(x₁), f(x₂), f(x₃)
```

Each copy of `f` reduces independently. IC's parallelism guarantee (non-interfering reductions) means this is automatically data-parallel.

### `jit(f)` = Lazy Net Construction

Instead of reducing the net immediately, collect it as a lazy structure and optimize before reducing. We already have this — TOP nodes are lazy until `thvm_reduce` is called.

## What To Build

### Phase A: Graph-Level Diff (next)

Move gradient computation from runtime tape to static graph rewriting:

```c
// Current (tape-based):
thvm_start_recording(ctx);
Term y = thvm_reduce(ctx, thvm_op(ctx, UOP_MUL, x, x));
thvm_stop_recording(ctx);
thvm_backward(ctx, y);  // walks tape

// Target (graph rewriting):
Term y = thvm_op(ctx, UOP_MUL, x, x);       // lazy graph
Term dy_dx = thvm_grad(ctx, y, x);           // rewrites graph to add backward ops
Term result = thvm_reduce(ctx, dy_dx);       // reduces backward graph
```

`thvm_grad(y, x)` doesn't execute anything. It builds a new interaction net that, when reduced, computes ∂y/∂x. The key difference: the gradient is an IC net, not a tape walk.

### Phase B: Forward Mode (JVP)

Add Jacobian-vector products for forward-mode autodiff:

```c
// Forward mode: compute f(x) and f'(x)·v simultaneously
Term tangent = thvm_jvp(ctx, f_net, x, v);
// tangent = directional derivative of f at x in direction v
```

Forward mode is cheaper for f: R→Rⁿ (one input, many outputs). Reverse mode is cheaper for f: Rⁿ→R (many inputs, one output = loss functions). Having both lets the user pick.

### Phase C: Higher-Order Gradients

Since `thvm_grad` returns an IC net:

```c
Term dy_dx = thvm_grad(ctx, y, x);
Term d2y_dx2 = thvm_grad(ctx, dy_dx, x);  // Hessian diagonal
```

This just works if the backward net uses differentiable ops. Each gradient op (like multiply-by-weight in MUL backward) is itself a TOP node that can be differentiated again.

## Current State → Target

| Feature | Current (Phase 2) | Target |
|---|---|---|
| Autodiff mode | Reverse (tape) | Reverse + Forward |
| Higher-order | No | Yes (grad of grad) |
| Fusion | Forward only | Forward + backward fused |
| API | `backward()` + `get_grad()` | `grad()` returns net |
| Implementation | Runtime tape walk | Static graph rewrite |

## What We Keep

The tape-based system we built isn't wasted:
1. It validates the gradient rules (ADD, MUL, MM, RELU) — same rules apply in graph-rewriting mode
2. The `tensor_transpose_2d`, `tensor_reduce_sum_to` helpers are needed regardless
3. `thvm_param_update` (SGD) works the same way — it doesn't care how gradients were computed

The transition is: replace the tape walk in `thvm_backward` with graph construction in `thvm_grad`. Everything downstream stays the same.

## Why IC Is Better Than JAX's Approach

JAX traces Python functions to build `jaxpr` IR, then transforms the IR. This works but has friction:
- Python control flow doesn't trace well (need `lax.cond`, `lax.while_loop`)
- Tracing is a separate step from execution
- `jaxpr` is a custom IR that doesn't compose with the rest of the system

IC doesn't have these problems:
- Functions *are* nets — no tracing step needed
- Control flow is just more net structure (branching = SUP, case analysis = pattern matching)
- The gradient net reduces with the same rules as the forward net
- Composition is DUP — a primitive that's already in the calculus

The catch: we need actual IC reduction for this to work (not just the tensor dispatch we have now). The forward-only TOP dispatch doesn't give us graph-level differentiation. We need the DUP-TOP, ERA-TOP, and SUP-TOP interaction rules to be real.
