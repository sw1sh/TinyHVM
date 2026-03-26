# Plan: IC-Native Backprop

## Context

TinyHVM's autograd (`UOP_GRAD` in `src/interact/_.c:35-306`) is standard reverse-mode AD:
a hand-written chain-rule switch dispatches on `creator_op`, building lazy `TAG_TOP(UOP_GRAD)`
terms. No DUP/ERA/SUP interaction rules participate in gradient computation. The backward pass
is an imperative case analysis wearing an IC costume.

This plan sketches what it would take to make backprop work through actual IC interaction rules,
what the realistic benefits are, and where the theory runs out.

**Prerequisite reading**: `resources/ic_autograd.md` (honest assessment of theory vs. implementation)

---

## What "IC-Native Backprop" Means Concretely

Replace the `UOP_GRAD` chain-rule switch with interaction rules that the reducer already knows
how to fire. Gradient computation becomes graph structure that reduces through the same enter/apply
trampoline as everything else — not a special-case handler.

**Current flow:**
```
thvm_grad(ctx, y, x)  →  TAG_TOP(UOP_GRAD, [y, ones, x])
  → reduce hits UOP_GRAD handler
  → switch(creator_op) { case UOP_MUL: ... case UOP_ADD: ... }
  → builds more TAG_TOP(UOP_GRAD, ...) terms
  → recurse until y == x
```

**Target flow:**
```
thvm_grad(ctx, y, x)  →  builds an IC net (LAM/APP/DUP/SUP/TOP)
  → net encodes: "given dy, produce dx by reversing each op"
  → reducing this net fires standard IC rules + a few new TOP interaction rules
  → no UOP_GRAD handler exists
```

The gradient "function" for each op is a small IC subnet. `thvm_grad` stitches these subnets
together by walking provenance. The reducer fires them.

---

## What This Is NOT

This is **not** "differentiation as duplication" from differential linear logic. The Ehrhard-Regnier
theory gives forward-mode AD (tangent propagation). Reverse-mode requires the adjoint/transpose,
which is a separate transformation. We are not implementing differential interaction nets.

What we **are** doing: expressing reverse-mode AD as IC graph structure instead of a runtime
switch statement. The benefits are compositional, not theoretical.

---

## Phase 0: Lazy DUP-TOP and ERA-TOP Primitive Rules

**Goal**: Add the missing IC interaction rules for TOP nodes. These are prerequisites for
everything that follows.

### ERA-TOP

When an eraser meets a TOP node, propagate erasure into its arguments:

```
ERA-TOP rule:
  ERA ⊗ TOP(uop, [a, b])  →  ERA(a), ERA(b)
```

Implementation in `src/reduce/_.c` apply phase (or `src/interact/_.c`):

```c
// In apply phase: TAG_ERA frame + TAG_TOP whnf
// Or: when reducing TAG_TOP and one arg is ERA
case TAG_ERA:
    if (term_tag(whnf) == TAG_TOP) {
        u64 loc = term_loc(whnf);
        // Propagate ERA into sub-terms
        heap[loc+0] = TAG_ERA_TERM;
        heap[loc+1] = TAG_ERA_TERM;
        // Result is ERA
        result = TAG_ERA_TERM;
    }
```

ERA-TEN (terminal case): when ERA reaches a materialized tensor, decrement refcount:

```c
case TAG_TEN:
    tensor_release(ctx, term_val(whnf));  // decrement refcount, free if zero
    result = TAG_ERA_TERM;
```

**Benefit**: Dead code elimination for unused gradient branches, and incremental activation
freeing. This replaces both the REACHES DFS (dead path pruning) and `thvm_reset` bulk-free.

**Complication**: tensor metadata (TensorMeta) and GPU buffer are separate. ERA frees the
logical tensor; buffer freeing needs a buffer-level refcount (multiple views can share a buffer).
Mostly handled by `buf_id` indirection but needs auditing.

### Lazy DUP-TOP

When a duplicator meets a TOP node, propagate DUP into sub-terms (standard constructor
duplication — TOP treated as a constructor with `uop` as tag and `[a, b]` as fields):

```
DUP-TOP rule (lazy):
  DP0/DP1 ⊗ TOP(uop, [a, b])  →  SUP(TOP(uop, [DP0(a), DP0(b)]),
                                       TOP(uop, [DP1(a), DP1(b)]))
```

Each DUP₀/DUP₁ gets its own label to avoid clashing with other DUPs.

```c
// In apply phase: TAG_DP0/DP1 frame + TAG_TOP whnf
u32 uop = term_uop(whnf);
u64 old_loc = term_loc(whnf);
u32 lab = term_lab(frame);  // DUP label

// Create DP nodes for each argument
u64 dp_a = heap_alloc(ctx, 1);
u64 dp_b = heap_alloc(ctx, 1);
heap[dp_a] = heap[old_loc+0];  // shared arg a
heap[dp_b] = heap[old_loc+1];  // shared arg b

// Create two TOP copies with DP0/DP1 on each arg
u64 top0_loc = heap_alloc(ctx, 2);
heap[top0_loc+0] = TAG_DP0(lab, dp_a);
heap[top0_loc+1] = TAG_DP0(lab, dp_b);

u64 top1_loc = heap_alloc(ctx, 2);
heap[top1_loc+0] = TAG_DP1(lab, dp_a);
heap[top1_loc+1] = TAG_DP1(lab, dp_b);

// Result is SUP of both copies
u64 sup_loc = heap_alloc(ctx, 2);
heap[sup_loc+0] = TAG_TOP(uop, top0_loc);
heap[sup_loc+1] = TAG_TOP(uop, top1_loc);
result = TAG_SUP(lab, sup_loc);
```

**Memoization prevents double work**: when both DP0/DP1 copies of `a` eventually reduce,
`reduce_memo` caches the result. The DUP propagation happens at the graph level; actual
GPU computation only fires once.

**When this matters**: weight tensor W used in both `MUL(X, W)` (forward) and `MUL(gy, W)`
(backward) — lazy DUP lets the reducer share W's reduction across both uses through
DUP-SUP annihilation.

### Files to modify

- `src/reduce/_.c`: Add ERA-TOP, ERA-TEN, DUP-TOP cases in apply phase (~50 lines total)
- `src/ops/_.c`: Add `tensor_retain` / `tensor_release` for refcount management

### Validation

- Test: `DUP(TOP(MUL, [a, b]))` reduces to SUP with correct values on both branches
- Test: `ERA(TOP(MUL, [a, b]))` propagates erasure, no leaked buffers
- Test: existing `test_grad_*` still pass (GRAD handler unchanged at this phase)

---

## Phase 1: Adjoint Subnets — IC Nets That Compute Gradients

**Goal**: For each differentiable op, define a small IC subnet that computes its gradient.
These replace the cases in the UOP_GRAD switch.

### The Adjoint Subnet Concept

Instead of:
```c
case UOP_MUL:
    da = thvm_op(ctx, UOP_MUL, gy, b);  // gy * b
    db = thvm_op(ctx, UOP_MUL, gy, a);  // gy * a
    return GRAD_COMBINE(GRAD3(a, da, x), GRAD3(b, db, x));
```

Build a function that, given `(gy, a, b)`, returns `(da, db)` as lazy TOP nodes:

```c
// adjoint_mul: (gy, a, b) → SUP(gy*b, gy*a)
Term adjoint_mul(TinyHVM *ctx, Term gy, Term a, Term b) {
    Term da = thvm_op(ctx, UOP_MUL, gy, b);
    Term db = thvm_op(ctx, UOP_MUL, gy, a);
    return thvm_sup(ctx, da, db);  // SUP encodes the pair
}
```

The key difference: returns lazy IC terms (TOP nodes), not a GRAD3 wrapper.
The gradient chain is an IC net — reducing it fires TOP interactions and DUP-SUP splits.

### Adjoint Registry

New file `src/adjoint/_.c`:

```c
typedef Term (*AdjointFn)(TinyHVM *ctx, Term gy, Term *saved, int n_saved);

AdjointFn adjoint_table[UOP_COUNT] = {
    [UOP_ADD]     = adjoint_add,      // (gy) → SUP(gy, gy)
    [UOP_SUB]     = adjoint_sub,      // (gy) → SUP(gy, NEG(gy))
    [UOP_MUL]     = adjoint_mul,      // (gy, a, b) → SUP(gy*b, gy*a)
    [UOP_DIV]     = adjoint_div,      // (gy, a, b) → SUP(gy/b, -gy*a/b²)
    [UOP_MM]      = adjoint_mm,       // (gy, A, B) → SUP(gy@Bᵀ, Aᵀ@gy)
    [UOP_RELU]    = adjoint_relu,     // (gy, a) → gy * (a > 0)
    [UOP_EXP]     = adjoint_exp,      // (gy, z) → gy * z
    [UOP_LOG]     = adjoint_log,      // (gy, a) → gy / a
    [UOP_SQRT]    = adjoint_sqrt,     // (gy, z) → gy / (2*z)
    [UOP_NEG]     = adjoint_neg,      // (gy) → -gy
    [UOP_SUM]     = adjoint_sum,      // (gy, a_shape) → expand(gy, a_shape)
    [UOP_EXPAND]  = adjoint_expand,   // (gy, a_shape) → sum_to_shape(gy)
    [UOP_RESHAPE] = adjoint_reshape,  // (gy, a_shape) → reshape(gy, a_shape)
    [UOP_PERMUTE] = adjoint_permute,  // (gy, inv_perm) → permute(gy, inv_perm)
    [UOP_PAD]     = adjoint_pad,      // (gy, a_shape) → shrink(gy, ...)
    [UOP_SHRINK]  = adjoint_shrink,   // (gy, a_shape) → pad(gy, ...)
    [UOP_MAX]     = adjoint_max,      // (gy, a, z) → gy * (a == z)
};
```

Each function is 5-15 lines. Total ~200 lines for the file.

### `thvm_grad` Rewrite

`build_grad_net` does **not** use REACHES. It builds both branches unconditionally for binary
ops. Dead branches receive ERA as their gradient seed when the provenance walk finds a leaf
(`creator_op == 0`) that isn't `x`. ERA-TOP (Phase 0) propagates through and kills the
dead subnet during reduction — the reducer does the DCE, not the graph builder.

```c
Term thvm_grad(TinyHVM *ctx, Term y, Term x) {
    u32 y_id = term_val(thvm_reduce(ctx, y));
    u32 x_id = term_val(thvm_reduce(ctx, x));
    Term gy = tensor_fill(ctx, meta[y_id].view.shape, 1.0f);  // seed
    return build_grad_net(ctx, y_id, gy, x_id);
}

static Term build_grad_net(TinyHVM *ctx, u32 y_id, Term gy, u32 x_id) {
    if (y_id == x_id) return gy;                  // base case: dy/dx = gy
    TensorMeta *md = &ctx->meta[y_id];
    if (md->creator_op == 0) return TAG_ERA_TERM;  // leaf, no grad path
                                                    // ERA propagates up and kills
                                                    // any ops that depended on this

    AdjointFn adj = adjoint_table[md->creator_op];
    Term saved[] = { TAG_TEN(md->src_ids[0]), TAG_TEN(md->src_ids[1]) };
    Term grads = adj(ctx, gy, saved, 2);  // SUP(da, db) or single term

    if (is_binary(md->creator_op)) {
        // Always build both branches. Dead ones return ERA from recursion,
        // which propagates up through ERA-TOP and gets cleaned up.
        u32 a_id = md->src_ids[0], b_id = md->src_ids[1];

        Term da = thvm_dp0(ctx, grads);  // DP0(SUP(da, db)) → da
        Term db = thvm_dp1(ctx, grads);  // DP1(SUP(da, db)) → db
        Term ga = build_grad_net(ctx, a_id, da, x_id);
        Term gb = build_grad_net(ctx, b_id, db, x_id);

        // If one branch returned ERA, ADD(ERA, other) → other
        // (already an existing ERA-as-identity rule in interact.c)
        return thvm_op(ctx, UOP_ADD, ga, gb);
    } else {
        return build_grad_net(ctx, md->src_ids[0], grads, x_id);
    }
}
```

**Why no REACHES**: the whole point of ERA-TOP is that dead branches self-eliminate during
reduction. `build_grad_net` recurses into a dead branch, hits a leaf, returns `TAG_ERA_TERM`.
That ERA flows back up: `ADD(ERA, live_grad)` → `live_grad` (existing ERA-as-identity rule).
The adjoint ops on the dead branch (`MUL(gy, b)`) never get reduced because their result
feeds into ERA. The reducer doesn't even enter them — ERA-TOP erases them on contact.

This is cleaner than REACHES (no depth limit, no DFS at graph-build time) and the DCE
happens where it belongs: in the reducer.

### Files to modify

- New `src/adjoint/_.c`: Adjoint subnet builders (~200 lines, one function per op)
- `src/grad/_.c`: Rewrite `thvm_grad` to call `build_grad_net`
- `src/interact/_.c`: Remove UOP_GRAD handler (~270 lines deleted)
- `src/tinyhvm.h`: Remove UOP_GRAD from enum, add adjoint_table declaration

### Validation

- All existing `test_grad_*` must pass with identical numerical results
- Verify lazy evaluation: gradient ops should only fire when reduced
- Memory: compare peak tensor count vs. old GRAD handler (should be similar or better)

---

## Phase 2: Higher-Order Gradients

**Goal**: `thvm_grad(ctx, thvm_grad(ctx, y, x), x)` produces a correct Hessian-vector product.

### Why this works with Phase 1

After Phase 1, `thvm_grad` returns a pure IC net of TOP nodes. Those TOP nodes have
`requires_grad` set (they were created by adjoint subnets using `thvm_op`). Their provenance
is recorded. Calling `thvm_grad` on the resulting net walks provenance again and builds
another layer of adjoint subnets.

```
y = x * x                      // creator_op = MUL, src_ids = [x, x]
dy_dx = thvm_grad(y, x)        // builds: TOP(MUL, [ones, x]) + TOP(MUL, [ones, x])
                                //       = 2*x (after ADD)
d2y_dx2 = thvm_grad(dy_dx, x)  // walks provenance of the ADD and MULs in dy_dx
                                // produces: 2 (constant)
```

### Requirements

- Adjoint subnets must use `thvm_op` (which records provenance), not raw buffer operations
- Movement ops (RESHAPE, EXPAND) must have correct second-order adjoints
- MM transpose-of-transpose must cancel correctly

### Validation

- `test_hessian_quadratic`: verify d²(x²)/dx² = 2
- `test_hessian_vector_product`: verify Hv for a small MLP loss function
- Numerical check: compare against finite differences (f(x+εv) - 2f(x) + f(x-εv)) / ε²

---

## Phase 3: Forward-Mode AD (JVP)

**Goal**: Add Jacobian-vector products for forward-mode autodiff. This is the mode that
differential linear logic actually describes — tangent propagation forward through the graph.

### Why forward-mode

Reverse-mode (VJP, what we have) is efficient for f: R^n → R (many inputs, scalar output —
loss functions). Forward-mode (JVP) is efficient for f: R → R^n (scalar input, many outputs).
Having both enables:

- **Hessian-vector products** via forward-over-reverse: `jvp(λx. vjp(f, x), x, v)` computes
  `H·v` without materializing the full Hessian
- **Jacobian columns** cheaply for low-input-dim functions
- **Directional derivatives** for sensitivity analysis

### Tangent Subnets

Mirror of adjoint subnets, but propagate a tangent `t` forward instead of a gradient `gy`
backward:

```c
// New file: src/tangent/_.c
typedef Term (*TangentFn)(TinyHVM *ctx, Term t, Term *args, int n_args);

TangentFn tangent_table[UOP_COUNT] = {
    [UOP_ADD]  = tangent_add,   // (ta, tb) → ta + tb
    [UOP_MUL]  = tangent_mul,   // (ta, tb, a, b) → ta*b + a*tb
    [UOP_RELU] = tangent_relu,  // (ta, a) → ta * (a > 0)
    [UOP_EXP]  = tangent_exp,   // (ta, a) → ta * exp(a)
    [UOP_LOG]  = tangent_log,   // (ta, a) → ta / a
    [UOP_MM]   = tangent_mm,    // (tA, tB, A, B) → tA@B + A@tB
    [UOP_SUM]  = tangent_sum,   // (ta) → sum(ta)
    // ...
};
```

Each tangent rule is the Jacobian-vector product: `J·t` where J is the Jacobian of the op.
These are the same rules as Ehrhard-Regnier's differential lambda calculus, applied to
tensor ops.

### `thvm_jvp` — Forward-Mode API

```c
// Compute f(x) and df(x)·v simultaneously
// Returns SUP(primal, tangent) — both values in one reduction
Term thvm_jvp(TinyHVM *ctx, Term y, Term x, Term v) {
    u32 y_id = term_val(thvm_reduce(ctx, y));
    u32 x_id = term_val(thvm_reduce(ctx, x));
    return build_tangent_net(ctx, y_id, x_id, v);
}

static Term build_tangent_net(TinyHVM *ctx, u32 y_id, u32 x_id, Term tangent_seed) {
    if (y_id == x_id) return tangent_seed;         // base case: dx/dx · v = v
    TensorMeta *md = &ctx->meta[y_id];
    if (md->creator_op == 0) return tensor_zeros(ctx, md->view.shape);  // leaf: 0

    // Recursively compute tangents of inputs
    u32 a_id = md->src_ids[0];
    Term ta = build_tangent_net(ctx, a_id, x_id, tangent_seed);

    if (is_binary(md->creator_op)) {
        u32 b_id = md->src_ids[1];
        Term tb = build_tangent_net(ctx, b_id, x_id, tangent_seed);
        Term args[] = { ta, tb, TAG_TEN(a_id), TAG_TEN(b_id) };
        return tangent_table[md->creator_op](ctx, ta, args, 4);
    } else {
        Term args[] = { ta, TAG_TEN(a_id) };
        return tangent_table[md->creator_op](ctx, ta, args, 2);
    }
}
```

**Key difference from reverse-mode**: forward-mode walks the provenance graph from inputs to
outputs (bottom-up in the DAG), not outputs to inputs. The tangent at each node depends on
tangents of its inputs, which must be computed first. This means `build_tangent_net` needs
topological order — either via recursive memoization (natural if we cache by `y_id`) or an
explicit topo sort.

### Forward-Over-Reverse for Hessian-Vector Products

```c
// H·v = d/dx [grad(f, x)] · v = jvp(x → grad(f(x), x), x, v)
//
// Step 1: build reverse-mode gradient net (Phase 1)
Term grad_net = thvm_grad(ctx, loss, x);
// Step 2: compute JVP of the gradient net w.r.t. x in direction v
Term hvp = thvm_jvp(ctx, grad_net, x, v);
// Result: Hessian-vector product without materializing the Hessian
```

This requires Phase 1 (adjoint subnets record provenance) so that `thvm_jvp` can walk
the provenance of the backward graph.

### Relationship to Differential Linear Logic

This is the mode where the theory actually applies. Each tangent rule is a linearization:

```
d(a * b)/dx · t = (da/dx · t) * b + a * (db/dx · t)    -- Leibniz rule
```

In differential LL terms: `t` is the codereliction `d(x)` — one linear probe of `x`.
The tangent propagates forward through each op, producing one linear response at the output.
DUP naturally appears when a variable is used twice: `d(x*x)/dx · t = t*x + x*t = 2*x*t`.

This is the one place where the "DUP = differentiation" slogan is literally correct:
forward-mode tangent splitting at a fan-out point IS contraction/DUP in linear logic.

### Files to modify

- New `src/tangent/_.c`: Tangent subnet builders (~150 lines, mirrors adjoint structure)
- `src/grad/_.c`: Add `thvm_jvp` and `build_tangent_net`
- `src/tinyhvm.h`: Add `tangent_table`, `thvm_jvp` declaration

### Validation

- `test_jvp_mul`: verify d(x*x)/dx · v = 2*x*v
- `test_jvp_mm`: verify d(X@W)/dW · V = X@V
- `test_hvp`: verify forward-over-reverse Hv against finite differences
- Numerical: compare all JVPs against `(f(x+εv) - f(x))/ε` for small ε

---

## Phase 4 (Speculative): Training Loop as Single IC Term

**Goal**: Eliminate the C training shell. The entire training loop is one IC net.

This is the full vision from `ic_pure_inet.md`. With Phases 0-2 done:

```
train = FIX(λself. λW. λdata. λn.
  IFZ n (W) (
    let loss = forward(W, next(data))
    let dW   = GRAD(loss, W)              // Phase 1: pure IC net
    let W'   = ASSIGN(W, W - lr * dW)
    self W' (advance data) (n-1)
  ))

result = thvm_reduce(ctx, APP(train, W0, data, 70))
```

This requires:
- Phase 1 (adjoint subnets) — grad is a net, not a handler
- Phase 0 (ERA propagation) — activations freed between steps without `thvm_reset`
- `TAG_REF` / fixpoint definitions — for the recursive `self` call
- `UOP_IFZ` already exists for the counter-based termination
- `UOP_ASSIGN` already exists for in-place weight update

**Realistic assessment**: Phases 0-1 are straightforward engineering. Phase 2 is mostly
validation. Phase 3 is a separate feature (forward-mode). Phase 4 is a significant
architectural change that depends on TAG_REF being solid, and the performance characteristics
are uncertain (IC overhead per training step vs. the current thin C shell).

---

## What This Does NOT Give Us

- **Differential linear logic**: We are not implementing codereliction or the exponential
  modality. The adjoints are hand-written chain rules, same as before — just expressed as
  IC subnets rather than a switch statement. (Phase 3's forward-mode is the closest thing
  to the actual theory, but still uses hand-written tangent rules, not derived from DUP.)

- **Automatic derivative discovery**: The adjoint/tangent for each op is still manually
  specified. The IC structure doesn't derive `∂MUL/∂a = b` from first principles.

- **Fusion of forward+backward kernels**: Would require the JIT/pattern matcher (see
  `resources/plans/jit_ranges_patterns.md`) to recognize backward ops and fuse them with
  forward. The IC structure makes the ops visible for pattern matching, but the actual
  fusion logic is separate work.

## What It DOES Give Us

- **Delete the UOP_GRAD handler**: 270 lines of switch statement → ~200 lines of small,
  composable adjoint functions in `src/adjoint/_.c` + a ~50-line graph builder. Easier to
  audit and extend.

- **Higher-order gradients**: `grad(grad(f))` works by construction once adjoints record
  provenance.

- **ERA-based DCE**: Dead gradient branches eliminated by the reducer during reduction,
  not by REACHES DFS at graph-build time. No depth limit, no separate reachability pass.

- **Forward-mode AD**: JVP via tangent subnets. Enables Hessian-vector products via
  forward-over-reverse without materializing the Hessian.

- **Path to single-term training**: Phases 0-1 are prerequisites for Phase 4. Even if we
  never ship Phase 4, the intermediate phases improve the codebase.

- **Composability**: Adjoint and tangent subnets are IC terms. They can be inspected,
  transformed, serialized, or optimized by any tool that operates on IC nets.
