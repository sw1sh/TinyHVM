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

## Phase 0: ERA-TOP and DUP-TOP Primitive Rules

**Goal**: Add the missing IC interaction rules for TOP nodes. These are needed regardless of
whether we reform the gradient system.

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

**Benefit**: Dead code elimination for unused gradient branches. Currently REACHES does a DFS
to check reachability, then skips branches. With ERA-TOP, unreachable branches get erased by
the reducer itself as ERA propagates through the graph. This also frees activation tensors
earlier (the "ERA frees activations" item in `ic_pure_inet.md`).

**Complication**: ERA-TOP must also decrement tensor refcounts. When ERA reaches a `TAG_TEN`,
the underlying buffer should be freed (or refcount decremented). Need:

```c
case TAG_TEN:
    tensor_release(ctx, term_val(whnf));  // decrement refcount, free if zero
    result = TAG_ERA_TERM;
```

### DUP-TOP

When a duplicator meets a TOP node, produce a SUP of two independent reductions:

```
DUP-TOP rule:
  DP0/DP1 ⊗ TOP(uop, [a, b])  →  reduce TOP to TEN(id), then SUP(TEN(id), TEN(id))
```

The simple version: force the TOP to reduce to a value (TEN), then duplicate the value.
This is the **strict** DUP-TOP rule — it doesn't try to lazily duplicate the computation.

```c
// In apply phase: TAG_DP0/DP1 frame + TAG_TOP whnf
// Force TOP to value first, then split
Term val = thvm_reduce(ctx, whnf);  // val is TAG_TEN
u64 sup_loc = heap_alloc(ctx, 2);
heap[sup_loc+0] = val;
heap[sup_loc+1] = val;  // same TEN, shared buffer (refcount++)
tensor_retain(ctx, term_val(val));
result = TAG_SUP(sup_loc);
// DP0 takes left, DP1 takes right
```

**Benefit**: Enables IC-level sharing of tensor computations. When a weight tensor is used in
both forward and backward, DUP-SUP handles the split cleanly.

**Alternative (lazy DUP-TOP)**: Duplicate the computation structure, don't force:

```
DP0/DP1 ⊗ TOP(uop, [a, b])  →  SUP(TOP(uop, [DP0(a), DP0(b)]),
                                     TOP(uop, [DP1(a), DP1(b)]))
```

This propagates DUP into sub-terms. More IC-native but risks double computation unless
memoization catches it. Defer to Phase 2.

### Files to modify

- `src/reduce/_.c`: Add ERA/DUP cases for TAG_TOP in apply phase (~20 lines each)
- `src/interact/_.c`: Alternatively handle here if interaction-rule style preferred
- `src/ops/_.c`: Add `tensor_retain` / `tensor_release` for refcount management

### Validation

- Test: build `DUP(TOP(MUL, [a, b]))`, reduce, verify SUP with correct values
- Test: build `ERA(TOP(MUL, [a, b]))`, reduce, verify erasure + buffer freed
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

Build a **lambda** that, given `(gy, a, b)`, returns `(da, db)`:

```c
// adjoint_mul: λgy. λa. λb. (gy*b, gy*a)
Term adjoint_mul(TinyHVM *ctx, Term gy, Term a, Term b) {
    Term da = thvm_op(ctx, UOP_MUL, gy, b);
    Term db = thvm_op(ctx, UOP_MUL, gy, a);
    return thvm_sup(ctx, da, db);  // SUP encodes the pair
}
```

The key difference: this function returns lazy IC terms (TOP nodes), not a GRAD3 wrapper.
The gradient chain is an IC net of `APP(adjoint_op, gy, saved_inputs...)` terms.

### Gradient Chain as IC Net

`thvm_grad(ctx, y, x)` walks provenance and stitches adjoint subnets:

```
y was created by MUL(a, b)
  → build: APP(adjoint_mul, gy, a, b)
  → this produces SUP(da, db)
  → recurse: connect da → grad_chain(a, x), db → grad_chain(b, x)
  → if a doesn't reach x: connect da → ERA (dead branch)
  → if a == x: da is the answer
```

The result is a pure IC net. Reducing it fires APP-LAM (beta), DUP-SUP (sharing),
ERA-TOP (dead branches), and TOP-TOP (the actual tensor ops inside adjoint subnets).

### Adjoint Registry

```c
// In tinyhvm.h or a new src/adjoint/_.c
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
    [UOP_SUM]     = adjoint_sum,      // (gy, a_shape) → expand(gy, a_shape)
    [UOP_EXPAND]  = adjoint_expand,   // (gy, a_shape) → sum_to_shape(gy)
    // ... movement ops, etc.
};
```

### `thvm_grad` Rewrite

```c
Term thvm_grad(TinyHVM *ctx, Term y, Term x) {
    Term gy = tensor_fill(ctx, y_shape, 1.0f);  // seed
    return build_grad_net(ctx, y_id, gy, x_id);
}

static Term build_grad_net(TinyHVM *ctx, u32 y_id, Term gy, u32 x_id) {
    if (y_id == x_id) return gy;                  // base case
    TensorMeta *md = &ctx->meta[y_id];
    if (md->creator_op == 0) return TAG_ERA_TERM;  // leaf, no path

    AdjointFn adj = adjoint_table[md->creator_op];
    Term saved[] = { TAG_TEN(md->src_ids[0]), TAG_TEN(md->src_ids[1]) };
    Term grads = adj(ctx, gy, saved, 2);  // returns SUP(da, db) or single term

    // For binary ops: split SUP, recurse on each branch
    if (is_binary(md->creator_op)) {
        u32 a_id = md->src_ids[0], b_id = md->src_ids[1];
        int a_reaches = reaches(ctx, a_id, x_id);
        int b_reaches = reaches(ctx, b_id, x_id);

        if (a_reaches && b_reaches) {
            // Both branches: DUP the SUP, recurse both
            Term da = thvm_dp0(ctx, grads);  // DP0(SUP(da, db)) → da
            Term db = thvm_dp1(ctx, grads);  // DP1(SUP(da, db)) → db
            Term ga = build_grad_net(ctx, a_id, da, x_id);
            Term gb = build_grad_net(ctx, b_id, db, x_id);
            return thvm_op(ctx, UOP_ADD, ga, gb);  // accumulate
        } else if (a_reaches) {
            Term da = thvm_dp0(ctx, grads);
            return build_grad_net(ctx, a_id, da, x_id);
        } else if (b_reaches) {
            Term db = thvm_dp1(ctx, grads);
            return build_grad_net(ctx, b_id, db, x_id);
        } else {
            return TAG_ERA_TERM;
        }
    } else {
        // Unary op: grads is the single gradient term
        return build_grad_net(ctx, md->src_ids[0], grads, x_id);
    }
}
```

This produces a pure IC net. No UOP_GRAD tag needed.

### Files to modify

- New `src/adjoint/_.c`: Adjoint subnet builders (~200 lines, one per op)
- `src/grad/_.c`: Rewrite `thvm_grad` to call `build_grad_net` instead of creating UOP_GRAD
- `src/interact/_.c`: Remove UOP_GRAD handler (~270 lines deleted)
- `src/tinyhvm.h`: Remove UOP_GRAD from enum, add adjoint_table declaration

### Validation

- All existing `test_grad_*` must pass with identical numerical results
- Verify lazy evaluation: gradient ops should only fire when reduced
- Memory: compare peak tensor count vs. old GRAD handler (should be similar or better)

---

## Phase 2: Lazy DUP-TOP and ERA Propagation for Activation Freeing

**Goal**: Make DUP-TOP propagate lazily into sub-terms (not force-reduce), and make ERA
propagation free tensor buffers.

### Lazy DUP-TOP

Phase 0's strict DUP-TOP forces the TOP to a value before splitting. This loses laziness.
Lazy DUP-TOP instead duplicates the computation graph:

```
DUP(TOP(MUL, [a, b]))  →  SUP(TOP(MUL, [DUP₀(a), DUP₀(b)]),
                               TOP(MUL, [DUP₁(a), DUP₁(b)]))
```

Each DUP₀/DUP₁ gets its own label to avoid clashing with other DUPs. This is standard
HVM duplication of constructors — TOP is treated as a constructor with `uop` as the tag
and `[a, b]` as fields.

**Memoization prevents double work**: when both copies of `a` reduce to the same value,
`reduce_memo` catches it. The DUP propagation happens at the graph level; actual computation
only happens once.

**When this matters**: if a weight tensor W is used in both `MUL(X, W)` (forward) and
`MUL(gy, W)` (backward), lazy DUP lets the reducer share W's reduction across both uses
through DUP-SUP annihilation.

### ERA Propagation and Buffer Freeing

When ERA meets a TAG_TEN, decrement the tensor's refcount. When refcount hits zero, free the
GPU buffer. This turns IC erasure into automatic memory management:

```
forward:  X @ W → H → RELU(H) → loss
backward: GRAD produces dW using H
          after dW is computed, H has no remaining references
          ERA propagates to H → buffer freed
```

Currently `thvm_reset(ctx, n_weights)` bulk-frees everything after each step. ERA propagation
would make this incremental — activations freed as soon as their last consumer reduces.

**Complication**: tensor metadata (TensorMeta) and GPU buffer are separate. ERA frees the
logical tensor; buffer freeing needs a buffer-level refcount (multiple views can share a buffer).
This is mostly already handled by `buf_id` indirection but needs auditing.

### Files to modify

- `src/reduce/_.c`: Lazy DUP-TOP in apply phase (~30 lines)
- `src/reduce/_.c` or `src/interact/_.c`: ERA-TEN handler for buffer freeing (~15 lines)
- `src/ops/_.c`: `tensor_release` implementation with buffer refcount
- Test: verify no double-free, no use-after-free in gradient chains

---

## Phase 3: Higher-Order Gradients

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

## Phase 4 (Speculative): Training Loop as Single IC Term

**Goal**: Eliminate the C training shell. The entire training loop is one IC net.

This is the full vision from `ic_pure_inet.md`. With Phases 0-3 done:

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
- Phase 2 (ERA propagation) — activations freed between steps without `thvm_reset`
- `TAG_REF` / fixpoint definitions — for the recursive `self` call
- `UOP_IFZ` already exists for the counter-based termination
- `UOP_ASSIGN` already exists for in-place weight update

**Realistic assessment**: Phases 0-1 are straightforward engineering (~1-2 weeks each).
Phase 2 requires careful memory management. Phase 3 is mostly validation. Phase 4 is a
significant architectural change that depends on TAG_REF being solid, and the performance
characteristics are uncertain (IC overhead per training step vs. the current thin C shell).

---

## What This Does NOT Give Us

- **Differential linear logic**: We are not implementing codereliction or the exponential
  modality. The adjoints are hand-written chain rules, same as before — just expressed as
  IC subnets rather than a switch statement.

- **Automatic derivative discovery**: The adjoint for each op is still manually specified.
  The IC structure doesn't derive `∂MUL/∂a = b` from first principles.

- **Forward-mode AD for free**: Forward-mode (JVP) would require separate tangent subnets
  propagating perturbations forward. Doable but orthogonal to this plan.

- **Fusion of forward+backward kernels**: Would require the JIT/pattern matcher (see
  `resources/plans/jit_ranges_patterns.md`) to recognize backward ops and fuse them with
  forward. The IC structure makes the ops visible for pattern matching, but the actual
  fusion logic is separate work.

## What It DOES Give Us

- **Delete the UOP_GRAD handler**: 270 lines of switch statement → ~200 lines of small,
  composable adjoint functions + a ~50-line graph builder. Easier to audit and extend.

- **Higher-order gradients**: `grad(grad(f))` works by construction once adjoints record
  provenance.

- **ERA-based DCE**: Dead gradient branches eliminated by the reducer, not by REACHES DFS.
  Cleaner and handles cases REACHES might miss (e.g., deeply nested dead paths beyond the
  64-depth limit).

- **Path to single-term training**: Phases 0-2 are prerequisites for Phase 4. Even if we
  never ship Phase 4, the intermediate phases improve the codebase.

- **Composability**: Adjoint subnets are IC terms. They can be inspected, transformed,
  serialized, or optimized by any tool that operates on IC nets.
