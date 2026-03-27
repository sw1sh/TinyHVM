# Plan: IC-Native Backprop

## Context

TinyHVM computes gradients via `UOP_GRAD` interaction rules in `src/interact/_.c`. The GRAD
handler fires lazily during `thvm_reduce` — it IS an interaction rule, not an imperative graph
walk. This is already IC-native in spirit: GRAD(y, gy, x) is a lazy TAG_TOP term that reduces
by dispatching on `creator_op` provenance and producing new GRAD terms via chain rule.

This plan documents the current design, its proven correctness properties, and the path to
optimization (DUP accumulation for shared tensors, adjoint table for cleaner code).

---

## Design Principles

1. **Everything is lazy reduction.** No graph walks. No `build_grad_net`. No eager backward
   pass. `thvm_grad(ctx, y, x)` creates a single GRAD term. `thvm_reduce` drives everything.

2. **Forward graph has no linearity enforcement.** `linear_use` is disabled. The same
   TAG_TOP term can appear in multiple heap slots. The trampoline handles this correctly:
   first entry reduces and overwrites the TOP's args; subsequent entries find already-reduced
   args and re-fire the rule (producing duplicate tensors with identical values). This is
   redundant but correct.

3. **GRAD is correct without DUP.** Diamond fan-outs (tensor h used by 2+ ops) produce
   redundant GRAD walks, but the multivariate chain rule guarantees correctness: each path
   independently computes its contribution, and ADD sums them. Proof:

   ```
   z = f(h, h), h = g(x)
   dz/dx = df/dh₁ · dh/dx + df/dh₂ · dh/dx    (chain rule for repeated variable)
         = (df/dh₁ + df/dh₂) · dh/dx            (linearity of multiplication)
   ```

   Without DUP: both terms computed separately, added.
   With DUP: (df/dh₁ + df/dh₂) accumulated at DUP, multiplied once by dh/dx.
   Same result, different computation order.

4. **DUP is an optimization, not a correctness requirement.** DUP accumulation reduces
   dispatch count at diamond nodes (O(1) backward walk per shared tensor instead of O(k)
   for k consumers). Added only after correctness is proven.

5. **Adjoint table is a refactoring, not a redesign.** The per-op gradient rules (the dagger
   structure) must be manually specified. Moving them from a switch to a table improves
   readability but doesn't change the mathematics.

---

## Current State (verified)

### What works

- **GRAD handler** fires lazily via TAG_TOP(UOP_GRAD) reduction
- **No DUP** — `linear_use` is a no-op, GRAD3_FWD always creates plain GRAD3
- **ERA propagation** for binary ops: ADD(ERA,x)→x, MUL(ERA,x)→ERA, Unary(ERA)→ERA
- **Gradient parity with numpy** verified across multiple architectures:
  - Simple MLP (no diamond): ALL ZERO (max 2e-8, BLAS noise)
  - 1-layer CNN + sum loss: EXACT (conv_w, conv_b, lin_w, lin_b all < 3e-6)
  - 2-layer CNN (Conv(1,4,3)→ReLU→Conv(4,8,3)→ReLU→Linear): ALL < 8e-6
  - CNN + cross-entropy loss (softmax diamonds): ALL < 4e-6
- **Single reduce** drives forward + backward + SGD: `thvm_reduce(ctx, chain)`
- **Pool chain backward** works correctly through PERMUTE→EXPAND→RESHAPE→SHRINK chain

### Bugs found and fixed

1. **ERA propagation missing**: ADD(ERA, tensor) dispatched ERA as tensor_id=0 instead of
   returning the live tensor. Fixed: universal ERA rules at start of binary dispatch.

2. **Multi-axis SUM fusion bug**: RESHAPE(SUM(MUL)) rewrite rule produced wrong results for
   multi-axis reduce (axes=[3,4] etc). The fused kernel's index computation was incorrect.
   Workaround: skip fusion for multi-axis SUM in rewrite rules. TODO: fix the codegen.

3. **Masked view reshape drops mask**: `view_reshape` for masked views (from PAD backward)
   returned a placeholder with `has_mask=0`, causing downstream ops to read unpadded data.
   Fixed: force masked views to lazy path (materialization during reduction).

4. **Metal codegen ignores masks**: `metal_contiguify` uses codegen that doesn't handle
   `has_mask`/`mask_begin`/`mask_end`. Fixed: use CPU fallback for masked views.
   TODO: add mask support to Metal codegen for performance.

5. **Virtual strides in pool chain**: repeat+expand creates stride-0 views. Subsequent
   reshape/shrink compute strides relative to virtual buffer, exceeding physical buffer.
   Forward works (fuser walks full view chain for correct indexing). Backward provenance
   references view tensors directly — strides are invalid for buffer access. Partially
   mitigated by fixes 3+4 (masked views materialize correctly now).

### DUP accumulation (restored, TAG_TOP only)

DUP was stripped for correctness verification, then restored as optimization:
- `linear_use` now only DUPs TAG_TOP terms (lazy ops), NOT TAG_TEN (weights)
- This prevents counter mismatch: weights appear in forward + SGD, GRAD only walks forward
- DUP reduces dispatches: 396→194 for 1-layer CNN (51% reduction)
- Correctness verified: all gradient tests still PASS with DUP enabled

### Current performance

| Architecture | Dispatches | Time/step | Status |
|---|---|---|---|
| 1-layer CNN + CE (BS=32, 4 params) | 194 | 37ms | ✓ converges |
| 2-layer CNN + CE (BS=32, 6 params) | 342 | ~1s | ✓ runs |
| Full CNN (14 params, BS=128) | — | too slow | needs optimization |
| tinygrad reference | 26 | ~100ms | target |

### Multi-target GRAD (`thvm_grad_multi`) — Pure IC

`thvm_grad_multi(ctx, loss, params, grad_slots, n_params)` returns a LAZY TERM that,
when reduced, walks the provenance chain ONCE and deposits gradients via ASSIGN.

**No mutable context state.** The multi-target encoding is a TAG_CTR heap node:
```
heap[loc]     = N (count)
heap[loc+1..] = param_0, slot_0, param_1, slot_1, ...
```

x = TAG_CTR(loc). When GRAD reaches a leaf matching param_i, it fires
`APP(ASSIGN(slot_i, gy), ERA)` — the ASSIGN deposits, APP-TEN→ERA propagates.

Usage: ONE `thvm_reduce` drives grad + SGD together:
```c
Term grad_term = thvm_grad_multi(ctx, loss, params, grad_slots, N);
Term sgd_chain = ...; // reads from grad_slots
thvm_reduce(ctx, thvm_app(ctx, grad_term, sgd_chain));
```

- Verified: exact match vs per-param thvm_grad on 2-layer CNN (all 6 params, zero diff)
- 1-layer CNN: 66 dispatches, 11ms/step

### Tinygrad comparison (Metal)

1-layer CNN (Conv(1,8,3)→ReLU→Flatten→Linear, BS=32):
| Metric | TinyHVM | Tinygrad | Ratio |
|---|---|---|---|
| Dispatches | 63 | ~26 | 2.4× |
| Time/step | 10-13ms | 8-9ms | **1.3×** |

2-layer CNN (Conv→ReLU→Conv→ReLU→Flatten→Linear, BS=64):
| Metric | TinyHVM | Tinygrad | Ratio |
|---|---|---|---|
| Dispatches | 94 | ~26 | 3.6× |
| Time/step | 128ms | 25ms | 5.1× |
| Accuracy (100 steps) | **84.1%** | **84.0%** | **=** |
| Gradient diff | < 8.3e-7 | — | exact |

### What needs work

- **Non-trailing axis fusion**: axes like [0,3,4] fall back to non-fused (63→26 gap)
- **Metal codegen mask bug**: float4 path ignores masks, CPU fallback used
- **GRAD_STEP tail call**: reduce recursion depth for deep networks
- **REDUCE_MAX_DEPTH**: increased to 256 (was 64)

### Key files

- `src/interact/_.c:38-248` — GRAD handler + BIN_GRAD/UN_GRAD macros
- `src/ctx/init.c:104-151` — `linear_use` (currently no-op)
- `src/ctx/init.c:153-270` — `thvm_op` with shape tracking
- `src/reduce/_.c` — trampoline (enter/apply)
- `src/tinyhvm.h:470-483` — TinyHVM context (dup_frozen, in_grad — to be cleaned up)

---

## How GRAD Fires

```
thvm_grad(ctx, loss, W1)
  → allocates 3 heap slots: [loss, ones(1), W1]
  → returns TAG_TOP(UOP_GRAD, loc)   ← lazy, no computation

thvm_reduce(grad_term)
  ENTER: TAG_TOP → push frame, enter heap[loc+0] (reduce loss)
  ... loss reduces through forward graph to TAG_TEN(loss_id) ...
  APPLY: pop frame, check heap[loc+1] (gy=ones) → TAG_TEN → ready
  → fire thvm_interact(GRAD)

thvm_interact:
  y = TAG_TEN(loss_id), gy = TAG_TEN(ones), x = TAG_TEN(W1)
  creator_op = loss's provenance op
  switch(creator_op) { case UOP_MUL: ... case UOP_ADD: ... }
  → builds GRAD3 terms for inputs
  → BIN_GRAD wraps in ADD, RETURN_REDUCED forces evaluation
  → recursion bottoms out when y_id == x_id (base case: return gy)
  → dead branches: y is leaf and y_id ≠ x_id → return ERA
```

### ERA propagation (current)

- `ADD(ERA, x) → x` — identity rule, already implemented
- `MUL(x, ERA) → ERA`, `MUL(ERA, x) → ERA` — annihilation
- Movement ops: not yet handled (RESHAPE(ERA) gets stuck)

---

## Phase 0: ERA-TOP for movement ops

**Status: not yet implemented**

When ERA meets a movement op (RESHAPE, PERMUTE, EXPAND, SHRINK, PAD), the result should
be ERA. Currently these ops try to compute on ERA and fail.

Add to `thvm_interact`, before the per-UOP compute dispatch:

```c
// Movement ops with ERA input: propagate ERA
if (term_tag(a) == TAG_ERA && !is_binary) {
    switch (uop) {
        case UOP_RESHAPE: case UOP_PERMUTE: case UOP_EXPAND:
        case UOP_SHRINK: case UOP_PAD: case UOP_SUM: case UOP_RMAX:
            RETURN_REDUCED(term_era());
    }
}
```

This replaces the need for REACHES DFS (dead path pruning). Dead GRAD branches naturally
produce ERA, which propagates through the chain and gets absorbed by ADD(ERA, live) → live.

### Files to modify

- `src/interact/_.c`: ~10 lines in the ERA propagation section

---

## Phase 1: DUP Accumulation (optimization)

**Status: disabled, needs redesign**

### Why DUP was broken

The previous DUP mechanism had a 4-slot heap node with a counter:
```
heap[dl+0] = value, heap[dl+1] = grad_accum, heap[dl+2] = pending, heap[dl+3] = original
```

`linear_use` in `thvm_op` created DUP nodes for ANY re-used grad-tracked term. The counter
was set to the total number of forward usages. The GRAD handler's `GRAD3_DUP` macro
accumulated gradients and fired when the counter reached 0.

**Problem 1: Counter mismatch.** `linear_use` counted ALL uses of a term (forward + SGD
chain). But GRAD only walks forward provenance. Example: weight W used in (1) conv2d forward,
(2) SUB for SGD update, (3) ASSIGN for in-place update → counter=3. But GRAD only arrives
once (from conv2d path). Counter stuck at 2, gradient never fires.

**Problem 2: Backward DUPs.** During GRAD handler execution, `thvm_op` calls for gradient
formulas also went through `linear_use`, creating DUP nodes for gradient intermediates.
These polluted `dup_loc` on gradient tensors, causing GRAD3_DUP to misaccumulate.

**Problem 3: DP passthrough dup_loc.** The `dup_frozen` flag was added to prevent DP
passthrough from setting `dup_loc` on gradient tensors during backward. But this only
works in the 2-pass pattern (forward then backward), not in the 1-pass pattern.

### Correct DUP design

DUP accumulation must be scoped to the **gradient computation graph**, not the forward
usage graph. The counter must equal the number of GRAD arrivals, which equals the number
of forward consumers that are ancestors of the loss in the provenance DAG.

**Approach: Forward-only DUP with provenance-based counting.**

1. `linear_use` only counts uses that go through `thvm_op` for differentiable ops
   (not ASSIGN, not the SGD chain).
2. Or: separate the SGD chain from the forward graph. The SGD chain reads current
   parameter values but is not part of the differentiable computation.
3. Or: count arrivals reactively. Use a hash table keyed by (tensor_id, x_id) to detect
   repeated GRAD arrivals at the same tensor. First arrival parks, subsequent arrivals
   accumulate, and... we still need to know when to fire.

**Simplest correct approach:**

Keep DUP nodes but move their creation to `thvm_grad` time. When `thvm_grad(ctx, y, x)` is
called, walk the provenance from y to find fan-out points (tensors with multiple consumers
on the path to loss). Create DUP accumulators only for those. This is a lightweight O(D)
walk (depth of provenance chain), not a full graph traversal.

Alternatively: **let the GRAD handler detect diamonds at runtime.** When GRAD(h, g, x)
fires and h has creator_op, check if any of h's inputs have already been visited by a
GRAD walk for the same x. If so, accumulate. This requires a visited-set per x_id.

**Deferred until Phase 0 and correctness are solid.**

### Files to modify (when ready)

- `src/ctx/init.c`: Revive `linear_use` with proper scoping
- `src/interact/_.c`: GRAD3_DUP with correct counter
- `src/tinyhvm.h`: Clean up `dup_frozen`, `in_grad` fields

---

## Phase 2: Adjoint Table (refactoring)

**Status: not yet started**

Extract per-op gradient rules from the UOP_GRAD switch into a dispatch table. The GRAD
handler becomes:

```c
if (uop == UOP_GRAD) {
    // ... reduce y, gy, x ...
    // ... lookup provenance ...
    AdjointFn adj = adjoint_table[cop];
    if (!adj) RETURN_REDUCED(term_era()); // non-differentiable op
    Term result = adj(ctx, y, gy, x, aid, bid, my, ma, mb_p);
    RETURN_REDUCED(result);
}
```

Each adjoint function is 5-15 lines, independently testable:

```c
// src/adjoint/_.c
static Term adjoint_mul(TinyHVM *ctx, ...) {
    Term da = sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, bt), ...);
    Term db = sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, at), ...);
    return thvm_op(ctx, UOP_ADD, GRAD3(aid, da, x), GRAD3(bid, db, x));
}
```

This is a pure refactoring — same math, cleaner code. The adjoint functions ARE the dagger
structure (Cockett et al. 2020, "Reverse Derivative Categories").

### Files to modify

- New `src/adjoint/_.c`: ~200 lines
- `src/interact/_.c`: Replace 200-line switch with table dispatch

---

## Phase 3: GRAD_STEP Tail Call

**Status: partially implemented**

For unary ops (RELU, NEG, EXP, LOG, SQRT, movement ops), the GRAD handler produces a
single GRAD3 term. Instead of RETURN_REDUCED (which recurses through thvm_reduce), use
GRAD_STEP:

```c
#define GRAD_STEP(result) do { t = (result); goto inet_step; } while(0)
```

This converts O(D) recursion into O(1) iteration for chains of unary ops. Essential for
deep networks where the chain REL→RESHAPE→EXPAND→SUM→... can be 50+ ops deep.

For binary ops, the ADD(GRAD3_left, GRAD3_right) must still go through thvm_reduce (the
ADD needs both args). But unary chains are the common case in practice.

### Files to modify

- `src/interact/_.c`: UN_GRAD macro uses GRAD_STEP instead of RETURN_REDUCED

---

## Phase 4: Higher-Order Gradients

`thvm_grad(ctx, thvm_grad(ctx, y, x), x)` should work if adjoint functions use `thvm_op`
(which records provenance). The second GRAD walk follows provenance of the first GRAD's
output terms.

Requires: all gradient intermediate tensors have provenance recorded via `thvm_op`.
Currently true by construction.

---

## Phase 5: Forward-Mode AD (JVP)

Tangent subnets via `tangent_table[cop]`. Forward-over-reverse for Hessian-vector products.
See previous plan version for details. Deferred.

---

## What This Does NOT Give Us

- **Reverse-mode from DUP**: DUP copies computations, it does not transpose linear maps.
  The adjoint (dagger) must be manually specified.
- **Automatic derivative discovery**: `∂MUL/∂a = b` is in `adjoint_mul`, not derived.
- **Net reversal**: Backprop as reversed continuations (Wang et al. 2019) is theoretically
  promising but requires CPS transform. Not in scope.

## What It DOES Give Us

- **Correct gradients without DUP** — proven via numpy parity
- **ERA-based dead branch elimination** — no REACHES DFS
- **Single reduce drives everything** — forward, backward, SGD in one term
- **Adjoint table** — each op's gradient independently testable
- **Path to DUP optimization** — add back once correctness is solid
