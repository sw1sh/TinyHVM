# Training Loop: Recursion Research

Status: **active research** — the correct encoding of training loops within TinyHVM's
interaction calculus is an open design problem.

## Decided Architecture (from Codex session 019d747c)

The loop is **purely functional** — no DETACH, no ASSIGN, no side effects:

```
train(counter)(w)(b) =
    IFZ counter
        → CTR(w, b)                              // base case: return params
        → λm.
            pred  = x @ w + b                     // forward (lazy TAG_TOP)
            loss  = mean((pred - y)²)
            grads = thvm_grad_multi_keep([w, b])   // backward → CTR(gw, gb)
            MAT grads λgw. λgb.                    // destructure (APP-MAT-CTR)
                w' = w - lr * gw                   // SGD update (lazy TAG_TOP)
                b' = b - lr * gb
                train(m)(w')(b')                    // recurse with lazy params
```

Key decisions:
- **Remove DETACH** — updated params are lazy TAG_TOP terms, not materialized
- **Remove ASSIGN** — no side-effectful weight updates
- **APP-MAT-CTR** for gradient bundle destructuring (HVM4-aligned)
- **CTR** for multi-value bundling
- Standard beta reduction for APP-LAM
- **REF** unfolds lazily via deep clone

## Execution Model (Target)

The crucial insight: **without DETACH**, `w'` and `b'` are TAG_TOP compute expressions.
They are WNF in phase 1. When substituted into the next iteration, the forward compute
builds TAG_TOP on TAG_TOP — this is just lazy graph building, not evaluation.

**But**: IFZ's counter is a concrete scalar (TAG_TEN), so IFZ fires immediately.
And REF unfolds unconditionally. So phase 1 still unrolls all iterations structurally.

**This is actually correct for the target model:**

```
Phase 1 (reduce):
    Unfold all N iterations structurally
    → Each iteration builds lazy TAG_TOP forward/backward/update chains
    → No compute fires — everything is WNF TAG_TOP
    → Result: deep nested TAG_TOP graph for N iterations
    → Combinators (IFZ/REF/APP/LAM/MAT) all fire and disappear

Phase 2 (UOP_FUSE):
    Walk the entire TAG_TOP graph
    → Fuse into kernels
    → Key insight: the scheduler sees the SAME compute pattern repeated N times
    → Multi-consumer deduplication: same subgraph structure → same kernel ID
    → Result: small set of unique kernels, reused across iterations

Phase 3 (dispatch):
    Fire kernels → TAG_TEN results
    → kid_results[] caches by kernel ID
    → Same kernel structure across iterations hits cache
    → N iterations worth of compute dispatched efficiently
```

The difference from current behavior: **DETACH currently calls thvm_eval per iteration**,
which means each iteration runs the full 3-phase pipeline independently. Without DETACH,
phase 1 builds the entire lazy graph, and phases 2-3 handle it as one unified schedule.

## What Needs to Change

### 1. Remove DETACH from the loop test
Replace:
```c
Term next_w_leaf = thvm_detach(ctx, next_w);
Term next_b_leaf = thvm_detach(ctx, next_b);
// ... recurse with next_w_leaf, next_b_leaf
```
With:
```c
// ... recurse directly with next_w, next_b (lazy TAG_TOP)
```

### 2. Fix term_clone for TAG_TOP shape metadata
Already done in Codex session — `term_clone(TAG_TOP)` now copies View metadata.
Verify this is committed.

### 3. Scheduler must handle deep TAG_TOP graphs efficiently
With N iterations unrolled as lazy TAG_TOP, the scheduler sees a deep graph.
The fusion boundary walker must:
- Recognize repeated subgraph structure
- Deduplicate kernels across iterations
- Not blow up in memory for large N

### 4. Verify kernel reuse across iterations
After phase 2, check that kernel count is O(1) in N, not O(N).
The same forward/backward/update pattern should produce the same kernel IDs.

## Open Questions

### Graph depth vs kernel count
With N=100 iterations unrolled lazily, the TAG_TOP graph is 100x deeper.
Is the scheduler's boundary walker O(graph_size) or O(unique_patterns)?
If O(graph_size), scheduling 100 iterations is 100x more work even with dedup.

### Memory for unrolled graph
N=100 iterations of lazy TAG_TOP chains means O(N) heap nodes in phase 1.
For large N, this could exhaust heap. Is there a way to schedule incrementally
(one iteration at a time) while still getting kernel reuse?

### Alternative: explicit iteration boundary
Instead of unrolling all N iterations in phase 1, could the program be structured
so that phase 1 exposes one iteration, phases 2-3 schedule and dispatch it, and
then reduction continues to the next iteration? This would require a mechanism
for the reducer to "pause" between iterations — which is essentially what
thvm_eval already does, but driven by the program structure rather than the
runtime.

## Design Constraints

These are firm decisions — do not revisit:

1. **No explicit loop/state combinator** — recursion uses standard HVM REF/LAM/APP/MAT
2. **Phase 1 is an initializer** — exposes the compute frontier
3. **Phase 3 re-fires same prescheduled kernels** across loop iterations
4. **Scheduler walks the whole graph** (even behind combinators), not just root terms
5. **Running eval in a loop is the wrong model** — single reduce should complete the loop
6. **Only single iteration is planned and JIT'd** — the loop body reuses the same kernels
7. **No DETACH in the loop** — parameters are lazy TAG_TOP, not materialized
8. **No ASSIGN in the loop** — purely functional parameter threading

## Test

`test/test_tiny_linear_sgd_loop.m` — functional recursive linear SGD with:
- 4x3 weight matrix, 4-element bias
- 2 training steps (configurable via `THVM_TRAIN_STEPS`)
- CTR bundle for gradient returns
- MAT destructuring for gradient extraction
- **Currently still uses DETACH** — needs to be removed per this plan

## Files

- `test/test_tiny_linear_sgd_loop.m` — functional loop test
- `src/interact/combinators.c` — APP/LAM/MAT/CTR/IFZ rules
- `src/interact/tensor_ops.c` — DETACH/IFZ handlers
- `src/schedule/_.c` — thvm_eval 3-phase orchestration
- `src/reduce/_.c` — WNF conditions, direct UOP list
- `src/clone/_.c` — term_clone (TAG_TOP shape metadata fix)
- HVM4 reference: `/Users/swish/src/HVM4/docs/hvm/interactions`
