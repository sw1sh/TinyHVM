# Plan: Finish The Recursive GRAD Learning Loop

## Current State

The recursive training loop in
[test/test_tiny_linear_sgd_loop.m](/Users/swish/src/TinyHVM/test/test_tiny_linear_sgd_loop.m)
still applies only the first SGD step under the default staged pipeline.

Current behavior:

```bash
THVM_TRAIN_STEPS=1 ./bin/test_tiny_linear_sgd_loop cpu 1
```

updates `w` and `b` once.

```bash
THVM_TRAIN_STEPS=2 ./bin/test_tiny_linear_sgd_loop cpu 1
```

still returns the one-step parameter values and stops at a structural
`KERNEL(UOP_COUNT, ...)` carrier instead of finishing the second update.

## What Is Already Fixed

- Recursive `GRAD` target copies now preserve stable tensor ids across `REF` /
  `ALO` realization.
- The staged collector no longer walks into the right branch of `SEQ` before the
  left branch settles.
- The same strictness rule now applies to `KERNEL(..., UOP_COUNT)` control
  shells.

These fixes were necessary: the second recursive `GRAD` pass now hits both
parameter targets again.

## Remaining Failure

The remaining bug is no longer in `GRAD` target matching or in early recursive
evaluation. The next recursive update eventually loses its `ASSIGN` destination:
the destination branch collapses to `ERA`, so the loop stalls after one
effective update.

Current best hypothesis:

- the remaining corruption happens in the `DP` / `ALO` alias path for threaded
  parameter handles (`w_next`, `b_next`, `w_assign`, `b_assign`)
- not in lowering or kernel dispatch

## Next Steps

1. Regenerate the `n=2` step trace and identify the first frame where the
   second-step `ASSIGN` target stops being the live parameter tensor.
2. Trace that target backward through `DP0` / `DP1`, `ALO`, and any shared alias
   fast-path in [src/interact/_.c](/Users/swish/src/TinyHVM/src/interact/_.c).
3. Check whether `ALO` realization is incorrectly collapsing a projected alias
   to a shared leaf, or otherwise dropping a `DP` wrapper too early.
4. If the imperative aliasing model is fundamentally unstable, switch this loop
   to the pure functional update shape:
   `train(m)(w')(b')` instead of threading the same in-place handles through
   recursive `ASSIGN`.

## Verification Targets

- `THVM_TRAIN_STEPS=1` still performs one correct update.
- `THVM_TRAIN_STEPS=2` performs a second distinct update.
- `THVM_TRAIN_STEPS=3` continues progressing.
- No `THVM_EVAL_MIXED_DISPATCH` override is needed.
