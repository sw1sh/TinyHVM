# Plan: Recursive GRAD Learning Loop — Status & Open Work

## Status (2026-04-17)

**Single-parameter elementwise loop: done.**

- `test_tiny_scalar_grad_loop` n=1, n=2, n=3 pass assertions
  against host-simulated SGD trajectory.
  - n=1: `[0.8000, 0.0000, -0.4000]` ✓
  - n=2: `[1.0400, 0.8000, -0.9200]` ✓
  - n=3: `[1.2320, 1.4400, -1.3360]` ✓
- `test_grad_fuse_minimal` passes.
- `test_tiny_linear_sgd_loop` (matmul-based) returns `CTR(w, b)` with
  readable tensors (smoke pass, no value asserts).

**Multi-parameter / matmul in loop: broken numerically.**

- `test_tiny_twoparam_grad_loop` (NEW diagnostic, elementwise
  `loss = sum((w + b - t)^2)` with multi-target GRAD + MAT):
  - n=1: w's grad is **exact**; b's grad is uniformly **0.8x**
    expected. Pattern holds across positions (uniform ratio).
  - n=2/3: ratios drift further — step 2 w becomes 1.025x, b becomes
    0.82x.
  - XFAIL'd (prints "XFAIL" instead of asserting).
- `test_tiny_linear_sgd_loop` numerics vs. host-simulated SGD:
  n=1 W[0,0] observed 0.1441, expected 0.1109. Not a clean scaling
  factor — matmul backward in the loop computes different values.

## Root cause

Two independent bugs:

1. **ALO multi-force vs. DP linearity** —
   [src/interact/_.c:914-932](/Users/swish/src/TinyHVM/src/interact/_.c#L914).
   Multiple consumer sites that share an ALO cell would each read
   the same memoised DP0/DP1 token from `heap[alo_loc+0]`. Since
   `port_slot` tracks only one consumer per DUP port, stale
   consumers cascaded ERA on later visits.
   **Fix:** fresh-DUP-wrap at second-and-later force — allocate a
   new DUP cell wrapping the memo, hand out one projection, push
   the other back as the new memo.

2. **Shape tracking lost on DP/VAR/ALO args in GRAD** —
   [src/interact/grad.c:291-292](/Users/swish/src/TinyHVM/src/interact/grad.c#L291).
   When an op's arg was a DP / VAR / ALO (not yet resolved to TEN
   or TOP), `a_shape` / `b_shape` defaulted to `SHAPE(1)` — a
   rank-1 scalar. `sum_to_shape(gy, y_shape, [1])` then
   **reduced the per-element gradient to a scalar** and
   broadcast it back, destroying the gradient's spatial
   structure.
   **Fix:** default `a_shape` / `b_shape` to `y_shape` (the
   output shape). Elementwise ops preserve shape, so when the
   arg isn't yet resolvable, `y_shape` is the correct guess.
   Reshape / expand / permute override via the TEN / TOP lookup
   paths that were already correct.

   Confirmed by instrumentation: step 1 reported
   `a=[3] b=[3]` for the SUB, step 2 reported `a=[1] b=[3]` —
   the scalar default fired at step 2 because the SUB's first
   arg resolved through a DP chain that hadn't been materialised
   yet.

## What made this hard to find

- The `scalar` test asserts exact SGD trajectory — it was the
  only signal that the `linear_sgd` smoke test was also wrong
  (same bug, but `linear_sgd` has no value asserts).
- The numeric pattern (`update - w_prev = uniform 0.52`) pointed
  at a reduce-then-broadcast bug, not the DUP / ASSIGN race that
  prior plan revisions fixated on.
- Two unrelated bugs had to be fixed together. The ALO bug
  surfaced first (it stalls the reducer at n≥2 without the
  fresh-DUP-wrap), hiding the shape-tracking bug until the first
  one was fixed.

## Verified correct (do not revisit)

- Fresh-DUP-wrap in `thvm_alo_force`.
- `thvm_alo_suspend_child` DP0/DP1 alias-skip.
- `APP-LAM` caller-APP-cell clear.
- FUSE-gated on structural shape (not APP / REF / ALO).
- FUSE ⊳ UOP_ASSIGN idempotent.
- Rule 2 (`ASSIGN ⊳ SUP`) landed.
- Backward through UOP_ASSIGN routing.
- Recursive GRAD target copy stability.
- Graph-dump directory announce.
- **Default `a_shape` / `b_shape` = `y_shape`** in grad-backward.

## Dead ends (do not retry)

- Rule 1 (`DUP ⊳ UOP_ASSIGN = t`). On top of fresh-DUP-wrap it
  causes unbounded heap growth; without the shape-tracking fix
  it's solving the wrong problem.
- Eager eval in interaction rules.
- Clearing `heap[alo_loc+0]` to ERA on first DP force.
- Reverting the fresh-DUP-wrap.
- Program-side `DETACH` on the recursive arg (breaks tensor-id
  identity).
- `ASSIGN ⊳ ERA` src short-circuit (masking, not fixing).
- An "ASSIGN driver" in the reducer fixed-point.
- Selective fuse sweeps, share-by-reference DUP ⊳ ASSIGN.

## Open work — multi-target GRAD in the loop

The twoparam XFAIL is the next investigation. Observed:

- w's gradient is correct at step 1 (matches host simulation).
- b's gradient is uniformly **0.8x** expected at step 1; drifts
  further at step 2/3.
- w and b have structurally identical 5-DUP chains and both feed
  `thvm_grad_multi_keep(loss, [w, b], 2)`. Suggests the bug is in
  how the second slot of a multi-keep bundle is wired (first slot
  seems fine).

Hypotheses to test next:

1. **Bundle slot routing** — `thvm_grad_multi_keep` might be
   incorrectly sharing the keep-target's DP chain between slots,
   with slot 1 getting a partially-consumed view.
2. **MAT destructuring** — the `thvm_mat(ctx, 2, lam_gw, era)`
   wrapping `λgw. λgb. body` might leak one of the bundled
   gradients.
3. **ASSIGN ordering** — both ASSIGNs run inside a `SEQ(assign_w,
   SEQ(assign_b, rec))`; the second ASSIGN may see a stale DP
   projection of its dst because the first ASSIGN already
   consumed some consumer.

For matmul: deeper issue. Linear_sgd_loop W[0,0] is ~4x expected
at step 1 — not a clean factor. Likely matmul backward in the
loop uses a stale or wrong input activations tensor after the
first step.
