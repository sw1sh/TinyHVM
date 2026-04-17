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

Bisection narrows where the bug appears, not yet why:

| Test                                                | Loss                         | Result                       |
|-----------------------------------------------------|------------------------------|------------------------------|
| `test_tiny_twoparam_noloop`                         | `sum((w+b-t)^2)`             | ✓ both grads exact           |
| `test_tiny_twoparam_lam` (LAM-APP, no recursion)    | `sum((w+b-t)^2)`             | ✗ bundle never materialises to TEN (raw slot stays `TOP/UOP_FUSE`) |
| `test_tiny_twoparam_sum_loop` (recursion, no MUL)   | `sum(w+b)` (only ADD backward) | ✓ both grads = 1.0 exact     |
| `test_tiny_twoparam_grad_loop` (recursion)          | `sum((w+b-t)^2)`             | ✗ w exact, b = 0.8× expected |

**Key narrowing:** the 0.8× bug only manifests when the backward
chain contains `MUL(diff, diff)` — i.e., the `BG` rule in grad.c
which duplicates both `at` and `bt` via `GRAD_SPLIT_AT()` +
`_bt_dup`. A plain `ADD` loss (`BG_GY` only, one branch per target)
works correctly at all n values.

Swapping `params = [b_grad, w_grad]` in `thvm_grad_multi_keep`
still gives b's tensor the 0.8× factor (not w's) — so the bug is
attached to the b tensor's DUP chain, not to bundle slot position.
The DP0-projection of gy always flows to the first arg of each
ADD backward (which resolves to w), DP1 to the second (b).

So multi-target GRAD works correctly when called on plain tensor
inputs (`noloop`), but breaks the moment the params arrive via a
lambda binder. Two distinct failure modes:

1. **LAM-APP never materialises the bundle**: after `thvm_eval`
   on `(λw. λb. multi_keep(...)) w0 b0`, `BUNDLE_GET` reports the
   raw slots still hold `TAG_TOP/UOP_FUSE` terms. The FUSE
   wrappers don't resolve through the scheduler. `thvm_to_host`
   returns NULL. Target `tid` entries stay `~0u` because
   `grad_resolve_target_term` walks DP→heap_read→VAR and breaks on
   the unbound `term_is_sub` cell.
2. **Loop case gives different wrong answer**: recursion makes the
   bundle materialise (the reducer fires through the loop further
   than the LAM-APP case), but b's grad slot lands at 0.8× the
   correct value while w's is exact. Both targets hit their slots
   twice (one STORE + one ADD — the two MUL(diff,diff) backward
   branches). w's two contributions sum correctly; b's sum
   under-counts by 20%.

Likely root cause for (1): targets are captured at
`thvm_grad_multi_keep` time, long before APP-LAM substitution
fills the binder cells. `grad_group_alloc` can't resolve them to
TEN yet, so `tid` is unset. The loop gets around this via
`thvm_book_from_dynamic_r` copying GRAD metadata into a fresh
dyn-loc and letting `find_term_at` resolve via ALO force later —
but that path has the multi-slot numerical bug in (2).

### Experiment: pre-reduce grad before clone (2026-04-18)

Inserting `grad = thvm_reduce(ctx, grad);` before `term_clone`
in `thvm_grad_bundle_accum` — **flips the bug**:

- Normal: w exact, b = 0.8× expected.
- With pre-reduce: w = 0 (no update), b = exact.

Striking. Reducing grad to TEN before cloning makes whichever
target is processed *second* correct and whichever is *first*
lose its deposit. Conclusion: the bug is **order-dependent** and
lives in how the two sides of BG's DUP chain interact when
multiple targets hit the same DUP cells (cell_209, cell_272)
with DP0 (→ w) and DP1 (→ b).

Normal operation: w (first hit) reads DP0 correctly, then b
(second hit) reads DP1 and the DUP has already fired for DP0 via
some side-effect path → b gets a partial view (80%).

Pre-reduced: w's pre-reduce fires full backward chain eagerly,
which consumes the subsequent path's contribution, landing b
correctly but w with nothing.

This isn't a 0.8× multiplier hiding anywhere — it's a linearity
violation where DP0 and DP1 of the same gy DUP cell resolve to
**different** effective values depending on which fires first.

### DUP_DIAG instrumentation added — lets us audit port_slot state at fire time

Added `DUP_FIRE` log (gated on `THVM_DUP_DIAG=1`) inside
`DUP_STATE_RETURN` in
[src/interact/combinators.c](/Users/swish/src/TinyHVM/src/interact/combinators.c):
prints `dup_loc`, firing DP index, val tag, port_s0, port_s1,
and a `missing` flag when either is 0.

### Findings

Counted DP-fire patterns (DP0/DP1, with each port slot
registered or not) across the three loop tests at n=1:

| Test             | DP0 both | DP0 s1=0 | DP1 both | DP1 s0=0 |
|------------------|---------:|---------:|---------:|---------:|
| scalar (PASS)    |       49 |        8 |       46 |       10 |
| twoparam_sum (PASS) |    29 |       10 |        0 |       30 |
| twoparam_grad (0.8×) | 55 |       21 |       10 |       33 |

Both passing tests also have cases where DUP fires with one
port_slot unregistered — so `missing=1` alone isn't the bug.

The cells the bundle reads from (cell 209, 272, etc. in the
BUNDLE_ACCUM log) don't appear in `DUP_FIRE` — the clones in
bundle_accum create fresh cells (221, 230, 280, 294) which *do*
fire. Each cloned DUP fires with only one port registered (the
one matching its firing DP) — expected, since cloning gave each
DP its own cell.

### Experiments tried and ruled out

- Skipping `term_clone` entirely in `thvm_grad_bundle_accum`:
  twoparam still 0.8×, scalar still passes. **term_clone is not
  the cause.**
- Fresh BG-label DUPs (was hardcoded `label=0`): no effect.
- Pre-reducing grad to WNF before clone: **flips** the bug —
  b becomes exact but w becomes zero. Order-dependent.

### Next diagnostic

Instrument the actual *tensor values* of b's two contributions
at deposit time. If each is g/2 (correct per-branch), b's sum
should be g. If each is 0.4g, something's halving. If one is g
and the other is -0.2g (negative), something is subtracting.
Knowing the per-contribution values will reveal whether the bug
is in a specific backward rule or in the accumulator itself.

Currently stuck on the mechanism — need deeper instrumentation
(live tensor value probing mid-reduction) to make further progress.

For matmul (`test_tiny_linear_sgd_loop`) W[0,0] is ~4x expected at
step 1 — separate deeper issue involving MM backward in the loop.
Investigate only after twoparam is closed.
