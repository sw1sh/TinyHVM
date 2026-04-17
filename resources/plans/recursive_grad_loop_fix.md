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

**Hypothesis**: the BG rule's `GRAD_SPLIT_AT()` + `_bt_dup` double
duplication interacts poorly with the LOOP's book-realize +
`thvm_alo_force` fresh-DUP-wrap. In a loop, each unfold
re-realizes the BG's internal DUPs through the book path, and one
of the projections (the one feeding b through DP1) ends up going
through an extra level of splitting that drops ~20% of its
contribution.

Specifically: w's DP0 flow reaches target via one `find_term_at`
resolve → hit. b's DP1 flow reaches target via `find_term_at`
too — but the DP1 projection has to walk one extra heap slot
through the BG's inner `_at_dup` DUP (because `at = DP1(_at_dup)`
is the outer rebind inside the `_db` block). That extra DUP layer,
re-realized through book/ALO, is the likely source of the 0.8×.

Concrete next diagnostic:
- Add a probe in `thvm_grad_bundle_accum` that forces
  `grad` to WNF (TEN) before cloning, and prints the actual
  tensor values for w and b contributions at each deposit. If
  b's deposits are tensor values with wrong numbers (not 0.8× w's),
  the DUP propagation is fine and the bug is in accumulation. If
  b's deposits numerically equal w's deposits, the bug is in how
  the add-chain reduces.

For matmul (`test_tiny_linear_sgd_loop`) W[0,0] is ~4x expected at
step 1 — separate deeper issue involving MM backward in the loop.
Investigate only after twoparam is closed.
