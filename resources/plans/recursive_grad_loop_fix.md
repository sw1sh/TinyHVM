# Plan: Recursive GRAD Learning Loop — Status & Open Work

## Status (2026-04-18, late) — MATMUL-IN-LOOP STILL OPEN

The twoparam_grad case is resolved (see below). A separate bug was
found in `test_tiny_linear_sgd_loop` (matmul in a training loop)
that cron-loop bisection has localised but not fixed.

### Matmul-in-loop bug — current narrowing

Minimum trigger (commit `0aa157e` and follow-ups):
- Forward contains a reduction (SUM/RMAX) producing a TEN with
  singleton dims (stride 0 on reduced axes)
- Downstream self-MUL on that reduction's output (e.g. MUL(X, X)
  where X = sum_axes(expand(w, …), {ax}))
- grad_multi_keep on this loss
- Forward expression evaluated through REF + book→dyn ALO realize

Observed: gw collapses to scalar — gw[k, j] = s · sum_i(x[i, k])
uniform over j, with s = the [0,0] element of the correct gy.

Bisection matrix:
| Path                                       | Result |
|--------------------------------------------|--------|
| Pure compute, no grad                      | PASS   |
| grad + ASSIGN, no loop/REF                 | PASS   |
| grad + MAT-CTR, pre-eval'd bundle, no REF  | PASS   |
| grad + REF with pre-eval'd bundle          | PASS   |
| grad + MAT-CTR defer via REF               | FAIL   |
| grad + REF body, no MAT-CTR                | FAIL   |
| grad OUTSIDE REF on loss from REF body     | FAIL   |
| direct LAM-APP in dyn heap                 | PASS   |
| forward-only (no grad) via REF             | PASS   |
| forward user-DUP⊳SUM via REF               | PASS   |

Key observations:
- MAT-CTR defer is NOT the trigger (commit `1a9dd51`).
- Direct LAM-APP works; the issue is specifically book→dyn ALO
  realization (commit `0aa157e`).
- Forward IC through ALO works (test_dup_sum_forward_ref,
  commit `afec3da`); only backward via BG on an ALO-realized
  SUM output breaks.
- EXPAND + self-MUL through ALO works (commit `a2dc053`); the
  reduction's singleton-dim output is what triggers.
- DETACH cleanly short-circuits — no memory corruption,
  confirming the collapse needs live gradient flow (commit
  `7ad295a`).

### Attempted fixes that did NOT work
1. `term_clone_r` TAG_TOP ShapeTracker preservation (`2e0e847`)
2. DUP⊳TOP commute ShapeTracker preservation (`ed54175`)
3. `thvm_kernel_normalize_compute` tracker preservation (`34e26fb`)
4. BG rule fresh labels instead of 0 (`8274630`) — defensively
   correct, no behavioural change on this bug
5. Witness heap slot for DP-token grad targets — reverted
6. `thvm_track_top_shape` recompute at ALO-realize time —
   catastrophic garbage values (ALO children views aren't
   resolved yet), reverted

### Tests isolating the bug
- `test_matmul_grad_mat_ref_noassign` — minimal failing repro
  (matmul variant)
- `test_sum_sq_ref_nomat` — simpler failing repro (SUM +
  self-MUL, no matmul needed)
- `test_sum_sq_noref` — passing counterpart (no REF)
- `test_sum_sq_lam_app` — passing direct LAM-APP variant
- `test_dup_sum_forward_ref` — forward user-DUP via REF works
- `test_kernel_out_sq_ref` — self-MUL on elementwise kernel
  output (no reduction) works

### Fix directions for future iterations
1. **ALO-force hook**: when an ALO-suspended SUM child finally
   resolves, recompute its parent's tracker using the resolved
   child's view. Tracker rebuild at ALO-realize time failed
   because children are still suspended then.
2. **Eager reduction realization**: at ALO realize for a
   reduction TOP (SUM/RMAX), force the reduction to dispatch
   early so it's a concrete TEN (atomic, no DUP commute
   issues). Violates "no eager eval in rules" policy but may
   be the only IC-clean fix.
3. **DUP⊳ALO'd-TOP specialization**: custom commute rule
   for DP firing on a TOP whose source is an ALO. Resolve
   the ALO first, then commute with the resolved dyn term.
4. **Fuse BG DUP with forward dedup**: forward MUL(X, X)
   dedups X in the kernel builder. Make backward's BG rule
   do the same — when at==bt semantically (even through
   ALO indirection), avoid parallel DUPs.

## Status (2026-04-18) — RESOLVED

Multi-target GRAD in the recursive training loop works correctly for the
elementwise elementwise case. Rounds 8–15 landed the stale-read fix.

| Test                             | n=1  | n=2  | n=3  | 20× determinism |
|----------------------------------|------|------|------|-----------------|
| `test_tiny_scalar_grad_loop`     | PASS | PASS | PASS | 20/20           |
| `test_tiny_twoparam_sum_loop`    | PASS | PASS | PASS | 20/20           |
| `test_tiny_twoparam_grad_loop`   | PASS | PASS | PASS | 20/20           |
| `test_grad_fuse_minimal`         | PASS | —    | —    | 20/20           |

XFAIL in `test_tiny_twoparam_grad_loop` flipped to a hard assertion.

## Resolution (round 15)

APP-MAT-CTR now defers (returns `t` unchanged, no `ctx->itrs++`)
when any CTR child is a TOP with a non-view uop
(ADD/MUL/SUB/SUM/RMAX/MM/.../KERNEL/EXEC). View-op children
(RESHAPE/PERMUTE/EXPAND/SHRINK/PAD) and FUSE are bound
directly — they don't feed a multi-leaf backward kernel that
would read forward tensors at dispatch time.

The change is a ~15-line addition in
[src/interact/combinators.c:139](/Users/swish/src/TinyHVM/src/interact/combinators.c#L139)
that inspects child TOP uops before firing the bind chain. No
changes to grad.c, tensor_ops.c, scheduler, or the IC framework.

### What the defer actually unlocks

Clarification: the defer has **nothing to do with fusion**. IC
fusion is the `FUSE ⊳ compute_uop` interaction rule in
[src/interact/tensor_ops.c:386](/Users/swish/src/TinyHVM/src/interact/tensor_ops.c#L386),
driven by the root FUSE wrapper. It doesn't reach the CTR's
children through `APP(MAT, CTR)` because `FUSE ⊳ APP` has no
explicit rule and passes through.

The materialisation happens via the scheduler pipeline in the
outer `thvm_eval_internal` loop
([src/schedule/_.c:2562](/Users/swish/src/TinyHVM/src/schedule/_.c#L2562)):

1. Phase 1 (`thvm_eval_reduce_fused`): MAT-CTR defers, nothing
   progresses through it.
2. Phase 2 (`thvm_run_global_passes` → `thvm_sched_global_pass`
   → `sched_all`): walks heap reachable from root, finds the
   compute TOPs at `ctr_loc[i]` as boundary roots, calls
   `fuse_build_kernel` on each, installs EXEC triggers,
   rewrites ADDs in place to EXEC terms.
3. Phase 3 (`thvm_eval_exec_fixed_point`): EXEC fires,
   dispatches kernel, produces TEN. `ctr_loc[i]` now holds
   TEN (or KERNEL if an EXEC isn't installed yet; either
   way, not the original compute TOP).
4. Phase 4 (follow-up round re-enters phase 1): APP-MAT-CTR
   re-fires, children are no longer raw compute TOPs → falls
   through to the bind loop → succeeds.

So the defer says "don't bind yet; let the outer scheduler
pipeline complete before I commit the binding" — not "wait
for fusion."

The view-op exemption is about stale-read risk, not about
the scheduler's reach. EXPAND(scalar) in particular is
trivial and the scheduler may never schedule it (no compute
to fuse), so deferring on views would livelock. Views are
also provably safe to bind lazily — they just re-view a
single source buffer, not a fused read of multiple forward
tensors, so there's no stale-read issue even if ASSIGNs
later mutate a forward input.

### Why this corresponds to user direction (b)

The user asked for "retry-from-fixed-point: after MAT-CTR
defers ... the outer reducer fixed-point would need to
re-drive the stuck APP-MAT pair once children materialize."
The retry is implicit in the existing pipeline: each
follow-up round in `thvm_eval_internal` re-enters phase 1,
which fires APP-MAT-CTR again with the now-materialised
children. No new scheduler machinery was required.

Earlier lazy attempts (rounds 12-13) failed because they
tried to express the defer as an IC-level primitive (wrap
CTR in FUSE, emit SEQ chains, etc.) — that broke the
trampoline's interaction of APP-MAT with CTR. The working
form is just "return `t` unchanged" so the phase pipeline
continues around it.

## Status (2026-04-17) — prior status kept below for history

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

### Clone+reduce probe tried (2026-04-18) — reveals chain isn't self-contained

Added `THVM_BUNDLE_PROBE=1` heavy probe: clones the grad term and
calls `thvm_reduce` on the clone, then prints the reduced Term.

**Finding:** at `BUNDLE_ACCUM` time, the cloned grad reduces only
to a `TAG_TOP/UOP_MUL` (tag 11, ext 10) — stops at an unresolved
MUL. The grad chain **is not self-contained**: it depends on
ambient heap state (forward tensors, scheduled kernel outputs)
that the clone has no access to. Cannot probe contribution
values via clone+reduce during mid-reduction.

### Round 2 (2026-04-18): MAT-CTR probe lands

Added a diagnostic at `APP-MAT-CTR` in `src/interact/combinators.c`
that prints each CTR child and recurses one level into TOP args
(showing the `heap[dup_loc]` of each DP).

**Finding:** at MAT-CTR time, both bundle slots hold unresolved
ADD ops:

```
MAT_CTR match_tag=2 ctr_tag=2 ctr_loc=219
  [0]=11/9@291 (ADD)
      arg0=DP0/20@221 dup[221]=DP0/20@222
      arg1=DP0/30@280 dup[280]=DP0/30@281
  [1]=11/9@305 (ADD)
      arg0=DP1/22@230 dup[230]=DP0/22@231
      arg1=DP1/32@294 dup[294]=DP0/32@295
```

Two observations:

1. **CTR children are not WNF** when MAT fires — they're still
   ADD compute ops. The reducer lets MAT destructure on a CTR
   with lazy children, because CTR itself is WNF (arity 2). So
   the bundle values aren't materialised at MAT time; they
   materialise later when ASSIGN's src is reduced.
2. **Outer wrapper asymmetry**: w's ADD args are DP0, b's args
   are DP1. Both inner `heap[dup_loc]` values are DP0 — because
   both clones cloned the *same* source `heap[cell_209]` which
   contained `DP0(cell_192)`. The outer wrapper differs only
   because `bundle_accum` cloned DP0 for w and DP1 for b.
   Structurally equivalent below the outer layer.

So the asymmetry at the term level is limited to that outermost
DP0-vs-DP1 wrapper. Everything inside the chain is the same
cloned structure. Yet b's final value is 0.8× w's.

### Round 3 (2026-04-18): DP0 vs DP1 chain firing asymmetry

Captured full `DUP_FIRE` sequence and diffed w vs b chains.

**Finding — the firing patterns are NOT symmetric:**

```
w's chain (outer label 20):
  dup=222 DP0  [ORPHAN s1=0]  (inner cdup, w only consumer)
  dup=281 DP0  [ORPHAN s1=0]  (inner cdup, w only consumer)
  dup=221 DP0  [ORPHAN s1=0]  (outer DUP from clone, w only)
  dup=280 DP0  [ORPHAN s1=0]
  dup=377 DP0  [s0=389 s1=375 non-orphan]  ← cdup with both
  dup=389 DP0  [s0=385 s1=387 non-orphan]
  dup=378 DP0
  ...all DP0 all the way down.

b's chain (outer label 22):
  dup=231 DP0  [ORPHAN s1=0]  (inner)
  dup=295 DP0  [ORPHAN s1=0]
  dup=230 DP1  [ORPHAN s0=0]  (OUTER — the DP1 twist!)
  dup=294 DP1  [ORPHAN s0=0]
  dup=527 DP0  [s0=539 s1=525 non-orphan]
  dup=539 DP1  [s0=535 s1=537 non-orphan]  ← ALTERNATES!
  dup=528 DP0
  dup=540 DP1
  ...alternates DP0, DP1 throughout.
```

**Root pattern**: `bundle_accum`'s `term_clone` on `DP1(cell_209)`
produces outer wrapper DP1(new_cell=230). Inner `heap[cell_209]`
= DP0(cell_192) so the inner is DP0. Result: b's outermost DUP
fires via DP1 (because the consumer is DP1 at ADD's arg slot),
while inner DUPs created by the commute have BOTH port_slots
registered (because both n0's and n1's args point to the same
cdups via `heap_set`).

When a non-orphan inner DUP fires via DP1, `DUP_STATE_RETURN`
writes `_v1` to the live consumer (b's arg slot) AND writes
`_v0` to the DEAD n0-side arg slot (the "sibling" MUL that was
orphaned at outer commute time). The write to the dead slot
has `port_forget`/`port_remember` side effects that can clear
or re-register DUP port_slots in the cdup's children.

### Round 4 (2026-04-18): teardown hypothesis FALSIFIED

Added `TEARDOWN` diagnostic inside `DUP_STATE_RETURN`: flags any
case where the cell being overwritten (`_c0` / `_c1`) is a DP of
SOME OTHER DUP (i.e., `term_val(_c) != dup_loc`), which would
mean the side-effect `heap_set` clears another DUP's port_slot.

**Count across all three tests at n=1: 0 TEARDOWN events.** The
overwritten cells are DPs of the firing dup itself (the standard
consumer-side writes), never of other DUPs. So the side-effect
writes do NOT cascade port_slot teardowns.

**Conclusion**: the DP0/DP1 alternation in b's chain is not
corrupting port_slot state. Both sides' writes land on their
intended consumers, and both n0 and n1 of each commute are
reduced by the trampoline. The bug must be in the NUMERIC
VALUES produced by the DP1-side chain, not in broken state.

### Round 5 (2026-04-18): FORCE_V0 experiment — v0 and v1 are NOT interchangeable

Added `THVM_DUP_FORCE_V0=1` env knob inside `DUP_STATE_RETURN`
that writes `_v0` to BOTH port_slots (replacing `_v1`). This
tests the hypothesis "DP1-side commute is the diverging path".

**Results:**

| Test | Baseline | With `FORCE_V0=1` |
|---|---|---|
| scalar (1 param) | pass [0.8, 0, -0.4] | **BREAKS** [0.65, -0.5, -0.075] |
| twoparam_sum | pass (no MUL backward) | pass |
| twoparam_grad | w=0.78, b=0.324 (0.8×) | **both worse**: w=0.64, b=0.226 |

**Findings:**

1. Forcing v0 on both sides does NOT fix b. w's result gets
   worse too. So DP1-side reduction is not "wrong" in a way
   that swapping to DP0 fixes.
2. `v0` and `v1` from a commute are NOT numerically
   interchangeable at the consumer level. This contradicts the
   naive IC semantics that `n0` and `n1` compute the same
   value — something about how their argument DPs share cdups
   creates real numerical divergence when you substitute one
   for the other.
3. Scalar broke because it ALSO has non-orphan commute DUPs
   (the diag shows ~49 DP0-both + 10 DP1-both firings in the
   scalar backward chain). Forcing v0 on DP1 consumers in
   scalar corrupts its normally-correct grad.

### Round 6 (2026-04-18): DUP⊳ERA hypothesis also FALSIFIED

Added `DUP_ON_ERA` log for the TAG_ERA branch of the DUP rule.

**Result**: 0 `DUP_ON_ERA` events across all three tests at
n=1. No DP ever fires on a pre-erased DUP cell. cdup-timing-ERA
hypothesis is wrong.

Also added a `heap[grad.val]` probe at bundle_accum_store:

```
BUNDLE_ACCUM_STORE loc=210 idx=0 src=4/0@209 heap[209]=4/0@192
BUNDLE_ACCUM_STORE loc=212 idx=1 src=5/0@209 heap[209]=4/0@192
```

`heap[209]` is IDENTICAL at both w's and b's deposit times
(DP0(192) both times). So both `term_clone` calls clone the
same source structure. The clones themselves end up with
structurally-equivalent inner chains (confirmed round 2).

## STUCK — proposing bisection that needs user input

After 6 rounds, ruled out:

- `term_clone` (not the cause)
- Fresh BG DUP labels (no effect)
- Port_slot teardowns via side-effect writes (0 events)
- DP0/DP1 interchangeability via `FORCE_V0` (breaks scalar, doesn't fix b)
- `DUP ⊳ ERA` cascades (0 events)
- `heap[gy_dup]` mutation between w and b hits (unchanged)

Confirmed bug locus:
- Only manifests when **all three** of the following hold:
  recursion, multi-target GRAD, and MUL(diff, diff) backward (BG rule).
- Dropping any one of these three makes the test pass.

Confirmed bug signature:
- b's gradient is uniformly 0.8× expected (ratios identical
  across all 3 positions).
- Swapping params = [b, w] in `multi_keep` keeps b at 0.8×
  (attached to b's tensor, not slot position).
- Pre-reducing grad in bundle_accum_store flips it: b becomes
  exact, w becomes zero. Suggests order-dependent interaction
  between the two deposits' reductions.

### Proposals requiring user input

1. **Re-examine the MUL `BG` rule's GRAD_SPLIT_AT + `_bt_dup` double
   duplication.** This is the only structural thing that appears only
   in the MUL case. Specifically: when `at == bt` (same tensor, as in
   `MUL(diff, diff)`), the two DUPs have REDUNDANT purpose — each
   duplicates `diff` on its own. The grad chain then has 4 DP tokens
   all pointing at `diff`'s materialised TEN (via separate DUP cells).
   Share-by-reference should make them all equivalent, but in the
   recursive case the ALO/book realization may create non-symmetric
   clone structures where one of those 4 tokens resolves differently.

   User could bisect by rewriting MUL's `BG` as a single-DUP rule
   specialised for `at==bt` — avoid creating `_bt_dup` entirely when
   the same tensor is on both sides.

2. **Compare MM(x, w) backward vs MUL(diff, diff) backward** in the
   loop. Both use `BG`, both duplicate both args. But matmul
   (`test_tiny_linear_sgd_loop`) is ~4× wrong, not 0.8×. If user sees
   MM misbehaving differently, that narrows the issue further (or
   confirms it's all one rule).

3. **Try without the fresh-DUP-wrap ALO fix**. The plan marked this
   as "verified correct", but our bug may be specific to a race
   between ALO fresh-DUP-wrap and multi-target backward. Test:
   temporarily remove the fresh-DUP-wrap and run scalar n=1 +
   twoparam_grad n=1. Expected: scalar n=1 fails (that was the
   reason we landed the wrap). But if twoparam_grad's b gradient
   improves, it's evidence that the wrap's handling of multi-consumer
   ALOs is creating the asymmetry.

Currently stuck without user guidance on which of these three
directions to pursue; each is a non-trivial change touching
code currently "verified correct" in the plan.

### Round 7 (2026-04-18): SCHED_DIAG confirms symmetric chain

Ran `THVM_SCHED_DIAG=1` on twoparam_grad n=1 and walked the
`GRAD_ENTRY` / `GRAD_TOP` sequence. Both MUL(diff,diff) backward
branches produce the same structural chain through SUB → ADD →
target hit:

```
Branch 1: SUM → MUL(diff,diff) → SUB(w+b, t) → ADD(w, b)
           hit w (gy=DP0) + hit b (gy=DP1)  [branch 1's contributions]
Branch 2: (MUL's other BG side) → SUB → ADD
           hit w (gy=DP0) + hit b (gy=DP1)  [branch 2's contributions]
```

All 4 target hits have structurally identical gy (DP0-tagged
for w, DP1-tagged for b). The DP-structure symmetry holds.
The numerical asymmetry must therefore come from how the
reducer resolves those DP chains, not from the backward-rule
construction.

### Round 7 addition to bisection proposals

**4. Dump all tensor IDs produced during the backward chain**
(both MUL intermediates and the final w/b grads). If b's path
reuses a buffer that's then overwritten by w's path or by an
ENSURE side-effect, the numerical corruption could originate at
the scheduler/materializer level — not in the interaction rules
at all. Would need a probe at `thvm_op_raw` and `ENSURE` sites
that logs `(op, input_tids, output_tid, buf_id)` for the grad
chain. If user can enable this trace and run twoparam_grad n=1,
the log would reveal any buffer aliasing or early-ENSURE.

## ROUND 8 (2026-04-18): BUG FOUND — stale forward leaf in backward kernel

User hint: "debugging wrong layer, did you verify fused kernels
and codegen? cpu vs metal?" — this was the unlock.

**CPU and Metal produce IDENTICAL 0.8× output.** So the bug is
in the shared layer (scheduler/fuser), not in backend numerics.

**Kernel diag (`THVM_LOOP_DIAG=1`) reveals the bug:**

```
Kernel for w (out=11, dispatched first):
  op[0]=ADD a=0 b=3     -- ADD leaf[0] leaf[3] = w + b
  leaf[0]=t1 v0=0.500   -- w (correct pre-update value)
  leaf[3]=t2 v0=0.100   -- b (original)
  ...
  → SCHED_DISPATCH_DONE out0=0.780   -- w's update computed correctly

Kernel for b (out=12, dispatched AFTER assign_w has blit t1):
  op[0]=ADD a=3 b=0     -- same ADD, args swapped
  leaf[0]=t2 v0=0.100   -- b
  leaf[3]=t1 v0=0.780   ← *** W's POST-UPDATE VALUE, NOT 0.5! ***
  ...
  → SCHED_DISPATCH_DONE out0=0.324
```

**Root cause:** the backward kernel for b references `t1` as a
leaf (the live w buffer). After `assign_w` fires and blits the
computed update into `t1`, b's backward kernel dispatches and
reads the mutated buffer instead of the pre-ASSIGN value.

Numerical verification:
```
With w_post=0.78, b=0.1, target=2:
  diff = 0.78 + 0.1 - 2 = -1.12
  g = 2 * diff = -2.24
  b_new = 0.1 - 0.1 * (-2.24) = 0.324
```
Exactly matches observed `b = [0.324, 0.968, -0.772]`.

**Why scalar passes:** only one target; no second ASSIGN mutates a
forward leaf between backward dispatches.

**Why twoparam_sum passes:** `sum(w+b)` backward doesn't depend
on forward tensor VALUES (gradient is uniform 1). Stale reads
don't matter when the read data is unused.

**Why noloop passes:** no in-place ASSIGN, so forward leaves
never mutate.

### The real fix direction

The bug is in the shared scheduler/fuser, NOT in DUP/GRAD/KEEP
interaction rules. Ruled-out hypotheses from rounds 1-7
(term_clone, fresh labels, port_slot teardowns, FORCE_V0,
DUP⊳ERA, heap mutation between hits) were all at the wrong
layer.

Fix options:

1. **Snapshot forward activations before ASSIGN can mutate.**
   In PyTorch terms: "saved tensors" in autograd. The backward
   graph should reference materialised TEN intermediates, not
   live input tensors. Would require changes to how grad
   construction interacts with the scheduler.

2. **Order ALL backward dispatches before ANY ASSIGN fires.**
   Currently SEQ orders the lazy `assign_w` and `assign_b`
   expressions, but evaluating `assign_w` triggers its own
   backward dispatch first, then blits. If we could force "all
   backwards, then all blits" ordering, the stale-read
   disappears.

3. **Copy-on-write ASSIGN buffers.** Expensive; defeats the
   in-place semantics of ASSIGN.

4. **Test-level workaround**: insert a DETACH or explicit
   materialisation on each `g_var` so grads resolve to fresh
   TENs before any ASSIGN executes.

Option 2 is the most IC-aligned. Needs scheduler-level work.

### Credit

User's one-line hint ("check fused kernels and codegen, cpu vs
metal?") unlocked 7 rounds of wrong-layer debugging. Moral:
when stuck at one layer, verify the layer before climbing deeper
into it.

## ROUND 9 (2026-04-18): diagnosis CONFIRMED, fix attempts failed

**Confirmation experiment**: flip `SEQ(assign_w, ...) → SEQ(assign_b, ...)` —
fire b's ASSIGN first. Result: **w becomes 0.8×, b becomes exact**.
Exactly the mirror predicted by the stale-read hypothesis.
With b_post=0.38, w=0.5, target=2: diff=-1.12, g=-2.24,
w_new=0.5+0.224=0.724 ✓ matches observed w. Diagnosis locked in.

### Failed fix attempts this round

1. **SEQ update_w and update_b before the assigns**:
   ```c
   seq_body = SEQ(update_w, SEQ(update_b, SEQ(assign_w, SEQ(assign_b, rec))))
   ```
   No effect — b still 0.8×. Root cause: IC reducer reduces
   the Term once (to TEN) but doesn't cache the TEN back at
   the original Term's heap location. The subsequent
   `assign_w` reading `update_w` re-traverses the SUB chain
   and re-reads the live forward tensors.

2. **DETACH(update_w) and DETACH(update_b)**:
   ```c
   update_w = DETACH(SUB(w_sgd, MUL(lr_w, gw_var)));
   update_b = DETACH(SUB(b_sgd, MUL(lr_b, gb_var)));
   ```
   w's ASSIGN fires correctly with the DETACHed TEN. But
   **b's ASSIGN doesn't fire at all** — final_b stays at its
   initial value. DETACH on the src term appears to break the
   ASSIGN chain when combined with the MAT-bound gb_var. The
   ASSIGN's dst (b_assign, a DP projection) doesn't resolve in
   time.

### Real mechanism

IC semantics: a Term is a REFERENCE (tag+loc). Reducing a Term
walks its heap content and computes a value. The VALUE is
returned to the caller, but the original heap content at the
Term's location is typically NOT overwritten with the reduced
value. So repeated reductions of the same Term re-compute.

For ASSIGN with in-place mutation of forward inputs:
- 1st ASSIGN reduces its src → walks forward chain → reads t1, t2 → computes → blits new w into t1.
- 2nd ASSIGN reduces its src → walks forward chain → re-reads t1 (now mutated) → computes with stale t1.

The fix must either:
(A) Cache the src TEN at the Term's heap location so second-read gets TEN directly (no re-traversal).
(B) Dispatch both srcs in one atomic pass, then both blits.
(C) Block ASSIGN's blit until all live backward dependencies on its dst have completed.

### Next round direction

(C) is the IC-natural fix — an ASSIGN dependency tracker. An
ASSIGN's dst blit should be deferred until no other live
computation references that dst's tid as a forward input. This
is analogous to a write-fence in concurrency terms.

Concretely: the scheduler maintains a refcount of "backward
readers" per tensor id. thvm_grad_multi_keep increments
the refcount on each param for the duration of the backward
dispatch window. ASSIGN fires its blit only when the refcount
drops to zero. With both w and b's backward sharing the
window, both backwards complete before either blit fires.

Next round: look at where the scheduler assigns kernel leaves
(fuse_build_kernel) and identify the right hook point for a
write-fence.

## ROUND 10 (2026-04-18): more fix attempts fail, cross-cutting change needed

### Failed fix attempt 3

**SEQ(gw_var, SEQ(gb_var, ...))** before the assigns — force each
lambda-bound gradient var to materialise to a TEN in its
binder slot before either ASSIGN fires.

Result: w exact, **b worse than ever** (`[-0.0060, 0.1080,
-0.2070]` vs expected `[0.38, 1.16, -0.89]`). Some side effect
between gw_var's reduction and gb_var's chain — possibly the
reduction chain for gw_var consumes shared DUP cells that
gb_var's chain relies on. The DUPs at cell 209 / 272 (the
ADD-backward gy-dups) have both DP0 (w side) and DP1 (b side)
consumers; when gw_var reduces, it fires these DUPs via its
DP0 side, which writes to both port_slots. But b's bundle slot
stores a CLONED DP1, not the original — so the port_slot write
to the original DP1 (in the backward chain) doesn't affect b's
cloned chain. Net effect: gw_var's reduction drains *some*
shared state without updating b's cloned path, leaving b's
reduction to hit dead cells.

### Why the scheduler/kernel layer is the right place to fix

Fundamentally, the backward kernels built by the scheduler
reference **forward input tensors** (t1, t2, t3) as leaves —
NOT materialised forward intermediate TENs (like diff's TEN).

In PyTorch autograd, forward ops save their outputs (activations)
and the backward references those saves. In TinyHVM IC, the
forward is lazy, so each grad target's backward chain
independently walks back to forward tensor reads.

When ASSIGN mutates a forward tensor between backward dispatches,
any subsequent backward reads get the stale value.

### The architectural fix options, ranked

1. **Forward-activation save** (PyTorch-style).
   At GRAD time, identify forward op outputs that the backward
   references and materialise them to fresh TENs. Store those
   TENs in the GRAD node's metadata. Backward references these
   saved TENs (stable tensor ids) instead of re-traversing
   forward chains.

   Pros: canonical autograd pattern, robust to any number of
   ASSIGNs.

   Cons: touches grad construction (src/grad/_.c) and possibly
   BG/BG_GY rules (src/interact/grad.c). Structural change.

2. **ASSIGN write-fence** (scheduler-level).
   ASSIGN's dst blit defers until no live backward kernel
   references the dst tid as a leaf. Refcount per tid of
   backward readers; multi_keep increments, bundle_accum
   decrements.

   Pros: isolates the fix to scheduler, doesn't change grad rules.

   Cons: need a mechanism to detect "live backward kernel
   references". fuse_build_kernel doesn't currently expose this.

3. **Two-phase ASSIGN** (split compute and blit).
   Add an op that's "compute src to fresh TEN, but don't blit
   yet". Pair with a deferred blit op. Then SEQ can sequence
   all computes before all blits.

   Pros: conceptually simple.

   Cons: introduces a new op pair, changes IC semantics, may
   conflict with existing fusion decisions.

### Still stuck on direct IC fix; architectural work needed

Rounds 1–7 debugged the wrong layer (DUP/GRAD/bundle mechanics).
Rounds 8–10 located the bug (stale forward read in backward
kernel leaf) and confirmed it via SEQ flip + numeric match.
Three fix attempts at IC term level (SEQ updates, SEQ gw_var,
DETACH updates) all fail due to IC laziness not caching
reductions back at the original Term's heap location.

The real fix requires either (1) or (2) above. Both are
non-trivial and need user guidance on which direction to
pursue.

## ROUND 11 (2026-04-18): WORKING FIX (env-gated) — force MAT-CTR children to WNF

Added `THVM_MAT_FORCE_WNF=1` env knob in APP-MAT-CTR
([src/interact/combinators.c](/Users/swish/src/TinyHVM/src/interact/combinators.c)
line ~192): when set, each CTR child of TAG_TOP kind is passed
through `thvm_eval` to materialise as a TEN BEFORE binding into
the lambda variables.

**Result: twoparam_grad passes at n=1, 2, 3 with EXACT SGD
trajectory match:**

| n | final_w                              | final_b                              | Status |
|---|--------------------------------------|--------------------------------------|--------|
| 1 | `[0.7800, -0.0400, -0.3400]`         | `[0.3800, 1.1600, -0.8900]`          | ✓ PASS |
| 2 | `[0.9480, 0.5360, -0.6940]`          | `[0.5480, 1.7360, -1.2440]`          | ✓ PASS |
| 3 | `[1.0488, 0.8816, -0.9064]`          | `[0.6488, 2.0816, -1.4564]`          | ✓ PASS |

Why it works: when the grad bundle's ADD-accumulator children are
materialised to fresh TENs before the lambda body runs, the
gradient TENs are cached (stable tensor ids). ASSIGN reads
`gw_var` / `gb_var` which now point to cached TENs, not to live
forward chains. No more stale-read when the other ASSIGN mutates
a forward input buffer.

### Known limitations

1. **Violates "no eager eval in interaction rules" policy.**
   `thvm_eval` called from APP-MAT-CTR is eager evaluation from
   an interaction rule. Gated to `THVM_MAT_FORCE_WNF=1` so it
   doesn't run by default. User's call on whether to relax the
   policy or find a different materialisation point.
2. **Aborts on twoparam_sum** (`SIGTRAP` exit 133). That test's
   bundle children are `TOP/UOP_EXPAND` (lazy view ops).
   `thvm_eval` on an EXPAND term hits an assertion somewhere in
   the scheduler. Would need to either (a) restrict force-WNF to
   non-view UOPs or (b) fix whatever scheduler assertion fires.

### Behaviour summary

- Default (no env flag): all tests behave exactly as before
  (scalar PASS, twoparam_sum PASS, twoparam_grad XFAIL,
  grad_fuse_minimal PASS). No regression.
- With flag: twoparam_grad PASSES, scalar still PASSES,
  grad_fuse_minimal still PASSES, twoparam_sum ABORTS.

### Next steps (require user decision)

The fix mechanism is proven. To land it cleanly:

1. **Decision needed**: is calling `thvm_eval` from an
   interaction rule acceptable for MAT-CTR destructuring? The
   rule's semantics arguably SHOULD force WNF children since it
   binds them as lambda arguments. Or prefer option 2/3 from
   round 10.

2. **Fix the UOP_EXPAND abort.** Restrict the force path to ops
   that reliably materialise, or fix the scheduler assertion
   that fires on EXPAND evaluation.

3. **Consider replacing `thvm_eval` with a less-eager
   alternative** — e.g., emit a FUSE wrapper and let the
   normal reducer handle materialisation through its fixed
   point.

## ROUND 12 (2026-04-18): FUSE-wrapping approaches FAIL

Tried two lazy alternatives to the `thvm_eval` eager path:

### Attempt 1 — `THVM_MAT_FUSE_CHILDREN=1`

Wrap each CTR child in a `FUSE(child)` term, then let the
lambda body consume the FUSE'd version. Result: all tests fail
— `final_b` stays at initial value (no ASSIGN fires). The lazy
FUSE doesn't materialise before the body runs; the ordering
guarantee needed for correctness isn't enforced.

### Attempt 2 — `THVM_MAT_SEQ_FUSE=1`

Same FUSE wrapping, PLUS prepend a SEQ chain `SEQ(FUSE(c0),
SEQ(FUSE(c1), original_APP_chain))` so each FUSE materialises
in order before the body runs. Also fails — same result as
above (b=initial, no progress). The SEQ-chain reduces each
FUSE and discards, but the APP chain still reads ctr_loc[i]
which holds FUSE — and the lambda binding receives FUSE, not
the materialised TEN. The FUSE-materialisation side effect
isn't propagating to where the body reads.

### Why lazy approaches fail, eager works

- `thvm_eval(child)` — runs the entire reducer fixed-point in
  the child's chain. Produces a TEN (or ERA/NUM). We then
  `heap_set(ctr_loc + i, reduced)` — the CTR slot now holds
  the cached TEN. APP chain reads the TEN. Body gets TEN.
  Key insight: `thvm_eval` COMPLETES before MAT-CTR returns,
  so the heap_set happens synchronously.

- `FUSE(child)` — just marks the child for materialisation.
  The SEQ chain reduces the FUSE — but the reduced value goes
  into the FUSE wrapper's inner cell, not back to the CTR slot.
  When APP chain reads ctr_loc[i], it gets the FUSE term, not
  a TEN. When the body reduces that FUSE later, it's TOO LATE
  — by then interleaving with ASSIGN has already happened.

### Fundamentally, the fix needs eager materialisation

The stale-read bug requires SYNCHRONOUS materialisation at MAT
time. Lazy IC primitives (FUSE, SEQ) can order work but can't
force synchronous completion across the boundary between
interaction rule firing and subsequent term consumption.

`thvm_eval` (round 11) is the only mechanism that achieves
this, and it remains the sole working fix — with the
documented policy-violation and UOP_EXPAND-abort caveats.

### Recommendation

Unless user has a less-intrusive alternative, the pragmatic
path is:
1. Accept `thvm_eval`-in-MAT-CTR for grad bundle destructuring.
2. Fix the UOP_EXPAND abort (likely an assertion the scheduler
   throws on certain view-op shapes mid-reduction).
3. Make `THVM_MAT_FORCE_WNF=1` the default once EXPAND is
   fixed.

Alternatively the scheduler-side write-fence (option B) is
the canonical IC-aligned solution but requires non-trivial
scheduler work.

## ROUND 13 (2026-04-18): third lazy attempt fails too

User feedback: "why you MAT to work on phase1 anyway? let it
remain WNF, fuse everything below, final reduce should have
materialized children"

This is the right IC intuition. Investigation found the
FUSE ⊳ CTR rule (tensor_ops.c ~line 411) already distributes
FUSE to each CTR child when fire on `FUSE(CTR(...))`. So the
IC primitive exists. Tried:

### Attempt 3 — `THVM_MAT_FUSE_CTR_ARG=1`

When MAT-CTR sees any unreduced TAG_TOP child, wrap the CTR
itself in FUSE and rewrite the APP's second arg to FUSE(CTR),
then return `t` to defer. The reducer's next step fires
FUSE ⊳ CTR which distributes FUSE to each child. Then the
children materialise via normal FUSE propagation. Eventually
MAT-CTR fires again with TEN children.

**Result**: fails like the other lazy attempts — `final_b`
stays at initial, ASSIGNs never fire. The FUSE-wrapped children
don't resolve through the scheduler in a way that reaches
MAT-CTR again with TEN values.

Same pattern as rounds 11-12 lazy attempts: IC-native FUSE
propagation CAN materialise things, but the trigger point
from APP-MAT-CTR doesn't reach completion before MAT's
continuation runs.

### All three lazy approaches now ruled out

| Attempt | Mechanism | Result |
|---------|-----------|--------|
| FUSE_CHILDREN | Wrap each child in FUSE | Fails: ASSIGNs never fire |
| SEQ_FUSE | FUSE + SEQ chain before APP | Fails: same |
| FUSE_CTR_ARG | FUSE the CTR itself, retry | Fails: same |

### What works: `THVM_MAT_FORCE_WNF=1` (eager `thvm_eval`)

The ONLY mechanism that fixes the bug remains eagerly calling
`thvm_eval` on each TAG_TOP CTR child from inside APP-MAT-CTR.
Synchronous materialisation is essential; lazy IC primitives
can't achieve it from within an interaction rule.

### Code state

Cleaned up in this round: removed the three failing env-gated
lazy experiments (THVM_MAT_FUSE_CHILDREN, THVM_MAT_SEQ_FUSE,
THVM_MAT_FUSE_CTR_ARG) from src/interact/combinators.c.
`THVM_MAT_FORCE_WNF=1` remains the sole env-gated fix. Default
behaviour unchanged (all regressions pass with flag off).

### Remaining path forward

User's "MAT should remain WNF, fuse below" is architecturally
correct but requires either:
(a) making the reducer's fixed point retry APP-MAT-CTR after
    CTR children materialise — requires the outer scheduler to
    know about the pending APP-MAT and re-drive it.
(b) accepting synchronous eager materialisation at MAT-CTR
    (round 11's thvm_eval path).
(c) scheduler-level write-fence on ASSIGN (plan option B):
    ASSIGN defers its blit until no backward kernel has the
    dst tid as a live leaf. Most IC-aligned but largest
    surgery.

Given 13 rounds of investigation, recommend user pick a
direction before further attempts — continued iteration on
APP-MAT-CTR alone isn't productive.

## ROUND 14 (2026-04-18): FORCE_WNF flakiness measured — NOT safe as default

Tested `THVM_MAT_FORCE_WNF=1` determinism with 20 runs each:

| Test           | Default-off (20 runs)   | With flag (20 runs)        |
|----------------|-------------------------|----------------------------|
| scalar n=1/2/3 | 20/20 PASS              | 20/20 PASS                 |
| twoparam_sum n=1 | 20/20 PASS (exit 0)  | 17/20 FAIL (exit 133, no output) ⚠️ |
| twoparam_grad n=1/2/3 | XFAIL (the bug) | 20/20 PASS with exact values |
| grad_fuse      | 20/20 PASS              | 20/20 PASS                 |

**The FORCE_WNF fix is 85% flaky-abort on twoparam_sum.** Likely
an assertion or heap-corruption race in the scheduler when
`thvm_eval` is called recursively on a UOP_EXPAND child from
within an interaction rule. The 3/20 runs that pass suggest
the race depends on heap allocation order.

This is NOT safe to land as default. The working fix delivers
correct numerics for the target bug but breaks a passing
regression non-deterministically.

### Remaining direction for a real fix

The bug is now understood, the symptomatic fix identified, and
the cost of that fix (flakiness, eager-eval policy violation)
measured. To land a stable fix, one of:

1. **Fix the EXPAND race.** Investigate what `thvm_eval` on a
   UOP_EXPAND child touches during a recursive re-entry that
   fails non-deterministically. Likely a scheduler assertion
   around tensor-id state or view-tracker state.

2. **Switch to option B (scheduler write-fence).** ASSIGN's blit
   defers until no live backward kernel references the dst tid.
   Bigger surgery; the most IC-aligned approach. Deterministic
   by design (no recursive eval from rules).

3. **Option A (PyTorch-style saved tensors).** At GRAD construction
   time, materialise forward activations that the backward
   references and use those saved TENs as leaves in backward
   kernels.

Options 2 and 3 are architecturally cleaner than option 1.
Option 1 is the minimum surgical fix if we accept the policy
violation. Option 2 is my strongest recommendation.

All 14 rounds of investigation are now documented. The cron
should stop; next step requires user direction.

For matmul (`test_tiny_linear_sgd_loop`) W[0,0] is ~4x expected at
step 1 — separate deeper issue involving MM backward in the loop.
Investigate only after twoparam is closed.
