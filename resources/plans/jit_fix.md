# Plan: Fix JIT Replay & Close the Gap

## Global Picture

| Metric | TinyHVM now | Tinygrad target | Gap |
|--------|-------------|-----------------|-----|
| Accuracy | 94.3% (BS=128) | 98.3% (BS=512) | BS + planner |
| Speed | 492ms/step | 79ms/step | 6.2x |
| Dispatches | 374 | ~20 | 18.7x |
| Memory | 2.1GB (BS=128) | 0.82GB (BS=512) | Need planner |

Critical path: **Fix JIT replay → planner works → BS=512 → 98%+. Speed from replay.**

---

## How Tinygrad Does It (reference design)

Tinygrad's JIT (`engine/jit.py`) uses a 3-phase model:
1. **Capture** (`cnt==1`): Collect all `ExecItem` dispatches into `_jit_cache` list
2. **Replay** (`cnt>=2`): Re-execute cached items with new inputs via `input_replace` dict
3. **Graph** (`runtime/graph/metal.py`): Batch entire replay into one Metal ICB (indirect command buffer)

Key design patterns:
- **`input_replace` dict**: Maps `(jit_item_idx, buf_idx) → input_buf_idx`. On replay, only input buffers are swapped — intermediates persist and get overwritten by the same kernels.
- **Intermediates persist**: Allocated once at first replay (`ensure_allocated()`), reused forever. NOT freed between steps. Each kernel writes before reads — no zeroing needed.
- **Variables via shared buffer**: Per-step varying scalars (like Adam bc1/bc2) stored in a shared int buffer. Metal graph updates `int_buf_view[idx] = var_vals[var]` before dispatch.
- **Memory planner**: TLSF allocator runs on the captured schedule. Buffers with non-overlapping lifetimes share memory. Only 71 lines (`engine/memory.py`).

**Key difference from TinyHVM**: Tinygrad doesn't restore consts or zero buffers between replays. Every intermediate buffer is simply overwritten by the kernel that produces it. The only per-step work is swapping input buffer pointers and updating variable values.

---

## Current JIT Bug

JIT replay produces 12.4% accuracy (random chance). Capture step 0 is correct.

### Confirmed facts
- All GPU dispatch paths record to JIT (comprehensive audit)
- JIT tiny test works perfectly (4 cmds, weights update correctly)
- Same failure with NO_PLAN=1 (independent buffers) — NOT planner bug
- Replay |grad| = 3022 vs capture |grad| = 359 (8.4x, wrong intermediates)

### Investigation results (2026-04-03)

**Confirmed working:**
- JIT tiny test (SGD): weights update correctly across replay steps
- All GPU dispatch paths record to JIT (comprehensive audit)
- JIT capture == non-JIT (cos_sim=1.0 for all non-zero params)
- Forward pass cmd 7 output matches capture EXACTLY for same-data replay
- Overfit test (same batch every step): gradients decrease (2323→81)
- Fixed: manual CE → cross_entropy_loss, thvm_tensor → thvm_shrink for input

**Still broken:**
- JIT BM replay: 9.5-9.7% accuracy at 70-500 steps despite non-zero grads
- |grad| decreases (2323→390 over 500 steps) but weights barely change (0.004)
- Grads cancel in direction: large |g| but near-zero momentum m
- NOT planner: same result with NO_PLAN=1 (16GB budget)
- NOT const aliasing, NOT GPU sync, NOT forward computation
- NOT one-hot labels: oh_bid is persistent, not in consts, buf_write works

**BREAKTHROUGH: CNN + SGD + sum loss — JIT replay WORKS!**
Weights explode (lr too high) but they DO update: lw[0] goes from 0.007 to -22M in 5 steps.
This PROVES the JIT replay infrastructure is correct — commands fire, buffers resolve,
weights update through the full CNN backward + SGD assign chain.

The issue is SPECIFICALLY Adam + CE during replay. Adam normalizes gradient direction.
With JIT replay, each step's gradient points in a slightly different direction (different
batch data + evolved weights). Adam momentum averages these directions. In non-JIT, the
same thing happens but the model converges. The JIT replay's numerical behavior differs
subtly enough that Adam momentum doesn't converge.

**Root cause hypothesis:** The fused backward kernels during replay compute gradients
that are numerically close to (but not identical with) what a fresh IC reduction would
produce. The differences are small per-step but accumulate in Adam's momentum, causing
the gradient direction to be biased. This bias prevents convergence.

**DEFINITIVE FINDING: beautiful_mnist ALSO has zero conv grads at step 0.**
Only lw=356.9, lb=0.6, bb2=1.5 are non-zero — SAME as JIT capture. This is
normal vanishing gradient behavior at initialization. The model trains purely
through the dense layer initially.

Non-JIT beautiful_mnist converges because it rebuilds the graph each step.
Over 70 steps, the dense layer trains → forward intermediates change → conv
grads eventually become non-zero → full model converges.

JIT replay fires the SAME backward commands. These commands SHOULD produce
non-zero conv grads when weights evolve. But 500 replay steps still shows
only dense-layer grads. The commands are generic (MUL/ADD/SUM) so they
should work on any input values. The question is whether the fused backward
kernels correctly propagate non-zero values through the BN/pool chain when
the forward intermediates change.

**Path forward:** The JIT replay infrastructure is mechanically correct (verified
with CNN+SGD). The convergence issue is that the captured backward dispatch
sequence doesn't produce non-zero conv grads even after dense training. Options:
(a) Capture at a later step (after warm-up) — requires multi-step non-JIT then JIT
(b) Disable backward fusion during capture so backward commands are unfused
(c) Adopt tinygrad approach: schedule → replay (no IC at all)

**WARM-UP TEST RESULT (2026-04-04):**
10 non-JIT warm-up steps → 68% accuracy. Then 60 JIT replay steps → **12.5%**.
JIT replay ACTIVELY DESTROYS training. The backward produces wrong gradient
directions that push weights away from the warm-up solution.

Forward pass is verified correct (cmd 7 output matches capture exactly).
ViewParams are baked but should be correct (same layout). The bug must be
in how the backward dispatch sequence reads from intermediate buffers.

**NARROWING (2026-04-04):**
- CE + Adam + LINEAR model: JIT replay WORKS (w changes correctly)
- CE + Adam + CNN model: JIT replay FAILS (destroys warm-up training)
- Sum loss + SGD + ANY model: JIT replay WORKS
- Issue is specifically conv backward during JIT replay
- Conv backward involves im2col-reversed view ops (EXPAND/SHRINK/RESHAPE)
  with ViewParams baked from capture time
- Next: test conv+dense (no BN, no pool) to isolate if it's conv backward
  specifically or the pool/reshape chain
- Consts ARE needed: NaN without restoration (86 consts = backward scalars + shapes)
- Loss readback broken: loss slot not found in JIT (buf_id 460 not in any slot)

**Root cause hypothesis:**
The JIT backward commands compute gradients of correct MAGNITUDE but wrong DIRECTION
for the current weights. Each step's gradient pushes in a slightly different random
direction. Over steps, the momentum averages to near-zero despite large |g| values.
This manifests as: m[0]=-0.000634 at step 69 (near-zero) despite |g|=686 (large).

This would happen if the backward commands read intermediate forward buffers that have
been overwritten by the memory planner (overlapping lifetime assignment). The intermediate
values would be numerically wrong, producing gradients that are large but incoherent.
However NO_PLAN also fails, ruling this out.

**Alternative hypothesis:** The captured backward dispatch sequence encodes a specific
gradient computation path from the GRAD handler's provenance walk. This path reads
from specific intermediate buffers in a specific order. When weights change, the forward
pass produces different intermediates. The backward commands read these new intermediates
and compute correct gradients FOR THE NEW DATA. But some buffer reuse during replay
(even without planner) corrupts an intermediate that the backward needs.

**Next step:** Build a minimal test that does non-JIT step 1 (with proper thvm_reset +
fresh graph) and JIT replay step 1 on the SAME data, comparing per-param gradient
vectors element-by-element. This will show if the gradients are merely scaled differently
or completely wrong in direction (cosine similarity test).

---

## Fix Plan

### Phase 1: Checksum Verification (find exact divergence point)

Add per-command output checksums during capture. During replay, verify after each command.

```c
// Capture: after each dispatch, flush and checksum output
static u32 jit_checksums[JIT_MAX_CMDS];  // capture-time checksums

// In jit_replay_commands with VERIFY=1:
metal_flush();
u32 out_bid = jit.slots[cmd->buf_slots[0]].buf_id;
u32 sum = fnv1a(BUF_CONTENTS(out_bid), jit.slots[cmd->buf_slots[0]].alloc_size);
if (sum != jit_checksums[ci])
    fprintf(stderr, "MISMATCH cmd %u: capture=%08x replay=%08x\n", ...);
```

**Deliverable**: Know the FIRST command that diverges and which input buffer is wrong.

### Phase 2: Eliminate const restoration (match tinygrad design)

Tinygrad's approach: intermediates are never restored. Each kernel overwrites its output buffer. Only input buffer pointers and variable values change per step.

**Fix**: Remove `jit_restore_consts()` entirely. Instead:
1. CPU-written scalars (shapes, axes, eps) → keep as **persistent** tensors created before capture. They never change, so they don't need restoration.
2. Per-step varying values (batch data, labels, Adam bc1/bc2) → write directly to persistent buffers before replay, like tinygrad's `input_replace`.
3. Grad accumulators → zero via a GPU kernel (already part of the capture), not CPU memset.

This requires restructuring the test to:
```c
// BEFORE capture: create ALL scalars as persistent
f32 eps = 1e-7f;
Term eps_t = thvm_tensor(ctx, &eps, SHAPE(1,1));  // persistent!
// ... build graph using persistent scalars ...

jit_begin_capture(ctx->tensor_count);  // ALL tensors are persistent
thvm_reduce(ctx, train_step);
adam_step_direct(ctx, &opt, gids);
jit_end_capture();  // no consts to snapshot!

// Replay: only overwrite input data + patch Adam params
for (step = 1; step < n_steps; step++) {
    buf_write(x_buf, new_batch_data);
    buf_write(oh_buf, new_labels);
    patch_adam_bc(jit, opt.t);
    zero_grad_bufs(gids);  // GPU zero_fill, not CPU memset
    jit_replay_commands();  // no const restoration
}
```

### Phase 3: GPU-written detection fix

Mark ALL buffer slots as GPU-written (not just slot 0):
```c
// In jit_end_capture:
for (u32 ci = 0; ci < jit.n_cmds; ci++) {
    JITCmd *cmd = &jit.cmds[ci];
    if (cmd->is_blit) {
        gpu_written[cmd->blit_dst_slot] = 1;
    } else if (cmd->is_mps) {
        gpu_written[cmd->mps_dst_slot] = 1;
    } else {
        // Adam writes to param[0], m[2], v[3]
        // Mark ALL slots as potentially GPU-written
        for (u32 j = 0; j < cmd->n_bufs; j++)
            gpu_written[cmd->buf_slots[j]] = 1;
    }
```

This is conservative but safe. With Phase 2's elimination of const restoration, this becomes moot — but fixes any remaining edge cases.

### Phase 4: Grad zeroing via GPU (not CPU memset)

CPU memset on GPU buffers requires synchronization. Replace with a GPU zero_fill dispatch that's part of the JIT capture:

```c
// During graph construction (before capture):
for (int i = 0; i < NP; i++) {
    // Zero grad is a GPU op, captured by JIT
    Term zero = thvm_assign(ctx, gs[i], thvm_fill(ctx, 0.0f, grad_shape));
    train_step = thvm_app(ctx, zero, train_step);
}
```

Or simpler: add explicit zero_fill dispatches at the start of capture:
```c
jit_begin_capture(n_persistent);
for (int i = 0; i < NP; i++)
    metal_op_zero_fill(ctx->tensors[gids[i]].buf_id, psz[i]);
thvm_reduce(ctx, train_step);
// ...
```

### Phase 5: Memory planner on verified JIT

With JIT replay verified correct, the existing planner code should work. The planner runs in `jit_end_capture()` and assigns non-overlapping lifetimes to shared buffers.

Verify:
- 2.1GB → ~0.8GB at BS=128
- BS=512 fits in GPU memory
- Accuracy unchanged

### Phase 6: Speed optimization

With correct replay, measure raw replay time (no IC reduction overhead):
- Target: <50ms/step at BS=128
- Bottleneck will be GPU kernel execution, not CPU
- Consider Metal ICB (indirect command buffer) for single-submission replay, like tinygrad's MetalGraph

---

## Implementation Order

```
Phase 1: Checksum verify     → find exact divergence point
Phase 2: Remove const restore → match tinygrad's "overwrite" model
Phase 3: Fix GPU-written      → conservative, prevents future bugs
Phase 4: GPU grad zeroing     → remove CPU/GPU sync in replay loop
Phase 5: Memory planner       → 2.1GB → 0.8GB, unlock BS=512
Phase 6: Speed                → <50ms/step via ICB batching
Phase 7: Range-based fusion   → principled kernel boundaries, extensible
```

---

## Phase 7: Range-Based Fusion Refactor

### Why

The current fusion system works (157 reduces, 200 ew fused) but uses ad-hoc
pattern matching: `rule_sum_fuse` checks if a SUM's child is ew/view,
`rule_elementwise_fuse` checks if an ew op has ew/view children,
`fuse_walk_inner` manually handles RESHAPE/PERMUTE/EXPAND composition.
Adding new patterns (SHRINK/PAD fusion, multi-reduce, grouped reduces)
requires modifying multiple switch/if chains.

Tinygrad's grouper uses a principled approach: each op has an iteration
domain (the loop bounds it executes over). Ops with compatible domains
fuse into one kernel. One reduce per kernel is the only structural rule.

### Design: RangeKey

```c
typedef struct {
    u32 dims[MAX_DIM];  // iteration dimensions
    u32 ndim;
} RangeKey;
```

- **Elementwise ops**: range = output shape
- **Reduce ops**: range = INPUT shape (iterate input, accumulate to output)
- **View ops** (reshape/permute/expand): range = output shape, but the VIEW
  composes onto leaf index expressions (same as current fuse_walk_inner)
- **Matmul**: range = (M, N, K) — not fusable with plain ew

Two ops fuse when their RangeKeys match, OR one is a reduce whose input
range matches the other's range (the ew chain feeds the reduce).

### Refactor Steps

1. **Add `range_key_for_op()`**: Compute RangeKey from tensor metadata.
   Replace the `is_elementwise(child_uop) || is_view_op(child_uop)` checks
   in rewrite rules with range compatibility checks.

2. **Unify walk eligibility**: `fuse_walk_inner` currently has separate
   branches for TAG_TEN (deferred), TAG_TOP (ew), TAG_TOP (view). Replace
   with: "walk if range-compatible with root, stop at range boundary".
   Range boundaries become leaves.

3. **Implement SHRINK/PAD composition**: Currently `fuse_walk_inner` line 106
   says "SHRINK, PAD: not implemented yet". With range keys, these are just
   view ops that compose onto leaf index expressions (offset + mask). The
   range key handles eligibility; the codegen handles index math.

4. **Multi-reduce fusion**: When `grad_refs > 1`, the GRAD handler creates
   `ADD(GRAD_a, GRAD_b)` which produces `SUM(da) + SUM(db)`. With range
   keys, detect that `SUM(ADD(da, db))` shares the same input range and
   fuse into a single reduce kernel. This is the 374→~50 dispatch win.

5. **Declarative rule table**: The current `rewrite_rules[]` table in
   `rewrite/_.c` already has the right structure. Extend it with range-aware
   eligibility: `{ match_uop, range_compat_fn, rewrite_fn }`.

### Files to Modify

| File | Change |
|------|--------|
| `src/fuse/_.c` | Add RangeKey, range_key_for_op, range-based walk eligibility |
| `src/fuse/materialize.c` | SHRINK/PAD view composition in materialize_walk |
| `src/rewrite/_.c` | Range-compatible eligibility in rule functions |
| `src/backend/metal/codegen.m` | SHRINK/PAD index math in codegen |
| `src/interact/grad.c` | Multi-reduce fusion for grad_refs > 1 |

### Expected Impact

| Change | Dispatches |
|--------|-----------|
| Current | 374 |
| SHRINK/PAD fusion (forward) | ~350 |
| Lazy ENSURE backward | ~170 |
| Multi-reduce fusion | ~50 |
| Target (tinygrad-like) | ~20 |

---

## Files to Modify (all phases)

| File | Phase | Change |
|------|-------|--------|
| `src/backend/metal/jit.m` | 1-4 | Checksums, remove const restore, fix GPU-written |
| `src/backend/metal/init.m` | 1 | Checksum storage in JIT struct |
| `test/test_jit_bm.m` | 2,4 | Restructure: persistent scalars, GPU grad zeroing |
| `src/backend/metal/pool.m` | 5 | Planner integration verified |
| `src/fuse/_.c` | 7 | RangeKey, range-based fusion eligibility |
| `src/fuse/materialize.c` | 7 | SHRINK/PAD view composition |
| `src/rewrite/_.c` | 7 | Range-compatible rewrite rules |
| `src/backend/metal/codegen.m` | 7 | SHRINK/PAD index codegen |
| `src/interact/grad.c` | 7 | Multi-reduce fusion |

## Expected Progression

| After | Accuracy | Speed | Memory | Dispatches |
|-------|----------|-------|--------|------------|
| Phase 1 (checksums) | debug only | — | — | 374 |
| Phase 2 (no const restore) | **94%+** | ~400ms | 2.1GB | 374 |
| Phase 4 (GPU grad zero) | 94%+ | ~100ms | 2.1GB | 374 |
| Phase 5 (planner) | **98%+ (BS=512)** | ~100ms | **~0.8GB** | 374 |
| Phase 6 (ICB) | 98%+ | **<50ms** | ~0.8GB | 374 |
| Phase 7 (ranges) | 98%+ | **<30ms** | ~0.8GB | **~50** |
