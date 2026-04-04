# Plan: Close the TinyHVM/Tinygrad Gap

## Current State (2026-04-04)

| Metric | TinyHVM (non-JIT) | TinyHVM (JIT) | Tinygrad | Gap |
|--------|-------------------|---------------|----------|-----|
| Accuracy | 94.3% (BS=128) | **96.6%** (BS=128) | 98.3% (BS=512) | BS=512 |
| Speed | 500ms/step | ~410ms/step | 79ms/step | 5x |
| Dispatches | 374 | 374 | ~20 | 18x |
| Memory | 2.1GB | 1.7GB (planner) | 0.82GB | 2x |

### What's Done

- [x] JIT capture/replay — 96.6% accuracy with 1 warm-up step
- [x] Memory planner — 12.5GB→1.7GB compression in jit_end_capture
- [x] ASSIGN fully lazy (0 thvm_reduce calls)
- [x] BN assigns wrapped in APP(ASSIGN,ERA) for inet_step safety
- [x] All gradient bugs fixed (CPU=Metal for all 14 params)
- [x] Deferred ew ops (forward + backward)
- [x] SUM/RMAX fusion with deferred ew chains
- [x] Primitive matmul (EXPAND+MUL+SUM)

### JIT Architecture

1. **Warm-up step** (1 non-JIT IC reduction): stabilizes dispatch count (371→374)
2. **Capture step** (non-JIT IC reduction + JIT recording): 374 dispatches captured
3. **Replay steps** (JIT replay): same 374 dispatches, new input/label data per step
4. **thvm_reset(nw) before eval**: cleans up JIT pool state for correct eval

Key: `jit_restore_consts` restores 87 CPU-written backward scalars (shapes, axes,
epsilon). `jit_alloc_ephemeral` allocates plan buffers. Adam bc1/bc2 patched per step.

### Open Issues

- **3-dispatch non-determinism** (371 at step 0, 374 at step 1+): cause unknown despite
  clearing defer_consumers, buf_cpu_only, grad_refs. Persistent tensor metadata identical
  between steps. Workaround: 1 warm-up step before capture.
- **JIT eval cleanup**: must call thvm_reset(nw) before eval or ephemeral pool entries
  corrupt eval's buffer allocation.

---

## Phase 1: Memory Planner → BS=512 (NEXT)

The planner already runs in `jit_end_capture` and compresses 12.5GB→1.7GB. Need to verify
it works for BS=512 training with the JIT.

### Steps

1. **Verify planner at BS=128**: Run JIT BM test with planner enabled (default).
   Already shown: 96.6% accuracy with planner. ✓
2. **Test BS=512**: Change BS=512 in JIT BM test. Planner should compress to ~1GB.
   Verify accuracy ≥ 98%.
3. **Fix any BS=512 issues**: May need larger HEAP_CAP, more JIT slots, etc.

### Expected Result
- BS=512: 98%+ accuracy, ~1GB memory, ~400ms/step

---

## Phase 2: Speed — JIT Replay Optimization

Current: ~410ms/step at BS=128. Target: <100ms/step.

### Steps

1. **Profile replay**: Where does time go? GPU kernel execution vs CPU encoding overhead.
2. **Metal ICB** (indirect command buffer): Batch ALL replay commands into one ICB.
   Submit once per step instead of encoding each command individually.
   Tinygrad's MetalGraph does this: one `executeCommandsInBuffer` call per step.
3. **Eliminate CPU/GPU sync in replay loop**: Move grad zeroing to GPU (zero_fill dispatch
   captured in JIT). Remove CPU memset + metal_flush between steps.

### Expected Result
- <100ms/step at BS=128, <200ms/step at BS=512

---

## Phase 3: Dispatch Reduction (374 → ~50)

### Eliminate Standalone EW Dispatches (200 fused_ew → ~0)

The 200 fused_ew dispatches are backward ew chains materialized at ENSURE boundaries.
In tinygrad, these are absorbed into reduce kernels.

Fix: Lazy ENSURE during backward — don't materialize deferred ew chains. Let them stay
deferred until a reduce (SUM/RMAX) absorbs them. Expected: 374 → ~171 dispatches.

### Multi-Reduce Fusion (~171 → ~50)

Shared tensors (grad_refs > 1) create ADD + SUM pairs. Fuse into single kernel:
`SUM(ADD(da, db))` instead of `SUM(da) + SUM(db)`.

### SHRINK/PAD in fuse_walk_inner

Currently stops at SHRINK/PAD ("not implemented yet"). Implementing these view
compositions fuses more forward ops.

---

## Phase 4: Range-Based Fusion Refactor (374 → ~20)

Replace ad-hoc pattern matching with principled range keys. Each op has an iteration
domain. Ops with compatible domains fuse. One reduce per kernel.

See detailed design in the earlier plan version (RangeKey struct, 5 refactor steps).

---

## Phase 5: Fix 3-Dispatch Non-Determinism

Eliminate the need for warm-up by making the dispatch sequence deterministic across steps.
The 3 extra fused dispatches at step 1 come from BN-forward related materializations.
Cause unknown — not defer_consumers, not buf_cpu_only, not grad_refs.

Low priority since warm-up is a cheap workaround (~500ms one-time cost).

---

## Expected Progression

| After | Accuracy | Dispatches | Speed | Memory |
|-------|----------|------------|-------|--------|
| Current (JIT BS=128) | 96.6% | 374 | 410ms | 1.7GB |
| Phase 1 (BS=512) | **98%+** | 374 | ~400ms | **~1GB** |
| Phase 2 (ICB speed) | 98%+ | 374 | **<100ms** | ~1GB |
| Phase 3 (dispatch cut) | 98%+ | **~50** | <50ms | ~1GB |
| Phase 4 (ranges) | 98%+ | **~20** | **<30ms** | ~1GB |

## Critical Files

| File | Phase | Change |
|------|-------|--------|
| `test/test_jit_bm.m` | 1 | BS=512 test |
| `src/backend/metal/jit.m` | 2 | ICB replay, eliminate CPU sync |
| `src/interact/tensor_ops.c` | 3 | Lazy ENSURE during backward |
| `src/fuse/_.c` | 3,4 | Multi-reduce fusion, RangeKey |
| `src/fuse/materialize.c` | 3 | SHRINK/PAD view composition |
| `src/backend/metal/codegen.m` | 3,4 | SHRINK/PAD index codegen |
