# Close the Gap: 76s → 20.5s Optimization Plan

## Current State
- **76s total** (tinygrad: 20.5s) = **3.7× gap**
- 251 dispatches/step at ~750ms GPU (tinygrad: 186 at 37ms)
- 96.6% accuracy (matched)

## Root Cause Breakdown

The 750ms GPU time per step decomposes:

| Category | Dispatches | Est. Time | Why Slow |
|----------|-----------|-----------|----------|
| mrs (mul_reduce_sum) | 38 | ~200ms | Serial inner loop, no parallel reduce, integer divisions per reduce iteration |
| contiguify | 46 | ~150ms | Unnecessary GPU copies before reduce/mm/reshape — strided_idx identity kernel |
| mdim binary | 37 | ~100ms | Already optimized; some still scalar (not float4 eligible) |
| fused_v2 | 20 | ~80ms | strided_idx with 4-8 integer divisions per element per leaf |
| bc2d | 24 | ~40ms | No float4, not fused with adjacent ops |
| reduce | 12 | ~40ms | Serial inner loop, no parallel reduce |
| adam | 14 | ~30ms | Scalar (no float4) |
| f4_bin + fast + mm | 38 | ~30ms | Already fast |
| other (conv helpers) | 22 | ~80ms | Inherent to im2col conv approach |

## Priority-Ordered Optimizations

### P0: Parallel reduction in mul_reduce_sum (~35ms saved)
**Files:** `shaders.metal`, `fused.m`
**What:** The `mul_reduce_sum` kernel does `for (r=0; r<reduce_dim; r++) acc += a[idx]*b[idx]` in ONE thread per output element. For reduce_dim=784, that's 784 serial iterations.
**Fix:** Use SIMD-width (32) parallel reduction: each thread in a SIMD group handles reduce_dim/32 elements, then `simd_sum()` across the group. For reduce_dim < 32, use direct serial (current). For reduce_dim >= 32, use `simd_sum`.
**Also:** Specialize for single-axis reduce (n_reduce==1) to skip the multi-axis coordinate decomposition loop.

### P1: Replace strided_idx in fused_v2 with mdim indexing (~10ms saved)
**Files:** `fused.m` lines 72-153
**What:** `fused_v2` kernels use generic `strided_idx` (integer division loop per leaf per element). The mdim codegen already solved this with compile-time coordinate decomposition.
**Fix:** In `codegen_fused_v2`, use 3D dispatch with the same group collapsing as mdim. Decompose output coords once, then compute each leaf's physical index via multiply-only expressions. Add float4 when all leaves' innermost stride is 0 or 1.

### P1: Strided reduce kernel — eliminate contiguify-before-reduce (~8ms, -12 dispatches)
**Files:** `shaders.metal`, `interact/_.c` line 955
**What:** Single-axis reduce on non-contiguous input does contiguify (1 dispatch) + reduce (1 dispatch). Two dispatches for one logical operation.
**Fix:** Add `reduce_sum_strided` kernel that takes ViewParams and reads input via strided indexing directly. Or use `mul_reduce_sum(input, ones, axis)` which already handles ViewParams (partially implemented at interact/_.c:935-960).

### P2: MPS transpose flags — eliminate contiguify-before-mm (~3ms, -4 dispatches)
**Files:** `ops.m` lines 233-244
**What:** Non-contiguous matmul inputs (transposed) get contiguified before MPS. But MPS supports `transposeLeft`/`transposeRight` natively.
**Fix:** Detect when the non-contiguous view is a simple transpose (dims swapped, strides swapped). If so, pass the base buffer with transpose flag instead of contiguifying.

### P2: Widen fusion scope in backward (~12ms, -20 dispatches)
**Files:** `grad/_.c` `fuse_walk_inner` line 47
**What:** `fuse_walk_inner` stops at TAG_TEN (materialized tensors). Adjacent ops that share a single consumer can't be fused.
**Fix:** Track use-counts during backward. When a materialized tensor has exactly one consumer, include it in the fused chain. This lets chains like `t1=mul(a,b); t2=sub(t1,c); t3=mul(t2,d)` fuse into one dispatch even if t1 was materialized.

### P2: Float4 for bc2d, adam, fused_v2 (~5ms)
**Files:** `shaders.metal`, `optim.m`, `fused.m`
**What:** These categories still use scalar (1 element/thread). Float4 gives 4× throughput for memory-bound ops.
**Fix:** Add float4 variants: `add_bc_2d_f4`, `adam_step_f4`. For fused_v2, emit float4 reads when eligible (same check as mdim float4).

### P3: MTLIndirectCommandBuffer for JIT replay (~5ms)
**Files:** `jit.m`
**What:** JIT replay re-encodes 251 dispatches via ObjC calls each step. An ICB encodes once and replays from GPU memory.
**Fix:** After first replay, convert the command sequence to an ICB. Subsequent replays use `executeCommandsInBuffer:` — zero CPU encoding.

## Expected Result

| Optimization | Time Saved | Dispatches Saved |
|-------------|-----------|-----------------|
| P0: Parallel mrs | ~35ms | 0 |
| P1: mdim fused_v2 | ~10ms | 0 |
| P1: Strided reduce | ~8ms | -12 |
| P2: MPS transpose | ~3ms | -4 |
| P2: Wider fusion | ~12ms | -20 |
| P2: Float4 everywhere | ~5ms | 0 |
| P3: ICB JIT replay | ~5ms | 0 |
| **Total** | **~78ms** | **-36** |

From 750ms → ~670ms GPU per step, 251 → ~215 dispatches.
Total wall time: ~76s → ~55-60s (tinygrad: 20.5s, gap: ~2.8×).

## Remaining Gap After All Above

The ~2.8× gap after all optimizations comes from:
1. **tinygrad's graph-level scheduling** — fuses entire conv (im2col+mul+sum) into single kernels
2. **tinygrad's lazy evaluation** — never materializes intermediate buffers
3. **tinygrad's TinyJit** — replays compiled graphs with zero Python/IC overhead
4. **tinygrad's optimal tiling** — beam-searched thread/group sizes per kernel shape

Closing this requires TinyHVM to adopt a **lazy graph compiler** architecture (building the full computation graph before scheduling) rather than the current eager IC reducer. This is a fundamental architectural change.
