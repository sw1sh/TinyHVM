# Plan: Close the Dispatch Gap with C-ML

## Current State (2026-03-28)

| Architecture | TinyHVM | Tinygrad | C-ML target | Gap |
|---|---|---|---|---|
| 1-layer CNN | **46** disp, 6ms | 16, ~8ms | ~8 | 5.8× |
| 2-conv+pool | **78** disp, 17ms | 33, 16ms | ~14 | 5.6× |
| 4-conv+pool | **124** disp, 30ms | 45, 61ms | ~22 | 5.6× |

TinyHVM is **matched or faster in wall time** than tinygrad for 2+ conv layers,
despite 2-3× more dispatches. Per-kernel efficiency is high thanks to Metal
command buffer batching and JIT codegen.

## Completed Optimizations

- [x] General codegen transpiler (any axis reduce, ReduceSpec)
- [x] Deferred elementwise dispatch (tensor_materialize fuses chains)
- [x] Multi-output kernels (shared intermediates as side buffers)
- [x] Shared tensor detection (defer_consumers tracking)
- [x] Lazy GRAD ENSURE (backward chains stay deferred longer)
- [x] RELU backward uses output not input (safe in fused reduces)
- [x] Deleted backward_local (pure IC gradient only)
- [x] All reduces through general codegen (eliminate reduce/mrs categories)
- [x] SUM deferral for unshared inputs
- [x] rule_sum_fuse materializes deferred children
- [x] Dead-branch skip in BIN_GRAD

## Remaining Dispatch Breakdown (78 for 2-conv+pool)

| Category | Count | Path to eliminate |
|----------|-------|-------------------|
| **MM** | 6 | Fused GEMM+bias+relu (Phase 5) |
| **Contiguify** | ~15 | View composition in materialize (Phase 1) |
| **Standalone reduce** | ~12 | SUM deferral for shared inputs (Phase 3) |
| **Single-op deferred** | ~12 | Longer chains via shared-input fusion (Phase 2) |
| **Fused ew/reduce** | ~33 | Already optimal |

## Phase 1: Eliminate Contiguify (-10 dispatches → ~68)

**15 contiguify = 6 ASSIGN blits + 9 RESHAPE materializations.**

### ASSIGN blits
SGD `ASSIGN(param, new_val)` calls `metal_contiguify`. If src is contiguous
(common after materialize), replace with `buf_copy` — zero dispatch overhead.

### RESHAPE materialization
Conv backward produces non-reshapable strides (PERMUTE → EXPAND → RESHAPE fails).
Fix: compose views in materialize_walk like fuse_walk_inner does. The attempt
earlier failed on SHRINK/PAD — need to handle those view types or treat them
as leaf boundaries (current fuser behavior).

## Phase 2: Longer Deferred Chains (-5 dispatches → ~63)

12 single-op deferred chains break because the next consumer ENSUREs immediately.
Fix: extend multi-output to let SUM/RMAX include deferred inputs as side outputs,
so the deferred chain extends through the reduce boundary.

## Phase 3: Fuse Reduce + Upstream Elementwise (-8 dispatches → ~55)

12 standalone reduces have materialized inputs. The softmax diamond (EXP shared by
SUM and DIV) blocks SUM deferral. Fix with multi-output: fused SUM(EXP) writes
both reduced output and EXP intermediate. DIV reads the side buffer.

Requires:
1. Diamond detection in materialize path
2. Side-output writes inside reduce loops (codegen change)
3. Update DIV to read from side buffer instead of deferred tensor

## Phase 4: Clean Legacy Paths (simplification, no dispatch change)

- Remove `metal_dispatch_kernel` old wrapper
- Remove `codegen_kernel` / `cg_get_pipe` old wrappers
- Remove `metal_mul_reduce_sum` (fully replaced)
- Remove pre-compiled reduce kernels from ops.m
- Remove `get_fused_pipe_v2` / `metal_codegen_fused` legacy codegen
- Migrate ops.m single-op dispatch to use metal_dispatch_kernel_rs directly

## Phase 5: MM Fusion (-6 dispatches → ~49)

Matmul via MPS is unfused. Options:
1. Custom tiled GEMM shader fusing bias_add + relu
2. MPS activation post-processing
3. Fuse bias_add into downstream elementwise (consumer fusion)

## Theoretical Minimum (~20 dispatches)

```
Forward:  conv1(1) + pool1(1) + conv2(1) + pool2(1) + linear(1) + softmax+CE(3) = 8
Backward: ~10 fused kernels (reduce+ew chains)                                  = 10
SGD:      1-2 fused weight updates                                               = 2
Total:    ~20 dispatches
```

With MPS matmul unfusable: 6 MM + 14 fused = 20.
Current gap: 78/20 = 3.9×. Most remaining dispatches are contiguify (15)
and standalone reduce (12) — both addressable with the phases above.
