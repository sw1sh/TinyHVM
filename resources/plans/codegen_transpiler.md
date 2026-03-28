# General Codegen Transpiler Plan

**Status: Phases 1-3 DONE** (2026-03-27). New `codegen_kernel_rs` handles any axis config.
Trailing restriction removed. Non-trailing reduces fuse through general codegen.

## Dispatch Gap Analysis (2026-03-27)

2-conv+pool CNN: **TinyHVM 105 vs C-ML ~14 dispatches** (7.5× gap).

Per-step breakdown (TinyHVM):
| Category | Count | Root cause | C-ML equivalent |
|----------|-------|-----------|----------------|
| MM | 6 | Irreducible | 6 |
| Unfused reduce | 18 | Eager dispatch in interact | 0 (fused) |
| MulReduceSum | 14 | Eager dispatch in interact | 0 (fused) |
| Contiguify | 15 | View materialization | 0 (movement folded) |
| Single-op codegen | 44 | GRAD ops dispatch individually | 0 (fused into chains) |
| Multi-op fused | 4 | Working correctly | 4 |
| Fused reduce | 4 | Working correctly | 4 |

Three targets (in priority order):
1. **44 single-op dispatches**: GRAD handler creates one lazy op, RETURN_REDUCED forces
   immediate reduction, op dispatches alone. Need: keep backward chain lazy so adjacent
   ops form fusable chains.
2. **15 contiguify**: View materialization for non-contiguous tensors. Need: fold view
   transforms into next kernel's index computation (the codegen already handles this
   for fused chains — extend to standalone ops).
3. **32 unfused reduces**: Sum-to-shape in GRAD handler dispatches eagerly via
   metal_mul_reduce_sum/op_reduce. Need: route through fuser or general codegen.

## Problem

The current `codegen_kernel()` in `src/backend/metal/codegen.m` only handles **trailing** reduce axes. Non-trailing reduces (e.g., SUM over axis 0 of [B,C,H,W]) require special-cased `metal_mul_reduce_sum` or fall back to unfused dispatch. This limits fusion opportunities and adds dispatch overhead.

## Goal

Make `codegen_kernel()` handle **any reduce axis configuration** — not just trailing. One general transpiler for all fused subgraphs (elementwise + reduce, any axis layout).

## Key Insight

The current codegen generates coordinates for output dimensions from the dispatch grid, then indexes leaves using those coordinates. For reduces, it uses a single flat loop (`for r=0..reduce_dim`) and a flat index (`ridx = base * reduce_dim + r`), which only works for trailing axes.

The fix: generate coordinates for ALL axes. Output-axis coordinates come from the dispatch grid. Reduce-axis coordinates come from a flat reduce loop with coordinate decomposition. Leaf index expressions use ALL coordinates: `offset + c0*s0 + c1*s1 + ... + cN*sN`.

## Architecture

### ReduceSpec

```c
typedef struct {
    u8  is_reduce[MAX_DIM]; // 1 = reduce axis, 0 = output axis
    u32 reduce_type;        // UOP_SUM, UOP_RMAX, or 0 (no reduce)
} ReduceSpec;
```

### Kernel Structure

```metal
kernel void K(device float *out, device const float *in0, ..., uint3 gid) {
    // 1. Grid bounds check (OUTPUT axes only)
    uint ix=gid.x, iy=gid.y, iz=gid.z;
    if (ix>=inner || iy>=mid || iz>=outer) return;

    // 2. Output-axis coordinates from grid
    uint c0 = iz;      // batch dim (output axis 0)
    uint c1 = iy;      // channel dim (output axis 1)
    // etc — one coordinate per output axis, derived from grid groups

    // 3. Reduce accumulator + loop
    float acc = 0.0f;
    for (uint r = 0; r < reduce_numel; r++) {
        // 4. Reduce-axis coordinates from flat index r
        uint c2 = r / W;   // height (reduce axis 0)
        uint c3 = r % W;   // width (reduce axis 1)

        // 5. Leaf index using ALL coordinates
        uint i0 = offset0 + c0*s0_0 + c1*s0_1 + c2*s0_2 + c3*s0_3;
        float t0 = in0[i0];
        // ...more leaves...

        // 6. Op chain
        float t2 = t0 * t1;

        // 7. Accumulate
        acc += t2;
    }

    // 8. Write output
    uint oi = iz * mid*inner + iy * inner + ix;
    out[oi] = acc;
}
```

### Leaf Index Paths

Three paths for per-leaf index computation (same for reduce and non-reduce):

1. **Flat** (non-reduce only): leaf is contiguous with same numel as output. `i = flat_grid_index`.

2. **Coordinate-based** (leaf rank == full rank): `i = offset + sum(c_d * stride[d])` skipping dims where `stride==0` or `shape[d]==1` (broadcast).

3. **Flat-index decomposition** (different rank fallback): compute flat full-shape index from all coordinates `fi = c0*S0 + c1*S1 + ...`, then decompose through leaf's own shape `i = offset + (fi % dim_last) * stride_last + ...`. Works for any rank mismatch.

### Float4

Disabled for reduces (same as current). Non-reduce float4 logic unchanged — checks innermost output axis stride.

## Callers to Update

### Direct callers of `metal_dispatch_kernel` (old interface)

| File | Usage | Migration |
|------|-------|-----------|
| `src/backend/metal/ops.m:44` | Unary op dispatch | Phase 4: update to new interface |
| `src/backend/metal/ops.m:119` | Binary op dispatch | Phase 4: update to new interface |
| `src/backend/metal/fused.m:212` | `metal_contiguify` | Phase 2: pass NULL reduce |
| `src/backend/metal/fused.m:227` | `metal_dispatch_fused_v2` non-reduce | Phase 2: pass NULL reduce |

### Callers of `metal_dispatch_fused_v2`

| File | Usage | Migration |
|------|-------|-----------|
| `src/fuse/_.c:516` | Fused elementwise+reduce dispatch | Phase 3: construct ReduceSpec |
| `src/fuse/materialize.c:103` | Lazy tensor materialization | Phase 3: pass NULL reduce |

### Callers of `metal_mul_reduce_sum` (to eventually eliminate)

| File | Lines | Usage |
|------|-------|-------|
| `src/fuse/_.c:328` | Non-trailing SUM(MUL) in fuser | Phase 3: remove, general codegen handles it |
| `src/interact/_.c:747,819` | Direct reduce ops in interaction handler | Phase 5: long-term migration |
| `src/grad/_.c:130,164,310,440,476` | Gradient sum-to-shape reductions | Phase 5: long-term migration |

## Implementation Phases

### Phase 1: New codegen internals (no interface change)

**Files**: `src/backend/metal/codegen.m`

1. Add `ReduceSpec` typedef at top of codegen.m
2. Write new `codegen_kernel_rs()` accepting `(ops, n_ops, n_leaves, leaf_views, full_shape, reduce_spec)`:
   - Axis classification from ReduceSpec
   - Output-only grid computation
   - Output-axis coordinate generation from grid
   - Reduce loop + reduce-axis coordinate generation
   - Unified leaf indexing (3 paths: flat, coord-based, flat-decomp)
   - Mask conditions using all coordinates
   - Unified read block (reduce and non-reduce share same index vars)
   - Op chain with indent
   - Output write (reduce accumulate or direct)
3. Write new `cg_hash_rs()` including ReduceSpec in hash
4. Write new `cg_get_pipe_rs()` and `metal_dispatch_kernel_rs()` with new interface
5. Keep old `codegen_kernel` / `metal_dispatch_kernel` unchanged (backward compat)

**Verify**: Compile. Existing tests pass (nothing calls new functions yet).

### Phase 2: Route fused dispatch through new codegen

**Files**: `src/backend/metal/fused.m`, `src/backend/metal/codegen.m`

1. Update `metal_dispatch_fused_v2` signature to accept `(out_buf, leaf_bufs, leaf_views, n_leaves, ops, n_ops, full_shape, reduce_spec)`
2. Route ALL cases (reduce + non-reduce) through `metal_dispatch_kernel_rs`
3. Remove legacy `get_fused_pipe_v2` codepath (the string-template MSL)
4. Update `metal_contiguify` to call new dispatch with NULL reduce
5. Update forward declarations in `src/tinyhvm.c`

**Verify**: `test_grad_exact`, `test_cnn_small` pass. Trailing reduces produce same results.

### Phase 3: Update fuser, remove trailing restriction

**Files**: `src/fuse/_.c`, `src/fuse/materialize.c`, `src/tinyhvm.c`

1. In `fuse_or_reduce`: construct ReduceSpec from `is_reduce_ax[]`
2. Remove the trailing-axis check (`trailing = 1; ... if (!trailing) { ... }`)
3. Remove `metal_mul_reduce_sum` special case in fuser (non-trailing SUM(MUL) now handled by general codegen)
4. Pass `&ew_view.shape` (full pre-reduction shape) and `&reduce_spec` to dispatch
5. Update `fuse/materialize.c` to pass NULL reduce (it's always elementwise)
6. Simplify phase-2 lazy re-walk (no trailing check needed)

**Verify**: Non-trailing SUM(MUL) produces correct results. CNN training accuracy matches previous. Compare against tinygrad/numpy.

### Phase 4: Migrate ops.m callers

**Files**: `src/backend/metal/ops.m`

1. Update `metal_op_unary` to call new dispatch interface (full_shape = dv->shape, reduce = NULL)
2. Update `metal_op_binary` to call new dispatch interface
3. Remove old `metal_dispatch_kernel` wrapper
4. Rename `metal_dispatch_kernel_rs` → `metal_dispatch_kernel`

**Verify**: All tests pass. No regressions.

### Phase 5: Dead code cleanup (longer term)

1. Remove `metal_mul_reduce_sum` once all callers in `interact/_.c` and `grad/_.c` are migrated to use fused codegen or the general reduce path
2. Remove legacy codegen functions: `metal_codegen_fused`, `get_fused_pipe_v2`
3. Remove ViewParams-based dispatch paths
4. Remove pre-compiled reduce kernels if all reduces go through JIT codegen

## Dispatch Count Impact

Non-trailing reduces that currently fall back to unfused dispatch (separate contiguify + reduce) will now fuse into a single kernel. Expected reduction: 2-4 fewer dispatches per non-trailing reduce in backward pass. For a 4-conv CNN, backward has ~8 non-trailing reduces → saves ~16-32 dispatches.

## Risk Mitigation

- Phase 1 adds code without changing behavior — zero risk
- Phase 2 changes reduce codepath — compare kernel output numerically (max_abs_diff < 1e-5)
- Phase 3 removes trailing restriction — run full training loop, check loss curve matches
- Keep `metal_mul_reduce_sum` alive until Phase 5 (interact/grad callers still need it)
