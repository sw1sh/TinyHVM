# General Codegen Transpiler

**Status: DONE.** `codegen_kernel_rs` handles any reduce axis config, multi-output
side buffers for shared intermediates, and all dispatch paths unified.

## Architecture

```
codegen_kernel_rs(out_buf, leaf_bufs, leaf_views, n_leaves,
                  ops, n_ops, full_shape, reduce_spec,
                  side_bufs, side_op_indices, n_side_outputs)
```

- **ReduceSpec**: per-axis `is_reduce[MAX_DIM]` + `reduce_type` (SUM/RMAX/0)
- **Multi-output**: side_bufs/side_op_indices write shared intermediates alongside main output
- **Axis classification**: output axes → 3D dispatch grid, reduce axes → inner loop
- **Leaf indexing**: 3 paths (flat, coordinate-based, flat-decomposition)
- **Float4**: auto-vectorization for aligned non-reduce, non-masked kernels

## Dispatch Paths

All compute now flows through ONE codegen:
1. **Forward fusion** (rewrite rules → fuse_or_reduce → metal_dispatch_fused_rs)
2. **Deferred elementwise** (interact handler → buf_id=0 → tensor_materialize → codegen)
3. **Standalone reduce** (interact handler → codegen with ReduceSpec)
4. **Contiguify** (metal_contiguify → codegen with 0 ops, 1 leaf)
5. **Legacy single-op** (ops.m → metal_dispatch_kernel → wrapper → codegen_kernel_rs)

The old `metal_dispatch_fused_v2` and `metal_mul_reduce_sum` paths are eliminated.
`reduce` and `mrs` dispatch categories no longer exist.

## Current Dispatch Breakdown (2-conv+pool CNN, 78 total)

| Category | Count | Notes |
|----------|-------|-------|
| MM (matmul) | 6 | Irreducible — MPS accelerated |
| Fused elementwise | ~20 | Forward + backward chains (2-8 ops each) |
| Fused reduce+ew | ~8 | SUM(MUL) from rewrite rules |
| Standalone reduce | ~12 | SUM/RMAX with materialized inputs |
| Contiguify | ~15 | RESHAPE materialization + ASSIGN blits |
| Multi-output fused | ~5 | Shared intermediates via side buffers |
| Deferred+materialized | ~12 | Single-op deferred chains |
