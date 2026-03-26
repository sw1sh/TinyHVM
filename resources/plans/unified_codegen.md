# Unified Codegen: One JIT Path for All Elementwise Ops

## Problem

9 different dispatch paths for elementwise ops:
- fast_un/fast_bin (compiled scalar flat)
- f4_un/f4_bin (compiled float4 flat)
- bc2d_* (compiled 2D broadcast)
- mdim_bin/mdim_f4/mdim_un (JIT 3D coordinate)
- slow_un/slow_bin (compiled strided_idx ViewParams)
- fused_v2 (JIT 1D strided_idx multi-op)
- contiguify (JIT 1D strided_idx identity)

## Solution: One JIT codegen function

```
codegen_kernel(ops, n_ops, leaves, n_leaves, out_shape, has_reduce, reduce_dim)
```

Generates a kernel that:
1. Uses 3D dispatch (gid.xyz) with output dims collapsed into 3 groups
2. Per-leaf: baked-in index expression using output-shape coordinate decomposition
   - stride=1 on innermost → float4 load (when all leaves qualify)
   - stride=0 → scalar broadcast (load once, reuse)
   - other → multiply-only index from coordinates
3. Multi-op chain in the body (same as current fused_v2 EMIT_OPS)
4. Optional trailing reduce loop
5. Float4 output when eligible

This replaces ALL of:
- fast_un/fast_bin → 1-op kernel with 1 contiguous leaf
- f4_un/f4_bin → 1-op kernel with 1 contiguous leaf, float4
- bc2d_* → 1-op kernel with 1 contiguous + 1 broadcast leaf
- mdim_bin/mdim_f4 → 1-op kernel with 2 arbitrary leaves
- mdim_un → 1-op kernel with 1 arbitrary leaf
- slow_un/slow_bin → 1-op kernel with arbitrary strides (coordinate decomp replaces strided_idx)
- fused_v2 → N-op kernel with M arbitrary leaves
- contiguify → 0-op kernel with 1 arbitrary leaf (identity copy)

## Cache key
(n_ops, op_chain_hash, per_leaf_stride_pattern, output_shape, float4_flag)

## Dispatch
3D grid: (inner, mid, outer) collapsed from output shape.
Threadgroup: (min(inner, 256), 1, 1).
Float4: inner/4 when eligible.

## What this enables
- The scheduler (lazy_graph_compiler.md) has ONE dispatch target
- Kernel quality is uniform — all ops get mdim coordinate decomposition
- No slow strided_idx path anywhere
- Float4 applies to all eligible ops, not just a few

## What gets deleted
- shaders.metal: All unary_*/binary_*/fast_*/fast4_* kernels (lines 48-800+)
- ops.m: The entire is_fast_view/bc2d/fast/slow dispatch tree (lines 1-226)
  Replaced with: call unified codegen for every binary/unary
- fused.m: codegen_fused_v2, try_mdim_float4, get_mdim_binary_pipe, mdim_un
  Replaced with: ONE codegen_kernel function
- init.m: All fpipe_*, f4pipe_*, bc2d_*, pipe_add/mul/etc. pipeline state objects
  Replaced with: JIT cache only
