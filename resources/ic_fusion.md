# IC Op Fusion

How TinyHVM fuses operations into single GPU kernel dispatches.

## Three Fusion Paths

### 1. Rewrite Rules (forward, lazy TAG_TOP chains)

Declarative rules in `src/rewrite/_.c` fire during reduction:

```
rule_reshape_reduce_fuse:  RESHAPE(SUM/RMAX(ew_chain)) → fused kernel
rule_sum_fuse:             SUM/RMAX(ew_chain)           → fused kernel
rule_elementwise_fuse:     EW(EW_chain)                 → fused kernel
```

The trampoline enters a TAG_TOP → `rewrite_apply` checks rules → `fuse_or_reduce`
walks the lazy chain via `fuse_walk_inner` → collects ops + leaf views → dispatches
via `metal_dispatch_fused_rs` with ReduceSpec.

### 2. Deferred Elementwise (backward, interact handler)

Elementwise ops in the interact handler create tensors with `buf_id=0` instead of
dispatching immediately. When a boundary (MM, SUM, ASSIGN, thvm_to_host) needs the
data, `ENSURE` → `tensor_materialize` walks the provenance chain and dispatches a
fused kernel.

```
GRAD: MUL(gy, bt)  → deferred (buf_id=0)
GRAD: ADD(da, db)   → deferred
ENSURE at SUM       → materialize_walk → fused kernel for MUL+ADD chain
```

Shared intermediates (defer_consumers > 0) get multi-output side buffers written by
the same kernel dispatch.

### 3. SUM Deferral (backward reduces)

When SUM fires and its input is a deferred elementwise op (buf_id=0, unshared):
- SUM itself is deferred (buf_id=0, creator_op=UOP_SUM)
- tensor_materialize detects deferred SUM → walks elementwise input chain
- Dispatches a fused reduce+elementwise kernel via codegen with ReduceSpec

Additionally, `rule_sum_fuse` materializes deferred TAG_TEN children before the SUM
dispatches (bridges deferred dispatch with rewrite rules).

## Codegen

All fused dispatches go through `codegen_kernel_rs`:
- **ReduceSpec**: per-axis reduce classification, any axis configuration
- **Multi-output**: side buffer params + extra write statements for shared intermediates
- **View composition**: fuse_walk_inner composes PERMUTE/EXPAND/RESHAPE into leaf index expressions
- **Float4 vectorization**: for aligned contiguous non-reduce non-masked kernels

## Safety Check (Fused Reduce Backward)

When a fused reduce chain has grad-tracked leaves, the backward needs intermediate
data from virtual tensors. Ops whose backward reads `at` (input data) from virtual
intermediates are rejected:

- **Blocked in fused reduces**: DIV, MAX, LOG (backward needs `at`)
- **Safe**: ADD, SUB, NEG, MUL, RELU, EXP, SQRT (backward uses `y` or leaves only)

## SUM Provenance for Backward

Fused SUM(elementwise) creates a virtual intermediate chain:
- Virtual intermediates (lower tensor IDs) with pre-reduce shapes
- dst (higher ID) with creator_op=UOP_SUM pointing to virtual ew_last
- Backward walks: SUM → expand gy → elementwise chain → leaves

This ensures GRAD applies SUM backward (expand) before elementwise backward,
with correct shapes for sum_to_shape.
