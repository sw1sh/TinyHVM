# Lazy Graph Compiler Plan

## The Problem

TinyHVM's eager IC reducer fires ops one at a time (251 GPU dispatches/step).
tinygrad builds the full graph then schedules fused kernels (186 dispatches, 20× faster GPU).
Per-kernel optimizations give diminishing returns. We need graph-level scheduling.

## Architecture

```
CURRENT:  thvm_op → TAG_TOP → thvm_reduce → thvm_interact → 1 GPU dispatch per op

NEW:      thvm_op → TAG_TOP → thvm_schedule → [ScheduleEntry*] → thvm_exec → batched dispatch
```

IC net stays as user-facing API. Scheduler walks the TAG_TOP DAG, partitions
into fusable kernel groups, codegens each, dispatches as one sequence.

## Core Data Structure

```c
typedef enum {
    SCHED_FUSED_EW,         // elementwise chain
    SCHED_FUSED_EW_REDUCE,  // elementwise + trailing reduce
    SCHED_MUL_REDUCE_SUM,   // existing mrs pattern
    SCHED_MM,               // MPS matmul (barrier)
    SCHED_CONTIGUIFY,       // forced materialization
    SCHED_ASSIGN,           // in-place update
} SchedKind;

typedef struct {
    SchedKind kind;
    FusedOp   ops[64];     u32 n_ops;
    u32       leaf_ids[32]; const View *leaf_views[32]; u32 n_leaves;
    u32       out_tensor_id; Shape out_shape; u32 out_numel;
    u32       reduce_dim;
    u32       mm_a_id, mm_b_id, M, K, N;  // MM fields
    Term      source_term;                  // provenance
    u32       creator_op; u32 src_ids[2]; u8 requires_grad;
} ScheduleEntry;
```

## Scheduling Algorithm

**Pre-pass: reference counting.** Walk DAG, count consumers per TAG_TOP node.
Multi-consumer nodes are fusion barriers (their output must be materialized).

**Main pass: post-order DAG traversal (`schedule_walk`).**

For each TAG_TOP node:
1. **Already scheduled?** (DAG dedup via heap_loc → entry map) → reuse output tensor
2. **Movement op?** (RESHAPE/EXPAND/PERMUTE/SHRINK/PAD) → inline view alias, no dispatch
3. **MM?** → barrier, schedule inputs first, emit SCHED_MM
4. **Elementwise?** → try to build fused group (greedy walk backward through ew chain)
5. **SUM/RMAX?** → if input is ew chain, fuse as SCHED_FUSED_EW_REDUCE
6. **SUM(MUL)?** → emit SCHED_MUL_REDUCE_SUM

**Fusion rules:**
- Elementwise ops fuse freely (up to 64 ops, 32 leaves)
- One reduce per fused kernel (terminates the chain)
- Multi-consumer nodes are barriers (output shared → can't be internal to a group)
- MM is always a barrier (MPS, separate encoding)
- Movement ops are view aliases (zero cost, resolved inline)

## Integration Points

### Forward Pass
Replace `thvm_reduce(ctx, loss)` with:
```c
Schedule sched;
thvm_schedule(ctx, loss_term, &sched);
thvm_exec(ctx, &sched);
```

### Backward Pass
Two-phase approach:

**Phase 1 (minimal change):** Keep `thvm_backward` as-is but collect all lazy gradient
Terms into a batch. After the backward loop, schedule them together:
```c
// backward_local creates lazy TAG_TOP gradient terms
// grad_accum stores them as TAG_TOP (not immediately reduced)
// After the loop:
for (u32 p = 0; p < n_params; p++) {
    thvm_schedule(ctx, ga_terms[p], &sched);
}
thvm_exec(ctx, &sched);
```

**Phase 2 (full refactor):** Make `ga[]` store `Term` instead of `u32`. All gradient
accumulation is lazy. The entire backward pass produces a single DAG that gets
scheduled in one shot.

### Provenance
Fused kernels get `creator_op = UOP_FUSING` with `fusing_loc` pointing to the
original TAG_TOP root. The existing FUSING backward handler re-reduces the unfused
subnet. This already works for SUM(MUL) fusion.

### JIT Replay
Unchanged — JIT records at dispatch level. Fewer dispatches from scheduler means
fewer JIT commands to record/replay.

## Expected Results

**Dispatch count:** 251 → ~100-120 per step
- Forward: ~120 → ~40-50 (movement ops become free, ew chains fuse)
- Backward: ~120 → ~40-50 (gradient ew chains fuse with reduces)
- Adam: 14 (unchanged)

**GPU time:** ~750ms → ~200-300ms per step
- Eliminated intermediate buffer reads/writes (fused kernels)
- Fewer kernel launches (5-10μs each × 130 fewer = ~1ms)
- Better data locality in fused kernels

**Wall time:** ~76s → ~30-40s (tinygrad: 20.5s, gap: ~1.5-2×)

**Remaining gap after scheduler:** tinygrad's beam-searched tiling, simdgroup_matrix
for matmul, and FP16 where safe. These are kernel-quality improvements on top of
the scheduler.

## Files to Create/Modify

### New: `src/schedule/_.c` (~400 lines)
- `thvm_schedule(ctx, root, sched)` — DAG walk + fusion grouping
- `thvm_exec(ctx, sched)` — linear dispatch through entries
- `schedule_walk(ctx, term, sched)` — recursive post-order traversal
- `try_build_fused_group(ctx, term, sched, entry)` — greedy fusion walk
- `refcount_pass(ctx, term)` — pre-pass consumer counting

### Modify: `src/grad/_.c`
- `thvm_backward`: Phase 1 — collect lazy gradient Terms, schedule at end
- `grad_accum`: store lazy Terms instead of calling fuse_or_reduce
- `backward_local`: return lazy Terms (no immediate reduction)

### Modify: `src/interact/_.c`
- SUM(MUL) fusion handler becomes redundant (scheduler handles it)
- Keep as fallback for `ctx->no_fuse` / debugging

### Modify: `src/backend/metal/fused.m`
- `codegen_fused_v2`: replace strided_idx with mdim coordinate decomposition
- Increase FUSED_CACHE_SIZE to 256
- Add float4 support for fused kernels

### Modify: `src/tinyhvm.h`
- Add Schedule/ScheduleEntry structs
- Add thvm_schedule/thvm_exec declarations

## Implementation Sequence

1. **Stub scheduler** — add Schedule struct, thvm_schedule/thvm_exec that just
   falls back to thvm_reduce. Verify nothing breaks.

2. **Forward scheduling** — implement schedule_walk for forward-only. Handle
   elementwise fusion + MM barriers + movement inlining. Compare dispatches.

3. **Forward execution** — implement thvm_exec. Verify numerical correctness
   against eager reduction on MNIST forward pass.

4. **Backward integration** — make grad_accum lazy, schedule gradients at end.
   Verify accuracy and dispatch reduction.

5. **Codegen upgrade** — mdim indexing in fused_v2, float4 for fused kernels.
   This is enabled by larger fusion groups from the scheduler.

6. **Tune** — fusion group size limits, caching, profiling.
