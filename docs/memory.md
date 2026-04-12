# Memory Planning

## Problem

The scheduler currently allocates physical tensor storage (`tensor_create()`) at schedule time
(phase 2). This is wrong — boundary tensors should be **logical planned values** until dispatch.

Reasons:
- Tensors allocated early cannot share buffers (no reuse analysis yet)
- Tinygrad achieves 0.82GB at BS=512 via 58→1 buffer reuse planner
- Physical allocation during scheduling mixes two abstraction layers

## Design

### Separation of Concerns

| Layer | Responsibility | When |
|-------|---------------|------|
| **Fusion** (UOP_FUSE) | Choose compute boundaries, create kernel nodes | Phase 2 |
| **Planning** (UOP_SCHED) | Assign logical slot IDs, compute use counts, map buffer reuse | Phase 2/3 boundary |
| **Dispatch** (UOP_KERNEL) | Allocate physical buffers, execute kernels | Phase 3 |

### UOP_SCHED: Planning Signal

When UOP_SCHED fires as an interaction:
1. Walk the kernel DAG (acyclic dependency graph of UOP_KERNEL nodes)
2. Assign logical value IDs to each kernel output
3. Compute use counts (how many consumers per value)
4. Plan buffer reuse: values with non-overlapping lifetimes share physical slots
5. No physical allocation — output is annotated kernel DAG with slot assignments

### Logical vs Physical Values

```
Phase 2 output:  K1[slot=0] → K2[slot=1] → K3[slot=0]  (slot 0 reused)
Phase 3 input:   Allocate physical buffers for unique active slots only
```

### Boundary Tensors

Current: `sched_prepare_boundary_output()` calls `tensor_create()` per kernel output.
Target: boundary outputs are logical slot IDs. Physical buffers allocated lazily at dispatch.

## Tinygrad Reference

Tinygrad's memory planner (`tinygrad/engine/memory.py`):
- Walks scheduled kernel DAG
- Assigns buffer numbers based on lifetime analysis
- Achieves massive reduction: 58 unique buffers → 1 via reuse at BS=512
- Key insight: non-overlapping lifetimes of intermediate values enable sharing

## Phase 3 Execution Net

The phase 3 runtime net is a **general IC graph** (not necessarily a DAG):
- Kernel dependencies form a DAG
- But the surrounding combinator structure (loop, conditionals) may have cycles
- The memory planner dependency graph must still be acyclic
- Plan the DAG portion; let combinators drive iteration

## Integration with Loop

For training loops with kernel re-firing:
- Single iteration's kernels are planned once
- Buffer slots are reused across iterations (same kernel, fresh data)
- The planner doesn't need to see the loop — it plans one iteration's worth
- Phase 3 reduction naturally re-fires planned kernels via combinator sequencing

## Files

- `src/schedule/_.c` — current `sched_all()`, `sched_prepare_boundary_output()`
- `src/interact/tensor_ops.c` — UOP_KERNEL dispatch, `thvm_sched_dispatch_kernel()`
- `src/tinyhvm.h` — KernelEntry, sched_slot_*, buffer management
- Reference: `resources/plans/memory_planner.md`
