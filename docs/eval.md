# Evaluation: 3-Phase Sequence

TinyHVM evaluates tensor programs through three phases of interaction-net reduction,
each with different semantics enabled.

## Overview

```
thvm_eval(ctx, t):
    Phase 1:  thvm_reduce(ctx, t)          // pure IC reduction
    Phase 2:  sched_all(ctx, t)            // lowering: TOP → UOP_KERNEL
    Phase 3:  thvm_reduce(ctx, t)          // dispatch + ASSIGN (with flag)
```

## Phase 1: Pure IC Reduction

**Goal**: Propagate GRAD, fire combinators, expose compute frontier.

The reducer runs standard interaction calculus rules:
- GRAD interactions fire eagerly (backward graph construction)
- APP/LAM beta reduction
- IFZ conditional evaluation
- MAT/CTR pattern matching
- DUP cloning, ERA erasure
- Compute ops (`TAG_TOP`) are **WNF** — they stay as lazy terms

**Result**: A term where all combinators have fired, leaving:
- `TAG_TOP` compute ops (the "compute frontier")
- `TAG_TEN` tensor leaves
- `TAG_ERA` for erased branches

## Phase 2: Lowering (Scheduling)

**Goal**: Convert compute ops into dispatchable kernel specs.

`sched_all(ctx, root)` is an imperative rewrite pass (not reduction):

1. **Boundary selection** (`sched_collect_boundaries`): Find all TAG_TOP nodes
2. **Pruning** (`sched_select_boundaries`): Keep only ENSURE points (nodes with consumers)
3. **Fusion** (`fuse_build_kernel`): Walk backward from each boundary, collect ops into fused kernels until hitting tensor leaves
4. **Installation** (`sched_install_kernel`): Rewrite heap — boundary TAG_TOP → `UOP_KERNEL(kid)`

Multi-consumer deduplication: if the same compute subgraph appears at multiple positions,
it gets one kernel ID and the result is cached in `kid_results[]`.

**Result**: Heap rewritten with `UOP_KERNEL` nodes carrying kernel IDs.

## Phase 3: Dispatch

**Goal**: Execute kernels, materialize tensors, fire ASSIGNs.

The reducer runs again with `_assign_dispatch_enabled=1`:

- `UOP_KERNEL` handler fires: reads kernel ID, dispatches to Metal/CPU backend
- Dependencies dispatched recursively (bottom-up)
- Results cached per kernel ID (fire once, reuse via `kid_results[]`)
- `ASSIGN` nodes fire: copy gradient into weight tensor slots
- `LOG_PRINT` sinks consume and display tensor values

**Result**: All compute materialized, gradient slots updated, program reduced to values.

## Future Direction: reduce → reduce → reduce

The goal is to replace the imperative `sched_all()` with interaction-driven lowering:

```
Phase 1:  thvm_reduce(ctx, t)                    // pure IC
Phase 2:  thvm_reduce(ctx, UOP_FUSE(t))          // fusion as interaction
Phase 3:  thvm_reduce(ctx, UOP_SCHED(t))         // planning + dispatch as interaction
```

New UOP signals:
- **UOP_FUSE**: lowering request — walks exposed blocked compute, creates UOP_KERNEL nodes
- **UOP_SCHED**: planning signal — walks kernel DAG, assigns buffer slots, plans memory
- **UOP_KERNEL** (was UOP_FUSING): the final runtime kernel node

This makes all three phases uniform IC reduction, with UOP signals driving transitions.

## Key State

| Global | Purpose |
|--------|---------|
| `sched_kernels[SCHED_MAX_KERNELS]` | Kernel specs (KernelEntry) |
| `kid_results[SCHED_MAX_KERNELS]` | Dispatch results (TAG_TEN or TAG_ERA=pending) |
| `_assign_dispatch_enabled` | Phase 3 gate flag |

## Files

- `src/schedule/_.c` — `thvm_eval`, `sched_all`, `thvm_sched_dispatch_kernel`
- `src/reduce/_.c` — `thvm_reduce` (trampoline reducer)
- `src/interact/tensor_ops.c` — `UOP_KERNEL` dispatch handler
- `src/tinyhvm.h` — KernelEntry, sched state, UOP enums
