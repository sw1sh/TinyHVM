# Evaluation: Unified IC Reduction with Scheduling Signals

## Core Principle

`thvm_reduce` is always pure phase 1 — it reduces standard IC interactions and
stops at WNF (compute ops stay as lazy `TAG_TOP` terms).

`thvm_eval` injects scheduling signals after reduce reaches WNF. The choice of
calling `thvm_reduce` vs `thvm_eval` determines whether you get phase 1 only
or the full pipeline.

## Overview

```
thvm_reduce(ctx, t)     — pure IC reduction, stops at WNF

thvm_eval(ctx, t):
    1. thvm_reduce(ctx, t)             — phase 1: pure IC to WNF
    2. thvm_reduce(ctx, UOP_FUSE(t))   — phase 2: fusion as interaction
    3. thvm_reduce(ctx, UOP_SCHED(t))  — phase 3: plan + dispatch as interaction
```

All three phases are uniform IC reduction. The scheduling signals (UOP_FUSE,
UOP_SCHED) are injected by eval between phases after the previous reduce
reaches WNF.

## Phase 1: Pure IC Reduction (`thvm_reduce`)

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

## Phase 2: Fusion (`UOP_FUSE` interaction)

**Goal**: Convert compute ops into dispatchable kernel specs.

When `UOP_FUSE` fires as an interaction, it calls `sched_all()` which:

1. **Boundary selection** (`sched_collect_boundaries`): Find all TAG_TOP nodes
2. **Pruning** (`sched_select_boundaries`): Keep only ENSURE points
3. **Fusion** (`fuse_build_kernel`): Walk backward, collect ops into fused kernels
4. **Installation** (`sched_install_kernel`): Rewrite heap — TAG_TOP → `UOP_KERNEL(kid)`

Multi-consumer deduplication: same subgraph gets one kernel ID, result cached.

**Result**: Heap rewritten with `UOP_KERNEL` nodes carrying kernel IDs.

## Phase 3: Planning + Dispatch (`UOP_SCHED` interaction)

**Goal**: Execute kernels, materialize tensors, fire ASSIGNs.

When `UOP_SCHED` fires, it enables dispatch and returns the payload for further
reduction:

- `UOP_KERNEL` handler fires: reads kernel ID, dispatches to Metal/CPU backend
- Dependencies dispatched recursively (bottom-up)
- Results cached per kernel ID (fire once, reuse via `kid_results[]`)
- `ASSIGN` nodes fire: copy gradient into weight tensor slots
- `LOG_PRINT` sinks consume and display tensor values

**Result**: All compute materialized, gradient slots updated, program reduced to values.

## Scheduling Signal UOPs

| UOP | Role | Fires during |
|-----|------|-------------|
| `UOP_FUSE` | Lowering request: boundary walk → create UOP_KERNEL | Phase 2 |
| `UOP_SCHED` | Planning signal: enable dispatch, (future: memory planning) | Phase 3 |
| `UOP_KERNEL` | Fused kernel node: dispatches to backend | Phase 3 |

## Key State

| Global | Purpose |
|--------|---------|
| `sched_kernels[SCHED_MAX_KERNELS]` | Kernel specs (KernelEntry) |
| `kid_results[SCHED_MAX_KERNELS]` | Dispatch results (TAG_TEN or TAG_ERA=pending) |
| `_assign_dispatch_enabled` | Phase 3 gate flag (set by UOP_SCHED) |

## Files

- `src/schedule/_.c` — `thvm_eval`, `sched_all`, `thvm_sched_dispatch_kernel`
- `src/reduce/_.c` — `thvm_reduce` (trampoline reducer)
- `src/interact/tensor_ops.c` — UOP_FUSE, UOP_SCHED, UOP_KERNEL handlers
- `src/tinyhvm.h` — KernelEntry, sched state, UOP enums
