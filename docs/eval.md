# Evaluation: Reduce Then Fuse

## Core Principle

`thvm_reduce` performs the pure interaction-calculus reduction and stops at WNF.
Lazy compute ops remain as `TAG_TOP` nodes.

`thvm_eval` runs one extra uniform IC pass by wrapping the reduced program in
`FUSE(...)`. That second pass:

- propagates `FUSE`
- materializes structural `KERNEL` nodes on the heap
- dispatches those kernels only when a strict context demands `TEN`

## Overview

```text
thvm_reduce(ctx, t)     -> pure IC reduction to WNF

thvm_eval(ctx, t):
  1. thvm_reduce(ctx, t)
  2. thvm_reduce(ctx, FUSE(t))
```

There is no separate scheduling phase in the live eval path anymore.
There is also no separate `UOP_SCHED` marker anymore: `FUSE` is the only
propagating fusion agent.

## Phase 1: Structural Reduction

The first pass performs the standard local rewrites:

- `APP/LAM`
- `IFZ`
- `GRAD`
- `MAT/CTR`
- `DUP` / `ERA`

Result: the program reaches a lazy frontier where tensor compute is still
represented as `TAG_TOP` structure.

## Phase 2: Fusion and Demand-Driven Dispatch

The second pass wraps the phase-1 result in `FUSE(...)`.

`FUSE` does not dispatch anything directly. Instead it rewrites fuseable
subgraphs into explicit `UOP_KERNEL` nodes:

```text
FUSE(MUL(a,b)) -> KERNEL(FUSE(a), FUSE(b), MUL)
```

Those `KERNEL` nodes stay visible until a strict context reaches them.
Typical strict contexts are:

- `ASSIGN(dst, src)` forcing `src`
- `LOG_PRINT`
- the final root result of `thvm_eval`

When a `KERNEL` is demanded:

1. its structural subtree is lowered into a `KernelEntry`
2. the runtime cache is consulted
3. epoch checks decide reuse vs re-dispatch
4. the handler returns a concrete `TAG_TEN`

## Observable Phases in Step Graphs

The step tracer should now show:

1. pure structural rewrites
2. `FUSE` propagation
3. visible `KERNEL` nodes in the heap
4. `KERNEL -> TEN` dispatch steps
5. `ASSIGN` / cleanup

That is the intended user-facing contract for fusion tracing.

## Key State

| Global | Purpose |
|--------|---------|
| `sched_kernels[SCHED_MAX_KERNELS]` | Lowered runtime kernels built from structural `KERNEL`s |
| `sched_kernel_locs[SCHED_MAX_KERNELS]` | Map runtime kernel entries back to heap-visible `KERNEL` nodes |
| `kid_results[SCHED_MAX_KERNELS]` | Cached dispatch results |
| `buf_epoch[]` | ASSIGN invalidation tracking |

## Files

- `src/schedule/_.c` — `thvm_eval`, dispatch/cache tables, step-graph driver
- `src/interact/tensor_ops.c` — `FUSE` and `KERNEL` handlers
- `src/interact/_.c` — shared helpers for structural kernels
- `src/reduce/_.c` — reducer trampoline
