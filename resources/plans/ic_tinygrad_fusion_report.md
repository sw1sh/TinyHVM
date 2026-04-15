# IC Tinygrad Fusion Report

## Goal

Move TinyHVM's fusion model closer to the architecture described in the tinygrad
blog:

- `FUSE` remains a pure propagating IC agent
- `KERNEL` becomes the explicit fused kernel boundary on the heap
- lowering, dispatch, and caching become downstream runtime details

## Tinygrad vs Old TinyHVM

### Tinygrad

- one graph IR (`UOp`) through the whole compiler
- single sink root for demanded outputs
- schedule / rangeify decides kernel boundaries
- explicit kernel-like nodes exist before lowering/runtime
- runtime executes the chosen kernel schedule

### Old TinyHVM

- `FUSE` propagated locally, which was good
- but absorbed ops were turned into `FUSE2(op, left, right)`
- `KERNEL` mostly acted like a dispatch handle (`kid`) rather than the visible
  fused structure itself
- step graphs often had to reconstruct kernels later from tensor provenance

### Mismatch

The visible IR and the runtime IR were different things:

```text
visible: FUSE / FUSE2 / ERA steps
runtime reality: hidden dispatch-handle KERNELs
```

That mismatch is what made kernel tracing feel synthetic.

## Implemented Direction

The current implementation now follows this shape:

```text
thvm_reduce(program)
-> thvm_reduce(FUSE(program))
-> visible structural KERNEL nodes
-> demand-driven KERNEL -> TEN dispatch
-> ASSIGN / sinks consume TEN
```

## What Changed

### 1. Structural KERNEL replaced FUSE2 in the live path

- `FUSE` now rewrites fuseable structure directly into `UOP_KERNEL`
- structural kernel heap layout is:

```text
KERNEL(left, right_or_meta, NUM(root_uop))
```

- `UOP_FUSE2` remains only as a deprecated compatibility shim

### 2. Runtime state moved behind the structural node

- runtime still uses `KernelEntry`, `kid_results[]`, and epoch tracking
- but those are now discovered from the structural kernel location
- `sched_kernel_locs[kid]` maps runtime entries back to heap-visible kernels

### 3. Dispatch became demand-driven again

- `KERNEL` stays visible until a strict context reaches it
- `ASSIGN`, `LOG_PRINT`, and final eval demand can force dispatch
- `SEQ`-shaped kernels degrade back to `SEQ`, preserving explicit ordering

### 4. Step graphs now show real kernel nodes

- fused traces contain explicit `KERNEL_*` steps
- the graph harness now passes with visible kernel steps in
  `n1/n2/n3_steps_fuse_vals`

## Why This Is More Faithful to Tinygrad

Tinygrad's architecture separates:

- inter-kernel structure
- intra-kernel lowering
- runtime execution

The structural-`KERNEL` model gives TinyHVM the same split while keeping the
mechanism local and IC-native:

- `FUSE` is the local graph-rewrite pressure
- `KERNEL` is the explicit inter-kernel boundary
- `fuse_build_kernel()` and backend codegen are the intra-kernel lowering
- cache/dispatch is runtime-only state

## Remaining Tradeoffs

- TinyHVM still lowers kernels lazily at dispatch time, not in one explicit
  schedule-building pass like tinygrad's rangeify pipeline.
- Runtime entries are still indexed by `kid`; they are just no longer mistaken
  for the public IR.
- The old `sched_all()` path still exists as legacy machinery and documentation,
  even though the live eval path is now `reduce(FUSE(program))`.

## Recommended Next Moves

1. Keep the public contract centered on structural `KERNEL`, not `kid`.
2. Treat `sched_all()` as legacy implementation support unless a
   new explicit planner is reintroduced.
3. If TinyHVM later grows a more explicit scheduler, make it operate over the
   visible `KERNEL` DAG instead of inventing a second hidden kernel IR.
4. Continue using step graphs as the primary oracle: if a kernel is not visible
   there, the architecture is drifting again.
