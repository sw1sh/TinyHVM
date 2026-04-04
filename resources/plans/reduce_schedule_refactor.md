# Plan: Single-Pass Fusion Optimization

## Architecture (revised)

Two-pass reduce→schedule→reduce doesn't work:
- IC reduction is destructive (modifies heap)
- Deferring ops breaks fusion contract (ew intermediates are virtual in kernels)

**Correct approach**: single-pass reduce with inline fusion decisions.
The trampoline walks the graph depth-first. At each node, the interact handler
makes fusion decisions. This IS the existing architecture — we optimize it.

## Current State
- 368 dispatches (reduce=159, fused=195, adam=14), 94.4% accuracy
- Tinygrad: 73 dispatches for same model
- MM decomposed to EXPAND+MUL+SUM (thvm_mm)
- TAG_TOP GRAD handler implemented (compiles, not yet active)
- Scheduler infrastructure: fusing specs, memory planner, absorbed marking

## Dispatch Analysis (from earlier tracing)
- 91% of dispatches from tensor_materialize (ENSURE cascade)
- Forward: 33 fused by rewrite rules, rest by tensor_materialize
- Backward: reduces dispatch eagerly, ew inputs materialized prematurely by ENSURE
- 4 reduces/step fall through (PERMUTE(RESHAPE(ew)) pattern not handled)

## Optimization Targets (single-pass)

### 1. Backward reduce deferral
Currently guarded by `!ctx->no_grad_alloc`. Removing the guard gets 347 dispatches
(21 fewer) but breaks accuracy (11.7%). The issue: some backward reduces defer
correctly but others produce wrong results when their ew input chain has been
partially consumed by another backward path.

Fix: only defer backward reduces when the ew input has `defer_consumers == 0`
(no other consumer). This is already checked. The 11.7% issue needs debugging.

### 2. PERMUTE(RESHAPE(ew)) fusion in interact handler
The PERMUTE fusion path at line 619 only handles PERMUTE(ew) directly.
PERMUTE(RESHAPE(ew)) falls through to ENSURE → 2 dispatches instead of 1.
Fix: walk through RESHAPE under PERMUTE to find ew base.

### 3. Post-reduce ew fusion in tensor_materialize
Already partially implemented. Can be extended to absorb more backward patterns.

### 4. Memory planner integration
The JIT memory planner works at JIT capture time. Can be activated earlier
(during tensor_materialize) for within-step buffer reuse.

## What Works
- Default path: 94.4%, 368 dispatches (correct baseline)
- MM decomposition: correct forward, backward uses thvm_mm
- TAG_TOP GRAD handler: compiles, ready for activation
- Scheduler infrastructure: available for future use
