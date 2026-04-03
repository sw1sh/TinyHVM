# Symbolic Shape Tracking for Backward Fusion

## Status: Partially Implemented

Steps 1-2 done. Steps 3-4 revealed architectural insights — see below.

## The Problem

TinyHVM has 375 dispatches/step. Tinygrad has ~20. The gap is:
- Forward: ~30 dispatches (fusion works via rewrite rules + deferred ew chains)
- Backward: ~330 dispatches (fusion partially disabled by `ctx->no_fuse=1`)
- Adam: ~14

### Dispatch Breakdown (after investigation)
- mm=3 (MPS matmul — opaque barrier, can't fuse through)
- reduce=154 (102 fused reduce+ew + 52 solo reduces on materialized inputs)
- fused=200 (ew-only fused kernels, forward + deferred backward chains)
- adam=14

## What Was Done

### Step 1: Shape Table for GRAD3 Terms (DONE)
All GRAD3 terms now have shape table entries via inline st_set in the GRAD3 macro.
This ensures fuse_walk_inner can get shapes for any TAG_TOP term.

### Step 2: Separate Concerns — no_grad_alloc (DONE)
Added `no_grad_alloc` flag to TinyHVM struct:
- `no_fuse` controls rewrite rule firing
- `no_grad_alloc` controls buffer allocation for virtual intermediates + reduce deferral
- During backward: no_grad_alloc=1 prevents separate buffer allocation AND reduce deferral

**Critical fix**: The reduce deferral guard at interact/_.c used `no_fuse` but should use
`no_grad_alloc`. This was the root cause of previous zero gradient bugs when trying to
enable backward fusion.

### Step 3-4: What Didn't Work As Planned

**Removing no_fuse entirely (setting no_fuse=0 during backward)**: Tested and works
(gradients correct) BUT doesn't reduce dispatches because:

1. **rule_elementwise_fuse during backward breaks deferred chains**: Backward ew ops
   should DEFER (buf_id=0, creator_op/src_ids set) so SUM/RMAX can fuse with the full
   chain. When rule_elementwise_fuse fires on backward ew chains, it dispatches a fused
   kernel IMMEDIATELY — materializing the chain. Then SUM sees a materialized input and
   can't fuse. Result: MORE dispatches, not fewer.

2. **The 52 solo reduces are irreducible**: These are SUMs on already-materialized inputs
   (e.g., MM outputs, ENSURE'd backward tensors). They can't fuse because there's no
   deferred ew chain to absorb. They represent the irreducible cost of shape broadcasting
   (sum_to_shape) in the backward.

## Fundamental Architecture Understanding

**Why tinygrad has ~20 dispatches**: It builds the ENTIRE fwd+bwd as one lazy graph.
The scheduler groups ops with "one reduce per kernel" — each kernel fuses all upstream
ew ops into the reduce. With ~10 reduces total, you get ~20 kernels.

**Why TinyHVM has ~375 dispatches**: The IC reducer processes ops eagerly. Deferral
helps (ew ops defer, SUM fuses with deferred chains) but:
- MM (matmul via MPS) is an opaque barrier — forces materialization of inputs
- Each conv backward creates ~3 MM dispatches
- sum_to_shape creates standalone SUMs for broadcasting
- The backward creates ~100+ individual reduces

**The irreducible gap**: With MPS matmul as an opaque barrier, each conv layer
creates materialization points. The backward has irreducible reduces for shape
broadcasting that can't be fused because their inputs are MM outputs.

## Path Forward

To close the dispatch gap:
1. **Memory planner** (Phase 2) — unlock BS=512 for 98%+ accuracy
2. **Fuse matmul via primitives** — implement MM as EXPAND+MUL+SUM (tinygrad-style)
   rather than MPS, so it fuses with upstream/downstream ew chains
3. **Global scheduling** — batch reduces that share inputs (SUM(gy, axes1) + SUM(gy, axes2))

## Files Modified

| File | Change |
|------|--------|
| `src/interact/_.c` | st_set in GRAD3 macro; no_grad_alloc save/restore; reduce deferral guard |
| `src/tinyhvm.h` | no_grad_alloc field in TinyHVM struct |
| `src/fuse/_.c` | Use no_grad_alloc for virtual intermediate buffer allocation + requires_grad |
| `src/rewrite/_.c` | Skip ew-only fusion during backward (no_grad_alloc check) |
