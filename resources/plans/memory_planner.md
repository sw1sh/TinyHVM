# Memory Planner — Status and Findings

## Blocked: Needs JIT First

The memory planner requires **deterministic execution order** across steps.
TinyHVM's lazy reduction means buffer allocation order on step 0 doesn't match
step 1+ — the mid-step steal changes effective buffer sizes, and deferred chains
delay buffer accesses beyond what any static analysis can predict.

### What Was Tried (All Failed)

1. **Dispatch log + suballocation**: Suballocation mechanism works (stable total,
   no OOM). But lifetime analysis corrupts training because:
   - Dispatch log misses CPU reads (buf_read, buf_read_nosync, META_READ)
   - Even with CPU reads logged, still corrupts (dispatch log misses contiguify blits, adam_step, etc.)

2. **Tensor graph + suballocation**: Compute [birth, death] from tensor src_ids.
   - Covers all dependencies including conv_input_id, conv_weight_id
   - Transitive closure through deferred chains (buf_id=0 → walk src_ids recursively)
   - Still corrupts because tensor creation order ≠ execution order (lazy eval)
   - peak=1838MB (correct) but regions overlap due to order mismatch

3. **Direct buffer reuse (no suballocation)**: Failed because
   - buf_decref frees buffers before pool_reset can save them
   - metal_bytes_inuse double-counts reused buffers within a step

### Root Cause

TinyHVM's execution model is **lazy reduction with deferred dispatch**. Buffers
are allocated eagerly but data is written lazily (at ENSURE time). Two buffers
that don't overlap in tensor creation order CAN overlap in execution order.

Tinygrad doesn't have this problem because its schedule is computed BEFORE
execution. All lifetimes are known at planning time. The planner runs on the
schedule, not on execution traces.

### The Path Forward

**Prerequisite: JIT replay.** Once JIT captures the deterministic dispatch
sequence from step 0, the memory planner can use the JIT's recorded alloc/free
sequence (which IS execution order) for lifetime analysis.

With JIT:
1. Step 0: JIT records every dispatch + every alloc in execution order
2. Between steps: planner computes lifetimes from JIT's recorded alloc sequence
3. Step 1+: JIT replays dispatches, planner provides suballocated buffers

This is exactly how tinygrad works — the schedule IS the execution plan.

### What Works Without Planner

BS=128: 2.1GB, trains fine
BS=256: ~4GB, trains fine (95.1%)
BS=512: ~8GB, trains fine (95.9%) — but slow (3133ms/step)

The free-list mechanism already provides cross-step buffer reuse. The missing
piece is within-step reuse, which requires the planner.
