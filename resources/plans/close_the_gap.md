# Plan: Close the TinyHVM/Tinygrad Gap

## Global Picture

**Goal**: Match tinygrad's beautiful_mnist: 98.3% accuracy, ~79ms/step, ~20 dispatches, 0.82GB.

**Current** (2026-04-03): 94.4% accuracy, 505ms/step, 374 dispatches, 2.1GB (BS=128).
BS=512 works at 95.9%/3133ms but needs memory planner for 98%+ at reasonable speed.

### Architecture Gap

**Tinygrad**: Lazy graph → schedule → memory plan → JIT. Everything determined before execution.
**TinyHVM**: Eager IC reduction + deferred dispatch. Execution order is non-deterministic
(mid-step steal, lazy ENSURE). This blocks both memory planning and JIT.

---

## Phase 1: JIT Replay (prerequisite for everything)

**Priority: HIGHEST. Unblocks memory planner AND speed.**

### Why JIT First

The memory planner CANNOT work without deterministic execution order:
- 5 attempts at suballocation all corrupted training data
- Root cause: buffer alloc/access order differs between steps (mid-step steal,
  lazy ENSURE, deferred chains delay access beyond static predictions)
- Tinygrad solves this by computing the schedule BEFORE execution
- TinyHVM equivalent: JIT captures step 0's execution, replays it deterministically

With JIT:
1. Step 0: JIT records every dispatch + alloc in actual execution order
2. JIT replay on step 1+: deterministic dispatch sequence = deterministic alloc sequence
3. Memory planner runs on the JIT's recorded sequence (exact lifetimes)
4. Speed: JIT replay skips IC reduction overhead → <100ms/step

### Current JIT State

Already partially implemented in `src/backend/metal/jit.m`:
- JIT_CAPTURE / JIT_REPLAY states
- `jit_record_dispatch_ids` records compute dispatches
- `jit_record_mps` records MPS matmul calls
- `jit.consts[]` saves CPU-written constants
- Missing: buffer allocation replay, memory planner integration

### Implementation

1. **Capture phase** (step 0): Record dispatch sequence + alloc sequence + buf writes
2. **Replay phase** (step 1+): Replay dispatches in recorded order, using recorded buffers
3. **Buffer mapping**: Map step 0's buf_ids to step 1+'s buf_ids (same alloc sequence)
4. **Memory planner**: After capture, compute lifetimes from the recorded sequence,
   assign offsets, allocate one big buffer. Replay uses suballocated offsets.

### Verification
- Step 1 accuracy = step 0 accuracy (deterministic)
- Dispatch count identical across steps
- Speed < 100ms/step at BS=128
- Memory planner reduces BS=512 to ~1GB

---

## Phase 2: Memory Planner (on top of JIT)

With JIT providing deterministic execution, the planner is straightforward:
- JIT records: alloc(buf_id, size) at position P, last_use(buf_id) at position Q
- Planner: for each alloc, find dead alloc with compatible size → assign same offset
- One big MTLBuffer with suballocated offsets via buf_offset[]
- All the infrastructure (buf_offset, BUF_OFFSET, BUF_CONTENTS) already exists

Expected: 2.1GB → ~0.8GB at BS=128. BS=512 fits comfortably.

---

## Phase 3: Eliminate Standalone EW Dispatches (200 → ~0)

The 200 fused_ew dispatches are backward ew chains materialized at ENSURE boundaries.
In tinygrad, these are absorbed into reduce kernels.

### Fix: Lazy ENSURE During Backward

ENSURE should NOT materialize deferred ew chains during backward. Instead, chains
stay deferred until a reduce (SUM/RMAX) absorbs them.

With primitive matmul (no MPS), backward has no ops that REQUIRE materialized buffers.
All backward ops (MUL, ADD, SUM, EXPAND, RESHAPE) work with deferred chains.

Implementation: `ENSURE` during backward only materializes if the tensor has no
creator_op (it's a leaf = weight/input). Deferred chains pass through.

Expected: 374 → ~171 dispatches (157 reduces + ~0 fused_ew + 14 adam)

---

## Phase 4: Reduce Fusion (~171 → ~50)

### Merge Multi-Input Reduces

Shared tensors (grad_refs > 1) create ADD + SUM pairs.
Fuse: `SUM(ADD(da, db))` instead of `SUM(da) + SUM(db)`.

### Forward SHRINK/PAD in Walker

fuse_walk_inner stops at SHRINK/PAD (line 106: "not implemented yet").
Implementing these view compositions fuses more forward ops.

Expected: ~50 dispatches total

---

## What's Already Done

- [x] Deferred ew ops (forward + backward)
- [x] SUM/RMAX fusion with deferred ew chains (102 fused reduces)
- [x] GRAD3 shape tracking (st_set)
- [x] no_grad_alloc flag (separate buffer allocation from rewrite control)
- [x] Primitive matmul (EXPAND+MUL+SUM, no MPS dependency)
- [x] All gradient bugs fixed (CPU=Metal for all 14 params)
- [x] Budget guard fix (metal_bytes_total, catches OOM before system kill)
- [x] BS=512 verified working (95.9% at 3133ms/step)

## What Failed (Memory Planner Without JIT)

- 5 suballocation attempts: all corrupt training data
- Root cause: non-deterministic execution order between steps
- dispatch-log lifetimes miss CPU reads, contiguify blits, adam_step
- tensor-graph lifetimes miss lazy execution ordering
- Direct buffer reuse: ARC + buf_decref lifecycle issues

## Expected Progression

| After | Accuracy | Dispatches | Speed | Memory |
|-------|----------|------------|-------|--------|
| Current | 94.4% (BS=128) | 374 | 505ms | 2.1GB |
| Phase 1 (JIT) | 94.4% | 374 | **<100ms** | 2.1GB |
| Phase 2 (Planner) | **98%+ (BS=512)** | 374 | <100ms | **~0.8GB** |
| Phase 3 (Lazy ENSURE) | 98%+ | **~171** | <50ms | ~0.8GB |
| Phase 4 (Reduce fusion) | 98%+ | **~50** | <30ms | ~0.8GB |
