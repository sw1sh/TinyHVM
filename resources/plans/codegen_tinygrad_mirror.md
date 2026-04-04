# Plan: Mirror Tinygrad Scheduling/Codegen Architecture

## Root Cause Analysis (verified by tracing)

### Why 368 Dispatches?

| Source | Count | % |
|--------|-------|---|
| tensor_materialize_chain | 321 | 91% |
| fuse_or_reduce (rewrite rules) | 33 | 9% |
| adam | 14 | 4% |

**91% of dispatches come from the materializer**, which processes deferred tensors one-at-a-time.

### The Sequential ENSURE Problem

The IC reducer processes backward ops sequentially. When GRAD creates:
1. `MUL(gy, x)` → defers (buf_id=0)
2. `PERMUTE(MUL, axes)` → defers (view propagation)
3. `SUM(PERMUTE, reduce_axes)` → needs to dispatch

At step 3, the interact handler calls `ENSURE(ctx, permute_tid)` which calls
`tensor_materialize_chain(ctx, permute_tid)`. This materializes the MUL ew chain
as a standalone fused_ew dispatch. THEN the SUM dispatches separately.

By the time the rewrite rules try to fuse `SUM(MUL_chain)`, the MUL chain is
already materialized (buf_id != 0). The rewrite filter at `rule_sum_fuse` line 154
(`if (_am->buf_id != 0) return t`) rejects it.

**Tinygrad doesn't have this problem** because everything is lazy until `realize()`.

### What Currently Works

- Forward: 33 fuse_or_reduce dispatches (rewrite rules see lazy TAG_TOP children)
- PERMUTE-of-ew: 21 per step fuse into reduce via interact handler line 619
- Post-reduce fusion: ew→reduce→ew chains in tensor_materialize_chain

### What Fails

- Most backward reduces (144/step): ew inputs already materialized by earlier ENSURE
- SUM(PERMUTE(RESHAPE(ew))): PERMUTE base is RESHAPE (view), not ew → path fails

---

## The Fix: Two-Pass Backward

Instead of materializing ew chains one-at-a-time during backward, collect the entire backward graph lazily, then dispatch optimally.

### Phase 1: Prevent Premature Materialization

**File**: `src/interact/tensor_ops.c`

When a backward reduce hits the ENSURE path (line 784), check if the input is a deferred
ew/view chain. If so, **don't ENSURE it**. Instead, create a deferred reduce tensor:

```c
// NEW: During backward, defer reduces instead of ENSURE-materializing their ew inputs
if (is_reduce && ctx->no_grad_alloc && ma->buf_id == 0 && ma->creator_op &&
    (is_elementwise(ma->creator_op) || is_view_op(ma->creator_op))) {
    // Create deferred reduce tensor (same as forward deferral path)
    md->buf_id = 0;
    md->creator_op = uop;
    md->src_ids[0] = a_id;
    md->src_ids[1] = b_id;
    ctx->itrs++;
    RETURN_REDUCED(term_ten(dst_id, ma->dtype));
}
```

This prevents `tensor_materialize_chain` from being called for the ew input.
The reduce stays deferred (buf_id=0) like forward reduces.

**Safety**: The previous attempt at deferring backward reduces failed because
"deferred reduces become TAG_TEN with buf_id=0, but the reducer treats TAG_TEN
as WNF and never materializes them." But the fix is: when the FINAL output
(loss + grads) is reduced by `thvm_reduce`, the ENSURE at that point will
trigger `tensor_materialize` on the entire deferred graph.

### Phase 2: Graph-Level Materialization

**File**: `src/fuse/materialize.c`

When `tensor_materialize` is called for a deferred reduce (from the final ENSURE),
it should recognize the deferred reduce and dispatch reduce+ew as one kernel:

```c
// In tensor_materialize_chain:
if ((m->creator_op == UOP_SUM || m->creator_op == UOP_RMAX) && m->src_ids[0]) {
    // Walk through view ops to find ew base
    u32 _rw = m->src_ids[0];
    for (u32 d = 0; d < 5; d++) {
        TensorMeta *vt = &ctx->tensors[_rw];
        if (vt->buf_id != 0) break;
        if (is_elementwise(vt->creator_op)) { walk_tid = _rw; break; }
        if (is_view_op(vt->creator_op)) _rw = vt->src_ids[0];
        else break;
    }
    reduce_type = m->creator_op;
    reduce_axes_id = m->src_ids[1];
}
```

### Phase 3: SHRINK/PAD View Composition

**Files**: `src/fuse/_.c`, `src/fuse/materialize.c`

Implement SHRINK and PAD handling in fuse_walk_inner and materialize_walk.
This fuses conv im2col chains into their consumers.

### Phase 4: Multi-Reduce Codegen

**File**: `src/backend/metal/codegen.m`

Already partially implemented (reduce2 support). Extend for N-phase reduces
to handle patterns like BN variance: `SUM((x - SUM(x)/n)^2)`.

### Phase 5: Metal ICB Replay

**File**: `src/backend/metal/jit.m`

With fewer dispatches, encode all into one Metal ICB per step.

---

## Expected Progression

| After | Dispatches | Speed |
|-------|-----------|-------|
| Current | 368 | 500ms |
| Phase 1+2 (backward reduce deferral) | ~80 | ~200ms |
| Phase 3 (SHRINK/PAD) | ~40 | ~150ms |
| Phase 4 (multi-reduce) | ~15 | ~100ms |
| Phase 5 (ICB) | ~15 | <50ms |
| Tinygrad | ~10 | 79ms |

## Critical Insight

The problem is NOT in the codegen or the fusion walker. Both work correctly.
The problem is **timing**: backward ew chains get materialized too early (by ENSURE)
before reduces can absorb them. The fix is to defer backward reduces so the ew
chains stay deferred until the global materialization pass.
