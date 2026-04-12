# Kernel Result Caching

## Current Model

`kid_results[kid]`: global array mapping kernel ID → dispatched TAG_TEN result.
Initialized to ERA (not yet dispatched). Once a kernel fires, its result is cached
and reused for all subsequent references to the same kid.

```c
if (term_tag(kid_results[kid]) != TAG_ERA)
    RETURN_REDUCED(kid_results[kid]);  // cache hit — return previous result
```

This works when:
- Each kernel fires exactly once per eval
- Input buffers are immutable after scheduling
- Same kid → same computation → same result

## Problem: ASSIGN Invalidation

In a training loop with ASSIGN-based parameter updates:

```
Iteration 1:  K1(w) → result1,  ASSIGN(w, w - lr*grad)
Iteration 2:  K1(w) → result2   ← should read UPDATED w, not cached result1
```

After ASSIGN writes to buffer `w`, K1's cached result is stale. K1 was scheduled
reading the original `w`. The cache returns `result1` (computed from original `w`)
instead of re-dispatching with the updated buffer.

## Requirements

1. After ASSIGN fires on buffer B, invalidate all cached kernels that read from B
2. Invalidation must be cheap (not O(all_kernels) per ASSIGN)
3. Support re-dispatch of invalidated kernels (same kernel spec, fresh inputs)
4. Must work across loop iterations driven by IC reduction

## Design Options

### A. Per-buffer generation counter

Each buffer gets a generation number. Incremented on ASSIGN write.
Each kernel result records the generation of each input buffer at dispatch time.
Cache lookup: hit only if all input buffer generations still match.

```
buf_generation[buf_id]++   on ASSIGN
kid_input_gens[kid][i]     recorded at dispatch time
cache_valid = all(buf_generation[kid_inputs[i]] == kid_input_gens[kid][i])
```

**Pros**: O(n_inputs) per cache check, O(1) per ASSIGN
**Cons**: per-kernel storage for input generations, global buf_generation array

### B. Invalidation list per buffer

Each buffer maintains a list of kid indices that read from it.
On ASSIGN, walk the list and set `kid_results[kid] = ERA` (invalidate).

```
buf_readers[buf_id] = [kid1, kid2, ...]   built at schedule time
ASSIGN(buf_id): for kid in buf_readers[buf_id]: kid_results[kid] = ERA
```

**Pros**: precise invalidation, O(readers) per ASSIGN
**Cons**: requires tracking readers at schedule time, list management

### C. Epoch-based invalidation

Global dispatch epoch counter. Incremented after each ASSIGN.
Kernel cache entries tagged with epoch. Hit only if epoch matches.

```
dispatch_epoch = 0
kid_epoch[kid]  = epoch at dispatch time
kid_results[kid] valid only if kid_epoch[kid] == dispatch_epoch
ASSIGN: dispatch_epoch++
```

**Pros**: simplest, O(1) per ASSIGN, O(1) per cache check
**Cons**: invalidates ALL cached kernels on any ASSIGN — too aggressive for
graphs with multiple independent ASSIGN targets

### D. Per-ASSIGN-target epoch

Like C, but scoped to each ASSIGN target buffer.

```
buf_epoch[buf_id] = 0
kid_buf_epochs[kid][i] = buf_epoch[input_buf_i] at dispatch time
cache_valid = all(buf_epoch[kid_inputs[i]] == kid_buf_epochs[kid][i])
ASSIGN(buf_id): buf_epoch[buf_id]++
```

Equivalent to option A but with "epoch" framing.

## Recommendation: Option D (per-buffer epoch)

Same as A, just named differently. Simple, efficient, precise:

- O(1) ASSIGN cost (increment one counter)
- O(n_inputs) cache check (compare epochs)
- No invalidation lists to manage
- Works for multiple independent ASSIGN targets

Implementation:
1. Add `u32 buf_epoch[MAX_BUFS]` global array (init to 0)
2. In ASSIGN handler: `buf_epoch[dst_buf_id]++`
3. In kernel dispatch: record `kid_buf_epochs[kid][i] = buf_epoch[input_buf_i]`
4. In kernel cache check: compare all input epochs before returning cached result

## IC Integration

The epoch check is inside UOP_KERNEL's interaction handler. When the handler
detects a stale cache (epoch mismatch), it re-dispatches the kernel.
This is transparent to the IC reduction — the kernel just takes longer to
return its result.

ASSIGN's epoch increment is a side effect of its interaction. After ASSIGN
fires and updates the buffer, it bumps the epoch. Downstream SEQ ensures
kernels reading that buffer are checked after the bump.

## Relation to SEQ

SEQ(ASSIGN, KERNEL) ensures ASSIGN fires first. After ASSIGN:
- Buffer is updated
- `buf_epoch[buf_id]` is incremented
Then KERNEL fires:
- Cache check: `kid_buf_epochs[kid][i] != buf_epoch[input_buf_i]` → stale
- Re-dispatch with updated buffer
- Cache new result with current epochs

Without SEQ, the kernel might fire before ASSIGN (confluence). With SEQ,
the ordering is guaranteed.

## Files

- `src/interact/tensor_ops.c` — UOP_KERNEL cache check, ASSIGN handler
- `src/schedule/_.c` — kid_results[], kernel dispatch
- `src/tinyhvm.h` — buf_epoch array, KernelEntry input buf tracking
