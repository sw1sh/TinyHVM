# Implementation Plan: JIT Capture/Replay, Range Fusion, Pattern Matching

## Context

TinyHVM's CNN training loop (`test/test_mnist.m:283-348`) repeats the same computation graph 70 times per run. Each step re-traverses the IC net, re-walks fusion trees, re-looks up cached kernels, and re-encodes ~300+ GPU dispatches. The computation structure is identical across steps — only buffer data and batch offsets change. This is the exact scenario JIT capture/replay is designed for.

Additionally, the current fusion system (`src/grad/_.c:94-232`, `fuse_or_reduce`) handles elementwise+reduce chains but uses ad-hoc pattern detection. Range-based fusion and declarative pattern matching would make this more principled and extensible.

**Ordering: JIT → Ranges → Pattern Matcher** (by immediate performance impact)

---

## Phase 1: JIT Capture/Replay

### Goal
Record the GPU dispatch sequence from step 0, replay it for steps 1-69. Skip all IC reduction, fusion walking, kernel cache lookup, and tensor metadata management on replay steps.

### Architecture

**Recording level: Metal dispatch commands.** We record at `dispatch_1d`/`dispatch_2d` (`src/backend/metal/dispatch.m:12-50`), capturing the pipeline state, buffer list, raw param bytes, and grid dimensions. This is the lowest level before GPU encoding, giving maximum replay speed.

**Buffer slot mapping:** Each Metal buffer referenced during capture gets assigned a slot index. Buffers are categorized:
- **Persistent** (slot 0..N-1): weights, adam m/v — exist before capture, same buffer IDs on replay
- **Ephemeral** (slot N+): intermediates, gradients — allocated during capture, freshly allocated on replay with same sizes
- **Input** (marked): batch data view — needs param patching on replay (shrink offset changes per step)

### Files to modify

#### New file: `src/backend/metal/jit.m`
```c
#define JIT_MAX_CMDS    4096
#define JIT_MAX_SLOTS   1024

typedef struct {
    id<MTLComputePipelineState> pipe;
    u32    slot_indices[16];   // buffer slots
    u32    n_bufs;
    u8     params[512];        // raw param bytes (ViewParams etc)
    u32    param_offsets[8];   // byte offset of each param in params[]
    u32    param_sizes[8];
    u32    n_params;
    u32    grid[3];
    u32    tg[3];
    u8     is_2d;
} JITCmd;

typedef struct {
    u64    alloc_size;         // bytes allocated for this slot
    u32    buf_id;             // resolved buffer ID (set during capture or replay alloc)
    u8     persistent;         // 1 if existed before capture
} JITSlot;

typedef struct {
    JITCmd  cmds[JIT_MAX_CMDS];
    u32     n_cmds;
    JITSlot slots[JIT_MAX_SLOTS];
    u32     n_slots;
    u32     persistent_count;  // slots 0..persistent_count-1 are persistent
    enum { JIT_OFF, JIT_CAPTURE, JIT_REPLAY } state;
} JITState;
```

Key functions:
- `jit_begin_capture(u32 n_persistent)` — snapshot existing buffer IDs as persistent slots
- `jit_record_dispatch(pipe, bufs, n_bufs, params, sizes, n_params, grid, tg)` — called from dispatch_1d/dispatch_2d during capture
- `jit_end_capture()` — finalize recording
- `jit_replay()` — allocate ephemeral buffers, encode all cmds into one command buffer, commit
- `jit_slot_for_buf(u32 buf_id)` — maps raw buffer ID to slot during capture; allocates new slot on first use
- `jit_buf_alloc_hook(u32 buf_id, u64 bytes)` — called from `metal_buf_alloc` during capture to track slot sizes

#### Modify: `src/backend/metal/dispatch.m` (lines 12-50)
Add JIT recording hooks to `dispatch_1d` and `dispatch_2d`:
```c
static void dispatch_1d(...) {
    // existing encoder setup...
    if (jit_state.state == JIT_CAPTURE) {
        jit_record_dispatch(pipe, bufs, n_bufs, params, param_sizes, n_params,
                           (u32[]){numel,1,1}, (u32[]){tpg,1,1});
    }
    // existing dispatch...
}
```

#### Modify: `src/backend/metal/init.m`
Add `jit_begin_capture`/`jit_replay` to Backend vtable or as direct Metal functions. Add buf_alloc hook for JIT slot tracking.

#### Modify: `src/tinyhvm.h` (lines 445-457)
Add JIT API functions:
```c
void thvm_jit_begin(TinyHVM *ctx);    // start capture for next step
int  thvm_jit_ready(TinyHVM *ctx);    // 1 if captured and ready to replay
void thvm_jit_replay(TinyHVM *ctx, Term batch_input);  // replay with new input
void thvm_jit_free(TinyHVM *ctx);     // free JIT state
```

#### Modify: `test/test_mnist.m` (lines 283-348)
Change training loop to use JIT:
```c
for (u32 step = 0; step < n_steps; step++) {
    if (step == 0) {
        thvm_jit_begin(ctx);
        // ... normal step (capture) ...
    } else if (thvm_jit_ready(ctx)) {
        // Replay: only update batch input + read loss
        thvm_jit_replay(ctx, /* batch offset, labels */);
    }
}
```

### Key challenges
1. **Batch offset patching**: The shrink view creates different ViewParams each step. Solution: mark the batch input slot during capture, patch its ViewParams offset before replay.
2. **Loss readback**: Need to identify which buffer slot holds the loss scalar so we can read it on replay.
3. **Adam state**: Adam updates weights in-place — the persistent buffers' data changes but their IDs don't. This works naturally with slot mapping.
4. **`thvm_reset` on replay**: Skip tensor metadata reset on replay; only reset the ephemeral buffer pool.

### Estimated dispatch savings
- Step 0: ~300 dispatches (normal, with capture overhead ~5%)
- Steps 1-69: ~300 dispatches each but **zero IC reduction, zero fusion walk, zero cache lookup**
- Total CPU savings: ~69 * (IC reduction + fusion + cache) per step

---

## Phase 2: Range-Based Fusion

### Goal
Replace the ad-hoc pattern matching in `fuse_or_reduce` with range annotations that determine fusion eligibility by iteration domain identity.

### Architecture

Currently `fuse_or_reduce` (`src/grad/_.c:94-232`) checks for specific patterns:
1. `elementwise_chain` → fuse
2. `SUM(elementwise_chain)` → fuse with reduce
3. `RESHAPE(SUM(elementwise_chain))` → fuse with reduce + reshape output

This is brittle. Range-based fusion assigns each op a **range signature** (the loop bounds it iterates over). Ops with the same range signature can be fused into one kernel.

### Design

```c
typedef struct {
    u32 dims[MAX_DIM];  // iteration dimensions
    u32 ndim;
} RangeKey;
```

- Elementwise ops: range = output shape
- Reduce ops: range = input shape (iterate input, accumulate to output)
- Matmul: range = (M, N, K) — custom, not fusable with elementwise

Two ops are fusable when their RangeKeys match OR one is a reduce whose output range matches the other's range.

### Files to modify

#### Modify: `src/grad/_.c`
- Add `range_key_for_op(TinyHVM *ctx, Term t)` — compute range from shape/strides
- Replace `fuse_or_reduce` pattern checks with range key comparison
- Keep existing `fuse_walk_inner` for collecting the op chain; just change the eligibility check

#### Modify: `src/backend/metal/fused.m`
- Extend `codegen_fused_v2` to handle range-annotated ops (currently assumes all ops share the output numel iteration domain)

### Dependency
Independent of JIT. Can be developed in parallel.

---

## Phase 3: Declarative Pattern Matcher

### Goal
Replace the hardcoded `switch` in `thvm_interact` (`src/interact/_.c:13+`) with a table-driven rewrite system. Rules are declared as `(pattern, replacement)` pairs and applied bottom-up.

### Architecture

Inspired by tinygrad's PatternMatcher but adapted for C:

```c
typedef struct {
    u32   match_uop;                    // UOP to match (or UOP_ANY)
    u32   match_arg_uops[2];            // optional: constrain arg UOPs
    Term (*rewrite)(TinyHVM *ctx, u64 loc, Term a, Term b);  // NULL = use default
    u32   early_reject;                 // bitmask for fast rejection
} RewriteRule;

static RewriteRule rules[] = {
    // Algebraic simplifications
    { UOP_ADD, {UOP_ANY, UOP_ANY}, rewrite_add, 0 },
    { UOP_MUL, {UOP_ANY, UOP_ANY}, rewrite_mul, 0 },
    // Constant folding
    { UOP_ADD, {/*scalar*/}, fold_add_const, 0 },
    // ...
};
```

The dispatch loop becomes:
```c
for (int i = 0; i < n_rules; i++) {
    if (rules[i].match_uop != uop) continue;
    if (rules[i].early_reject & ...) continue;
    Term result = rules[i].rewrite(ctx, loc, a, b);
    if (result != t) return result;
}
```

### Files to modify

#### New file: `src/rewrite/_.c`
- Rule table definition
- `rewrite_apply(TinyHVM *ctx, Term t)` — try all matching rules
- Individual rewrite functions extracted from current `thvm_interact` switch cases

#### Modify: `src/interact/_.c`
- Replace UOP dispatch switch with call to `rewrite_apply`
- Keep non-UOP cases (APP, DP0, DP1, GRAD) in the switch

### Benefits
- Adding new optimization rules = adding a table entry, not modifying a switch
- Rules are composable and order-independent (fixpoint iteration)
- Enables algebraic simplification passes (x*0→0, x+0→x, x*1→x)

### Dependency
Independent of JIT and ranges. Lowest priority — code quality improvement, not performance.

---

## Verification

### Phase 1 (JIT)
1. Run `make test` — MNIST CNN must still achieve >90% accuracy
2. Add JIT to training loop, verify step 0 capture produces same dispatch count
3. Verify steps 1-69 replay without IC reduction (check `ctx->itrs` doesn't increase)
4. Compare wall time: expect 20-40% speedup on 70-step training (CPU-bound portion eliminated)
5. Verify loss values match non-JIT run within floating point tolerance

### Phase 2 (Ranges)
1. Run `make test` — accuracy unchanged
2. Verify fusion count (`fuse_fused_count`) is >= current count
3. Check dispatch breakdown — expect fewer total dispatches if more ops are fused

### Phase 3 (Pattern Matcher)
1. Run `make test` — accuracy unchanged
2. Verify all existing UOP interactions produce identical results
3. Add 2-3 algebraic simplification rules, verify they fire on synthetic tests

---

## Implementation Order

```
Phase 1 (JIT):     ~3-4 sessions
  1a. JITState struct + slot mapping in jit.m
  1b. Recording hooks in dispatch.m
  1c. Replay logic (buffer alloc + re-encoding)
  1d. API (thvm_jit_begin/replay/free)
  1e. Integrate into test_mnist.m training loop
  1f. Batch offset patching + loss readback

Phase 2 (Ranges):  ~2 sessions
  2a. RangeKey computation
  2b. Replace fuse_or_reduce eligibility checks
  2c. Update codegen for range-annotated fusion

Phase 3 (Pattern Matcher): ~2 sessions
  3a. Rule table + rewrite_apply
  3b. Extract existing switch cases into rule functions
  3c. Add algebraic simplification rules
```
