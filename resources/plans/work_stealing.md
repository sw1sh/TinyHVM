# Multi-Threaded Work-Stealing Plan

## The Problem

TinyHVM's reducer (`thvm_reduce`) is **single-threaded**. The IC net often contains
independent subgraphs that could reduce in parallel — e.g., the two args of a binary
TOP, independent gradient branches in backward, or separate layer computations. All
CPU work (IC reduction, fusion walking, kernel cache lookup) runs on one core while
the other cores are idle.

HVM4 achieves near-ideal speedup (12x on 16 cores) via per-thread heap banks and
Chase-Lev work-stealing deques. TinyHVM should adopt this architecture.

## Current State

**Thread-unsafe components** (everything touches shared `TinyHVM *ctx`):

| Component | Location | Issue |
|-----------|----------|-------|
| `heap_alloc()` | `src/heap/alloc.c` | Non-atomic `ctx->heap_pos += w` |
| `heap_read/set()` | `src/heap/read.c`, `set.c` | Raw array access, no barriers |
| `tensor_count` | `src/tinyhvm.h:449` | Non-atomic `ctx->tensor_count++` |
| `tensor_create()` | `src/ctx/tensor.c` | Bumps `tensor_count`, writes to `tensors[]` |
| Metal encoder | `src/backend/metal/batch.m` | Shared `batch_cmd`/`batch_encoder` |
| Buffer pool | `src/backend/metal/pool.m` | Shared `metal_pool.count` |
| `term_clone()` | `src/clone/_.c` | Calls `heap_alloc`, `heap_set`, `tensor_incref` |
| `reduce_pool` TLS | `src/reduce/_.c:46-47` | Already per-thread (ready) |

**Key constants:**
- `HEAP_CAP = 2M terms` (16MB) — `src/tinyhvm.h:324`
- `MAX_TENSORS = 524288` — `src/tinyhvm.h:342`
- `REDUCE_SLICE = 256`, `REDUCE_MAX_DEPTH = 512` — `src/reduce/_.c:44-45`

## Architecture

### Level 1: Per-Thread Heap Banks (HVM4 pattern)

Partition the heap into N banks (one per thread). Each thread allocates from its
own bank with zero contention — no atomics on the hot path.

```c
#define MAX_THREADS 16
#define HEAP_BANK_SIZE (HEAP_CAP / MAX_THREADS)  // 128K terms per bank

typedef struct {
    u64 next;   // next allocation position within bank
    u64 end;    // end of this bank's region
} HeapBank;

// Per-thread state
typedef struct {
    u32       tid;
    HeapBank  hbank;           // this thread's heap bank
    u32       tensor_next;     // this thread's tensor ID range start
    u32       tensor_end;      // this thread's tensor ID range end
    u64       itrs;            // interaction counter (local, summed at end)
} ThreadState;
```

**New `heap_alloc`:**
```c
static inline u64 heap_alloc(TinyHVM *ctx, u64 w) {
    ThreadState *ts = &ctx->threads[THREAD_ID];  // TLS thread ID
    u64 l = ts->hbank.next;
    ts->hbank.next += w;
    assert(ts->hbank.next <= ts->hbank.end);
    return l;
}
```

Wait-free. No atomics. Each bank is a contiguous slice of `ctx->heap`.

### Level 2: Per-Thread Tensor ID Ranges

Same pattern for tensor allocation. Pre-assign disjoint ID ranges:

```c
// Thread 0: tensor IDs [keep..keep+chunk)
// Thread 1: tensor IDs [keep+chunk..keep+2*chunk)
// ...

static inline u32 tensor_alloc_id(TinyHVM *ctx) {
    ThreadState *ts = &ctx->threads[THREAD_ID];
    assert(ts->tensor_next < ts->tensor_end);
    return ts->tensor_next++;
}
```

Tensor metadata writes go to `ctx->tensors[id]` which is safe because IDs are disjoint.

### Level 3: Chase-Lev Work-Stealing Deque

Each thread owns a deque of pending reduction work (heap locations to normalize).
Owner pushes/pops from the bottom. Thieves steal from the top via atomic CAS.

```c
#define WS_CAP (1 << 16)  // 64K slots per deque

typedef struct {
    _Alignas(128) _Atomic(u64) top;   // steal end
    _Alignas(128) _Atomic(u64) bot;   // owner end
    u64 buf[WS_CAP];
    u64 mask;                          // WS_CAP - 1
} WsDeque;
```

**Owner push** (wait-free):
```c
static inline void ws_push(WsDeque *q, u64 task) {
    u64 b = atomic_load_explicit(&q->bot, memory_order_relaxed);
    q->buf[b & q->mask] = task;
    atomic_thread_fence(memory_order_release);
    atomic_store_explicit(&q->bot, b + 1, memory_order_relaxed);
}
```

**Owner pop** (wait-free, handles race with thieves):
```c
static inline int ws_pop(WsDeque *q, u64 *out) {
    u64 b = atomic_load_explicit(&q->bot, memory_order_relaxed) - 1;
    atomic_store_explicit(&q->bot, b, memory_order_relaxed);
    atomic_thread_fence(memory_order_seq_cst);
    u64 t = atomic_load_explicit(&q->top, memory_order_relaxed);
    if (t <= b) {
        *out = q->buf[b & q->mask];
        if (t == b) {
            // Last element — race with thief
            u64 expected = t;
            if (!atomic_compare_exchange_strong_explicit(&q->top, &expected, t + 1,
                    memory_order_seq_cst, memory_order_relaxed)) {
                atomic_store_explicit(&q->bot, b + 1, memory_order_relaxed);
                return 0;  // thief got it
            }
            atomic_store_explicit(&q->bot, b + 1, memory_order_relaxed);
        }
        return 1;
    }
    atomic_store_explicit(&q->bot, b + 1, memory_order_relaxed);
    return 0;  // empty
}
```

**Thief steal** (lock-free CAS):
```c
static inline int ws_steal(WsDeque *q, u64 *out) {
    u64 t = atomic_load_explicit(&q->top, memory_order_acquire);
    atomic_thread_fence(memory_order_seq_cst);
    u64 b = atomic_load_explicit(&q->bot, memory_order_acquire);
    if (t >= b) return 0;
    *out = q->buf[t & q->mask];
    return atomic_compare_exchange_strong_explicit(&q->top, &t, t + 1,
        memory_order_acq_rel, memory_order_relaxed);
}
```

### Level 4: Parallel Reduction Strategy

**Where parallelism lives in TinyHVM:**

1. **Binary TOP args**: `TOP(uop, arg0, arg1)` — arg0 and arg1 are independent
   until both are realized. Reduce them in parallel.
2. **Backward branches**: `backward_local` computes d_a and d_b independently
   for binary ops. Each branch can reduce on a separate thread.
3. **Multi-layer forward**: In `thvm_sequential`, each layer depends on the
   previous, but within a layer (e.g., conv + bias + relu), some computation
   is independent.
4. **REF unfolding**: `term_clone` produces independent subgraphs.

**The core loop** (adapted from HVM4's `eval_normalize`):

```c
static void parallel_reduce_worker(ParallelCtx *pctx, u32 me) {
    WsDeque *myq = &pctx->deques[me];
    u32 n = pctx->n_threads;
    u32 prng = me * 0x9e3779b9;  // per-thread PRNG seed

    for (;;) {
        // Check global termination
        if (atomic_load_explicit(&pctx->pending, memory_order_acquire) == 0)
            break;

        u64 task;
        if (ws_pop(myq, &task)) {
            reduce_one(pctx, me, task);  // normalize one subgraph
            continue;
        }

        // Try stealing (random victim)
        int stolen = 0;
        u32 start = (me + 1 + (prng & 7)) % n;
        prng = prng * 1103515245 + 12345;
        for (u32 k = 0; k < n - 1; k++) {
            u32 vic = (start + k) % n;
            if (vic == me) continue;
            if (ws_steal(&pctx->deques[vic], &task)) {
                reduce_one(pctx, me, task);
                stolen = 1;
                break;
            }
        }
        if (!stolen) sched_yield();
    }
}
```

**Work discovery**: When `thvm_reduce` encounters a TAG_TOP with two unreduced
args, instead of pushing both onto the local stack (sequential), it:
1. Pushes arg0 onto its own deque (will reduce locally)
2. Pushes arg1 onto its own deque (stealable by another thread)

If arg1 is stolen, two threads reduce the two args in parallel. When both finish,
the thread that finishes second fires the binary op.

**Synchronization for join**: When a TOP's two args are both reduced, the TOP
can fire. Use an atomic counter per TOP node:

```c
// In the TOP node's heap, reserve a slot for the "args_ready" counter
// heap[loc+0] = arg0, heap[loc+1] = arg1, heap[loc+2] = atomic ready count
// When a thread finishes reducing an arg, it does:
u64 prev = atomic_fetch_add(&heap[loc+2], 1);
if (prev == 1) {
    // Both args ready — this thread fires the TOP interaction
    enqueue(myq, loc);  // re-enqueue for firing
}
```

This is the **fork-join** pattern at the granularity of individual TOP nodes.

### Level 5: GPU Dispatch Serialization

Metal command encoding is NOT thread-safe — all dispatch calls must happen on one
thread. Two options:

**Option A: Dedicated GPU thread.** Worker threads produce "dispatch requests"
into a lock-free queue. A dedicated GPU thread consumes them and encodes.

**Option B: Reducer-only parallelism.** Threads do IC reduction + fusion walking
in parallel but serialize at the GPU dispatch point. When a thread is ready to
dispatch a kernel, it acquires a lightweight spinlock, encodes, releases.

**Recommendation: Option B** — simpler, and the dispatch encoding itself is fast
(~1us per dispatch). The bottleneck is IC reduction + fusion walking, not encoding.

```c
static _Atomic(int) dispatch_lock = 0;

static void dispatch_1d_mt(...) {
    while (atomic_exchange(&dispatch_lock, 1)) cpu_relax();  // acquire
    dispatch_1d(pipe, bufs, n_bufs, params, param_sizes, n_params, numel);
    atomic_store(&dispatch_lock, 0);  // release
}
```

---

## Files to Create/Modify

### New: `src/parallel/wsdeque.c`
Chase-Lev work-stealing deque implementation.
- `ws_init()`, `ws_push()`, `ws_pop()`, `ws_steal()`, `ws_can_steal()`
- Cache-line aligned (`_Alignas(128)`)
- ~80 lines

### New: `src/parallel/workers.c`
Thread pool and parallel reduction coordinator.
- `ThreadState` struct, per-thread heap banks, tensor ID ranges
- `parallel_reduce(TinyHVM *ctx, Term root, u32 n_threads)` — entry point
- `parallel_reduce_worker()` — per-thread main loop
- Worker creation via `pthread_create`, join via `pthread_join`
- `pending` atomic counter for termination detection
- ~200 lines

### Modify: `src/heap/alloc.c`
Replace single `heap_pos` bump with per-thread bank allocation:
```c
// Old: u64 l = ctx->heap_pos; ctx->heap_pos += w;
// New: u64 l = ctx->threads[TID].hbank.next; ctx->threads[TID].hbank.next += w;
```
Keep the old path for single-threaded mode (n_threads == 1, no overhead).

### Modify: `src/tinyhvm.h`
- Add `ThreadState threads[MAX_THREADS]` to `TinyHVM` struct
- Add `u32 n_threads` field
- Add `_Thread_local u32 THREAD_ID` extern declaration
- Add `parallel_reduce()` API declaration
- Increase `HEAP_CAP` if needed (currently 2M terms = 16MB; may need 4M for multi-thread headroom)

### Modify: `src/reduce/_.c`
- In the enter phase for TAG_TOP: instead of always entering arg0 sequentially,
  check if n_threads > 1 and enqueue arg1 as stealable work
- Add atomic ready-counter check in the apply phase for TOP/TOP1 frames
- Keep single-threaded fast path untouched (zero overhead when n_threads == 1)

### Modify: `src/backend/metal/dispatch.m`
- Add `dispatch_lock` spinlock for multi-threaded dispatch serialization
- `dispatch_1d_mt()` and `dispatch_2d_mt()` wrappers that acquire lock before encoding

### Modify: `src/backend/metal/pool.m`
- `metal_buf_alloc()`: use atomic `pool.count++` (or per-thread buffer ID ranges)
- `metal_pool_reset()`: already called from single thread (between training steps)

### Modify: `src/ctx/tensor.c`
- `tensor_create()`: use per-thread tensor ID range instead of global `tensor_count++`
- `tensor_incref()`/`tensor_decref()`: use `atomic_fetch_add` for refcount

### Modify: `src/clone/_.c`
- Already thread-safe given per-thread heap banks (calls `heap_alloc`, `heap_set`)
- `tensor_incref` needs to be atomic (see above)

---

## Phased Implementation

### Phase 1: Infrastructure (~1 session)
- Implement `WsDeque` in `src/parallel/wsdeque.c`
- Add `ThreadState` to `TinyHVM` struct
- Per-thread heap banks with bank initialization in `thvm_init()`
- Per-thread tensor ID ranges
- Atomic refcounts on `TensorMeta`
- Keep everything single-threaded — just restructure allocation to be bank-based
- **Test**: `make test` passes, zero performance regression

### Phase 2: Parallel Reducer (~2 sessions)
- Implement `parallel_reduce_worker()` in `src/parallel/workers.c`
- Modify `thvm_reduce` enter phase: enqueue stealable arg1 for TAG_TOP when n_threads > 1
- Atomic ready-counter for TOP node join
- `pthread_create`/`pthread_join` thread pool
- `pending` counter for termination detection
- `dispatch_lock` spinlock for Metal encoding serialization
- **Test**: `make test` passes with `n_threads=1` and `n_threads=4`

### Phase 3: Backward Parallelism (~1 session)
- In `backward_local` (`src/grad/_.c:256+`): for binary ops, enqueue d_a and d_b
  reduction as separate stealable tasks
- In `grad_accum`: atomic CAS loop for gradient accumulation (two threads may
  accumulate to the same parameter)
- **Test**: MNIST CNN accuracy unchanged with 4 threads

### Phase 4: Tuning (~1 session)
- Profile with `THVM_PROFILE=1`: measure actual speedup vs single-threaded
- Tune `WS_CAP`, steal period, PRNG seed strategy
- Consider: minimum subgraph size threshold to avoid overhead for trivial reductions
  (don't spawn work for scalar ops)
- Benchmark: expect 2-4x speedup on CPU-bound portion of training step
- **Test**: wall time comparison, dispatch breakdown comparison

---

## Key Design Decisions

### Why Chase-Lev (not work-sharing, not task queues)

| Approach | Pros | Cons |
|----------|------|------|
| Work-sharing (centralized queue) | Simple | Single point of contention, cache thrashing |
| Task queue per thread (no stealing) | No contention | Load imbalance, idle threads |
| **Work-stealing (Chase-Lev)** | **No contention on hot path, load balances automatically** | **Slightly more complex steal logic** |

Chase-Lev is the standard for graph reduction (used by HVM4, Cilk, Go, Tokio).
Owner operations are wait-free. Stealing is rare and lock-free.

### Why NOT parallel GPU dispatch

Metal's `MTLComputeCommandEncoder` is not thread-safe. Options:
- One encoder per thread → multiple command buffers → extra GPU overhead
- Shared encoder with lock → simple, dispatch encoding is fast (~1us)

We chose shared encoder with spinlock. The encoding cost is negligible compared
to IC reduction. If profiling shows encoding contention, switch to per-thread
encoders with explicit GPU barriers.

### Single-threaded fast path

When `n_threads == 1`, the entire parallel infrastructure is bypassed:
- `heap_alloc` uses bank[0] (no extra indirection vs current code)
- No atomic operations anywhere
- No deque push/pop
- Zero overhead vs current single-threaded reducer

This is critical — we must not slow down the default case.

### Fork-join at TOP granularity

The natural parallelism unit is a TOP node's two args. This is fine-grained
(each TOP is a potential fork point) but cheap (just a deque push + atomic counter).
Coarser-grained parallelism (e.g., per-layer) would miss intra-layer parallelism
and require more complex scheduling.

---

## Verification

1. **Correctness**: `make test` — MNIST CNN >90% accuracy with 1, 2, 4, 8 threads
2. **Determinism**: Same random seed produces same loss trajectory regardless of thread count
   (may need deterministic reduction order for floating-point reproducibility)
3. **Race detection**: Run with ThreadSanitizer (`-fsanitize=thread`) — zero data races
4. **Performance**: Measure wall time for 70-step MNIST training:
   - 1 thread: baseline (should match current single-threaded performance exactly)
   - 4 threads: expect 1.5-2.5x speedup (limited by GPU dispatch serialization)
   - 8 threads: expect 2-4x speedup (diminishing returns as GPU becomes bottleneck)
5. **Overhead**: Profile single-threaded path to confirm zero regression from bank-based allocation
6. **Stress test**: Run with `REDUCE_MAX_DEPTH` cases to verify TLS stack pool works correctly across threads

---

## Dependencies & Prerequisites

- **No dependency on lazy graph compiler** (`lazy_graph_compiler.md`): work-stealing
  parallelizes the existing eager reducer. The graph compiler can later produce
  batched work items that feed into the parallel reducer.
- **No dependency on JIT** (`jit_ranges_patterns.md`): JIT replays skip IC reduction
  entirely. Work-stealing helps the non-JIT path (step 0 capture, eval, etc.).
- **Requires**: `<stdatomic.h>` (C11), `<pthread.h>` (POSIX). Both available on macOS.
