# Multi-Threaded Work-Stealing Plan

## Current Benchmarks (2026-03-28)

### Single-threaded IC reduction rate

| Path | Rate | Latency | Notes |
|------|------|---------|-------|
| TAG_NUM (inline scalar) | **137 Mitrs/s** | 7.3 ns/itr | No tensors, no buffers |
| TAG_TEN + thvm_op_raw | 22–36 Mitrs/s | 27–45 ns/itr | Tensor create + CPU dispatch |
| TAG_TEN + thvm_op | 9–18 Mitrs/s | 56–113 ns/itr | Full linear_use + shape tracking |
| HVM4 reference | ~200 Mitrs/s | ~5 ns/itr | Pure lambda calculus |

### Parallel scaling (independent contexts, TAG_NUM ADD 50K chain)

| Threads | Total Mitrs/s | Speedup | Per-thread |
|---------|---------------|---------|------------|
| 1 | 57 | 1.0× | 57 |
| 2 | 108 | 1.9× | 54 |
| 4 | 230 | 4.0× | 57 |
| 6 | 393 | 6.9× | 66 |
| 8 | 460 | 8.1× | 58 |
| 12 | 551 | 9.7× | 46 |

Near-linear scaling up to 6 threads. Diminishes at 8+ due to M3 Max efficiency
core mix and TLS memory pressure (4MB reduce_pool per thread).

### Parallel scaling (shared context, TAG_NUM ADD balanced binary tree)

All threads reduce subtrees of the SAME TinyHVM context (shared heap).
Subtasks sized to ~4K nodes (64KB) for L1 cache friendliness.

| Tree depth | Heap (MB) | Threads | Mitrs/s | Speedup | Notes |
|------------|-----------|---------|---------|---------|-------|
| 18 | 4 | 1 | 69 | 1.0× | DFS tree traversal slower than chain |
| 18 | 4 | 4 | 174 | 2.5× | |
| 18 | 4 | 6 | **218** | 3.2× | Peak for 4MB working set |
| 20 | 16 | 1 | 86 | 1.0× | |
| 20 | 16 | 4 | **215** | 2.5× | Peak for 16MB |
| 22 | 64 | 1 | 87 | 1.0× | |
| 22 | 64 | 3 | **201** | 2.3× | Peak for 64MB |
| 22 | 64 | 8 | 92 | 1.1× | L2 contention kills scaling |

**Key findings:**
- Peak throughput: ~220 Mitrs/s (3–6 threads), regardless of working set size.
- Cache-limited: per-thread throughput degrades when total working set > L2 (48MB).
- 8+ threads: E-cores + shared L2 contention cause slowdown, not speedup.
- Tree DFS traversal is ~40% slower than chain per interaction (cache access pattern).
- Independent-context (551 Mitrs/s) >> shared-context (220 Mitrs/s) due to zero
  cache interference in independent mode.

### CNN training (Metal GPU)

| Architecture | Dispatches | Time/step |
|---|---|---|
| 1-conv CNN | 25 | 4ms |
| 4-conv CNN | 78 | 19ms |

---

## What Was Done (Phase 1)

### 1. TAG_NUM inline scalar compute (`src/interact/_.c`)

Added a fast path in the interact handler: when inputs are TAG_NUM (32-bit
float embedded in the Term's VAL bits), compute the result inline and return
TAG_NUM. No tensor creation, no buffer allocation, no backend dispatch.

```
TAG_NUM: 137 Mitrs/s (7.3 ns/itr) — within 1.5× of HVM4
TAG_TEN: 22 Mitrs/s (45 ns/itr)  — 6× slower
```

**Key change**: also fixed the trampoline (`src/reduce/_.c`) to accept TAG_NUM
as a resolved arg type. Previously TAG_NUM was treated as "not ready", causing
stuck TAG_TOP nodes.

### 2. thvm_op_raw (`src/ctx/init.c`)

Lean 2-slot term creation that skips linear_use, shape tracking, and shadow
slots. 3ns vs 113ns for thvm_op. For internal backward ops where sharing is
managed explicitly (grad_cache, not DUP).

### 3. Rewrite + fusion skip on CPU (`src/rewrite/_.c`, `src/fuse/_.c`)

Skip `rewrite_apply` and `fuse_or_reduce` when the backend has no codegen
(dispatch_kernel_rs = NULL). Prevents stack overflow in `fuse_walk_inner`
for long lazy chains on CPU, and saves ~10ns per interaction.

### 4. Chase-Lev work-stealing deque (`src/parallel/wsdeque.c`)

Lock-free deque implementation: owner push/pop are wait-free, thief steal
uses atomic CAS. 64K slots per deque, cache-line aligned.

### 5. ThvmThread struct (`src/tinyhvm.h`, `src/parallel/workers.c`)

Per-thread state with heap bank range and tensor ID range. Added to the
TinyHVM struct as `ThvmThread threads[16]`. `threads_init()` partitions
heap and tensor space across N threads.

### 6. Parallel benchmark (`test/bench_parallel.m`)

Independent-context benchmark: each thread creates its own TinyHVM context
and reduces a TAG_NUM chain. Measures raw scaling potential without
shared-state contention.

### 7. TLS reduce_pool tuning (`src/reduce/_.c`)

`REDUCE_SLICE = 16384` (was 4096), `REDUCE_MAX_DEPTH = 32`. TLS pool is
4MB per thread. Supports chains up to 16K deep and 32 levels of recursive
thvm_reduce.

### 8. Shared-context parallel benchmark (`test/bench_shared_parallel.m`)

Balanced binary tree of TAG_NUM ADD operations. All threads reduce subtrees
of the SAME TinyHVM context (shared heap). Uses atomic task counter for
work distribution. Subtasks sized to ~4K nodes (64KB, fits L1 cache).

**Result:** 220 Mitrs/s peak at 3–6 threads. Cache-limited beyond that —
per-thread throughput degrades when total working set > L2 (48MB on M3 Max).

---

## What Remains (Phase 2+)

### Phase 2a: Shared-context infrastructure (DONE)

Shared-context parallel reduction works. Threads operate on the SAME heap.
No per-thread heap banks needed for TAG_NUM (no allocation during reduce).
No atomic refcounts needed (no tensors in TAG_NUM path).

**What worked:**
- Atomic task counter distributes subtrees to workers
- Workers call thvm_reduce on their subtree, write TAG_NUM result back
- pthread_join provides memory fence for result visibility
- DFS heap layout gives cache-friendly per-subtree access

**Cache bottleneck (not fixable without hardware change):**
- Independent contexts: 551 Mitrs/s at 12 threads (zero cache interference)
- Shared context: 220 Mitrs/s at 3–6 threads (shared L2 contention)
- Gap is entirely explained by cache: independent heaps fit in per-core L2,
  shared heap exceeds L2 at 6+ threads
- Tree DFS is ~40% slower than chain per-itr (branch misprediction, pointer chasing)

### Phase 2b: Per-thread heap banks + tensor IDs (for tensor ops)

For shared-context reduction of TAG_TEN operations (not just TAG_NUM),
threads need to allocate heap slots and tensor IDs without contention.

1. **Wire per-thread heap banks into heap_alloc**
   ```c
   static inline u64 heap_alloc(TinyHVM *ctx, u64 w) {
       if (ctx->n_threads <= 1) return global_bump(ctx, w);
       return thread_heap_alloc(ctx, w);  // tl_thread_id → bank
   }
   ```

2. **Wire per-thread tensor IDs into tensor_create**
   Same pattern: disjoint ID ranges per thread.

3. **Atomic refcounts on TensorMeta**
   `tensor_incref/decref` need `atomic_fetch_add` for safe sharing.

4. **GPU dispatch serialization**
   Metal command encoding is single-threaded. Spinlock or submit queue.

### Phase 2c: Fork-join in the trampoline

The current approach (pre-scatter subtrees) works but requires knowing the
tree structure upfront. True work-stealing forks dynamically inside the
trampoline:

1. In the apply phase for TAG_TOP: when arg1 is unreduced and n_threads > 1,
   push a "reduce arg1" task to the local WsDeque instead of processing inline
2. Continue reducing other work from own deque (LIFO) or steal (FIFO)
3. When returning to this TAG_TOP frame, check if arg1 was resolved by a thief
4. If so, fire immediately. If not, reduce it locally.

**Challenge:** the trampoline's stack encodes continuations. Forking requires
either copying stack frames or switching to a continuation-passing style.
The pre-scatter approach avoids this by keeping the trampoline unchanged.

### Phase 3: Backward parallelism

BIN_GRAD creates ADD(GRAD3(at, da, x), GRAD3(bt, db, x)). The two GRAD3
subtrees are independent — natural fork points. With work-stealing, each
branch reduces on a separate thread.

**Challenge:** gradient deposits use ASSIGN which writes to shared grad
slots. Need atomic CAS for gradient accumulation.

### Phase 4: Overhead reduction

1. **Eliminate thvm_op overhead for backward ops**: The GRAD handler creates
   many lazy ops via thvm_op (113ns each). Use thvm_op_raw where safe
   (backward chain ops without sharing).

2. **TAG_NUM for gradient scalars**: Loss scalars, learning rates, and
   gradient accumulators could use TAG_NUM instead of TAG_TEN. This would
   make the SGD step nearly free.

3. **Batch tensor allocation**: Instead of one tensor_create per interaction,
   pre-allocate a batch of tensors and bump through them.

4. **Reduce thvm_op to 2 heap slots**: The current 4-slot layout (with
   shadow slots for DUP) doubles heap usage. For non-shared terms, 2 slots
   suffice.

---

## Architecture Notes

### Why independent contexts scale but shared doesn't

Independent contexts: 551 Mitrs/s at 12 threads (9.7× speedup).
Shared context: 220 Mitrs/s at 4 threads (2.5× speedup).

The gap is **cache interference**, not contention. There are no locks,
no atomics on the hot path, and threads access disjoint heap regions.
But the regions share cache lines in the same physical memory:

- **Independent**: each thread's 160KB chain fits entirely in per-core L1/L2.
  Zero cross-core cache traffic.
- **Shared (depth 22)**: 64MB heap. Each thread's subtree is ~8MB.
  Exceeds per-core L2 (~12MB). When 8+ threads run, total working set
  exceeds shared L2 (48MB). Cache lines are evicted and reloaded.

Evidence:
- Depth 18 (4MB heap): 3.2× at 6 threads — fits in shared L2
- Depth 20 (16MB heap): 2.5× at 4 threads — exceeds per-core L2
- Depth 22 (64MB heap): 2.3× at 3 threads — exceeds shared L2

**Implication for CNN training:** the IC reduction between GPU dispatches
typically involves O(100–1000) interactions with a ~10KB working set.
This fits in L1 cache. Parallel backward should see near-linear scaling
for the IC reduction portion (though GPU dispatch remains serial).

### Why TAG_NUM is 1.5× slower than HVM4

HVM4 at 200 Mitrs/s does ~5ns/itr. TinyHVM TAG_NUM at 137 Mitrs/s does 7.3ns.
The 2.3ns gap is:

- **Trampoline overhead (~1.5ns)**: TinyHVM's stack-based trampoline does
  push/pop/goto per interaction. HVM4 uses a tighter CPS-style loop.
- **interact handler dispatch (~0.5ns)**: TinyHVM checks TAG_TOP + uop +
  multiple fast paths before reaching the TAG_NUM switch. HVM4 has a
  single combined tag-match.
- **heap_alloc (~0.3ns)**: TinyHVM allocates heap slots even for TAG_NUM
  results (through thvm_op_raw). HVM4's NUM nodes are passed by value.

To close the gap: eliminate heap allocation for TAG_NUM chains entirely.
Use a register-passing convention where TAG_NUM results flow through the
trampoline without heap storage.

### Single-threaded fast path

When `n_threads <= 1`, the entire parallel infrastructure is bypassed:
- `heap_alloc` uses the global bump pointer (no bank indirection)
- No atomic operations anywhere
- No deque push/pop
- Zero overhead vs current single-threaded reducer

This is critical — the default case must not slow down.
