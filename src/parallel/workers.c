// parallel/workers.c — Per-thread state and parallel reduction coordinator
//
// Phase 1: ThvmThread with per-thread heap banks and tensor ID ranges.
// Phase 2: parallel_reduce with work-stealing (TODO).
//
// Single-threaded mode (n_threads <= 1): zero overhead. bank[0] covers
// the entire heap, and tensor_alloc_id uses the global counter directly.

#include <stdatomic.h>
#include <pthread.h>

#define MAX_THREADS 16

typedef struct {
    u32 tid;
    // Heap bank: this thread allocates from [bank_start, bank_end)
    u64 bank_start;
    u64 bank_next;
    u64 bank_end;
    // Tensor ID range: this thread creates tensors in [tensor_start, tensor_end)
    u32 tensor_start;
    u32 tensor_next;
    u32 tensor_end;
    // Per-thread interaction counter
    u64 itrs;
} ThvmThread;

// Thread-local thread ID (set by worker init, 0 for main thread)
static _Thread_local u32 tl_thread_id = 0;

// Initialize thread states for n threads, partitioning heap and tensor space
static void threads_init(TinyHVM *ctx, u32 n_threads) {
    if (n_threads <= 1) return;  // single-threaded: use global allocators
    // Partition heap into n banks
    u64 heap_per = (HEAP_CAP - ctx->heap_pos) / n_threads;
    u64 base = ctx->heap_pos;
    // Partition tensor IDs into n ranges
    u32 tensors_per = (MAX_TENSORS - ctx->tensor_count) / n_threads;
    u32 tid_base = ctx->tensor_count;
    for (u32 i = 0; i < n_threads; i++) {
        ThvmThread *ts = &ctx->threads[i];
        ts->tid = i;
        ts->bank_start = base + i * heap_per;
        ts->bank_next  = ts->bank_start;
        ts->bank_end   = (i == n_threads - 1) ? HEAP_CAP : ts->bank_start + heap_per;
        ts->tensor_start = tid_base + i * tensors_per;
        ts->tensor_next  = ts->tensor_start;
        ts->tensor_end   = (i == n_threads - 1) ? MAX_TENSORS : ts->tensor_start + tensors_per;
        ts->itrs = 0;
    }
}

// Per-thread heap allocation (zero contention)
static inline u64 thread_heap_alloc(TinyHVM *ctx, u64 w) {
    ThvmThread *ts = &ctx->threads[tl_thread_id];
    u64 l = ts->bank_next;
    ts->bank_next += w;
    if (ts->bank_next > ts->bank_end) {
        fprintf(stderr, "THREAD_HEAP_OVERFLOW: tid=%u pos=%llu end=%llu\n",
            ts->tid, (unsigned long long)ts->bank_next, (unsigned long long)ts->bank_end);
        assert(0 && "thread heap bank overflow");
    }
    return l;
}

// Per-thread tensor ID allocation (zero contention)
static inline u32 thread_tensor_alloc(TinyHVM *ctx) {
    ThvmThread *ts = &ctx->threads[tl_thread_id];
    u32 id = ts->tensor_next++;
    if (id >= ts->tensor_end) {
        fprintf(stderr, "THREAD_TENSOR_OVERFLOW: tid=%u id=%u end=%u\n",
            ts->tid, id, ts->tensor_end);
        assert(0 && "thread tensor ID overflow");
    }
    return id;
}
