// parallel/workers.c — Per-thread utility functions
//
// ThvmThread and tl_thread_id are defined in tinyhvm.h / tinyhvm.c.
// This file provides per-thread allocation helpers used by parallel code.

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
