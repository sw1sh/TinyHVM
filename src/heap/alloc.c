// heap/alloc.c — heap_alloc(): bump-pointer allocator
// When n_threads > 1: per-thread bank allocation (zero contention).
// When n_threads <= 1: global bump pointer (zero overhead).
static inline u64 heap_alloc(TinyHVM *ctx, u64 w) {
  if (ctx->n_threads > 1) {
    ThvmThread *ts = &ctx->threads[tl_thread_id];
    u64 l = ts->bank_next;
    ts->bank_next += w;
    if (ts->bank_next > ts->bank_end) {
      fprintf(stderr, "THREAD_HEAP_OVERFLOW: tid=%u pos=%llu end=%llu\n",
              ts->tid, (unsigned long long)ts->bank_next, (unsigned long long)ts->bank_end);
      fflush(stderr); assert(0);
    }
    return l;
  }
  u64 l = ctx->heap_pos;
  ctx->heap_pos += w;
  if (ctx->heap_pos >= HEAP_CAP) { fprintf(stderr, "HEAP_OVERFLOW: pos=%llu cap=%llu\n", (unsigned long long)ctx->heap_pos, (unsigned long long)HEAP_CAP); fflush(stderr); assert(0); }
  return l;
}
