// heap/set.c — heap_set(): write a term to the heap
static inline void heap_set(TinyHVM *ctx, u64 l, Term t) {
    (void)ctx;
    ctx->heap[l] = t;
}
