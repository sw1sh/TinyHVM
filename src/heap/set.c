// heap/set.c — heap_set(): write a term to the heap
static inline void heap_set(TinyHVM *ctx, u64 l, Term t) {
    Term old = ctx->heap[l];
    thvm_dup_port_forget(ctx, l, old);
    ctx->heap[l] = t;
    thvm_dup_port_remember(ctx, l, t);
}
