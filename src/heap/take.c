// heap/take.c — atomic take: swap cell with 0, spin if 0.
// Used for DP0/DP1 shared cells: first DP to arrive owns the value,
// second spins until the result is written back with SUB flag.
// Single-threaded fast path: plain read (no contention possible).
static inline Term heap_take(TinyHVM *ctx, u64 loc) {
    if (__builtin_expect(ctx->n_threads <= 1, 1)) {
        return ctx->heap[loc];
    }
    for (;;) {
        Term prev = __atomic_exchange_n(&ctx->heap[loc], 0, __ATOMIC_ACQUIRE);
        if (__builtin_expect(prev != 0, 1)) return prev;
#if defined(__aarch64__)
        __asm__ volatile("yield" ::: "memory");
#elif defined(__x86_64__)
        __asm__ volatile("pause");
#endif
    }
}
