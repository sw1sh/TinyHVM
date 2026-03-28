// parallel/wsdeque.c — Chase-Lev work-stealing deque
// Lock-free: owner push/pop are wait-free, thief steal is lock-free CAS.
// Adapted from HVM4 and the Chase-Lev paper.

#include <stdatomic.h>

#define WS_CAP (1u << 16)  // 64K slots per deque

typedef struct {
    _Alignas(128) _Atomic(u64) top;   // steal end (thieves CAS here)
    _Alignas(128) _Atomic(u64) bot;   // owner end (push/pop)
    u64 buf[WS_CAP];
    u64 mask;
} WsDeque;

static inline void ws_init(WsDeque *q) {
    atomic_store(&q->top, 0);
    atomic_store(&q->bot, 0);
    q->mask = WS_CAP - 1;
}

// Owner push (wait-free)
static inline void ws_push(WsDeque *q, u64 task) {
    u64 b = atomic_load_explicit(&q->bot, memory_order_relaxed);
    q->buf[b & q->mask] = task;
    atomic_thread_fence(memory_order_release);
    atomic_store_explicit(&q->bot, b + 1, memory_order_relaxed);
}

// Owner pop (wait-free, handles race with thief on last element)
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
                // Thief got it
                atomic_store_explicit(&q->bot, b + 1, memory_order_relaxed);
                return 0;
            }
            atomic_store_explicit(&q->bot, b + 1, memory_order_relaxed);
        }
        return 1;
    }
    // Empty
    atomic_store_explicit(&q->bot, b + 1, memory_order_relaxed);
    return 0;
}

// Thief steal (lock-free CAS)
static inline int ws_steal(WsDeque *q, u64 *out) {
    u64 t = atomic_load_explicit(&q->top, memory_order_acquire);
    atomic_thread_fence(memory_order_seq_cst);
    u64 b = atomic_load_explicit(&q->bot, memory_order_acquire);
    if (t >= b) return 0;  // empty
    *out = q->buf[t & q->mask];
    return atomic_compare_exchange_strong_explicit(&q->top, &t, t + 1,
        memory_order_acq_rel, memory_order_relaxed);
}

static inline u64 ws_size(WsDeque *q) {
    u64 b = atomic_load_explicit(&q->bot, memory_order_relaxed);
    u64 t = atomic_load_explicit(&q->top, memory_order_relaxed);
    return (b > t) ? b - t : 0;
}
