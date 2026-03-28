// parallel/shared.c — Shared-context parallel reduction
// Multiple threads reduce subtrees of the SAME TinyHVM context.
// Uses atomic task counter for work distribution.

#include <pthread.h>
#include <stdatomic.h>

// ─── Thread pool (create once, reuse across reductions) ──────────────────────

#define PAR_MAX_TASKS 4096

typedef struct { TinyHVM *ctx; Term term; u64 write_loc; } ParTask;

static ParTask       par_tasks[PAR_MAX_TASKS];
static _Atomic(u32)  par_task_next;
static u32           par_task_count;

// Collect subtrees at target depth. write_loc = heap slot for result.
static void par_collect(TinyHVM *ctx, Term t, u64 write_loc, u32 d, u32 target) {
    if (par_task_count >= PAR_MAX_TASKS) return;
    if (d >= target || term_tag(t) != TAG_TOP) {
        par_tasks[par_task_count++] = (ParTask){ctx, t, write_loc};
        return;
    }
    u64 loc = term_val(t);
    par_collect(ctx, heap_read(ctx, loc),   loc,   d + 1, target);
    par_collect(ctx, heap_read(ctx, loc + 1), loc + 1, d + 1, target);
}

static void *par_worker(void *arg) {
    u32 tid = (u32)(uintptr_t)arg;
    tl_thread_id = tid;
    for (;;) {
        u32 i = atomic_fetch_add_explicit(&par_task_next, 1, memory_order_relaxed);
        if (i >= par_task_count) break;
        ParTask *t = &par_tasks[i];
        Term r = thvm_reduce(t->ctx, t->term);
        t->ctx->heap[t->write_loc] = r;
    }
    return NULL;
}

// Initialize per-thread banks on ctx for n_threads workers.
// Must be called before parallel_reduce_tree when heap_alloc/tensor_create
// may be called during reduction (e.g., TAG_TEN ops, GRAD).
static void par_init_banks(TinyHVM *ctx, u32 n_threads) {
    if (n_threads <= 1) return;
    u64 heap_per = (HEAP_CAP - ctx->heap_pos) / n_threads;
    u64 base = ctx->heap_pos;
    u32 tensors_per = (MAX_TENSORS - ctx->tensor_count) / n_threads;
    u32 tid_base = ctx->tensor_count;
    for (u32 i = 0; i < n_threads && i < 16; i++) {
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
    ctx->n_threads = n_threads;
}

// Finalize: update ctx->heap_pos and ctx->tensor_count from bank watermarks.
static void par_finalize_banks(TinyHVM *ctx) {
    if (ctx->n_threads <= 1) return;
    u64 max_heap = ctx->heap_pos;
    u32 max_tensor = ctx->tensor_count;
    for (u32 i = 0; i < ctx->n_threads; i++) {
        if (ctx->threads[i].bank_next > max_heap)
            max_heap = ctx->threads[i].bank_next;
        if (ctx->threads[i].tensor_next > max_tensor)
            max_tensor = ctx->threads[i].tensor_next;
    }
    ctx->heap_pos = max_heap;
    ctx->tensor_count = max_tensor;
    ctx->n_threads = 0;  // back to single-threaded mode
}

// Parallel reduce: distribute subtrees of `root` across n_threads.
// Returns the fully reduced result.
// For TAG_NUM trees: no heap/tensor allocation during reduce, banks not needed.
// For TAG_TEN trees: call par_init_banks first.
static Term parallel_reduce_tree(TinyHVM *ctx, Term root, u32 n_threads) {
    if (n_threads <= 1 || term_tag(root) != TAG_TOP)
        return thvm_reduce(ctx, root);

    // Target depth: subtasks of ~4K nodes (64KB, fits L1)
    u32 subtask_depth = 12;
    // Estimate tree depth from heap usage (rough heuristic)
    u32 td = subtask_depth;  // default: many fine tasks
    if (td < 2) td = 2;

    // Collect subtrees
    par_task_count = 0;
    atomic_store(&par_task_next, 0);
    u64 root_loc = term_val(root);
    par_collect(ctx, heap_read(ctx, root_loc),     root_loc,     1, td);
    par_collect(ctx, heap_read(ctx, root_loc + 1), root_loc + 1, 1, td);

    if (par_task_count <= 1) return thvm_reduce(ctx, root);

    // Cap thread count to available tasks
    u32 nt = (n_threads < par_task_count) ? n_threads : par_task_count;

    // Spawn workers
    pthread_t threads[16];
    for (u32 i = 0; i < nt; i++)
        pthread_create(&threads[i], NULL, par_worker, (void *)(uintptr_t)i);
    for (u32 i = 0; i < nt; i++)
        pthread_join(threads[i], NULL);

    // Final merge (upper-level interactions)
    tl_thread_id = 0;
    return thvm_reduce(ctx, root);
}
