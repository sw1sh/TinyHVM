// parallel/reduce.c — Multi-threaded IC reduction benchmarking
#include <pthread.h>

Term parallel_reduce(TinyHVM *ctx, Term root, u32 n_threads) {
    (void)n_threads;
    return thvm_reduce(ctx, root);
}

// Worker: build + reduce on a pre-allocated context
typedef struct {
    TinyHVM *ctx;  // pre-allocated, one per thread
    u32 chain_len;
    f32 result;
    u64 itrs;
    double reduce_ms; // reduce-only time
} ChainWorkerArg;

static void *chain_worker_fn(void *arg) {
    ChainWorkerArg *cw = (ChainWorkerArg *)arg;
    TinyHVM *ctx = cw->ctx;
    // Build chain
    Term t = term_num_f32(0.0f);
    for (u32 i = 0; i < cw->chain_len; i++)
        t = thvm_op_raw(ctx, UOP_ADD, t, term_num_f32(1.0f));
    // Reduce (timed separately)
    ctx->itrs = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    t = thvm_reduce(ctx, t);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    cw->reduce_ms = (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_nsec-t0.tv_nsec)/1e6;
    cw->result = term_as_f32(t);
    cw->itrs = ctx->itrs;
    return NULL;
}

void parallel_bench(u32 n_threads, u32 chain_len) {
    // Pre-allocate contexts (outside timing)
    TinyHVM **ctxs = calloc(n_threads, sizeof(TinyHVM*));
    for (u32 i = 0; i < n_threads; i++) ctxs[i] = thvm_init("cpu");

    ChainWorkerArg *args = calloc(n_threads, sizeof(ChainWorkerArg));
    pthread_t *threads = calloc(n_threads, sizeof(pthread_t));
    for (u32 i = 0; i < n_threads; i++) {
        args[i].ctx = ctxs[i];
        args[i].chain_len = chain_len;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (u32 i = 0; i < n_threads; i++)
        pthread_create(&threads[i], NULL, chain_worker_fn, &args[i]);
    for (u32 i = 0; i < n_threads; i++)
        pthread_join(threads[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double wall_ms = (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_nsec-t0.tv_nsec)/1e6;

    u64 total_itrs = 0;
    double max_reduce_ms = 0;
    for (u32 i = 0; i < n_threads; i++) {
        total_itrs += args[i].itrs;
        if (args[i].reduce_ms > max_reduce_ms) max_reduce_ms = args[i].reduce_ms;
    }
    // Throughput = total interactions / wall time
    printf("  %2u thr: %5.0f Mitrs/s total (%4.0f/thr) wall=%.1fms reduce=%.1fms sum=%.0f\n",
        n_threads, total_itrs / (wall_ms/1000) / 1e6,
        total_itrs / (wall_ms/1000) / 1e6 / n_threads,
        wall_ms, max_reduce_ms, args[0].result);

    for (u32 i = 0; i < n_threads; i++) thvm_free(ctxs[i]);
    free(ctxs); free(args); free(threads);
}
