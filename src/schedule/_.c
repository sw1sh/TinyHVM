// schedule/_.c — Lazy graph compiler
// Walks TAG_TOP DAG, partitions into fusable kernel groups, dispatches.
//
// Phase 1: stub that falls through to thvm_reduce (no behavior change).
// Phase 2: forward scheduling with fusion.
// Phase 3: backward integration.

#define SCHED_MAX_ENTRIES 512

typedef struct {
    // For now, just track the root term and whether it was scheduled
    Term root;
    u32  result_tensor_id;
    u8   realized;
} SchedEntry;

typedef struct {
    SchedEntry entries[SCHED_MAX_ENTRIES];
    u32 n_entries;
} Schedule;

// Schedule a lazy term for execution.
// Phase 1: just calls thvm_reduce (no actual scheduling).
Term thvm_schedule(TinyHVM *ctx, Term t) {
    // TODO: walk TAG_TOP DAG, partition into fused kernels
    // For now, fall through to eager reduction
    return thvm_reduce(ctx, t);
}

// Execute a schedule (batch dispatch all kernels).
// Phase 1: no-op (thvm_schedule already reduced).
void thvm_exec(TinyHVM *ctx, Schedule *sched) {
    (void)ctx; (void)sched;
}
