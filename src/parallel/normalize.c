// parallel/normalize.c — deep WHNF normalization via root-reachable walk.
//
// HVM4 equivalent: eval/normalize.c.  Replaces reduce_net_quiesce's
// whole-heap sweep with a principled walk: start at root, thvm_reduce
// each reachable heap slot to WHNF, enqueue the WHNF's arg slots,
// repeat until fixed point.
//
// Key differences from quiesce:
//   - Only visits slots reachable from root (not every heap cell).
//   - Calls thvm_reduce (full wnf) at each slot, not the ad-hoc fire_one.
//   - Seen-set prevents redundant work on shared slots.
//   - Uses the WsDeque primitive from parallel/wsdeque.c so the future
//     multi-thread version is a drop-in extension (cf. HVM4's pattern).
//
// Serial for now: only the main thread owns the deque via ws_push/ws_pop.
// Parallelization (steal-from-victim) can be added later mirroring
// thvm_collapse_par in src/inet/_.c.

static u8 *g_norm_seen       = NULL;
static u64  g_norm_seen_cap  = 0;

static inline int norm_seen_get(u64 loc) {
    if (loc >= g_norm_seen_cap) return 0;
    return g_norm_seen[loc];
}

static inline void norm_seen_set(u64 loc) {
    if (loc >= g_norm_seen_cap) {
        u64 new_cap = g_norm_seen_cap ? g_norm_seen_cap * 2 : 4096;
        while (new_cap <= loc) new_cap *= 2;
        u8 *new_buf = (u8 *)realloc(g_norm_seen, (size_t)new_cap);
        if (!new_buf) return;
        memset(new_buf + g_norm_seen_cap, 0, (size_t)(new_cap - g_norm_seen_cap));
        g_norm_seen     = new_buf;
        g_norm_seen_cap = new_cap;
    }
    g_norm_seen[loc] = 1;
}

// Deep-reduce `root` so every reachable arg slot holds WHNF.  Analogue
// of HVM4 eval_normalize, but serial: one owner using a LIFO deque.
Term thvm_normalize(TinyHVM *ctx, Term root) {
    if (wnf_is_atom(term_tag(root))) return root;

    // Reset seen-set for this normalization pass.
    if (g_norm_seen && g_norm_seen_cap > 0)
        memset(g_norm_seen, 0, (size_t)g_norm_seen_cap);

    // Anchor root in a heap slot so we can address it by loc and pick
    // the final value back up after possible in-place updates.
    u64 root_slot = heap_alloc(ctx, 1);
    heap_set(ctx, root_slot, root);

    WsDeque dq; ws_init(&dq);
    ws_push(&dq, root_slot);

    u64 loc;
    while (ws_pop(&dq, &loc)) {
        if (loc == 0 || loc >= ctx->heap_pos) continue;
        if (norm_seen_get(loc)) continue;
        norm_seen_set(loc);

        Term t = heap_read(ctx, loc);
        if (wnf_is_atom(term_tag(t))) continue;

        Term w = thvm_reduce(ctx, t);
        if (w != t) heap_set(ctx, loc, w);
        if (wnf_is_atom(term_tag(w))) continue;

        // DP0/DP1/VAR/UDP point to a single body slot, not arg slots
        // owned by this term — but enqueuing the target is still correct
        // (reduce at that slot is a no-op if it's already WHNF).
        // Push order: default LIFO (right-to-left, preserves original
        // lazy evaluation order).  Under step-graph tracing, push reverse
        // so slot 0 pops first (left-to-right — matches IC spec
        // convention where the principal port is the leftmost operand,
        // e.g. Leibniz chain-MUL sub_a annihilates before sub_b).
        int left_first = (getenv("THVM_STEP_GRAPH") != NULL);
        u32 ar = reduce_net_term_arity(w);
        u64 cloc = term_val(w);
        if (cloc == 0) continue;
        if (left_first) {
            for (u32 i = ar; i > 0; i--) {
                u64 child = cloc + (i - 1);
                if (child >= ctx->heap_pos) continue;
                ws_push(&dq, child);
            }
        } else {
            for (u32 i = 0; i < ar; i++) {
                u64 child = cloc + i;
                if (child >= ctx->heap_pos) break;
                ws_push(&dq, child);
            }
        }
    }

    Term out = heap_read(ctx, root_slot);
    heap_set(ctx, root_slot, term_era());
    return out;
}
