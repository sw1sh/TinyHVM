// schedule/_.c — Lazy graph scheduler
// Walks the lazy inet, identifies kernel boundaries (SUM/RMAX/MM),
// and wraps elementwise+view subgraphs in FUSE nodes.
//
// Boundaries: SUM, RMAX, MM — these define kernels.
// Transparent: RESHAPE, PERMUTE, EXPAND, SHRINK, PAD — view ops.
// Fusable: elementwise ops between boundaries.
//
// The scheduler rewrites the graph BEFORE reduction. The trampoline
// then reduces FUSE nodes via fuse_or_reduce, dispatching fused kernels.

// Check if UOP is a kernel boundary (starts a new kernel)
static int is_boundary(u32 uop) {
    return uop == UOP_SUM || uop == UOP_RMAX || uop == UOP_MM;
}

// Check if UOP is transparent (view op — doesn't generate compute)
static int is_view_op(u32 uop) {
    return uop == UOP_RESHAPE || uop == UOP_PERMUTE || uop == UOP_EXPAND ||
           uop == UOP_SHRINK || uop == UOP_PAD;
}

// Check if a term's subgraph is a fusable elementwise+view chain.
// Returns 1 if the term is elementwise or a view op with elementwise below.
static int is_fusable_chain(TinyHVM *ctx, Term t) {
    if (term_tag(t) != TAG_TOP) return 0;
    u32 uop = term_ext(t);
    if (is_elementwise(uop)) return 1;
    if (is_view_op(uop)) {
        // View op: check if input is fusable
        Term input = heap_read(ctx, term_val(t));
        return is_fusable_chain(ctx, input);
    }
    return 0;
}

// Wrap a fusable chain in a FUSE node.
// FUSE(chain, ERA) — the trampoline handles this in enter phase.
static Term wrap_fuse(TinyHVM *ctx, Term chain) {
    u64 floc = heap_alloc(ctx, 2);
    heap_set(ctx, floc, chain);
    heap_set(ctx, floc + 1, term_era());
    return term_top(UOP_FUSING, floc);
}

// Walk the lazy graph, inject FUSE nodes at boundary inputs.
// Each TAG_TOP is visited once. The scheduler modifies the heap in-place
// (replacing boundary inputs with FUSE-wrapped versions).
static void sched_walk(TinyHVM *ctx, Term t, int depth) {
    if (depth > 128) return;
    if (term_tag(t) != TAG_TOP) return;

    u32 uop = term_ext(t);
    u64 loc = term_val(t);

    // Recurse into children first (bottom-up)
    Term a = heap_read(ctx, loc);
    if (term_tag(a) == TAG_TOP) sched_walk(ctx, a, depth + 1);

    // Binary ops: recurse into second arg too
    int has_b = is_binary(uop) || is_view_op(uop) || is_boundary(uop) ||
                uop == UOP_ASSIGN || uop == UOP_WHERE || uop == UOP_IFZ;
    if (has_b) {
        Term b = heap_read(ctx, loc + 1);
        if (term_tag(b) == TAG_TOP) sched_walk(ctx, b, depth + 1);
    }

    // At boundary ops: wrap fusable children in FUSE
    if (is_boundary(uop)) {
        a = heap_read(ctx, loc); // re-read (may have been modified by recursion)
        if (is_fusable_chain(ctx, a)) {
            heap_set(ctx, loc, wrap_fuse(ctx, a));
        }
    }
}

Term thvm_schedule(TinyHVM *ctx, Term t) {
    // Walk graph, inject FUSE at kernel boundaries
    sched_walk(ctx, t, 0);
    // Reduce the annotated graph
    return thvm_reduce(ctx, t);
}
