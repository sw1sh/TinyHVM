// schedule/_.c — Scheduler: pure rewrite (no dispatch)
// Writes fusing_loc specs on deferred TensorMeta.
// Second thvm_reduce dispatches via ASSIGN → ENSURE → fusing_loc.

static u8 sched_absorbed[MAX_TENSORS];

static void mark_absorbed(TinyHVM *ctx, u32 tid) {
    TensorMeta *m = &ctx->tensors[tid];
    if (m->buf_id != 0 || sched_absorbed[tid]) return;
    if (!m->creator_op) return;
    if (!is_elementwise(m->creator_op) && !is_view_op(m->creator_op)) return;
    sched_absorbed[tid] = 1;
    if (m->src_ids[0]) mark_absorbed(ctx, m->src_ids[0]);
    if (m->src_ids[1] && is_binary(m->creator_op)) mark_absorbed(ctx, m->src_ids[1]);
}

// Write kernel spec for a fused reduce+ew kernel.
// Stores FusedOps, leaf_ids, views, ReduceSpec, full_shape on the heap.
// Returns spec heap location, or 0 on failure.
static u64 sched_write_reduce_spec(TinyHVM *ctx, u32 tid) {
    TensorMeta *m = &ctx->tensors[tid];
    u32 walk_tid = m->src_ids[0];
    if (!walk_tid) return 0;

    u32 ew_base = walk_tid;
    while (ctx->tensors[ew_base].buf_id == 0 &&
           ctx->tensors[ew_base].creator_op &&
           is_view_op(ctx->tensors[ew_base].creator_op) &&
           ctx->tensors[ew_base].src_ids[0])
        ew_base = ctx->tensors[ew_base].src_ids[0];

    if (!(ctx->tensors[ew_base].buf_id == 0 &&
          is_elementwise(ctx->tensors[ew_base].creator_op)))
        return 0; // can't fuse

    FusedOp ops[FUSE_MAX_OPS]; u32 n_ops = 0, op_tids[FUSE_MAX_OPS];
    u32 leaf_ids[FUSE_MAX_LEAVES]; const View *leaf_views[FUSE_MAX_LEAVES]; u32 n_leaves = 0;
    ref_prescan(ctx);
    walk_no_reshape_through = 1;
    int r = materialize_walk(ctx, ew_base, ops, &n_ops, op_tids, leaf_ids, leaf_views, &n_leaves);
    walk_no_reshape_through = 0;
    if (r < 0 || n_ops == 0) return 0;

    for (u32 i = 0; i < n_ops; i++) {
        if (ops[i].arg_a >= FUSE_MAX_LEAVES) ops[i].arg_a = n_leaves + (ops[i].arg_a - FUSE_MAX_LEAVES);
        if (ops[i].arg_b >= FUSE_MAX_LEAVES) ops[i].arg_b = n_leaves + (ops[i].arg_b - FUSE_MAX_LEAVES);
    }

    ReduceSpec rs = {0}; rs.reduce_type = m->creator_op;
    Shape fs = ctx->tensors[ew_base].view.shape;
    u32 axes_id = m->src_ids[1];
    if (axes_id && ctx->tensors[axes_id].buf_id != 0) {
        TensorMeta *axt = &ctx->tensors[axes_id];
        f32 af[MAX_DIM]; META_READ(axt->backend, axt->buf_id, af, axt->view.numel*4);
        for (u32 a = 0; a < axt->view.numel; a++) {
            u32 ax = (u32)af[a]; if (ax < fs.rank) rs.is_reduce[ax] = 1;
        }
    } else {
        for (int d = (int)fs.rank-1; d >= 0; d--)
            if (fs.dims[d] > 1) { rs.is_reduce[d] = 1; break; }
    }

    // Pack spec to heap: [n_ops, n_leaves, rank, reduce_type, out_numel,
    //   ops(uop,a,b)*n_ops, leaf_ids*n_leaves, dims*rank, reduce_axes*rank,
    //   per-leaf views]
    u32 rank = fs.rank;
    u32 vw = 1 + MAX_DIM*2 + 2 + 1 + MAX_DIM*2; // per-leaf view words
    u32 total = 5 + 3*n_ops + n_leaves + rank + rank + n_leaves*vw;
    u64 loc = heap_alloc(ctx, total);
    u32 p = 0;
    heap_set(ctx, loc+p++, term_num_u32(n_ops));
    heap_set(ctx, loc+p++, term_num_u32(n_leaves));
    heap_set(ctx, loc+p++, term_num_u32(rank));
    heap_set(ctx, loc+p++, term_num_u32(rs.reduce_type));
    heap_set(ctx, loc+p++, term_num_u32(m->view.numel));
    for (u32 i = 0; i < n_ops; i++) {
        heap_set(ctx, loc+p++, term_num_u32(ops[i].uop));
        heap_set(ctx, loc+p++, term_num_u32(ops[i].arg_a));
        heap_set(ctx, loc+p++, term_num_u32(ops[i].arg_b));
    }
    for (u32 i = 0; i < n_leaves; i++)
        heap_set(ctx, loc+p++, term_num_u32(leaf_ids[i]));
    for (u32 d = 0; d < rank; d++)
        heap_set(ctx, loc+p++, term_num_u32(fs.dims[d]));
    for (u32 d = 0; d < rank; d++)
        heap_set(ctx, loc+p++, term_num_u32(rs.is_reduce[d]));
    for (u32 i = 0; i < n_leaves; i++) {
        const View *v = leaf_views[i];
        heap_set(ctx, loc+p++, term_num_u32(v->shape.rank));
        for (u32 d = 0; d < MAX_DIM; d++) heap_set(ctx, loc+p++, term_num_u32(v->shape.dims[d]));
        for (u32 d = 0; d < MAX_DIM; d++) heap_set(ctx, loc+p++, term_num_u32((u32)v->strides[d]));
        heap_set(ctx, loc+p++, term_num_u32((u32)v->offset));
        heap_set(ctx, loc+p++, term_num_u32(v->numel));
        heap_set(ctx, loc+p++, term_num_u32(v->has_mask));
        for (u32 d = 0; d < MAX_DIM; d++) heap_set(ctx, loc+p++, term_num_u32(v->mask_begin[d]));
        for (u32 d = 0; d < MAX_DIM; d++) heap_set(ctx, loc+p++, term_num_u32(v->mask_end[d]));
    }
    return loc;
}

// Dispatch a fusing_loc kernel spec (called from ENSURE macro)
static void sched_dispatch_fusing(TinyHVM *ctx, u32 tid) {
    TensorMeta *m = &ctx->tensors[tid];
    if (m->buf_id != 0 || !m->fusing_loc) return;
    u64 loc = m->fusing_loc;
    u32 p = 0;
    u32 n_ops    = term_as_u32(heap_read(ctx, loc+p++));
    u32 n_leaves = term_as_u32(heap_read(ctx, loc+p++));
    u32 rank     = term_as_u32(heap_read(ctx, loc+p++));
    u32 red_type = term_as_u32(heap_read(ctx, loc+p++));
    u32 out_numel= term_as_u32(heap_read(ctx, loc+p++));
    FusedOp ops[FUSE_MAX_OPS];
    for (u32 i = 0; i < n_ops && i < FUSE_MAX_OPS; i++) {
        ops[i].uop   = term_as_u32(heap_read(ctx, loc+p++));
        ops[i].arg_a = term_as_u32(heap_read(ctx, loc+p++));
        ops[i].arg_b = term_as_u32(heap_read(ctx, loc+p++));
    }
    u32 leaf_ids[FUSE_MAX_LEAVES];
    for (u32 i = 0; i < n_leaves && i < FUSE_MAX_LEAVES; i++)
        leaf_ids[i] = term_as_u32(heap_read(ctx, loc+p++));
    Shape fs = {.rank = rank};
    for (u32 d = 0; d < rank && d < MAX_DIM; d++)
        fs.dims[d] = term_as_u32(heap_read(ctx, loc+p++));
    ReduceSpec rs = {0};
    if (red_type) {
        rs.reduce_type = red_type;
        for (u32 d = 0; d < rank && d < MAX_DIM; d++)
            rs.is_reduce[d] = (u8)term_as_u32(heap_read(ctx, loc+p++));
    }
    View leaf_views[FUSE_MAX_LEAVES];
    const View *lv_ptrs[FUSE_MAX_LEAVES];
    for (u32 i = 0; i < n_leaves && i < FUSE_MAX_LEAVES; i++) {
        View *v = &leaf_views[i]; memset(v, 0, sizeof(*v));
        v->shape.rank = term_as_u32(heap_read(ctx, loc+p++));
        for (u32 d = 0; d < MAX_DIM; d++) v->shape.dims[d] = term_as_u32(heap_read(ctx, loc+p++));
        for (u32 d = 0; d < MAX_DIM; d++) v->strides[d] = (i32)term_as_u32(heap_read(ctx, loc+p++));
        v->offset = (i32)term_as_u32(heap_read(ctx, loc+p++));
        v->numel = term_as_u32(heap_read(ctx, loc+p++));
        v->has_mask = term_as_u32(heap_read(ctx, loc+p++));
        for (u32 d = 0; d < MAX_DIM; d++) v->mask_begin[d] = term_as_u32(heap_read(ctx, loc+p++));
        for (u32 d = 0; d < MAX_DIM; d++) v->mask_end[d] = term_as_u32(heap_read(ctx, loc+p++));
        lv_ptrs[i] = v;
    }
    // Ensure leaves
    for (u32 i = 0; i < n_leaves; i++) {
        u32 lid = leaf_ids[i];
        if (lid && ctx->tensors[lid].buf_id == 0)
            ENSURE(ctx, lid); // recursive: may dispatch other fusing specs
    }
    // Allocate + dispatch
    m->buf_id = m->backend->buf_alloc((u64)out_numel * sizeof(f32));
    u32 bufs[FUSE_MAX_LEAVES];
    for (u32 i = 0; i < n_leaves; i++) bufs[i] = ctx->tensors[leaf_ids[i]].buf_id;
    m->backend->dispatch_kernel_rs(m->buf_id, bufs, lv_ptrs, n_leaves,
        n_ops > 0 ? ops : NULL, n_ops, &fs, red_type ? &rs : NULL, NULL, NULL, 0);
    m->fusing_loc = 0;
}

// Schedule: mark absorbed, write specs, set fusing_loc
static void schedule_rewrite(TinyHVM *ctx, u32 from, u32 to) {
    memset(sched_absorbed, 0, to);

    // Write specs for reduces, and mark absorbed chains only for successful fusions
    u32 n_specs = 0;
    for (u32 t = from; t < to; t++) {
        TensorMeta *m = &ctx->tensors[t];
        if (m->buf_id != 0 || !m->creator_op) continue;
        if (m->creator_op != UOP_SUM && m->creator_op != UOP_RMAX) continue;
        u64 spec = sched_write_reduce_spec(ctx, t);
        if (spec) {
            m->fusing_loc = spec;
            m->fusing_uop = m->creator_op;
            // Mark input chain as absorbed (virtual — no separate buffer)
            if (m->src_ids[0]) mark_absorbed(ctx, m->src_ids[0]);
            n_specs++;
        }
    }

    // Mark absorbed tensors with sentinel buf_id so second reduce skips them
    for (u32 t = from; t < to; t++) {
        if (sched_absorbed[t] && ctx->tensors[t].buf_id == 0)
            ctx->tensors[t].buf_id = 1; // sentinel
    }

    if (getenv("THVM_SCHED_DIAG"))
        fprintf(stderr, "SCHED: %u kernel specs written\n", n_specs);
}

Term thvm_eval(TinyHVM *ctx, Term t) {
    if (!getenv("THVM_SCHED")) return thvm_reduce(ctx, t);
    u32 tc = ctx->tensor_count;
    ctx->defer_all = 1;
    t = thvm_reduce(ctx, t);
    ctx->defer_all = 0;
    schedule_rewrite(ctx, tc, ctx->tensor_count);
    return thvm_reduce(ctx, t); // ASSIGN→ENSURE→fusing_loc dispatch
}

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_eval(ctx, t);
}
