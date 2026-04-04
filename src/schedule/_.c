// schedule/_.c — Scheduler: fuse, plan memory, rewrite
// Memory planning happens entirely in schedule_rewrite (before dispatch).
// Buffer assignments are pre-computed via interval coloring.

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

// ── Kernel spec writing ──────────────────────────────────────────────
static u64 sched_write_reduce_spec(TinyHVM *ctx, u32 tid) {
    TensorMeta *m = &ctx->tensors[tid];
    u32 walk_tid = m->src_ids[0];
    if (!walk_tid) return 0;
    u32 ew_base = walk_tid;
    while (ctx->tensors[ew_base].buf_id == 0 && ctx->tensors[ew_base].creator_op &&
           is_view_op(ctx->tensors[ew_base].creator_op) && ctx->tensors[ew_base].src_ids[0])
        ew_base = ctx->tensors[ew_base].src_ids[0];
    if (!(ctx->tensors[ew_base].buf_id == 0 && is_elementwise(ctx->tensors[ew_base].creator_op)))
        return 0;
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
        for (u32 a = 0; a < axt->view.numel; a++) { u32 ax=(u32)af[a]; if(ax<fs.rank) rs.is_reduce[ax]=1; }
    } else { for (int d=(int)fs.rank-1;d>=0;d--) if(fs.dims[d]>1){rs.is_reduce[d]=1;break;} }
    u32 rank = fs.rank, vw = 1+MAX_DIM*2+2+1+MAX_DIM*2;
    u32 total = 5+3*n_ops+n_leaves+rank+rank+n_leaves*vw;
    u64 loc = heap_alloc(ctx, total); u32 p = 0;
    heap_set(ctx, loc+p++, term_num_u32(n_ops));
    heap_set(ctx, loc+p++, term_num_u32(n_leaves));
    heap_set(ctx, loc+p++, term_num_u32(rank));
    heap_set(ctx, loc+p++, term_num_u32(rs.reduce_type));
    heap_set(ctx, loc+p++, term_num_u32(m->view.numel));
    for (u32 i=0;i<n_ops;i++) { heap_set(ctx,loc+p++,term_num_u32(ops[i].uop)); heap_set(ctx,loc+p++,term_num_u32(ops[i].arg_a)); heap_set(ctx,loc+p++,term_num_u32(ops[i].arg_b)); }
    for (u32 i=0;i<n_leaves;i++) heap_set(ctx,loc+p++,term_num_u32(leaf_ids[i]));
    for (u32 d=0;d<rank;d++) heap_set(ctx,loc+p++,term_num_u32(fs.dims[d]));
    for (u32 d=0;d<rank;d++) heap_set(ctx,loc+p++,term_num_u32(rs.is_reduce[d]));
    for (u32 i=0;i<n_leaves;i++) {
        const View *v=leaf_views[i];
        heap_set(ctx,loc+p++,term_num_u32(v->shape.rank));
        for(u32 d=0;d<MAX_DIM;d++) heap_set(ctx,loc+p++,term_num_u32(v->shape.dims[d]));
        for(u32 d=0;d<MAX_DIM;d++) heap_set(ctx,loc+p++,term_num_u32((u32)v->strides[d]));
        heap_set(ctx,loc+p++,term_num_u32((u32)v->offset));
        heap_set(ctx,loc+p++,term_num_u32(v->numel));
        heap_set(ctx,loc+p++,term_num_u32(v->has_mask));
        for(u32 d=0;d<MAX_DIM;d++) heap_set(ctx,loc+p++,term_num_u32(v->mask_begin[d]));
        for(u32 d=0;d<MAX_DIM;d++) heap_set(ctx,loc+p++,term_num_u32(v->mask_end[d]));
    }
    return loc;
}

// ── Pre-assigned buffer table ────────────────────────────────────────
// sched_planned_buf[tid] = pre-assigned Metal buf_id (0 = not planned)
static u32 sched_planned_buf[MAX_TENSORS];

// ── Dispatch a fusing_loc kernel spec (uses pre-assigned buffer) ─────
static void fusing_interact(TinyHVM *ctx, u32 tid) {
    TensorMeta *m = &ctx->tensors[tid];
    if (m->buf_id != 0 || !m->fusing_loc) return;
    u64 loc = m->fusing_loc; u32 p = 0;
    u32 n_ops=term_as_u32(heap_read(ctx,loc+p++)), n_leaves=term_as_u32(heap_read(ctx,loc+p++));
    u32 rank=term_as_u32(heap_read(ctx,loc+p++)), red_type=term_as_u32(heap_read(ctx,loc+p++));
    u32 out_numel=term_as_u32(heap_read(ctx,loc+p++));
    FusedOp ops[FUSE_MAX_OPS];
    for(u32 i=0;i<n_ops&&i<FUSE_MAX_OPS;i++){ops[i].uop=term_as_u32(heap_read(ctx,loc+p++));ops[i].arg_a=term_as_u32(heap_read(ctx,loc+p++));ops[i].arg_b=term_as_u32(heap_read(ctx,loc+p++));}
    u32 leaf_ids[FUSE_MAX_LEAVES];
    for(u32 i=0;i<n_leaves&&i<FUSE_MAX_LEAVES;i++) leaf_ids[i]=term_as_u32(heap_read(ctx,loc+p++));
    Shape fs={.rank=rank};
    for(u32 d=0;d<rank&&d<MAX_DIM;d++) fs.dims[d]=term_as_u32(heap_read(ctx,loc+p++));
    ReduceSpec rs={0};
    if(red_type){rs.reduce_type=red_type; for(u32 d=0;d<rank&&d<MAX_DIM;d++) rs.is_reduce[d]=(u8)term_as_u32(heap_read(ctx,loc+p++));}
    View leaf_views[FUSE_MAX_LEAVES]; const View *lv_ptrs[FUSE_MAX_LEAVES];
    for(u32 i=0;i<n_leaves&&i<FUSE_MAX_LEAVES;i++){
        View *v=&leaf_views[i]; memset(v,0,sizeof(*v));
        v->shape.rank=term_as_u32(heap_read(ctx,loc+p++));
        for(u32 d=0;d<MAX_DIM;d++) v->shape.dims[d]=term_as_u32(heap_read(ctx,loc+p++));
        for(u32 d=0;d<MAX_DIM;d++) v->strides[d]=(i32)term_as_u32(heap_read(ctx,loc+p++));
        v->offset=(i32)term_as_u32(heap_read(ctx,loc+p++)); v->numel=term_as_u32(heap_read(ctx,loc+p++));
        v->has_mask=term_as_u32(heap_read(ctx,loc+p++));
        for(u32 d=0;d<MAX_DIM;d++) v->mask_begin[d]=term_as_u32(heap_read(ctx,loc+p++));
        for(u32 d=0;d<MAX_DIM;d++) v->mask_end[d]=term_as_u32(heap_read(ctx,loc+p++));
        lv_ptrs[i]=v;
    }
    for(u32 i=0;i<n_leaves;i++) if(leaf_ids[i]&&ctx->tensors[leaf_ids[i]].buf_id==0) ENSURE(ctx,leaf_ids[i]);
    // Use pre-assigned buffer if planned, else allocate fresh
    m->buf_id = sched_planned_buf[tid] ? sched_planned_buf[tid] : m->backend->buf_alloc((u64)out_numel*sizeof(f32));
    u32 bufs[FUSE_MAX_LEAVES];
    for(u32 i=0;i<n_leaves;i++) bufs[i]=ctx->tensors[leaf_ids[i]].buf_id;
    m->backend->dispatch_kernel_rs(m->buf_id,bufs,lv_ptrs,n_leaves,n_ops>0?ops:NULL,n_ops,&fs,red_type?&rs:NULL,NULL,NULL,0);
    m->fusing_loc = 0;
}

// ── Memory planner: interval coloring ────────────────────────────────
// Ported from JIT planner (jit.m:186-249).
// Input: spec_tids[] (kernel output tensor IDs in execution order).
// For each: compute out_bytes, leaf_tids. Assign buffers with reuse.
static void schedule_plan_memory(TinyHVM *ctx, u32 *spec_tids, u32 n_specs) {
    if (n_specs == 0) return;

    // Compute last_read[k] = last spec index that reads spec k's output as a leaf
    u32 last_read[512];
    for (u32 k = 0; k < n_specs; k++) last_read[k] = k; // default: self

    for (u32 k = 0; k < n_specs; k++) {
        TensorMeta *m = &ctx->tensors[spec_tids[k]];
        if (!m->fusing_loc) continue;
        u64 loc = m->fusing_loc;
        u32 no = term_as_u32(heap_read(ctx, loc));
        u32 nl = term_as_u32(heap_read(ctx, loc+1));
        u32 leaf_off = 5 + 3*no; // offset to leaf_ids in spec
        for (u32 i = 0; i < nl && i < FUSE_MAX_LEAVES; i++) {
            u32 lid = term_as_u32(heap_read(ctx, loc + leaf_off + i));
            // Find which spec wrote this leaf
            for (u32 j = 0; j < k; j++) {
                if (spec_tids[j] == lid && k > last_read[j])
                    last_read[j] = k;
            }
        }
    }

    // Greedy interval coloring (from JIT planner)
    struct { u32 buf_id; u64 size; } fpool[256];
    u32 fpool_n = 0;
    u32 assignment[512]; // spec index → buf_id
    memset(assignment, 0, n_specs * sizeof(u32));

    for (u32 k = 0; k < n_specs; k++) {
        // Release: specs whose output is no longer needed
        for (u32 j = 0; j < k; j++) {
            if (assignment[j] && last_read[j] < k) {
                // Return to free pool
                if (fpool_n < 256) {
                    fpool[fpool_n].buf_id = assignment[j];
                    fpool[fpool_n].size = (u64)ctx->tensors[spec_tids[j]].view.numel * sizeof(f32);
                    fpool_n++;
                }
                assignment[j] = 0; // released
            }
        }
        // Allocate: best-fit from free pool
        u64 need = (u64)ctx->tensors[spec_tids[k]].view.numel * sizeof(f32);
        int best = -1; u64 best_sz = UINT64_MAX;
        for (u32 f = 0; f < fpool_n; f++) {
            if (fpool[f].size >= need && fpool[f].size < best_sz) {
                best = (int)f; best_sz = fpool[f].size;
            }
        }
        if (best >= 0) {
            assignment[k] = fpool[best].buf_id;
            fpool[best] = fpool[--fpool_n];
        } else {
            assignment[k] = ctx->tensors[spec_tids[k]].backend->buf_alloc(need);
        }
        sched_planned_buf[spec_tids[k]] = assignment[k];
    }

    if (getenv("THVM_SCHED_DIAG")) {
        u64 total_alloc = 0, without_reuse = 0;
        for (u32 k = 0; k < n_specs; k++) without_reuse += (u64)ctx->tensors[spec_tids[k]].view.numel * 4;
        // Count unique buf_ids
        u32 unique[512]; u32 nu = 0;
        for (u32 k = 0; k < n_specs; k++) {
            int found = 0;
            for (u32 j = 0; j < nu; j++) if (unique[j] == assignment[k]) { found = 1; break; }
            if (!found && nu < 512) unique[nu++] = assignment[k];
        }
        for (u32 j = 0; j < nu; j++) {
            u64 mx = 0;
            for (u32 k = 0; k < n_specs; k++)
                if (assignment[k] == unique[j]) { u64 s = (u64)ctx->tensors[spec_tids[k]].view.numel*4; if(s>mx) mx=s; }
            total_alloc += mx;
        }
        fprintf(stderr, "PLAN: %u specs → %u bufs (%.1fMB, was %.1fMB, %.1fx reuse)\n",
            n_specs, nu, total_alloc/1e6, without_reuse/1e6, without_reuse>0?(double)without_reuse/total_alloc:0);
    }
}

// ── Schedule: build specs, plan memory, mark absorbed ────────────────
static void schedule_rewrite(TinyHVM *ctx, u32 from, u32 to) {
    memset(sched_absorbed, 0, to);
    memset(sched_planned_buf, 0, to * sizeof(u32));

    u32 spec_tids[512]; u32 n_specs = 0;
    for (u32 t = from; t < to; t++) {
        TensorMeta *m = &ctx->tensors[t];
        if (m->buf_id != 0 || !m->creator_op) continue;
        if (m->creator_op != UOP_SUM && m->creator_op != UOP_RMAX) continue;
        u64 spec = sched_write_reduce_spec(ctx, t);
        if (spec) {
            m->fusing_loc = spec; m->fusing_uop = m->creator_op;
            if (m->src_ids[0]) mark_absorbed(ctx, m->src_ids[0]);
            if (n_specs < 512) spec_tids[n_specs] = t;
            n_specs++;
        }
    }

    // Plan memory: pre-assign buffers with reuse
    schedule_plan_memory(ctx, spec_tids, n_specs < 512 ? n_specs : 512);

    // Don't mark absorbed tensors — let the old path handle everything.
    // The fusing specs provide optimized dispatch for reduces;
    // absorbed intermediates get materialized normally by tensor_materialize.

    if (getenv("THVM_SCHED_DIAG"))
        fprintf(stderr, "SCHED: %u kernel specs, %u planned bufs\n", n_specs, n_specs);
}

Term thvm_eval(TinyHVM *ctx, Term t) {
    if (!getenv("THVM_SCHED")) return thvm_reduce(ctx, t);
    if (!getenv("THVM_DEFER")) return thvm_reduce(ctx, t);
    u32 tc = ctx->tensor_count;
    ctx->defer_all = 1;
    t = thvm_reduce(ctx, t);
    ctx->defer_all = 0;
    schedule_rewrite(ctx, tc, ctx->tensor_count);
    return thvm_reduce(ctx, t);
}

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_eval(ctx, t);
}
