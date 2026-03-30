TinyHVM *thvm_init(const char *default_device) {
    TinyHVM *ctx = calloc(1, sizeof(TinyHVM));
    ctx->tensors = calloc(MAX_TENSORS, sizeof(TensorMeta));
    ctx->heap = calloc(HEAP_CAP, sizeof(Term));
    ctx->heap_pos = 1;
    ctx->heap[0] = term_era();  // sentinel: prevent TAG_APP(0) self-loop
    ctx->tensor_count = 1;      // reserve tensor 0 as sentinel (0 = "no tensor")

    // Always register CPU backend
    ctx->backends[THVM_DEV_CPU] = &cpu_backend;
    ctx->n_backends = 1;
    ctx->default_device = THVM_DEV_CPU;
    if (cpu_backend.init) cpu_backend.init();

    // Register Metal backend if available and requested
    #ifdef __APPLE__
    if (default_device && (strcmp(default_device, "metal") == 0 || strcmp(default_device, "gpu") == 0)) {
        ctx->backends[THVM_DEV_METAL] = &metal_backend;
        ctx->n_backends = 2;
        ctx->default_device = THVM_DEV_METAL;
        if (metal_backend.init) metal_backend.init();
    }
    #else
    (void)default_device;
    #endif

    // Allocate trace buffer
    ctx->trace_cap = 1024;
    ctx->trace_buf = calloc(ctx->trace_cap, sizeof(struct InteractionTrace));

    return ctx;
}

// DUP use tracking (used by thvm_reset, thvm_free, and linear_use)
#define TERM_USE_SIZE 4096
static struct { Term term; u64 first_loc; u64 dup_loc; } term_use_table[TERM_USE_SIZE];

void thvm_free(TinyHVM *ctx) {
    // Free GPU buffers via each tensor's own backend
    for (u32 i = 0; i < ctx->tensor_count; i++) {
        if (ctx->tensors[i].buf_id && ctx->tensors[i].backend)
            ctx->tensors[i].backend->buf_free(ctx->tensors[i].buf_id);
    }
    // Shutdown all registered backends
    for (u32 i = 0; i < ctx->n_backends; i++) {
        if (ctx->backends[i] && ctx->backends[i]->shutdown)
            ctx->backends[i]->shutdown();
    }
    for (u32 i = 0; i < ctx->tensor_count; i++) {
        if (ctx->tensors[i].host_ptr) free(ctx->tensors[i].host_ptr);
    }
    free(ctx->trace_buf);
    free(ctx->tensors);
    free(ctx->heap);
    free(ctx);
    // Clear stale DUP tracking — TAG_TEN terms encode low tensor IDs
    // (0,1,2...) that collide across ctx instances.
    memset(term_use_table, 0, sizeof(term_use_table));
}

void thvm_reset(TinyHVM *ctx, u32 keep) {
    // Free ephemeral tensors (above `keep`), reset heap
    // Weight tensors [0..keep) are preserved
    if (thvm_prof_global.enabled) {
        thvm_prof_global.tensor_freed += ctx->tensor_count - keep;
        thvm_prof_global.heap_at_reset = ctx->heap_pos;
    }
    // Sum bytes being freed for memory tracking
    // Decref shared buffers before pool_reset — ensures correct per-buffer freeing
    for (u32 i = keep; i < ctx->tensor_count; i++) {
        TensorMeta *m = &ctx->tensors[i];
        if (m->buf_id && m->backend && m->backend->buf_decref)
            m->backend->buf_decref(m->buf_id);
        if (m->host_ptr) free(m->host_ptr);
        memset(m, 0, sizeof(TensorMeta));
    }
    // Compute max persistent buf_id from kept tensors
    u32 max_persistent_buf = 0;
    for (u32 i = 0; i < keep; i++)
        if (ctx->tensors[i].buf_id > max_persistent_buf)
            max_persistent_buf = ctx->tensors[i].buf_id;

    // Reset all backend buffer pools (frees GPU/CPU buffers above keep)
    for (u32 bi = 0; bi < ctx->n_backends; bi++) {
        if (ctx->backends[bi] && ctx->backends[bi]->pool_reset)
            ctx->backends[bi]->pool_reset(keep);
        if (ctx->backends[bi] && ctx->backends[bi]->pool_set_persistent)
            ctx->backends[bi]->pool_set_persistent(max_persistent_buf);
    }
    ctx->tensor_count = keep > 0 ? keep : 1; // keep sentinel at 0
    // Clear grad_refs/grad_cache for preserved tensors (re-counted each step)
    for (u32 i = 1; i < ctx->tensor_count; i++) {
        ctx->tensors[i].grad_refs = 0;
        ctx->tensors[i].grad_cache = 0;
    }
    ctx->heap_pos = 1;
    ctx->heap[0] = term_era();
    ctx->prescan_done = 0; // reset for next backward pass
    // Clear shape tracker — stale entries from old heap locs cause wrong
    // view compositions after heap reuse.
    memset(st_keys, 0, sizeof(st_keys));
    memset(term_use_table, 0, sizeof(term_use_table));
}

Term thvm_tensor(TinyHVM *ctx, const f32 *data, Shape s) {
    u32 id = tensor_create(ctx, s, DTYPE_F32);
    TensorMeta *m = &ctx->tensors[id];
    if (m->backend && data) {
        m->backend->buf_write(m->buf_id, data, (u64)m->view.numel * dtype_size(m->dtype));
    }
    // Cache host data for small metadata tensors (shapes, axes, permutations).
    // This allows shape tracking in thvm_op to read data without GPU buffer access.
    if (data && m->view.numel <= MAX_DIM) {
        m->host_ptr = malloc(m->view.numel * sizeof(f32));
        memcpy(m->host_ptr, data, m->view.numel * sizeof(f32));
    }
    return term_ten(id, DTYPE_F32);
}

// Get the output view of any term (TAG_TEN or TAG_TOP with shape tracking)
static const View *term_view(TinyHVM *ctx, Term t) {
    if (term_tag(t) == TAG_TEN) return &ctx->tensors[(u32)term_val(t)].view;
    if (term_tag(t) == TAG_TOP) return st_get(term_val(t));
    // Look through DUP nodes: DP0/DP1 share a value at heap[dl+0]
    if (term_tag(t) == TAG_DP0 || term_tag(t) == TAG_DP1)
        return term_view(ctx, heap_read(ctx, term_val(t)));
    return NULL;
}

// Read cached metadata from a TAG_TEN's host_ptr (no GPU access).
// Returns NULL if not cached.
static const f32 *tensor_host_f32(TinyHVM *ctx, u32 tid) {
    return (const f32 *)ctx->tensors[tid].host_ptr;
}

// Check if a term is a lazy elementwise op (not a chain — just 1 op with TAG_TEN args)
// These are safe to FUSE because single-op backward works with standard rules.
static int is_ew_single(Term t) {
    return (term_tag(t) == TAG_TOP && is_elementwise(term_ext(t)));
}

// DUP tracking: detect when the same term is used in multiple op slots.
// Creates a 1-slot DUP node (SUP with shared value) so the inet reducer
// can evaluate the shared term once and cache the result for both projections.
// Applies to ALL tags (TAG_TOP and TAG_TEN alike).
static Term linear_use(TinyHVM *ctx, Term t, u64 dest_loc) {
    u8 tag = term_tag(t);
    // ERA is the erasure — always safe to duplicate trivially
    if (tag == TAG_ERA || tag == TAG_NUM) return t;
    // Already a DUP projection — don't double-wrap
    if (tag == TAG_DP0 || tag == TAG_DP1) return t;

    u32 h = (u32)(t % TERM_USE_SIZE);
    for (u32 i = 0; i < 8; i++) {
        u32 idx = (h + i) % TERM_USE_SIZE;
        if (term_use_table[idx].term == 0) {
            // First use: record
            term_use_table[idx].term = t;
            term_use_table[idx].first_loc = dest_loc;
            term_use_table[idx].dup_loc = 0;
            return t;
        }
        if (term_use_table[idx].term == t) {
            // Second+ use: create or reuse DUP node
            u64 dl = term_use_table[idx].dup_loc;
            if (!dl) {
                // Create 1-slot DUP: just the shared value. No counters.
                // Both DP0 and DP1 read from heap[dl]. First to reduce
                // caches the result; second reads the cache.
                dl = heap_alloc(ctx, 1);
                heap_set(ctx, dl, t);
                term_use_table[idx].dup_loc = dl;
                // Patch first use site to DP0
                heap_set(ctx, term_use_table[idx].first_loc, term_new(TAG_DP0, 0, dl));
            }
            // N-way sharing: all 2nd+ uses get DP1 to the same 1-slot node.
            // First projection to reduce forces the value and caches the result;
            // all subsequent projections read the cache. This gives optimal sharing
            // for atoms (TEN, NUM) — equivalent to a binary SUP tree but simpler.
            return term_new(TAG_DP1, 0, dl);
        }
    }
    return t; // table full
}

// Fast path: skip linear_use + shape tracking. For internal backward ops
// where sharing is managed explicitly (grad_cache, not DUP).
Term thvm_op_raw(TinyHVM *ctx, u32 uop, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc, a);
    heap_set(ctx, loc + 1, b);
    return term_new(TAG_TOP, uop, loc);
}

Term thvm_op(TinyHVM *ctx, u32 uop, Term a, Term b) {
    // MM decomposition moved to interact handler (preserves creator_op for GRAD).

    u64 loc = heap_alloc(ctx, 4); // 0-1: args (trampoline overwrites), 2-3: shadow (preserved)
    a = linear_use(ctx, a, loc);
    heap_set(ctx, loc, a);              // write before second linear_use
    b = linear_use(ctx, b, loc + 1);    // may patch heap[loc] to DP0
    heap_set(ctx, loc + 1, b);
    heap_set(ctx, loc + 2, heap_read(ctx, loc)); // shadow: picks up DP0 patch
    heap_set(ctx, loc + 3, b);

    // Shape tracking: eagerly compute and store output view.
    // View ops with metadata (RESHAPE, EXPAND, SHRINK, PAD, PERMUTE) store
    // shape even without input view. Elementwise/reduce need input view.
    const View *va = term_view(ctx, a);
    {
        View out = {0};
        int stored = 0;

        // View ops: output shape deterministic from metadata
        if (term_tag(b) == TAG_TEN) {
            const f32 *bf = tensor_host_f32(ctx, (u32)term_val(b));
            u32 bn = bf ? ctx->tensors[(u32)term_val(b)].view.numel : 0;
            if (bf && uop == UOP_RESHAPE) {
                Shape ns = {.rank = bn}; u32 nn = 1;
                for (u32 i = 0; i < bn; i++) { ns.dims[i] = (u32)bf[i]; nn *= ns.dims[i]; }
                if (va && nn == va->numel) out = view_reshape(*va, ns);
                else out = view_create(ns);
                st_set(loc, &out); stored = 1;
            } else if (bf && uop == UOP_EXPAND) {
                if (va) {
                    out = *va; out.shape.rank = bn; out.numel = 1;
                    for (u32 i = 0; i < bn; i++) {
                        u32 nd = (u32)bf[i];
                        if (i < va->shape.rank && va->shape.dims[i] == 1 && nd > 1) out.strides[i] = 0;
                        out.shape.dims[i] = nd; out.numel *= nd;
                    }
                    out.contiguous = 0;
                } else {
                    Shape ns = {.rank = bn};
                    for (u32 i = 0; i < bn; i++) ns.dims[i] = (u32)bf[i];
                    out = view_create(ns);
                }
                st_set(loc, &out); stored = 1;
            } else if (bf && uop == UOP_SHRINK) {
                u32 ndim = bn / 2;
                Shape ns = {.rank = ndim}; u32 nn = 1;
                for (u32 i = 0; i < ndim; i++) { ns.dims[i] = (u32)bf[i*2+1] - (u32)bf[i*2]; nn *= ns.dims[i]; }
                out = view_create(ns);
                st_set(loc, &out); stored = 1;
            } else if (bf && uop == UOP_PAD && va) {
                u32 ndim = bn / 2;
                out = *va;
                for (u32 i = 0; i < ndim; i++)
                    out.shape.dims[i] += (u32)bf[i*2] + (u32)bf[i*2+1];
                out.numel = 1;
                for (u32 i = 0; i < out.shape.rank; i++) out.numel *= out.shape.dims[i];
                out = view_create(out.shape);
                st_set(loc, &out); stored = 1;
            } else if (bf && uop == UOP_PERMUTE && va) {
                out = (View){0}; out.offset = va->offset; out.shape.rank = bn; out.numel = va->numel;
                for (u32 i = 0; i < bn; i++) {
                    out.shape.dims[i] = va->shape.dims[(u32)bf[i]];
                    out.strides[i] = va->strides[(u32)bf[i]];
                }
                out.contiguous = 0;
                st_set(loc, &out); stored = 1;
            } else if ((uop == UOP_SUM || uop == UOP_RMAX)) {
                if (bf) {
                if (!va) goto skip_sum;
                // va available:
                out = *va;
                for (u32 i = 0; i < bn; i++) {
                    u32 ax = (u32)bf[i];
                    if (ax < out.shape.rank) out.shape.dims[ax] = 1;
                }
                out = view_create(out.shape);
                st_set(loc, &out); stored = 1;
                skip_sum:;
            }} // close bf check + view op metadata block
        }

        // SUM/RMAX with no explicit axes (b=ERA): reduce last non-1 dim
        if (!stored && va && (uop == UOP_SUM || uop == UOP_RMAX) && term_tag(b) == TAG_ERA) {
            out = *va;
            for (int d = (int)va->shape.rank - 1; d >= 0; d--)
                if (va->shape.dims[d] > 1) { out.shape.dims[d] = 1; break; }
            out = view_create(out.shape);
            st_set(loc, &out); stored = 1;
        }

        // MM: (M, K) @ (K, N) → (M, N)
        if (!stored && uop == UOP_MM && va) {
            const View *vb = (term_tag(b) != TAG_ERA) ? term_view(ctx, b) : NULL;
            if (vb) {
                Shape os = {.rank = 2, .dims = {va->shape.dims[0], vb->shape.dims[1]}};
                out = view_create(os);
                st_set(loc, &out); stored = 1;
            }
        }

        // Elementwise: broadcast if both views available, propagate if one
        if (!stored && is_elementwise(uop)) {
            const View *vb = (term_tag(b) != TAG_ERA) ? term_view(ctx, b) : NULL;
            if (va && vb) {
                View av_bc, bv_bc; u32 bc_shape[MAX_DIM], bc_ndim;
                if (view_broadcast(va, vb, &av_bc, &bv_bc, bc_shape, &bc_ndim))
                    out = view_create(shape_of(bc_shape, bc_ndim));
                else out = *va;
            } else if (va) out = *va;
            else if (vb) out = *vb;
            else goto skip_st;
            st_set(loc, &out);
            skip_st:;
        }
    }

    return term_top(uop, loc);
}

// Multi-axis SUM: reduces along specified axes in one pass.
// Matches tinygrad's .sum(axis=[...]) — the axes tensor is stored as the
// second heap slot, just like reshape stores shape.
Term thvm_sum_axes(TinyHVM *ctx, Term x, const u32 *axes, u32 n_axes) {
    f32 axes_f[MAX_DIM];
    for (u32 i = 0; i < n_axes; i++) axes_f[i] = (f32)axes[i];
    Term axes_t = thvm_tensor(ctx, axes_f, SHAPE(n_axes));
    return thvm_op(ctx, UOP_SUM, x, axes_t);
}

Term thvm_rmax_axes(TinyHVM *ctx, Term x, const u32 *axes, u32 n_axes) {
    f32 axes_f[MAX_DIM];
    for (u32 i = 0; i < n_axes; i++) axes_f[i] = (f32)axes[i];
    Term axes_t = thvm_tensor(ctx, axes_f, SHAPE(n_axes));
    return thvm_op(ctx, UOP_RMAX, x, axes_t);
}

// Movement ops: eager when input is TAG_TEN (zero GPU alloc, just view transform)
// Falls back to lazy TOP with shape-tensor only for unreduced inputs.

Term thvm_reshape(TinyHVM *ctx, Term t, Shape new_shape) {
    if (term_tag(t) == TAG_TEN) {
        u32 src_id = (u32)term_val(t);
        TensorMeta *m = &ctx->tensors[src_id];
        // Use view_reshape's merge-split algorithm to compute new strides.
        // Returns valid strides for contiguous, expanded (stride-0), and
        // compatible permuted views. Falls to lazy only when strides are
        // truly incompatible (mixed stride-0/non-zero, or non-contiguous merge).
        // Masked views (from PAD) must be materialized before reshape —
        // view_reshape drops the mask, causing wrong data downstream.
        if (m->view.has_mask) goto lazy;
        View rv = view_reshape(m->view, new_shape);
        // view_reshape returns contiguous=0 in two cases:
        // 1. Reshapable but non-contiguous (stride-0 dims) — strides are valid
        // 2. Not reshapable — strides are placeholder, needs materialization
        // Check: if any stride-0 dim in source has no corresponding stride-0
        // in result, it's case 2 (not reshapable). Simple heuristic: if result
        // has stride-0 dims or is contiguous, it's valid.
        int valid = rv.contiguous;
        if (!valid) {
            // Check if view_reshape produced valid non-contiguous strides
            // (stride-0 dims properly propagated). If any non-trivial dim
            // has stride 0, it's a valid expanded view alias.
            int has_stride0 = 0;
            for (u32 d = 0; d < new_shape.rank; d++)
                if (new_shape.dims[d] > 1 && rv.strides[d] == 0) has_stride0 = 1;
            // If source had stride-0 and result has stride-0, the merge-split
            // algorithm succeeded. If source had stride-0 but result has none,
            // it's the fallback (not reshapable).
            int src_has_stride0 = 0;
            for (u32 d = 0; d < m->view.shape.rank; d++)
                if (m->view.shape.dims[d] > 1 && m->view.strides[d] == 0) src_has_stride0 = 1;
            if (src_has_stride0 && has_stride0) valid = 1;
            if (!src_has_stride0 && !has_stride0) valid = 0; // permuted, needs materialize
        }
        if (!valid) goto lazy;
        // Reshape is valid as a view alias — same buffer, computed strides
        u32 id = ctx->tensor_count++;
        ctx->tensors[id] = *m; ctx->tensors[id].creator_loc = 0;
        ctx->tensors[id].view = rv;
        ctx->tensors[id].host_ptr = NULL;
        ctx->tensors[id].refcount = 1;
        ctx->tensors[id].creator_op = UOP_RESHAPE;
        ctx->tensors[id].src_ids[0] = src_id;
        if (m->buf_id && m->backend && m->backend->buf_incref)
            m->backend->buf_incref(m->buf_id);
        if (m->requires_grad) ctx->tensors[id].requires_grad = 1;
        return term_ten(id, m->dtype);
    }
lazy:;
    // Lazy fallback
    f32 dims[MAX_DIM];
    for (u32 i = 0; i < new_shape.rank; i++) dims[i] = (f32)new_shape.dims[i];
    Term shape_t = thvm_tensor(ctx, dims, SHAPE(new_shape.rank));
    return thvm_op(ctx, UOP_RESHAPE, t, shape_t);
}

Term thvm_expand(TinyHVM *ctx, Term t, Shape new_shape) {
    if (term_tag(t) == TAG_TEN) {
        u32 src_id = (u32)term_val(t);
        TensorMeta *m = &ctx->tensors[src_id];
        // Expand: set stride=0 where dim goes from 1→N
        u32 id = ctx->tensor_count++;
        ctx->tensors[id] = *m; ctx->tensors[id].creator_loc = 0;
        ctx->tensors[id].host_ptr = NULL;
        ctx->tensors[id].refcount = 1;
        if (m->buf_id && m->backend && m->backend->buf_incref)
            m->backend->buf_incref(m->buf_id);
        View *v = &ctx->tensors[id].view;
        if (v->shape.rank != new_shape.rank) {
            fprintf(stderr, "expand rank mismatch: tensor %u has rank %u, target rank %u\n"
                            "  tensor shape: [", src_id, v->shape.rank, new_shape.rank);
            for (u32 d=0;d<v->shape.rank;d++) fprintf(stderr, "%u%s", v->shape.dims[d], d+1<v->shape.rank?",":"");
            fprintf(stderr, "]  target: [");
            for (u32 d=0;d<new_shape.rank;d++) fprintf(stderr, "%u%s", new_shape.dims[d], d+1<new_shape.rank?",":"");
            fprintf(stderr, "]\n  creator_op=%u\n", ctx->tensors[src_id].creator_op);
            assert(0 && "expand rank mismatch");
        }
        u32 numel = 1;
        for (u32 i = 0; i < new_shape.rank; i++) {
            if (v->shape.dims[i] == 1 && new_shape.dims[i] > 1) {
                v->strides[i] = 0;
            }
            v->shape.dims[i] = new_shape.dims[i];
            numel *= new_shape.dims[i];
        }
        v->numel = numel;
        v->contiguous = 0;
        ctx->tensors[id].creator_op = UOP_EXPAND;
        ctx->tensors[id].src_ids[0] = src_id;
        if (m->requires_grad) ctx->tensors[id].requires_grad = 1;
        return term_ten(id, m->dtype);
    }
    f32 dims[MAX_DIM];
    for (u32 i = 0; i < new_shape.rank; i++) dims[i] = (f32)new_shape.dims[i];
    Term shape_t = thvm_tensor(ctx, dims, SHAPE(new_shape.rank));
    return thvm_op(ctx, UOP_EXPAND, t, shape_t);
}

Term thvm_permute(TinyHVM *ctx, Term t, const u32 *axes, u32 rank) {
    if (term_tag(t) == TAG_TEN) {
        u32 src_id = (u32)term_val(t);
        TensorMeta *m = &ctx->tensors[src_id];
        u32 id = ctx->tensor_count++;
        ctx->tensors[id] = *m; ctx->tensors[id].creator_loc = 0;
        ctx->tensors[id].host_ptr = NULL;
        ctx->tensors[id].refcount = 1;
        if (m->buf_id && m->backend && m->backend->buf_incref)
            m->backend->buf_incref(m->buf_id);
        ctx->tensors[id].view = view_permute(m->view, axes);
        ctx->tensors[id].creator_op = UOP_PERMUTE;
        ctx->tensors[id].src_ids[0] = src_id;
        // Store axes in src_ids[1] as tensor for backward (need this for inverse permute)
        f32 axes_f[MAX_DIM];
        for (u32 i = 0; i < rank; i++) axes_f[i] = (f32)axes[i];
        Term axes_t = thvm_tensor(ctx, axes_f, SHAPE(rank));
        ctx->tensors[id].src_ids[1] = (u32)term_val(axes_t);
        if (m->requires_grad) ctx->tensors[id].requires_grad = 1;
        return term_ten(id, m->dtype);
    }
    f32 axes_f[MAX_DIM];
    for (u32 i = 0; i < rank; i++) axes_f[i] = (f32)axes[i];
    Term axes_t = thvm_tensor(ctx, axes_f, SHAPE(rank));
    return thvm_op(ctx, UOP_PERMUTE, t, axes_t);
}

// Pad: pairs = [before0, after0, before1, after1, ...]
Term thvm_pad(TinyHVM *ctx, Term t, const u32 *pairs, u32 ndim) {
    if (term_tag(t) == TAG_TEN) {
        u32 src_id = (u32)term_val(t);
        TensorMeta *m = &ctx->tensors[src_id];
        u32 pad_before[MAX_DIM], pad_after[MAX_DIM];
        for (u32 i = 0; i < ndim; i++) {
            pad_before[i] = pairs[i*2];
            pad_after[i] = pairs[i*2+1];
        }
        // Zero-copy: create view with mask
        u32 id = ctx->tensor_count++;
        ctx->tensors[id] = *m; ctx->tensors[id].creator_loc = 0;
        ctx->tensors[id].host_ptr = NULL;
        ctx->tensors[id].refcount = 1;
        if (m->buf_id && m->backend && m->backend->buf_incref)
            m->backend->buf_incref(m->buf_id);
        ctx->tensors[id].view = view_pad(m->view, pad_before, pad_after);
        ctx->tensors[id].creator_op = UOP_PAD;
        ctx->tensors[id].src_ids[0] = src_id;
        f32 pairs_f[MAX_DIM * 2];
        for (u32 i2 = 0; i2 < ndim * 2; i2++) pairs_f[i2] = (f32)pairs[i2];
        Term pt = thvm_tensor(ctx, pairs_f, SHAPE(ndim * 2));
        ctx->tensors[id].src_ids[1] = (u32)term_val(pt);
        if (m->requires_grad) ctx->tensors[id].requires_grad = 1;
        return term_ten(id, m->dtype);
    }
    // Lazy fallback
    f32 pairs_f[MAX_DIM * 2];
    for (u32 i = 0; i < ndim * 2; i++) pairs_f[i] = (f32)pairs[i];
    Term pairs_t = thvm_tensor(ctx, pairs_f, SHAPE(ndim * 2));
    return thvm_op(ctx, UOP_PAD, t, pairs_t);
}

// Shrink: pairs = [start0, end0, start1, end1, ...]
Term thvm_shrink(TinyHVM *ctx, Term t, const u32 *pairs, u32 ndim) {
    if (term_tag(t) == TAG_TEN) {
        u32 src_id = (u32)term_val(t);
        TensorMeta *m = &ctx->tensors[src_id];
        // Shrink = adjust offset + shape (zero-copy if contiguous, otherwise materialize)
        u32 new_dims[MAX_DIM];
        i32 offset = m->view.offset;
        for (u32 i = 0; i < ndim; i++) {
            new_dims[i] = pairs[i*2+1] - pairs[i*2];
            offset += (i32)pairs[i*2] * m->view.strides[i];
        }
        // ns not needed — dims set directly on the view below
        u32 id = ctx->tensor_count++;
        ctx->tensors[id] = *m; ctx->tensors[id].creator_loc = 0;
        ctx->tensors[id].host_ptr = NULL;
        ctx->tensors[id].refcount = 1;
        if (m->buf_id && m->backend && m->backend->buf_incref)
            m->backend->buf_incref(m->buf_id);
        View *v = &ctx->tensors[id].view;
        for (u32 i = 0; i < ndim; i++) v->shape.dims[i] = new_dims[i];
        v->offset = offset;
        u32 numel = 1;
        for (u32 i = 0; i < ndim; i++) numel *= new_dims[i];
        v->numel = numel;
        // Keep original strides — they're still valid for the shrunk region
        // Contiguity check
        v->contiguous = 0;
        ctx->tensors[id].creator_op = UOP_SHRINK;
        ctx->tensors[id].src_ids[0] = src_id;
        // Store pairs as tensor for backward
        f32 shrink_f[MAX_DIM * 2];
        for (u32 i2 = 0; i2 < ndim * 2; i2++) shrink_f[i2] = (f32)pairs[i2];
        Term st = thvm_tensor(ctx, shrink_f, SHAPE(ndim * 2));
        ctx->tensors[id].src_ids[1] = (u32)term_val(st);
        if (m->requires_grad) ctx->tensors[id].requires_grad = 1;
        return term_ten(id, m->dtype);
    }
    f32 pairs_f[MAX_DIM * 2];
    for (u32 i = 0; i < ndim * 2; i++) pairs_f[i] = (f32)pairs[i];
    Term pairs_t = thvm_tensor(ctx, pairs_f, SHAPE(ndim * 2));
    return thvm_op(ctx, UOP_SHRINK, t, pairs_t);
}

// ============================================================
// _pool — tinygrad's sliding window via movement UOps
// Full implementation: handles k>s (complex/repeat) and k<=s (simple)
// From tensor.py:2278-2301
// ============================================================

