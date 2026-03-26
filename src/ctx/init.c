Backend *thvm_device(const char *name) {
    #ifdef __APPLE__
    if (strcmp(name, "metal") == 0 || strcmp(name, "gpu") == 0)
        return &metal_backend;
    #endif
    (void)name;
    return &cpu_backend;
}

TinyHVM *thvm_init(Backend *backend) {
    TinyHVM *ctx = calloc(1, sizeof(TinyHVM));
    ctx->tensors = calloc(MAX_TENSORS, sizeof(TensorMeta));
    ctx->heap = calloc(HEAP_CAP, sizeof(Term));
    ctx->heap_pos = 1;
    ctx->heap[0] = term_era();  // sentinel: prevent TAG_APP(0) self-loop
    ctx->backend = backend;
    if (backend && backend->init) backend->init();
    return ctx;
}

void thvm_free(TinyHVM *ctx) {
    // Free GPU buffers first
    if (ctx->backend) {
        for (u32 i = 0; i < ctx->tensor_count; i++) {
            if (ctx->tensors[i].buf_id)
                ctx->backend->buf_free(ctx->tensors[i].buf_id);
        }
    }
    if (ctx->backend && ctx->backend->shutdown) ctx->backend->shutdown();
    for (u32 i = 0; i < ctx->tensor_count; i++) {
        if (ctx->tensors[i].host_ptr) free(ctx->tensors[i].host_ptr);
    }
    free(ctx->tensors);
    free(ctx->heap);
    free(ctx);
}

void thvm_reset(TinyHVM *ctx, u32 keep) {
    // Free ephemeral tensors (above `keep`), reset heap
    // Weight tensors [0..keep) are preserved
    if (thvm_prof_global.enabled) {
        thvm_prof_global.tensor_freed += ctx->tensor_count - keep;
        thvm_prof_global.heap_at_reset = ctx->heap_pos;
    }
    // Sum bytes being freed for memory tracking
    for (u32 i = keep; i < ctx->tensor_count; i++) {
        if (ctx->tensors[i].host_ptr) free(ctx->tensors[i].host_ptr);
        memset(&ctx->tensors[i], 0, sizeof(TensorMeta));
    }
    // Reset backend buffer pool (frees GPU/CPU buffers above keep)
    if (ctx->backend && ctx->backend->pool_reset)
        ctx->backend->pool_reset(keep);
    ctx->tensor_count = keep;
    ctx->heap_pos = 1;  // reset heap (keep weight terms as raw IDs)
    ctx->heap[0] = term_era();  // sentinel
}

Term thvm_tensor(TinyHVM *ctx, const f32 *data, Shape s) {
    u32 id = tensor_create(ctx, s, DTYPE_F32);
    TensorMeta *m = &ctx->tensors[id];
    if (ctx->backend && data) {
        ctx->backend->buf_write(m->buf_id, data, (u64)m->view.numel * dtype_size(m->dtype));
    }
    return term_ten(id, DTYPE_F32);
}

// Get the output view of any term (TAG_TEN or TAG_TOP with shape tracking)
static const View *term_view(TinyHVM *ctx, Term t) {
    if (term_tag(t) == TAG_TEN)
        return &ctx->tensors[(u32)term_val(t)].view;
    if (term_tag(t) == TAG_TOP)
        return st_get(term_val(t));
    return NULL;
}

Term thvm_op(TinyHVM *ctx, u32 uop, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc, a);
    heap_set(ctx, loc + 1, b);

    // Shape tracking: compute output view eagerly
    const View *va = term_view(ctx, a);
    if (va) {
        View out;
        if (is_elementwise(uop)) {
            // Elementwise: output = broadcast(a, b) or just a for unary
            const View *vb = (term_tag(b) != TAG_ERA) ? term_view(ctx, b) : NULL;
            if (vb) {
                View av_bc, bv_bc;
                u32 bc_shape[MAX_DIM], bc_ndim;
                if (view_broadcast(va, vb, &av_bc, &bv_bc, bc_shape, &bc_ndim))
                    out = view_create(shape_of(bc_shape, bc_ndim));
                else
                    out = *va; // fallback
            } else {
                out = *va; // unary
            }
            st_set(loc, &out);
        } else if (uop == UOP_SUM || uop == UOP_RMAX) {
            // Reduce: output shape = input with last non-1 dim → 1
            // (exact axis determined at reduce time, approximate here)
            out = *va;
            for (int d = (int)va->shape.rank - 1; d >= 0; d--) {
                if (va->shape.dims[d] > 1) {
                    out.shape.dims[d] = 1;
                    out.numel = va->numel / va->shape.dims[d];
                    break;
                }
            }
            st_set(loc, &out);
        } else if (uop == UOP_RESHAPE) {
            // Reshape: target shape from b (metadata tensor)
            if (term_tag(b) == TAG_TEN) {
                TensorMeta *mb = &ctx->tensors[(u32)term_val(b)];
                u32 rank = mb->view.numel;
                f32 sf[MAX_DIM];
                META_READ(ctx, mb->buf_id, sf, rank * sizeof(f32));
                Shape ns = {.rank = rank};
                for (u32 i = 0; i < rank; i++) ns.dims[i] = (u32)sf[i];
                out = view_reshape(*va, ns);
                st_set(loc, &out);
            }
        } else if (uop == UOP_EXPAND) {
            if (term_tag(b) == TAG_TEN) {
                TensorMeta *mb = &ctx->tensors[(u32)term_val(b)];
                u32 rank = mb->view.numel;
                f32 sf[MAX_DIM];
                META_READ(ctx, mb->buf_id, sf, rank * sizeof(f32));
                out = *va;
                out.shape.rank = rank;
                out.numel = 1;
                for (u32 i = 0; i < rank; i++) {
                    u32 nd = (u32)sf[i];
                    if (i < va->shape.rank && va->shape.dims[i] == 1 && nd > 1)
                        out.strides[i] = 0;
                    out.shape.dims[i] = nd;
                    out.numel *= nd;
                }
                out.contiguous = 0;
                st_set(loc, &out);
            }
        } else if (uop == UOP_PERMUTE) {
            if (term_tag(b) == TAG_TEN) {
                TensorMeta *mb = &ctx->tensors[(u32)term_val(b)];
                u32 rank = mb->view.numel;
                f32 pf[MAX_DIM];
                META_READ(ctx, mb->buf_id, pf, rank * sizeof(f32));
                out = (View){0};
                out.offset = va->offset;
                out.shape.rank = rank;
                out.numel = va->numel;
                for (u32 i = 0; i < rank; i++) {
                    out.shape.dims[i] = va->shape.dims[(u32)pf[i]];
                    out.strides[i] = va->strides[(u32)pf[i]];
                }
                out.contiguous = 0;
                st_set(loc, &out);
            }
        }
        // MM, SHRINK, PAD, etc. — not tracked yet (add as needed)
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
        ctx->tensors[id] = *m;
        ctx->tensors[id].view = rv;
        ctx->tensors[id].host_ptr = NULL;
        ctx->tensors[id].creator_op = UOP_RESHAPE;
        ctx->tensors[id].src_ids[0] = src_id;
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
        ctx->tensors[id] = *m;
        ctx->tensors[id].host_ptr = NULL;
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
        ctx->tensors[id] = *m;
        ctx->tensors[id].host_ptr = NULL;
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
        ctx->tensors[id] = *m;
        ctx->tensors[id].host_ptr = NULL;
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
        ctx->tensors[id] = *m;
        ctx->tensors[id].host_ptr = NULL;
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

