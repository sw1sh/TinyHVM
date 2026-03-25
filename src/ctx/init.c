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

Term thvm_op(TinyHVM *ctx, u32 uop, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc, a);
    heap_set(ctx, loc + 1, b);
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
        // Create a view alias: same buffer, new shape
        u32 id = ctx->tensor_count++;
        ctx->tensors[id] = *m;
        ctx->tensors[id].view = view_create(new_shape);
        ctx->tensors[id].view.offset = m->view.offset;
        ctx->tensors[id].host_ptr = NULL;
        // Track provenance for autograd
        ctx->tensors[id].creator_op = UOP_RESHAPE;
        ctx->tensors[id].src_ids[0] = src_id;
        if (m->requires_grad) ctx->tensors[id].requires_grad = 1;
        return term_ten(id, m->dtype);
    }
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
        // Pad requires materializing: new buffer with zeros + copy
        // We must allocate a new buffer and copy data
        View old_v = m->view;
        u32 new_dims[MAX_DIM];
        for (u32 i = 0; i < ndim; i++)
            new_dims[i] = old_v.shape.dims[i] + pairs[i*2] + pairs[i*2+1];
        Shape ns = shape_of(new_dims, ndim);
        u32 new_numel = 1;
        for (u32 i = 0; i < ndim; i++) new_numel *= new_dims[i];

        u32 id = tensor_create(ctx, ns, m->dtype);

        if (ctx->backend) {
            u32 dsz = dtype_size(m->dtype);
            // Zero the new buffer
            f32 *tmp = calloc(new_numel, dsz);
            // Copy old data at offset
            f32 *src = malloc(old_v.numel * dsz);
            ctx->backend->buf_read(m->buf_id, src, old_v.numel * dsz);

            // Copy element by element respecting padding offset
            for (u32 flat = 0; flat < old_v.numel; flat++) {
                u32 coords[MAX_DIM], rem = flat;
                for (int d = (int)ndim - 1; d >= 0; d--) {
                    coords[d] = rem % old_v.shape.dims[d];
                    rem /= old_v.shape.dims[d];
                }
                // Compute destination flat index
                u32 dst_flat = 0, dst_stride = 1;
                for (int d = (int)ndim - 1; d >= 0; d--) {
                    dst_flat += (coords[d] + pairs[d*2]) * dst_stride;
                    dst_stride *= new_dims[d];
                }
                tmp[dst_flat] = src[flat];
            }

            ctx->backend->buf_write(ctx->tensors[id].buf_id, tmp, new_numel * dsz);
            free(src); free(tmp);
        }

        ctx->tensors[id].creator_op = UOP_PAD;
        ctx->tensors[id].src_ids[0] = src_id;
        // Store pairs as tensor for backward
        f32 pairs_f[MAX_DIM * 2];
        for (u32 i2 = 0; i2 < ndim * 2; i2++) pairs_f[i2] = (f32)pairs[i2];
        Term pt = thvm_tensor(ctx, pairs_f, SHAPE(ndim * 2));
        ctx->tensors[id].src_ids[1] = (u32)term_val(pt);
        if (m->requires_grad) ctx->tensors[id].requires_grad = 1;
        return term_ten(id, m->dtype);
    }
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

static Term thvm_repeat(TinyHVM *ctx, Term x, const u32 *repeats, u32 ndim) {
    // repeat = reshape(unsqueezed).expand(expanded).reshape(final)
    Term xr = thvm_reduce(ctx, x);
    TensorMeta *mx = &ctx->tensors[(u32)term_val(xr)];

    u32 unsq[MAX_DIM], exp[MAX_DIM], fin[MAX_DIM];
    for (u32 i = 0; i < ndim; i++) {
        unsq[i * 2]     = 1;
        unsq[i * 2 + 1] = mx->view.shape.dims[i];
        exp[i * 2]      = repeats[i];
        exp[i * 2 + 1]  = mx->view.shape.dims[i];
        fin[i]          = repeats[i] * mx->view.shape.dims[i];
    }
    Term t = thvm_reshape(ctx, xr, shape_of(unsq, ndim * 2));
    t = thvm_expand(ctx, t, shape_of(exp, ndim * 2));
    t = thvm_reshape(ctx, t, shape_of(fin, ndim));
    return t;
}

Term thvm_pool(TinyHVM *ctx, Term x, const u32 *kernel, const u32 *stride_,
               u32 n_spatial) {
    Term xr = thvm_reduce(ctx, x);
    TensorMeta *mx = &ctx->tensors[(u32)term_val(xr)];
    u32 ndim = mx->view.shape.rank;
    u32 bd = ndim - n_spatial;  // batch dims count

    u32 i_[MAX_DIM], o_[MAX_DIM], s_[MAX_DIM], k_[MAX_DIM];
    for (u32 j = 0; j < n_spatial; j++) {
        i_[j] = mx->view.shape.dims[bd + j];
        s_[j] = stride_[j];
        k_[j] = kernel[j];
        o_[j] = (i_[j] - k_[j]) / s_[j] + 1;  // floor division for conv
    }

    // Check if we need the complex path (k > s for any spatial dim)
    int need_complex = 0;
    for (u32 j = 0; j < n_spatial; j++)
        if (k_[j] > s_[j]) need_complex = 1;

    Term t = xr;

    if (need_complex) {
        // Tinygrad's _pool pattern: repeat → shrink → reshape → shrink → reshape → permute
        // All standard UOps — backward flows through standard rules, no special POOL_GATHER op.
        // See tinygrad tensor.py _pool() lines 2285-2296
        assert(n_spatial == 2);

        // f_ = scaling factors (always 1 for s=1, d=1)
        u32 f_[2];
        for (u32 j = 0; j < n_spatial; j++)
            f_[j] = 1 + (o_[j] * s_[j] > (i_[j] - (k_[j] - 1)) ? 1 : 0);

        // repeat: tile input so we can extract overlapping windows without padding
        u32 reps[MAX_DIM];
        for (u32 j = 0; j < bd; j++) reps[j] = 1;
        for (u32 j = 0; j < n_spatial; j++) {
            u32 ij = i_[j], kj = k_[j], fj = f_[j];
            reps[bd + j] = (kj * (ij * fj + 1) + ij - 1) / ij;  // ceildiv
        }
        t = thvm_repeat(ctx, t, reps, ndim);

        // shrink to [batch, k0*(i0*f0+1), k1*(i1*f1+1)]
        u32 sh1[MAX_DIM * 2];
        for (u32 j = 0; j < bd; j++) { sh1[j*2] = 0; sh1[j*2+1] = mx->view.shape.dims[j]; }
        for (u32 j = 0; j < n_spatial; j++) {
            sh1[(bd+j)*2] = 0;
            sh1[(bd+j)*2+1] = k_[j] * (i_[j] * f_[j] + 1);
        }
        t = thvm_shrink(ctx, t, sh1, ndim);

        // reshape to [batch, k0, i0*f0+1, k1, i1*f1+1]
        u32 rs1[MAX_DIM], rs1_rank = bd + n_spatial * 2;
        for (u32 j = 0; j < bd; j++) rs1[j] = mx->view.shape.dims[j];
        for (u32 j = 0; j < n_spatial; j++) {
            rs1[bd + j*2] = k_[j];
            rs1[bd + j*2 + 1] = i_[j] * f_[j] + 1;
        }
        t = thvm_reshape(ctx, t, shape_of(rs1, rs1_rank));

        // shrink to [batch, k0, o0*s0, k1, o1*s1]
        u32 sh2[MAX_DIM * 2];
        for (u32 j = 0; j < bd; j++) { sh2[j*2] = 0; sh2[j*2+1] = mx->view.shape.dims[j]; }
        for (u32 j = 0; j < n_spatial; j++) {
            sh2[(bd + j*2)*2] = 0;
            sh2[(bd + j*2)*2 + 1] = k_[j];
            sh2[(bd + j*2+1)*2] = 0;
            sh2[(bd + j*2+1)*2 + 1] = o_[j] * s_[j];
        }
        t = thvm_shrink(ctx, t, sh2, rs1_rank);

        // reshape to [batch, k0, o0, s0, k1, o1, s1]
        u32 rs2[MAX_DIM], rs2_rank = bd + n_spatial * 3;
        for (u32 j = 0; j < bd; j++) rs2[j] = mx->view.shape.dims[j];
        for (u32 j = 0; j < n_spatial; j++) {
            rs2[bd + j*3] = k_[j];
            rs2[bd + j*3 + 1] = o_[j];
            rs2[bd + j*3 + 2] = s_[j];
        }
        t = thvm_reshape(ctx, t, shape_of(rs2, rs2_rank));

        // shrink stride dim to 1: [batch, k0, o0, 1, k1, o1, 1]
        u32 sh3[MAX_DIM * 2];
        for (u32 j = 0; j < rs2_rank; j++) { sh3[j*2] = 0; sh3[j*2+1] = rs2[j]; }
        for (u32 j = 0; j < n_spatial; j++) {
            u32 si = bd + j*3 + 2;
            sh3[si*2+1] = 1;
        }
        t = thvm_shrink(ctx, t, sh3, rs2_rank);

        // reshape to [batch, k0, o0, k1, o1]
        u32 rs3[MAX_DIM], rs3_rank = bd + n_spatial * 2;
        for (u32 j = 0; j < bd; j++) rs3[j] = mx->view.shape.dims[j];
        for (u32 j = 0; j < n_spatial; j++) {
            rs3[bd + j*2] = k_[j];
            rs3[bd + j*2+1] = o_[j];
        }
        t = thvm_reshape(ctx, t, shape_of(rs3, rs3_rank));

        // permute to [batch, o0, o1, k0, k1]
        u32 perm[MAX_DIM], pi = 0;
        for (u32 j = 0; j < bd; j++) perm[pi++] = j;
        for (u32 j = 0; j < n_spatial; j++) perm[pi++] = bd + j*2 + 1;  // o dims
        for (u32 j = 0; j < n_spatial; j++) perm[pi++] = bd + j*2;      // k dims
        t = thvm_permute(ctx, t, perm, rs3_rank);

        return t;
    }

    // Simple path: k <= s (e.g., maxpool 2x2/2)
    // pad → shrink → reshape → shrink → permute

    // Step 1: pad to make divisible, then shrink to o*s
    u32 pad_pairs[MAX_DIM * 2];
    memset(pad_pairs, 0, sizeof(pad_pairs));
    int need_pad = 0;
    for (u32 j = 0; j < n_spatial; j++) {
        u32 pad_amt = (o_[j] * s_[j] > i_[j]) ? (o_[j] * s_[j] - i_[j]) : 0;
        pad_pairs[(bd + j) * 2 + 1] = pad_amt;
        if (pad_amt > 0) need_pad = 1;
    }
    if (need_pad) t = thvm_pad(ctx, t, pad_pairs, ndim);

    // Shrink to [batch..., o*s]
    u32 shrink_pairs[MAX_DIM * 2];
    for (u32 j = 0; j < ndim; j++) {
        shrink_pairs[j * 2] = 0;
        shrink_pairs[j * 2 + 1] = (j >= bd) ? o_[j - bd] * s_[j - bd] : mx->view.shape.dims[j];
    }
    t = thvm_shrink(ctx, t, shrink_pairs, ndim);

    // Step 2: reshape to [batch..., o0, s0, o1, s1, ...]
    u32 rs_dims[MAX_DIM], rs_rank = bd + n_spatial * 2;
    for (u32 j = 0; j < bd; j++) rs_dims[j] = mx->view.shape.dims[j];
    for (u32 j = 0; j < n_spatial; j++) {
        rs_dims[bd + j * 2]     = o_[j];
        rs_dims[bd + j * 2 + 1] = s_[j];
    }
    t = thvm_reshape(ctx, t, shape_of(rs_dims, rs_rank));

    // Step 3: shrink (o, k) from (o, s)
    u32 shrink2[MAX_DIM * 2];
    for (u32 j = 0; j < rs_rank; j++) {
        shrink2[j * 2] = 0;
        shrink2[j * 2 + 1] = rs_dims[j];
    }
    for (u32 j = 0; j < n_spatial; j++) {
        u32 dim_idx = bd + j * 2 + 1;
        shrink2[dim_idx * 2 + 1] = k_[j];
    }
    t = thvm_shrink(ctx, t, shrink2, rs_rank);

    // Step 4: permute to [batch..., o0, o1, ..., k0, k1, ...]
    u32 perm[MAX_DIM], pi = 0;
    for (u32 j = 0; j < bd; j++) perm[pi++] = j;
    for (u32 j = 0; j < n_spatial; j++) perm[pi++] = bd + j * 2;
    for (u32 j = 0; j < n_spatial; j++) perm[pi++] = bd + j * 2 + 1;
    t = thvm_permute(ctx, t, perm, rs_rank);

    return t;
}

// ============================================================
// Conv2d as UOp composition — matches tinygrad tensor.py:2476-2484
// ============================================================

Term thvm_conv2d(TinyHVM *ctx, Term x, Term w, Term bias,
                 u32 groups, const u32 *stride_, const u32 *padding_) {
    // x: [BS, Cin, H, W], w: [Cout, Cin/groups, KH, KW], bias: [Cout] or NULL
    Term xr = thvm_reduce(ctx, x);
    Term wr = thvm_reduce(ctx, w);
    TensorMeta *mx = &ctx->tensors[(u32)term_val(xr)];
    TensorMeta *mw = &ctx->tensors[(u32)term_val(wr)];

    u32 bs = mx->view.shape.dims[0];
    u32 cin = mx->view.shape.dims[1];
    u32 cout = mw->view.shape.dims[0];
    u32 cin_g = mw->view.shape.dims[1];  // cin/groups
    u32 KH = mw->view.shape.dims[2];
    u32 KW = mw->view.shape.dims[3];
    (void)cin_g;
    assert(groups * cin_g == cin);

    // Step 1: pad input
    u32 pad_pairs[MAX_DIM * 2] = {0};
    pad_pairs[2*2] = padding_[0]; pad_pairs[2*2+1] = padding_[1];  // H before/after
    pad_pairs[3*2] = padding_[2]; pad_pairs[3*2+1] = padding_[3];  // W before/after
    Term padded = xr;
    if (padding_[0] || padding_[1] || padding_[2] || padding_[3]) {
        padded = thvm_pad(ctx, xr, pad_pairs, 4);
    }

    // Step 2: _pool to create sliding windows
    u32 k[] = {KH, KW};
    u32 s[] = {stride_[0], stride_[1]};
    // Compute output spatial dims algebraically — no thvm_reduce needed.
    // input after padding: H + pad[0]+pad[1], W + pad[2]+pad[3]
    u32 IH = mx->view.shape.dims[2] + padding_[0] + padding_[1];
    u32 IW = mx->view.shape.dims[3] + padding_[2] + padding_[3];
    u32 oy = (IH - KH) / stride_[0] + 1;
    u32 ox = (IW - KW) / stride_[1] + 1;
    Term pooled = thvm_pool(ctx, padded, k, s, 2);
    // pooled: [BS, Cin, OY, OX, KH, KW]
    Term pr = pooled;  // lazy — shape is now known algebraically

    u32 rcout = cout / groups;

    // Step 3: reshape + expand + permute for broadcasting
    // pooled: [BS, groups, cin_g, 1, OY, OX, KH, KW]
    Term x_rs = thvm_reshape(ctx, pr,
        shape_of((u32[]){bs, groups, cin_g, 1, oy, ox, KH, KW}, 8));
    // expand to: [BS, groups, cin_g, rcout, OY, OX, KH, KW]
    Term x_exp = thvm_expand(ctx, x_rs,
        shape_of((u32[]){bs, groups, cin_g, rcout, oy, ox, KH, KW}, 8));
    // permute to: [BS, groups, rcout, OY, OX, cin_g, KH, KW]
    u32 conv_perm[] = {0, 1, 3, 4, 5, 2, 6, 7};
    Term x_perm = thvm_permute(ctx, x_exp, conv_perm, 8);

    // Step 4: reshape weight to [1, groups, rcout, 1, 1, cin_g, KH, KW]
    Term w_rs = thvm_reshape(ctx, wr,
        shape_of((u32[]){1, groups, rcout, 1, 1, cin_g, KH, KW}, 8));

    // prod: [BS, groups, rcout, OY, OX, cin_g, KH, KW]
    Term prod = thvm_op(ctx, UOP_MUL, x_perm, w_rs);
    // Fused multi-axis SUM(MUL) — reduce axes 5,6,7 (cin_g, KH, KW)
    u32 reduce_axes[] = {5, 6, 7};
    Term summed = thvm_sum_axes(ctx, prod, reduce_axes, 3);
    // → [BS, groups, rcout, OY, OX, 1, 1, 1]
    Term out = thvm_reshape(ctx, summed, shape_of((u32[]){bs, cout, oy, ox}, 4));

    // Add bias
    if (term_tag(bias) != TAG_ERA) {
        Term b_rs = thvm_reshape(ctx, bias, shape_of((u32[]){1, cout, 1, 1}, 4));
        out = thvm_op(ctx, UOP_ADD, out, b_rs);
    }

    return out;
}

// ============================================================
// MaxPool2d as UOp composition — tinygrad tensor.py:2404-2405
// ============================================================

Term thvm_maxpool2d(TinyHVM *ctx, Term x, const u32 *kernel, const u32 *stride_) {
    u32 k[] = {kernel[0], kernel[1]};
    u32 s[] = {stride_[0], stride_[1]};

    // Compute output spatial dims from input shape
    Term pr = x;
    if (term_tag(x) != TAG_TEN) pr = thvm_reduce(ctx, x);
    TensorMeta *mp = &ctx->tensors[(u32)term_val(pr)];
    u32 bs = mp->view.shape.dims[0];
    u32 c  = mp->view.shape.dims[1];
    u32 IH = mp->view.shape.dims[2];
    u32 IW = mp->view.shape.dims[3];
    u32 kh = kernel[0], kw2 = kernel[1];
    u32 oy = (IH - kh) / stride_[0] + 1;
    u32 ox = (IW - kw2) / stride_[1] + 1;

    // Single pool call — pooled shape: [BS, C, OY, OX, KH, KW]
    Term pool_t = thvm_pool(ctx, pr, k, s, 2);

    Term r1 = thvm_op(ctx, UOP_RMAX, pool_t, term_era());
    // → [BS, C, OY, OX, KH, 1], squeeze:
    r1 = thvm_reshape(ctx, r1, shape_of((u32[]){bs, c, oy, ox, kh}, 5));

    Term r2 = thvm_op(ctx, UOP_RMAX, r1, term_era());
    // → [BS, C, OY, OX, 1], squeeze:
    (void)kw2;
    return thvm_reshape(ctx, r2, shape_of((u32[]){bs, c, oy, ox}, 4));
}

