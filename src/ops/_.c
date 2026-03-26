void thvm_realize(TinyHVM *ctx, Term t) {
    thvm_reduce(ctx, t);
}

f32 *thvm_to_host(TinyHVM *ctx, Term t) {
    t = thvm_reduce(ctx, t);
    if (term_tag(t) != TAG_TEN) return NULL;
    u32 id = (u32)term_val(t);
    TensorMeta *m = &ctx->tensors[id];

    if (m->view.contiguous) {
        // Contiguous: direct read
        if (!m->host_ptr) m->host_ptr = malloc((size_t)m->view.numel * dtype_size(m->dtype));
        if (ctx->backend) ctx->backend->buf_read(m->buf_id, m->host_ptr,
                                                  (u64)m->view.numel * dtype_size(m->dtype));
        return (f32 *)m->host_ptr;
    }

    // Non-contiguous (e.g. expand with stride=0): need strided copy
    // Read the underlying buffer (may be smaller than numel)
    // Sum all dimension extents (not max!) to get total buffer span
    u32 src_numel = (m->view.offset > 0) ? (u32)m->view.offset : 0;
    for (u32 d = 0; d < m->view.shape.rank; d++) {
        i32 stride = m->view.strides[d];
        if (stride > 0)
            src_numel += (m->view.shape.dims[d] - 1) * (u32)stride;
    }
    src_numel += 1;
    if (src_numel == 0) src_numel = 1;

    f32 *src_buf = malloc((size_t)src_numel * sizeof(f32));
    if (ctx->backend) ctx->backend->buf_read(m->buf_id, src_buf,
                                              (u64)src_numel * sizeof(f32));

    if (!m->host_ptr) m->host_ptr = malloc((size_t)m->view.numel * sizeof(f32));
    f32 *dst = (f32 *)m->host_ptr;

    // Strided copy (with mask support)
    for (u32 flat = 0; flat < m->view.numel; flat++) {
        u32 rem = flat;
        i32 src_idx_s = m->view.offset;
        int masked_out = 0;
        for (i32 d = (i32)m->view.shape.rank - 1; d >= 0; d--) {
            u32 coord = rem % m->view.shape.dims[d];
            rem /= m->view.shape.dims[d];
            if (m->view.has_mask && (coord < m->view.mask_begin[d] || coord >= m->view.mask_end[d]))
                masked_out = 1;
            src_idx_s += (i32)coord * (m->view.strides[d] > 0 ? m->view.strides[d] : 0);
        }
        dst[flat] = masked_out ? 0.0f : src_buf[(u32)src_idx_s];
    }
    free(src_buf);
    return dst;
}

// ============================================================
// autograd.c — Backward pass, gradient descent
// ============================================================

// Create a tensor filled with a constant
static u32 tensor_fill(TinyHVM *ctx, Shape s, f32 val) {
    u32 id = tensor_create(ctx, s, DTYPE_F32);
    TensorMeta *m = &ctx->tensors[id];
    u32 n = m->view.numel;
    f32 *tmp = malloc(n * sizeof(f32));
    for (u32 i = 0; i < n; i++) tmp[i] = val;
    if (ctx->backend) ctx->backend->buf_write(m->buf_id, tmp, (u64)n * dtype_size(DTYPE_F32));
    free(tmp);
    return id;
}

// Transpose 2D: just swap axes via permute (zero-copy, stride swap)
static u32 tensor_transpose_2d(TinyHVM *ctx, u32 src_id) {
    u32 axes[] = {1, 0};
    Term t = thvm_permute(ctx, term_ten(src_id, ctx->tensors[src_id].dtype), axes, 2);
    t = thvm_reduce(ctx, t);
    return (u32)term_val(t);
}



// Sum-to-shape: reduce broadcast dimensions to match target shape.
// Returns lazy Term (TAG_TOP). No thvm_reduce calls.
static Term sum_to_shape(TinyHVM *ctx, Term grad, Shape src_shape, Shape target) {
    // Check if shapes already match
    if (src_shape.rank == target.rank) {
        int same = 1;
        for (u32 i = 0; i < target.rank; i++)
            if (src_shape.dims[i] != target.dims[i]) same = 0;
        if (same) return grad;
    }

    u32 reduce_axes[MAX_DIM];
    u32 n_reduce = 0;

    if (src_shape.rank == target.rank) {
        // Same rank: sum over axes where src is larger than target
        for (u32 d = 0; d < target.rank; d++) {
            if (src_shape.dims[d] != target.dims[d])
                reduce_axes[n_reduce++] = d;
        }
    } else if (src_shape.rank > target.rank) {
        // src has more dims: sum leading dims first
        u32 n_leading = src_shape.rank - target.rank;
        for (u32 d = 0; d < n_leading; d++)
            reduce_axes[n_reduce++] = d;
        for (u32 d = 0; d < target.rank; d++) {
            if (src_shape.dims[n_leading + d] != target.dims[d])
                reduce_axes[n_reduce++] = n_leading + d;
        }
    } else {
        // src.rank < target.rank: src was broadcast from a scalar/reduced form
        // This happens when gy is a keepdims-reduced shape like [1,1] but
        // the target has more dims. Just reshape src to target if numel matches or is 1.
        // Generally this shouldn't happen in correct backward — but handle gracefully.
        printf("sum_to_shape: src.rank=%u < target.rank=%u src=[", src_shape.rank, target.rank);
        for (u32 d=0;d<src_shape.rank;d++) printf("%u,",src_shape.dims[d]);
        printf("] target=[");
        for (u32 d=0;d<target.rank;d++) printf("%u,",target.dims[d]);
        printf("]\n");
        // Reshape grad to target shape (broadcast; assumes numel(src) == 1 or numel matches)
        if (target.rank > 0) {
            grad = thvm_reshape(ctx, grad, target);
        }
        return grad;
    }

    if (n_reduce > 0) {
        grad = thvm_sum_axes(ctx, grad, reduce_axes, n_reduce);
        grad = thvm_reshape(ctx, grad, target);
    }
    return grad;
}

void thvm_set_requires_grad(TinyHVM *ctx, Term t) {
    if (term_tag(t) == TAG_TEN) {
        ctx->tensors[(u32)term_val(t)].requires_grad = 1;
    }
}





// ============================================================
// Eval helpers — argmax + accuracy
// ============================================================

Term thvm_argmax(TinyHVM *ctx, Term x, u32 rows, u32 cols) {
    x = thvm_reduce(ctx, x);
    f32 *data = thvm_to_host(ctx, x);
    u32 *preds = malloc(rows * sizeof(u32));
    for (u32 i = 0; i < rows; i++) {
        u32 best = 0; f32 mv = data[i * cols];
        for (u32 j = 1; j < cols; j++)
            if (data[i * cols + j] > mv) { mv = data[i * cols + j]; best = j; }
        preds[i] = best;
    }
    u32 id = ctx->tensor_count++;
    u32 buf = ctx->backend->buf_alloc(rows * sizeof(u32));
    ctx->tensors[id] = (TensorMeta){ .buf_id = buf, .dtype = DTYPE_U32, .view = view_create(SHAPE(rows)) };
    ctx->backend->buf_write(buf, preds, rows * sizeof(u32));
    free(preds);
    return term_ten(id, DTYPE_U32);
}

f32 thvm_eval_accuracy(TinyHVM *ctx, Term logits, const u8 *labels, u32 n_samples, u32 n_classes) {
    logits = thvm_reduce(ctx, logits);
    f32 *data = thvm_to_host(ctx, logits);
    u32 correct = 0;
    for (u32 i = 0; i < n_samples; i++) {
        u32 best = 0; f32 mv = data[i * n_classes];
        for (u32 j = 1; j < n_classes; j++)
            if (data[i * n_classes + j] > mv) { mv = data[i * n_classes + j]; best = j; }
        if (best == labels[i]) correct++;
    }
    return 100.0f * (f32)correct / (f32)n_samples;
}

void thvm_profile_report(TinyHVM *ctx) {
    if (ctx->backend && ctx->backend->profile_report) ctx->backend->profile_report();
}
void thvm_profile_reset(TinyHVM *ctx) {
    if (ctx->backend && ctx->backend->profile_reset) ctx->backend->profile_reset();
}

