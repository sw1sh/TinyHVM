void thvm_realize(TinyHVM *ctx, Term t) {
    thvm_reduce(ctx, t);
}

static int thvm_is_view_top(Term t) {
    return term_tag(t) == TAG_TOP &&
           term_ext(t) >= UOP_RESHAPE &&
           term_ext(t) <= UOP_PAD;
}

static Term thvm_force_view_chain(TinyHVM *ctx, Term t) {
    for (u32 depth = 0; depth < 32; depth++) {
        if (!thvm_is_view_top(t)) return t;
        u64 loc = term_val(t);
        Term a = thvm_reduce(ctx, heap_read(ctx, loc + 0));
        if (thvm_is_view_top(a)) a = thvm_force_view_chain(ctx, a);
        heap_set(ctx, loc + 0, a);

        Term b = thvm_reduce(ctx, heap_read(ctx, loc + 1));
        if (thvm_is_view_top(b)) b = thvm_force_view_chain(ctx, b);
        heap_set(ctx, loc + 1, b);

        Term r = thvm_interact(ctx, t);
        if (r == t) return t;
        t = thvm_reduce(ctx, r);
    }
    return t;
}

void *thvm_to_host_raw(TinyHVM *ctx, Term t, u32 *out_dtype, Shape *out_shape) {
    t = thvm_reduce(ctx, t);
    if (term_tag(t) != TAG_TEN) {
        // After scheduler: TAG_TOPs may need dispatch_mode to resolve
        ctx->dispatch_mode = 1;
        t = thvm_reduce(ctx, t);
        ctx->dispatch_mode = 0;
    }
    if (thvm_is_view_top(t))
        t = thvm_force_view_chain(ctx, t);
    if (term_tag(t) != TAG_TEN) return NULL;
    u32 id = (u32)term_val(t);
    ENSURE(ctx, id);
    TensorMeta *m = &ctx->tensors[id];
    u32 elem_bytes = dtype_size(m->dtype);
    if (out_dtype) *out_dtype = m->dtype;
    if (out_shape) *out_shape = m->view.shape;

    if (m->view.contiguous) {
        // Contiguous: direct read
        if (!m->host_ptr) m->host_ptr = malloc((size_t)m->view.numel * elem_bytes);
        if (m->backend) m->backend->buf_read(m->buf_id, m->host_ptr,
                                                  (u64)m->view.numel * elem_bytes);
        return m->host_ptr;
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

    u8 *src_buf = malloc((size_t)src_numel * elem_bytes);
    if (m->backend) m->backend->buf_read(m->buf_id, src_buf,
                                              (u64)src_numel * elem_bytes);

    if (!m->host_ptr) m->host_ptr = malloc((size_t)m->view.numel * elem_bytes);
    u8 *dst = (u8 *)m->host_ptr;

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
        if (masked_out) memset(dst + (size_t)flat * elem_bytes, 0, elem_bytes);
        else memcpy(dst + (size_t)flat * elem_bytes,
                    src_buf + (size_t)(u32)src_idx_s * elem_bytes, elem_bytes);
    }
    free(src_buf);
    return dst;
}

f32 *thvm_to_host(TinyHVM *ctx, Term t) {
    u32 dtype = DTYPE_F32;
    void *raw = thvm_to_host_raw(ctx, t, &dtype, NULL);
    if (!raw || dtype != DTYPE_F32) return NULL;
    return (f32 *)raw;
}

i32 *thvm_to_host_i32(TinyHVM *ctx, Term t) {
    u32 dtype = DTYPE_I32;
    void *raw = thvm_to_host_raw(ctx, t, &dtype, NULL);
    if (!raw || dtype != DTYPE_I32) return NULL;
    return (i32 *)raw;
}

u32 *thvm_to_host_u32(TinyHVM *ctx, Term t) {
    u32 dtype = DTYPE_U32;
    void *raw = thvm_to_host_raw(ctx, t, &dtype, NULL);
    if (!raw || dtype != DTYPE_U32) return NULL;
    return (u32 *)raw;
}

// Lazy device transfer — creates a UOP_TODEVICE node, realized at reduce time
Term thvm_to_device(TinyHVM *ctx, Term t, u32 device_idx) {
    Term dev = thvm_scalar_u32(ctx, device_idx);
    return thvm_op(ctx, UOP_TODEVICE, t, dev);
}

// ============================================================
// autograd.c — Backward pass, gradient descent
// ============================================================

// Create a tensor filled with a constant
static u32 tensor_fill_typed(TinyHVM *ctx, Shape s, f32 val, u32 dtype) {
    u32 id = tensor_create(ctx, s, dtype);
    TensorMeta *m = &ctx->tensors[id];
    u32 n = m->view.numel;
    u64 nbytes = (u64)n * dtype_size(dtype);
    void *tmp = malloc((size_t)nbytes);
    for (u32 i = 0; i < n; i++) dtype_store_from_f32(tmp, dtype, i, val);
    if (m->backend) m->backend->buf_write(m->buf_id, tmp, nbytes);
    free(tmp);
    return id;
}

static u32 tensor_fill(TinyHVM *ctx, Shape s, f32 val) {
    return tensor_fill_typed(ctx, s, val, DTYPE_F32);
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
        fprintf(stderr, "sum_to_shape RANK MISMATCH: src.rank=%u < target.rank=%u\n", src_shape.rank, target.rank);
        // Abort to get stack trace
        // assert(0 && "sum_to_shape rank mismatch");
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
    u32 dtype = DTYPE_F32;
    void *data = thvm_to_host_raw(ctx, x, &dtype, NULL);
    if (!data) return term_era();
    u32 *preds = malloc(rows * sizeof(u32));
    for (u32 i = 0; i < rows; i++) {
        u32 best = 0; f32 mv = dtype_load_as_f32(data, dtype, i * cols);
        for (u32 j = 1; j < cols; j++)
            if (dtype_load_as_f32(data, dtype, i * cols + j) > mv) {
                mv = dtype_load_as_f32(data, dtype, i * cols + j);
                best = j;
            }
        preds[i] = best;
    }
    u32 id = ctx->tensor_count++;
    Backend *be = ctx_default_backend(ctx);
    u32 buf = be->buf_alloc(rows * sizeof(u32));
    ctx->tensors[id] = (TensorMeta){ .buf_id = buf, .dtype = DTYPE_U32, .view = view_create(SHAPE(rows)), .backend = be };
    be->buf_write(buf, preds, rows * sizeof(u32));
    free(preds);
    return term_ten(id, DTYPE_U32);
}

f32 thvm_eval_accuracy(TinyHVM *ctx, Term logits, const u8 *labels, u32 n_samples, u32 n_classes) {
    logits = thvm_reduce(ctx, logits);
    u32 dtype = DTYPE_F32;
    void *data = thvm_to_host_raw(ctx, logits, &dtype, NULL);
    if (!data) return 0.0f;
    u32 correct = 0;
    for (u32 i = 0; i < n_samples; i++) {
        u32 best = 0; f32 mv = dtype_load_as_f32(data, dtype, i * n_classes);
        for (u32 j = 1; j < n_classes; j++)
            if (dtype_load_as_f32(data, dtype, i * n_classes + j) > mv) {
                mv = dtype_load_as_f32(data, dtype, i * n_classes + j);
                best = j;
            }
        if (best == labels[i]) correct++;
    }
    return 100.0f * (f32)correct / (f32)n_samples;
}

void thvm_profile_report(TinyHVM *ctx) {
    for (u32 i = 0; i < ctx->n_backends; i++)
        if (ctx->backends[i] && ctx->backends[i]->profile_report) ctx->backends[i]->profile_report();
}
void thvm_profile_reset(TinyHVM *ctx) {
    for (u32 i = 0; i < ctx->n_backends; i++)
        if (ctx->backends[i] && ctx->backends[i]->profile_reset) ctx->backends[i]->profile_reset();
}

// ============================================================
// Interaction tracing
// ============================================================

void thvm_trace_enable(TinyHVM *ctx, int enabled) { ctx->trace_enabled = (u8)enabled; }
void thvm_trace_clear(TinyHVM *ctx) { ctx->trace_count = 0; }
