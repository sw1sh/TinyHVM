void thvm_realize(TinyHVM *ctx, Term t) {
    thvm_reduce(ctx, t);
}

static Term thvm_force_tensor_term(TinyHVM *ctx, Term t);
static int thvm_kernel_to_compute(TinyHVM *ctx, Term t, Term *out_compute, u32 depth);

static int thvm_is_view_top(Term t) {
    return term_tag(t) == TAG_TOP &&
           term_ext(t) >= UOP_RESHAPE &&
           term_ext(t) <= UOP_PAD;
}

static Term thvm_force_view_chain(TinyHVM *ctx, Term t) {
    for (u32 depth = 0; depth < 32; depth++) {
        if (!thvm_is_view_top(t)) return t;
        u64 loc = term_val(t);
        Term a = heap_read(ctx, loc + 0);
        if (term_tag(a) == TAG_TOP || term_tag(a) == TAG_SEQ) {
            a = thvm_force_tensor_term(ctx, a);
        } else {
            a = thvm_reduce(ctx, a);
        }
        if (thvm_is_view_top(a)) a = thvm_force_view_chain(ctx, a);
        heap_set(ctx, loc + 0, a);

        Term b = heap_read(ctx, loc + 1);
        if (term_tag(b) == TAG_TOP || term_tag(b) == TAG_SEQ) {
            b = thvm_force_tensor_term(ctx, b);
        } else {
            b = thvm_reduce(ctx, b);
        }
        if (thvm_is_view_top(b)) b = thvm_force_view_chain(ctx, b);
        heap_set(ctx, loc + 1, b);

        Term r = thvm_interact(ctx, t);
        if (r == t) return t;
        t = thvm_reduce(ctx, r);
    }
    return t;
}

static Term thvm_force_dispatch_kid(TinyHVM *ctx, u32 kid, u32 depth) {
    extern u32 sched_kernel_count;
    extern Term kid_results[];
    extern KernelEntry sched_kernels[];

    if (kid >= sched_kernel_count || depth >= SCHED_MAX_KERNELS)
        return term_era();
    if (term_tag(kid_results[kid]) != TAG_ERA)
        return kid_results[kid];

    KernelEntry *ke = &sched_kernels[kid];
    for (u32 di = 0; di < ke->n_deps; di++) {
        u32 dep_kid = ke->dep_kids[di];
        Term dep = thvm_force_dispatch_kid(ctx, dep_kid, depth + 1);
        if (dep_kid < sched_kernel_count && term_tag(dep) == TAG_ERA && term_val(dep) == 0)
            return dep;
    }
    return thvm_sched_dispatch_kernel(ctx, kid);
}

static Term thvm_force_kernel_term(TinyHVM *ctx, Term t) {
    if (term_tag(t) != TAG_TOP || term_ext(t) != UOP_KERNEL) return t;

    u32 kid = 0;
    int saved_dispatch = ctx->dispatch_enabled;
    ctx->dispatch_enabled = 1;
    int registered = thvm_kernel_register(ctx, t, &kid);
    ctx->dispatch_enabled = saved_dispatch;
    if (getenv("THVM_LOOP_DIAG")) {
        extern u32 sched_kernel_count;
        fprintf(stderr, "FORCE_TENSOR register kernel=%llu ok=%d kid=%u sched_kernels=%u\n",
                (unsigned long long)term_val(t), registered, kid, sched_kernel_count);
    }
    if (registered) {
        Term dispatched = thvm_force_dispatch_kid(ctx, kid, 0);
        if (getenv("THVM_LOOP_DIAG")) {
            fprintf(stderr, "FORCE_TENSOR after_direct_dispatch tag=%u ext=%u val=%llu kid=%u\n",
                    (u32)term_tag(dispatched), (u32)term_ext(dispatched),
                    (unsigned long long)term_val(dispatched), kid);
        }
        if (term_tag(dispatched) == TAG_TEN) return dispatched;
        return t;
    }

    Term compute = term_era();
    if (thvm_kernel_to_compute(ctx, t, &compute, 0) && thvm_is_view_top(compute)) {
        Term forced = thvm_force_view_chain(ctx, compute);
        if (getenv("THVM_LOOP_DIAG")) {
            fprintf(stderr,
                    "FORCE_TENSOR view_kernel_direct tag=%u ext=%u val=%llu\n",
                    (u32)term_tag(forced), (u32)term_ext(forced),
                    (unsigned long long)term_val(forced));
        }
        return forced;
    }
    return t;
}

static Term thvm_force_tensor_term(TinyHVM *ctx, Term t) {
    if (getenv("THVM_LOOP_DIAG")) {
        fprintf(stderr, "FORCE_TENSOR in tag=%u ext=%u val=%llu\n",
                (u32)term_tag(t), (u32)term_ext(t), (unsigned long long)term_val(t));
    }
    if (term_tag(t) == TAG_TOP && term_ext(t) == UOP_DETACH) {
        u64 loc = term_val(t);
        if (loc != 0 && loc + 1 < ctx->heap_pos) {
            Term memo = heap_read(ctx, loc + 1);
            if (term_tag(memo) == TAG_TEN) return memo;
        }
    }
    t = thvm_reduce(ctx, t);
    if (getenv("THVM_LOOP_DIAG")) {
        fprintf(stderr, "FORCE_TENSOR after_reduce tag=%u ext=%u val=%llu\n",
                (u32)term_tag(t), (u32)term_ext(t), (unsigned long long)term_val(t));
        if (term_tag(t) == TAG_TOP && term_ext(t) == UOP_KERNEL) {
            u64 loc = term_val(t);
            if (loc != 0 && loc + 2 < ctx->heap_pos) {
                Term a = heap_read(ctx, loc + 0);
                Term b = heap_read(ctx, loc + 1);
                Term c = heap_read(ctx, loc + 2);
                fprintf(stderr,
                        "FORCE_TENSOR kernel_slots a=(tag=%u ext=%u val=%llu) b=(tag=%u ext=%u val=%llu) c=(tag=%u ext=%u val=%llu)\n",
                        (u32)term_tag(a), (u32)term_ext(a), (unsigned long long)term_val(a),
                        (u32)term_tag(b), (u32)term_ext(b), (unsigned long long)term_val(b),
                        (u32)term_tag(c), (u32)term_ext(c), (unsigned long long)term_val(c));
            }
        }
        if (term_tag(t) == TAG_TOP && term_ext(t) == UOP_EXEC) {
            u64 loc = term_val(t);
            if (loc != 0 && loc + 2 < ctx->heap_pos) {
                Term kid = heap_read(ctx, loc + 0);
                Term deps = heap_read(ctx, loc + 1);
                Term flags = heap_read(ctx, loc + 2);
                fprintf(stderr,
                        "FORCE_TENSOR exec_slots kid=(tag=%u ext=%u val=%llu) deps=(tag=%u ext=%u val=%llu) flags=(tag=%u ext=%u val=%llu)\n",
                        (u32)term_tag(kid), (u32)term_ext(kid), (unsigned long long)term_val(kid),
                        (u32)term_tag(deps), (u32)term_ext(deps), (unsigned long long)term_val(deps),
                        (u32)term_tag(flags), (u32)term_ext(flags), (unsigned long long)term_val(flags));
                if (term_tag(deps) == TAG_CTR) {
                    u64 dloc = term_val(deps);
                    if (dloc != 0 && dloc + 1 < ctx->heap_pos) {
                        Term d0 = heap_read(ctx, dloc + 0);
                        Term d1 = heap_read(ctx, dloc + 1);
                        fprintf(stderr,
                                "FORCE_TENSOR exec_deps d0=(tag=%u ext=%u val=%llu) d1=(tag=%u ext=%u val=%llu)\n",
                                (u32)term_tag(d0), (u32)term_ext(d0), (unsigned long long)term_val(d0),
                                (u32)term_tag(d1), (u32)term_ext(d1), (unsigned long long)term_val(d1));
                    }
                }
            }
        }
        if (term_tag(t) == TAG_SEQ) {
            u64 loc = term_val(t);
            if (loc != 0 && loc + 1 < ctx->heap_pos) {
                Term a = heap_read(ctx, loc + 0);
                Term b = heap_read(ctx, loc + 1);
                fprintf(stderr,
                        "FORCE_TENSOR seq_slots a=(tag=%u ext=%u val=%llu) b=(tag=%u ext=%u val=%llu)\n",
                        (u32)term_tag(a), (u32)term_ext(a), (unsigned long long)term_val(a),
                        (u32)term_tag(b), (u32)term_ext(b), (unsigned long long)term_val(b));
                if (term_tag(a) == TAG_TOP && term_ext(a) == UOP_EXEC) {
                    u64 aloc = term_val(a);
                    if (aloc != 0 && aloc + 2 < ctx->heap_pos) {
                        Term ak = heap_read(ctx, aloc + 0);
                        Term ad = heap_read(ctx, aloc + 1);
                        fprintf(stderr,
                                "FORCE_TENSOR seq_a_exec kid=(tag=%u ext=%u val=%llu) deps=(tag=%u ext=%u val=%llu)\n",
                                (u32)term_tag(ak), (u32)term_ext(ak), (unsigned long long)term_val(ak),
                                (u32)term_tag(ad), (u32)term_ext(ad), (unsigned long long)term_val(ad));
                        if (term_tag(ad) == TAG_CTR && term_val(ad) + 1 < ctx->heap_pos) {
                            Term ad0 = heap_read(ctx, term_val(ad) + 0);
                            Term ad1 = heap_read(ctx, term_val(ad) + 1);
                            fprintf(stderr,
                                    "FORCE_TENSOR seq_a_exec_deps d0=(tag=%u ext=%u val=%llu) d1=(tag=%u ext=%u val=%llu)\n",
                                    (u32)term_tag(ad0), (u32)term_ext(ad0), (unsigned long long)term_val(ad0),
                                    (u32)term_tag(ad1), (u32)term_ext(ad1), (unsigned long long)term_val(ad1));
                        }
                    }
                }
                if (term_tag(b) == TAG_TOP && term_ext(b) == UOP_EXEC) {
                    u64 bloc = term_val(b);
                    if (bloc != 0 && bloc + 2 < ctx->heap_pos) {
                        Term bk = heap_read(ctx, bloc + 0);
                        Term bd = heap_read(ctx, bloc + 1);
                        fprintf(stderr,
                                "FORCE_TENSOR seq_b_exec kid=(tag=%u ext=%u val=%llu) deps=(tag=%u ext=%u val=%llu)\n",
                                (u32)term_tag(bk), (u32)term_ext(bk), (unsigned long long)term_val(bk),
                                (u32)term_tag(bd), (u32)term_ext(bd), (unsigned long long)term_val(bd));
                        if (term_tag(bd) == TAG_CTR && term_val(bd) + 1 < ctx->heap_pos) {
                            Term bd0 = heap_read(ctx, term_val(bd) + 0);
                            Term bd1 = heap_read(ctx, term_val(bd) + 1);
                            fprintf(stderr,
                                    "FORCE_TENSOR seq_b_exec_deps d0=(tag=%u ext=%u val=%llu) d1=(tag=%u ext=%u val=%llu)\n",
                                    (u32)term_tag(bd0), (u32)term_ext(bd0), (unsigned long long)term_val(bd0),
                                    (u32)term_tag(bd1), (u32)term_ext(bd1), (unsigned long long)term_val(bd1));
                        }
                    }
                }
            }
        }
    }
    if (term_tag(t) == TAG_TOP && term_ext(t) == UOP_KERNEL)
        t = thvm_force_kernel_term(ctx, t);
    if (term_tag(t) == TAG_TOP && term_ext(t) == UOP_ASSIGN) {
        u64 loc = term_val(t);
        if (loc != 0 && loc + 1 < ctx->heap_pos) {
            Term dst = heap_read(ctx, loc + 0);
            Term src = heap_read(ctx, loc + 1);
            Term forced_src = thvm_force_tensor_term(ctx, src);
            if (term_tag(forced_src) == TAG_TEN) {
                if (getenv("THVM_LOOP_DIAG") && term_tag(dst) == TAG_TEN) {
                    u32 src_dtype = DTYPE_F32, dst_dtype = DTYPE_F32;
                    f32 *src_host = (f32 *)thvm_to_host_raw(ctx, forced_src, &src_dtype, NULL);
                    f32 *dst_host = (f32 *)thvm_to_host_raw(ctx, dst, &dst_dtype, NULL);
                    fprintf(stderr,
                            "FORCE_TENSOR assign_ready dst=%llu src=%llu src0=%.6f dst0=%.6f\n",
                            (unsigned long long)term_val(dst),
                            (unsigned long long)term_val(forced_src),
                            (src_host && src_dtype == DTYPE_F32) ? src_host[0] : 0.0f,
                            (dst_host && dst_dtype == DTYPE_F32) ? dst_host[0] : 0.0f);
                }
                heap_set(ctx, loc + 1, forced_src);
                int saved_dispatch = ctx->dispatch_enabled;
                ctx->dispatch_enabled = 1;
                Term reduced = thvm_reduce(ctx, t);
                ctx->dispatch_enabled = saved_dispatch;
                if (getenv("THVM_LOOP_DIAG") && term_tag(reduced) == TAG_TEN) {
                    u32 out_dtype = DTYPE_F32;
                    f32 *out_host = (f32 *)thvm_to_host_raw(ctx, reduced, &out_dtype, NULL);
                    fprintf(stderr,
                            "FORCE_TENSOR assign_done out=%llu out0=%.6f\n",
                            (unsigned long long)term_val(reduced),
                            (out_host && out_dtype == DTYPE_F32) ? out_host[0] : 0.0f);
                }
                if (term_tag(reduced) == TAG_TEN)
                    t = reduced;
            }
        }
    }
    if (term_tag(t) == TAG_TOP && term_ext(t) == UOP_KERNEL) {
        Term rewritten = thvm_sched_rewrite_get(ctx, t);
        if (term_tag(rewritten) != TAG_ERA) {
            t = thvm_eval_exec_fixed_point(ctx, rewritten);
            if (getenv("THVM_LOOP_DIAG")) {
                fprintf(stderr, "FORCE_TENSOR after_rewrite_exec tag=%u ext=%u val=%llu\n",
                        (u32)term_tag(t), (u32)term_ext(t), (unsigned long long)term_val(t));
            }
        }
    }
    if (term_tag(t) == TAG_SEQ ||
        (term_tag(t) == TAG_TOP &&
         (term_ext(t) == UOP_EXEC || term_ext(t) == UOP_FUSE))) {
        t = thvm_eval_exec_fixed_point(ctx, t);
        if (getenv("THVM_LOOP_DIAG")) {
            fprintf(stderr, "FORCE_TENSOR after_exec_eval tag=%u ext=%u val=%llu\n",
                    (u32)term_tag(t), (u32)term_ext(t), (unsigned long long)term_val(t));
        }
    } else if (term_tag(t) != TAG_TEN && term_tag(t) == TAG_TOP) {
        // Host materialization must honor the same staged pipeline as normal
        // eval. Mixed fuse+dispatch is enough for legacy cases, but staged keep
        // bundles can require global passes before the subtree becomes an
        // executable tensor term.
        t = thvm_eval(ctx, t);
        if (getenv("THVM_LOOP_DIAG")) {
            fprintf(stderr, "FORCE_TENSOR after_eval tag=%u ext=%u val=%llu\n",
                    (u32)term_tag(t), (u32)term_ext(t), (unsigned long long)term_val(t));
        }
    }
    if (term_tag(t) != TAG_TEN) {
        int saved_dispatch = ctx->dispatch_enabled;
        ctx->dispatch_enabled = 1;
        t = thvm_reduce(ctx, t);
        ctx->dispatch_enabled = saved_dispatch;
        if (getenv("THVM_LOOP_DIAG")) {
            fprintf(stderr, "FORCE_TENSOR after_dispatch_reduce tag=%u ext=%u val=%llu\n",
                    (u32)term_tag(t), (u32)term_ext(t), (unsigned long long)term_val(t));
        }
    }
    if (term_tag(t) == TAG_TOP && term_ext(t) == UOP_KERNEL)
        t = thvm_force_kernel_term(ctx, t);
    if (term_tag(t) == TAG_TOP && term_ext(t) == UOP_KERNEL) {
        extern Term kid_results[];
        extern u32 sched_kernel_count;
        extern u64 sched_kernel_locs[];
        u64 kloc = term_val(t);
        for (u32 kid = 0; kid < sched_kernel_count; kid++) {
            if (sched_kernel_locs[kid] == kloc && term_tag(kid_results[kid]) != TAG_ERA) {
                t = kid_results[kid];
                break;
            }
        }
    }
    if (thvm_is_view_top(t))
        t = thvm_force_view_chain(ctx, t);
    if (getenv("THVM_LOOP_DIAG")) {
        fprintf(stderr, "FORCE_TENSOR out tag=%u ext=%u val=%llu\n",
                (u32)term_tag(t), (u32)term_ext(t), (unsigned long long)term_val(t));
    }
    return t;
}

void *thvm_to_host_raw(TinyHVM *ctx, Term t, u32 *out_dtype, Shape *out_shape) {
    t = thvm_force_tensor_term(ctx, t);
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
        // src.rank < target.rank: reinsert singleton axes where needed so the
        // source dims line up with the target dims in order. This preserves
        // missing reduced axes in the middle, e.g. [2,4] -> [2,1,4].
        fprintf(stderr, "sum_to_shape RANK MISMATCH: src.rank=%u < target.rank=%u\n", src_shape.rank, target.rank);
        printf("sum_to_shape: src.rank=%u < target.rank=%u src=[", src_shape.rank, target.rank);
        for (u32 d=0;d<src_shape.rank;d++) printf("%u,",src_shape.dims[d]);
        printf("] target=[");
        for (u32 d=0;d<target.rank;d++) printf("%u,",target.dims[d]);
        printf("]\n");

        Shape padded = {.rank = target.rank};
        u32 s = 0;
        for (u32 d = 0; d < target.rank; d++) {
            if (s < src_shape.rank &&
                (src_shape.dims[s] == target.dims[d] || src_shape.dims[s] == 1)) {
                padded.dims[d] = src_shape.dims[s++];
            } else {
                padded.dims[d] = 1;
            }
        }
        if (s != src_shape.rank) {
            u32 pad = target.rank - src_shape.rank;
            for (u32 d = 0; d < pad; d++) padded.dims[d] = 1;
            for (u32 d = 0; d < src_shape.rank; d++) padded.dims[pad + d] = src_shape.dims[d];
        }

        grad = thvm_reshape(ctx, grad, padded);
        if (target.rank > 0 && !shape_eq(padded, target))
            grad = thvm_expand(ctx, grad, target);
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
