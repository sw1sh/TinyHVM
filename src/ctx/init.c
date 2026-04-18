TinyHVM *thvm_init(const char *default_device) {
    TinyHVM *ctx = calloc(1, sizeof(TinyHVM));
    ctx->tensors = calloc(MAX_TENSORS, sizeof(TensorMeta));
    ctx->heap = calloc(HEAP_CAP, sizeof(Term));
    ctx->book_heap_cap = (1u << 20);
    ctx->book_heap = calloc((size_t)ctx->book_heap_cap, sizeof(Term));
    ctx->book_heap_pos = 1;
    ctx->dup_ports = calloc(DUP_PORT_CAP, sizeof(DupPortEntry));
    ctx->sched_rewrites = calloc(SCHED_REWRITE_CAP, sizeof(SchedRewriteEntry));
    ctx->alo_state_cap = (1u << 16);
    ctx->alo_states = calloc(ctx->alo_state_cap, sizeof(AloState));
    ctx->alo_state_count = 1; // 0 = empty root state
    ctx->lower_ctx.root = term_era();
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
        int metal_ok = 1;
        if (metal_backend.init) metal_ok = (metal_backend.init() == 0);
        if (metal_ok) {
            ctx->backends[THVM_DEV_METAL] = &metal_backend;
            ctx->n_backends = 2;
            ctx->default_device = THVM_DEV_METAL;
        } else {
            fprintf(stderr, "TinyHVM: requested metal backend unavailable; using cpu\n");
        }
    }
    #else
    (void)default_device;
    #endif

    // Allocate trace buffer
    ctx->trace_cap = 1024;
    ctx->trace_buf = calloc(ctx->trace_cap, sizeof(struct InteractionTrace));
    // Keep movement ops lazy by default. Explicit RESHAPE/EXPAND/PERMUTE/PAD/
    // SHRINK terms are the real IR we want to reduce and schedule; eager
    // TAG_TEN alias allocation only hides that structure from phase 1/2.
    ctx->no_grad_alloc = 1;
    thvm_register_pass(ctx, thvm_sched_global_pass, "sched");

    return ctx;
}

// DUP use tracking (used by thvm_reset, thvm_free, and linear_use)
#define TERM_USE_SIZE 4096
static struct {
    Term term;
    u64 first_loc;
    u64 dup_loc;
    u64 tail_loc;
    u64 tail_dup;
} term_use_table[TERM_USE_SIZE];
void term_use_clear(void) { memset(term_use_table, 0, sizeof(term_use_table)); }

void thvm_free(TinyHVM *ctx) {
    thvm_grad_targets_clear(ctx);
    thvm_sched_reset_runtime();
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
    free(ctx->book_heap);
    free(ctx->lower_ctx.heap);
    free(ctx->dup_ports);
    free(ctx->sched_rewrites);
    free(ctx->alo_states);
    free(ctx);
    memset(st_keys, 0, sizeof(st_keys));
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
    // Decref shared buffers before pool_reset — ensures correct per-buffer freeing.
    // Backends gate persistent buf_ids via their own persistent_floor.
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
    // Clear per-step metadata for preserved tensors
    for (u32 i = 1; i < ctx->tensor_count; i++) {
        ctx->tensors[i].defer_consumers = 0;
    }
    fuse_wrap_memo_reset();
    ctx->heap_pos = 1;
    ctx->heap[0] = term_era();
    if (ctx->lower_ctx.heap) {
        ctx->lower_ctx.heap_pos = 1;
        ctx->lower_ctx.heap[0] = term_era();
    }
    ctx->lower_ctx.root = term_era();
    ctx->lower_ctx.rewrite_count = 0;
    ctx->lower_ctx.normalized_sig = 0;
    if (ctx->dup_ports) memset(ctx->dup_ports, 0, DUP_PORT_CAP * sizeof(DupPortEntry));
    if (ctx->sched_rewrites) memset(ctx->sched_rewrites, 0, SCHED_REWRITE_CAP * sizeof(SchedRewriteEntry));
    if (ctx->alo_states) {
        memset(ctx->alo_states, 0, (size_t)ctx->alo_state_cap * sizeof(AloState));
        ctx->alo_state_count = 1;
    }
    ctx->step_graph_settled_replay = 0;
    // Clear shape tracker — stale entries from old heap locs cause wrong
    // view compositions after heap reuse.
    memset(st_keys, 0, sizeof(st_keys));
    memset(term_use_table, 0, sizeof(term_use_table));
}

Term thvm_tensor(TinyHVM *ctx, const f32 *data, Shape s) {
    return thvm_tensor_typed(ctx, data, s, DTYPE_F32);
}

Term thvm_tensor_typed(TinyHVM *ctx, const void *data, Shape s, u32 dtype) {
    u32 id = tensor_create(ctx, s, dtype);
    TensorMeta *m = &ctx->tensors[id];
    if (m->backend && data) {
        m->backend->buf_write(m->buf_id, data, (u64)m->view.numel * dtype_size(dtype));
    }
    // Cache small host payloads (including shape/axes/pairs metadata) so shape
    // tracking and debug code can read them without GPU round-trips.
    if (data && m->view.numel <= MAX_DIM * 2) {
        size_t bytes = (size_t)m->view.numel * dtype_size(dtype);
        m->host_ptr = malloc(bytes);
        memcpy(m->host_ptr, data, bytes);
    }
    return term_ten(id, dtype);
}

Term thvm_tensor_i32(TinyHVM *ctx, const i32 *data, Shape s) {
    return thvm_tensor_typed(ctx, data, s, DTYPE_I32);
}

Term thvm_tensor_u32(TinyHVM *ctx, const u32 *data, Shape s) {
    return thvm_tensor_typed(ctx, data, s, DTYPE_U32);
}

// Get the output view of any term (TAG_TEN or TAG_TOP with shape tracking)
static const View *term_view(TinyHVM *ctx, Term t) {
    if (term_tag(t) == TAG_TEN) return tensor_view_get(&ctx->tensors[(u32)term_val(t)]);
    if (term_tag(t) == TAG_TOP) return st_get(term_val(t));
    if (term_tag(t) == TAG_VAR) {
        u64 loc = term_val(t);
        Term sub = heap_read(ctx, loc);
        if (!term_is_sub(sub)) return term_view(ctx, sub);
        return st_get(loc);
    }
    if (term_tag(t) == TAG_ANN) return term_view(ctx, heap_read(ctx, term_val(t)));
    // Look through DUP nodes: DP0/DP1 share a value at heap[dl+0]
    if (term_tag(t) == TAG_DP0 || term_tag(t) == TAG_DP1)
        return term_view(ctx, heap_read(ctx, term_val(t)));
    return NULL;
}

// Read small metadata tensors (shape/axes/pairs) as u32 values.
// Accepts legacy f32 metadata and new i32/u32 metadata.
static u32 tensor_meta_read_u32(TinyHVM *ctx, u32 tid, u32 *out, u32 max_n) {
    if (tid >= ctx->tensor_count) return 0;
    TensorMeta *m = &ctx->tensors[tid];
    u32 n = m->view.numel;
    if (n == 0 || n > max_n) return 0;

    switch (m->dtype) {
        case DTYPE_I32: {
            i32 tmp[MAX_DIM * 2];
            const i32 *src = (const i32 *)m->host_ptr;
            if (!src) {
                if (!m->backend) return 0;
                META_READ(m->backend, m->buf_id, tmp, n * sizeof(i32));
                src = tmp;
            }
            for (u32 i = 0; i < n; i++) out[i] = (u32)src[i];
            return n;
        }
        case DTYPE_U32: {
            u32 tmp[MAX_DIM * 2];
            const u32 *src = (const u32 *)m->host_ptr;
            if (!src) {
                if (!m->backend) return 0;
                META_READ(m->backend, m->buf_id, tmp, n * sizeof(u32));
                src = tmp;
            }
            memcpy(out, src, n * sizeof(u32));
            return n;
        }
        case DTYPE_F32: {
            f32 tmp[MAX_DIM * 2];
            const f32 *src = (const f32 *)m->host_ptr;
            if (!src) {
                if (!m->backend) return 0;
                META_READ(m->backend, m->buf_id, tmp, n * sizeof(f32));
                src = tmp;
            }
            for (u32 i = 0; i < n; i++) out[i] = (u32)src[i];
            return n;
        }
        default:
            return 0;
    }
}

// Check if a term is a lazy elementwise op (not a chain — just 1 op with TAG_TEN args)
// These are safe to FUSE because single-op backward works with standard rules.
static int is_ew_single(Term t) {
    return (term_tag(t) == TAG_TOP && is_elementwise(term_ext(t)));
}

static int shape_eq(Shape a, Shape b) {
    if (a.rank != b.rank) return 0;
    for (u32 i = 0; i < a.rank; i++)
        if (a.dims[i] != b.dims[i]) return 0;
    return 1;
}

static Shape shape_left_pad_ones(Shape s, u32 rank) {
    Shape out = {.rank = rank};
    u32 pad = rank - s.rank;
    for (u32 i = 0; i < pad; i++) out.dims[i] = 1;
    for (u32 i = 0; i < s.rank; i++) out.dims[pad + i] = s.dims[i];
    return out;
}

static Term thvm_explicit_broadcast_term(TinyHVM *ctx, Term t, Shape target) {
    const View *tv = term_view(ctx, t);
    if (!tv) return t;

    Shape cur = tv->shape;
    if (cur.rank > target.rank) return t;

    if (cur.rank < target.rank) {
        Shape padded = shape_left_pad_ones(cur, target.rank);
        if (!shape_eq(cur, padded)) {
            t = thvm_reshape(ctx, t, padded);
            cur = padded;
        }
    }

    if (!shape_eq(cur, target)) {
        for (u32 i = 0; i < target.rank; i++) {
            if (cur.dims[i] == target.dims[i]) continue;
            if (!(cur.dims[i] == 1 && target.dims[i] > 1)) return t;
        }
        t = thvm_expand(ctx, t, target);
    }
    return t;
}

static void thvm_maybe_normalize_binary_broadcast(TinyHVM *ctx, u32 uop, Term *a, Term *b) {
    if (!is_elementwise(uop) || !is_binary(uop)) return;

    const View *va = term_view(ctx, *a);
    const View *vb = term_view(ctx, *b);
    if (!va || !vb) return;

    View av_bc, bv_bc;
    u32 bc_shape[MAX_DIM], bc_ndim;
    if (!view_broadcast(va, vb, &av_bc, &bv_bc, bc_shape, &bc_ndim)) return;

    Shape target = shape_of(bc_shape, bc_ndim);
    *a = thvm_explicit_broadcast_term(ctx, *a, target);
    *b = thvm_explicit_broadcast_term(ctx, *b, target);
}

Term thvm_sum_axes(TinyHVM *ctx, Term x, const u32 *axes, u32 n_axes);

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
            term_use_table[idx].tail_loc = 0;
            term_use_table[idx].tail_dup = 0;
            return t;
        }
        if (term_use_table[idx].term == t) {
            // Second+ use: maintain a binary right-branching DUP chain.
            // This guarantees each DUP port has at most one direct consumer.
            u64 dl = term_use_table[idx].dup_loc;
            if (!dl) {
                u64 fl = term_use_table[idx].first_loc;
                if (fl == 0 || fl >= ctx->heap_pos || ctx->heap[fl] != t) {
                    // Stale entry — restart tracking at this use site.
                    term_use_table[idx].first_loc = dest_loc;
                    term_use_table[idx].dup_loc = 0;
                    term_use_table[idx].tail_loc = 0;
                    term_use_table[idx].tail_dup = 0;
                    return t;
                }
                dl = heap_alloc(ctx, 1);
                heap_set(ctx, dl, t);
                term_use_table[idx].dup_loc = dl;
                term_use_table[idx].tail_dup = dl;
                term_use_table[idx].tail_loc = dest_loc;
                heap_set(ctx, fl, term_new(TAG_DP0, 0, dl));
                return term_new(TAG_DP1, 0, dl);
            }

            u64 tail_loc = term_use_table[idx].tail_loc;
            u64 tail_dup = term_use_table[idx].tail_dup ? term_use_table[idx].tail_dup : dl;
            Term expect_tail = term_new(TAG_DP1, 0, tail_dup);
            if (tail_loc == 0 || tail_loc >= ctx->heap_pos || ctx->heap[tail_loc] != expect_tail) {
                // Stale chain (heap evolved) — restart at this site.
                term_use_table[idx].first_loc = dest_loc;
                term_use_table[idx].dup_loc = 0;
                term_use_table[idx].tail_loc = 0;
                term_use_table[idx].tail_dup = 0;
                return t;
            }

            u64 nd = heap_alloc(ctx, 1);
            heap_set(ctx, nd, expect_tail);
            heap_set(ctx, tail_loc, term_new(TAG_DP0, 0, nd));
            term_use_table[idx].tail_dup = nd;
            term_use_table[idx].tail_loc = dest_loc;
            return term_new(TAG_DP1, 0, nd);
        }
    }
    return t; // table full
}

static void thvm_track_top_shape(TinyHVM *ctx, u64 loc, u32 uop, Term a, Term b) {
    const View *va = term_view(ctx, a);
    const View *vb_dbg = (term_tag(b) != TAG_ERA) ? term_view(ctx, b) : NULL;
    const ShapeTracker *ast =
        (term_tag(a) == TAG_TOP || term_tag(a) == TAG_VAR)
            ? st_get_tracker(term_val(a))
            : NULL;
    {
        View out = {0};
        int stored = 0;

        // View ops: output shape deterministic from metadata
        if (term_tag(b) == TAG_TEN) {
            u32 bf[MAX_DIM * 2];
            u32 bn = tensor_meta_read_u32(ctx, (u32)term_val(b), bf, MAX_DIM * 2);
            if (bn && uop == UOP_RESHAPE) {
                Shape ns = {.rank = bn};
                for (u32 i = 0; i < bn; i++) ns.dims[i] = bf[i];
                u32 nn = 1;
                for (u32 i = 0; i < ns.rank; i++) nn *= ns.dims[i];
                if (ast && va && nn == va->numel) {
                    ShapeTracker composed = st_reshape(*ast, ns);
                    st_set_tracker(loc, &composed);
                } else if (va && nn == va->numel) {
                    ShapeTracker composed = st_reshape(st_from_view(*va), ns);
                    st_set_tracker(loc, &composed);
                } else {
                    out = view_create(ns);
                    st_set(loc, &out);
                }
                stored = 1;
            } else if (bn && uop == UOP_EXPAND) {
                if (ast && va) {
                    ShapeTracker composed = st_expand(*ast, bf, bn);
                    st_set_tracker(loc, &composed);
                } else if (va) {
                    ShapeTracker composed = st_expand(st_from_view(*va), bf, bn);
                    st_set_tracker(loc, &composed);
                } else {
                    Shape ns = {.rank = bn};
                    for (u32 i = 0; i < bn; i++) ns.dims[i] = bf[i];
                    out = view_create(ns);
                    st_set(loc, &out);
                }
                stored = 1;
            } else if (bn && uop == UOP_SHRINK) {
                u32 ndim = bn / 2;
                u32 starts[MAX_DIM], ends[MAX_DIM];
                for (u32 i = 0; i < ndim; i++) { starts[i] = bf[i * 2]; ends[i] = bf[i * 2 + 1]; }
                if (ast && va) {
                    ShapeTracker composed = st_shrink(*ast, starts, ends, ndim);
                    st_set_tracker(loc, &composed);
                } else {
                    Shape ns = {.rank = ndim};
                    for (u32 i = 0; i < ndim; i++) ns.dims[i] = ends[i] - starts[i];
                    out = view_create(ns);
                    st_set(loc, &out);
                }
                stored = 1;
            } else if (bn && uop == UOP_PAD) {
                u32 ndim = bn / 2;
                u32 pb[MAX_DIM], pa[MAX_DIM];
                for (u32 i = 0; i < ndim; i++) { pb[i] = bf[i * 2]; pa[i] = bf[i * 2 + 1]; }
                if (ast && va) {
                    ShapeTracker composed = st_pad(*ast, pb, pa);
                    st_set_tracker(loc, &composed);
                } else if (va) {
                    ShapeTracker composed = st_pad(st_from_view(*va), pb, pa);
                    st_set_tracker(loc, &composed);
                } else {
                    Shape ns = {.rank = ndim};
                    for (u32 i = 0; i < ndim; i++) ns.dims[i] = pb[i] + pa[i];
                    out = view_create(ns);
                    st_set(loc, &out);
                }
                stored = 1;
            } else if (bn && uop == UOP_PERMUTE && va) {
                u32 axes[MAX_DIM];
                for (u32 i = 0; i < bn; i++) axes[i] = bf[i];
                if (ast) {
                    ShapeTracker composed = st_permute(*ast, axes, bn);
                    st_set_tracker(loc, &composed);
                } else {
                    ShapeTracker composed = st_permute(st_from_view(*va), axes, bn);
                    st_set_tracker(loc, &composed);
                }
                stored = 1;
            } else if ((uop == UOP_SUM || uop == UOP_RMAX)) {
                if (!va) goto skip_sum;
                out = *va;
                for (u32 i = 0; i < bn; i++) {
                    u32 ax = bf[i];
                    if (ax < out.shape.rank) out.shape.dims[ax] = 1;
                }
                out = view_create(out.shape);
                st_set(loc, &out);
                stored = 1;
                skip_sum:;
            }
        }

        if (!stored && va && (uop == UOP_SUM || uop == UOP_RMAX) && term_tag(b) == TAG_ERA) {
            out = *va;
            for (int d = (int)va->shape.rank - 1; d >= 0; d--)
                if (va->shape.dims[d] > 1) { out.shape.dims[d] = 1; break; }
            out = view_create(out.shape);
            st_set(loc, &out);
            stored = 1;
        }

        if (!stored && uop == UOP_MM && va) {
            const View *vb = (term_tag(b) != TAG_ERA) ? term_view(ctx, b) : NULL;
            if (vb) {
                Shape os = {.rank = 2, .dims = {va->shape.dims[0], vb->shape.dims[1]}};
                out = view_create(os);
                st_set(loc, &out);
                stored = 1;
            }
        }

        if (!stored && uop == UOP_CAST && va) {
            out = *va;
            st_set(loc, &out);
            stored = 1;
        }

        if (!stored && uop == UOP_DETACH && va) {
            out = *va;
            st_set(loc, &out);
            stored = 1;
        }

        if (!stored && is_elementwise(uop)) {
            const View *vb = (term_tag(b) != TAG_ERA) ? term_view(ctx, b) : NULL;
            if (va && vb) {
                View av_bc, bv_bc;
                u32 bc_shape[MAX_DIM], bc_ndim;
                if (view_broadcast(va, vb, &av_bc, &bv_bc, bc_shape, &bc_ndim))
                    out = view_create(shape_of(bc_shape, bc_ndim));
                else
                    out = *va;
            } else if (va) {
                out = *va;
            } else {
                const View *vb = (term_tag(b) != TAG_ERA) ? term_view(ctx, b) : NULL;
                if (vb) out = *vb;
                else goto skip_st;
            }
            st_set(loc, &out);
            skip_st:;
        }
    }
    if (getenv("THVM_SCHED_DIAG") && loc >= 300) {
        const View *ov = st_get(loc);
        fprintf(stderr, "  OP_TRACK: loc=%llu uop=%s out=[",
                (unsigned long long)loc, uop < UOP_COUNT ? uop_names[uop] : "?");
        if (ov) for (u32 _d = 0; _d < ov->shape.rank; _d++)
            fprintf(stderr, "%u%s", ov->shape.dims[_d], _d + 1 < ov->shape.rank ? "," : "");
        fprintf(stderr, "] a=[");
        if (va) for (u32 _d = 0; _d < va->shape.rank; _d++)
            fprintf(stderr, "%u%s", va->shape.dims[_d], _d + 1 < va->shape.rank ? "," : "");
        fprintf(stderr, "] b=[");
        if (vb_dbg) for (u32 _d = 0; _d < vb_dbg->shape.rank; _d++)
            fprintf(stderr, "%u%s", vb_dbg->shape.dims[_d], _d + 1 < vb_dbg->shape.rank ? "," : "");
        fprintf(stderr, "]\n");
    }
}

// Fast path: skip linear_use/shadow bookkeeping, but still track lazy shapes.
Term thvm_op_raw(TinyHVM *ctx, u32 uop, Term a, Term b) {
    thvm_maybe_normalize_binary_broadcast(ctx, uop, &a, &b);
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc, a);
    heap_set(ctx, loc + 1, b);
    thvm_track_top_shape(ctx, loc, uop, a, b);
    return term_new(TAG_TOP, uop, loc);
}

Term thvm_op(TinyHVM *ctx, u32 uop, Term a, Term b) {
    // Decompose MM into expand+MUL+SUM when BOTH inputs are lazy (TAG_TOP).
    // This enables fusion with downstream ops (bias+relu) in forward pass.
    // When inputs are materialized (TAG_TEN, from GRAD handler), keep UOP_MM
    // for efficient MPS dispatch (avoids creating huge expanded intermediates).
    if (uop == UOP_MM) {
        const View *va = term_view(ctx, a);
        const View *vb = term_view(ctx, b);
        if (va && vb && va->shape.rank == 2 && vb->shape.rank == 2) {
            int a_lazy = (term_tag(a) == TAG_TOP);
            int b_lazy = (term_tag(b) == TAG_TOP);
            // Primitive matmul: EXPAND+MUL+SUM. Fuses with surrounding ops.
            {
                u32 M = va->shape.dims[0], K = va->shape.dims[1], N = vb->shape.dims[1];
                extern Term thvm_sum_axes(TinyHVM *ctx, Term x, const u32 *axes, u32 n_axes);
                Term a3 = thvm_expand(ctx, thvm_reshape(ctx, a, SHAPE(M, K, 1)), SHAPE(M, K, N));
                Term b3 = thvm_expand(ctx, thvm_reshape(ctx, b, SHAPE(1, K, N)), SHAPE(M, K, N));
                Term prod = thvm_op(ctx, UOP_MUL, a3, b3);
                Term sum = thvm_sum_axes(ctx, prod, (u32[]){1}, 1);
                return thvm_reshape(ctx, sum, SHAPE(M, N));
            }
        }
    }

    // UOP graphs share heap references directly; DUP boundaries should come
    // from explicit combinator semantics, not constructor-time linear_use.
    return thvm_op_raw(ctx, uop, a, b);
}

Term thvm_cast(TinyHVM *ctx, Term t, u32 dtype) {
    return thvm_op(ctx, UOP_CAST, t, thvm_scalar_u32(ctx, dtype));
}

Term thvm_detach(TinyHVM *ctx, Term t) {
    return thvm_op(ctx, UOP_DETACH, t, term_era());
}

// Matmul: [M,K] @ [K,N] → [M,N]. Decomposed to EXPAND+MUL+SUM primitives.
// Codegen can recognize this pattern and emit MPS matmul as optimization.
Term thvm_mm(TinyHVM *ctx, Term a, Term b) {
    // Get shapes from TAG_TEN or shape table
    const View *va = term_view(ctx, a);
    const View *vb = term_view(ctx, b);
    Shape sa = va ? va->shape : SHAPE(1);
    Shape sb = vb ? vb->shape : SHAPE(1);
    u32 M = sa.dims[0], K = sa.dims[1], N = sb.dims[1];
    // a: [M,K] → [M,K,1] → expand [M,K,N]
    Term a3 = thvm_reshape(ctx, a, SHAPE(M, K, 1));
    Term a_exp = thvm_expand(ctx, a3, SHAPE(M, K, N));
    // b: [K,N] → [1,K,N] → expand [M,K,N]
    Term b3 = thvm_reshape(ctx, b, SHAPE(1, K, N));
    Term b_exp = thvm_expand(ctx, b3, SHAPE(M, K, N));
    // mul: [M,K,N], sum over K (axis 1) → [M,1,N]
    Term mul = thvm_op(ctx, UOP_MUL, a_exp, b_exp);
    Term summed = thvm_sum_axes(ctx, mul, (u32[]){1}, 1);
    // reshape [M,1,N] → [M,N]
    return thvm_reshape(ctx, summed, SHAPE(M, N));
}

// Multi-axis SUM: reduces along specified axes in one pass.
// Matches tinygrad's .sum(axis=[...]) — the axes tensor is stored as the
// second heap slot, just like reshape stores shape.
Term thvm_sum_axes(TinyHVM *ctx, Term x, const u32 *axes, u32 n_axes) {
    i32 axes_i[MAX_DIM];
    for (u32 i = 0; i < n_axes; i++) axes_i[i] = (i32)axes[i];
    Term axes_t = thvm_tensor_i32(ctx, axes_i, SHAPE(n_axes));
    return thvm_op(ctx, UOP_SUM, x, axes_t);
}

Term thvm_rmax_axes(TinyHVM *ctx, Term x, const u32 *axes, u32 n_axes) {
    i32 axes_i[MAX_DIM];
    for (u32 i = 0; i < n_axes; i++) axes_i[i] = (i32)axes[i];
    Term axes_t = thvm_tensor_i32(ctx, axes_i, SHAPE(n_axes));
    return thvm_op(ctx, UOP_RMAX, x, axes_t);
}

// Movement ops: eager when input is TAG_TEN (zero GPU alloc, just view transform)
// Falls back to lazy TOP with shape-tensor only for unreduced inputs.

Term thvm_reshape(TinyHVM *ctx, Term t, Shape new_shape) {
    if (term_tag(t) == TAG_TEN) {
        if (ctx->no_grad_alloc) goto lazy;
        u32 src_id = (u32)term_val(t);
        TensorMeta *m = &ctx->tensors[src_id];
        // Use view_reshape's merge-split algorithm to compute new strides.
        // Masked views (from PAD) must be materialized before reshape —
        // view_reshape drops the mask, causing wrong data downstream.
        if (m->view.has_mask) goto lazy;
        View rv = view_reshape(m->view, new_shape);
        // view_reshape returns numel=0 when merge fails → go to lazy path
        // (lazy path uses ShapeTracker composition which pushes a new view)
        if (rv.numel == 0) goto lazy;
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
    { i32 dims[MAX_DIM];
    for (u32 i = 0; i < new_shape.rank; i++) dims[i] = (i32)new_shape.dims[i];
    Term shape_t = thvm_tensor_i32(ctx, dims, SHAPE(new_shape.rank));
    // thvm_op composes ShapeTracker via st_reshape — always succeeds.
    return thvm_op(ctx, UOP_RESHAPE, t, shape_t); }
}

Term thvm_expand(TinyHVM *ctx, Term t, Shape new_shape) {
    if (term_tag(t) == TAG_TEN) {
        if (ctx->no_grad_alloc) goto expand_lazy;
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
expand_lazy:;
    { i32 dims[MAX_DIM];
    for (u32 i = 0; i < new_shape.rank; i++) dims[i] = (i32)new_shape.dims[i];
    Term shape_t = thvm_tensor_i32(ctx, dims, SHAPE(new_shape.rank));
    Term r = thvm_op(ctx, UOP_EXPAND, t, shape_t);
    const View *iv = term_view(ctx, t);
    if (iv) { View ov = *iv; ov.shape = new_shape; ov.numel = 1;
        for (u32 i=0;i<new_shape.rank;i++) {
            if (i<iv->shape.rank && iv->shape.dims[i]==1 && new_shape.dims[i]>1) ov.strides[i]=0;
            ov.shape.dims[i]=new_shape.dims[i]; ov.numel*=new_shape.dims[i];
        }
        ov.contiguous=0; st_set(term_val(r), &ov); }
    return r; }
}

Term thvm_permute(TinyHVM *ctx, Term t, const u32 *axes, u32 rank) {
    if (term_tag(t) == TAG_TEN) {
        if (ctx->no_grad_alloc) goto permute_lazy;
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
        i32 axes_i[MAX_DIM];
        for (u32 i = 0; i < rank; i++) axes_i[i] = (i32)axes[i];
        Term axes_t = thvm_tensor_i32(ctx, axes_i, SHAPE(rank));
        ctx->tensors[id].src_ids[1] = (u32)term_val(axes_t);
        if (m->requires_grad) ctx->tensors[id].requires_grad = 1;
        return term_ten(id, m->dtype);
    }
permute_lazy:;
    { i32 axes_i[MAX_DIM];
    for (u32 i = 0; i < rank; i++) axes_i[i] = (i32)axes[i];
    Term axes_t = thvm_tensor_i32(ctx, axes_i, SHAPE(rank));
    Term r = thvm_op(ctx, UOP_PERMUTE, t, axes_t);
    const View *iv = term_view(ctx, t);
    if (iv) { View ov = view_permute(*iv, axes); st_set(term_val(r), &ov); }
    return r; }
}

// Pad: pairs = [before0, after0, before1, after1, ...]
Term thvm_pad(TinyHVM *ctx, Term t, const u32 *pairs, u32 ndim) {
    if (term_tag(t) == TAG_TEN) {
        if (ctx->no_grad_alloc) goto pad_lazy;
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
        i32 pairs_i[MAX_DIM * 2];
        for (u32 i2 = 0; i2 < ndim * 2; i2++) pairs_i[i2] = (i32)pairs[i2];
        Term pt = thvm_tensor_i32(ctx, pairs_i, SHAPE(ndim * 2));
        ctx->tensors[id].src_ids[1] = (u32)term_val(pt);
        if (m->requires_grad) ctx->tensors[id].requires_grad = 1;
        return term_ten(id, m->dtype);
    }
pad_lazy:;
    { u32 pb[MAX_DIM], pa[MAX_DIM];
    for (u32 i=0;i<ndim;i++){pb[i]=pairs[i*2];pa[i]=pairs[i*2+1];}
    i32 pairs_i[MAX_DIM * 2];
    for (u32 i = 0; i < ndim * 2; i++) pairs_i[i] = (i32)pairs[i];
    Term pairs_t = thvm_tensor_i32(ctx, pairs_i, SHAPE(ndim * 2));
    Term r = thvm_op(ctx, UOP_PAD, t, pairs_t);
    const View *iv = term_view(ctx, t);
    if (iv) { View ov = view_pad(*iv, pb, pa); st_set(term_val(r), &ov); }
    return r; }
}

// Shrink: pairs = [start0, end0, start1, end1, ...]
Term thvm_shrink(TinyHVM *ctx, Term t, const u32 *pairs, u32 ndim) {
    if (term_tag(t) == TAG_TEN) {
        if (ctx->no_grad_alloc) goto shrink_lazy;
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
        i32 shrink_i[MAX_DIM * 2];
        for (u32 i2 = 0; i2 < ndim * 2; i2++) shrink_i[i2] = (i32)pairs[i2];
        Term st = thvm_tensor_i32(ctx, shrink_i, SHAPE(ndim * 2));
        ctx->tensors[id].src_ids[1] = (u32)term_val(st);
        if (m->requires_grad) ctx->tensors[id].requires_grad = 1;
        return term_ten(id, m->dtype);
    }
shrink_lazy:;
    { u32 ss[MAX_DIM], se[MAX_DIM];
    for (u32 i=0;i<ndim;i++){ss[i]=pairs[i*2];se[i]=pairs[i*2+1];}
    i32 pairs_i[MAX_DIM * 2];
    for (u32 i = 0; i < ndim * 2; i++) pairs_i[i] = (i32)pairs[i];
    Term pairs_t = thvm_tensor_i32(ctx, pairs_i, SHAPE(ndim * 2));
    Term r = thvm_op(ctx, UOP_SHRINK, t, pairs_t);
    const View *iv = term_view(ctx, t);
    if (iv) { View ov = view_shrink(*iv, ss, se); st_set(term_val(r), &ov); }
    return r; }
}

// ============================================================
// _pool — tinygrad's sliding window via movement UOps
// Full implementation: handles k>s (complex/repeat) and k<=s (simple)
// From tensor.py:2278-2301
// ============================================================
