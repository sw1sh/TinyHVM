// tinyhvm.c — Hub file. Include-based single-translation-unit build.
// Inspired by HVM4's hvm.c architecture.

#include "tinyhvm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// ============================================================
// term.c — Term bit packing/unpacking
// ============================================================

static inline Term term_new(u32 tag, u32 ext, u64 val) {
    return ((u64)(tag & TAG_MASK) << TAG_SHIFT) |
           ((u64)(ext & EXT_MASK) << EXT_SHIFT) |
           (val & VAL_MASK);
}

static inline u32 term_tag(Term t) { return (u32)((t >> TAG_SHIFT) & TAG_MASK); }
static inline u32 term_ext(Term t) { return (u32)((t >> EXT_SHIFT) & EXT_MASK); }
static inline u64 term_val(Term t) { return t & VAL_MASK; }

static inline Term term_set_sub(Term t) { return t | (1ULL << SUB_SHIFT); }
static inline int  term_is_sub(Term t)  { return (t >> SUB_SHIFT) & 1; }

// --- Constructors ---

static inline Term term_num_u32(u32 n) {
    return term_new(TAG_NUM, NUM_U32, (u64)n);
}

static inline Term term_num_f32(f32 f) {
    u32 bits; memcpy(&bits, &f, 4);
    return term_new(TAG_NUM, NUM_F32, (u64)bits);
}

static inline f32 term_as_f32(Term t) {
    u32 bits = (u32)term_val(t); f32 f; memcpy(&f, &bits, 4); return f;
}

static inline u32 term_as_u32(Term t) {
    return (u32)term_val(t);
}

static inline Term term_era(void)              { return term_new(TAG_ERA, 0, 0); }
static inline Term term_var(u64 loc)           { return term_new(TAG_VAR, 0, loc); }
static inline Term term_ten(u32 tid, u32 dt)   { return term_new(TAG_TEN, dt, (u64)tid); }
static inline Term term_top(u32 uop, u64 loc)  { return term_new(TAG_TOP, uop, loc); }

// ============================================================
// heap.c — Single-threaded bump allocator
// ============================================================

static inline u64  heap_alloc(TinyHVM *ctx, u64 w) { u64 l = ctx->heap_pos; ctx->heap_pos += w; assert(ctx->heap_pos < HEAP_CAP); return l; }
static inline Term heap_read(TinyHVM *ctx, u64 l)  { return ctx->heap[l]; }
static inline void heap_set(TinyHVM *ctx, u64 l, Term t) { ctx->heap[l] = t; }

// ============================================================
// view.c — View helpers (shape/stride arithmetic)
// ============================================================

static View view_create(Shape s) {
    View v = {0};
    v.shape = s;
    v.numel = 1;
    for (u32 i = 0; i < s.rank; i++) {
        v.numel *= s.dims[i];
    }
    // Row-major strides
    for (u32 i = 0; i < s.rank; i++) {
        i32 st = 1;
        for (u32 j = i + 1; j < s.rank; j++) st *= (i32)s.dims[j];
        v.strides[i] = st;
    }
    v.offset = 0;
    v.contiguous = 1;
    return v;
}

static View view_permute(View v, const u32 *axes) {
    View r = v;
    for (u32 i = 0; i < v.shape.rank; i++) {
        r.shape.dims[i] = v.shape.dims[axes[i]];
        r.strides[i] = v.strides[axes[i]];
    }
    r.contiguous = 0;
    return r;
}

// Reshape: validate numel, recompute strides (row-major)
static View view_reshape(View v, Shape new_shape) {
    u32 new_numel = 1;
    for (u32 i = 0; i < new_shape.rank; i++) new_numel *= new_shape.dims[i];
    assert(new_numel == v.numel && "reshape: numel mismatch");
    View r = {0};
    r.shape = new_shape;
    r.numel = new_numel;
    r.offset = v.offset;
    // New row-major strides
    for (u32 i = 0; i < new_shape.rank; i++) {
        i32 st = 1;
        for (u32 j = i + 1; j < new_shape.rank; j++) st *= (i32)new_shape.dims[j];
        r.strides[i] = st;
    }
    r.contiguous = v.contiguous;
    return r;
}

// Expand: set stride=0 for broadcast dimensions (dim must be 1 → n)
static View view_expand(View v, Shape new_shape) {
    assert(v.shape.rank == new_shape.rank);
    View r = v;
    r.numel = 1;
    for (u32 i = 0; i < new_shape.rank; i++) {
        if (v.shape.dims[i] == 1 && new_shape.dims[i] > 1) {
            r.strides[i] = 0;  // broadcast!
        } else {
            assert(v.shape.dims[i] == new_shape.dims[i]);
        }
        r.shape.dims[i] = new_shape.dims[i];
        r.numel *= new_shape.dims[i];
    }
    r.contiguous = 0;
    return r;
}

// Shrink: slice — adjust offset and shape
static View view_shrink(View v, const u32 *starts, const u32 *ends) {
    View r = v;
    r.numel = 1;
    for (u32 i = 0; i < v.shape.rank; i++) {
        r.offset += (i32)starts[i] * v.strides[i];
        r.shape.dims[i] = ends[i] - starts[i];
        r.numel *= r.shape.dims[i];
    }
    r.contiguous = 0;
    return r;
}

// Pad: extend shape, adjust offset (data starts at offset with original strides)
// Padding requires physical copy to a bigger buffer, so this is NOT free.
// However we can handle it at dispatch time.
static View view_pad(View v, const u32 *pad_before, const u32 *pad_after) {
    View r = v;
    r.numel = 1;
    for (u32 i = 0; i < v.shape.rank; i++) {
        r.shape.dims[i] = v.shape.dims[i] + pad_before[i] + pad_after[i];
        r.numel *= r.shape.dims[i];
    }
    r.contiguous = 0;
    return r;
}

// Stride: modify strides for pooling window extraction
static View view_stride(View v, const u32 *strides_mult) {
    View r = v;
    r.numel = 1;
    for (u32 i = 0; i < v.shape.rank; i++) {
        r.shape.dims[i] = (v.shape.dims[i] + strides_mult[i] - 1) / strides_mult[i];
        r.strides[i] = v.strides[i] * (i32)strides_mult[i];
        r.numel *= r.shape.dims[i];
    }
    r.contiguous = 0;
    return r;
}



// Broadcast two views to a common shape. Returns 1 on success.
static int view_broadcast(const View *a, const View *b, View *out_a, View *out_b, u32 *out_shape, u32 *out_ndim) {
    // Align from the right, pad with 1s on the left
    u32 ndim = a->shape.rank > b->shape.rank ? a->shape.rank : b->shape.rank;
    *out_ndim = ndim;

    // Build padded shapes (right-aligned)
    u32 sa[MAX_DIM] = {0}, sb[MAX_DIM] = {0};
    i32 sta[MAX_DIM] = {0}, stb[MAX_DIM] = {0};
    u32 off_a = ndim - a->shape.rank, off_b = ndim - b->shape.rank;

    for (u32 i = 0; i < ndim; i++) {
        sa[i] = (i >= off_a) ? a->shape.dims[i - off_a] : 1;
        sb[i] = (i >= off_b) ? b->shape.dims[i - off_b] : 1;
        sta[i] = (i >= off_a) ? a->strides[i - off_a] : 0;
        stb[i] = (i >= off_b) ? b->strides[i - off_b] : 0;
    }

    // Compute broadcast shape
    u32 numel = 1;
    for (u32 i = 0; i < ndim; i++) {
        if (sa[i] == sb[i])       out_shape[i] = sa[i];
        else if (sa[i] == 1)      out_shape[i] = sb[i];
        else if (sb[i] == 1)      out_shape[i] = sa[i];
        else return 0;  // incompatible
        numel *= out_shape[i];
    }

    // Build broadcast views (stride=0 where expanded)
    *out_a = (View){0};
    *out_b = (View){0};
    out_a->shape.rank = out_b->shape.rank = ndim;
    out_a->numel = out_b->numel = numel;
    out_a->offset = a->offset;
    out_b->offset = b->offset;
    for (u32 i = 0; i < ndim; i++) {
        out_a->shape.dims[i] = out_b->shape.dims[i] = out_shape[i];
        out_a->strides[i] = (sa[i] == 1 && out_shape[i] > 1) ? 0 : sta[i];
        out_b->strides[i] = (sb[i] == 1 && out_shape[i] > 1) ? 0 : stb[i];
    }
    return 1;
}

// ============================================================
// tensor.c — Tensor metadata registry
// ============================================================

static u32 tensor_create(TinyHVM *ctx, Shape s, u32 dtype) {
    assert(s.rank <= MAX_DIM && ctx->tensor_count < MAX_TENSORS);
    u32 id = ctx->tensor_count++;
    TensorMeta *m = &ctx->tensors[id];
    memset(m, 0, sizeof(*m));
    m->dtype = dtype;
    m->refcount = 1;
    m->view = view_create(s);

    if (ctx->backend) {
        m->buf_id = ctx->backend->buf_alloc((u64)m->view.numel * dtype_size(dtype));
    }
    return id;
}

// Create a tensor that shares another's buffer but has a different View
// No GPU allocation — just metadata.
static u32 tensor_view_of(TinyHVM *ctx, u32 src_id, View new_view) {
    assert(ctx->tensor_count < MAX_TENSORS);
    u32 id = ctx->tensor_count++;
    TensorMeta *m = &ctx->tensors[id];
    TensorMeta *ms = &ctx->tensors[src_id];
    memset(m, 0, sizeof(*m));
    m->dtype = ms->dtype;
    m->refcount = 1;
    m->buf_id = ms->buf_id;    // SHARE the buffer
    m->view = new_view;
    return id;
}



// ============================================================
// reduce.c — WNF reduction engine
// ============================================================

Term thvm_reduce(TinyHVM *ctx, Term t) {
    u32 tag = term_tag(t);

    switch (tag) {
        case TAG_TOP: {
            u32 uop = term_ext(t);
            u64 loc = term_val(t);

            // Movement ops: modify View, share buffer
            int is_movement = (uop >= UOP_RESHAPE && uop <= UOP_PAD);

            // Reduce arguments
            Term a = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, a);

            int is_binary = (uop >= UOP_ADD && uop <= UOP_SUB) || uop == UOP_MM;
            int is_cnn_input = (uop >= UOP_IM2COL && uop <= UOP_NHWC2NCHW);
            int is_reduce = (uop == UOP_SUM || uop == UOP_RMAX);
            Term b = term_era();
            if (is_binary || is_movement || is_cnn_input) {
                b = thvm_reduce(ctx, heap_read(ctx, loc + 1));
                heap_set(ctx, loc + 1, b);
            }

            if (term_tag(a) != TAG_TEN) return t;
            if (is_binary && term_tag(b) != TAG_TEN) return t;

            u32 a_id = (u32)term_val(a);
            TensorMeta *ma = &ctx->tensors[a_id];

            // Movement ops: create view, share buffer, no compute
            if (is_movement) {
                // b encodes the new shape/args as a tensor with shape data
                u32 b_id = 0;
                if (term_tag(b) == TAG_TEN) b_id = (u32)term_val(b);
                View new_view;
                switch (uop) {
                    case UOP_RESHAPE: {
                        // b is a 1D tensor whose elements are the new dims
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        Shape ns = {.rank = mb->view.numel};
                        f32 *dims = malloc(ns.rank * sizeof(f32));
                        ctx->backend->buf_read(mb->buf_id, dims, ns.rank * sizeof(f32));
                        for (u32 i = 0; i < ns.rank; i++) ns.dims[i] = (u32)dims[i];
                        free(dims);
                        new_view = view_reshape(ma->view, ns);
                        break;
                    }
                    case UOP_EXPAND: {
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        Shape ns = {.rank = mb->view.numel};
                        f32 *dims = malloc(ns.rank * sizeof(f32));
                        ctx->backend->buf_read(mb->buf_id, dims, ns.rank * sizeof(f32));
                        for (u32 i = 0; i < ns.rank; i++) ns.dims[i] = (u32)dims[i];
                        free(dims);
                        new_view = view_expand(ma->view, ns);
                        break;
                    }
                    case UOP_PERMUTE: {
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        u32 rank = mb->view.numel;
                        f32 *axes_f = malloc(rank * sizeof(f32));
                        ctx->backend->buf_read(mb->buf_id, axes_f, rank * sizeof(f32));
                        u32 axes[MAX_DIM];
                        for (u32 i = 0; i < rank; i++) axes[i] = (u32)axes_f[i];
                        free(axes_f);
                        new_view = view_permute(ma->view, axes);
                        break;
                    }
                    default:
                        // PAD, SHRINK need physical copy — fall through to compute
                        // For now, assert
                        assert(0 && "pad/shrink not yet implemented in reduce");
                        new_view = ma->view;
                }
                u32 dst_id = tensor_view_of(ctx, a_id, new_view);
                // Record provenance
                if (ctx->recording && ma->requires_grad) {
                    TensorMeta *md = &ctx->tensors[dst_id];
                    md->requires_grad = 1;
                    md->creator_op = uop;
                    md->src_ids[0] = a_id;
                    md->src_ids[1] = b_id;
                }
                ctx->itrs++;
                return term_ten(dst_id, ma->dtype);
            }

            if (!ctx->backend) return t;

            u32 b_id = is_binary ? (u32)term_val(b) : 0;
            TensorMeta *mb = is_binary ? &ctx->tensors[b_id] : NULL;

            // CNN-specific ops: a=input, b=params tensor (contains Conv2dParams or LayoutParams)
            int is_cnn_op = (uop >= UOP_IM2COL && uop <= UOP_NHWC2NCHW);
            if (is_cnn_op) {
                // b holds conv params as raw bytes
                assert(term_tag(b) == TAG_TEN);
                b_id = (u32)term_val(b);
                mb = &ctx->tensors[b_id];
            }

            // Determine output shape
            u32 out_shape[MAX_DIM];
            u32 out_ndim;
            View av_bc, bv_bc;  // broadcast views

            if (uop == UOP_MM) {
                // matmul: [M,K] x [K,N] → [M,N]
                assert(ma->view.shape.rank == 2 && mb->view.shape.rank == 2);
                assert(ma->view.shape.dims[1] == mb->view.shape.dims[0]);
                out_shape[0] = ma->view.shape.dims[0];
                out_shape[1] = mb->view.shape.dims[1];
                out_ndim = 2;
            } else if (is_cnn_op) {
                // Read Conv2dParams from params tensor buffer
                // (Layout ops use B,C,H,W from shape directly)
                if (uop == UOP_IM2COL) {
                    // Output: [n_patches, patch_size]
                    // Read params: {B,Cin,H,W,KH,KW,OH,OW,patch_size,n_patches}
                    u32 params[10];
                    ctx->backend->buf_read(mb->buf_id, params, sizeof(params));
                    out_shape[0] = params[9]; // n_patches
                    out_shape[1] = params[8]; // patch_size
                    out_ndim = 2;
                } else if (uop == UOP_COL2IM) {
                    // Output: [B,Cin,H,W]
                    u32 params[10];
                    ctx->backend->buf_read(mb->buf_id, params, sizeof(params));
                    out_shape[0] = params[0]; // B
                    out_shape[1] = params[1]; // Cin
                    out_shape[2] = params[2]; // H
                    out_shape[3] = params[3]; // W
                    out_ndim = 4;
                } else if (uop == UOP_MAXPOOL) {
                    // Output: [B,C,OH,OW] where OH=H/2, OW=W/2
                    assert(ma->view.shape.rank == 4);
                    out_shape[0] = ma->view.shape.dims[0]; // B
                    out_shape[1] = ma->view.shape.dims[1]; // C
                    out_shape[2] = ma->view.shape.dims[2] / 2; // OH
                    out_shape[3] = ma->view.shape.dims[3] / 2; // OW
                    out_ndim = 4;
                } else if (uop == UOP_NCHW2NHWC || uop == UOP_NHWC2NCHW) {
                    // Same total elements, just reordered
                    u32 n = ma->view.numel;
                    out_shape[0] = n;
                    out_ndim = 1;
                } else {
                    assert(0 && "unknown CNN UOp");
                    out_ndim = 0;
                }
            } else if (is_reduce) {
                // Reduce: collapse last dim to 1
                out_ndim = ma->view.shape.rank;
                for (u32 i = 0; i < out_ndim; i++) out_shape[i] = ma->view.shape.dims[i];
                out_shape[out_ndim - 1] = 1;
            } else if (is_binary) {
                // Binary: broadcast shapes
                int ok = view_broadcast(&ma->view, &mb->view, &av_bc, &bv_bc, out_shape, &out_ndim);
                assert(ok && "shape broadcast failed");
            } else {
                // Unary: output = input shape
                out_ndim = ma->view.shape.rank;
                for (u32 i = 0; i < out_ndim; i++) out_shape[i] = ma->view.shape.dims[i];
            }

            u32 dst_id = tensor_create(ctx, shape_of(out_shape, out_ndim), ma->dtype);
            TensorMeta *md = &ctx->tensors[dst_id];

            // Record provenance for autograd
            if (ctx->recording) {
                int needs = ma->requires_grad || (mb && mb->requires_grad);
                if (needs) {
                    md->requires_grad = 1;
                    md->creator_op = uop;
                    md->src_ids[0] = a_id;
                    md->src_ids[1] = b_id;
                }
            }

            // Dispatch
            if (uop == UOP_MM) {
                u32 M = ma->view.shape.dims[0], K = ma->view.shape.dims[1], N = mb->view.shape.dims[1];
                ctx->backend->op_mm(md->buf_id, ma->buf_id, &ma->view,
                                mb->buf_id, &mb->view, M, K, N);
            } else if (is_cnn_op) {
                // CNN ops dispatched via backend op_cnn
                ctx->backend->op_cnn(uop, md->buf_id, ma->buf_id, mb->buf_id);
            } else if (is_reduce) {
                u32 reduce_dim = ma->view.shape.dims[ma->view.shape.rank - 1];
                ctx->backend->op_reduce(uop, md->buf_id, md->view.numel,
                                        ma->buf_id, ma->view.numel, reduce_dim);
            } else if (is_binary) {
                ctx->backend->op_binary(uop, md->buf_id, &md->view,
                                    ma->buf_id, &av_bc,
                                    ctx->tensors[b_id].buf_id, &bv_bc);
            } else {
                ctx->backend->op_unary(uop, md->buf_id, &md->view,
                                   ma->buf_id, &ma->view);
            }

            ctx->itrs++;
            return term_ten(dst_id, ma->dtype);
        }

        case TAG_APP: {
            u64 loc = term_val(t);
            Term fun = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, fun);
            if (term_tag(fun) == TAG_LAM) {
                u64 lam_loc = term_val(fun);
                Term arg = heap_read(ctx, loc + 1);
                heap_set(ctx, lam_loc, arg);
                Term body = heap_read(ctx, lam_loc + 1);
                ctx->itrs++;
                return thvm_reduce(ctx, body);
            }
            return t;
        }

        case TAG_OP2: {
            u64 loc = term_val(t);
            u32 opr = term_ext(t);
            Term x = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, x);
            Term y = thvm_reduce(ctx, heap_read(ctx, loc + 1));
            heap_set(ctx, loc + 1, y);
            if (term_tag(x) == TAG_NUM && term_tag(y) == TAG_NUM) {
                u32 xv = term_as_u32(x), yv = term_as_u32(y), r;
                switch (opr) {
                    case 0: r = xv + yv; break;
                    case 1: r = xv - yv; break;
                    case 2: r = xv * yv; break;
                    case 3: r = yv ? xv / yv : 0; break;
                    default: r = 0;
                }
                ctx->itrs++;
                return term_num_u32(r);
            }
            return t;
        }

        case TAG_VAR: {
            u64 loc = term_val(t);
            Term sub = heap_read(ctx, loc);
            if (term_is_sub(sub)) return t;
            return thvm_reduce(ctx, sub);
        }

        default:
            return t;
    }
}

// ============================================================
// print.c — Debug printer
// ============================================================

static const char *tag_names[] = {
    "APP", "LAM", "VAR", "SUP", "DP0", "DP1", "ERA",
    "NUM", "REF", "OP2", "TEN", "TOP", "CTR"
};

static const char *uop_names[] = {
    "load", "store", "copy",
    "neg", "exp", "log", "relu", "cast", "sqrt",
    "add", "mul", "div", "max", "cmp", "sub",
    "sum", "rmax",
    "mm",
    "reshape", "permute", "expand", "shrink", "pad"
};

void thvm_print_term(TinyHVM *ctx, Term t) {
    u32 tag = term_tag(t);
    (void)ctx;
    if (tag < TAG_COUNT) printf("%s", tag_names[tag]);
    else printf("?%u", tag);

    switch (tag) {
        case TAG_NUM:
            if (term_ext(t) == NUM_F32) printf("(%.4f)", term_as_f32(t));
            else printf("(%u)", term_as_u32(t));
            break;
        case TAG_TEN: {
            u32 tid = (u32)term_val(t);
            printf("(id=%u", tid);
            if (ctx && tid < ctx->tensor_count) {
                View *v = &ctx->tensors[tid].view;
                printf(" [");
                for (u32 i = 0; i < v->shape.rank; i++) printf("%s%u", i?",":"", v->shape.dims[i]);
                printf("]");
            }
            printf(")");
            break;
        }
        case TAG_TOP:
            if (term_ext(t) < UOP_COUNT) printf("(%s @%llu)", uop_names[term_ext(t)], (unsigned long long)term_val(t));
            else printf("(uop=%u @%llu)", term_ext(t), (unsigned long long)term_val(t));
            break;
        default:
            if (term_val(t) || term_ext(t))
                printf("(ext=%u, val=%llu)", term_ext(t), (unsigned long long)term_val(t));
            break;
    }
}

// ============================================================
// api.c — High-level tensor API
// ============================================================

// Device registry — both backends are always linked in.
extern Backend cpu_backend;
#ifdef __APPLE__
extern Backend metal_backend;
#endif

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
    ctx->heap = calloc(HEAP_CAP, sizeof(Term));
    ctx->heap_pos = 1;
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
    free(ctx->heap);
    free(ctx);
}

void thvm_reset(TinyHVM *ctx, u32 keep) {
    // Free ephemeral tensors (above `keep`), reset heap
    // Weight tensors [0..keep) are preserved
    for (u32 i = keep; i < ctx->tensor_count; i++) {
        if (ctx->tensors[i].host_ptr) free(ctx->tensors[i].host_ptr);
        memset(&ctx->tensors[i], 0, sizeof(TensorMeta));
    }
    // Reset backend buffer pool (frees GPU/CPU buffers above keep)
    if (ctx->backend && ctx->backend->pool_reset)
        ctx->backend->pool_reset(keep);
    ctx->tensor_count = keep;
    ctx->heap_pos = 1;  // reset heap (keep weight terms as raw IDs)
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

// Movement ops: encode shape as a 1D f32 tensor, build lazy TOP
Term thvm_reshape(TinyHVM *ctx, Term t, Shape new_shape) {
    f32 dims[MAX_DIM];
    for (u32 i = 0; i < new_shape.rank; i++) dims[i] = (f32)new_shape.dims[i];
    Term shape_t = thvm_tensor(ctx, dims, SHAPE(new_shape.rank));
    return thvm_op(ctx, UOP_RESHAPE, t, shape_t);
}

Term thvm_expand(TinyHVM *ctx, Term t, Shape new_shape) {
    f32 dims[MAX_DIM];
    for (u32 i = 0; i < new_shape.rank; i++) dims[i] = (f32)new_shape.dims[i];
    Term shape_t = thvm_tensor(ctx, dims, SHAPE(new_shape.rank));
    return thvm_op(ctx, UOP_EXPAND, t, shape_t);
}

Term thvm_permute(TinyHVM *ctx, Term t, const u32 *axes, u32 rank) {
    f32 axes_f[MAX_DIM];
    for (u32 i = 0; i < rank; i++) axes_f[i] = (f32)axes[i];
    Term axes_t = thvm_tensor(ctx, axes_f, SHAPE(rank));
    return thvm_op(ctx, UOP_PERMUTE, t, axes_t);
}

void thvm_realize(TinyHVM *ctx, Term t) {
    thvm_reduce(ctx, t);
}

f32 *thvm_to_host(TinyHVM *ctx, Term t) {
    t = thvm_reduce(ctx, t);
    if (term_tag(t) != TAG_TEN) return NULL;
    u32 id = (u32)term_val(t);
    TensorMeta *m = &ctx->tensors[id];
    if (!m->host_ptr) m->host_ptr = malloc((size_t)m->view.numel * dtype_size(m->dtype));
    if (ctx->backend) ctx->backend->buf_read(m->buf_id, m->host_ptr, (u64)m->view.numel * dtype_size(m->dtype));
    return (f32 *)m->host_ptr;
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

// Reduce-sum a tensor to target shape (for gradient accumulation after broadcast)
// Uses UOP_SUM iteratively along broadcast dimensions
static u32 tensor_reduce_sum_to(TinyHVM *ctx, u32 grad_id, Shape target) {
    TensorMeta *mg = &ctx->tensors[grad_id];

    // Check if shapes already match
    if (mg->view.shape.rank == target.rank) {
        int same = 1;
        for (u32 i = 0; i < target.rank; i++)
            if (mg->view.shape.dims[i] != target.dims[i]) same = 0;
        if (same) return grad_id;
    }

    // Reduce along each broadcast dimension using UOP_SUM
    // For now, fall back to host-side for complex shape mismatches
    // (full UOp reduce-along-axis needs axis parameter in UOP_SUM)
    u32 n_out = 1;
    for (u32 i = 0; i < target.rank; i++) n_out *= target.dims[i];

    u32 out_id = tensor_fill(ctx, target, 0.0f);
    u32 n_grad = mg->view.numel;
    u32 dsz = dtype_size(mg->dtype);
    f32 *g_data = malloc(n_grad * dsz);
    f32 *o_data = calloc(n_out, dsz);
    if (ctx->backend) ctx->backend->buf_read(mg->buf_id, g_data, n_grad * dsz);

    u32 off = mg->view.shape.rank - target.rank;
    for (u32 i = 0; i < n_grad; i++) {
        u32 out_idx = 0, rem = i, out_stride = 1;
        for (i32 d = (i32)mg->view.shape.rank - 1; d >= 0; d--) {
            u32 coord = rem % mg->view.shape.dims[d];
            rem /= mg->view.shape.dims[d];
            if ((u32)d >= off) {
                u32 td = (u32)d - off;
                u32 tc = (target.dims[td] == 1) ? 0 : coord;
                out_idx += tc * out_stride;
                out_stride *= target.dims[td];
            }
        }
        o_data[out_idx] += g_data[i];
    }

    if (ctx->backend) ctx->backend->buf_write(ctx->tensors[out_id].buf_id, o_data, n_out * dsz);
    free(g_data);
    free(o_data);
    return out_id;
}

void thvm_set_requires_grad(TinyHVM *ctx, Term t) {
    if (term_tag(t) == TAG_TEN) {
        ctx->tensors[(u32)term_val(t)].requires_grad = 1;
    }
}

void thvm_start_recording(TinyHVM *ctx) {
    ctx->recording = 1;
}

void thvm_stop_recording(TinyHVM *ctx) {
    ctx->recording = 0;
}



// ============================================================
// grad.c — Graph-level gradient (JAX-style)
// ============================================================
//
// thvm_grad(ctx, y, x) returns a lazy Term.
// When reduced, it computes ∂y/∂x.
// Because gradient ops are built with thvm_op(), they go through
// thvm_reduce → TAG_TOP dispatch → get taped if recording is on.
// This means grad(grad(f)) works: reduce the first gradient with
// recording on, then call thvm_grad again on the result.

// Accumulate gradient: grad_map[id] += new_grad (lazy ADD)
static void grad_graph_accum(TinyHVM *ctx, Term *grad_map, u32 id, Term new_grad) {
    if (!ctx->tensors[id].requires_grad) return;
    if (term_tag(grad_map[id]) == TAG_ERA) {
        grad_map[id] = new_grad;
    } else {
        grad_map[id] = thvm_op(ctx, UOP_ADD, grad_map[id], new_grad);
    }
}

Term thvm_grad(TinyHVM *ctx, Term y, Term x) {
    // Reduce y so we have tensor IDs and a tape
    y = thvm_reduce(ctx, y);
    assert(term_tag(y) == TAG_TEN && term_tag(x) == TAG_TEN);
    u32 y_id = (u32)term_val(y);
    u32 x_id = (u32)term_val(x);

    // Gradient map: tensor_id → gradient Term (ERA = no gradient yet)
    u32 tc = ctx->tensor_count;
    Term *gm = malloc(tc * sizeof(Term));
    for (u32 i = 0; i < tc; i++) gm[i] = term_era();

    // Seed: ∂y/∂y = 1 (ones with same shape as y)
    TensorMeta *my = &ctx->tensors[y_id];
    u32 ones_id = tensor_fill(ctx, my->view.shape, 1.0f);
    gm[y_id] = term_ten(ones_id, my->dtype);

    // Walk tensors backward (provenance stored in TensorMeta)
    for (i32 i = (i32)tc - 1; i >= 0; i--) {
        TensorMeta *e = &ctx->tensors[i];
        if (!e->creator_op && i != 0) continue; // not a computed tensor
        Term grad = gm[i];
        if (term_tag(grad) == TAG_ERA) continue;

        u32 uop = e->creator_op;
        u32 a_id = e->src_ids[0];
        u32 b_id = e->src_ids[1];
        int is_binary = (uop >= UOP_ADD && uop <= UOP_SUB) || uop == UOP_MM;
        TensorMeta *ma = &ctx->tensors[a_id];
        TensorMeta *mb = is_binary ? &ctx->tensors[b_id] : NULL;

        switch (uop) {
            case UOP_ADD:
                // ∂(a+b)/∂a = 1 → pass grad through
                grad_graph_accum(ctx, gm, a_id, grad);
                if (mb) grad_graph_accum(ctx, gm, b_id, grad);
                break;

            case UOP_SUB:
                // ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
                grad_graph_accum(ctx, gm, a_id, grad);
                if (mb && mb->requires_grad)
                    grad_graph_accum(ctx, gm, b_id,
                        thvm_op(ctx, UOP_NEG, grad, term_era()));
                break;

            case UOP_MUL:
                // ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
                if (ma->requires_grad)
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_op(ctx, UOP_MUL, grad,
                                term_ten(b_id, mb->dtype)));
                if (mb && mb->requires_grad)
                    grad_graph_accum(ctx, gm, b_id,
                        thvm_op(ctx, UOP_MUL, grad,
                                term_ten(a_id, ma->dtype)));
                break;

            case UOP_MM: {
                if (!mb) break;
                // z = mm(A[M,K], B[K,N])
                // ∂z/∂A = mm(grad, Bᵀ)
                // ∂z/∂B = mm(Aᵀ, grad)
                if (ma->requires_grad) {
                    u32 bt_id = tensor_transpose_2d(ctx, b_id);
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_op(ctx, UOP_MM, grad,
                                term_ten(bt_id, mb->dtype)));
                }
                if (mb && mb->requires_grad) {
                    u32 at_id = tensor_transpose_2d(ctx, a_id);
                    grad_graph_accum(ctx, gm, b_id,
                        thvm_op(ctx, UOP_MM,
                                term_ten(at_id, ma->dtype), grad));
                }
                break;
            }

            case UOP_RELU: {
                // ∂relu(a)/∂a = (a > 0) ? 1 : 0
                // mask is computed eagerly (not differentiable)
                if (ma->requires_grad) {
                    u32 zero_id = tensor_fill(ctx, ma->view.shape, 0.0f);
                    u32 mask_id = tensor_create(ctx, ma->view.shape, ma->dtype);
                    View mv, zv; u32 os[MAX_DIM], on;
                    view_broadcast(&ma->view, &ctx->tensors[zero_id].view, &mv, &zv, os, &on);
                    ctx->backend->op_binary(UOP_CMP, ctx->tensors[mask_id].buf_id,
                                        &ctx->tensors[mask_id].view,
                                        ma->buf_id, &mv,
                                        ctx->tensors[zero_id].buf_id, &zv);
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_op(ctx, UOP_MUL, grad,
                                term_ten(mask_id, ma->dtype)));
                }
                break;
            }

            case UOP_NEG: {
                // ∂(-a)/∂a = -grad
                if (ma->requires_grad)
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_op(ctx, UOP_NEG, grad, term_era()));
                break;
            }

            case UOP_EXP: {
                // ∂exp(a)/∂a = exp(a) * grad  (output = exp(a))
                if (ma->requires_grad)
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_op(ctx, UOP_MUL, grad,
                                term_ten((u32)i, e->dtype)));
                break;
            }

            case UOP_LOG: {
                // ∂log(a)/∂a = grad / a
                if (ma->requires_grad)
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_op(ctx, UOP_DIV, grad,
                                term_ten(a_id, ma->dtype)));
                break;
            }

            case UOP_SQRT: {
                // ∂sqrt(a)/∂a = grad / (2 * sqrt(a))
                if (ma->requires_grad) {
                    f32 two_val = 2.0f;
                    Term two_t = thvm_tensor(ctx, &two_val, SHAPE(1));
                    Term denom = thvm_op(ctx, UOP_MUL,
                                    two_t, term_ten((u32)i, e->dtype));
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_op(ctx, UOP_DIV, grad, denom));
                }
                break;
            }

            case UOP_DIV: {
                if (!mb) break;
                // ∂(a/b)/∂a = grad / b
                if (ma->requires_grad)
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_op(ctx, UOP_DIV, grad,
                                term_ten(b_id, mb->dtype)));
                // ∂(a/b)/∂b = -grad * a / b²
                if (mb->requires_grad) {
                    Term neg_grad = thvm_op(ctx, UOP_NEG, grad, term_era());
                    Term num = thvm_op(ctx, UOP_MUL, neg_grad,
                                       term_ten(a_id, ma->dtype));
                    Term b_sq = thvm_op(ctx, UOP_MUL,
                                        term_ten(b_id, mb->dtype),
                                        term_ten(b_id, mb->dtype));
                    grad_graph_accum(ctx, gm, b_id,
                        thvm_op(ctx, UOP_DIV, num, b_sq));
                }
                break;
            }

            case UOP_SUM: {
                // ∂sum(a)/∂a = broadcast grad to original shape
                if (ma->requires_grad) {
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_expand(ctx,
                            thvm_reshape(ctx, grad, e->view.shape),
                            ma->view.shape));
                }
                break;
            }

            case UOP_RMAX: {
                // ∂max(a)/∂a = grad * (a == max)
                // Skip for now — rarely needed directly in backprop
                break;
            }

            // === Movement op DUP rules (inverse movement) ===
            case UOP_RESHAPE: {
                // DUP through reshape = reshape grad back to original shape
                if (ma->requires_grad) {
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_reshape(ctx, grad, ma->view.shape));
                }
                break;
            }

            case UOP_EXPAND: {
                // DUP through expand = sum along expanded axes
                // (expand broadcasts dims that were 1; backward sums them back)
                if (ma->requires_grad) {
                    // For now, use tensor_reduce_sum_to to handle shape reduction
                    // This is eagerly computed, but correct
                    Term g_reduced = thvm_reduce(ctx, grad);
                    if (term_tag(g_reduced) == TAG_TEN) {
                        u32 g_id = (u32)term_val(g_reduced);
                        u32 reduced = tensor_reduce_sum_to(ctx, g_id, ma->view.shape);
                        grad_graph_accum(ctx, gm, a_id, term_ten(reduced, ma->dtype));
                    }
                }
                break;
            }

            case UOP_PERMUTE: {
                // DUP through permute = permute with inverse axes
                if (ma->requires_grad && b_id) {
                    TensorMeta *mb_p = &ctx->tensors[b_id];
                    u32 rank = mb_p->view.numel;
                    f32 *axes_f = malloc(rank * sizeof(f32));
                    ctx->backend->buf_read(mb_p->buf_id, axes_f, rank * sizeof(f32));
                    // Compute inverse permutation
                    u32 inv_axes[MAX_DIM];
                    for (u32 j = 0; j < rank; j++)
                        inv_axes[(u32)axes_f[j]] = j;
                    free(axes_f);
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_permute(ctx, grad, inv_axes, rank));
                }
                break;
            }
            // === CNN op DUP rules ===
            case UOP_IM2COL: {
                // DUP through im2col = col2im (they are inverses)
                // Gradient of im2col(x, params) w.r.t. x is col2im(grad, params)
                if (ma->requires_grad) {
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_op(ctx, UOP_COL2IM, grad,
                                term_ten(b_id, DTYPE_U32)));
                }
                break;
            }

            case UOP_COL2IM: {
                // DUP through col2im = im2col (they are inverses)
                if (ma->requires_grad) {
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_op(ctx, UOP_IM2COL, grad,
                                term_ten(b_id, DTYPE_U32)));
                }
                break;
            }

            case UOP_NCHW2NHWC: {
                // DUP through NCHW→NHWC = NHWC→NCHW (inverse layout)
                if (ma->requires_grad) {
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_op(ctx, UOP_NHWC2NCHW, grad,
                                term_ten(b_id, DTYPE_U32)));
                }
                break;
            }

            case UOP_NHWC2NCHW: {
                // DUP through NHWC→NCHW = NCHW→NHWC (inverse layout)
                if (ma->requires_grad) {
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_op(ctx, UOP_NCHW2NHWC, grad,
                                term_ten(b_id, DTYPE_U32)));
                }
                break;
            }

            case UOP_MAXPOOL: {
                // DUP through maxpool = scatter grad to argmax positions
                // This needs the argmax mask from forward — stored in b_id
                // For now, skip (maxpool backward is handled separately)
                break;
            }

            default:
                break;
        }
    }

    Term result = gm[x_id];
    free(gm);
    return result;  // lazy — reduce to get the value
}

