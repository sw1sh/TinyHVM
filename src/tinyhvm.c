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

static View view_create(const u32 *shape, u32 ndim) {
    View v = {0};
    v.ndim = ndim;
    v.numel = 1;
    for (u32 i = 0; i < ndim; i++) {
        v.shape[i] = shape[i];
        v.numel *= shape[i];
    }
    // Row-major strides
    for (u32 i = 0; i < ndim; i++) {
        i32 s = 1;
        for (u32 j = i + 1; j < ndim; j++) s *= (i32)shape[j];
        v.strides[i] = s;
    }
    v.offset = 0;
    v.contiguous = 1;
    return v;
}

static View view_permute(View v, const u32 *axes) {
    View r = v;
    for (u32 i = 0; i < v.ndim; i++) {
        r.shape[i]   = v.shape[axes[i]];
        r.strides[i] = v.strides[axes[i]];
    }
    r.contiguous = 0;
    return r;
}

static View view_expand(View v, const u32 *new_shape) {
    // Broadcast: where shape[i]==1 and new_shape[i]>1, set stride=0
    View r = v;
    r.numel = 1;
    for (u32 i = 0; i < v.ndim; i++) {
        if (v.shape[i] == 1 && new_shape[i] > 1) {
            r.shape[i] = new_shape[i];
            r.strides[i] = 0;
        }
        r.numel *= r.shape[i];
    }
    r.contiguous = 0;
    return r;
}

// Broadcast two views to a common shape. Returns 1 on success.
static int view_broadcast(const View *a, const View *b, View *out_a, View *out_b, u32 *out_shape, u32 *out_ndim) {
    // Align from the right, pad with 1s on the left
    u32 ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    *out_ndim = ndim;

    // Build padded shapes (right-aligned)
    u32 sa[MAX_DIM] = {0}, sb[MAX_DIM] = {0};
    i32 sta[MAX_DIM] = {0}, stb[MAX_DIM] = {0};
    u32 off_a = ndim - a->ndim, off_b = ndim - b->ndim;

    for (u32 i = 0; i < ndim; i++) {
        sa[i] = (i >= off_a) ? a->shape[i - off_a] : 1;
        sb[i] = (i >= off_b) ? b->shape[i - off_b] : 1;
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
    out_a->ndim = out_b->ndim = ndim;
    out_a->numel = out_b->numel = numel;
    out_a->offset = a->offset;
    out_b->offset = b->offset;
    for (u32 i = 0; i < ndim; i++) {
        out_a->shape[i] = out_b->shape[i] = out_shape[i];
        out_a->strides[i] = (sa[i] == 1 && out_shape[i] > 1) ? 0 : sta[i];
        out_b->strides[i] = (sb[i] == 1 && out_shape[i] > 1) ? 0 : stb[i];
    }
    return 1;
}

// ============================================================
// tensor.c — Tensor metadata registry
// ============================================================

static u32 tensor_create(TinyHVM *ctx, const u32 *shape, u32 ndim, u32 dtype) {
    assert(ndim <= MAX_DIM && ctx->tensor_count < MAX_TENSORS);
    u32 id = ctx->tensor_count++;
    TensorMeta *m = &ctx->tensors[id];
    memset(m, 0, sizeof(*m));
    m->dtype = dtype;
    m->refcount = 1;
    m->view = view_create(shape, ndim);

    if (ctx->gpu) {
        m->buf_id = ctx->gpu->buf_alloc((u64)m->view.numel * 4);
    }
    return id;
}

// Create a tensor that shares a buffer (for views/broadcasts)
static u32 tensor_view(TinyHVM *ctx, u32 src_id, View v) {
    assert(ctx->tensor_count < MAX_TENSORS);
    u32 id = ctx->tensor_count++;
    TensorMeta *m = &ctx->tensors[id];
    memset(m, 0, sizeof(*m));
    m->buf_id = ctx->tensors[src_id].buf_id;
    m->dtype  = ctx->tensors[src_id].dtype;
    m->refcount = 1;
    m->view = v;
    // bump refcount on original buffer's source
    ctx->tensors[src_id].refcount++;
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

            // Reduce arguments
            Term a = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, a);

            int is_binary = (uop >= UOP_ADD && uop <= UOP_SUB) || uop == UOP_MM;
            Term b = term_era();
            if (is_binary) {
                b = thvm_reduce(ctx, heap_read(ctx, loc + 1));
                heap_set(ctx, loc + 1, b);
            }

            if (term_tag(a) != TAG_TEN) return t;
            if (is_binary && term_tag(b) != TAG_TEN) return t;
            if (!ctx->gpu) return t;

            u32 a_id = (u32)term_val(a);
            u32 b_id = is_binary ? (u32)term_val(b) : 0;
            TensorMeta *ma = &ctx->tensors[a_id];
            TensorMeta *mb = is_binary ? &ctx->tensors[b_id] : NULL;

            // Determine output shape
            u32 out_shape[MAX_DIM];
            u32 out_ndim;
            View av_bc, bv_bc;  // broadcast views

            if (uop == UOP_MM) {
                // matmul: [M,K] x [K,N] → [M,N]
                assert(ma->view.ndim == 2 && mb->view.ndim == 2);
                assert(ma->view.shape[1] == mb->view.shape[0]);
                out_shape[0] = ma->view.shape[0];
                out_shape[1] = mb->view.shape[1];
                out_ndim = 2;
            } else if (is_binary) {
                // Binary: broadcast shapes
                int ok = view_broadcast(&ma->view, &mb->view, &av_bc, &bv_bc, out_shape, &out_ndim);
                assert(ok && "shape broadcast failed");
            } else {
                // Unary: output = input shape
                out_ndim = ma->view.ndim;
                for (u32 i = 0; i < out_ndim; i++) out_shape[i] = ma->view.shape[i];
            }

            u32 dst_id = tensor_create(ctx, out_shape, out_ndim, ma->dtype);
            TensorMeta *md = &ctx->tensors[dst_id];

            // Record to tape if any input requires grad
            if (ctx->recording) {
                int needs = ma->requires_grad || (mb && mb->requires_grad);
                if (needs && ctx->tape_len < MAX_TAPE) {
                    TapeEntry *e = &ctx->tape[ctx->tape_len++];
                    e->uop = uop;
                    e->out_id = dst_id;
                    e->src_ids[0] = a_id;
                    e->src_ids[1] = b_id;
                    md->requires_grad = 1;
                    md->creator_op = uop;
                    md->src_ids[0] = a_id;
                    md->src_ids[1] = b_id;
                }
            }

            // Dispatch
            if (uop == UOP_MM) {
                u32 M = ma->view.shape[0], K = ma->view.shape[1], N = mb->view.shape[1];
                ctx->gpu->op_mm(md->buf_id, ma->buf_id, &ma->view,
                                mb->buf_id, &mb->view, M, K, N);
            } else if (is_binary) {
                ctx->gpu->op_binary(uop, md->buf_id, &md->view,
                                    ma->buf_id, &av_bc,
                                    ctx->tensors[b_id].buf_id, &bv_bc);
            } else {
                ctx->gpu->op_unary(uop, md->buf_id, &md->view,
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
    "neg", "exp", "log", "relu", "cast",
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
                for (u32 i = 0; i < v->ndim; i++) printf("%s%u", i?",":"", v->shape[i]);
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

TinyHVM *thvm_init(GpuBackend *gpu) {
    TinyHVM *ctx = calloc(1, sizeof(TinyHVM));
    ctx->heap = calloc(HEAP_CAP, sizeof(Term));
    ctx->heap_pos = 1;
    ctx->gpu = gpu;
    if (gpu && gpu->init) gpu->init();
    return ctx;
}

void thvm_free(TinyHVM *ctx) {
    if (ctx->gpu && ctx->gpu->shutdown) ctx->gpu->shutdown();
    for (u32 i = 0; i < ctx->tensor_count; i++) {
        if (ctx->tensors[i].host_ptr) free(ctx->tensors[i].host_ptr);
    }
    free(ctx->heap);
    free(ctx);
}

Term thvm_tensor(TinyHVM *ctx, const f32 *data, const u32 *shape, u32 ndim) {
    u32 id = tensor_create(ctx, shape, ndim, DTYPE_F32);
    TensorMeta *m = &ctx->tensors[id];
    if (ctx->gpu && data) {
        ctx->gpu->buf_write(m->buf_id, data, (u64)m->view.numel * 4);
    }
    return term_ten(id, DTYPE_F32);
}

Term thvm_op(TinyHVM *ctx, u32 uop, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc, a);
    heap_set(ctx, loc + 1, b);
    return term_top(uop, loc);
}

void thvm_realize(TinyHVM *ctx, Term t) {
    thvm_reduce(ctx, t);
}

f32 *thvm_to_host(TinyHVM *ctx, Term t) {
    t = thvm_reduce(ctx, t);
    if (term_tag(t) != TAG_TEN) return NULL;
    u32 id = (u32)term_val(t);
    TensorMeta *m = &ctx->tensors[id];
    if (!m->host_ptr) m->host_ptr = malloc((size_t)m->view.numel * sizeof(f32));
    if (ctx->gpu) ctx->gpu->buf_read(m->buf_id, m->host_ptr, (u64)m->view.numel * 4);
    return (f32 *)m->host_ptr;
}
