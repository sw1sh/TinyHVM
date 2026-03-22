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
        r.shape.dims[i]   = v.shape.dims[axes[i]];
        r.strides[i] = v.strides[axes[i]];
    }
    r.contiguous = 0;
    return r;
}

static View view_expand(View v, const u32 *new_shape) {
    // Broadcast: where shape[i]==1 and new_shape[i]>1, set stride=0
    View r = v;
    r.numel = 1;
    for (u32 i = 0; i < v.shape.rank; i++) {
        if (v.shape.dims[i] == 1 && new_shape[i] > 1) {
            r.shape.dims[i] = new_shape[i];
            r.strides[i] = 0;
        }
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
                assert(ma->view.shape.rank == 2 && mb->view.shape.rank == 2);
                assert(ma->view.shape.dims[1] == mb->view.shape.dims[0]);
                out_shape[0] = ma->view.shape.dims[0];
                out_shape[1] = mb->view.shape.dims[1];
                out_ndim = 2;
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
                u32 M = ma->view.shape.dims[0], K = ma->view.shape.dims[1], N = mb->view.shape.dims[1];
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

Term thvm_tensor(TinyHVM *ctx, const f32 *data, Shape s) {
    u32 id = tensor_create(ctx, s, DTYPE_F32);
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
    if (ctx->gpu) ctx->gpu->buf_write(m->buf_id, tmp, (u64)n * 4);
    free(tmp);
    return id;
}

// Physical 2D transpose: create a new buffer with swapped layout
static u32 tensor_transpose_2d(TinyHVM *ctx, u32 src_id) {
    TensorMeta *ms = &ctx->tensors[src_id];
    assert(ms->view.shape.rank == 2);
    u32 M = ms->view.shape.dims[0], N = ms->view.shape.dims[1];
    u32 t_shape[] = {N, M};
    u32 tid = tensor_create(ctx, shape_of(t_shape, 2), ms->dtype);

    // Read src, transpose, write dst
    u32 n = M * N;
    f32 *src = malloc(n * sizeof(f32));
    f32 *dst = malloc(n * sizeof(f32));
    if (ctx->gpu) ctx->gpu->buf_read(ms->buf_id, src, n * 4);
    for (u32 i = 0; i < M; i++)
        for (u32 j = 0; j < N; j++)
            dst[j * M + i] = src[i * N + j];
    if (ctx->gpu) ctx->gpu->buf_write(ctx->tensors[tid].buf_id, dst, n * 4);
    free(src);
    free(dst);
    return tid;
}

// Reduce-sum a tensor to target shape (for gradient accumulation after broadcast)
static u32 tensor_reduce_sum_to(TinyHVM *ctx, u32 grad_id, Shape target) {
    TensorMeta *mg = &ctx->tensors[grad_id];

    // Check if shapes already match
    if (mg->view.shape.rank == target.rank) {
        int same = 1;
        for (u32 i = 0; i < target.rank; i++)
            if (mg->view.shape.dims[i] != target.dims[i]) same = 0;
        if (same) return grad_id;
    }

    // Sum-reduce along broadcast dimensions
    u32 n_out = 1;
    for (u32 i = 0; i < target.rank; i++) n_out *= target.dims[i];

    u32 out_id = tensor_fill(ctx, target, 0.0f);
    u32 n_grad = mg->view.numel;

    f32 *g_data = malloc(n_grad * sizeof(f32));
    f32 *o_data = calloc(n_out, sizeof(f32));
    if (ctx->gpu) ctx->gpu->buf_read(mg->buf_id, g_data, n_grad * 4);

    // Map each grad element to its target element via modular indexing
    for (u32 i = 0; i < n_grad; i++) {
        u32 out_idx = 0, rem = i, out_stride = 1;
        // Right-align target shape within grad shape
        u32 off = mg->view.shape.rank - target.rank;
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

    if (ctx->gpu) ctx->gpu->buf_write(ctx->tensors[out_id].buf_id, o_data, n_out * 4);
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

void thvm_clear_tape(TinyHVM *ctx) {
    ctx->tape_len = 0;
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

    // Walk tape backward, build lazy gradient graph
    for (i32 i = (i32)ctx->tape_len - 1; i >= 0; i--) {
        TapeEntry *e = &ctx->tape[i];
        Term grad = gm[e->out_id];
        if (term_tag(grad) == TAG_ERA) continue;

        u32 a_id = e->src_ids[0];
        u32 b_id = e->src_ids[1];
        int is_binary = (e->uop >= UOP_ADD && e->uop <= UOP_SUB) || e->uop == UOP_MM;
        TensorMeta *ma = &ctx->tensors[a_id];
        TensorMeta *mb = is_binary ? &ctx->tensors[b_id] : NULL;

        switch (e->uop) {
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
                    ctx->gpu->op_binary(UOP_CMP, ctx->tensors[mask_id].buf_id,
                                        &ctx->tensors[mask_id].view,
                                        ma->buf_id, &mv,
                                        ctx->tensors[zero_id].buf_id, &zv);
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_op(ctx, UOP_MUL, grad,
                                term_ten(mask_id, ma->dtype)));
                }
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

