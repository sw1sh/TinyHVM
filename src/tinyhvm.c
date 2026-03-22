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

static inline u32 term_tag(Term t) {
    return (u32)((t >> TAG_SHIFT) & TAG_MASK);
}

static inline u32 term_ext(Term t) {
    return (u32)((t >> EXT_SHIFT) & EXT_MASK);
}

static inline u64 term_val(Term t) {
    return t & VAL_MASK;
}

static inline Term term_set_sub(Term t) {
    return t | (1ULL << SUB_SHIFT);
}

static inline int term_is_sub(Term t) {
    return (t >> SUB_SHIFT) & 1;
}

// --- Constructors ---

static inline Term term_num_u32(u32 n) {
    return term_new(TAG_NUM, NUM_U32, (u64)n);
}

static inline Term term_num_f32(f32 f) {
    u32 bits;
    memcpy(&bits, &f, 4);
    return term_new(TAG_NUM, NUM_F32, (u64)bits);
}

static inline f32 term_as_f32(Term t) {
    u32 bits = (u32)term_val(t);
    f32 f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline u32 term_as_u32(Term t) {
    return (u32)term_val(t);
}

static inline Term term_era(void) {
    return term_new(TAG_ERA, 0, 0);
}

static inline Term term_var(u64 loc) {
    return term_new(TAG_VAR, 0, loc);
}

static inline Term term_ten(u32 buf_id, u32 dtype) {
    return term_new(TAG_TEN, dtype, (u64)buf_id);
}

static inline Term term_top(u32 uop, u64 loc) {
    return term_new(TAG_TOP, uop, loc);
}

// ============================================================
// heap.c — Single-threaded bump allocator
// ============================================================

static inline u64 heap_alloc(TinyHVM *ctx, u64 words) {
    u64 loc = ctx->heap_pos;
    ctx->heap_pos += words;
    assert(ctx->heap_pos < HEAP_CAP && "heap overflow");
    return loc;
}

static inline Term heap_read(TinyHVM *ctx, u64 loc) {
    return ctx->heap[loc];
}

static inline void heap_set(TinyHVM *ctx, u64 loc, Term t) {
    ctx->heap[loc] = t;
}

// ============================================================
// tensor.c — Tensor metadata registry
// ============================================================

static u32 tensor_create(TinyHVM *ctx, const u32 *shape, u32 ndim, u32 dtype) {
    assert(ndim <= MAX_DIM);
    assert(ctx->tensor_count < MAX_TENSORS);

    u32 id = ctx->tensor_count++;
    TensorMeta *m = &ctx->tensors[id];
    m->buf_id   = 0;  // filled by gpu->buf_alloc
    m->dtype    = dtype;
    m->ndim     = ndim;
    m->refcount = 1;
    m->host_ptr = NULL;

    u32 numel = 1;
    for (u32 i = 0; i < ndim; i++) {
        m->shape[i] = shape[i];
        numel *= shape[i];
    }
    m->numel = numel;

    // Allocate GPU buffer
    if (ctx->gpu) {
        u64 bytes = (u64)numel * 4;  // f32 = 4 bytes
        m->buf_id = ctx->gpu->buf_alloc(bytes);
    }

    return id;
}

static u32 tensor_dtype_bytes(u32 dtype) {
    switch (dtype) {
        case DTYPE_F32: return 4;
        case DTYPE_F16: return 2;
        case DTYPE_I32: return 4;
        case DTYPE_U32: return 4;
        default:        return 4;
    }
}

// ============================================================
// reduce.c — WNF reduction engine
// ============================================================

// Evaluate a term to weak normal form (WNF).
// For Phase 1, this handles:
//   APP-LAM  (beta reduction)
//   OP2-NUM  (arithmetic)
//   TOP-TEN  (tensor op dispatch)

Term thvm_reduce(TinyHVM *ctx, Term t) {
    u32 tag = term_tag(t);

    switch (tag) {
        case TAG_TOP: {
            // Tensor op node. Layout: HEAP[loc] = arg0, HEAP[loc+1] = arg1
            u32 uop = term_ext(t);
            u64 loc = term_val(t);

            // Recursively reduce arguments
            Term a = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, a);

            // Binary ops have second argument
            int is_binary = (uop >= UOP_ADD && uop <= UOP_CMP) || uop == UOP_MM;
            Term b = term_era();
            if (is_binary) {
                b = thvm_reduce(ctx, heap_read(ctx, loc + 1));
                heap_set(ctx, loc + 1, b);
            }

            // Both args must be tensors to dispatch
            if (term_tag(a) != TAG_TEN) return t;
            if (is_binary && term_tag(b) != TAG_TEN) return t;

            u32 a_id = (u32)term_val(a);
            u32 b_id = is_binary ? (u32)term_val(b) : 0;
            TensorMeta *ma = &ctx->tensors[a_id];

            if (!ctx->gpu) return t;  // no backend, can't dispatch

            // Compute output shape and allocate result tensor
            u32 out_shape[MAX_DIM];
            u32 out_ndim = ma->ndim;
            u32 out_numel = ma->numel;

            if (uop == UOP_MM) {
                // matmul: [M,K] x [K,N] → [M,N]
                TensorMeta *mb = &ctx->tensors[b_id];
                assert(ma->ndim == 2 && mb->ndim == 2);
                assert(ma->shape[1] == mb->shape[0]);
                out_shape[0] = ma->shape[0];
                out_shape[1] = mb->shape[1];
                out_ndim = 2;
                out_numel = out_shape[0] * out_shape[1];
            } else {
                for (u32 i = 0; i < out_ndim; i++)
                    out_shape[i] = ma->shape[i];
            }

            u32 dst_id = tensor_create(ctx, out_shape, out_ndim, ma->dtype);
            u32 dst_buf = ctx->tensors[dst_id].buf_id;
            u32 a_buf = ctx->tensors[a_id].buf_id;
            u32 b_buf = is_binary ? ctx->tensors[b_id].buf_id : 0;

            // Dispatch to GPU backend
            switch (uop) {
                case UOP_ADD:  ctx->gpu->op_add(dst_buf, a_buf, b_buf, out_numel); break;
                case UOP_MUL:  ctx->gpu->op_mul(dst_buf, a_buf, b_buf, out_numel); break;
                case UOP_RELU: ctx->gpu->op_relu(dst_buf, a_buf, out_numel); break;
                case UOP_NEG:  ctx->gpu->op_neg(dst_buf, a_buf, out_numel); break;
                case UOP_MM: {
                    TensorMeta *mb = &ctx->tensors[b_id];
                    ctx->gpu->op_mm(dst_buf, a_buf, b_buf,
                                    ma->shape[0], ma->shape[1], mb->shape[1]);
                    break;
                }
                default:
                    fprintf(stderr, "tinyhvm: unimplemented uop %u\n", uop);
                    break;
            }

            ctx->itrs++;
            return term_ten(dst_id, ma->dtype);
        }

        case TAG_APP: {
            // APP: HEAP[loc] = function, HEAP[loc+1] = argument
            u64 loc = term_val(t);
            Term fun = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, fun);

            if (term_tag(fun) == TAG_LAM) {
                // Beta reduction: (λx.body) arg → body[x := arg]
                u64 lam_loc = term_val(fun);
                Term arg = heap_read(ctx, loc + 1);
                // Substitute: write arg into the variable slot
                heap_set(ctx, lam_loc, arg);
                Term body = heap_read(ctx, lam_loc + 1);
                ctx->itrs++;
                return thvm_reduce(ctx, body);
            }
            return t;
        }

        case TAG_OP2: {
            // OP2: EXT = opcode, HEAP[loc] = x, HEAP[loc+1] = y
            u64 loc = term_val(t);
            u32 opr = term_ext(t);
            Term x = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, x);
            Term y = thvm_reduce(ctx, heap_read(ctx, loc + 1));
            heap_set(ctx, loc + 1, y);

            if (term_tag(x) == TAG_NUM && term_tag(y) == TAG_NUM) {
                u32 xv = term_as_u32(x);
                u32 yv = term_as_u32(y);
                u32 r;
                switch (opr) {
                    case 0:  r = xv + yv; break; // add
                    case 1:  r = xv - yv; break; // sub
                    case 2:  r = xv * yv; break; // mul
                    case 3:  r = yv ? xv / yv : 0; break; // div
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
            if (term_is_sub(sub)) {
                return t;  // unbound variable
            }
            return thvm_reduce(ctx, sub);
        }

        default:
            return t;  // already in WNF
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
    "add", "mul", "div", "max", "cmp",
    "sum", "rmax",
    "mm"
};

void thvm_print_term(TinyHVM *ctx, Term t) {
    u32 tag = term_tag(t);
    (void)ctx;

    if (tag < TAG_COUNT) {
        printf("%s", tag_names[tag]);
    } else {
        printf("?%u", tag);
    }

    switch (tag) {
        case TAG_NUM:
            if (term_ext(t) == NUM_F32)
                printf("(%.4f)", term_as_f32(t));
            else
                printf("(%u)", term_as_u32(t));
            break;
        case TAG_TEN:
            printf("(buf=%llu, dtype=%u)", term_val(t), term_ext(t));
            break;
        case TAG_TOP:
            if (term_ext(t) < UOP_COUNT)
                printf("(%s @%llu)", uop_names[term_ext(t)], term_val(t));
            else
                printf("(uop=%u @%llu)", term_ext(t), term_val(t));
            break;
        default:
            if (term_val(t) || term_ext(t))
                printf("(ext=%u, val=%llu)", term_ext(t), term_val(t));
            break;
    }
}

// ============================================================
// api.c — High-level tensor API
// ============================================================

TinyHVM *thvm_init(GpuBackend *gpu) {
    TinyHVM *ctx = calloc(1, sizeof(TinyHVM));
    ctx->heap = calloc(HEAP_CAP, sizeof(Term));
    ctx->heap_pos = 1;  // slot 0 reserved
    ctx->gpu = gpu;
    if (gpu && gpu->init) gpu->init();
    return ctx;
}

void thvm_free(TinyHVM *ctx) {
    if (ctx->gpu && ctx->gpu->shutdown) ctx->gpu->shutdown();
    // Free tensor host shadows
    for (u32 i = 0; i < ctx->tensor_count; i++) {
        if (ctx->tensors[i].host_ptr)
            free(ctx->tensors[i].host_ptr);
    }
    free(ctx->heap);
    free(ctx);
}

Term thvm_tensor(TinyHVM *ctx, const f32 *data, const u32 *shape, u32 ndim) {
    u32 id = tensor_create(ctx, shape, ndim, DTYPE_F32);
    TensorMeta *m = &ctx->tensors[id];
    if (ctx->gpu && data) {
        ctx->gpu->buf_write(m->buf_id, data, (u64)m->numel * 4);
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

    // Allocate host buffer if needed
    if (!m->host_ptr) {
        m->host_ptr = malloc((size_t)m->numel * sizeof(f32));
    }

    if (ctx->gpu) {
        ctx->gpu->buf_read(m->buf_id, m->host_ptr, (u64)m->numel * 4);
    }

    return (f32 *)m->host_ptr;
}
