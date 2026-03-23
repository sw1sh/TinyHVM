// tinyhvm.c — Hub file. Include-based single-translation-unit build.
// Inspired by HVM4's hvm.c architecture.

#include "tinyhvm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

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
    if (new_numel != v.numel) {
        fprintf(stderr, "reshape: numel mismatch old=%u (rank=%u: ", v.numel, v.shape.rank);
        for (u32 i = 0; i < v.shape.rank; i++) fprintf(stderr, "%u%s", v.shape.dims[i], i<v.shape.rank-1?",":"");
        fprintf(stderr, ") new=%u (rank=%u: ", new_numel, new_shape.rank);
        for (u32 i = 0; i < new_shape.rank; i++) fprintf(stderr, "%u%s", new_shape.dims[i], i<new_shape.rank-1?",":"");
        fprintf(stderr, ")\n");
    }
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
        u64 bytes = (u64)m->view.numel * dtype_size(dtype);
        m->buf_id = ctx->backend->buf_alloc(bytes);
    }
    thvm_prof_tensor_created(0);
    thvm_prof_update_watermarks(ctx->tensor_count, ctx->heap_pos);
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
// fusion.c — IC op fusion via intermediate accumulator nodes
// ============================================================
// See resources/ic_fusion.md for design.
// FUSING node absorbs fuseable ops pairwise, emits one fused dispatch.

#define MAX_FUSE_OPS 16
#define MAX_FUSE_INPUTS 4

typedef struct {
    u32 ops[MAX_FUSE_OPS];       // op chain (outermost first)
    u32 n_ops;
    u32 input_ids[MAX_FUSE_INPUTS]; // materialized input tensor IDs
    u32 n_inputs;
    Shape out_shape;              // shape of the final output
    Shape reduce_src_shape;       // shape before reduce (for reduce_dim)
    int has_reduce;
} FuseState;

static int is_elementwise_uop(u32 uop) {
    return uop == UOP_RELU || uop == UOP_NEG || uop == UOP_EXP ||
           uop == UOP_LOG || uop == UOP_SQRT ||
           uop == UOP_ADD || uop == UOP_MUL || uop == UOP_DIV || uop == UOP_SUB;
}

// Walk the unreduced graph pairwise, absorbing fuseable ops.
// Returns 1 if fusion found, 0 if not.
static int try_fuse(TinyHVM *ctx, Term t, FuseState *fs) {
    if (term_tag(t) != TAG_TOP) return 0;
    u32 uop = term_ext(t);
    u64 loc = term_val(t);

    int is_reduce = (uop == UOP_SUM || uop == UOP_RMAX);
    int is_elem = is_elementwise_uop(uop);
    int is_binary_op = (uop >= UOP_ADD && uop <= UOP_SUB);

    // Can't fuse: not elementwise or reduce
    if (!is_elem && !is_reduce) return 0;
    // Can't fuse: binary ops need broadcast-aware indexing (future work)
    if (is_binary_op) return 0;
    // Can't fuse: second reduce
    if (is_reduce && fs->has_reduce) return 0;
    // Overflow
    if (fs->n_ops >= MAX_FUSE_OPS) return 0;

    // Absorb this op
    fs->ops[fs->n_ops++] = uop;
    if (is_reduce) fs->has_reduce = 1;

    // Peek at child_a (DON'T reduce yet)
    Term child_a = heap_read(ctx, loc);

    // For binary ops, the second input must already be materialized
    // AND have the same numel (no broadcasting — we don't handle strided fusion yet)
    if (is_binary_op) {
        Term child_b = heap_read(ctx, loc + 1);
        Term b_reduced = thvm_reduce(ctx, child_b);
        heap_set(ctx, loc + 1, b_reduced);
        if (term_tag(b_reduced) != TAG_TEN) return 0;
        u32 b_id = (u32)term_val(b_reduced);
        TensorMeta *mb = &ctx->tensors[b_id];
        // Bail if any stride is 0 (expanded view) or if shapes won't match
        for (u32 d = 0; d < mb->view.shape.rank; d++) {
            if (mb->view.strides[d] == 0 && mb->view.shape.dims[d] > 1) return 0;
        }
        if (fs->n_inputs >= MAX_FUSE_INPUTS) return 0;
        fs->input_ids[fs->n_inputs++] = b_id;
    }

    // Try to absorb child_a recursively
    if (term_tag(child_a) == TAG_TOP && try_fuse(ctx, child_a, fs)) {
        return 1; // chain continues deeper
    }

    // child_a is a leaf — reduce it and mark as input
    Term a_reduced = thvm_reduce(ctx, child_a);
    heap_set(ctx, loc, a_reduced);
    if (term_tag(a_reduced) != TAG_TEN) return 0;
    u32 a_id = (u32)term_val(a_reduced);
    TensorMeta *ma_leaf = &ctx->tensors[a_id];
    // Bail if primary input has stride=0 (expanded view)
    for (u32 d = 0; d < ma_leaf->view.shape.rank; d++) {
        if (ma_leaf->view.strides[d] == 0 && ma_leaf->view.shape.dims[d] > 1) return 0;
    }
    if (fs->n_inputs >= MAX_FUSE_INPUTS) return 0;
    // Primary input goes at the START
    // Shift existing inputs right, insert at 0
    for (int i = (int)fs->n_inputs; i > 0; i--)
        fs->input_ids[i] = fs->input_ids[i-1];
    fs->input_ids[0] = a_id;
    fs->n_inputs++;

    return fs->n_ops >= 2; // fuse only if >1 op absorbed
}

// Execute fused op chain on CPU without intermediate buffers.
// ops[] is outermost-first: e.g. [SUM, MUL, RELU]
// means: SUM(MUL(input_a, input_b) where input_a = RELU(primary_input))
static Term dispatch_fused_cpu(TinyHVM *ctx, FuseState *fs) {
    // Primary input is input_ids[0], secondary inputs are [1..n_inputs-1]
    u32 primary_id = fs->input_ids[0];
    TensorMeta *mp = &ctx->tensors[primary_id];
    u32 n_primary = mp->view.numel;

    // Read primary input to host
    f32 *primary = (f32 *)thvm_to_host(ctx, term_ten(primary_id, mp->dtype));
    f32 *primary_copy = malloc(n_primary * sizeof(f32));
    memcpy(primary_copy, primary, n_primary * sizeof(f32));

    // Read secondary inputs
    f32 *secondary[MAX_FUSE_INPUTS - 1] = {0};
    for (u32 i = 1; i < fs->n_inputs; i++) {
        u32 sid = fs->input_ids[i];
        TensorMeta *ms = &ctx->tensors[sid];
        f32 *s = (f32 *)thvm_to_host(ctx, term_ten(sid, ms->dtype));
        secondary[i-1] = malloc(ms->view.numel * sizeof(f32));
        memcpy(secondary[i-1], s, ms->view.numel * sizeof(f32));
    }

    // Determine output shape
    // If has_reduce: reduce collapses the primary input
    u32 out_numel;
    Shape out_shape;
    u32 reduce_dim = 1;
    if (fs->has_reduce) {
        // Find the reduce dim (innermost non-1 dim)
        out_shape = mp->view.shape;
        for (int d = (int)out_shape.rank - 1; d >= 0; d--) {
            if (out_shape.dims[d] > 1) {
                reduce_dim = out_shape.dims[d];
                out_shape.dims[d] = 1;
                break;
            }
        }
        out_numel = 1;
        for (u32 i = 0; i < out_shape.rank; i++) out_numel *= out_shape.dims[i];
    } else {
        out_shape = mp->view.shape;
        out_numel = n_primary;
    }

    // Allocate output
    u32 out_id = tensor_create(ctx, out_shape, DTYPE_F32);
    f32 *output = malloc(out_numel * sizeof(f32));

    // Execute fused loop
    u32 sec_idx = 0; // index into secondary inputs for binary ops
    if (fs->has_reduce) {
        u32 outer = out_numel;
        for (u32 i = 0; i < outer; i++) {
            f32 acc = 0;
            for (u32 j = 0; j < reduce_dim; j++) {
                u32 flat = i * reduce_dim + j;
                f32 v = primary_copy[flat < n_primary ? flat : 0];

                // Apply elementwise ops (innermost-first = reversed)
                sec_idx = 0;
                for (int k = (int)fs->n_ops - 1; k >= 0; k--) {
                    switch (fs->ops[k]) {
                        case UOP_RELU: v = v > 0 ? v : 0; break;
                        case UOP_NEG:  v = -v; break;
                        case UOP_EXP:  v = expf(v); break;
                        case UOP_LOG:  v = logf(v); break;
                        case UOP_SQRT: v = sqrtf(v); break;
                        case UOP_MUL:
                            if (secondary[sec_idx])
                                v *= secondary[sec_idx][flat < n_primary ? flat : 0];
                            sec_idx++;
                            break;
                        case UOP_ADD:
                            if (secondary[sec_idx])
                                v += secondary[sec_idx][flat < n_primary ? flat : 0];
                            sec_idx++;
                            break;
                        case UOP_DIV:
                            if (secondary[sec_idx])
                                v /= secondary[sec_idx][flat < n_primary ? flat : 0];
                            sec_idx++;
                            break;
                        case UOP_SUB:
                            if (secondary[sec_idx])
                                v -= secondary[sec_idx][flat < n_primary ? flat : 0];
                            sec_idx++;
                            break;
                        case UOP_SUM:  break; // handled outside
                        case UOP_RMAX: break;
                    }
                }
                // Accumulate for reduce
                if (fs->ops[0] == UOP_SUM) acc += v;
                else if (fs->ops[0] == UOP_RMAX) acc = (j == 0) ? v : (v > acc ? v : acc);
            }
            output[i] = acc;
        }
    } else {
        // Pure elementwise fusion (no reduce)
        for (u32 i = 0; i < out_numel; i++) {
            f32 v = primary_copy[i];
            sec_idx = 0;
            for (int k = (int)fs->n_ops - 1; k >= 0; k--) {
                switch (fs->ops[k]) {
                    case UOP_RELU: v = v > 0 ? v : 0; break;
                    case UOP_NEG:  v = -v; break;
                    case UOP_EXP:  v = expf(v); break;
                    case UOP_LOG:  v = logf(v); break;
                    case UOP_SQRT: v = sqrtf(v); break;
                    case UOP_MUL:
                        if (secondary[sec_idx]) v *= secondary[sec_idx][i];
                        sec_idx++;
                        break;
                    case UOP_ADD:
                        if (secondary[sec_idx]) v += secondary[sec_idx][i];
                        sec_idx++;
                        break;
                    case UOP_DIV:
                        if (secondary[sec_idx]) v /= secondary[sec_idx][i];
                        sec_idx++;
                        break;
                    case UOP_SUB:
                        if (secondary[sec_idx]) v -= secondary[sec_idx][i];
                        sec_idx++;
                        break;
                    default: break;
                }
            }
            output[i] = v;
        }
    }

    // Write output to backend
    if (ctx->backend)
        ctx->backend->buf_write(ctx->tensors[out_id].buf_id, output, out_numel * sizeof(f32));

    // Cleanup
    free(primary_copy);
    free(output);
    for (u32 i = 0; i < MAX_FUSE_INPUTS - 1; i++) free(secondary[i]);

    ctx->itrs++;
    return term_ten(out_id, DTYPE_F32);
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

            // === IC FUSION: try to absorb elementwise chain into reduce ===
            // Opt-in via THVM_FUSE=1 env var (prototype)
            // Skip during gradient recording — gradient graphs are complex
            if ((uop == UOP_SUM || uop == UOP_RMAX) && !ctx->recording && getenv("THVM_FUSE")) {
                FuseState fs = {0};
                if (try_fuse(ctx, t, &fs) && fs.n_ops >= 2) {
                    return dispatch_fused_cpu(ctx, &fs);
                }
            }

            // Movement ops: modify View, share buffer
            int is_movement = (uop >= UOP_RESHAPE && uop <= UOP_PAD);

            // Reduce arguments
            Term a = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, a);

            int is_binary = (uop >= UOP_ADD && uop <= UOP_SUB) || uop == UOP_MM;
            int is_reduce = (uop == UOP_SUM || uop == UOP_RMAX);
            Term b = term_era();
            if (is_binary || is_movement) {
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
                    case UOP_SHRINK: {
                        // b is a 1D tensor: [start0, end0, start1, end1, ...]
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        u32 n_pairs = mb->view.numel;
                        f32 *pairs = malloc(n_pairs * sizeof(f32));
                        ctx->backend->buf_read(mb->buf_id, pairs, n_pairs * sizeof(f32));
                        u32 starts[MAX_DIM], ends[MAX_DIM];
                        for (u32 i = 0; i < n_pairs / 2; i++) {
                            starts[i] = (u32)pairs[i * 2];
                            ends[i]   = (u32)pairs[i * 2 + 1];
                        }
                        free(pairs);
                        new_view = view_shrink(ma->view, starts, ends);
                        break;
                    }
                    case UOP_PAD: {
                        // b is a 1D tensor: [before0, after0, before1, after1, ...]
                        // PAD requires physical copy: alloc zeroed buffer, copy src into it
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        u32 n_pairs = mb->view.numel;
                        f32 *pairs = malloc(n_pairs * sizeof(f32));
                        ctx->backend->buf_read(mb->buf_id, pairs, n_pairs * sizeof(f32));
                        u32 pad_before[MAX_DIM], pad_after[MAX_DIM];
                        for (u32 i = 0; i < n_pairs / 2; i++) {
                            pad_before[i] = (u32)pairs[i * 2];
                            pad_after[i]  = (u32)pairs[i * 2 + 1];
                        }
                        free(pairs);

                        // Compute padded shape
                        Shape ps = {.rank = ma->view.shape.rank};
                        u32 numel = 1;
                        for (u32 i = 0; i < ps.rank; i++) {
                            ps.dims[i] = ma->view.shape.dims[i] + pad_before[i] + pad_after[i];
                            numel *= ps.dims[i];
                        }

                        // Allocate zeroed output
                        u32 dst_id = tensor_create(ctx, ps, ma->dtype);
                        TensorMeta *md = &ctx->tensors[dst_id];

                        if (ctx->backend) {
                            u32 src_numel = ma->view.numel;
                            f32 *src_data = malloc(src_numel * sizeof(f32));
                            f32 *dst_data = calloc(numel, sizeof(f32));
                            ctx->backend->buf_read(ma->buf_id, src_data, src_numel * sizeof(f32));

                            // Scatter src into padded dst at offset
                            for (u32 flat = 0; flat < src_numel; flat++) {
                                u32 coords[MAX_DIM], rem = flat;
                                for (int d = (int)ma->view.shape.rank - 1; d >= 0; d--) {
                                    coords[d] = rem % ma->view.shape.dims[d];
                                    rem /= ma->view.shape.dims[d];
                                }
                                u32 dst_idx = 0, stride = 1;
                                for (int d = (int)ps.rank - 1; d >= 0; d--) {
                                    dst_idx += (coords[d] + pad_before[d]) * stride;
                                    stride *= ps.dims[d];
                                }
                                dst_data[dst_idx] = src_data[flat];
                            }

                            ctx->backend->buf_write(md->buf_id, dst_data, numel * sizeof(f32));
                            free(src_data);
                            free(dst_data);
                        }

                        // Record provenance
                        if (ctx->recording && ma->requires_grad) {
                            md->requires_grad = 1;
                            md->creator_op = uop;
                            md->src_ids[0] = a_id;
                            md->src_ids[1] = b_id;
                        }
                        ctx->itrs++;
                        return term_ten(dst_id, ma->dtype);
                    }
                    default:
                        assert(0 && "unknown movement op");
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
            } else if (is_reduce) {
                // Reduce: find the innermost non-1 dim to collapse
                out_ndim = ma->view.shape.rank;
                for (u32 i = 0; i < out_ndim; i++) out_shape[i] = ma->view.shape.dims[i];
                // Find the reduce axis: last dim with size > 1
                int reduce_axis = -1;
                for (int i = (int)out_ndim - 1; i >= 0; i--) {
                    if (out_shape[i] > 1) { reduce_axis = i; break; }
                }
                if (reduce_axis >= 0) {
                    out_shape[reduce_axis] = 1;
                } else {
                    // All dims are 1 — nothing to reduce, output = input
                    out_shape[out_ndim - 1] = 1;
                }
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
            } else if (is_reduce) {
                // reduce_dim = product of dims from reduce_axis to end
                // The kernel sees flat [outer × reduce_dim]
                u32 reduce_dim = 1;
                int ra = -1;
                for (int i = (int)ma->view.shape.rank - 1; i >= 0; i--) {
                    reduce_dim *= ma->view.shape.dims[i];
                    if (ma->view.shape.dims[i] > 1 && ra < 0) ra = i;
                    if (ra >= 0 && i <= ra) break;
                }
                // If we found a non-1 dim, reduce_dim is the product from that dim to end
                if (ra >= 0) {
                    reduce_dim = 1;
                    for (u32 i = (u32)ra; i < ma->view.shape.rank; i++)
                        reduce_dim *= ma->view.shape.dims[i];
                }
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

// uop_names now in tinyhvm.h

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
        if (ctx->recording && m->requires_grad) ctx->tensors[id].requires_grad = 1;
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
        assert(v->shape.rank == new_shape.rank);
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
        if (ctx->recording && m->requires_grad) ctx->tensors[id].requires_grad = 1;
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
        if (ctx->recording && m->requires_grad) ctx->tensors[id].requires_grad = 1;
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
        if (ctx->recording && m->requires_grad) ctx->tensors[id].requires_grad = 1;
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
        Shape ns = shape_of(new_dims, ndim);
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
        if (ctx->recording && m->requires_grad) ctx->tensors[id].requires_grad = 1;
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
        // Differentiable path using only traceable UOps:
        // For each spatial dim j with k > s:
        //   Create k_j shifted copies via shrink, each of size o_j
        //   Stack along new kernel dimension
        //
        // For conv2d (s=1): x[b,c, oh+kh, ow+kw] for each (kh,kw)
        // = shrink x along spatial dims to [oh+kh : oh+kh + 1] for each kernel pos
        //
        // Strategy: process one spatial dim at a time.
        // For dim j: reshape to (..., i_j) then for each k in 0..k_j-1,
        //   shrink to [k, k+o_j) to get shifted copy of size o_j
        //   pad to full size i_j with zeros, accumulate via reshape
        // Finally reshape to (..., o_j, k_j)
        //
        // Simpler: for 2D conv (the common case), create the full [BS, Cin, OH, OW, KH, KW]
        // tensor by iterating over kernel positions and writing shrunk slices.
        // Each shrink IS differentiable (has gradient rule now).

        // For 2D spatial: iterate over (kh, kw), shrink input for each position
        assert(n_spatial == 2);  // 2D pooling for now
        u32 bs_dims[MAX_DIM];
        for (u32 j = 0; j < bd; j++) bs_dims[j] = mx->view.shape.dims[j];

        // We need: result[batch, o0, o1, k0, k1] = x[batch, o0*s0+k0, o1*s1+k1]
        // Build by creating k0*k1 shrunk slices and concatenating

        // Output shape: [batch_dims..., o0, o1, k0, k1]
        u32 out_dims[MAX_DIM], out_rank = ndim + n_spatial;
        for (u32 j = 0; j < bd; j++) out_dims[j] = mx->view.shape.dims[j];
        for (u32 j = 0; j < n_spatial; j++) {
            out_dims[bd + j]             = o_[j];
            out_dims[bd + n_spatial + j] = k_[j];
        }

        // For each kernel position (kh, kw), produce a shrunk view of x
        // with shape [batch, o0, o1] and place it at position [kh, kw]
        // in the kernel dimensions.
        //
        // In the output tensor [batch, o0, o1, k0, k1]:
        //   element [b, oh, ow, kh, kw] = x[b, oh*s0+kh, ow*s1+kw]
        //
        // We create this by doing k0*k1 shrink ops and summing into the right spot.
        // Each shrink produces [batch, o0, o1].
        // Reshape to [batch, o0, o1, 1, 1], pad to [batch, o0, o1, k0, k1] at (kh,kw).

        // Actually simplest: build entire output on CPU using shrink results.
        // Each shrink IS a traceable UOp. Read shrunk data, write to output position.

        // Even simpler and fully lazy: build the output as a sum of padded shrinks.
        // For each (kh,kw): shrink → reshape to [batch,o0,o1,1,1] → pad to place at (kh,kw)
        // Then sum all k0*k1 terms. But that's really wasteful in memory.

        // Most practical: CPU gather but WITH proper provenance tracking.
        // Store source tensor + params, implement scatter-add backward.

        // Since we can't avoid materialization for overlapping windows without
        // a dedicated unfolding UOp, materialize on CPU but record provenance.
        u32 out_numel = 1;
        for (u32 j = 0; j < out_rank; j++) out_numel *= out_dims[j];

        Shape os = {.rank = out_rank};
        for (u32 j = 0; j < out_rank; j++) os.dims[j] = out_dims[j];
        u32 dst_id = tensor_create(ctx, os, mx->dtype);

        if (ctx->backend) {
            u32 src_numel = mx->view.numel;
            f32 *src = malloc(src_numel * sizeof(f32));
            f32 *dst = malloc(out_numel * sizeof(f32));
            ctx->backend->buf_read(mx->buf_id, src, src_numel * sizeof(f32));

            for (u32 flat = 0; flat < out_numel; flat++) {
                u32 coords[MAX_DIM], rem = flat;
                for (int d = (int)out_rank - 1; d >= 0; d--) {
                    coords[d] = rem % out_dims[d];
                    rem /= out_dims[d];
                }
                u32 src_idx = 0, src_stride = 1;
                for (int d = (int)ndim - 1; d >= 0; d--) {
                    u32 coord;
                    if ((u32)d < bd) {
                        coord = coords[d];
                    } else {
                        u32 si = (u32)d - bd;
                        coord = coords[bd + si] * s_[si] + coords[bd + n_spatial + si];
                    }
                    src_idx += coord * src_stride;
                    src_stride *= mx->view.shape.dims[d];
                }
                dst[flat] = src[src_idx];
            }

            ctx->backend->buf_write(ctx->tensors[dst_id].buf_id, dst, out_numel * sizeof(f32));
            free(src);
            free(dst);
        }

        // Record provenance: custom POOL op for autograd
        // Store source id + pool params (n_spatial, kernel, stride, input spatial dims)
        ctx->tensors[dst_id].creator_op = UOP_POOL_GATHER;
        ctx->tensors[dst_id].src_ids[0] = (u32)term_val(xr);
        // Store pool params: [n_spatial, k0, k1, s0, s1, i0, i1, o0, o1, bd]
        f32 params[MAX_DIM * 2];  // needs 1 + 4*n_spatial + 1 entries
        params[0] = (f32)n_spatial;
        for (u32 j = 0; j < n_spatial; j++) {
            params[1 + j]               = (f32)k_[j];
            params[1 + n_spatial + j]    = (f32)s_[j];
            params[1 + 2*n_spatial + j]  = (f32)i_[j];
            params[1 + 3*n_spatial + j]  = (f32)o_[j];
        }
        params[1 + 4*n_spatial] = (f32)bd;
        u32 plen = 2 + 4*n_spatial;
        Term pt = thvm_tensor(ctx, params, SHAPE(plen));
        ctx->tensors[dst_id].src_ids[1] = (u32)term_val(pt);

        if (ctx->recording && mx->requires_grad) {
            ctx->tensors[dst_id].requires_grad = 1;
        }
        ctx->itrs++;
        return term_ten(dst_id, mx->dtype);
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
    Term pooled = thvm_pool(ctx, padded, k, s, 2);
    // pooled: [BS, Cin, OY, OX, KH, KW]

    // Get output spatial dims
    Term pr = thvm_reduce(ctx, pooled);
    TensorMeta *mp = &ctx->tensors[(u32)term_val(pr)];
    u32 oy = mp->view.shape.dims[2];
    u32 ox = mp->view.shape.dims[3];

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

    // Step 5: multiply + sum over (cin_g, KH, KW) = last 3 dims
    Term prod = thvm_op(ctx, UOP_MUL, x_perm, w_rs);

    // Sum reduces last dim to 1 (keeps rank). Reshape to drop trailing 1.
    // prod: [BS, groups, rcout, OY, OX, cin_g, KH, KW]
    Term s1 = thvm_op(ctx, UOP_SUM, prod, term_era());
    // → [BS, groups, rcout, OY, OX, cin_g, KH, 1], squeeze:
    s1 = thvm_reshape(ctx, s1, shape_of((u32[]){bs, groups, rcout, oy, ox, cin_g, KH}, 7));

    Term s2 = thvm_op(ctx, UOP_SUM, s1, term_era());
    // → [BS, groups, rcout, OY, OX, cin_g, 1], squeeze:
    s2 = thvm_reshape(ctx, s2, shape_of((u32[]){bs, groups, rcout, oy, ox, cin_g}, 6));

    Term s3 = thvm_op(ctx, UOP_SUM, s2, term_era());
    // → [BS, groups, rcout, OY, OX, 1], squeeze:
    s3 = thvm_reshape(ctx, s3, shape_of((u32[]){bs, groups, rcout, oy, ox}, 5));

    // Reshape to [BS, Cout, OY, OX]
    Term out = thvm_reshape(ctx, s3, shape_of((u32[]){bs, cout, oy, ox}, 4));

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
    // pooled = _pool(x, kernel, stride)
    // return pooled.max(kernel_axes)
    u32 k[] = {kernel[0], kernel[1]};
    u32 s[] = {stride_[0], stride_[1]};
    Term pooled = thvm_pool(ctx, x, k, s, 2);
    // pooled shape: [BS, C, OY, OX, KH, KW]
    // reduce max over last 2 dims (KH, KW), squeezing between

    // Get pool dims
    Term pr = thvm_reduce(ctx, pooled);
    TensorMeta *mp = &ctx->tensors[(u32)term_val(pr)];
    u32 bs = mp->view.shape.dims[0];
    u32 c  = mp->view.shape.dims[1];
    u32 oy = mp->view.shape.dims[2];
    u32 ox = mp->view.shape.dims[3];
    u32 kh = mp->view.shape.dims[4];

    Term r1 = thvm_op(ctx, UOP_RMAX, pr, term_era());
    // → [BS, C, OY, OX, KH, 1], squeeze:
    r1 = thvm_reshape(ctx, r1, shape_of((u32[]){bs, c, oy, ox, kh}, 5));

    Term r2 = thvm_op(ctx, UOP_RMAX, r1, term_era());
    // → [BS, C, OY, OX, 1], squeeze:
    return thvm_reshape(ctx, r2, shape_of((u32[]){bs, c, oy, ox}, 4));
}

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
    u32 src_numel = 1;
    for (u32 d = 0; d < m->view.shape.rank; d++) {
        u32 dim_extent = m->view.shape.dims[d];
        i32 stride = m->view.strides[d];
        if (stride != 0) {
            u32 end_idx = m->view.offset + (dim_extent - 1) * (u32)stride;
            if (end_idx + 1 > src_numel) src_numel = end_idx + 1;
        }
    }
    // Fallback: compute from buffer if we can't determine exact extents
    if (src_numel == 0) src_numel = 1;

    f32 *src_buf = malloc((size_t)src_numel * sizeof(f32));
    if (ctx->backend) ctx->backend->buf_read(m->buf_id, src_buf,
                                              (u64)src_numel * sizeof(f32));

    if (!m->host_ptr) m->host_ptr = malloc((size_t)m->view.numel * sizeof(f32));
    f32 *dst = (f32 *)m->host_ptr;

    // Strided copy
    for (u32 flat = 0; flat < m->view.numel; flat++) {
        u32 rem = flat;
        u32 src_idx = m->view.offset;
        for (i32 d = (i32)m->view.shape.rank - 1; d >= 0; d--) {
            u32 coord = rem % m->view.shape.dims[d];
            rem /= m->view.shape.dims[d];
            src_idx += coord * (u32)m->view.strides[d];
        }
        dst[flat] = src_buf[src_idx];
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

    u32 n_out = 1;
    for (u32 i = 0; i < target.rank; i++) n_out *= target.dims[i];

    u32 out_id = tensor_fill(ctx, target, 0.0f);
    u32 n_grad = mg->view.numel;
    u32 dsz = dtype_size(mg->dtype);

    // Read grad values via strided access (handles expanded views)
    f32 *g_data = (f32 *)thvm_to_host(ctx, term_ten(grad_id, mg->dtype));
    // thvm_to_host may have modified the tensor's host_ptr, copy it
    f32 *g_copy = malloc(n_grad * dsz);
    memcpy(g_copy, g_data, n_grad * dsz);

    f32 *o_data = calloc(n_out, dsz);

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
        o_data[out_idx] += g_copy[i];
    }

    if (ctx->backend) ctx->backend->buf_write(ctx->tensors[out_id].buf_id, o_data, n_out * dsz);
    free(g_copy);
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
            case UOP_ADD: {
                // ∂(a+b)/∂a = 1 → pass grad through, reduce if broadcast
                Term gr = thvm_reduce(ctx, grad);
                if (ma->requires_grad) {
                    Term ga = gr;
                    if (term_tag(gr) == TAG_TEN) {
                        u32 g_id = (u32)term_val(gr);
                        if (ctx->tensors[g_id].view.numel != ma->view.numel) {
                            u32 r_id = tensor_reduce_sum_to(ctx, g_id, ma->view.shape);
                            ga = term_ten(r_id, ma->dtype);
                        }
                    }
                    grad_graph_accum(ctx, gm, a_id, ga);
                }
                if (mb && mb->requires_grad) {
                    Term gb = gr;
                    if (term_tag(gr) == TAG_TEN) {
                        u32 g_id = (u32)term_val(gr);
                        if (ctx->tensors[g_id].view.numel != mb->view.numel) {
                            u32 r_id = tensor_reduce_sum_to(ctx, g_id, mb->view.shape);
                            gb = term_ten(r_id, mb->dtype);
                        }
                    }
                    grad_graph_accum(ctx, gm, b_id, gb);
                }
                break;
            }

            case UOP_SUB: {
                // ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
                Term gr = thvm_reduce(ctx, grad);
                if (ma->requires_grad) {
                    Term ga = gr;
                    if (term_tag(gr) == TAG_TEN) {
                        u32 g_id = (u32)term_val(gr);
                        if (ctx->tensors[g_id].view.numel != ma->view.numel) {
                            u32 r_id = tensor_reduce_sum_to(ctx, g_id, ma->view.shape);
                            ga = term_ten(r_id, ma->dtype);
                        }
                    }
                    grad_graph_accum(ctx, gm, a_id, ga);
                }
                if (mb && mb->requires_grad) {
                    Term neg_grad = thvm_op(ctx, UOP_NEG, gr, term_era());
                    Term ngr = thvm_reduce(ctx, neg_grad);
                    if (term_tag(ngr) == TAG_TEN) {
                        u32 g_id = (u32)term_val(ngr);
                        if (ctx->tensors[g_id].view.numel != mb->view.numel) {
                            u32 r_id = tensor_reduce_sum_to(ctx, g_id, mb->view.shape);
                            ngr = term_ten(r_id, mb->dtype);
                        }
                    }
                    grad_graph_accum(ctx, gm, b_id, ngr);
                }
                break;
            }

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
                        thvm_expand(ctx, grad, ma->view.shape));
                }
                break;
            }

            case UOP_RMAX: {
                // ∂max(a)/∂a = grad * (a == max_value)
                // max_value is the output (tensor i), broadcast back to input shape
                if (ma->requires_grad) {
                    // Expand max result back to input shape, compare to get mask
                    Term max_bc = thvm_expand(ctx,
                        thvm_reshape(ctx, term_ten((u32)i, e->dtype), e->view.shape),
                        ma->view.shape);
                    // mask = (a == max)
                    Term mask = thvm_op(ctx, UOP_CMP, term_ten(a_id, ma->dtype), max_bc);
                    // grad * mask, broadcast grad to input shape
                    Term grad_bc = thvm_expand(ctx, grad, ma->view.shape);
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_op(ctx, UOP_MUL, grad_bc, mask));
                }
                break;
            }

            case UOP_PAD: {
                // ∂pad(a)/∂a = shrink(grad) — remove the padded regions
                if (ma->requires_grad && b_id) {
                    TensorMeta *mb_pad = &ctx->tensors[b_id];
                    u32 ndim_p = mb_pad->view.numel / 2;
                    f32 *pf = malloc(mb_pad->view.numel * sizeof(f32));
                    ctx->backend->buf_read(mb_pad->buf_id, pf, mb_pad->view.numel * sizeof(f32));
                    // Shrink pairs: start=pad_before, end=pad_before+original_dim
                    u32 shrink_p[MAX_DIM * 2];
                    for (u32 j = 0; j < ndim_p; j++) {
                        u32 pad_before = (u32)pf[j * 2];
                        shrink_p[j * 2]     = pad_before;
                        shrink_p[j * 2 + 1] = pad_before + ma->view.shape.dims[j];
                    }
                    free(pf);
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_shrink(ctx, grad, shrink_p, ndim_p));
                }
                break;
            }

            case UOP_SHRINK: {
                // ∂shrink(a)/∂a = pad(grad) — pad with zeros to restore original shape
                if (ma->requires_grad && b_id) {
                    TensorMeta *mb_sh = &ctx->tensors[b_id];
                    u32 ndim_s = mb_sh->view.numel / 2;
                    f32 *sf = malloc(mb_sh->view.numel * sizeof(f32));
                    ctx->backend->buf_read(mb_sh->buf_id, sf, mb_sh->view.numel * sizeof(f32));
                    // Pad pairs: before=shrink_start, after=original_dim-shrink_end
                    u32 pad_p[MAX_DIM * 2];
                    for (u32 j = 0; j < ndim_s; j++) {
                        u32 s_start = (u32)sf[j * 2];
                        u32 s_end   = (u32)sf[j * 2 + 1];
                        pad_p[j * 2]     = s_start;
                        pad_p[j * 2 + 1] = ma->view.shape.dims[j] - s_end;
                    }
                    free(sf);
                    grad_graph_accum(ctx, gm, a_id,
                        thvm_pad(ctx, grad, pad_p, ndim_s));
                }
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

            case UOP_POOL_GATHER: {
                // ∂pool_gather(a)/∂a = scatter-add (col2im equivalent)
                // Forward gathered: out[batch, o0, o1, k0, k1] = in[batch, o0*s0+k0, o1*s1+k1]
                // Backward: scatter-add gradient back to input shape
                if (ma->requires_grad && b_id) {
                    TensorMeta *mb_pg = &ctx->tensors[b_id];
                    u32 plen = mb_pg->view.numel;
                    f32 *pf = malloc(plen * sizeof(f32));
                    ctx->backend->buf_read(mb_pg->buf_id, pf, plen * sizeof(f32));
                    u32 ns = (u32)pf[0];  // n_spatial
                    u32 pk[2], ps[2], pi[2], po[2], pbd;
                    for (u32 j = 0; j < ns; j++) {
                        pk[j] = (u32)pf[1 + j];
                        ps[j] = (u32)pf[1 + ns + j];
                        pi[j] = (u32)pf[1 + 2*ns + j];
                        po[j] = (u32)pf[1 + 3*ns + j];
                    }
                    pbd = (u32)pf[1 + 4*ns];
                    free(pf);

                    // Materialize gradient and scatter-add back
                    Term g_r = thvm_reduce(ctx, grad);
                    u32 g_id = (u32)term_val(g_r);
                    u32 g_numel = ctx->tensors[g_id].view.numel;
                    u32 in_numel = ma->view.numel;
                    f32 *g_data = malloc(g_numel * sizeof(f32));
                    f32 *dx = calloc(in_numel, sizeof(f32));
                    ctx->backend->buf_read(ctx->tensors[g_id].buf_id, g_data, g_numel * sizeof(f32));

                    u32 ndim_out = ma->view.shape.rank + ns;
                    u32 od[MAX_DIM];
                    for (u32 j = 0; j < pbd; j++) od[j] = ma->view.shape.dims[j];
                    for (u32 j = 0; j < ns; j++) {
                        od[pbd + j]      = po[j];
                        od[pbd + ns + j] = pk[j];
                    }

                    for (u32 flat = 0; flat < g_numel; flat++) {
                        u32 coords[MAX_DIM], rem = flat;
                        for (int d = (int)ndim_out - 1; d >= 0; d--) {
                            coords[d] = rem % od[d];
                            rem /= od[d];
                        }
                        // Map back to source index
                        u32 src_idx = 0, src_stride = 1;
                        u32 ndim_in = ma->view.shape.rank;
                        for (int d = (int)ndim_in - 1; d >= 0; d--) {
                            u32 coord;
                            if ((u32)d < pbd) {
                                coord = coords[d];
                            } else {
                                u32 si = (u32)d - pbd;
                                coord = coords[pbd + si] * ps[si] + coords[pbd + ns + si];
                            }
                            src_idx += coord * src_stride;
                            src_stride *= ma->view.shape.dims[d];
                        }
                        dx[src_idx] += g_data[flat];
                    }

                    u32 dx_id = tensor_fill(ctx, ma->view.shape, 0.0f);
                    ctx->backend->buf_write(ctx->tensors[dx_id].buf_id, dx, in_numel * sizeof(f32));
                    free(g_data); free(dx);
                    grad_graph_accum(ctx, gm, a_id, term_ten(dx_id, ma->dtype));
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

// thvm_backward removed — use thvm_grad(ctx, loss, param) per parameter instead.
// Each gradient is a lazy IC term that reduces through the standard engine.






// ============================================================
// Profiling — dispatch to backend
// ============================================================

void thvm_profile_report(TinyHVM *ctx) {
    if (ctx->backend && ctx->backend->profile_report)
        ctx->backend->profile_report();
}

void thvm_profile_reset(TinyHVM *ctx) {
    if (ctx->backend && ctx->backend->profile_reset)
        ctx->backend->profile_reset();
}


// ============================================================
// Eval helpers — argmax + accuracy
// ============================================================

Term thvm_argmax(TinyHVM *ctx, Term x, u32 rows, u32 cols) {
    // Read logits to host, compute argmax per row
    x = thvm_reduce(ctx, x);
    f32 *data = thvm_to_host(ctx, x);

    u32 *preds = malloc(rows * sizeof(u32));
    for (u32 i = 0; i < rows; i++) {
        u32 best = 0;
        f32 mv = data[i * cols];
        for (u32 j = 1; j < cols; j++) {
            if (data[i * cols + j] > mv) {
                mv = data[i * cols + j];
                best = j;
            }
        }
        preds[i] = best;
    }

    // Store as u32 tensor
    u32 id = ctx->tensor_count++;
    u32 buf = ctx->backend->buf_alloc(rows * sizeof(u32));
    ctx->tensors[id] = (TensorMeta){
        .buf_id = buf, .dtype = DTYPE_U32,
        .view = view_create(SHAPE(rows)),
    };
    ctx->backend->buf_write(buf, preds, rows * sizeof(u32));
    free(preds);
    return term_ten(id, DTYPE_U32);
}

f32 thvm_eval_accuracy(TinyHVM *ctx, Term logits, const u8 *labels,
                       u32 n_samples, u32 n_classes) {
    logits = thvm_reduce(ctx, logits);
    f32 *data = thvm_to_host(ctx, logits);

    u32 correct = 0;
    for (u32 i = 0; i < n_samples; i++) {
        u32 best = 0;
        f32 mv = data[i * n_classes];
        for (u32 j = 1; j < n_classes; j++) {
            if (data[i * n_classes + j] > mv) {
                mv = data[i * n_classes + j];
                best = j;
            }
        }
        if (best == labels[i]) correct++;
    }
    return 100.0f * (f32)correct / (f32)n_samples;
}
