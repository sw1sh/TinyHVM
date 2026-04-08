// tinyhvm.c — hub file (HVM4 style: path = function name, one file per fn)
// This file is a single-translation-unit build aggregator.
// Each section is extracted to its canonical sub-file.

#include "tinyhvm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdatomic.h>

// Thread-local ID (0 = main thread). Set by worker init, read by heap_alloc/tensor_create.
static _Thread_local u32 tl_thread_id = 0;

// ── term/ — term packing/unpacking ───────────────────────────────────────────
#include "term/new.c"
#include "term/tag.c"
#include "term/ext.c"
#include "term/val.c"
#include "term/sub.c"
#include "term/con.c"   // term_era / term_var / term_ten / term_top
                        // TAG_NUM removed: use thvm_scalar() for numeric consts

// ── heap/ — bump allocator ───────────────────────────────────────────────────
#include "heap/alloc.c"
#include "heap/read.c"
#include "heap/set.c"

// ── tensor/view/ — zero-copy view arithmetic ─────────────────────────────────
#include "tensor/view/create.c"
#include "tensor/view/permute.c"
#include "tensor/view/reshape.c"
#include "tensor/view/expand.c"
#include "tensor/view/shrink.c"
#include "tensor/view/pad.c"
#include "tensor/view/stride.c"
#include "tensor/view/broadcast.c"
#include "tensor/view/shapetracker.c"

// ── tensor/ — tensor metadata and scalar helper ───────────────────────────────
// tensor/create.c and tensor/scalar.c are included AFTER tensor_fill (below)
// since thvm_scalar calls thvm_tensor which is defined later.
#include "tensor/create.c"

// ── TAG_NUM compatibility stubs (DEPRECATED — migrate to thvm_scalar) ─────────
// These will be removed once UOP_ITE / UOP_LOAD_BATCH are migrated to scalar
// tensor predicates and the side-channel registry replaces the ptr-in-NUM hack.
static inline Term term_num_u32(u32 n) { return term_new(TAG_NUM, NUM_U32, (u64)n); }
static inline Term term_num_f32(f32 f) {
  u32 b; memcpy(&b, &f, 4);
  return term_new(TAG_NUM, NUM_F32, (u64)b);
}
static inline f32 term_as_f32(Term t) {
  u32 b = (u32)term_val(t); f32 f; memcpy(&f, &b, 4); return f;
}
static inline u32 term_as_u32(Term t) { return (u32)term_val(t); }

// Metal backend extern (only needed by ctx/init.c for registration)
#ifdef __APPLE__
extern Backend metal_backend;
#endif

// ── Forward declarations (needed by interact handlers) ───────────────────────
static u32  tensor_fill(TinyHVM *ctx, Shape s, f32 val);
static u32  tensor_transpose_2d(TinyHVM *ctx, u32 src_id);
static Term sum_to_shape(TinyHVM *ctx, Term grad, Shape src_shape, Shape target);

// Forward declarations for cross-module dependencies
static u32 reduce_id(TinyHVM *ctx, Term t);
static int is_elementwise(u32 uop);
static int is_binary(u32 uop);
static void tensor_materialize(TinyHVM *ctx, u32 tid);
static void tensor_materialize_chain(TinyHVM *ctx, u32 tid);
static int  tensor_materialize_reduce(TinyHVM *ctx, u32 input_tid, u32 out_buf, const ReduceSpec *rs);
static int  materialize_walk(TinyHVM *ctx, u32 tid,
                              FusedOp *ops, u32 *n_ops, u32 *op_tids,
                              u32 *leaf_ids, const View **leaf_views, u32 *n_leaves);
Term thvm_op2(TinyHVM *ctx, u32 opr, Term x, Term y);
Term thvm_op_raw(TinyHVM *ctx, u32 uop, Term a, Term b);
Term thvm_sup(TinyHVM *ctx, u32 label, Term a, Term b);
Term thvm_app(TinyHVM *ctx, Term fun, Term arg);
u32  thvm_fresh_label(TinyHVM *ctx);
Term thvm_bri(TinyHVM *ctx, Term *var_out, Term body);
Term thvm_ann(TinyHVM *ctx, Term term, Term type);
Term thvm_dsu(TinyHVM *ctx, Term label_expr, Term a, Term b);
Term thvm_ddu(TinyHVM *ctx, Term label_expr, Term val, Term bod);
Term thvm_inc(TinyHVM *ctx, Term term);
Term thvm_mm(TinyHVM *ctx, Term a, Term b);
static u32 _ensure_count = 0, _ensure_alloc_count = 0;
#define ENSURE(c,t) do{if((t)&&c->tensors[t].buf_id==0&&c->tensors[t].creator_op){_ensure_count++;tensor_materialize(c,t);if(c->tensors[t].buf_id)_ensure_alloc_count++;}}while(0)

// Read strided view to contiguous host buffer
static f32 *thvm_to_host_view(Backend *be, u32 buf_id, const View *v, u32 numel) {
    u32 max_idx = 0;
    for (u32 d = 0; d < v->shape.rank; d++)
        if (v->strides[d] > 0) max_idx += (v->shape.dims[d]-1) * (u32)v->strides[d];
    max_idx += (v->offset > 0) ? (u32)v->offset : 0;
    f32 *raw = malloc((max_idx+1)*sizeof(f32));
    be->buf_read(buf_id, raw, (u64)(max_idx+1)*sizeof(f32));
    f32 *out = malloc(numel * sizeof(f32));
    for (u32 flat = 0; flat < numel; flat++) {
        u32 rem = flat; i32 idx = v->offset; int msk = 0;
        for (int d = (int)v->shape.rank-1; d >= 0; d--) {
            u32 c = rem % v->shape.dims[d]; rem /= v->shape.dims[d];
            if (v->has_mask && (c < v->mask_begin[d] || c >= v->mask_end[d])) msk = 1;
            idx += (i32)c * (v->strides[d] > 0 ? v->strides[d] : 0);
        }
        out[flat] = (msk || idx < 0 || (u32)idx > max_idx) ? 0.f : raw[(u32)idx];
    }
    free(raw);
    return out;
}

// ── clone/ — deep-copy (ALO) for REF unfolding ──────────────────────────────
#include "clone/_.c"

// ── interact/ — one interaction rule per active pair ─────────────────────────
#include "interact/_.c"

// ── reduce/ — enter/apply trampoline ─────────────────────────────────────────
#include "reduce/_.c"

// ── ctx/ — init, free, device, reset, tensor, profiling ──────────────────────
#include "ctx/init.c"
#include "tensor/scalar.c"

// ── ops/ — thvm_realize, supporting kernel functions ─────────────────────────
#include "ops/_.c"

// ── fuse/ — elementwise fusion (fuse_walk_inner, fuse_or_reduce) ─────────────
#include "fuse/_.c"
#include "fuse/materialize.c"

// ── grad/ — thvm_grad, thvm_grad_multi (pure IC term constructors) ────────────
#include "grad/_.c"

// ── parallel/ — work-stealing deque + per-thread state ──────────────────────
#include "parallel/wsdeque.c"
#include "parallel/workers.c"

// ── inet/ — interaction combinator term constructors ─────────────────────────
#include "inet/_.c"

// ── schedule/ — lazy graph compiler ──────────────────────────────────────────
#include "schedule/_.c"

// ── debug/ — graph dump (DOT/JSON) ──────────────────────────────────────────
#include "debug/dump.c"
#include "debug/graph.c"

