// tinyhvm.c — hub file (HVM4 style: path = function name, one file per fn)
// This file is a single-translation-unit build aggregator.
// Each section is extracted to its canonical sub-file.

#include "tinyhvm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

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

// ── Metal GPU forward declarations (defined in metal.m) ───────────────────────
#ifdef __APPLE__
extern void metal_mul_reduce_sum(u32 dst, u32 dst_numel,
                                 u32 a_buf, const View *av,
                                 u32 b_buf, const View *bv,
                                 const View *ov,
                                 u32 n_reduce,
                                 const u32 *reduce_dims,
                                 const u32 *reduce_strides_a,
                                 const u32 *reduce_strides_b);
typedef struct { u32 uop; u32 arg_a; u32 arg_b; } FusedOp;
extern void metal_dispatch_fused_v2(u32 out_buf, u32 out_numel,
                                     u32 *leaf_bufs, const View **leaf_views, u32 n_leaves,
                                     FusedOp *ops, u32 n_ops,
                                     int has_reduce, u32 reduce_dim);
extern void metal_contiguify(u32 dst_buf, u32 numel, u32 src_buf, const View *src_view);
extern void metal_buf_read_nosync(u32 id, void *out, u64 bytes);
extern Backend metal_backend;
#endif

// ── Forward declarations (needed by interact handlers) ───────────────────────
static u32  tensor_fill(TinyHVM *ctx, Shape s, f32 val);
static u32  tensor_transpose_2d(TinyHVM *ctx, u32 src_id);
static Term sum_to_shape(TinyHVM *ctx, Term grad, Shape src_shape, Shape target);

// ── clone/ — deep-copy (ALO) for REF unfolding ──────────────────────────────
#include "clone/_.c"

// ── interact/ — one interaction rule per active pair ─────────────────────────
#include "interact/_.c"

// ── reduce/ — enter/apply trampoline ─────────────────────────────────────────
#include "reduce/_.c"

// ── ctx/ — init, free, device, reset, tensor, profiling ──────────────────────
#include "ctx/init.c"

// ── ops/ — thvm_realize, supporting kernel functions ─────────────────────────
#include "ops/_.c"

// ── grad/ — thvm_grad, thvm_backward, lam/app/ref/ite/load/assign ────────────
#include "grad/_.c"

