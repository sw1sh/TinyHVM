// tinyhvm.h — TinyHVM: interaction-net runtime for tensor computation
// Minimal. No dependencies. Inspired by HVM4's bit layout.

#ifndef TINYHVM_H
#define TINYHVM_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

// ============================================================
// Types
// ============================================================

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t  i32;
typedef float    f32;

typedef u64 Term;

// ============================================================
// Term Bit Layout (64 bits)
// ============================================================
//
//  63     62..56    55..38        37..0
//  [SUB]  [TAG:7]   [EXT:18]     [VAL:38]
//
// SUB  = substitution flag (1 bit)
// TAG  = term type (7 bits, 128 possible)
// EXT  = extension/label (18 bits)
// VAL  = value/pointer (38 bits, 256GB address space)

#define SUB_BITS  1
#define TAG_BITS  7
#define EXT_BITS  18
#define VAL_BITS  38

#define SUB_SHIFT 63
#define TAG_SHIFT 56
#define EXT_SHIFT 38
#define VAL_SHIFT 0

#define SUB_MASK  0x1ULL
#define TAG_MASK  0x7FULL
#define EXT_MASK  0x3FFFFULL
#define VAL_MASK  0x3FFFFFFFFFULL

// ============================================================
// Tags — Minimal set for tensor computation
// ============================================================
//
// Core interaction calculus (from HVM4):
#define TAG_APP  0   // Application: (f x)
#define TAG_LAM  1   // Lambda: λx.body
#define TAG_VAR  2   // Variable: bound reference
#define TAG_SUP  3   // Superposition: {a, b}
#define TAG_DP0  4   // Dup projection 0
#define TAG_DP1  5   // Dup projection 1
#define TAG_ERA  6   // Eraser

// Data:
#define TAG_NUM  7   // 32-bit number (u32/f32 via EXT flag)
#define TAG_REF  8   // Named definition reference
#define TAG_OP2  9   // Binary operation: Op2(opr, x, y)

// Tensor (new):
#define TAG_TEN  10  // Tensor handle: VAL = tensor_id, EXT = dtype
#define TAG_TOP  11  // Tensor op node (lazy): EXT = uop code
#define TAG_CTR  12  // Constructor (for multi-arg nodes)

// ICC (Interaction Calculus of Constructions):
#define TAG_BRI  13  // Bridge: θx.body (dual of lambda, contra-variant binding)
#define TAG_ANN  14  // Annotation: {term : type} (transparent — strips during reduce)

// Dynamic labels + priority:
#define TAG_DSU  15  // Dynamic SUP: &(label_expr){a, b}. Heap: [label_expr, a, b]
#define TAG_DDU  16  // Dynamic DUP: !&(label_expr){val, bod}. Heap: [label_expr, val, bod]
#define TAG_INC  17  // Priority wrapper: transparent to reduce, lower priority in collapse

// Type enumeration machinery (ported from HVM4):
#define TAG_EQL  18  // Structural equality: (a === b). Heap: [a, b]. Strict both sides.
#define TAG_AND  19  // Short-circuit AND: (a .&. b). Heap: [a, b]. 0→0, n→b.
#define TAG_OR   20  // Short-circuit OR:  (a .|. b). Heap: [a, b]. 0→b, n→1.
#define TAG_MAT  21  // Pattern match: λ{#K:h; m}. Heap: [handler, fallback]. EXT=match_tag. WNF atom.
#define TAG_ANY  22  // Wildcard: matches anything. No heap (atom). EQL(ANY,x)→1, DUP→ANY.
// Unordered superpositions (from Taelin's spec, not yet in HVM4):
#define TAG_USP  23  // Unordered SUP: %L{a, b}. Heap: [a, b], EXT = label. {a,b}=={b,a}.
#define TAG_UDP  24  // Unordered DUP: !%L{x}=v. Heap: [val], EXT = label. Single output port.

#define TAG_COUNT 25

// ============================================================
// UOps — Minimal tensor operations (tinygrad-inspired)
// ============================================================
// Stored in EXT field of TAG_TOP nodes.

// Movement (no compute, metadata only):
#define UOP_LOAD    0   // Load tensor from host
#define UOP_STORE   1   // Store tensor to host
#define UOP_COPY    2   // Physical copy (resolve strides)

// Elementwise (unary):
#define UOP_NEG     3
#define UOP_EXP     4
#define UOP_LOG     5
#define UOP_RELU    6
#define UOP_CAST    7   // dtype cast
#define UOP_SQRT    8   // square root

// Elementwise (binary):
#define UOP_ADD     9
#define UOP_MUL    10
#define UOP_DIV    11
#define UOP_MAX    12
#define UOP_CMP    13   // compare (returns 0/1)
#define UOP_SUB    14   // subtract (for gradient descent)

// Reduce:
#define UOP_SUM    15   // reduce sum over axis
#define UOP_RMAX   16   // reduce max over axis

// Matmul (special — dispatches to BLAS/MPS):
#define UOP_MM     17   // matrix multiply

// Movement (metadata-only, stride manipulation):
#define UOP_RESHAPE   18
#define UOP_PERMUTE   19
#define UOP_EXPAND    20  // = broadcast
#define UOP_SHRINK    21  // = slice
#define UOP_PAD       22

// Fusion: wraps any fused subgraph. Realized tensor carries the original subnet
// heap root in src_ids, so GRAD can walk back through the unfused lazy graph.
#define UOP_FUSING    23  // fused kernel (SUM(MUL) etc); see ic_fusion.md
#define UOP_ASSIGN    24  // in-place weight update: blit src buf into dst buf
#define UOP_WHERE     25  // elementwise select: WHERE(cond_ten, then_ten, else_ten)
#define UOP_IFZ       26  // if-zero branch: IFZ(counter_ten, zero_case, succ_lam)
                          // counter==0 → zero_case; counter>0 → APP(succ_lam, counter-1)
#define UOP_LOG_PRINT 27  // print scalar tensor value, return tensor unchanged

#define UOP_GRAD      28   // IC gradient: DUP-op interaction in the reducer
#define UOP_TODEVICE  29   // lazy device transfer: TODEVICE(tensor, device_idx_scalar)
#define UOP_COUNT     30

// (LAYER_OP_POOL_GATHER and LAYER_OP_BATCHNORM removed — both are now
// composed from standard UOps with standard backward rules.)

// UOp name table (device-agnostic)
static const char *uop_names[] = {
    "LOAD","STORE","COPY","NEG","EXP","LOG","RELU","CAST","SQRT",
    "ADD","MUL","DIV","MAX","CMP","SUB","SUM","RMAX","MM",
    "RESHAPE","PERMUTE","EXPAND","SHRINK","PAD","FUSING","ASSIGN","WHERE",
    "IFZ","LOG_PRINT","GRAD","TODEVICE"
};

// ============================================================
// Device-agnostic profiling (enabled by THVM_PROFILE env)
// ============================================================

#include <time.h>

// Extended UOp range to cover POOL_GATHER
#define PROF_UOP_MAX 128

// Phase IDs for step-level tracking
#define PHASE_FORWARD   0
#define PHASE_BACKWARD  1
#define PHASE_ADAM      2
#define PHASE_RESET     3
#define PHASE_OTHER     4
#define PHASE_COUNT     5

static const char *phase_names[] = {"forward", "backward", "adam", "reset", "other"};

typedef struct {
    // Per-UOp dispatch stats (compute ops)
    u64 uop_ns[PROF_UOP_MAX];       // total nanoseconds per UOp
    u32 uop_cnt[PROF_UOP_MAX];      // dispatch count per UOp

    // Per-UOp tensor creation stats
    u32 uop_tensors[PROF_UOP_MAX];  // tensors created per UOp type

    // Memory tracking
    u64 buf_bytes_alloc;             // total bytes allocated this step
    u32 buf_alloc_cnt;               // number of buf_alloc calls
    u64 buf_bytes_peak;              // peak cumulative bytes (lifetime)
    u64 buf_bytes_current;           // current live bytes

    // Tensor watermarks
    u32 tensor_peak;                 // peak tensor_count reached
    u32 tensor_created;              // total tensors created this step
    u32 tensor_freed;                // tensors freed by reset

    // Heap usage
    u64 heap_peak;                   // peak heap_pos reached
    u64 heap_at_reset;               // heap_pos before reset

    // Phase timing
    u64 phase_ns[PHASE_COUNT];
    u64 phase_start;
    u32 current_phase;

    // CPU-side data transfer
    u64 cpu_read_bytes;              // bytes read from GPU to CPU (buf_read)
    u32 cpu_read_cnt;
    u64 cpu_write_bytes;             // bytes written from CPU to GPU (buf_write)
    u32 cpu_write_cnt;

    int enabled;
} ThvmProfile;

// Global profile state
static ThvmProfile thvm_prof_global;

static inline void thvm_prof_init(void) {
    thvm_prof_global.enabled = getenv("THVM_PROFILE") != NULL;
    memset(&thvm_prof_global, 0, sizeof(thvm_prof_global));
    thvm_prof_global.enabled = getenv("THVM_PROFILE") != NULL;
}

// Reset per-step counters (keep cumulative ones like buf_bytes_current)
static inline void thvm_prof_step_reset(void) {
    if (!thvm_prof_global.enabled) return;
    memset(thvm_prof_global.uop_ns, 0, sizeof(thvm_prof_global.uop_ns));
    memset(thvm_prof_global.uop_cnt, 0, sizeof(thvm_prof_global.uop_cnt));
    memset(thvm_prof_global.uop_tensors, 0, sizeof(thvm_prof_global.uop_tensors));
    thvm_prof_global.buf_bytes_alloc = 0;
    thvm_prof_global.buf_alloc_cnt = 0;
    thvm_prof_global.tensor_created = 0;
    thvm_prof_global.tensor_freed = 0;
    thvm_prof_global.tensor_peak = 0;
    thvm_prof_global.heap_peak = 0;
    thvm_prof_global.heap_at_reset = 0;
    memset(thvm_prof_global.phase_ns, 0, sizeof(thvm_prof_global.phase_ns));
    thvm_prof_global.cpu_read_bytes = 0;
    thvm_prof_global.cpu_read_cnt = 0;
    thvm_prof_global.cpu_write_bytes = 0;
    thvm_prof_global.cpu_write_cnt = 0;
}

// Portable nanosecond timer
static inline u64 thvm_prof_tick(void) {
    if (!thvm_prof_global.enabled) return 0;
#ifdef __APPLE__
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ULL + (u64)ts.tv_nsec;
#endif
}

static inline void thvm_prof_record(u32 uop, u64 t0) {
    if (!thvm_prof_global.enabled || t0 == 0) return;
    u64 ns = thvm_prof_tick() - t0;
    if (uop < PROF_UOP_MAX) {
        thvm_prof_global.uop_ns[uop] += ns;
        thvm_prof_global.uop_cnt[uop]++;
    }
}

static inline void thvm_prof_tensor_created(u32 uop_hint) {
    if (!thvm_prof_global.enabled) return;
    thvm_prof_global.tensor_created++;
    if (uop_hint < PROF_UOP_MAX)
        thvm_prof_global.uop_tensors[uop_hint]++;
}

static inline void thvm_prof_buf_alloc(u64 bytes) {
    if (!thvm_prof_global.enabled) return;
    thvm_prof_global.buf_bytes_alloc += bytes;
    thvm_prof_global.buf_alloc_cnt++;
    thvm_prof_global.buf_bytes_current += bytes;
    if (thvm_prof_global.buf_bytes_current > thvm_prof_global.buf_bytes_peak)
        thvm_prof_global.buf_bytes_peak = thvm_prof_global.buf_bytes_current;
}

static inline void thvm_prof_buf_free(u64 bytes) {
    if (!thvm_prof_global.enabled) return;
    if (thvm_prof_global.buf_bytes_current >= bytes)
        thvm_prof_global.buf_bytes_current -= bytes;
}

static inline void thvm_prof_buf_read(u64 bytes) {
    if (!thvm_prof_global.enabled) return;
    thvm_prof_global.cpu_read_bytes += bytes;
    thvm_prof_global.cpu_read_cnt++;
}

static inline void thvm_prof_buf_write(u64 bytes) {
    if (!thvm_prof_global.enabled) return;
    thvm_prof_global.cpu_write_bytes += bytes;
    thvm_prof_global.cpu_write_cnt++;
}

static inline void thvm_prof_phase(u32 phase) {
    if (!thvm_prof_global.enabled) return;
    u64 now = thvm_prof_tick();
    if (thvm_prof_global.phase_start) {
        u32 prev = thvm_prof_global.current_phase;
        thvm_prof_global.phase_ns[prev] += now - thvm_prof_global.phase_start;
    }
    thvm_prof_global.current_phase = phase;
    thvm_prof_global.phase_start = now;
}

static inline void thvm_prof_phase_end(void) {
    if (!thvm_prof_global.enabled || !thvm_prof_global.phase_start) return;
    u64 now = thvm_prof_tick();
    u32 prev = thvm_prof_global.current_phase;
    thvm_prof_global.phase_ns[prev] += now - thvm_prof_global.phase_start;
    thvm_prof_global.phase_start = 0;
}

static inline void thvm_prof_update_watermarks(u32 tensor_count, u64 heap_pos) {
    if (!thvm_prof_global.enabled) return;
    if (tensor_count > thvm_prof_global.tensor_peak)
        thvm_prof_global.tensor_peak = tensor_count;
    if (heap_pos > thvm_prof_global.heap_peak)
        thvm_prof_global.heap_peak = heap_pos;
}

// Conv2d parameter structs (used by layers.c internal kernels)
typedef struct { u32 B, Cin, H, W, KH, KW, OH, OW, patch_size, n_patches; } Conv2dParams;
typedef struct { u32 B, C, H, W; } LayoutParams;
typedef struct { u32 B, C, H, W, OH, OW; } Pool2dParams;

// ============================================================
// NUM encoding
// ============================================================
// EXT[0] = 0 → VAL is u32, EXT[0] = 1 → VAL is f32 (bitcast)

#define NUM_U32  0
#define NUM_F32  1

// ============================================================
// Heap
// ============================================================

#ifndef HEAP_CAP
#define HEAP_CAP (1ULL << 25)  // 32M terms = 256MB
#endif

// ============================================================
// Tensor metadata
// ============================================================

#define DTYPE_F32  0
#define DTYPE_F16  1
#define DTYPE_I32  2
#define DTYPE_U32  3
#define DTYPE_COUNT 4

static inline u32 dtype_size(u32 dtype) {
    static const u32 sizes[] = {4, 2, 4, 4};
    return (dtype < DTYPE_COUNT) ? sizes[dtype] : 4;
}

#define MAX_DIM 8
#define MAX_TENSORS 16384

// Shape: dims + rank bundled together
typedef struct {
    u32 dims[MAX_DIM];
    u32 rank;
} Shape;

// Convenience: SHAPE(2,3) → (Shape){.dims={2,3}, .rank=2}
#define SHAPE(...) ((Shape){.dims={__VA_ARGS__}, \
    .rank=sizeof((u32[]){__VA_ARGS__})/sizeof(u32)})

static inline Shape shape_of(const u32 *dims, u32 rank) {
    Shape s = {.rank = rank};
    for (u32 i = 0; i < rank; i++) s.dims[i] = dims[i];
    return s;
}

// View: shape + strides + offset (tinygrad-inspired)
// Movement ops modify this without touching GPU buffers.
typedef struct {
    Shape shape;
    i32 strides[MAX_DIM];   // can be 0 (broadcast) or negative (flip)
    i32 offset;             // starting element in buffer
    u32 numel;              // product of shape (logical element count)
    u8  contiguous;         // 1 if standard row-major, no offset
    u8  has_mask;           // 1 if view has a per-dim valid region mask
    u32 mask_begin[MAX_DIM]; // per-dim start of valid region
    u32 mask_end[MAX_DIM];   // per-dim end of valid region
    // Compound masks: validity checks on linear combinations of coordinates.
    // Used by pool views where validity is: begin <= c_a*s + c_b < end.
    // n_compound_masks=0 means no compound masks (backward compat).
    u8  n_compound_masks;
    struct { u8 dim_a, dim_b; i32 stride_a; u32 begin, end; } compound_masks[2]; // max 2 spatial dims
    // Modular indexing (reserved for future use)
    u32 mod_size[MAX_DIM];
} View;

// ShapeTracker: stack of views for composed transformations.
// Matches tinygrad's ShapeTracker(views: tuple[View, ...]).
// When a reshape can't be expressed as a single view, views are stacked.
// At codegen time, views_to_idx() composes them right-to-left via unravel.
#define ST_MAX_VIEWS 4
typedef struct {
    View views[ST_MAX_VIEWS];
    u8   n_views;  // 0 = unused (use tensor's view directly), 1+ = stacked
} ShapeTracker;

// ShapeTracker from a single View (no forward decl needed).
static inline ShapeTracker st_from_view(View v) {
    ShapeTracker st = { .n_views = 1 };
    st.views[0] = v;
    return st;
}

// Last view = output shape/strides.
static inline const View *st_last_view(const ShapeTracker *st) {
    return (st && st->n_views > 0) ? &st->views[st->n_views - 1] : NULL;
}

// Composition helpers (st_reshape, st_permute, st_expand) defined in
// tensor/view/shapetracker.c — included after view_create/view_reshape.

// ============================================================
// Shape tracking for lazy TAG_TOP nodes (ShapeTracker)
// ============================================================
// Every TAG_TOP gets its output view stored eagerly at creation time.
// Keyed by heap loc. The fuser reads these to get pre-computed views
// for lazy nodes without reducing them.
// Hash table with collision detection via stored key (heap_loc).
#define ST_TABLE_SIZE (1 << 18)  // 256K — much larger than heap usage per step
static ShapeTracker st_trackers[ST_TABLE_SIZE];
static u64          st_keys[ST_TABLE_SIZE]; // 0 = empty

// Store a ShapeTracker for a heap location.
static inline void st_set_tracker(u64 heap_loc, const ShapeTracker *st) {
    u64 key = heap_loc + 1;
    u32 idx = (u32)(heap_loc % ST_TABLE_SIZE);
    for (u32 p = 0; p < 16; p++) {
        u32 i = (idx + p) % ST_TABLE_SIZE;
        if (st_keys[i] == 0 || st_keys[i] == key) {
            st_trackers[i] = *st;
            st_keys[i] = key;
            return;
        }
    }
    st_trackers[idx] = *st;
    st_keys[idx] = key;
}

// Store a single View as a 1-view ShapeTracker (convenience).
static inline void st_set(u64 heap_loc, const View *v) {
    ShapeTracker st = { .n_views = 1 };
    st.views[0] = *v;
    st_set_tracker(heap_loc, &st);
}

// Get the full ShapeTracker for a heap location.
static inline const ShapeTracker *st_get_tracker(u64 heap_loc) {
    u64 key = heap_loc + 1;
    u32 idx = (u32)(heap_loc % ST_TABLE_SIZE);
    for (u32 p = 0; p < 16; p++) {
        u32 i = (idx + p) % ST_TABLE_SIZE;
        if (st_keys[i] == key) return &st_trackers[i];
        if (st_keys[i] == 0) return NULL;
    }
    return NULL;
}

// Get the output View (last view in the ShapeTracker) — backward compat.
static inline const View *st_get(u64 heap_loc) {
    const ShapeTracker *st = st_get_tracker(heap_loc);
    if (!st || st->n_views == 0) return NULL;
    return &st->views[st->n_views - 1];
}

// ============================================================
// Fused Op / ReduceSpec (used by fuser + codegen + scheduler)
// ============================================================
#define FUSE_MAX_OPS    64
#define FUSE_MAX_LEAVES 64

typedef struct { u32 uop; u32 arg_a; u32 arg_b; } FusedOp;
typedef struct {
    u8 is_reduce[MAX_DIM];
    u32 reduce_type;
    u32 post_reduce_start; // index into ops[] where post-reduce ops begin (0 = no post-reduce)
    u32 n_post_leaves;     // number of post-reduce-only leaves (appended after pre-reduce leaves)
    // Multi-reduce: second reduce phase (0 = single reduce)
    // Layout: ops[0..post_reduce_start-1] = pre-reduce1
    //         ops[post_reduce_start..reduce2_start-1] = between-reduce ew
    //         ops[reduce2_start..n_ops-1] = pre-reduce2 + post-reduce2
    u32 reduce2_type;      // 0 = single reduce, else UOP_SUM/UOP_RMAX
    u32 reduce2_start;     // first op index for reduce2 phase
    u8 is_reduce2[MAX_DIM]; // reduce axes for phase 2 (may differ from phase 1)
} ReduceSpec;

// KernelEntry: scheduler's output — one fused kernel ready for dispatch.
// Produced by fuse_build_kernel (pure walk), consumed by UOP_FUSING handler.
#define SCHED_MAX_KERNELS 512
typedef struct {
    FusedOp    ops[FUSE_MAX_OPS]; u32 n_ops;
    u32        leaf_ids[FUSE_MAX_LEAVES];
    View       leaf_views[FUSE_MAX_LEAVES]; u32 n_leaves;
    ShapeTracker leaf_sts[FUSE_MAX_LEAVES]; // multi-view STs for cross-rank indexing
    Shape      full_shape;       // ew iteration domain
    Shape      out_shape;        // output shape (after reduce)
    ReduceSpec reduce;
    u32        has_reduce;       // 0 or UOP_SUM/UOP_RMAX
    Term       reshape_term;     // post-reduce RESHAPE (or TAG_ERA)
    Term       sum_term;         // the SUM/RMAX TAG_TOP (for axes)
    Term       original_term;    // original TAG_TOP before scheduling (for multi-consumer propagation)
    Term       leaf_terms[FUSE_MAX_LEAVES]; // original leaf terms (for resolving UOP_FUSING deps)
    int        fail_code;        // diagnostic: last failure reason from fuse_build_kernel
} KernelEntry;

typedef struct Backend Backend;

// Per-thread state for parallel reduction (work_stealing.md)
typedef struct ThvmThread {
    u32 tid;
    u64 bank_start, bank_next, bank_end;  // heap bank
    u32 tensor_start, tensor_next, tensor_end;  // tensor ID range
    u64 itrs;
} ThvmThread;

typedef struct {
    u32         buf_id;     // GPU buffer handle
    u32         dtype;
    u32         refcount;
    View        view;       // logical view (shape for broadcasting etc.)
    ShapeTracker st;        // view stack for codegen (0 views = use view directly)
    void       *host_ptr;   // cached host copy
    Backend    *backend;    // which backend owns this tensor's buffer

    // Autograd provenance
    u8          requires_grad;
    u32         creator_op; // UOP that created this tensor
    u32         src_ids[2]; // input tensor ids (for backward chain rule)
    u64         creator_loc; // heap location of the TAG_TOP that created this tensor

    // Deferred dispatch: how many deferred ops consume this tensor as input.
    u8          defer_consumers;

    u32         assign_target; // ASSIGN elision: write fused output here instead of new buf

    // Fusion metadata (only when creator_op == UOP_FUSING)
    u64         fusing_loc; // heap loc of the original subnet root TAG_TOP
    u32         fusing_uop; // UOP of the subnet root (e.g. UOP_SUM)

    // Conv backward metadata: enables specialized MPS matmul backward
    // instead of generic 8-dim unfuse+contiguify. Set by thvm_conv2d.
    u32         conv_input_id;  // input tensor before pool/pad (for im2col)
    u32         conv_weight_id; // weight tensor
    u16         conv_groups;
    u8          conv_KH, conv_KW;
    u8          conv_stride[2];
    u8          conv_padding[4];

    // Pool view metadata: enables correct backward for strided pool views.
    // When pool_n_spatial > 0, this is a pool strided view [batch,C,OY,OX,KH,KW].
    // Backward must scatter-add to input [batch,C,H,W], not just reshape.
    u8          pool_n_spatial;  // 0 = not a pool view, 2 = 2D pool
    u8          pool_kernel[2];  // KH, KW
    u8          pool_stride[2];  // stride_h, stride_w

} TensorMeta;

// ============================================================
// Interaction Trace (for single-step reduction debugging)
// ============================================================

struct InteractionTrace {
    u64    before_loc;
    u32    before_tag;
    u32    before_ext;
    u64    after_loc;
    u32    after_tag;
    u32    after_ext;
    u32    rule_id;
};

// ============================================================
// Backend Interface
// ============================================================

struct Backend {
    // Lifecycle
    int   (*init)(void);
    void  (*shutdown)(void);

    // Buffer management
    u32   (*buf_alloc)(u64 bytes);
    void  (*buf_free)(u32 id);
    void  (*buf_incref)(u32 id);   // increment buffer-level refcount (shared views)
    void  (*buf_decref)(u32 id);   // decrement; enqueues free when refcount→0
    void  (*mem_checkpoint)(u32 *leaf_buf_ids, u32 n_leaves); // recycle consumed buffers
    void  (*buf_mark_use)(u32 id); // pre-scan: mark buffer as having a future dispatch use
    void  (*buf_write)(u32 id, const void *data, u64 bytes);
    void  (*buf_read)(u32 id, void *out, u64 bytes);

    // UOp dispatch — the only compute primitives a backend must implement
    void  (*op_unary)(u32 uop, u32 dst, const View *dv,
                      u32 src, const View *sv);
    void  (*op_binary)(u32 uop, u32 dst, const View *dv,
                       u32 a, const View *av, u32 b, const View *bv);
    void  (*op_mm)(u32 dst, u32 a, const View *av, u32 b, const View *bv,
                   u32 M, u32 K, u32 N);
    void  (*op_reduce)(u32 uop, u32 dst, u32 dst_numel,
                       u32 src, u32 src_numel, u32 reduce_dim);

    // Advanced dispatch (optional — NULL means unsupported, fall back to CPU)
    void  (*dispatch_kernel_rs)(u32 out_buf,
                                u32 *leaf_bufs, const View **leaf_views,
                                const ShapeTracker *const *leaf_sts, u32 n_leaves,
                                FusedOp *ops, u32 n_ops,
                                const Shape *full_shape, const ReduceSpec *reduce,
                                u32 *side_bufs, const u32 *side_op_indices, u32 n_side_outputs);
    void  (*contiguify)(u32 dst_buf, u32 numel, u32 src_buf, const View *src_view);
    void  (*buf_copy)(u32 dst_buf, u32 src_buf, u64 nbytes);
    void  (*buf_read_nosync)(u32 id, void *out, u64 bytes);
    // Optimized optimizer kernel (optional — NULL means use IC graph)
    void  (*adam_step)(u32 param, u32 grad, u32 m, u32 v,
                       f32 lr, f32 beta1, f32 beta2, f32 eps,
                       f32 bc1, f32 bc2, u32 n);

    // Pool management
    void  (*pool_reset)(u32 keep);
    void  (*pool_set_persistent)(u32 max_buf); // mark bufs ≤ max_buf as persistent (skip flush on read)
    // GPU batching
    void  (*begin_batch)(void);
    void  (*end_batch)(void);
    // Profiling
    void  (*profile_report)(void);
    void  (*profile_reset)(void);
};

// ============================================================
// Context
// ============================================================

#define THVM_MAX_BACKENDS 4
#define THVM_DEV_CPU   0
#define THVM_DEV_METAL 1

// Hash-consing for APP nodes (optional)
#define HC_TABLE_BITS 18
#define HC_TABLE_CAP  (1U << HC_TABLE_BITS)

typedef struct {
    Term fun;      // key part 1
    Term arg;      // key part 2
    Term result;   // TAG_APP term (== heap loc since TAG_APP=0)
} HCSlot;

typedef struct {
    Term       *heap;
    u64         heap_pos;
    TensorMeta *tensors;
    u32         tensor_count;

    Backend    *backends[THVM_MAX_BACKENDS]; // [0]=cpu, [1]=metal, ...
    u32         n_backends;
    u32         default_device;              // index into backends[]

    u64         itrs;       // interaction count
    u8          no_fuse;    // (legacy, unused — compute ops always WNF)
    u8          no_grad_alloc; // (legacy, unused)
    u8          no_dup;     // 1 to skip linear_use DUP (used inside GRAD handler)
    u8          defer_all;   // (legacy, unused)
    u8          dispatch_mode; // (legacy, unused)

    // Named definitions for TAG_REF (global def table)
    Term        defs[256];   // defs[name] = heap loc or TAG_TOP term
    u32         def_count;

    // Interaction tracing (Phase 4)
    struct InteractionTrace *trace_buf;
    u32         trace_count;
    u32         trace_cap;
    u8          trace_enabled;

    // Step-limited reduction
    u32         step_budget;    // 0 = unlimited
    u32         steps_taken;

    // Parallel reduction (work_stealing.md)
    // threads[0] = main thread. When n_threads <= 1, parallel infra is bypassed.
    struct ThvmThread threads[16];
    u32         n_threads;     // 0 or 1 = single-threaded

    // SUP/DUP labels — monotonic counter, only incremented at construction time
    u32         next_sup_label;

    // Hash-consing table (NULL = disabled, lazy-allocated by thvm_app_hc)
    HCSlot     *hc_table;

} TinyHVM;

// Per-tensor backend accessor
static inline Backend *tensor_backend(TinyHVM *ctx, u32 tid) {
    return ctx->tensors[tid].backend;
}

// Default backend for new tensor creation
static inline Backend *ctx_default_backend(TinyHVM *ctx) {
    return ctx->backends[ctx->default_device];
}

// ============================================================
// API
// ============================================================

// Term manipulation
static inline Term term_new(u32 tag, u32 ext, u64 val);
static inline u32  term_tag(Term t);
static inline u32  term_ext(Term t);
static inline u64  term_val(Term t);

// Heap
static inline u64  heap_alloc(TinyHVM *ctx, u64 words);
static inline Term heap_read(TinyHVM *ctx, u64 loc);
static inline void heap_set(TinyHVM *ctx, u64 loc, Term t);

// Context
TinyHVM *thvm_init(const char *default_device);  // "cpu", "metal" — inits all backends, sets default
void     thvm_free(TinyHVM *ctx);
void     thvm_reset(TinyHVM *ctx, u32 keep);  // free tensors above `keep`, reset heap

// Device
Term     thvm_to_device(TinyHVM *ctx, Term t, u32 device_idx);

// JIT capture/replay (Metal only — bypasses IC for repeated training steps)
#ifdef __APPLE__
void     jit_begin_capture(u32 n_persistent_bufs);
void     jit_end_capture(void);
void     jit_flush(void);
void     jit_replay(void);
#endif

// Reduction
Term     thvm_reduce(TinyHVM *ctx, Term t);
Term     thvm_reduce_steps(TinyHVM *ctx, Term t, u32 max_steps);

// Tensor API
Term     thvm_tensor(TinyHVM *ctx, const f32 *data, Shape s);
Term     thvm_op(TinyHVM *ctx, u32 uop, Term a, Term b);
void     thvm_realize(TinyHVM *ctx, Term t);
f32     *thvm_to_host(TinyHVM *ctx, Term t);

// Movement ops (lazy — modify View, share buffer)
Term     thvm_reshape(TinyHVM *ctx, Term t, Shape new_shape);
Term     thvm_expand(TinyHVM *ctx, Term t, Shape new_shape);
Term     thvm_permute(TinyHVM *ctx, Term t, const u32 *axes, u32 rank);

// Print
void     thvm_print_term(TinyHVM *ctx, Term t);

// Heap graph (for visualization)
void     thvm_heap_graph(TinyHVM *ctx, Term root,
                         i32 **out_nodes, u32 *out_n_nodes,
                         i32 **out_edges, u32 *out_n_edges);

// Autograd
void     thvm_set_requires_grad(TinyHVM *ctx, Term t);

// Lambda / inet combinators
Term     thvm_lam(TinyHVM *ctx, Term *var_out, Term body);   // allocate LAM node
Term     thvm_app(TinyHVM *ctx, Term fun, Term arg);         // allocate APP node
Term     thvm_app_hc(TinyHVM *ctx, Term fun, Term arg);    // hash-consed APP
void     thvm_hc_clear(TinyHVM *ctx);                      // clear HC table
u32      thvm_define(TinyHVM *ctx, Term body);               // register def, return name id
Term     thvm_ref(TinyHVM *ctx, u32 name);                   // TAG_REF(name)
Term     thvm_sup(TinyHVM *ctx, u32 label, Term a, Term b);   // TAG_SUP with label
void     thvm_dup(TinyHVM *ctx, u32 label, Term z, Term *out0, Term *out1); // DUP with label
u32      thvm_fresh_label(TinyHVM *ctx);                     // allocate next label

// ICC: Bridge + Annotation
Term     thvm_bri(TinyHVM *ctx, Term *var_out, Term body);   // θx.body (dual of lambda)
Term     thvm_ann(TinyHVM *ctx, Term term, Term type);       // {term : type}

// Dynamic labels + priority
Term     thvm_dsu(TinyHVM *ctx, Term label_expr, Term a, Term b);  // dynamic SUP
Term     thvm_ddu(TinyHVM *ctx, Term label_expr, Term val, Term bod); // dynamic DUP
Term     thvm_inc(TinyHVM *ctx, Term term);                  // priority wrapper

// Collapse: extract all branches from a superposed term
typedef struct {
    Term   *terms;   // flat array of leaf terms
    u32     count;
    u32     cap;
} CollapseResult;
CollapseResult thvm_collapse(TinyHVM *ctx, Term t);
void           thvm_collapse_free(CollapseResult *cr);
CollapseResult thvm_collapse_par(TinyHVM *ctx, Term t, u32 n_threads);

// Grouped collapse: each result carries its (label, branch) path
typedef struct { u32 label; u8 branch; } LabelChoice;
typedef struct { Term term; LabelChoice *path; u32 path_len; } GroupedLeaf;
typedef struct { GroupedLeaf *leaves; u32 count; u32 cap; } GroupedCollapseResult;
GroupedCollapseResult thvm_collapse_grouped(TinyHVM *ctx, Term t);
void thvm_collapse_grouped_free(GroupedCollapseResult *gr);

// Type enumeration + unordered SUPs
Term     thvm_eql(TinyHVM *ctx, Term a, Term b);
Term     thvm_and(TinyHVM *ctx, Term a, Term b);
Term     thvm_or(TinyHVM *ctx, Term a, Term b);
Term     thvm_mat(TinyHVM *ctx, u32 match_tag, Term handler, Term fallback);
static inline Term thvm_any(void) { return term_new(TAG_ANY, 0, 0); }
Term     thvm_usp(TinyHVM *ctx, u32 label, Term a, Term b);
Term     thvm_udp(TinyHVM *ctx, u32 label, Term val);
CollapseResult thvm_collapse_ordered(TinyHVM *ctx, Term t, u32 limit);

// Inet ops
Term     thvm_where(TinyHVM *ctx, Term cond, Term then_t, Term else_t);
Term     thvm_assign(TinyHVM *ctx, Term dst, Term src);      // in-place weight update
Term     thvm_ifz(TinyHVM *ctx, Term counter, Term zero_case, Term succ_lam); // if-zero branch
Term     thvm_log_print(TinyHVM *ctx, Term tensor);  // print scalar value, return tensor

// Profiling (dispatches to backend->profile_report/reset)
void     thvm_profile_report(TinyHVM *ctx);
void     thvm_profile_reset(TinyHVM *ctx);

// Interaction tracing
void     thvm_trace_enable(TinyHVM *ctx, int enabled);
void     thvm_trace_clear(TinyHVM *ctx);

// ============================================================
// Layer abstraction — sequential composition
// ============================================================

typedef enum {
    LAYER_CONV2D, LAYER_BN, LAYER_MAXPOOL,
    LAYER_FLATTEN, LAYER_LINEAR, LAYER_FN
} LayerType;

typedef Term (*LayerFn)(TinyHVM *ctx, Term x);

typedef struct {
    LayerType type;
    union {
        struct { Term w, b; u32 ci, co, k; }    conv;
        struct { Term gamma, beta, rmean, rvar; u32 c, h, w; } bn;
        struct { u32 ks; }                       pool;
        struct { u32 flat_features; }             flat;
        struct { Term w, b; u32 in_f, out_f; }   lin;
        LayerFn fn;
    };
} Layer;

// Build model as layer list, run with thvm_sequential
Term     thvm_sequential(TinyHVM *ctx, Term x, Layer *layers, u32 n,
                         u32 BS, int training);

// Eval: argmax along last axis, returns u32 tensor of predictions
Term     thvm_argmax(TinyHVM *ctx, Term x, u32 rows, u32 cols);
// Eval accuracy: (argmax(logits) == labels).mean() * 100
f32      thvm_eval_accuracy(TinyHVM *ctx, Term logits, const u8 *labels,
                            u32 n_samples, u32 n_classes);

// Graph-level gradient (JAX-style)
// Returns a lazy Term — when reduced, computes ∂y/∂x.
// Gradient ops go through thvm_op → get taped → grad(grad(f)) works.
Term     thvm_grad(TinyHVM *ctx, Term y, Term x);
Term     thvm_grad_multi(TinyHVM *ctx, Term loss, Term *params, Term *grad_slots, u32 n_params);

// Internal GRAD target registry used by phase-1 GRAD interactions and debug labels.
void     thvm_grad_targets_clear(TinyHVM *ctx);
void     thvm_grad_target_set(TinyHVM *ctx, u64 grad_loc, Term x);
Term     thvm_grad_target_get(TinyHVM *ctx, u64 grad_loc);
void     thvm_grad_targets_set(TinyHVM *ctx, Term *params, Term *grad_slots, u32 n_params);
int      thvm_grad_targets_find_slot(TinyHVM *ctx, u32 tid, Term *out_slot);
u32      thvm_grad_targets_count(TinyHVM *ctx);
u32      thvm_grad_targets_get_tid(TinyHVM *ctx, u32 index);
void     term_use_clear(void);

// Movement ops
Term     thvm_pad(TinyHVM *ctx, Term t, const u32 *pairs, u32 ndim);
Term     thvm_shrink(TinyHVM *ctx, Term t, const u32 *pairs, u32 ndim);

// Tinygrad-style UOp compositions
Term     thvm_pool(TinyHVM *ctx, Term x, const u32 *kernel, const u32 *stride_,
                   u32 n_spatial);
Term     thvm_conv2d(TinyHVM *ctx, Term x, Term w, Term bias,
                     u32 groups, const u32 *stride_, const u32 *padding_);
Term     thvm_maxpool2d(TinyHVM *ctx, Term x, const u32 *kernel, const u32 *stride_);

#endif // TINYHVM_H
