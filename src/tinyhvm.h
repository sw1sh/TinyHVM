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

#define TAG_COUNT 13

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

#define UOP_COUNT     28

// Internal ops — not part of tinyspec, only used for autograd provenance
#define UOP_POOL_GATHER 100  // sliding window gather (im2col equivalent)
#define UOP_GRAD        101  // IC gradient: DUP-op interaction in the reducer

// UOp name table (device-agnostic)
static const char *uop_names[] = {
    "LOAD","STORE","COPY","NEG","EXP","LOG","RELU","CAST","SQRT",
    "ADD","MUL","DIV","MAX","CMP","SUB","SUM","RMAX","MM",
    "RESHAPE","PERMUTE","EXPAND","SHRINK","PAD","FUSING","ASSIGN","WHERE",
    "IFZ","LOG_PRINT"
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

#define HEAP_CAP (1ULL << 28)  // 256M terms = 2GB

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
#define MAX_TENSORS 524288

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
} View;

typedef struct {
    u32         buf_id;     // GPU buffer handle
    u32         dtype;
    u32         refcount;
    View        view;
    void       *host_ptr;   // cached host copy

    // Autograd provenance
    u8          requires_grad;
    u32         creator_op; // UOP that created this tensor
    u32         src_ids[2]; // input tensor ids (for backward rules + REACHES traversal)

    // Fusion metadata (only when creator_op == UOP_FUSING)
    u64         fusing_loc; // heap loc of the original subnet root TAG_TOP
    u32         fusing_uop; // UOP of the subnet root (e.g. UOP_SUM)
} TensorMeta;



// ============================================================
// Backend Interface
// ============================================================

typedef struct {
    int   (*init)(void);
    void  (*shutdown)(void);
    u32   (*buf_alloc)(u64 bytes);
    void  (*buf_free)(u32 id);
    void  (*buf_write)(u32 id, const void *data, u64 bytes);
    void  (*buf_read)(u32 id, void *out, u64 bytes);
    // Strided ops — full View info for broadcasting/transpose
    void  (*op_unary)(u32 uop, u32 dst, const View *dv,
                      u32 src, const View *sv);
    void  (*op_binary)(u32 uop, u32 dst, const View *dv,
                       u32 a, const View *av, u32 b, const View *bv);
    void  (*op_mm)(u32 dst, u32 a, const View *av, u32 b, const View *bv,
                   u32 M, u32 K, u32 N);
    // Reduce op: reduce src along last axis into dst
    // dst shape = src shape with last dim collapsed to 1
    void  (*op_reduce)(u32 uop, u32 dst, u32 dst_numel,
                       u32 src, u32 src_numel, u32 reduce_dim);

    // CNN ops — used by layers.c, device-agnostic
    void  (*op_im2col)(u32 dst, u32 src, Conv2dParams p);
    void  (*op_col2im)(u32 dst, u32 src, Conv2dParams p);
    void  (*op_nhwc_to_nchw)(u32 dst, u32 src, u32 B, u32 C, u32 H, u32 W);
    void  (*op_nchw_to_nhwc)(u32 dst, u32 src, u32 B, u32 C, u32 H, u32 W);
    void  (*op_bias_add)(u32 buf, u32 bias, u32 C, u32 n);
    void  (*op_col_sum)(u32 dst, u32 src, u32 N, u32 C);
    void  (*op_transpose)(u32 dst, u32 src, u32 M, u32 N);
    void  (*op_maxpool_fwd)(u32 out, u32 mask, u32 src, u32 B, u32 C, u32 H, u32 W);
    void  (*op_maxpool_bwd)(u32 dx, u32 dout, u32 mask, u32 B, u32 C, u32 H, u32 W);
    void  (*op_relu_bwd)(u32 dx, u32 dout, u32 x, u32 n);
    void  (*op_zero_fill)(u32 buf, u32 n);
    void  (*op_adam_step)(u32 param, u32 grad, u32 m, u32 v,
                          f32 lr, f32 beta1, f32 beta2, f32 eps, f32 bc1, f32 bc2, u32 n);

    // Pool management: reset buffer pool to `keep` entries
    void  (*pool_reset)(u32 keep);
    // Command buffer batching: accumulate GPU work without sync
    void  (*begin_batch)(void);
    void  (*end_batch)(void);
    // Profiling: kernel-level timing (like tinygrad's GlobalCounters)
    void  (*profile_report)(void);
    void  (*profile_reset)(void);
} Backend;

// ============================================================
// Context
// ============================================================

typedef struct {
    Term       *heap;
    u64         heap_pos;
    TensorMeta  tensors[MAX_TENSORS];
    u32         tensor_count;
    Backend *backend;
    u64         itrs;       // interaction count
    u8          no_fuse;    // 1 to skip fusion (used during GRAD subnet re-reduction)

    // Named definitions for TAG_REF (global def table)
    Term        defs[256];   // defs[name] = heap loc or TAG_TOP term
    u32         def_count;
} TinyHVM;

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
Backend *thvm_device(const char *name);  // "cpu", "metal"
TinyHVM *thvm_init(Backend *backend);
void     thvm_free(TinyHVM *ctx);
void     thvm_reset(TinyHVM *ctx, u32 keep);  // free tensors above `keep`, reset heap

// Reduction
Term     thvm_reduce(TinyHVM *ctx, Term t);

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

// Autograd
void     thvm_set_requires_grad(TinyHVM *ctx, Term t);

// Lambda / inet combinators
Term     thvm_lam(TinyHVM *ctx, Term *var_out, Term body);   // allocate LAM node
Term     thvm_app(TinyHVM *ctx, Term fun, Term arg);         // allocate APP node
u32      thvm_define(TinyHVM *ctx, Term body);               // register def, return name id
Term     thvm_ref(TinyHVM *ctx, u32 name);                   // TAG_REF(name)
Term     thvm_sup(TinyHVM *ctx, Term a, Term b);             // TAG_SUP(a, b)

// Inet ops
Term     thvm_where(TinyHVM *ctx, Term cond, Term then_t, Term else_t);
Term     thvm_assign(TinyHVM *ctx, Term dst, Term src);      // in-place weight update
Term     thvm_ifz(TinyHVM *ctx, Term counter, Term zero_case, Term succ_lam); // if-zero branch
Term     thvm_log_print(TinyHVM *ctx, Term tensor);  // print scalar value, return tensor

// Profiling (dispatches to backend->profile_report/reset)
void     thvm_profile_report(TinyHVM *ctx);
void     thvm_profile_reset(TinyHVM *ctx);

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
        struct { Term gamma, beta, rmean, rvar; u32 c; } bn;
        struct { u32 ks; }                       pool;
        struct { u32 from_dim; }                  flat;
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
void     thvm_backward(TinyHVM *ctx, Term loss, Term *params, Term *grads, u32 n_params);

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
