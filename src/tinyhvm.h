// tinyhvm.h — TinyHVM: interaction-net runtime for tensor computation
// Minimal. No dependencies. Inspired by HVM4's bit layout.

#ifndef TINYHVM_H
#define TINYHVM_H

#include <stdint.h>
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

// Elementwise (binary):
#define UOP_ADD     8
#define UOP_MUL     9
#define UOP_DIV    10
#define UOP_MAX    11
#define UOP_CMP    12   // compare (returns 0/1)
#define UOP_SUB    13   // subtract (for gradient descent)

// Reduce:
#define UOP_SUM    14   // reduce sum over axis
#define UOP_RMAX   15   // reduce max over axis

// Matmul (special — dispatches to BLAS/MPS):
#define UOP_MM     16   // matrix multiply

// Movement (metadata-only, stride manipulation):
#define UOP_RESHAPE   17
#define UOP_PERMUTE   18
#define UOP_EXPAND    19  // = broadcast
#define UOP_SHRINK    20  // = slice
#define UOP_PAD       21

#define UOP_COUNT     22

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

#define MAX_DIM 8
#define MAX_TENSORS 4096
#define MAX_TAPE 4096

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
    u32         src_ids[2]; // input tensor ids (for backward rules)
} TensorMeta;

// Autograd tape entry (records forward ops for backward pass)
typedef struct {
    u32 uop;
    u32 out_id;
    u32 src_ids[2];
} TapeEntry;

// ============================================================
// GPU Backend Interface
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
} GpuBackend;

// ============================================================
// Context
// ============================================================

typedef struct {
    Term       *heap;
    u64         heap_pos;
    TensorMeta  tensors[MAX_TENSORS];
    u32         tensor_count;
    GpuBackend *gpu;
    u64         itrs;       // interaction count

    // Autograd
    TapeEntry   tape[MAX_TAPE];
    u32         tape_len;
    u8          recording;  // 1 if taping forward ops
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
GpuBackend *thvm_device(const char *name);  // "cpu", "metal"
TinyHVM *thvm_init(GpuBackend *gpu);
void     thvm_free(TinyHVM *ctx);

// Reduction
Term     thvm_reduce(TinyHVM *ctx, Term t);

// Tensor API
Term     thvm_tensor(TinyHVM *ctx, const f32 *data, Shape s);
Term     thvm_op(TinyHVM *ctx, u32 uop, Term a, Term b);
void     thvm_realize(TinyHVM *ctx, Term t);
f32     *thvm_to_host(TinyHVM *ctx, Term t);

// Print
void     thvm_print_term(TinyHVM *ctx, Term t);

// Autograd
void     thvm_set_requires_grad(TinyHVM *ctx, Term t);
void     thvm_start_recording(TinyHVM *ctx);
void     thvm_stop_recording(TinyHVM *ctx);
void     thvm_clear_tape(TinyHVM *ctx);

// Graph-level gradient (JAX-style)
// Returns a lazy Term — when reduced, computes ∂y/∂x.
// Gradient ops go through thvm_op → get taped → grad(grad(f)) works.
Term     thvm_grad(TinyHVM *ctx, Term y, Term x);

#endif // TINYHVM_H
