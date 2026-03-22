// gpu_cpu.c — CPU fallback backend with strided kernel dispatch
// Uses Accelerate BLAS for matmul, strided loops for everything else.

#include "tinyhvm.h"
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define HAS_BLAS 1
#else
#define HAS_BLAS 0
#endif

// ============================================================
// Buffer pool: ID → pointer
// ============================================================

#define MAX_BUFS 8192

static struct {
    void *bufs[MAX_BUFS];
    u64   sizes[MAX_BUFS];
    u32   count;
} cpu_pool;

static int  cpu_init(void)          { memset(&cpu_pool, 0, sizeof(cpu_pool)); cpu_pool.count = 1; return 0; }
static void cpu_shutdown(void)      { for (u32 i = 1; i < cpu_pool.count; i++) free(cpu_pool.bufs[i]); memset(&cpu_pool, 0, sizeof(cpu_pool)); }
static u32  cpu_buf_alloc(u64 b)    { u32 id = cpu_pool.count++; cpu_pool.bufs[id] = calloc(1, b); cpu_pool.sizes[id] = b; return id; }
static void cpu_buf_free(u32 id)    { free(cpu_pool.bufs[id]); cpu_pool.bufs[id] = NULL; }
static void cpu_buf_write(u32 id, const void *d, u64 b) { memcpy(cpu_pool.bufs[id], d, b); }
static void cpu_buf_read(u32 id, void *o, u64 b)        { memcpy(o, cpu_pool.bufs[id], b); }

// ============================================================
// Strided indexing helpers
// ============================================================

// Convert flat output index → strided input index
static inline u32 strided_index(u32 flat, const View *v) {
    u32 idx = (u32)v->offset;
    u32 rem = flat;
    for (i32 d = (i32)v->shape.rank - 1; d >= 0; d--) {
        u32 coord = rem % v->shape.dims[d];
        rem /= v->shape.dims[d];
        idx += coord * (u32)v->strides[d];
    }
    return idx;
}

// ============================================================
// Strided unary op
// ============================================================

static void cpu_op_unary(u32 uop, u32 dst, const View *dv,
                         u32 src, const View *sv) {
    f32 *pd = cpu_pool.bufs[dst];
    f32 *ps = cpu_pool.bufs[src];
    u32 n = dv->numel;

    for (u32 i = 0; i < n; i++) {
        u32 si = strided_index(i, sv);
        f32 val = ps[si];
        switch (uop) {
            case UOP_NEG:  pd[i] = -val; break;
            case UOP_RELU: pd[i] = val > 0.0f ? val : 0.0f; break;
            case UOP_EXP:  pd[i] = __builtin_expf(val); break;
            case UOP_LOG:  pd[i] = __builtin_logf(val); break;
            default:       pd[i] = val; break;
        }
    }
}

// ============================================================
// Strided binary op (handles broadcasting via stride=0)
// ============================================================

static void cpu_op_binary(u32 uop, u32 dst, const View *dv,
                          u32 a, const View *av, u32 b, const View *bv) {
    f32 *pd = cpu_pool.bufs[dst];
    f32 *pa = cpu_pool.bufs[a];
    f32 *pb = cpu_pool.bufs[b];
    u32 n = dv->numel;

    for (u32 i = 0; i < n; i++) {
        u32 ai = strided_index(i, av);
        u32 bi = strided_index(i, bv);
        f32 va = pa[ai], vb = pb[bi];
        switch (uop) {
            case UOP_ADD: pd[i] = va + vb; break;
            case UOP_MUL: pd[i] = va * vb; break;
            case UOP_DIV: pd[i] = vb != 0.0f ? va / vb : 0.0f; break;
            case UOP_SUB: pd[i] = va - vb; break;
            case UOP_MAX: pd[i] = va > vb ? va : vb; break;
            case UOP_CMP: pd[i] = va > vb ? 1.0f : 0.0f; break;
            default:      pd[i] = va; break;
        }
    }
}

// ============================================================
// Matmul (BLAS or naive, contiguous assumed)
// ============================================================

static void cpu_op_mm(u32 dst, u32 a, const View *av, u32 b, const View *bv,
                      u32 M, u32 K, u32 N) {
    (void)av; (void)bv;  // TODO: handle non-contiguous via copy
#if HAS_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K, 1.0f,
                (f32 *)cpu_pool.bufs[a], (int)K,
                (f32 *)cpu_pool.bufs[b], (int)N,
                0.0f, (f32 *)cpu_pool.bufs[dst], (int)N);
#else
    f32 *pd = cpu_pool.bufs[dst], *pa = cpu_pool.bufs[a], *pb = cpu_pool.bufs[b];
    for (u32 i = 0; i < M; i++)
        for (u32 j = 0; j < N; j++) {
            f32 s = 0;
            for (u32 k = 0; k < K; k++) s += pa[i*K+k] * pb[k*N+j];
            pd[i*N+j] = s;
        }
#endif
}

GpuBackend gpu_cpu_backend = {
    .init      = cpu_init,
    .shutdown  = cpu_shutdown,
    .buf_alloc = cpu_buf_alloc,
    .buf_free  = cpu_buf_free,
    .buf_write = cpu_buf_write,
    .buf_read  = cpu_buf_read,
    .op_unary  = cpu_op_unary,
    .op_binary = cpu_op_binary,
    .op_mm     = cpu_op_mm,
};
