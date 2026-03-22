// gpu_cpu.c — CPU fallback backend using Accelerate BLAS
// No GPU, pure CPU computation for correctness testing.

#include "tinyhvm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define HAS_BLAS 1
#else
#define HAS_BLAS 0
#endif

// Simple buffer pool: ID → pointer
#define MAX_BUFS 8192

static struct {
    void *bufs[MAX_BUFS];
    u64   sizes[MAX_BUFS];
    u32   count;
} cpu_pool;

static int cpu_init(void) {
    memset(&cpu_pool, 0, sizeof(cpu_pool));
    cpu_pool.count = 1;  // ID 0 reserved
    return 0;
}

static void cpu_shutdown(void) {
    for (u32 i = 1; i < cpu_pool.count; i++) {
        free(cpu_pool.bufs[i]);
    }
    memset(&cpu_pool, 0, sizeof(cpu_pool));
}

static u32 cpu_buf_alloc(u64 bytes) {
    u32 id = cpu_pool.count++;
    cpu_pool.bufs[id] = calloc(1, bytes);
    cpu_pool.sizes[id] = bytes;
    return id;
}

static void cpu_buf_free(u32 id) {
    free(cpu_pool.bufs[id]);
    cpu_pool.bufs[id] = NULL;
}

static void cpu_buf_write(u32 id, const void *data, u64 bytes) {
    memcpy(cpu_pool.bufs[id], data, bytes);
}

static void cpu_buf_read(u32 id, void *out, u64 bytes) {
    memcpy(out, cpu_pool.bufs[id], bytes);
}

static void cpu_op_add(u32 dst, u32 a, u32 b, u32 n) {
    f32 *pd = cpu_pool.bufs[dst];
    f32 *pa = cpu_pool.bufs[a];
    f32 *pb = cpu_pool.bufs[b];
    for (u32 i = 0; i < n; i++) pd[i] = pa[i] + pb[i];
}

static void cpu_op_mul(u32 dst, u32 a, u32 b, u32 n) {
    f32 *pd = cpu_pool.bufs[dst];
    f32 *pa = cpu_pool.bufs[a];
    f32 *pb = cpu_pool.bufs[b];
    for (u32 i = 0; i < n; i++) pd[i] = pa[i] * pb[i];
}

static void cpu_op_relu(u32 dst, u32 src, u32 n) {
    f32 *pd = cpu_pool.bufs[dst];
    f32 *ps = cpu_pool.bufs[src];
    for (u32 i = 0; i < n; i++) pd[i] = ps[i] > 0.0f ? ps[i] : 0.0f;
}

static void cpu_op_neg(u32 dst, u32 src, u32 n) {
    f32 *pd = cpu_pool.bufs[dst];
    f32 *ps = cpu_pool.bufs[src];
    for (u32 i = 0; i < n; i++) pd[i] = -ps[i];
}

static void cpu_op_mm(u32 dst, u32 a, u32 b, u32 M, u32 K, u32 N) {
#if HAS_BLAS
    // Use Accelerate BLAS: C = A * B
    // cblas_sgemm(order, trA, trB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                1.0f,
                (f32 *)cpu_pool.bufs[a], (int)K,
                (f32 *)cpu_pool.bufs[b], (int)N,
                0.0f,
                (f32 *)cpu_pool.bufs[dst], (int)N);
#else
    // Naive matmul fallback
    f32 *pd = cpu_pool.bufs[dst];
    f32 *pa = cpu_pool.bufs[a];
    f32 *pb = cpu_pool.bufs[b];
    for (u32 i = 0; i < M; i++) {
        for (u32 j = 0; j < N; j++) {
            f32 sum = 0.0f;
            for (u32 k = 0; k < K; k++)
                sum += pa[i * K + k] * pb[k * N + j];
            pd[i * N + j] = sum;
        }
    }
#endif
}

GpuBackend gpu_cpu_backend = {
    .init     = cpu_init,
    .shutdown = cpu_shutdown,
    .buf_alloc = cpu_buf_alloc,
    .buf_free  = cpu_buf_free,
    .buf_write = cpu_buf_write,
    .buf_read  = cpu_buf_read,
    .op_add  = cpu_op_add,
    .op_mul  = cpu_op_mul,
    .op_relu = cpu_op_relu,
    .op_neg  = cpu_op_neg,
    .op_mm   = cpu_op_mm,
};
