// shaders.metal — TinyHVM compute kernels
// Pre-built kernels for unary, binary, and strided ops.
// MPS handles matmul separately.

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Strided indexing (matches cpu's strided_index)
// ============================================================

struct ViewParams {
    int32_t strides[8];
    uint32_t shape[8];
    int32_t  offset;
    uint32_t rank;
    uint32_t numel;
};

static inline uint strided_idx(uint flat, constant ViewParams &v) {
    uint idx = uint(v.offset);
    uint rem = flat;
    for (int d = int(v.rank) - 1; d >= 0; d--) {
        uint coord = rem % v.shape[d];
        rem /= v.shape[d];
        idx += coord * uint(v.strides[d]);
    }
    return idx;
}

// ============================================================
// Unary ops (strided)
// ============================================================

kernel void unary_neg(device float *dst [[buffer(0)]],
                      device const float *src [[buffer(1)]],
                      constant ViewParams &dv [[buffer(2)]],
                      constant ViewParams &sv [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    dst[i] = -src[strided_idx(i, sv)];
}

kernel void unary_relu(device float *dst [[buffer(0)]],
                       device const float *src [[buffer(1)]],
                       constant ViewParams &dv [[buffer(2)]],
                       constant ViewParams &sv [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    float val = src[strided_idx(i, sv)];
    dst[i] = val > 0.0f ? val : 0.0f;
}

kernel void unary_exp(device float *dst [[buffer(0)]],
                      device const float *src [[buffer(1)]],
                      constant ViewParams &dv [[buffer(2)]],
                      constant ViewParams &sv [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    dst[i] = exp(src[strided_idx(i, sv)]);
}

kernel void unary_log(device float *dst [[buffer(0)]],
                      device const float *src [[buffer(1)]],
                      constant ViewParams &dv [[buffer(2)]],
                      constant ViewParams &sv [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    dst[i] = log(src[strided_idx(i, sv)]);
}

// ============================================================
// Binary ops (strided, handles broadcasting via stride=0)
// ============================================================

kernel void binary_add(device float *dst [[buffer(0)]],
                       device const float *a [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       constant ViewParams &dv [[buffer(3)]],
                       constant ViewParams &av [[buffer(4)]],
                       constant ViewParams &bv [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    dst[i] = a[strided_idx(i, av)] + b[strided_idx(i, bv)];
}

kernel void binary_mul(device float *dst [[buffer(0)]],
                       device const float *a [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       constant ViewParams &dv [[buffer(3)]],
                       constant ViewParams &av [[buffer(4)]],
                       constant ViewParams &bv [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    dst[i] = a[strided_idx(i, av)] * b[strided_idx(i, bv)];
}

kernel void binary_sub(device float *dst [[buffer(0)]],
                       device const float *a [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       constant ViewParams &dv [[buffer(3)]],
                       constant ViewParams &av [[buffer(4)]],
                       constant ViewParams &bv [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    dst[i] = a[strided_idx(i, av)] - b[strided_idx(i, bv)];
}

kernel void binary_div(device float *dst [[buffer(0)]],
                       device const float *a [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       constant ViewParams &dv [[buffer(3)]],
                       constant ViewParams &av [[buffer(4)]],
                       constant ViewParams &bv [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    float vb = b[strided_idx(i, bv)];
    dst[i] = vb != 0.0f ? a[strided_idx(i, av)] / vb : 0.0f;
}

kernel void binary_max(device float *dst [[buffer(0)]],
                       device const float *a [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       constant ViewParams &dv [[buffer(3)]],
                       constant ViewParams &av [[buffer(4)]],
                       constant ViewParams &bv [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    float va = a[strided_idx(i, av)], vb = b[strided_idx(i, bv)];
    dst[i] = va > vb ? va : vb;
}

kernel void binary_cmp(device float *dst [[buffer(0)]],
                       device const float *a [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       constant ViewParams &dv [[buffer(3)]],
                       constant ViewParams &av [[buffer(4)]],
                       constant ViewParams &bv [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    float va = a[strided_idx(i, av)], vb = b[strided_idx(i, bv)];
    dst[i] = va > vb ? 1.0f : 0.0f;
}

// ============================================================
// Naive matmul (fallback — MPS is the fast path)
// C[M,N] = A[M,K] * B[K,N]
// ============================================================

kernel void matmul_f32(device float *C [[buffer(0)]],
                       device const float *A [[buffer(1)]],
                       device const float *B [[buffer(2)]],
                       constant uint &M [[buffer(3)]],
                       constant uint &K [[buffer(4)]],
                       constant uint &N [[buffer(5)]],
                       uint2 pos [[thread_position_in_grid]]) {
    uint row = pos.y, col = pos.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (uint k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}
