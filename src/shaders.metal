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

kernel void unary_sqrt(device float *dst [[buffer(0)]],
                       device const float *src [[buffer(1)]],
                       constant ViewParams &dv [[buffer(2)]],
                       constant ViewParams &sv [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    dst[i] = sqrt(src[strided_idx(i, sv)]);
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

// ============================================================
// Reduce ops — one thread per output element
// src is [outer × reduce_dim], dst is [outer]
// ============================================================

kernel void reduce_sum(device float *dst [[buffer(0)]],
                       device const float *src [[buffer(1)]],
                       constant uint &reduce_dim [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    float acc = 0.0f;
    uint base = i * reduce_dim;
    for (uint r = 0; r < reduce_dim; r++)
        acc += src[base + r];
    dst[i] = acc;
}

kernel void reduce_max(device float *dst [[buffer(0)]],
                       device const float *src [[buffer(1)]],
                       constant uint &reduce_dim [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    float acc = -1e30f;
    uint base = i * reduce_dim;
    for (uint r = 0; r < reduce_dim; r++)
        acc = max(acc, src[base + r]);
    dst[i] = acc;
}

// ============================================================
// Im2col: [B, Cin, H, W] → [B*OH*OW, Cin*KH*KW]
// Thread per output element (row, col) in the col matrix
// ============================================================

struct Conv2dParams {
    uint B, Cin, H, W, KH, KW, OH, OW;
    uint patch_size;  // Cin * KH * KW
    uint n_patches;   // B * OH * OW
};

kernel void im2col(device float *col [[buffer(0)]],
                   device const float *x [[buffer(1)]],
                   constant Conv2dParams &p [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    // gid indexes into col[n_patches * patch_size]
    if (gid >= p.n_patches * p.patch_size) return;

    uint row = gid / p.patch_size;  // which patch
    uint col_idx = gid % p.patch_size;  // position within patch

    uint b  = row / (p.OH * p.OW);
    uint rem = row % (p.OH * p.OW);
    uint oh = rem / p.OW;
    uint ow = rem % p.OW;

    uint c  = col_idx / (p.KH * p.KW);
    uint rem2 = col_idx % (p.KH * p.KW);
    uint kh = rem2 / p.KW;
    uint kw = rem2 % p.KW;

    col[gid] = x[((b * p.Cin + c) * p.H + (oh + kh)) * p.W + (ow + kw)];
}

// ============================================================
// Col2im: [B*OH*OW, Cin*KH*KW] → [B, Cin, H, W] (scatter-add)
// Thread per output element in x (B*Cin*H*W)
// ============================================================

kernel void col2im(device float *dx [[buffer(0)]],
                   device const float *dcol [[buffer(1)]],
                   constant Conv2dParams &p [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    // gid indexes into dx[B * Cin * H * W]
    uint total = p.B * p.Cin * p.H * p.W;
    if (gid >= total) return;

    uint b = gid / (p.Cin * p.H * p.W);
    uint rem = gid % (p.Cin * p.H * p.W);
    uint c = rem / (p.H * p.W);
    rem = rem % (p.H * p.W);
    uint h = rem / p.W;
    uint w = rem % p.W;

    float sum = 0.0f;
    // For each (oh, ow, kh, kw) that maps to (b, c, h, w):
    // oh + kh = h, ow + kw = w → oh = h - kh, ow = w - kw
    for (uint kh = 0; kh < p.KH; kh++) {
        if (h < kh || h - kh >= p.OH) continue;
        uint oh = h - kh;
        for (uint kw = 0; kw < p.KW; kw++) {
            if (w < kw || w - kw >= p.OW) continue;
            uint ow = w - kw;
            uint row = b * p.OH * p.OW + oh * p.OW + ow;
            uint col_off = c * p.KH * p.KW + kh * p.KW + kw;
            sum += dcol[row * p.patch_size + col_off];
        }
    }
    dx[gid] = sum;
}

// ============================================================
// NHWC ↔ NCHW layout transpose (for matmul output → conv output)
// Thread per element
// ============================================================

struct LayoutParams {
    uint B, C, H, W;
};

// [B, H, W, C] → [B, C, H, W]
kernel void nhwc_to_nchw(device float *dst [[buffer(0)]],
                         device const float *src [[buffer(1)]],
                         constant LayoutParams &p [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    uint n = p.B * p.C * p.H * p.W;
    if (gid >= n) return;

    // dst index: in NCHW layout
    uint b = gid / (p.C * p.H * p.W);
    uint rem = gid % (p.C * p.H * p.W);
    uint c = rem / (p.H * p.W);
    rem = rem % (p.H * p.W);
    uint h = rem / p.W;
    uint w = rem % p.W;

    // src index: in NHWC layout
    uint src_idx = ((b * p.H + h) * p.W + w) * p.C + c;
    dst[gid] = src[src_idx];
}

// [B, C, H, W] → [B, H, W, C]
kernel void nchw_to_nhwc(device float *dst [[buffer(0)]],
                         device const float *src [[buffer(1)]],
                         constant LayoutParams &p [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    uint n = p.B * p.C * p.H * p.W;
    if (gid >= n) return;

    // gid indexes into NHWC output
    uint b = gid / (p.H * p.W * p.C);
    uint rem = gid % (p.H * p.W * p.C);
    uint h = rem / (p.W * p.C);
    rem = rem % (p.W * p.C);
    uint w = rem / p.C;
    uint c = rem % p.C;

    // src index: in NCHW layout
    uint src_idx = ((b * p.C + c) * p.H + h) * p.W + w;
    dst[gid] = src[src_idx];
}

// ============================================================
// Bias add: dst[i] += bias[i % C] (for conv output [n_patches, Cout])
// ============================================================

kernel void bias_add(device float *dst [[buffer(0)]],
                     device const float *bias [[buffer(1)]],
                     constant uint &C [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    dst[gid] += bias[gid % C];
}

// ============================================================
// Column-wise sum: sum along rows to get [C] from [N, C]
// Thread per output element
// ============================================================

kernel void col_sum(device float *dst [[buffer(0)]],
                    device const float *src [[buffer(1)]],
                    constant uint &N [[buffer(2)]],
                    constant uint &C [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= C) return;
    float sum = 0.0f;
    for (uint i = 0; i < N; i++)
        sum += src[i * C + gid];
    dst[gid] = sum;
}
