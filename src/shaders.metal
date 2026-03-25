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
    uint32_t has_mask;
    uint32_t mask_begin[8];
    uint32_t mask_end[8];
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

static inline float masked_read(device const float *buf, uint flat,
                                 constant ViewParams &v) {
    if (v.has_mask) {
        uint rem = flat;
        for (int d = int(v.rank) - 1; d >= 0; d--) {
            uint coord = rem % v.shape[d];
            rem /= v.shape[d];
            if (coord < v.mask_begin[d] || coord >= v.mask_end[d])
                return 0.0f;
        }
    }
    return buf[strided_idx(flat, v)];
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
    dst[i] = -masked_read(src, i, sv);
}

kernel void unary_relu(device float *dst [[buffer(0)]],
                       device const float *src [[buffer(1)]],
                       constant ViewParams &dv [[buffer(2)]],
                       constant ViewParams &sv [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    float val = masked_read(src, i, sv);
    dst[i] = val > 0.0f ? val : 0.0f;
}

kernel void unary_exp(device float *dst [[buffer(0)]],
                      device const float *src [[buffer(1)]],
                      constant ViewParams &dv [[buffer(2)]],
                      constant ViewParams &sv [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    dst[i] = exp(masked_read(src, i, sv));
}

kernel void unary_log(device float *dst [[buffer(0)]],
                      device const float *src [[buffer(1)]],
                      constant ViewParams &dv [[buffer(2)]],
                      constant ViewParams &sv [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    dst[i] = log(masked_read(src, i, sv));
}

kernel void unary_sqrt(device float *dst [[buffer(0)]],
                       device const float *src [[buffer(1)]],
                       constant ViewParams &dv [[buffer(2)]],
                       constant ViewParams &sv [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    dst[i] = sqrt(masked_read(src, i, sv));
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
    dst[i] = masked_read(a, i, av) + masked_read(b, i, bv);
}

kernel void binary_mul(device float *dst [[buffer(0)]],
                       device const float *a [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       constant ViewParams &dv [[buffer(3)]],
                       constant ViewParams &av [[buffer(4)]],
                       constant ViewParams &bv [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    dst[i] = masked_read(a, i, av) * masked_read(b, i, bv);
}

kernel void binary_sub(device float *dst [[buffer(0)]],
                       device const float *a [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       constant ViewParams &dv [[buffer(3)]],
                       constant ViewParams &av [[buffer(4)]],
                       constant ViewParams &bv [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    dst[i] = masked_read(a, i, av) - masked_read(b, i, bv);
}

kernel void binary_div(device float *dst [[buffer(0)]],
                       device const float *a [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       constant ViewParams &dv [[buffer(3)]],
                       constant ViewParams &av [[buffer(4)]],
                       constant ViewParams &bv [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    float vb = masked_read(b, i, bv);
    dst[i] = vb != 0.0f ? masked_read(a, i, av) / vb : 0.0f;
}

kernel void binary_max(device float *dst [[buffer(0)]],
                       device const float *a [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       constant ViewParams &dv [[buffer(3)]],
                       constant ViewParams &av [[buffer(4)]],
                       constant ViewParams &bv [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= dv.numel) return;
    float va = masked_read(a, i, av), vb = masked_read(b, i, bv);
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
    float va = masked_read(a, i, av), vb = masked_read(b, i, bv);
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

// ============================================================
// Adam optimizer step (per-element)
// param[i] -= lr * m_hat / (sqrt(v_hat) + eps)
// ============================================================

struct AdamParams {
    float lr, beta1, beta2, eps;
    float bc1, bc2;  // bias correction: 1 - beta^t
};

kernel void adam_step(device float *param [[buffer(0)]],
                      device const float *grad [[buffer(1)]],
                      device float *m [[buffer(2)]],
                      device float *v [[buffer(3)]],
                      constant AdamParams &p [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
    float g = grad[gid];
    float mi = p.beta1 * m[gid] + (1.0f - p.beta1) * g;
    float vi = p.beta2 * v[gid] + (1.0f - p.beta2) * g * g;
    m[gid] = mi;
    v[gid] = vi;
    float m_hat = mi / p.bc1;
    float v_hat = vi / p.bc2;
    param[gid] -= p.lr * m_hat / (sqrt(v_hat) + p.eps);
}

// ============================================================
// MaxPool2d 2x2 stride 2 — forward with argmax mask
// Thread per output element
// ============================================================

kernel void maxpool2d_fwd(device float *out [[buffer(0)]],
                          device uchar *mask [[buffer(1)]],
                          device const float *x [[buffer(2)]],
                          constant LayoutParams &p [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    uint OH = p.H / 2, OW = p.W / 2;
    uint total = p.B * p.C * OH * OW;
    if (gid >= total) return;

    uint b = gid / (p.C * OH * OW);
    uint rem = gid % (p.C * OH * OW);
    uint c = rem / (OH * OW);
    rem = rem % (OH * OW);
    uint oh = rem / OW;
    uint ow = rem % OW;

    uint base = (b * p.C + c) * p.H * p.W;
    uint h0 = oh * 2, w0 = ow * 2;
    float v0 = x[base + h0 * p.W + w0];
    float v1 = x[base + h0 * p.W + w0 + 1];
    float v2 = x[base + (h0+1) * p.W + w0];
    float v3 = x[base + (h0+1) * p.W + w0 + 1];

    uchar mi = 0;
    float mx = v0;
    if (v1 > mx) { mx = v1; mi = 1; }
    if (v2 > mx) { mx = v2; mi = 2; }
    if (v3 > mx) { mx = v3; mi = 3; }

    out[gid] = mx;
    mask[gid] = mi;
}

// MaxPool2d backward: scatter gradient using mask
kernel void maxpool2d_bwd(device float *dx [[buffer(0)]],
                          device const float *dout [[buffer(1)]],
                          device const uchar *mask [[buffer(2)]],
                          constant LayoutParams &p [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    uint OH = p.H / 2, OW = p.W / 2;
    uint total = p.B * p.C * OH * OW;
    if (gid >= total) return;

    uint b = gid / (p.C * OH * OW);
    uint rem = gid % (p.C * OH * OW);
    uint c = rem / (OH * OW);
    rem = rem % (OH * OW);
    uint oh = rem / OW;
    uint ow = rem % OW;

    uchar mi = mask[gid];
    uint h0 = oh * 2 + mi / 2;
    uint w0 = ow * 2 + mi % 2;
    // Note: dx must be zeroed first
    dx[(b * p.C + c) * p.H * p.W + h0 * p.W + w0] = dout[gid];
}

// ============================================================
// ReLU backward: dx = x > 0 ? dout : 0
// ============================================================

kernel void relu_bwd(device float *dx [[buffer(0)]],
                     device const float *dout [[buffer(1)]],
                     device const float *x [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    dx[gid] = x[gid] > 0.0f ? dout[gid] : 0.0f;
}

// ============================================================
// Matrix transpose: [M, N] → [N, M]
// ============================================================

struct TransposeParams {
    uint M, N;
};

kernel void matrix_transpose(device float *dst [[buffer(0)]],
                              device const float *src [[buffer(1)]],
                              constant TransposeParams &p [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= p.M * p.N) return;
    uint i = gid / p.N;
    uint j = gid % p.N;
    dst[j * p.M + i] = src[gid];
}

// ============================================================
// Zero fill
// ============================================================

kernel void zero_fill(device float *dst [[buffer(0)]],
                      uint gid [[thread_position_in_grid]]) {
    dst[gid] = 0.0f;
}

// ============================================================
// Fused MUL + SUM — avoids materializing the full MUL intermediate.
// Thread per output element. Supports multi-axis reduction.
// ============================================================

struct MulReduceParams {
    uint n_reduce;           // number of reduce axes
    uint reduce_numel;       // product of all reduce dims
    uint reduce_dims[8];     // size of each reduce axis
    uint reduce_strides_a[8]; // stride in a for each reduce axis
    uint reduce_strides_b[8]; // stride in b for each reduce axis
};

kernel void mul_reduce_sum(device float *dst [[buffer(0)]],
                           device const float *a [[buffer(1)]],
                           device const float *b [[buffer(2)]],
                           constant ViewParams &av [[buffer(3)]],
                           constant ViewParams &bv [[buffer(4)]],
                           constant ViewParams &ov [[buffer(5)]],
                           constant MulReduceParams &rp [[buffer(6)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= ov.numel) return;

    // Compute N-d coordinates from flat output index using output shape
    uint coords[8];
    uint rem = i;
    for (int d = int(ov.rank) - 1; d >= 0; d--) {
        coords[d] = rem % ov.shape[d];
        rem /= ov.shape[d];
    }

    // Compute base indices into a and b for these coordinates
    uint base_a = uint(av.offset);
    uint base_b = uint(bv.offset);
    for (uint d = 0; d < ov.rank; d++) {
        base_a += coords[d] * uint(av.strides[d]);
        base_b += coords[d] * uint(bv.strides[d]);
    }

    // Accumulate over all reduce axes
    float acc = 0.0f;
    for (uint r = 0; r < rp.reduce_numel; r++) {
        uint off_a = 0, off_b = 0, rr = r;
        for (uint ri = 0; ri < rp.n_reduce; ri++) {
            uint rc = rr % rp.reduce_dims[ri];
            rr /= rp.reduce_dims[ri];
            off_a += rc * rp.reduce_strides_a[ri];
            off_b += rc * rp.reduce_strides_b[ri];
        }
        // Mask check for a and b
        float va = a[base_a + off_a];
        float vb = b[base_b + off_b];
        // TODO: add full mask checking for mul_reduce_sum inputs
        // For now, inputs with masks should be materialized before reaching this kernel
        acc += va * vb;
    }
    dst[i] = acc;
}

// ============================================================
// PAD: zero-pad a tensor (thread per OUTPUT element)
// ============================================================

kernel void pad_kernel(device float *dst [[buffer(0)]],
                       device const float *src [[buffer(1)]],
                       constant ViewParams &sv [[buffer(2)]],
                       constant int *pad_before [[buffer(3)]],
                       constant uint *src_shape [[buffer(4)]],
                       constant uint *dst_shape [[buffer(5)]],
                       constant uint &rank [[buffer(6)]],
                       constant uint &dst_numel [[buffer(7)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= dst_numel) return;
    // Decompose gid into dst coordinates
    uint coords[8], rem = gid;
    for (int d = int(rank) - 1; d >= 0; d--) {
        coords[d] = rem % dst_shape[d];
        rem /= dst_shape[d];
    }
    // Check if inside source region
    uint src_idx = 0, src_stride = 1;
    bool inside = true;
    for (int d = int(rank) - 1; d >= 0; d--) {
        int sc = int(coords[d]) - pad_before[d];
        if (sc < 0 || uint(sc) >= src_shape[d]) { inside = false; break; }
        src_idx += uint(sc) * src_stride;
        src_stride *= src_shape[d];
    }
    dst[gid] = inside ? src[src_idx] : 0.0f;
}
