// layers.c — High-level layer functions for CNN training
// Conv2d/Linear matmuls dispatch through backend->op_mm.
// im2col, pooling, BN, CE, Adam are host-side (small data or data movement).
//
// Convention: functions take TinyHVM *ctx, work with tensor IDs (u32).

#include "tinyhvm.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

// ============================================================
// Buffer helpers
// ============================================================

static f32 *buf_read_f32(TinyHVM *ctx, u32 tid, u32 n) {
    f32 *data = malloc(n * sizeof(f32));
    ctx->backend->buf_read(ctx->tensors[tid].buf_id, data, n * sizeof(f32));
    return data;
}

static void buf_write_f32(TinyHVM *ctx, u32 tid, const f32 *data, u32 n) {
    ctx->backend->buf_write(ctx->tensors[tid].buf_id, data, n * sizeof(f32));
}

// Allocate tensor with shape, return ID
static u32 tensor_alloc(TinyHVM *ctx, Shape s) {
    u32 id = ctx->tensor_count++;
    assert(id < MAX_TENSORS);
    TensorMeta *m = &ctx->tensors[id];
    memset(m, 0, sizeof(TensorMeta));
    m->dtype = DTYPE_F32;
    m->view.shape = s;
    m->view.numel = 1;
    for (u32 i = 0; i < s.rank; i++) m->view.numel *= s.dims[i];
    u32 stride = 1;
    for (i32 d = (i32)s.rank - 1; d >= 0; d--) {
        m->view.strides[d] = (i32)stride;
        stride *= s.dims[d];
    }
    m->view.contiguous = 1;
    m->buf_id = ctx->backend->buf_alloc((u64)m->view.numel * sizeof(f32));
    return id;
}

// Allocate + fill from host data
static u32 tensor_from_data(TinyHVM *ctx, const f32 *data, Shape s) {
    u32 id = tensor_alloc(ctx, s);
    buf_write_f32(ctx, id, data, ctx->tensors[id].view.numel);
    return id;
}

// Backend matmul: dst[M,N] = a[M,K] @ b[K,N]
static u32 mm(TinyHVM *ctx, u32 a, u32 b, u32 M, u32 K, u32 N) {
    u32 dst = tensor_alloc(ctx, SHAPE(M, N));
    ctx->backend->op_mm(ctx->tensors[dst].buf_id,
                        ctx->tensors[a].buf_id, &ctx->tensors[a].view,
                        ctx->tensors[b].buf_id, &ctx->tensors[b].view,
                        M, K, N);
    return dst;
}

// ============================================================
// Softmax + Cross-Entropy
// ============================================================

typedef struct {
    f32 loss;
    u32 softmax_id;
    u32 batch_size, n_classes;
} CrossEntropyResult;

static CrossEntropyResult cross_entropy_forward(TinyHVM *ctx, u32 logits_id,
                                                 const u8 *labels, u32 B, u32 C) {
    f32 *logits = buf_read_f32(ctx, logits_id, B * C);
    f32 *sm = malloc(B * C * sizeof(f32));
    f32 total_loss = 0;

    for (u32 b = 0; b < B; b++) {
        f32 *row = &logits[b * C];
        f32 mx = row[0];
        for (u32 j = 1; j < C; j++) if (row[j] > mx) mx = row[j];
        f32 se = 0;
        for (u32 j = 0; j < C; j++) { sm[b*C+j] = expf(row[j]-mx); se += sm[b*C+j]; }
        for (u32 j = 0; j < C; j++) sm[b*C+j] /= se;
        f32 p = sm[b*C+labels[b]];
        total_loss += -logf(p > 1e-7f ? p : 1e-7f);
    }

    u32 sm_id = tensor_from_data(ctx, sm, SHAPE(B, C));
    free(logits); free(sm);
    return (CrossEntropyResult){total_loss/(f32)B, sm_id, B, C};
}

static u32 cross_entropy_backward(TinyHVM *ctx, CrossEntropyResult *ce, const u8 *labels) {
    u32 B = ce->batch_size, C = ce->n_classes;
    f32 *sm = buf_read_f32(ctx, ce->softmax_id, B * C);
    for (u32 b = 0; b < B; b++) {
        sm[b*C+labels[b]] -= 1.0f;
        for (u32 j = 0; j < C; j++) sm[b*C+j] /= (f32)B;
    }
    u32 g = tensor_from_data(ctx, sm, SHAPE(B, C));
    free(sm);
    return g;
}

// ============================================================
// ReLU
// ============================================================

static u32 relu_forward(TinyHVM *ctx, u32 input_id, u32 n) {
    f32 *x = buf_read_f32(ctx, input_id, n);
    for (u32 i = 0; i < n; i++) x[i] = x[i] > 0 ? x[i] : 0;
    u32 out = tensor_alloc(ctx, ctx->tensors[input_id].view.shape);
    buf_write_f32(ctx, out, x, n);
    free(x);
    return out;
}

static u32 relu_backward(TinyHVM *ctx, u32 grad_id, u32 input_id, u32 n) {
    f32 *g = buf_read_f32(ctx, grad_id, n);
    f32 *x = buf_read_f32(ctx, input_id, n);
    for (u32 i = 0; i < n; i++) g[i] = x[i] > 0 ? g[i] : 0;
    u32 out = tensor_from_data(ctx, g, ctx->tensors[grad_id].view.shape);
    free(g); free(x);
    return out;
}

// ============================================================
// Linear — dispatches matmul through backend
// ============================================================

static u32 linear_forward(TinyHVM *ctx, u32 input_id, u32 weight_id, u32 bias_id,
                           u32 B, u32 K, u32 N) {
    // out[B,N] = input[B,K] @ weight[K,N]
    u32 out_id = mm(ctx, input_id, weight_id, B, K, N);

    // Add bias (host-side, cheap)
    f32 *out = buf_read_f32(ctx, out_id, B * N);
    f32 *bias = buf_read_f32(ctx, bias_id, N);
    for (u32 b = 0; b < B; b++)
        for (u32 j = 0; j < N; j++)
            out[b * N + j] += bias[j];
    buf_write_f32(ctx, out_id, out, B * N);
    free(out); free(bias);
    return out_id;
}

typedef struct { u32 d_input, d_weight, d_bias; } LinearGrads;

static LinearGrads linear_backward(TinyHVM *ctx, u32 grad_id, u32 input_id, u32 weight_id,
                                    u32 B, u32 K, u32 N) {
    // d_input[B,K] = d_out[B,N] @ weight[K,N]^T
    // Need weight transposed: create [N,K] on the fly
    f32 *w = buf_read_f32(ctx, weight_id, K * N);
    f32 *wt = malloc(N * K * sizeof(f32));
    for (u32 k = 0; k < K; k++)
        for (u32 n = 0; n < N; n++)
            wt[n * K + k] = w[k * N + n];
    u32 wt_id = tensor_from_data(ctx, wt, SHAPE(N, K));
    u32 di = mm(ctx, grad_id, wt_id, B, N, K);  // [B,N] @ [N,K] → [B,K]
    free(w); free(wt);

    // d_weight[K,N] = input[B,K]^T @ d_out[B,N]
    f32 *inp = buf_read_f32(ctx, input_id, B * K);
    f32 *inpt = malloc(K * B * sizeof(f32));
    for (u32 b = 0; b < B; b++)
        for (u32 k = 0; k < K; k++)
            inpt[k * B + b] = inp[b * K + k];
    u32 inpt_id = tensor_from_data(ctx, inpt, SHAPE(K, B));
    u32 dw = mm(ctx, inpt_id, grad_id, K, B, N);  // [K,B] @ [B,N] → [K,N]
    free(inp); free(inpt);

    // d_bias[N] = sum(d_out, axis=0)
    f32 *dout = buf_read_f32(ctx, grad_id, B * N);
    f32 *db = calloc(N, sizeof(f32));
    for (u32 b = 0; b < B; b++)
        for (u32 n = 0; n < N; n++)
            db[n] += dout[b * N + n];
    u32 db_id = tensor_from_data(ctx, db, SHAPE(N));
    free(dout); free(db);

    return (LinearGrads){di, dw, db_id};
}

// ============================================================
// Conv2d — im2col (host) + matmul (backend)
// ============================================================

// im2col: [B,C,H,W] → [B*OH*OW, C*KH*KW]
static f32 *im2col(const f32 *input, u32 B, u32 C, u32 H, u32 W,
                    u32 KH, u32 KW, u32 *out_rows, u32 *col_cols) {
    u32 OH = H - KH + 1, OW = W - KW + 1;
    *out_rows = B * OH * OW;
    *col_cols = C * KH * KW;
    f32 *col = malloc((u64)(*out_rows) * (*col_cols) * sizeof(f32));
    for (u32 b = 0; b < B; b++)
        for (u32 oh = 0; oh < OH; oh++)
            for (u32 ow = 0; ow < OW; ow++) {
                u32 row = b * OH * OW + oh * OW + ow;
                for (u32 c = 0; c < C; c++)
                    for (u32 kh = 0; kh < KH; kh++)
                        for (u32 kw = 0; kw < KW; kw++)
                            col[row * (*col_cols) + c*KH*KW + kh*KW + kw] =
                                input[b*C*H*W + c*H*W + (oh+kh)*W + (ow+kw)];
            }
    return col;
}

// col2im: [B*OH*OW, C*KH*KW] → [B,C,H,W] (accumulate)
static f32 *col2im(const f32 *col, u32 B, u32 C, u32 H, u32 W, u32 KH, u32 KW) {
    u32 OH = H - KH + 1, OW = W - KW + 1;
    f32 *img = calloc(B * C * H * W, sizeof(f32));
    for (u32 b = 0; b < B; b++)
        for (u32 oh = 0; oh < OH; oh++)
            for (u32 ow = 0; ow < OW; ow++) {
                u32 row = b * OH * OW + oh * OW + ow;
                for (u32 c = 0; c < C; c++)
                    for (u32 kh = 0; kh < KH; kh++)
                        for (u32 kw = 0; kw < KW; kw++)
                            img[b*C*H*W + c*H*W + (oh+kh)*W + (ow+kw)]
                                += col[row * (C*KH*KW) + c*KH*KW + kh*KW + kw];
            }
    return img;
}

typedef struct {
    u32 out_id;
    u32 col_id;     // saved im2col for backward
    u32 B, IC, H, W, OC, KH, KW, OH, OW;
} Conv2dCache;

// Forward: out[B,OC,OH,OW] = conv(input[B,IC,H,W], weight[OC,IC*KH*KW]) + bias[OC]
// Uses backend->op_mm for the core matmul
static Conv2dCache conv2d_forward(TinyHVM *ctx, u32 input_id,
                                   u32 weight_id, u32 bias_id,
                                   u32 B, u32 IC, u32 H, u32 W,
                                   u32 OC, u32 KH, u32 KW) {
    u32 OH = H - KH + 1, OW = W - KW + 1;
    u32 col_rows, col_cols;

    // im2col: host-side data reorganization
    f32 *input = buf_read_f32(ctx, input_id, B * IC * H * W);
    f32 *col = im2col(input, B, IC, H, W, KH, KW, &col_rows, &col_cols);
    free(input);

    u32 col_id = tensor_from_data(ctx, col, SHAPE(col_rows, col_cols));
    free(col);

    // weight as [OC, col_cols] — transpose to [col_cols, OC] for col @ wt
    f32 *w = buf_read_f32(ctx, weight_id, OC * col_cols);
    f32 *wt = malloc(col_cols * OC * sizeof(f32));
    for (u32 i = 0; i < OC; i++)
        for (u32 j = 0; j < col_cols; j++)
            wt[j * OC + i] = w[i * col_cols + j];
    u32 wt_id = tensor_from_data(ctx, wt, SHAPE(col_cols, OC));
    free(w); free(wt);

    // CORE MATMUL through backend: out[col_rows, OC] = col[col_rows, col_cols] @ wt[col_cols, OC]
    u32 out_flat = mm(ctx, col_id, wt_id, col_rows, col_cols, OC);

    // Add bias
    f32 *out = buf_read_f32(ctx, out_flat, col_rows * OC);
    f32 *bias = buf_read_f32(ctx, bias_id, OC);
    for (u32 i = 0; i < col_rows; i++)
        for (u32 j = 0; j < OC; j++)
            out[i * OC + j] += bias[j];
    free(bias);

    // Reshape [B*OH*OW, OC] → [B, OC, OH, OW] (NCHW)
    f32 *nchw = malloc(B * OC * OH * OW * sizeof(f32));
    for (u32 b = 0; b < B; b++)
        for (u32 oc = 0; oc < OC; oc++)
            for (u32 oh = 0; oh < OH; oh++)
                for (u32 ow = 0; ow < OW; ow++)
                    nchw[b*OC*OH*OW + oc*OH*OW + oh*OW + ow] =
                        out[(b*OH*OW + oh*OW + ow) * OC + oc];
    free(out);

    Conv2dCache cache;
    cache.out_id = tensor_from_data(ctx, nchw, (Shape){.dims={B,OC,OH,OW}, .rank=4});
    cache.col_id = col_id;
    cache.B=B; cache.IC=IC; cache.H=H; cache.W=W;
    cache.OC=OC; cache.KH=KH; cache.KW=KW; cache.OH=OH; cache.OW=OW;
    free(nchw);
    return cache;
}

typedef struct { u32 d_input, d_weight, d_bias; } Conv2dGrads;

static Conv2dGrads conv2d_backward(TinyHVM *ctx, u32 grad_id, Conv2dCache *cache,
                                    u32 weight_id) {
    u32 B = cache->B, IC = cache->IC, OC = cache->OC;
    u32 KH = cache->KH, KW = cache->KW, OH = cache->OH, OW = cache->OW;
    u32 col_cols = IC * KH * KW;
    u32 col_rows = B * OH * OW;

    // Reshape grad from NCHW [B,OC,OH,OW] → [B*OH*OW, OC]
    f32 *gnchw = buf_read_f32(ctx, grad_id, B * OC * OH * OW);
    f32 *dout = malloc(col_rows * OC * sizeof(f32));
    for (u32 b = 0; b < B; b++)
        for (u32 oc = 0; oc < OC; oc++)
            for (u32 oh = 0; oh < OH; oh++)
                for (u32 ow = 0; ow < OW; ow++)
                    dout[(b*OH*OW + oh*OW + ow) * OC + oc] =
                        gnchw[b*OC*OH*OW + oc*OH*OW + oh*OW + ow];
    free(gnchw);
    u32 dout_id = tensor_from_data(ctx, dout, SHAPE(col_rows, OC));

    // d_weight[OC, col_cols] = dout^T[OC, col_rows] @ col[col_rows, col_cols]
    // Transpose dout → [OC, col_rows]
    f32 *dout_t = malloc(OC * col_rows * sizeof(f32));
    for (u32 r = 0; r < col_rows; r++)
        for (u32 c = 0; c < OC; c++)
            dout_t[c * col_rows + r] = dout[r * OC + c];
    u32 dout_t_id = tensor_from_data(ctx, dout_t, SHAPE(OC, col_rows));
    free(dout_t);

    u32 dw_id = mm(ctx, dout_t_id, cache->col_id, OC, col_rows, col_cols);

    // d_col[col_rows, col_cols] = dout[col_rows, OC] @ weight[OC, col_cols]
    u32 dcol_id = mm(ctx, dout_id, weight_id, col_rows, OC, col_cols);

    // col2im: d_col → d_input
    f32 *dcol = buf_read_f32(ctx, dcol_id, col_rows * col_cols);
    f32 *dinput = col2im(dcol, B, IC, cache->H, cache->W, KH, KW);
    u32 di_id = tensor_from_data(ctx, dinput, (Shape){.dims={B,IC,cache->H,cache->W}, .rank=4});
    free(dcol); free(dinput);

    // d_bias[OC] = sum(dout, axis=0)
    f32 *db = calloc(OC, sizeof(f32));
    for (u32 r = 0; r < col_rows; r++)
        for (u32 c = 0; c < OC; c++)
            db[c] += dout[r * OC + c];
    u32 db_id = tensor_from_data(ctx, db, SHAPE(OC));
    free(dout); free(db);

    return (Conv2dGrads){di_id, dw_id, db_id};
}

// ============================================================
// MaxPool2d (2×2 stride 2)
// ============================================================

typedef struct {
    u32 out_id, mask_id;
    u32 B, C, H, W, OH, OW;
} MaxPool2dCache;

static MaxPool2dCache maxpool2d_forward(TinyHVM *ctx, u32 input_id,
                                         u32 B, u32 C, u32 H, u32 W) {
    u32 OH = H / 2, OW = W / 2;
    f32 *x = buf_read_f32(ctx, input_id, B*C*H*W);
    f32 *out = malloc(B*C*OH*OW * sizeof(f32));
    f32 *mask = calloc(B*C*H*W, sizeof(f32));

    for (u32 b = 0; b < B; b++)
        for (u32 c = 0; c < C; c++)
            for (u32 oh = 0; oh < OH; oh++)
                for (u32 ow = 0; ow < OW; ow++) {
                    f32 mv = -1e30f; u32 mi = 0;
                    for (u32 dh = 0; dh < 2; dh++)
                        for (u32 dw = 0; dw < 2; dw++) {
                            u32 idx = b*C*H*W + c*H*W + (oh*2+dh)*W + (ow*2+dw);
                            if (x[idx] > mv) { mv = x[idx]; mi = idx; }
                        }
                    out[b*C*OH*OW + c*OH*OW + oh*OW + ow] = mv;
                    mask[mi] = 1.0f;
                }

    MaxPool2dCache cache;
    cache.out_id  = tensor_from_data(ctx, out, (Shape){.dims={B,C,OH,OW}, .rank=4});
    cache.mask_id = tensor_from_data(ctx, mask, (Shape){.dims={B,C,H,W}, .rank=4});
    cache.B=B; cache.C=C; cache.H=H; cache.W=W; cache.OH=OH; cache.OW=OW;
    free(x); free(out); free(mask);
    return cache;
}

static u32 maxpool2d_backward(TinyHVM *ctx, u32 grad_id, MaxPool2dCache *cache) {
    u32 B=cache->B, C=cache->C, H=cache->H, W=cache->W;
    u32 OH=cache->OH, OW=cache->OW;
    f32 *g = buf_read_f32(ctx, grad_id, B*C*OH*OW);
    f32 *m = buf_read_f32(ctx, cache->mask_id, B*C*H*W);
    f32 *di = calloc(B*C*H*W, sizeof(f32));

    for (u32 b = 0; b < B; b++)
        for (u32 c = 0; c < C; c++)
            for (u32 oh = 0; oh < OH; oh++)
                for (u32 ow = 0; ow < OW; ow++) {
                    f32 gv = g[b*C*OH*OW + c*OH*OW + oh*OW + ow];
                    for (u32 dh = 0; dh < 2; dh++)
                        for (u32 dw = 0; dw < 2; dw++) {
                            u32 idx = b*C*H*W + c*H*W + (oh*2+dh)*W + (ow*2+dw);
                            di[idx] += gv * m[idx];
                        }
                }
    u32 out = tensor_from_data(ctx, di, (Shape){.dims={B,C,H,W}, .rank=4});
    free(g); free(m); free(di);
    return out;
}

// ============================================================
// BatchNorm (channel-wise, NCHW)
// ============================================================

typedef struct {
    u32 out_id;
    u32 B, C, spatial;
    f32 *x_hat, *inv_std;
} BatchNormCache;

static BatchNormCache batchnorm_forward(TinyHVM *ctx, u32 input_id,
                                         u32 gamma_id, u32 beta_id,
                                         u32 rmean_id, u32 rvar_id,
                                         u32 B, u32 C, u32 H, u32 W, int training) {
    u32 sp = H * W, N = B * sp;
    f32 eps = 1e-5f, mom = 0.1f;
    f32 *x = buf_read_f32(ctx, input_id, B*C*sp);
    f32 *gamma = buf_read_f32(ctx, gamma_id, C);
    f32 *beta  = buf_read_f32(ctx, beta_id, C);
    f32 *mean = calloc(C, sizeof(f32));
    f32 *var  = calloc(C, sizeof(f32));

    if (training) {
        for (u32 b=0;b<B;b++) for (u32 c=0;c<C;c++) for (u32 s=0;s<sp;s++)
            mean[c] += x[b*C*sp+c*sp+s];
        for (u32 c=0;c<C;c++) mean[c] /= (f32)N;
        for (u32 b=0;b<B;b++) for (u32 c=0;c<C;c++) for (u32 s=0;s<sp;s++) {
            f32 d = x[b*C*sp+c*sp+s]-mean[c]; var[c] += d*d;
        }
        for (u32 c=0;c<C;c++) var[c] /= (f32)N;
        // Update running stats
        f32 *rm = buf_read_f32(ctx, rmean_id, C);
        f32 *rv = buf_read_f32(ctx, rvar_id, C);
        for (u32 c=0;c<C;c++) { rm[c]=(1-mom)*rm[c]+mom*mean[c]; rv[c]=(1-mom)*rv[c]+mom*var[c]; }
        buf_write_f32(ctx, rmean_id, rm, C);
        buf_write_f32(ctx, rvar_id, rv, C);
        free(rm); free(rv);
    } else {
        free(mean); free(var);
        mean = buf_read_f32(ctx, rmean_id, C);
        var  = buf_read_f32(ctx, rvar_id, C);
    }

    f32 *is = malloc(C * sizeof(f32));
    f32 *xh = malloc(B*C*sp * sizeof(f32));
    f32 *out = malloc(B*C*sp * sizeof(f32));
    for (u32 c=0;c<C;c++) is[c] = 1.0f/sqrtf(var[c]+eps);
    for (u32 b=0;b<B;b++) for (u32 c=0;c<C;c++) for (u32 s=0;s<sp;s++) {
        u32 i = b*C*sp+c*sp+s;
        xh[i] = (x[i]-mean[c])*is[c];
        out[i] = gamma[c]*xh[i]+beta[c];
    }

    BatchNormCache cache;
    cache.out_id = tensor_from_data(ctx, out, (Shape){.dims={B,C,H,W}, .rank=4});
    cache.B=B; cache.C=C; cache.spatial=sp; cache.x_hat=xh; cache.inv_std=is;
    free(x); free(gamma); free(beta); free(mean); free(var); free(out);
    return cache;
}

static void batchnorm_backward(TinyHVM *ctx, u32 grad_id, BatchNormCache *cache,
                                u32 gamma_id, u32 *di, u32 *dg, u32 *db) {
    u32 B=cache->B, C=cache->C, sp=cache->spatial, N=B*sp;
    f32 *dout = buf_read_f32(ctx, grad_id, B*C*sp);
    f32 *gamma = buf_read_f32(ctx, gamma_id, C);
    f32 *d_gamma = calloc(C, sizeof(f32));
    f32 *d_beta  = calloc(C, sizeof(f32));

    for (u32 b=0;b<B;b++) for (u32 c=0;c<C;c++) for (u32 s=0;s<sp;s++) {
        u32 i = b*C*sp+c*sp+s;
        d_gamma[c] += dout[i]*cache->x_hat[i];
        d_beta[c]  += dout[i];
    }

    f32 *dinput = malloc(B*C*sp*sizeof(f32));
    for (u32 c=0;c<C;c++) {
        f32 s1=0, s2=0;
        for (u32 b=0;b<B;b++) for (u32 s=0;s<sp;s++) {
            u32 i=b*C*sp+c*sp+s;
            f32 dg=dout[i]*gamma[c]; s1+=dg; s2+=dg*cache->x_hat[i];
        }
        for (u32 b=0;b<B;b++) for (u32 s=0;s<sp;s++) {
            u32 i=b*C*sp+c*sp+s;
            dinput[i] = cache->inv_std[c]/(f32)N*((f32)N*dout[i]*gamma[c]-s1-cache->x_hat[i]*s2);
        }
    }

    u32 sqsp = (u32)sqrtf((f32)sp);
    *di = tensor_from_data(ctx, dinput, (Shape){.dims={B,C,sqsp,sqsp}, .rank=4});
    *dg = tensor_from_data(ctx, d_gamma, SHAPE(C));
    *db = tensor_from_data(ctx, d_beta, SHAPE(C));
    free(dout); free(gamma); free(dinput); free(d_gamma); free(d_beta);
    free(cache->x_hat); free(cache->inv_std);
    cache->x_hat = NULL; cache->inv_std = NULL;
}

// ============================================================
// Adam optimizer
// ============================================================

typedef struct {
    u32 n_params;
    u32 *param_ids;
    f32 **m, **v;
    u32 *sizes;
    f32 lr, beta1, beta2, eps;
    u32 t;
} Adam;

static Adam adam_init(f32 lr, u32 n_params) {
    Adam o;
    o.n_params=n_params; o.lr=lr; o.beta1=0.9f; o.beta2=0.999f; o.eps=1e-8f; o.t=0;
    o.param_ids=calloc(n_params,sizeof(u32)); o.m=calloc(n_params,sizeof(f32*));
    o.v=calloc(n_params,sizeof(f32*)); o.sizes=calloc(n_params,sizeof(u32));
    return o;
}

static void adam_add_param(Adam *o, u32 idx, u32 pid, u32 sz) {
    o->param_ids[idx]=pid; o->sizes[idx]=sz;
    o->m[idx]=calloc(sz,sizeof(f32)); o->v[idx]=calloc(sz,sizeof(f32));
}

static void adam_step(TinyHVM *ctx, Adam *o, u32 *grad_ids) {
    o->t++;
    f32 bc1=1-powf(o->beta1,(f32)o->t), bc2=1-powf(o->beta2,(f32)o->t);
    for (u32 i=0; i<o->n_params; i++) {
        u32 n=o->sizes[i];
        f32 *p=buf_read_f32(ctx, o->param_ids[i], n);
        f32 *g=buf_read_f32(ctx, grad_ids[i], n);
        for (u32 j=0; j<n; j++) {
            o->m[i][j] = o->beta1*o->m[i][j]+(1-o->beta1)*g[j];
            o->v[i][j] = o->beta2*o->v[i][j]+(1-o->beta2)*g[j]*g[j];
            p[j] -= o->lr * (o->m[i][j]/bc1) / (sqrtf(o->v[i][j]/bc2)+o->eps);
        }
        buf_write_f32(ctx, o->param_ids[i], p, n);
        free(p); free(g);
    }
}

static void adam_free(Adam *o) {
    for (u32 i=0; i<o->n_params; i++) { free(o->m[i]); free(o->v[i]); }
    free(o->param_ids); free(o->m); free(o->v); free(o->sizes);
}
