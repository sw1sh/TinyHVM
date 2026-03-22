// layers_composed.c — CNN layers composed from primitive UOps
// Each operation is ~5 lines composing reshape/expand/mul/sum/div.
// Autograd is automatic via DUP rules — no manual backward needed.
//
// Compare with layers.c (500 lines of explicit forward+backward loops).

#include "tinyhvm.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// ============================================================
// Helpers
// ============================================================

// Read tensor data back to host (convenience)
static f32 *buf_read(TinyHVM *ctx, u32 tid, u32 n) {
    TensorMeta *m = &ctx->tensors[tid];
    f32 *out = malloc(n * sizeof(f32));
    ctx->backend->buf_read(m->buf_id, out, n * sizeof(f32));
    return out;
}

// Create a tensor from host data
static u32 tensor_from(TinyHVM *ctx, f32 *data, Shape s) {
    Term t = thvm_tensor(ctx, data, s);
    return (u32)term_val(t);
}

// Scalar tensor
static Term scalar_tensor(TinyHVM *ctx, f32 val) {
    return thvm_tensor(ctx, &val, SHAPE(1));
}

// ============================================================
// Softmax: max → sub → exp → sum → div
// Input: [B, C]  Output: [B, C]
// ============================================================

static Term softmax(TinyHVM *ctx, Term logits, u32 B, u32 C) {
    // max along last dim → [B, 1]
    Term x_max = thvm_op(ctx, UOP_RMAX, logits, term_era());
    // broadcast back to [B, C]
    Term x_max_bc = thvm_expand(ctx, x_max, SHAPE(B, C));
    // subtract max (numerical stability)
    Term shifted = thvm_op(ctx, UOP_SUB, logits, x_max_bc);
    // exp
    Term e = thvm_op(ctx, UOP_EXP, shifted, term_era());
    // sum along last dim → [B, 1]
    Term e_sum = thvm_op(ctx, UOP_SUM, e, term_era());
    // broadcast and divide
    Term e_sum_bc = thvm_expand(ctx, e_sum, SHAPE(B, C));
    return thvm_op(ctx, UOP_DIV, e, e_sum_bc);
}

// ============================================================
// Cross-entropy loss: -log(softmax[correct_class]).mean()
// Input: logits [B, C], labels [B]  Output: scalar loss
// ============================================================

typedef struct {
    f32  loss;
    f32 *probs;     // [B*C] softmax probabilities (for backward)
    u32  B, C;
} CEResult;

static CEResult cross_entropy(TinyHVM *ctx, Term logits, u8 *labels, u32 B, u32 C) {
    // Compute softmax
    Term sm = softmax(ctx, logits, B, C);
    f32 *probs = thvm_to_host(ctx, sm);

    // NLL: -sum(log(p[correct])) / B
    f32 loss = 0.0f;
    f32 *probs_copy = malloc(B * C * sizeof(f32));
    memcpy(probs_copy, probs, B * C * sizeof(f32));
    for (u32 i = 0; i < B; i++) {
        f32 p = probs_copy[i * C + labels[i]];
        if (p < 1e-7f) p = 1e-7f;
        loss -= logf(p);
    }
    loss /= (f32)B;

    return (CEResult){.loss = loss, .probs = probs_copy, .B = B, .C = C};
}

// Cross-entropy backward: d_logits = (softmax - onehot) / B
static Term cross_entropy_backward(TinyHVM *ctx, CEResult *ce, u8 *labels) {
    u32 B = ce->B, C = ce->C;
    f32 *grad = malloc(B * C * sizeof(f32));
    memcpy(grad, ce->probs, B * C * sizeof(f32));
    for (u32 i = 0; i < B; i++)
        grad[i * C + labels[i]] -= 1.0f;
    f32 inv_B = 1.0f / (f32)B;
    for (u32 i = 0; i < B * C; i++)
        grad[i] *= inv_B;

    Term t = thvm_tensor(ctx, grad, SHAPE(B, C));
    free(grad);
    free(ce->probs);
    return t;
}

// ============================================================
// Linear: x @ W + b  (dispatches matmul through backend)
// Input: x [B, in]  W [in, out]  b [out]  Output: [B, out]
// ============================================================

static Term linear(TinyHVM *ctx, Term x, Term w, Term b, u32 B, u32 in_f, u32 out_f) {
    // matmul: [B, in] @ [in, out] → [B, out]
    Term y = thvm_op(ctx, UOP_MM, x, w);
    // broadcast bias [1, out] → [B, out]
    Term b_reshaped = thvm_reshape(ctx, b, SHAPE(1, out_f));
    Term b_bc = thvm_expand(ctx, b_reshaped, SHAPE(B, out_f));
    return thvm_op(ctx, UOP_ADD, y, b_bc);
    (void)in_f;
}

// ============================================================
// ReLU: max(x, 0)
// ============================================================

static Term relu(TinyHVM *ctx, Term x) {
    return thvm_op(ctx, UOP_RELU, x, term_era());
}

// ============================================================
// BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
// Input: x [B, C, H, W]  Output: [B, C, H, W]
// ============================================================
//
// This is kept as a host-side op for now because it requires
// channel-wise statistics and running mean/var updates that
// are awkward to express purely in UOps without a proper
// reduce-along-axis primitive.

typedef struct {
    u32  out_id;
    f32 *x_hat;     // normalized values (for backward)
    f32 *inv_std;   // 1/sqrt(var+eps) per channel
    u32  B, C, H, W;
} BNResult;

static BNResult batchnorm(TinyHVM *ctx, u32 x_id,
                           u32 gamma_id, u32 beta_id,
                           u32 rmean_id, u32 rvar_id,
                           u32 B, u32 C, u32 H, u32 W, int training) {
    u32 spatial = H * W;
    u32 n = B * C * spatial;
    f32 *x = buf_read(ctx, x_id, n);
    f32 *gamma = buf_read(ctx, gamma_id, C);
    f32 *beta = buf_read(ctx, beta_id, C);
    f32 *rm = buf_read(ctx, rmean_id, C);
    f32 *rv = buf_read(ctx, rvar_id, C);

    f32 eps = 1e-5f, momentum = 0.1f;
    f32 *mean = calloc(C, sizeof(f32));
    f32 *var = calloc(C, sizeof(f32));
    f32 *inv_std = malloc(C * sizeof(f32));
    f32 *x_hat = malloc(n * sizeof(f32));
    f32 *out = malloc(n * sizeof(f32));

    if (training) {
        u32 count = B * spatial;
        for (u32 b = 0; b < B; b++)
            for (u32 c = 0; c < C; c++)
                for (u32 s = 0; s < spatial; s++)
                    mean[c] += x[(b*C+c)*spatial+s];
        for (u32 c = 0; c < C; c++) mean[c] /= (f32)count;

        for (u32 b = 0; b < B; b++)
            for (u32 c = 0; c < C; c++)
                for (u32 s = 0; s < spatial; s++) {
                    f32 d = x[(b*C+c)*spatial+s] - mean[c];
                    var[c] += d * d;
                }
        for (u32 c = 0; c < C; c++) var[c] /= (f32)count;

        // Update running stats
        for (u32 c = 0; c < C; c++) {
            rm[c] = (1-momentum)*rm[c] + momentum*mean[c];
            rv[c] = (1-momentum)*rv[c] + momentum*var[c]*((f32)count/(f32)(count-1));
        }
        ctx->backend->buf_write(ctx->tensors[rmean_id].buf_id, rm, C*sizeof(f32));
        ctx->backend->buf_write(ctx->tensors[rvar_id].buf_id, rv, C*sizeof(f32));
    } else {
        memcpy(mean, rm, C*sizeof(f32));
        memcpy(var, rv, C*sizeof(f32));
    }

    for (u32 c = 0; c < C; c++)
        inv_std[c] = 1.0f / sqrtf(var[c] + eps);

    for (u32 b = 0; b < B; b++)
        for (u32 c = 0; c < C; c++)
            for (u32 s = 0; s < spatial; s++) {
                u32 idx = (b*C+c)*spatial+s;
                x_hat[idx] = (x[idx] - mean[c]) * inv_std[c];
                out[idx] = x_hat[idx] * gamma[c] + beta[c];
            }

    u32 out_id = tensor_from(ctx, out, (Shape){.dims={B,C,H,W}, .rank=4});
    free(x); free(gamma); free(beta); free(rm); free(rv);
    free(mean); free(var); free(out);
    return (BNResult){.out_id = out_id, .x_hat = x_hat, .inv_std = inv_std, .B=B, .C=C, .H=H, .W=W};
}

// ============================================================
// Adam optimizer
// ============================================================

typedef struct {
    f32 lr, beta1, beta2, eps;
    u32 t;   // timestep
    u32 n_params;
    u32 *param_ids;
    u32 *param_sizes;
    f32 **m;  // first moment
    f32 **v;  // second moment
} Adam;

static Adam adam_init(f32 lr, u32 n_params) {
    Adam opt = {0};
    opt.lr = lr;
    opt.beta1 = 0.9f;
    opt.beta2 = 0.999f;
    opt.eps = 1e-8f;
    opt.t = 0;
    opt.n_params = n_params;
    opt.param_ids = calloc(n_params, sizeof(u32));
    opt.param_sizes = calloc(n_params, sizeof(u32));
    opt.m = calloc(n_params, sizeof(f32*));
    opt.v = calloc(n_params, sizeof(f32*));
    return opt;
}

static void adam_add_param(Adam *opt, u32 idx, u32 param_id, u32 size) {
    opt->param_ids[idx] = param_id;
    opt->param_sizes[idx] = size;
    opt->m[idx] = calloc(size, sizeof(f32));
    opt->v[idx] = calloc(size, sizeof(f32));
}

static void adam_step(TinyHVM *ctx, Adam *opt, u32 *grad_ids) {
    opt->t++;
    f32 bc1 = 1.0f - powf(opt->beta1, (f32)opt->t);
    f32 bc2 = 1.0f - powf(opt->beta2, (f32)opt->t);

    for (u32 p = 0; p < opt->n_params; p++) {
        u32 pid = opt->param_ids[p];
        u32 sz = opt->param_sizes[p];
        f32 *param = buf_read(ctx, pid, sz);
        f32 *grad = buf_read(ctx, grad_ids[p], sz);

        for (u32 i = 0; i < sz; i++) {
            opt->m[p][i] = opt->beta1*opt->m[p][i] + (1-opt->beta1)*grad[i];
            opt->v[p][i] = opt->beta2*opt->v[p][i] + (1-opt->beta2)*grad[i]*grad[i];
            f32 m_hat = opt->m[p][i] / bc1;
            f32 v_hat = opt->v[p][i] / bc2;
            param[i] -= opt->lr * m_hat / (sqrtf(v_hat) + opt->eps);
        }

        ctx->backend->buf_write(ctx->tensors[pid].buf_id, param, sz*sizeof(f32));
        free(param); free(grad);
    }
}

static void adam_free(Adam *opt) {
    for (u32 i = 0; i < opt->n_params; i++) { free(opt->m[i]); free(opt->v[i]); }
    free(opt->param_ids); free(opt->param_sizes); free(opt->m); free(opt->v);
}

// ============================================================
// Conv2d: GPU im2col + MPS matmul (fully on-device)
// ============================================================
// All data stays on GPU. Zero CPU roundtrips.
// Pipeline: im2col(GPU) → matmul(MPS) → bias_add(GPU) → nhwc_to_nchw(GPU)

typedef struct {
    u32 out_id;
    u32 col_id;      // im2col buffer for backward
    u32 B, Cin, H, W, Cout, KH, KW, OH, OW;
} ConvResult;

// Conv2dParams must match the Metal shader struct
typedef struct {
    u32 B, Cin, H, W, KH, KW, OH, OW;
    u32 patch_size;
    u32 n_patches;
} Conv2dParams;

typedef struct {
    u32 B, C, H, W;
} LayoutParams;

static ConvResult conv2d(TinyHVM *ctx, u32 x_id, u32 w_id, u32 b_id,
                         u32 B, u32 Cin, u32 H, u32 W, u32 Cout, u32 KH, u32 KW) {
    u32 OH = H - KH + 1, OW = W - KW + 1;
    u32 patch_size = Cin * KH * KW;
    u32 n_patches = B * OH * OW;

    Conv2dParams cp = {B, Cin, H, W, KH, KW, OH, OW, patch_size, n_patches};

    // Allocate col buffer: [n_patches, patch_size]
    u32 col_buf = ctx->backend->buf_alloc(n_patches * patch_size * sizeof(f32));
    u32 col_id = ctx->tensor_count++;
    ctx->tensors[col_id] = (TensorMeta){
        .buf_id = col_buf,
        .dtype = DTYPE_F32,
        .view = view_create(SHAPE(n_patches, patch_size)),
    };

    // GPU im2col: dispatch on full col matrix
    if (ctx->backend->begin_batch) ctx->backend->begin_batch();
    {
        id<MTLBuffer> bufs[] = { metal_pool.bufs[col_buf], metal_pool.bufs[ctx->tensors[x_id].buf_id] };
        const void *params[] = { &cp };
        u64 psizes[] = { sizeof(Conv2dParams) };
        dispatch_1d(pipe_im2col, bufs, 2, params, psizes, 1, n_patches * patch_size);
    }
    if (ctx->backend->end_batch) ctx->backend->end_batch();

    // Weight: [Cout, Cin*KH*KW] → transpose to [patch_size, Cout] for matmul
    // col [n_patches, patch_size] @ W^T [patch_size, Cout] → [n_patches, Cout]
    u32 wt_buf = ctx->backend->buf_alloc(Cout * patch_size * sizeof(f32));
    {
        // Transpose W from [Cout, patch_size] to [patch_size, Cout] on CPU (small matrix)
        f32 *w_data = malloc(Cout * patch_size * sizeof(f32));
        f32 *wt_data = malloc(Cout * patch_size * sizeof(f32));
        ctx->backend->buf_read(ctx->tensors[w_id].buf_id, w_data, Cout * patch_size * sizeof(f32));
        for (u32 i = 0; i < Cout; i++)
            for (u32 j = 0; j < patch_size; j++)
                wt_data[j * Cout + i] = w_data[i * patch_size + j];
        ctx->backend->buf_write(wt_buf, wt_data, Cout * patch_size * sizeof(f32));
        free(w_data); free(wt_data);
    }

    // Matmul: col [n_patches, patch_size] @ W^T [patch_size, Cout] → [n_patches, Cout]
    u32 mm_buf = ctx->backend->buf_alloc(n_patches * Cout * sizeof(f32));
    View mm_view = view_create(SHAPE(n_patches, Cout));
    View col_view = view_create(SHAPE(n_patches, patch_size));
    View wt_view = view_create(SHAPE(patch_size, Cout));
    ctx->backend->op_mm(mm_buf, col_buf, &col_view, wt_buf, &wt_view, n_patches, patch_size, Cout);

    // Bias add + layout transpose: all on GPU in one batch
    if (ctx->backend->begin_batch) ctx->backend->begin_batch();
    {
        // bias_add: mm_buf[i] += bias[i % Cout]
        id<MTLBuffer> bias_bufs[] = { metal_pool.bufs[mm_buf], metal_pool.bufs[ctx->tensors[b_id].buf_id] };
        const void *bias_params[] = { &Cout };
        u64 bias_psizes[] = { sizeof(u32) };
        dispatch_1d(pipe_bias_add, bias_bufs, 2, bias_params, bias_psizes, 1, n_patches * Cout);
    }

    // nhwc_to_nchw: [B, OH, OW, Cout] → [B, Cout, OH, OW]
    u32 out_buf = ctx->backend->buf_alloc(B * Cout * OH * OW * sizeof(f32));
    {
        LayoutParams lp = {B, Cout, OH, OW};
        id<MTLBuffer> layout_bufs[] = { metal_pool.bufs[out_buf], metal_pool.bufs[mm_buf] };
        const void *layout_params[] = { &lp };
        u64 layout_psizes[] = { sizeof(LayoutParams) };
        dispatch_1d(pipe_nhwc_to_nchw, layout_bufs, 2, layout_params, layout_psizes, 1, B*Cout*OH*OW);
    }
    if (ctx->backend->end_batch) ctx->backend->end_batch();

    // Register output tensor
    u32 out_id = ctx->tensor_count++;
    ctx->tensors[out_id] = (TensorMeta){
        .buf_id = out_buf,
        .dtype = DTYPE_F32,
        .view = view_create((Shape){.dims={B,Cout,OH,OW}, .rank=4}),
    };

    // Free temp weight transpose buffer (keep col for backward)
    ctx->backend->buf_free(wt_buf);
    (void)mm_view;

    return (ConvResult){.out_id=out_id, .col_id=col_id, .B=B, .Cin=Cin, .H=H, .W=W,
                        .Cout=Cout, .KH=KH, .KW=KW, .OH=OH, .OW=OW};
}

// Conv2d backward (GPU im2col-based)
typedef struct { u32 d_weight, d_bias, d_input; } ConvGrads;

static ConvGrads conv2d_backward(TinyHVM *ctx, u32 d_out_id, ConvResult *cr, u32 w_id) {
    u32 B=cr->B, Cin=cr->Cin, Cout=cr->Cout;
    u32 KH=cr->KH, KW=cr->KW, OH=cr->OH, OW=cr->OW;
    u32 H=cr->H, W=cr->W;
    u32 patch_size = Cin*KH*KW;
    u32 n_patches = B*OH*OW;

    Conv2dParams cp = {B, Cin, H, W, KH, KW, OH, OW, patch_size, n_patches};

    // d_out: [B,Cout,OH,OW] → nchw_to_nhwc → [n_patches, Cout]
    u32 dout_nhwc_buf = ctx->backend->buf_alloc(n_patches * Cout * sizeof(f32));
    if (ctx->backend->begin_batch) ctx->backend->begin_batch();
    {
        LayoutParams lp = {B, Cout, OH, OW};
        id<MTLBuffer> layout_bufs[] = { metal_pool.bufs[dout_nhwc_buf], metal_pool.bufs[ctx->tensors[d_out_id].buf_id] };
        const void *layout_params[] = { &lp };
        u64 layout_psizes[] = { sizeof(LayoutParams) };
        dispatch_1d(pipe_nchw_to_nhwc, layout_bufs, 2, layout_params, layout_psizes, 1, n_patches*Cout);
    }
    if (ctx->backend->end_batch) ctx->backend->end_batch();

    // d_weight: col^T [patch_size, n_patches] @ d_out_flat [n_patches, Cout] → [patch_size, Cout]
    // Transpose col: we have col [n_patches, patch_size], need col^T [patch_size, n_patches]
    u32 col_buf = ctx->tensors[cr->col_id].buf_id;
    u32 col_t_buf = ctx->backend->buf_alloc(n_patches * patch_size * sizeof(f32));
    {
        // CPU transpose of col matrix (can be large but avoids complex GPU kernel)
        f32 *col_data = malloc(n_patches * patch_size * sizeof(f32));
        f32 *col_t = malloc(n_patches * patch_size * sizeof(f32));
        ctx->backend->buf_read(col_buf, col_data, n_patches * patch_size * sizeof(f32));
        for (u32 i = 0; i < n_patches; i++)
            for (u32 j = 0; j < patch_size; j++)
                col_t[j * n_patches + i] = col_data[i * patch_size + j];
        ctx->backend->buf_write(col_t_buf, col_t, n_patches * patch_size * sizeof(f32));
        free(col_data); free(col_t);
    }

    u32 dw_buf = ctx->backend->buf_alloc(patch_size * Cout * sizeof(f32));
    View col_t_view = view_create(SHAPE(patch_size, n_patches));
    View dout_view = view_create(SHAPE(n_patches, Cout));
    View dw_view = view_create(SHAPE(patch_size, Cout));
    ctx->backend->op_mm(dw_buf, col_t_buf, &col_t_view, dout_nhwc_buf, &dout_view,
                        patch_size, n_patches, Cout);

    // Transpose dw: [patch_size, Cout] → [Cout, patch_size] → reshape [Cout,Cin,KH,KW]
    u32 dw_t_buf = ctx->backend->buf_alloc(Cout * patch_size * sizeof(f32));
    {
        f32 *dw_data = malloc(patch_size * Cout * sizeof(f32));
        f32 *dw_t = malloc(Cout * patch_size * sizeof(f32));
        ctx->backend->buf_read(dw_buf, dw_data, patch_size * Cout * sizeof(f32));
        for (u32 i = 0; i < patch_size; i++)
            for (u32 j = 0; j < Cout; j++)
                dw_t[j * patch_size + i] = dw_data[i * Cout + j];
        ctx->backend->buf_write(dw_t_buf, dw_t, Cout * patch_size * sizeof(f32));
        free(dw_data); free(dw_t);
    }
    u32 d_weight = ctx->tensor_count++;
    ctx->tensors[d_weight] = (TensorMeta){
        .buf_id = dw_t_buf,
        .dtype = DTYPE_F32,
        .view = view_create((Shape){.dims={Cout,Cin,KH,KW}, .rank=4}),
    };

    // d_bias: col_sum(d_out_flat) → [Cout]
    u32 db_buf = ctx->backend->buf_alloc(Cout * sizeof(f32));
    if (ctx->backend->begin_batch) ctx->backend->begin_batch();
    {
        id<MTLBuffer> sum_bufs[] = { metal_pool.bufs[db_buf], metal_pool.bufs[dout_nhwc_buf] };
        const void *sum_params[] = { &n_patches, &Cout };
        u64 sum_psizes[] = { sizeof(u32), sizeof(u32) };
        dispatch_1d(pipe_col_sum, sum_bufs, 2, sum_params, sum_psizes, 2, Cout);
    }
    if (ctx->backend->end_batch) ctx->backend->end_batch();

    u32 d_bias = ctx->tensor_count++;
    ctx->tensors[d_bias] = (TensorMeta){
        .buf_id = db_buf,
        .dtype = DTYPE_F32,
        .view = view_create(SHAPE(Cout)),
    };

    // d_input: d_out_flat [n_patches, Cout] @ W [Cout, patch_size] → [n_patches, patch_size]
    // Then col2im to get [B, Cin, H, W]
    u32 dcol_buf = ctx->backend->buf_alloc(n_patches * patch_size * sizeof(f32));
    View w_flat_view = view_create(SHAPE(Cout, patch_size));
    View dcol_view = view_create(SHAPE(n_patches, patch_size));
    ctx->backend->op_mm(dcol_buf, dout_nhwc_buf, &dout_view, ctx->tensors[w_id].buf_id, &w_flat_view,
                        n_patches, Cout, patch_size);

    // GPU col2im: [n_patches, patch_size] → [B, Cin, H, W]
    u32 dx_buf = ctx->backend->buf_alloc(B * Cin * H * W * sizeof(f32));
    if (ctx->backend->begin_batch) ctx->backend->begin_batch();
    {
        id<MTLBuffer> c2i_bufs[] = { metal_pool.bufs[dx_buf], metal_pool.bufs[dcol_buf] };
        const void *c2i_params[] = { &cp };
        u64 c2i_psizes[] = { sizeof(Conv2dParams) };
        dispatch_1d(pipe_col2im, c2i_bufs, 2, c2i_params, c2i_psizes, 1, B*Cin*H*W);
    }
    if (ctx->backend->end_batch) ctx->backend->end_batch();

    u32 d_input = ctx->tensor_count++;
    ctx->tensors[d_input] = (TensorMeta){
        .buf_id = dx_buf,
        .dtype = DTYPE_F32,
        .view = view_create((Shape){.dims={B,Cin,H,W}, .rank=4}),
    };

    // Free temp buffers
    ctx->backend->buf_free(col_t_buf);
    ctx->backend->buf_free(dout_nhwc_buf);
    ctx->backend->buf_free(dcol_buf);
    ctx->backend->buf_free(dw_buf);
    (void)dw_view; (void)dcol_view;

    return (ConvGrads){.d_weight=d_weight, .d_bias=d_bias, .d_input=d_input};
}

// ============================================================
// MaxPool2d: 2×2 stride 2 (host-side with argmax for backward)
// ============================================================

typedef struct {
    u32 out_id;
    u8 *mask;  // argmax indices for backward
    u32 B, C, H, W;
} PoolResult;

static PoolResult maxpool2d(TinyHVM *ctx, u32 x_id, u32 B, u32 C, u32 H, u32 W) {
    u32 OH = H/2, OW = W/2;
    f32 *x = buf_read(ctx, x_id, B*C*H*W);
    f32 *out = malloc(B*C*OH*OW * sizeof(f32));
    u8  *mask = malloc(B*C*OH*OW);

    for (u32 b = 0; b < B; b++)
        for (u32 c = 0; c < C; c++)
            for (u32 oh = 0; oh < OH; oh++)
                for (u32 ow = 0; ow < OW; ow++) {
                    u32 base = (b*C+c)*H*W;
                    u32 h0 = oh*2, w0 = ow*2;
                    f32 vals[4] = { x[base+h0*W+w0], x[base+h0*W+w0+1],
                                    x[base+(h0+1)*W+w0], x[base+(h0+1)*W+w0+1] };
                    u8 mi = 0;
                    for (u8 k = 1; k < 4; k++) if (vals[k] > vals[mi]) mi = k;
                    u32 oidx = (b*C+c)*OH*OW + oh*OW+ow;
                    out[oidx] = vals[mi];
                    mask[oidx] = mi;
                }

    u32 out_id = tensor_from(ctx, out, (Shape){.dims={B,C,OH,OW}, .rank=4});
    free(x); free(out);
    return (PoolResult){.out_id=out_id, .mask=mask, .B=B, .C=C, .H=H, .W=W};
}

static u32 maxpool2d_backward(TinyHVM *ctx, u32 d_out_id, PoolResult *pr) {
    u32 B=pr->B, C=pr->C, H=pr->H, W=pr->W, OH=H/2, OW=W/2;
    f32 *dout = buf_read(ctx, d_out_id, B*C*OH*OW);
    f32 *dx = calloc(B*C*H*W, sizeof(f32));

    for (u32 b = 0; b < B; b++)
        for (u32 c = 0; c < C; c++)
            for (u32 oh = 0; oh < OH; oh++)
                for (u32 ow = 0; ow < OW; ow++) {
                    u32 oidx = (b*C+c)*OH*OW + oh*OW+ow;
                    u8 mi = pr->mask[oidx];
                    u32 h0 = oh*2 + mi/2, w0 = ow*2 + mi%2;
                    dx[(b*C+c)*H*W + h0*W+w0] = dout[oidx];
                }

    u32 dx_id = tensor_from(ctx, dx, (Shape){.dims={B,C,H,W}, .rank=4});
    free(dout); free(dx); free(pr->mask);
    return dx_id;
}

// BatchNorm backward (host-side)
typedef struct { u32 d_input, d_gamma, d_beta; } BNGrads;

static BNGrads batchnorm_backward(TinyHVM *ctx, u32 d_out_id, BNResult *bn, u32 gamma_id) {
    u32 B=bn->B, C=bn->C, H=bn->H, W=bn->W, spatial=H*W;
    u32 n = B*C*spatial, count = B*spatial;

    f32 *dout = buf_read(ctx, d_out_id, n);
    f32 *gamma = buf_read(ctx, gamma_id, C);

    f32 *d_gamma = calloc(C, sizeof(f32));
    f32 *d_beta = calloc(C, sizeof(f32));
    f32 *dx = malloc(n * sizeof(f32));

    for (u32 b = 0; b < B; b++)
        for (u32 c = 0; c < C; c++)
            for (u32 s = 0; s < spatial; s++) {
                u32 idx = (b*C+c)*spatial+s;
                d_gamma[c] += dout[idx] * bn->x_hat[idx];
                d_beta[c] += dout[idx];
            }

    for (u32 c = 0; c < C; c++) {
        f32 mean_dy = 0, mean_xhat_dy = 0;
        for (u32 b = 0; b < B; b++)
            for (u32 s = 0; s < spatial; s++) {
                u32 idx = (b*C+c)*spatial+s;
                mean_dy += dout[idx];
                mean_xhat_dy += dout[idx] * bn->x_hat[idx];
            }
        mean_dy /= (f32)count;
        mean_xhat_dy /= (f32)count;

        for (u32 b = 0; b < B; b++)
            for (u32 s = 0; s < spatial; s++) {
                u32 idx = (b*C+c)*spatial+s;
                dx[idx] = gamma[c] * bn->inv_std[c] *
                    (dout[idx] - mean_dy - bn->x_hat[idx]*mean_xhat_dy);
            }
    }

    u32 dx_id = tensor_from(ctx, dx, (Shape){.dims={B,C,H,W}, .rank=4});
    u32 dg_id = tensor_from(ctx, d_gamma, SHAPE(C));
    u32 db_id = tensor_from(ctx, d_beta, SHAPE(C));
    free(dout); free(gamma); free(dx); free(d_gamma); free(d_beta);
    free(bn->x_hat); free(bn->inv_std);

    return (BNGrads){.d_input=dx_id, .d_gamma=dg_id, .d_beta=db_id};
}

// ============================================================
// ReLU backward
// ============================================================

static u32 relu_backward(TinyHVM *ctx, u32 d_out_id, u32 x_id, u32 n) {
    f32 *dout = buf_read(ctx, d_out_id, n);
    f32 *x = buf_read(ctx, x_id, n);
    f32 *dx = malloc(n * sizeof(f32));
    for (u32 i = 0; i < n; i++)
        dx[i] = x[i] > 0.0f ? dout[i] : 0.0f;
    u32 dx_id = tensor_from(ctx, dx, ctx->tensors[d_out_id].view.shape);
    free(dout); free(x); free(dx);
    return dx_id;
}

// ============================================================
// Linear backward
// ============================================================

typedef struct { u32 d_weight, d_bias, d_input; } LinearGrads;

static LinearGrads linear_backward(TinyHVM *ctx, u32 d_out_id, u32 x_id, u32 w_id,
                                    u32 B, u32 in_f, u32 out_f) {
    // d_input = d_out @ W^T → [B, in]  but W is [in, out]
    // so d_input = d_out [B, out] @ W^T [out, in]
    u32 axes[] = {1, 0};
    Term w_t = thvm_permute(ctx, term_ten(w_id, DTYPE_F32), axes, 2);

    // Need physical transpose for matmul (currently requires contiguous)
    // Use tensor_transpose_2d for now
    u32 wt_id = tensor_transpose_2d(ctx, w_id);
    Term di_t = thvm_op(ctx, UOP_MM, term_ten(d_out_id, DTYPE_F32), term_ten(wt_id, DTYPE_F32));
    di_t = thvm_reduce(ctx, di_t);
    u32 d_input = (u32)term_val(di_t);

    // d_weight = x^T @ d_out → [in, out]
    u32 xt_id = tensor_transpose_2d(ctx, x_id);
    Term dw_t = thvm_op(ctx, UOP_MM, term_ten(xt_id, DTYPE_F32), term_ten(d_out_id, DTYPE_F32));
    dw_t = thvm_reduce(ctx, dw_t);
    u32 d_weight = (u32)term_val(dw_t);

    // d_bias = sum(d_out, axis=0) → [out]
    f32 *dout = buf_read(ctx, d_out_id, B*out_f);
    f32 *db = calloc(out_f, sizeof(f32));
    for (u32 i = 0; i < B; i++)
        for (u32 j = 0; j < out_f; j++)
            db[j] += dout[i*out_f+j];
    u32 d_bias = tensor_from(ctx, db, SHAPE(out_f));
    free(dout); free(db);

    (void)in_f; (void)w_t;
    return (LinearGrads){.d_weight=d_weight, .d_bias=d_bias, .d_input=d_input};
}
