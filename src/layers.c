// layers.c — CNN layers using thvm UOp composition
// Conv2d: thvm_conv2d (pad → pool → reshape → expand → permute → mul → sum)
// MaxPool2d: thvm_maxpool2d (pool → rmax)
// All forwarded through the thvm API. Zero Metal-specific code.

#include "tinyhvm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================
// Helpers
// ============================================================

static f32 *buf_read(TinyHVM *ctx, u32 tid, u32 n) {
    TensorMeta *m = &ctx->tensors[tid];
    f32 *out = malloc(n * sizeof(f32));
    ctx->backend->buf_read(m->buf_id, out, n * sizeof(f32));
    return out;
}

static u32 tensor_from(TinyHVM *ctx, f32 *data, Shape s) {
    Term t = thvm_tensor(ctx, data, s);
    return (u32)term_val(t);
}

// ============================================================
// Softmax: max → sub → exp → sum → div
// ============================================================

static Term softmax(TinyHVM *ctx, Term logits, u32 B, u32 C) {
    Term x_max = thvm_op(ctx, UOP_RMAX, logits, term_era());
    Term x_max_bc = thvm_expand(ctx, x_max, SHAPE(B, C));
    Term shifted = thvm_op(ctx, UOP_SUB, logits, x_max_bc);
    Term e = thvm_op(ctx, UOP_EXP, shifted, term_era());
    Term e_sum = thvm_op(ctx, UOP_SUM, e, term_era());
    Term e_sum_bc = thvm_expand(ctx, e_sum, SHAPE(B, C));
    return thvm_op(ctx, UOP_DIV, e, e_sum_bc);
}

// ============================================================
// Cross-entropy loss
// ============================================================

typedef struct {
    f32  loss;
    f32 *probs;
    u32  B, C;
} CEResult;

static CEResult cross_entropy(TinyHVM *ctx, Term logits, u8 *labels, u32 B, u32 C) {
    Term sm = softmax(ctx, logits, B, C);
    f32 *probs = thvm_to_host(ctx, sm);

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
// Conv2d: thvm_conv2d UOp composition
// x:[BS,Cin,H,W] w:[Cout,Cin,KH,KW] b:[Cout] → [BS,Cout,OH,OW]
// ============================================================

typedef struct {
    u32 out_id;
    u32 x_id;      // input (for backward)
    u32 col_id;    // pooled/windowed intermediate (for backward via im2col)
    u32 B, Cin, H, W, Cout, KH, KW, OH, OW;
} ConvResult;

static ConvResult conv2d(TinyHVM *ctx, u32 x_id, u32 w_id, u32 b_id,
                         u32 B, u32 Cin, u32 H, u32 W, u32 Cout, u32 KH, u32 KW) {
    u32 OH = H - KH + 1, OW = W - KW + 1;

    // thvm_conv2d: pad → _pool → reshape → expand → permute → mul → sum → add bias
    u32 padding[] = {0, 0, 0, 0};
    u32 stride[] = {1, 1};
    Term x_t = term_ten(x_id, DTYPE_F32);
    Term w_t = term_ten(w_id, DTYPE_F32);
    Term b_t = term_ten(b_id, DTYPE_F32);

    Term out = thvm_conv2d(ctx, x_t, w_t, b_t, 1, stride, padding);
    Term out_r = thvm_reduce(ctx, out);

    return (ConvResult){.out_id=(u32)term_val(out_r), .x_id=x_id, .col_id=0,
                        .B=B, .Cin=Cin, .H=H, .W=W,
                        .Cout=Cout, .KH=KH, .KW=KW, .OH=OH, .OW=OW};
}

// ============================================================
// Conv2d backward — via backend vtable (im2col/col2im/matmul)
// ============================================================

typedef struct { u32 d_weight, d_bias, d_input; } ConvGrads;

static ConvGrads conv2d_backward(TinyHVM *ctx, u32 d_out_id, ConvResult *cr, u32 w_id) {
    u32 B=cr->B, Cin=cr->Cin, Cout=cr->Cout;
    u32 KH=cr->KH, KW=cr->KW, OH=cr->OH, OW=cr->OW;
    u32 H=cr->H, W=cr->W;
    u32 patch_size = Cin*KH*KW;
    u32 n_patches = B*OH*OW;

    Conv2dParams cp = {B, Cin, H, W, KH, KW, OH, OW, patch_size, n_patches};

    // d_out is [B, Cout, OH, OW] → NHWC: [n_patches, Cout]
    u32 dout_nhwc_buf = ctx->backend->buf_alloc(n_patches * Cout * sizeof(f32));
    ctx->backend->op_nchw_to_nhwc(dout_nhwc_buf, ctx->tensors[d_out_id].buf_id, B, Cout, OH, OW);

    // d_bias = col_sum → [Cout]
    u32 db_buf = ctx->backend->buf_alloc(Cout * sizeof(f32));
    ctx->backend->op_col_sum(db_buf, dout_nhwc_buf, n_patches, Cout);
    u32 d_bias = ctx->tensor_count++;
    ctx->tensors[d_bias] = (TensorMeta){
        .buf_id = db_buf, .dtype = DTYPE_F32,
        .view = view_create(SHAPE(Cout)),
    };

    // im2col the input (needed for d_weight computation)
    u32 col_buf = ctx->backend->buf_alloc(n_patches * patch_size * sizeof(f32));
    ctx->backend->op_im2col(col_buf, ctx->tensors[cr->x_id].buf_id, cp);

    // d_weight = col^T @ d_out_nhwc → [patch_size, Cout] then transpose to [Cout, patch_size]
    u32 colt_buf = ctx->backend->buf_alloc(patch_size * n_patches * sizeof(f32));
    ctx->backend->op_transpose(colt_buf, col_buf, n_patches, patch_size);

    u32 dw_buf = ctx->backend->buf_alloc(patch_size * Cout * sizeof(f32));
    {
        View colt_view = view_create(SHAPE(patch_size, n_patches));
        View dout_view = view_create(SHAPE(n_patches, Cout));
        ctx->backend->op_mm(dw_buf, colt_buf, &colt_view, dout_nhwc_buf, &dout_view,
                           patch_size, n_patches, Cout);
    }
    u32 dw_t_buf = ctx->backend->buf_alloc(Cout * patch_size * sizeof(f32));
    ctx->backend->op_transpose(dw_t_buf, dw_buf, patch_size, Cout);
    u32 d_weight = ctx->tensor_count++;
    ctx->tensors[d_weight] = (TensorMeta){
        .buf_id = dw_t_buf, .dtype = DTYPE_F32,
        .view = view_create((Shape){.dims={Cout,Cin,KH,KW}, .rank=4}),
    };

    // d_input: d_col = d_out_nhwc @ W → [n_patches, patch_size], then col2im
    u32 dcol_buf = ctx->backend->buf_alloc(n_patches * patch_size * sizeof(f32));
    {
        View dout_view = view_create(SHAPE(n_patches, Cout));
        View w_view = view_create(SHAPE(Cout, patch_size));
        ctx->backend->op_mm(dcol_buf, dout_nhwc_buf, &dout_view,
                           ctx->tensors[w_id].buf_id, &w_view,
                           n_patches, Cout, patch_size);
    }
    u32 dx_buf = ctx->backend->buf_alloc(B * Cin * H * W * sizeof(f32));
    ctx->backend->op_col2im(dx_buf, dcol_buf, cp);
    u32 d_input = ctx->tensor_count++;
    ctx->tensors[d_input] = (TensorMeta){
        .buf_id = dx_buf, .dtype = DTYPE_F32,
        .view = view_create((Shape){.dims={B,Cin,H,W}, .rank=4}),
    };

    return (ConvGrads){.d_weight=d_weight, .d_bias=d_bias, .d_input=d_input};
}

// ============================================================
// MaxPool2d: thvm_maxpool2d UOp composition
// ============================================================

typedef struct {
    u32 out_id;
    u32 mask_buf;  // argmax mask for backward (via backend)
    u32 x_id;
    u32 B, C, H, W;
} PoolResult;

static PoolResult maxpool2d(TinyHVM *ctx, u32 x_id, u32 B, u32 C, u32 H, u32 W) {
    // Use thvm_maxpool2d: pool → rmax (pure UOp composition)
    u32 kernel[] = {2, 2}, stride[] = {2, 2};
    Term x_t = term_ten(x_id, DTYPE_F32);
    Term out = thvm_maxpool2d(ctx, x_t, kernel, stride);
    Term out_r = thvm_reduce(ctx, out);

    u32 OH = H / 2, OW = W / 2;
    u32 out_n = B * C * OH * OW;

    // Also compute mask via backend for backward pass
    u32 mask_buf = ctx->backend->buf_alloc(out_n * sizeof(u8));
    u32 out_buf2 = ctx->backend->buf_alloc(out_n * sizeof(f32));
    ctx->backend->op_maxpool_fwd(out_buf2, mask_buf, ctx->tensors[x_id].buf_id, B, C, H, W);
    // We use the thvm result as the forward output (correct by UOp definition)
    // but store the mask for backward

    return (PoolResult){.out_id=(u32)term_val(out_r), .mask_buf=mask_buf, .x_id=x_id,
                        .B=B, .C=C, .H=H, .W=W};
}

// ============================================================
// MaxPool2d backward: via backend vtable
// ============================================================

static u32 maxpool2d_backward(TinyHVM *ctx, u32 d_out_id, PoolResult *pr) {
    u32 B=pr->B, C=pr->C, H=pr->H, W=pr->W;
    u32 in_n = B * C * H * W;

    u32 dx_buf = ctx->backend->buf_alloc(in_n * sizeof(f32));
    ctx->backend->op_zero_fill(dx_buf, in_n);
    ctx->backend->op_maxpool_bwd(dx_buf, ctx->tensors[d_out_id].buf_id, pr->mask_buf, B, C, H, W);

    u32 d_input = ctx->tensor_count++;
    ctx->tensors[d_input] = (TensorMeta){
        .buf_id = dx_buf, .dtype = DTYPE_F32,
        .view = view_create((Shape){.dims={B,C,H,W}, .rank=4}),
    };
    return d_input;
}

// ============================================================
// Linear: x @ W + b (via UOps)
// ============================================================

static Term linear(TinyHVM *ctx, Term x, Term w, Term b, u32 B, u32 in_f, u32 out_f) {
    Term y = thvm_op(ctx, UOP_MM, x, w);
    Term b_reshaped = thvm_reshape(ctx, b, SHAPE(1, out_f));
    Term b_bc = thvm_expand(ctx, b_reshaped, SHAPE(B, out_f));
    return thvm_op(ctx, UOP_ADD, y, b_bc);
    (void)in_f;
}

// ============================================================
// BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
// ============================================================

typedef struct {
    u32  out_id;
    f32 *x_hat;
    f32 *inv_std;
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
// Adam optimizer (via backend vtable)
// ============================================================

typedef struct {
    f32 lr, beta1, beta2, eps;
    u32 t;
    u32 n_params;
    u32 *param_ids;
    u32 *param_sizes;
    u32 *m_bufs;
    u32 *v_bufs;
} Adam;

static Adam adam_init(TinyHVM *ctx, f32 lr, u32 n_params) {
    Adam opt = {0};
    opt.lr = lr;
    opt.beta1 = 0.9f;
    opt.beta2 = 0.999f;
    opt.eps = 1e-8f;
    opt.t = 0;
    opt.n_params = n_params;
    opt.param_ids = calloc(n_params, sizeof(u32));
    opt.param_sizes = calloc(n_params, sizeof(u32));
    opt.m_bufs = calloc(n_params, sizeof(u32));
    opt.v_bufs = calloc(n_params, sizeof(u32));
    (void)ctx;
    return opt;
}

static void adam_add_param(TinyHVM *ctx, Adam *opt, u32 idx, u32 param_id, u32 size) {
    opt->param_ids[idx] = param_id;
    opt->param_sizes[idx] = size;
    f32 *zeros = calloc(size, sizeof(f32));
    opt->m_bufs[idx] = tensor_from(ctx, zeros, SHAPE(size));
    opt->v_bufs[idx] = tensor_from(ctx, zeros, SHAPE(size));
    free(zeros);
}

static void adam_step(TinyHVM *ctx, Adam *opt, u32 *grad_ids) {
    opt->t++;
    f32 bc1 = 1.0f - powf(opt->beta1, (f32)opt->t);
    f32 bc2 = 1.0f - powf(opt->beta2, (f32)opt->t);

    for (u32 i = 0; i < opt->n_params; i++) {
        u32 pid = opt->param_ids[i];
        u32 sz = opt->param_sizes[i];
        ctx->backend->op_adam_step(
            ctx->tensors[pid].buf_id,
            ctx->tensors[grad_ids[i]].buf_id,
            ctx->tensors[opt->m_bufs[i]].buf_id,
            ctx->tensors[opt->v_bufs[i]].buf_id,
            opt->lr, opt->beta1, opt->beta2, opt->eps, bc1, bc2, sz);
    }
}

static void adam_free(Adam *opt) {
    free(opt->param_ids); free(opt->param_sizes);
    free(opt->m_bufs); free(opt->v_bufs);
}

// ============================================================
// ReLU backward (via backend vtable)
// ============================================================

static u32 relu_backward(TinyHVM *ctx, u32 d_out_id, u32 x_id, u32 n) {
    u32 dx_buf = ctx->backend->buf_alloc(n * sizeof(f32));
    ctx->backend->op_relu_bwd(dx_buf, ctx->tensors[d_out_id].buf_id,
                               ctx->tensors[x_id].buf_id, n);
    u32 dx_id = ctx->tensor_count++;
    ctx->tensors[dx_id] = (TensorMeta){
        .buf_id = dx_buf, .dtype = DTYPE_F32,
        .view = ctx->tensors[d_out_id].view,
    };
    return dx_id;
}

// ============================================================
// Linear backward (via backend vtable)
// ============================================================

typedef struct { u32 d_weight, d_bias, d_input; } LinearGrads;

static LinearGrads linear_backward(TinyHVM *ctx, u32 d_out_id, u32 x_id, u32 w_id,
                                    u32 B, u32 in_f, u32 out_f) {
    // d_input = d_out @ W^T
    u32 wt_buf = ctx->backend->buf_alloc(in_f * out_f * sizeof(f32));
    ctx->backend->op_transpose(wt_buf, ctx->tensors[w_id].buf_id, in_f, out_f);
    u32 wt_id = ctx->tensor_count++;
    ctx->tensors[wt_id] = (TensorMeta){
        .buf_id = wt_buf, .dtype = DTYPE_F32,
        .view = view_create(SHAPE(out_f, in_f)),
    };

    View dout_view = view_create(SHAPE(B, out_f));
    View wt_view = view_create(SHAPE(out_f, in_f));
    u32 di_buf = ctx->backend->buf_alloc(B * in_f * sizeof(f32));
    ctx->backend->op_mm(di_buf, ctx->tensors[d_out_id].buf_id, &dout_view,
                        wt_buf, &wt_view, B, out_f, in_f);
    u32 d_input = ctx->tensor_count++;
    ctx->tensors[d_input] = (TensorMeta){
        .buf_id = di_buf, .dtype = DTYPE_F32,
        .view = view_create(SHAPE(B, in_f)),
    };

    // d_weight = x^T @ d_out
    u32 xt_buf = ctx->backend->buf_alloc(B * in_f * sizeof(f32));
    ctx->backend->op_transpose(xt_buf, ctx->tensors[x_id].buf_id, B, in_f);

    View xt_view = view_create(SHAPE(in_f, B));
    u32 dw_buf = ctx->backend->buf_alloc(in_f * out_f * sizeof(f32));
    ctx->backend->op_mm(dw_buf, xt_buf, &xt_view,
                        ctx->tensors[d_out_id].buf_id, &dout_view, in_f, B, out_f);
    u32 d_weight = ctx->tensor_count++;
    ctx->tensors[d_weight] = (TensorMeta){
        .buf_id = dw_buf, .dtype = DTYPE_F32,
        .view = view_create(SHAPE(in_f, out_f)),
    };

    // d_bias = col_sum(d_out)
    u32 db_buf = ctx->backend->buf_alloc(out_f * sizeof(f32));
    ctx->backend->op_col_sum(db_buf, ctx->tensors[d_out_id].buf_id, B, out_f);
    u32 d_bias = ctx->tensor_count++;
    ctx->tensors[d_bias] = (TensorMeta){
        .buf_id = db_buf, .dtype = DTYPE_F32,
        .view = view_create(SHAPE(out_f)),
    };

    return (LinearGrads){.d_weight=d_weight, .d_bias=d_bias, .d_input=d_input};
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
// Sequential composition
// ============================================================

Term thvm_sequential(TinyHVM *ctx, Term x, Layer *layers, u32 n,
                     u32 BS, int training) {
    for (u32 i = 0; i < n; i++) {
        Layer *l = &layers[i];
        switch (l->type) {
            case LAYER_CONV2D: {
                Term xr = thvm_reduce(ctx, x);
                TensorMeta *mx = &ctx->tensors[(u32)term_val(xr)];
                u32 Cin = mx->view.shape.dims[1];
                u32 H   = mx->view.shape.dims[2];
                u32 W   = mx->view.shape.dims[3];
                ConvResult cr = conv2d(ctx, (u32)term_val(xr),
                    (u32)term_val(thvm_reduce(ctx, l->conv.w)),
                    (u32)term_val(thvm_reduce(ctx, l->conv.b)),
                    BS, Cin, H, W, l->conv.co, l->conv.k, l->conv.k);
                x = term_ten(cr.out_id, DTYPE_F32);
                break;
            }
            case LAYER_BN: {
                Term xr = thvm_reduce(ctx, x);
                TensorMeta *mx = &ctx->tensors[(u32)term_val(xr)];
                u32 C = mx->view.shape.dims[1];
                u32 H = mx->view.shape.dims[2];
                u32 W = mx->view.shape.dims[3];
                BNResult bnr = batchnorm(ctx, (u32)term_val(xr),
                    (u32)term_val(thvm_reduce(ctx, l->bn.gamma)),
                    (u32)term_val(thvm_reduce(ctx, l->bn.beta)),
                    (u32)term_val(thvm_reduce(ctx, l->bn.rmean)),
                    (u32)term_val(thvm_reduce(ctx, l->bn.rvar)),
                    BS, C, H, W, training);
                x = term_ten(bnr.out_id, DTYPE_F32);
                break;
            }
            case LAYER_MAXPOOL: {
                Term xr = thvm_reduce(ctx, x);
                TensorMeta *mx = &ctx->tensors[(u32)term_val(xr)];
                u32 C = mx->view.shape.dims[1];
                u32 H = mx->view.shape.dims[2];
                u32 W = mx->view.shape.dims[3];
                PoolResult pr = maxpool2d(ctx, (u32)term_val(xr), BS, C, H, W);
                x = term_ten(pr.out_id, DTYPE_F32);
                break;
            }
            case LAYER_FLATTEN: {
                Term xr = thvm_reduce(ctx, x);
                TensorMeta *mx = &ctx->tensors[(u32)term_val(xr)];
                u32 numel = mx->view.numel / BS;
                mx->view = view_create(SHAPE(BS, numel));
                x = xr;
                break;
            }
            case LAYER_LINEAR: {
                Term xr = thvm_reduce(ctx, x);
                Term lo = linear(ctx, xr, l->lin.w, l->lin.b,
                    BS, l->lin.in_f, l->lin.out_f);
                x = thvm_reduce(ctx, lo);
                break;
            }
            case LAYER_FN:
                x = l->fn(ctx, x);
                break;
        }
    }
    return x;
}
