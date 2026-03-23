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
// Cross-entropy loss (pure UOp composition)
// loss = -mean(sum(one_hot * log(softmax(logits))))
// ============================================================

static Term cross_entropy_loss(TinyHVM *ctx, Term logits, u8 *labels, u32 B, u32 C) {
    // softmax (already UOp-based)
    Term probs = softmax(ctx, logits, B, C);

    // clamp for numerical stability: max(probs, 1e-7)
    f32 eps = 1e-7f;
    Term eps_t = thvm_expand(ctx, thvm_tensor(ctx, &eps, SHAPE(1, 1)), SHAPE(B, C));
    Term clamped = thvm_op(ctx, UOP_MAX, probs, eps_t);

    // log(softmax)
    Term log_probs = thvm_op(ctx, UOP_LOG, clamped, term_era());

    // one-hot labels: [B, C] with 1.0 at correct class
    f32 *oh = calloc(B * C, sizeof(f32));
    for (u32 i = 0; i < B; i++) oh[i * C + labels[i]] = 1.0f;
    Term one_hot = thvm_tensor(ctx, oh, SHAPE(B, C));
    free(oh);

    // -sum(one_hot * log_probs) / B
    Term masked = thvm_op(ctx, UOP_MUL, one_hot, log_probs);
    // Sum over classes (last dim), then sum over batch
    Term sum_c = thvm_op(ctx, UOP_SUM, masked, term_era());  // [B, 1]
    Term sum_b = thvm_op(ctx, UOP_SUM, sum_c, term_era());  // [1, 1]
    Term neg = thvm_op(ctx, UOP_NEG, sum_b, term_era());

    f32 inv_B = 1.0f / (f32)B;
    Term scale = thvm_tensor(ctx, &inv_B, SHAPE(1, 1));
    return thvm_op(ctx, UOP_MUL, neg, scale);  // scalar loss
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
    u32 patch_size = Cin * KH * KW;
    u32 n_patches = B * OH * OW;
    Conv2dParams cp = {B, Cin, H, W, KH, KW, OH, OW, patch_size, n_patches};

    // Direct Metal dispatch: im2col → matmul → bias_add → nhwc_to_nchw
    u32 col_buf = ctx->backend->buf_alloc(n_patches * patch_size * sizeof(f32));
    ctx->backend->op_im2col(col_buf, ctx->tensors[x_id].buf_id, cp);

    // matmul: col[n_patches, patch_size] @ W^T → [n_patches, Cout]
    u32 out_nhwc = ctx->backend->buf_alloc(n_patches * Cout * sizeof(f32));
    {
        View col_view = view_create(SHAPE(n_patches, patch_size));
        View w_view = view_create(SHAPE(Cout, patch_size));  // W is [Cout, Cin*KH*KW]
        // Need W transposed: [patch_size, Cout]
        u32 wt_buf = ctx->backend->buf_alloc(patch_size * Cout * sizeof(f32));
        ctx->backend->op_transpose(wt_buf, ctx->tensors[w_id].buf_id, Cout, patch_size);
        View wt_view = view_create(SHAPE(patch_size, Cout));
        ctx->backend->op_mm(out_nhwc, col_buf, &col_view, wt_buf, &wt_view,
                           n_patches, patch_size, Cout);
    }

    // bias_add
    ctx->backend->op_bias_add(out_nhwc, ctx->tensors[b_id].buf_id, Cout, n_patches);

    // nhwc_to_nchw: [B,OH,OW,Cout] → [B,Cout,OH,OW]
    u32 out_buf = ctx->backend->buf_alloc(n_patches * Cout * sizeof(f32));
    ctx->backend->op_nhwc_to_nchw(out_buf, out_nhwc, B, Cout, OH, OW);

    u32 out_id = ctx->tensor_count++;
    ctx->tensors[out_id] = (TensorMeta){
        .buf_id = out_buf, .dtype = DTYPE_F32,
        .view = view_create((Shape){.dims={B,Cout,OH,OW}, .rank=4}),
    };

    // Store col_id for backward
    u32 col_id = ctx->tensor_count++;
    ctx->tensors[col_id] = (TensorMeta){
        .buf_id = col_buf, .dtype = DTYPE_F32,
        .view = view_create(SHAPE(n_patches, patch_size)),
    };

    return (ConvResult){.out_id=out_id, .x_id=x_id, .col_id=col_id,
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
    u32 OH = H / 2, OW = W / 2;
    u32 out_n = B * C * OH * OW;

    // Single direct Metal dispatch: maxpool_fwd produces output + argmax mask
    u32 out_buf = ctx->backend->buf_alloc(out_n * sizeof(f32));
    u32 mask_buf = ctx->backend->buf_alloc(out_n * sizeof(u8));
    ctx->backend->op_maxpool_fwd(out_buf, mask_buf, ctx->tensors[x_id].buf_id, B, C, H, W);

    u32 out_id = ctx->tensor_count++;
    ctx->tensors[out_id] = (TensorMeta){
        .buf_id = out_buf, .dtype = DTYPE_F32,
        .view = view_create((Shape){.dims={B,C,OH,OW}, .rank=4}),
    };

    return (PoolResult){.out_id=out_id, .mask_buf=mask_buf, .x_id=x_id,
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
    u32  B, C, H, W;
} BNResult;

static BNResult batchnorm(TinyHVM *ctx, u32 x_id,
                           u32 gamma_id, u32 beta_id,
                           u32 rmean_id, u32 rvar_id,
                           u32 B, u32 C, u32 H, u32 W, int training) {
    f32 eps = 1e-5f, momentum = 0.1f;
    u32 count = B * H * W;

    Term x = term_ten(x_id, DTYPE_F32);
    Term mean_t, var_t;

    if (training) {
        // Compute mean per channel via UOps:
        // x:[B,C,H,W] → permute to [C,B,H,W] → reshape to [C, B*H*W]
        // → SUM → [C,1] → reshape [C] → DIV by count
        Term x_perm = thvm_permute(ctx, x, (u32[]){1,0,2,3}, 4);
        Term x_flat = thvm_reshape(ctx, x_perm, SHAPE(C, count));
        Term x_sum = thvm_op(ctx, UOP_SUM, x_flat, term_era());
        // x_sum: [C, 1]
        Term x_sum_sq = thvm_reshape(ctx, x_sum, SHAPE(C));
        f32 inv_count = 1.0f / (f32)count;
        Term inv_n = thvm_tensor(ctx, &inv_count, SHAPE(1));
        Term inv_n_bc = thvm_expand(ctx, inv_n, SHAPE(C));
        mean_t = thvm_op(ctx, UOP_MUL, x_sum_sq, inv_n_bc);
        // mean_t: [C]

        // Compute variance: var = sum((x - mean)^2) / count
        // Broadcast mean to [B,C,H,W]
        Term mean_4d = thvm_expand(ctx,
            thvm_reshape(ctx, mean_t, SHAPE(1, C, 1, 1)),
            (Shape){.dims={B,C,H,W}, .rank=4});
        Term diff = thvm_op(ctx, UOP_SUB, x, mean_4d);
        Term diff2 = thvm_op(ctx, UOP_MUL, diff, diff);
        // Sum diff2 per channel
        Term d2_perm = thvm_permute(ctx, diff2, (u32[]){1,0,2,3}, 4);
        Term d2_flat = thvm_reshape(ctx, d2_perm, SHAPE(C, count));
        Term d2_sum = thvm_op(ctx, UOP_SUM, d2_flat, term_era());
        Term d2_sum_sq = thvm_reshape(ctx, d2_sum, SHAPE(C));
        var_t = thvm_op(ctx, UOP_MUL, d2_sum_sq, inv_n_bc);
        // var_t: [C]

        // Update running stats (CPU-side, not in gradient graph)
        f32 *mean_host = thvm_to_host(ctx, thvm_reduce(ctx, mean_t));
        f32 *var_host = thvm_to_host(ctx, thvm_reduce(ctx, var_t));
        f32 *rm = malloc(C * sizeof(f32)), *rv = malloc(C * sizeof(f32));
        ctx->backend->buf_read(ctx->tensors[rmean_id].buf_id, rm, C*sizeof(f32));
        ctx->backend->buf_read(ctx->tensors[rvar_id].buf_id, rv, C*sizeof(f32));
        f32 bessel = (f32)count / (f32)(count - 1);
        for (u32 c = 0; c < C; c++) {
            rm[c] = (1-momentum)*rm[c] + momentum*mean_host[c];
            rv[c] = (1-momentum)*rv[c] + momentum*var_host[c]*bessel;
        }
        ctx->backend->buf_write(ctx->tensors[rmean_id].buf_id, rm, C*sizeof(f32));
        ctx->backend->buf_write(ctx->tensors[rvar_id].buf_id, rv, C*sizeof(f32));
        free(rm); free(rv);
    } else {
        mean_t = term_ten(rmean_id, DTYPE_F32);
        var_t = term_ten(rvar_id, DTYPE_F32);
    }

    // Normalize: x_hat = (x - mean) / sqrt(var + eps)
    Term mean_bc = thvm_expand(ctx,
        thvm_reshape(ctx, mean_t, SHAPE(1, C, 1, 1)),
        (Shape){.dims={B,C,H,W}, .rank=4});
    Term centered = thvm_op(ctx, UOP_SUB, x, mean_bc);

    f32 eps_f = eps;
    Term eps_t = thvm_tensor(ctx, &eps_f, SHAPE(1));
    Term eps_bc = thvm_expand(ctx, eps_t, SHAPE(C));
    Term var_eps = thvm_op(ctx, UOP_ADD, var_t, eps_bc);
    Term std = thvm_op(ctx, UOP_SQRT, var_eps, term_era());
    Term inv_std = thvm_expand(ctx,
        thvm_reshape(ctx, std, SHAPE(1, C, 1, 1)),
        (Shape){.dims={B,C,H,W}, .rank=4});
    Term x_hat = thvm_op(ctx, UOP_DIV, centered, inv_std);

    // Scale: out = x_hat * gamma + beta
    Term gamma_bc = thvm_expand(ctx,
        thvm_reshape(ctx, term_ten(gamma_id, DTYPE_F32), SHAPE(1, C, 1, 1)),
        (Shape){.dims={B,C,H,W}, .rank=4});
    Term beta_bc = thvm_expand(ctx,
        thvm_reshape(ctx, term_ten(beta_id, DTYPE_F32), SHAPE(1, C, 1, 1)),
        (Shape){.dims={B,C,H,W}, .rank=4});
    Term out = thvm_op(ctx, UOP_ADD,
                       thvm_op(ctx, UOP_MUL, x_hat, gamma_bc),
                       beta_bc);

    Term out_r = thvm_reduce(ctx, out);
    return (BNResult){.out_id=(u32)term_val(out_r), .B=B, .C=C, .H=H, .W=W};
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

// batchnorm_backward removed — autograd handles BN gradients via UOp composition

// ============================================================
// Sequential composition
// ============================================================

Term thvm_sequential(TinyHVM *ctx, Term x, Layer *layers, u32 n,
                     u32 BS, int training) {
    for (u32 i = 0; i < n; i++) {
        Layer *l = &layers[i];
        switch (l->type) {
            case LAYER_CONV2D: {
                // Pure UOp composition: thvm_conv2d
                u32 padding[] = {0, 0, 0, 0};
                u32 stride[] = {1, 1};
                x = thvm_conv2d(ctx, x, l->conv.w, l->conv.b,
                                1, stride, padding);
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
                // Pure UOp composition: thvm_maxpool2d
                u32 kernel[] = {l->pool.ks, l->pool.ks};
                u32 stride[] = {l->pool.ks, l->pool.ks};
                x = thvm_maxpool2d(ctx, x, kernel, stride);
                break;
            }
            case LAYER_FLATTEN: {
                Term xr = thvm_reduce(ctx, x);
                TensorMeta *mx = &ctx->tensors[(u32)term_val(xr)];
                u32 numel = mx->view.numel / BS;
                x = thvm_reshape(ctx, xr, SHAPE(BS, numel));
                break;
            }
            case LAYER_LINEAR: {
                x = linear(ctx, x, l->lin.w, l->lin.b,
                           BS, l->lin.in_f, l->lin.out_f);
                break;
            }
            case LAYER_FN:
                x = l->fn(ctx, x);
                break;
        }
    }
    return x;
}
