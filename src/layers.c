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

// Conv2d and MaxPool2d forward+backward are now handled by
// thvm_conv2d/thvm_maxpool2d (UOp composition) + thvm_grad (IC autograd).
// See tinyhvm.c for the implementations.



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
// BatchNorm: pure IC Term composition
// (x - mean) / sqrt(var + eps) * gamma + beta
// ============================================================

static Term batchnorm_term(TinyHVM *ctx, Term x,
                            Term gamma, Term beta, Term rmean, Term rvar,
                            u32 B, u32 C, u32 H, u32 W, int training) {
    f32 eps = 1e-5f, momentum = 0.1f;
    u32 count = B * H * W;

    Term mean_t, var_t;

    if (training) {
        // Mean per channel: x→permute[C,B,H,W]→reshape[C,count]→SUM→/count
        Term x_perm = thvm_permute(ctx, x, (u32[]){1,0,2,3}, 4);
        Term x_flat = thvm_reshape(ctx, x_perm, SHAPE(C, count));
        Term x_sum = thvm_reshape(ctx,
            thvm_op(ctx, UOP_SUM, x_flat, term_era()), SHAPE(C));
        f32 inv_count = 1.0f / (f32)count;
        Term inv_n = thvm_expand(ctx, thvm_tensor(ctx, &inv_count, SHAPE(1)), SHAPE(C));
        mean_t = thvm_op(ctx, UOP_MUL, x_sum, inv_n);

        // Variance: sum((x - mean)²) / count
        Term mean_4d = thvm_expand(ctx,
            thvm_reshape(ctx, mean_t, SHAPE(1, C, 1, 1)),
            (Shape){.dims={B,C,H,W}, .rank=4});
        Term diff = thvm_op(ctx, UOP_SUB, x, mean_4d);
        Term diff2 = thvm_op(ctx, UOP_MUL, diff, diff);
        Term d2_perm = thvm_permute(ctx, diff2, (u32[]){1,0,2,3}, 4);
        Term d2_flat = thvm_reshape(ctx, d2_perm, SHAPE(C, count));
        Term d2_sum = thvm_reshape(ctx,
            thvm_op(ctx, UOP_SUM, d2_flat, term_era()), SHAPE(C));
        var_t = thvm_op(ctx, UOP_MUL, d2_sum, inv_n);

        // Update running stats (side effect, not in gradient graph)
        f32 *m_host = thvm_to_host(ctx, thvm_reduce(ctx, mean_t));
        f32 *v_host = thvm_to_host(ctx, thvm_reduce(ctx, var_t));
        f32 *rm = thvm_to_host(ctx, thvm_reduce(ctx, rmean));
        f32 *rv = thvm_to_host(ctx, thvm_reduce(ctx, rvar));
        f32 bessel = (f32)count / (f32)(count - 1);
        f32 *rm_new = malloc(C * sizeof(f32)), *rv_new = malloc(C * sizeof(f32));
        for (u32 c = 0; c < C; c++) {
            rm_new[c] = (1-momentum)*rm[c] + momentum*m_host[c];
            rv_new[c] = (1-momentum)*rv[c] + momentum*v_host[c]*bessel;
        }
        // Write back via thvm_to_host's existing buf
        Term rmean_r = thvm_reduce(ctx, rmean);
        Term rvar_r = thvm_reduce(ctx, rvar);
        ctx->backend->buf_write(ctx->tensors[(u32)term_val(rmean_r)].buf_id,
                                rm_new, C*sizeof(f32));
        ctx->backend->buf_write(ctx->tensors[(u32)term_val(rvar_r)].buf_id,
                                rv_new, C*sizeof(f32));
        free(rm_new); free(rv_new);
    } else {
        mean_t = rmean;
        var_t = rvar;
    }

    // Normalize: x_hat = (x - mean) / sqrt(var + eps)
    Term mean_bc = thvm_expand(ctx,
        thvm_reshape(ctx, mean_t, SHAPE(1, C, 1, 1)),
        (Shape){.dims={B,C,H,W}, .rank=4});
    Term centered = thvm_op(ctx, UOP_SUB, x, mean_bc);

    f32 eps_f = eps;
    Term eps_bc = thvm_expand(ctx, thvm_tensor(ctx, &eps_f, SHAPE(1)), SHAPE(C));
    Term var_eps = thvm_op(ctx, UOP_ADD, var_t, eps_bc);
    Term inv_std = thvm_expand(ctx,
        thvm_reshape(ctx, thvm_op(ctx, UOP_SQRT, var_eps, term_era()),
                     SHAPE(1, C, 1, 1)),
        (Shape){.dims={B,C,H,W}, .rank=4});
    Term x_hat = thvm_op(ctx, UOP_DIV, centered, inv_std);

    // Scale: out = x_hat * gamma + beta
    Term gamma_bc = thvm_expand(ctx,
        thvm_reshape(ctx, gamma, SHAPE(1, C, 1, 1)),
        (Shape){.dims={B,C,H,W}, .rank=4});
    Term beta_bc = thvm_expand(ctx,
        thvm_reshape(ctx, beta, SHAPE(1, C, 1, 1)),
        (Shape){.dims={B,C,H,W}, .rank=4});
    return thvm_op(ctx, UOP_ADD,
                   thvm_op(ctx, UOP_MUL, x_hat, gamma_bc),
                   beta_bc);
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

// All backward functions (conv2d_backward, relu_backward, linear_backward,
// maxpool2d_backward) removed — IC autograd (thvm_grad) handles all gradients.

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
                // Reduce x first — BN needs concrete [B,C,H,W] shape
                Term xr = thvm_reduce(ctx, x);
                TensorMeta *mx = &ctx->tensors[(u32)term_val(xr)];
                u32 C = mx->view.shape.dims[1];
                u32 H = mx->view.shape.dims[2];
                u32 W = mx->view.shape.dims[3];
                x = batchnorm_term(ctx, xr,
                    l->bn.gamma, l->bn.beta, l->bn.rmean, l->bn.rvar,
                    BS, C, H, W, training);
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
