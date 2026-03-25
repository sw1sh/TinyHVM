// nn/batchnorm.c — Batch normalization (pure IC Term composition)

static Term batchnorm_term(TinyHVM *ctx, Term x,
                            Term gamma, Term beta, Term rmean, Term rvar,
                            u32 B, u32 C, u32 H, u32 W, int training) {
    f32 eps = 1e-5f, momentum = 0.1f;
    u32 count = B * H * W;

    Term mean_t, var_t, centered;

    if (training) {
        // Compute batch mean
        Term x_perm = thvm_permute(ctx, x, (u32[]){1,0,2,3}, 4);
        Term x_flat = thvm_reshape(ctx, x_perm, SHAPE(C, count));
        Term x_sum = thvm_reshape(ctx,
            thvm_op(ctx, UOP_SUM, x_flat, term_era()), SHAPE(C));
        f32 inv_count = 1.0f / (f32)count;
        Term inv_n = thvm_expand(ctx, thvm_tensor(ctx, &inv_count, SHAPE(1)), SHAPE(C));
        mean_t = thvm_op(ctx, UOP_MUL, x_sum, inv_n);

        // centered = x - mean (compute ONCE, reuse for var AND output)
        Term mean_4d = thvm_expand(ctx,
            thvm_reshape(ctx, mean_t, SHAPE(1, C, 1, 1)),
            (Shape){.dims={B,C,H,W}, .rank=4});
        centered = thvm_op(ctx, UOP_SUB, x, mean_4d);

        // Compute batch variance from centered (shares the same diff tensor)
        Term diff2 = thvm_op(ctx, UOP_MUL, centered, centered);
        Term d2_perm = thvm_permute(ctx, diff2, (u32[]){1,0,2,3}, 4);
        Term d2_flat = thvm_reshape(ctx, d2_perm, SHAPE(C, count));
        Term d2_sum = thvm_reshape(ctx,
            thvm_op(ctx, UOP_SUM, d2_flat, term_era()), SHAPE(C));
        var_t = thvm_op(ctx, UOP_MUL, d2_sum, inv_n);

        // Update running stats (side effect, not part of gradient graph)
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
        Term mean_bc = thvm_expand(ctx,
            thvm_reshape(ctx, mean_t, SHAPE(1, C, 1, 1)),
            (Shape){.dims={B,C,H,W}, .rank=4});
        centered = thvm_op(ctx, UOP_SUB, x, mean_bc);
    }

    // Normalize: x_hat = centered / sqrt(var + eps)
    f32 eps_f = eps;
    Term eps_bc = thvm_expand(ctx, thvm_tensor(ctx, &eps_f, SHAPE(1)), SHAPE(C));
    Term var_eps = thvm_op(ctx, UOP_ADD, var_t, eps_bc);
    Term std_t = thvm_reduce(ctx, thvm_op(ctx, UOP_SQRT, var_eps, term_era())); // [C]
    Term inv_std = thvm_expand(ctx,
        thvm_reshape(ctx, std_t, SHAPE(1, C, 1, 1)),
        (Shape){.dims={B,C,H,W}, .rank=4});
    Term inv_std_r = thvm_reduce(ctx, inv_std);
    Term x_hat = thvm_reduce(ctx, thvm_op(ctx, UOP_DIV, centered, inv_std_r));

    // Scale + shift: gamma * x_hat + beta
    Term gamma_bc = thvm_expand(ctx,
        thvm_reshape(ctx, gamma, SHAPE(1, C, 1, 1)),
        (Shape){.dims={B,C,H,W}, .rank=4});
    Term beta_bc = thvm_expand(ctx,
        thvm_reshape(ctx, beta, SHAPE(1, C, 1, 1)),
        (Shape){.dims={B,C,H,W}, .rank=4});
    Term out = thvm_op(ctx, UOP_ADD,
                       thvm_op(ctx, UOP_MUL, x_hat, gamma_bc),
                       beta_bc);

    // Record BN as a single provenance node for correct backward.
    // IC autograd through BN composition gets wrong gradients because it
    // doesn't properly handle the statistical dependencies (mean/var of x).
    // Instead, backward_local implements the standard BN backward formula directly.
    Term out_r = thvm_reduce(ctx, out);
    if (term_tag(out_r) == TAG_TEN && training) {
        Term xh_r = x_hat;    // already reduced above
        Term is_r = inv_std_r; // already reduced above
        u32 out_id = (u32)term_val(out_r);
        ctx->tensors[out_id].creator_op = UOP_BATCHNORM;
        ctx->tensors[out_id].src_ids[0] = (u32)term_val(thvm_reduce(ctx, x));
        ctx->tensors[out_id].src_ids[1] = (u32)term_val(thvm_reduce(ctx, gamma));
        ctx->tensors[out_id].bn_x_hat_id = (u32)term_val(xh_r);
        ctx->tensors[out_id].bn_inv_std_id = (u32)term_val(is_r);
        ctx->tensors[out_id].bn_count = count;
        ctx->tensors[out_id].bn_x_id = (u32)term_val(thvm_reduce(ctx, x));
        ctx->tensors[out_id].bn_gamma_id = (u32)term_val(thvm_reduce(ctx, gamma));
        ctx->tensors[out_id].bn_beta_id = (u32)term_val(thvm_reduce(ctx, beta));
        if (ctx->tensors[(u32)term_val(thvm_reduce(ctx, x))].requires_grad ||
            ctx->tensors[(u32)term_val(thvm_reduce(ctx, gamma))].requires_grad)
            ctx->tensors[out_id].requires_grad = 1;
    }
    return out_r;
}
