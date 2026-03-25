// nn/batchnorm.c — Batch normalization (pure IC Term composition)
// Forward: lazy graph composition. Running stats: single eager side-effect.
// Detach mean for variance (d(var)/d(mean)=0), live mean+var for output.

static Term batchnorm_term(TinyHVM *ctx, Term x,
                            Term gamma, Term beta, Term rmean, Term rvar,
                            u32 B, u32 C, u32 H, u32 W, int training) {
    f32 eps_val = 1e-5f;
    u32 count = B * H * W;
    f32 inv_count = 1.0f / (f32)count;

    Term batch_mean, batch_var;

    if (training) {
        // batch_mean = mean(x, axis=(0,2,3)) — lazy
        Term x_perm = thvm_permute(ctx, x, (u32[]){1,0,2,3}, 4);
        Term x_flat = thvm_reshape(ctx, x_perm, SHAPE(C, count));
        Term x_sum = thvm_reshape(ctx,
            thvm_op(ctx, UOP_SUM, x_flat, term_era()), SHAPE(C));
        Term inv_n = thvm_expand(ctx, thvm_tensor(ctx, &inv_count, SHAPE(1)), SHAPE(C));
        batch_mean = thvm_op(ctx, UOP_MUL, x_sum, inv_n);

        // Detach mean for variance computation: reduce once, create leaf copy
        Term mean_r = thvm_reduce(ctx, batch_mean);
        u32 mean_det_id = ctx->tensor_count++;
        ctx->tensors[mean_det_id] = ctx->tensors[(u32)term_val(mean_r)];
        ctx->tensors[mean_det_id].host_ptr = NULL;
        ctx->tensors[mean_det_id].creator_op = 0;   // DETACH
        ctx->tensors[mean_det_id].requires_grad = 0;
        Term mean_det = term_ten(mean_det_id, DTYPE_F32);

        // batch_var = mean((x - detach(mean))^2, axis=(0,2,3)) — lazy
        Term md4 = thvm_expand(ctx, thvm_reshape(ctx, mean_det, SHAPE(1,C,1,1)),
            (Shape){.dims={B,C,H,W},.rank=4});
        Term y = thvm_op(ctx, UOP_SUB, x, md4);
        Term y2 = thvm_op(ctx, UOP_MUL, y, y);
        Term y2p = thvm_permute(ctx, y2, (u32[]){1,0,2,3}, 4);
        Term y2f = thvm_reshape(ctx, y2p, SHAPE(C, count));
        Term y2s = thvm_reshape(ctx, thvm_op(ctx, UOP_SUM, y2f, term_era()), SHAPE(C));
        batch_var = thvm_op(ctx, UOP_MUL, y2s, inv_n);

        // Running stats update — the ONLY eager part
        f32 momentum = 0.1f;
        Term var_r = thvm_reduce(ctx, batch_var);
        f32 *m = thvm_to_host(ctx, mean_r);
        f32 *v = thvm_to_host(ctx, var_r);
        f32 *rm = thvm_to_host(ctx, thvm_reduce(ctx, rmean));
        f32 *rv = thvm_to_host(ctx, thvm_reduce(ctx, rvar));
        f32 bessel = (f32)count / (f32)(count - 1);
        for (u32 c = 0; c < C; c++) {
            rm[c] = (1-momentum)*rm[c] + momentum*m[c];
            rv[c] = (1-momentum)*rv[c] + momentum*v[c]*bessel;
        }
        ctx->backend->buf_write(ctx->tensors[(u32)term_val(thvm_reduce(ctx, rmean))].buf_id, rm, C*sizeof(f32));
        ctx->backend->buf_write(ctx->tensors[(u32)term_val(thvm_reduce(ctx, rvar))].buf_id, rv, C*sizeof(f32));
    } else {
        batch_mean = rmean;
        batch_var = rvar;
    }

    // Output: (x - mean) * rsqrt(var + eps) * gamma + beta — lazy
    Term mean_bc = thvm_expand(ctx, thvm_reshape(ctx, batch_mean, SHAPE(1,C,1,1)),
        (Shape){.dims={B,C,H,W},.rank=4});
    Term eps_bc = thvm_expand(ctx, thvm_tensor(ctx, &eps_val, SHAPE(1)), SHAPE(C));
    Term invstd = thvm_op(ctx, UOP_DIV,
        thvm_tensor(ctx, &(f32){1.0f}, SHAPE(1)),
        thvm_op(ctx, UOP_SQRT, thvm_op(ctx, UOP_ADD, batch_var, eps_bc), term_era()));
    Term invstd_bc = thvm_expand(ctx, thvm_reshape(ctx, invstd, SHAPE(1,C,1,1)),
        (Shape){.dims={B,C,H,W},.rank=4});
    Term gamma_bc = thvm_expand(ctx, thvm_reshape(ctx, gamma, SHAPE(1,C,1,1)),
        (Shape){.dims={B,C,H,W},.rank=4});
    Term beta_bc = thvm_expand(ctx, thvm_reshape(ctx, beta, SHAPE(1,C,1,1)),
        (Shape){.dims={B,C,H,W},.rank=4});

    Term centered = thvm_op(ctx, UOP_SUB, x, mean_bc);
    return thvm_op(ctx, UOP_ADD,
        thvm_op(ctx, UOP_MUL, thvm_op(ctx, UOP_MUL, centered, invstd_bc), gamma_bc),
        beta_bc);
}
