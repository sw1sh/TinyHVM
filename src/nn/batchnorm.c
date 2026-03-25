// nn/batchnorm.c — Batch normalization (pure IC Term composition)
// Matches tinygrad: detach mean for variance, live mean+var for output.
// Running stats via buf_write (equivalent to tinygrad's .assign()).

static Term thvm_detach(TinyHVM *ctx, Term t) {
    t = thvm_reduce(ctx, t);
    if (term_tag(t) != TAG_TEN) return t;
    u32 src = (u32)term_val(t);
    u32 id = ctx->tensor_count++;
    ctx->tensors[id] = ctx->tensors[src];
    ctx->tensors[id].host_ptr = NULL;
    ctx->tensors[id].creator_op = 0;
    ctx->tensors[id].requires_grad = 0;
    return term_ten(id, ctx->tensors[src].dtype);
}

static Term batchnorm_term(TinyHVM *ctx, Term x,
                            Term gamma, Term beta, Term rmean, Term rvar,
                            u32 B, u32 C, u32 H, u32 W, int training) {
    f32 eps_val = 1e-5f;
    u32 count = B * H * W;
    f32 inv_count = 1.0f / (f32)count;

    Term batch_mean, batch_var;

    if (training) {
        // batch_mean = x.mean(axis=(0,2,3))
        Term x_perm = thvm_permute(ctx, x, (u32[]){1,0,2,3}, 4);
        Term x_flat = thvm_reshape(ctx, x_perm, SHAPE(C, count));
        Term x_sum = thvm_reshape(ctx,
            thvm_op(ctx, UOP_SUM, x_flat, term_era()), SHAPE(C));
        Term inv_n = thvm_expand(ctx, thvm_tensor(ctx, &inv_count, SHAPE(1)), SHAPE(C));
        batch_mean = thvm_op(ctx, UOP_MUL, x_sum, inv_n);

        // batch_var = ((x - detach(mean))^2).mean(axis=(0,2,3))
        Term mean_det = thvm_detach(ctx, batch_mean);
        Term md4 = thvm_expand(ctx, thvm_reshape(ctx, mean_det, SHAPE(1,C,1,1)),
            (Shape){.dims={B,C,H,W},.rank=4});
        Term y = thvm_op(ctx, UOP_SUB, x, md4);
        Term y2 = thvm_op(ctx, UOP_MUL, y, y);
        Term y2p = thvm_permute(ctx, y2, (u32[]){1,0,2,3}, 4);
        Term y2f = thvm_reshape(ctx, y2p, SHAPE(C, count));
        Term y2s = thvm_reshape(ctx, thvm_op(ctx, UOP_SUM, y2f, term_era()), SHAPE(C));
        batch_var = thvm_op(ctx, UOP_MUL, y2s, inv_n);

        // running_mean.assign((1-m)*running_mean + m*detach(batch_mean))
        // running_var.assign((1-m)*running_var + m*bessel*detach(batch_var))
        f32 mom = 0.1f, bessel = (f32)count / (f32)(count - 1);
        Term mean_d = thvm_detach(ctx, batch_mean);
        Term var_d = thvm_detach(ctx, batch_var);
        f32 omm = 1.0f - mom;
        Term omm_t = thvm_tensor(ctx, &omm, SHAPE(1));
        Term mom_t = thvm_tensor(ctx, &mom, SHAPE(1));
        f32 mb = mom * bessel;
        Term mb_t = thvm_tensor(ctx, &mb, SHAPE(1));
        // new_rm = (1-m)*rm + m*mean
        Term new_rm = thvm_reduce(ctx, thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MUL, rmean, omm_t),
            thvm_op(ctx, UOP_MUL, mean_d, mom_t)));
        // new_rv = (1-m)*rv + m*bessel*var
        Term new_rv = thvm_reduce(ctx, thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MUL, rvar, omm_t),
            thvm_op(ctx, UOP_MUL, var_d, mb_t)));
        // assign: copy new_rm/new_rv GPU buffers into rmean/rvar GPU buffers
        u32 rm_id = (u32)term_val(thvm_reduce(ctx, rmean));
        u32 rv_id = (u32)term_val(thvm_reduce(ctx, rvar));
        { f32 tmp[MAX_TENSORS > 256 ? 256 : 64]; // enough for C channels
          ctx->backend->buf_read(ctx->tensors[(u32)term_val(new_rm)].buf_id, tmp, C*sizeof(f32));
          ctx->backend->buf_write(ctx->tensors[rm_id].buf_id, tmp, C*sizeof(f32));
          ctx->backend->buf_read(ctx->tensors[(u32)term_val(new_rv)].buf_id, tmp, C*sizeof(f32));
          ctx->backend->buf_write(ctx->tensors[rv_id].buf_id, tmp, C*sizeof(f32));
        }
    } else {
        batch_mean = rmean;
        batch_var = rvar;
    }

    // x.batchnorm(gamma, beta, batch_mean, (batch_var + eps).rsqrt())
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

    return thvm_op(ctx, UOP_ADD,
        thvm_op(ctx, UOP_MUL, thvm_op(ctx, UOP_MUL,
            thvm_op(ctx, UOP_SUB, x, mean_bc), invstd_bc), gamma_bc),
        beta_bc);
}
