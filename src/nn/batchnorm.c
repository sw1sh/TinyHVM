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
    Term assign_rm = term_era(), assign_rv = term_era();

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
        // Lazy assigns — chained into the output so the outer reduce executes them.
        f32 mom = 0.1f, bessel = (f32)count / (f32)(count - 1);
        Term mean_d = thvm_detach(ctx, batch_mean);
        Term var_d = thvm_detach(ctx, batch_var);
        f32 omm = 1.0f - mom, mb = mom * bessel;
        Term new_rm = thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MUL, rmean, thvm_tensor(ctx, &omm, SHAPE(1))),
            thvm_op(ctx, UOP_MUL, mean_d, thvm_tensor(ctx, &mom, SHAPE(1))));
        Term new_rv = thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MUL, rvar, thvm_tensor(ctx, &omm, SHAPE(1))),
            thvm_op(ctx, UOP_MUL, var_d, thvm_tensor(ctx, &mb, SHAPE(1))));
        assign_rm = thvm_assign(ctx, rmean, new_rm);
        assign_rv = thvm_assign(ctx, rvar, new_rv);
    } else {
        batch_mean = rmean;
        batch_var = rvar;
    }

    // x.batchnorm(gamma, beta, batch_mean, (batch_var + eps).rsqrt())
    // Live mean in output path (gradient flows through mean for BN backward correction)
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

    Term out = thvm_op(ctx, UOP_ADD,
        thvm_op(ctx, UOP_MUL, thvm_op(ctx, UOP_MUL,
            thvm_op(ctx, UOP_SUB, x, mean_bc), invstd_bc), gamma_bc),
        beta_bc);

    // Chain running stats assigns: APP(assign, continuation) forces assign via APP-TEN rule.
    // The outer reduce executes: assign_rm → assign_rv → out.
    if (term_tag(assign_rm) != TAG_ERA)
        out = thvm_app(ctx, assign_rm, thvm_app(ctx, assign_rv, out));
    return out;
}
