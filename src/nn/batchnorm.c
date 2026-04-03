// nn/batchnorm.c — Batch normalization (pure IC Term composition)
// Computes mean/var independently for each use to avoid DUP shape issues.

typedef struct { Term output; Term assigns; } BNResult;

// Helper: compute batch mean as [1,C,1,1]-shaped tensor (avoids extra reshape)
static Term bn_batch_mean_4d(TinyHVM *ctx, Term x, u32 B, u32 C, u32 H, u32 W) {
    u32 count = B * H * W;
    f32 inv_count = 1.0f / (f32)count;
    // mean = sum(x, axis=0,2,3) / count → shape [1,C,1,1]
    // Use SUM over axes [0,2,3]
    Term sum_bhw = thvm_sum_axes(ctx, x, (u32[]){0,2,3}, 3);
    // sum_bhw has shape [1,C,1,1]
    Term inv_t = thvm_expand(ctx,
        thvm_reshape(ctx, thvm_tensor(ctx, &inv_count, SHAPE(1)), SHAPE(1,1,1,1)),
        (Shape){.dims={1,C,1,1},.rank=4});
    return thvm_op(ctx, UOP_MUL, sum_bhw, inv_t);
}

// Helper: compute batch var as [1,C,1,1]-shaped tensor
static Term bn_batch_var_4d(TinyHVM *ctx, Term x, Term mean_4d, u32 B, u32 C, u32 H, u32 W) {
    u32 count = B * H * W;
    f32 inv_count = 1.0f / (f32)count;
    // mean_4d is [1,C,1,1], broadcast to [B,C,H,W]
    Term mean_bc = thvm_expand(ctx, mean_4d, (Shape){.dims={B,C,H,W},.rank=4});
    Term y = thvm_op(ctx, UOP_SUB, x, mean_bc);
    Term y2 = thvm_op(ctx, UOP_MUL, y, y);
    Term var_sum = thvm_sum_axes(ctx, y2, (u32[]){0,2,3}, 3); // [1,C,1,1]
    Term inv_t = thvm_expand(ctx,
        thvm_reshape(ctx, thvm_tensor(ctx, &inv_count, SHAPE(1)), SHAPE(1,1,1,1)),
        (Shape){.dims={1,C,1,1},.rank=4});
    return thvm_op(ctx, UOP_MUL, var_sum, inv_t);
}

static BNResult batchnorm_forward(TinyHVM *ctx, Term x,
                            Term gamma, Term beta, Term rmean, Term rvar,
                            u32 B, u32 C, u32 H, u32 W, int training) {
    f32 eps_val = 1e-5f;
    u32 count = B * H * W;

    if (training) {
        // All computations in [1,C,1,1] shape — no reshape to [C], avoids DUP shape issues.
        // Keep lazy so GRAD can walk through mean/var → x for full BN gradient.
        Term mean_4d = bn_batch_mean_4d(ctx, x, B, C, H, W);
        Term var_4d = bn_batch_var_4d(ctx, x, mean_4d, B, C, H, W);

        // Running stats update: rmean/rvar are [C], need reshape for compat
        f32 mom = 0.1f, bessel = (f32)count / (f32)(count - 1);
        f32 omm = 1.0f - mom, mb = mom * bessel;
        Term mean_1d = thvm_reshape(ctx, mean_4d, SHAPE(C));
        Term var_1d = thvm_reshape(ctx, var_4d, SHAPE(C));
        Term new_rm = thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MUL, rmean, thvm_tensor(ctx, &omm, SHAPE(1))),
            thvm_op(ctx, UOP_MUL, mean_1d, thvm_tensor(ctx, &mom, SHAPE(1))));
        Term new_rv = thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MUL, rvar, thvm_tensor(ctx, &omm, SHAPE(1))),
            thvm_op(ctx, UOP_MUL, var_1d, thvm_tensor(ctx, &mb, SHAPE(1))));

        // Output: (x - mean) * invstd * gamma + beta — all in [B,C,H,W]
        Term eps_4d = thvm_expand(ctx, thvm_tensor(ctx, &eps_val, SHAPE(1,1,1,1)),
            (Shape){.dims={1,C,1,1},.rank=4});
        Term invstd_4d = thvm_op(ctx, UOP_DIV,
            thvm_tensor(ctx, &(f32){1.0f}, SHAPE(1)),
            thvm_op(ctx, UOP_SQRT, thvm_op(ctx, UOP_ADD, var_4d, eps_4d), term_era()));
        Term mean_bc = thvm_expand(ctx, mean_4d, (Shape){.dims={B,C,H,W},.rank=4});
        Term invstd_bc = thvm_expand(ctx, invstd_4d, (Shape){.dims={B,C,H,W},.rank=4});
        Term gamma_bc = thvm_expand(ctx, thvm_reshape(ctx, gamma, SHAPE(1,C,1,1)),
            (Shape){.dims={B,C,H,W},.rank=4});
        Term beta_bc = thvm_expand(ctx, thvm_reshape(ctx, beta, SHAPE(1,C,1,1)),
            (Shape){.dims={B,C,H,W},.rank=4});
        Term out = thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MUL, thvm_op(ctx, UOP_MUL,
                thvm_op(ctx, UOP_SUB, x, mean_bc), invstd_bc), gamma_bc),
            beta_bc);

        // Wrap each ASSIGN in APP(ASSIGN, ERA) so that ASSIGN enters
        // as the fun of an APP (trampoline reduces args) rather than
        // the arg (inet_step skips arg reduction).
        Term a_rm = thvm_app(ctx, thvm_assign(ctx, rmean, new_rm), term_era());
        Term a_rv = thvm_app(ctx, thvm_assign(ctx, rvar, new_rv), term_era());
        return (BNResult){
            .output = out,
            .assigns = thvm_app(ctx, a_rm, a_rv),
        };
    }

    // Eval mode
    Term mean_bc = thvm_expand(ctx, thvm_reshape(ctx, rmean, SHAPE(1,C,1,1)),
        (Shape){.dims={B,C,H,W},.rank=4});
    Term eps_bc = thvm_expand(ctx, thvm_tensor(ctx, &eps_val, SHAPE(1)), SHAPE(C));
    Term invstd = thvm_op(ctx, UOP_DIV,
        thvm_tensor(ctx, &(f32){1.0f}, SHAPE(1)),
        thvm_op(ctx, UOP_SQRT, thvm_op(ctx, UOP_ADD, rvar, eps_bc), term_era()));
    Term invstd_bc = thvm_expand(ctx, thvm_reshape(ctx, invstd, SHAPE(1,C,1,1)),
        (Shape){.dims={B,C,H,W},.rank=4});
    Term gamma_bc = thvm_expand(ctx, thvm_reshape(ctx, gamma, SHAPE(1,C,1,1)),
        (Shape){.dims={B,C,H,W},.rank=4});
    Term beta_bc = thvm_expand(ctx, thvm_reshape(ctx, beta, SHAPE(1,C,1,1)),
        (Shape){.dims={B,C,H,W},.rank=4});
    return (BNResult){
        .output = thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MUL, thvm_op(ctx, UOP_MUL,
                thvm_op(ctx, UOP_SUB, x, mean_bc), invstd_bc), gamma_bc),
            beta_bc),
        .assigns = term_era()
    };
}

// Compat wrapper for sequential.c
static Term batchnorm_term(TinyHVM *ctx, Term x,
                            Term gamma, Term beta, Term rmean, Term rvar,
                            u32 B, u32 C, u32 H, u32 W, int training) {
    BNResult r = batchnorm_forward(ctx, x, gamma, beta, rmean, rvar, B, C, H, W, training);
    if (training) return thvm_app(ctx, r.assigns, r.output);
    return r.output;
}
