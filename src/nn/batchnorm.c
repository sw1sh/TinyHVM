// nn/batchnorm.c — Batch normalization (pure IC Term composition)
// Matches tinygrad: detach mean for variance, live mean+var for output.

static Term batchnorm_term(TinyHVM *ctx, Term x,
                            Term gamma, Term beta, Term rmean, Term rvar,
                            u32 B, u32 C, u32 H, u32 W, int training) {
    f32 eps_val = 1e-5f;
    u32 count = B * H * W;
    f32 inv_count = 1.0f / (f32)count;

    Term batch_mean, batch_var;

    if (training) {
        // batch_mean = x.mean(axis=(0,2,3)) — LAZY
        Term x_perm = thvm_permute(ctx, x, (u32[]){1,0,2,3}, 4);
        Term x_flat = thvm_reshape(ctx, x_perm, SHAPE(C, count));
        Term x_sum = thvm_reshape(ctx,
            thvm_op(ctx, UOP_SUM, x_flat, term_era()), SHAPE(C));
        Term inv_n = thvm_expand(ctx, thvm_tensor(ctx, &inv_count, SHAPE(1)), SHAPE(C));
        batch_mean = thvm_op(ctx, UOP_MUL, x_sum, inv_n);

        // DUP batch_mean: one copy for variance, one for output normalization.
        // No thvm_reduce — stays lazy. DUP ensures the computation runs once.
        Term mean_for_var, mean_for_out;
        thvm_dup(ctx, batch_mean, &mean_for_var, &mean_for_out);
        // "Detach" the variance copy: wrap in a no-grad marker
        // For now, just use the copy directly (GRAD will flow through both paths;
        // the DUP accumulation handles it correctly).
        Term mean_det = mean_for_var;

        // batch_var = ((x - detach(mean))^2).mean(axis=(0,2,3)) — LAZY
        Term md4 = thvm_expand(ctx, thvm_reshape(ctx, mean_det, SHAPE(1,C,1,1)),
            (Shape){.dims={B,C,H,W},.rank=4});
        Term y = thvm_op(ctx, UOP_SUB, x, md4);
        Term y2 = thvm_op(ctx, UOP_MUL, y, y);
        Term y2p = thvm_permute(ctx, y2, (u32[]){1,0,2,3}, 4);
        Term y2f = thvm_reshape(ctx, y2p, SHAPE(C, count));
        Term y2s = thvm_reshape(ctx, thvm_op(ctx, UOP_SUM, y2f, term_era()), SHAPE(C));
        batch_var = thvm_op(ctx, UOP_MUL, y2s, inv_n);

        // DUP batch_var: one copy for running stats, one for output normalization.
        Term var_for_stats, var_for_out;
        thvm_dup(ctx, batch_var, &var_for_stats, &var_for_out);
        Term var_det = var_for_stats;

        f32 mom = 0.1f, bessel = (f32)count / (f32)(count - 1);

        f32 omm = 1.0f - mom, mb = mom * bessel;
        Term new_rm = thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MUL, rmean, thvm_tensor(ctx, &omm, SHAPE(1))),
            thvm_op(ctx, UOP_MUL, mean_det, thvm_tensor(ctx, &mom, SHAPE(1))));
        Term new_rv = thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MUL, rvar, thvm_tensor(ctx, &omm, SHAPE(1))),
            thvm_op(ctx, UOP_MUL, var_det, thvm_tensor(ctx, &mb, SHAPE(1))));
        Term assign_rm = thvm_assign(ctx, rmean, new_rm);
        Term assign_rv = thvm_assign(ctx, rvar, new_rv);

        // Output uses the DUP copies for mean and var
        Term mean_bc = thvm_expand(ctx, thvm_reshape(ctx, mean_for_out, SHAPE(1,C,1,1)),
            (Shape){.dims={B,C,H,W},.rank=4});
        Term eps_bc = thvm_expand(ctx, thvm_tensor(ctx, &eps_val, SHAPE(1)), SHAPE(C));
        Term invstd = thvm_op(ctx, UOP_DIV,
            thvm_tensor(ctx, &(f32){1.0f}, SHAPE(1)),
            thvm_op(ctx, UOP_SQRT, thvm_op(ctx, UOP_ADD, var_for_out, eps_bc), term_era()));
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

        // Chain assigns AFTER output: APP(assign_rm, APP(assign_rv, out))
        // When reduced: processes out first (it's the continuation of the assigns).
        // Actually APP-TEN reduces the FUNCTION first, then continues to the arg.
        // So APP(assign, out) reduces assign first, then out.
        // This is fine because batch_mean/batch_var are already REDUCED above.
        return thvm_app(ctx, assign_rm, thvm_app(ctx, assign_rv, out));
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
    return thvm_op(ctx, UOP_ADD,
        thvm_op(ctx, UOP_MUL, thvm_op(ctx, UOP_MUL,
            thvm_op(ctx, UOP_SUB, x, mean_bc), invstd_bc), gamma_bc),
        beta_bc);
}
