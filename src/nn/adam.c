// nn/adam.c — Adam optimizer (via UOp composition + ASSIGN)
//
// Adam update expressed as lazy UOp graph:
//   m = beta1 * m + (1 - beta1) * grad
//   v = beta2 * v + (1 - beta2) * grad^2
//   param -= lr * (m / bc1) / (sqrt(v / bc2) + eps)
//
// The fuser merges this into efficient kernels; ASSIGN blits results
// back into the original param/m/v buffers.

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

    Term t_beta1 = thvm_scalar(ctx, opt->beta1);
    Term t_beta2 = thvm_scalar(ctx, opt->beta2);
    Term t_1mb1  = thvm_scalar(ctx, 1.0f - opt->beta1);
    Term t_1mb2  = thvm_scalar(ctx, 1.0f - opt->beta2);
    Term t_lr    = thvm_scalar(ctx, opt->lr);
    Term t_bc1   = thvm_scalar(ctx, bc1);
    Term t_bc2   = thvm_scalar(ctx, bc2);
    Term t_eps   = thvm_scalar(ctx, opt->eps);

    for (u32 i = 0; i < opt->n_params; i++) {
        if (grad_ids[i]) ENSURE(ctx, grad_ids[i]);

        Term param = term_new(TAG_TEN, 0, opt->param_ids[i]);
        Term grad  = term_new(TAG_TEN, 0, grad_ids[i]);
        Term m     = term_new(TAG_TEN, 0, opt->m_bufs[i]);
        Term v     = term_new(TAG_TEN, 0, opt->v_bufs[i]);

        // m_new = beta1 * m + (1 - beta1) * grad
        Term m_new = thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MUL, t_beta1, m),
            thvm_op(ctx, UOP_MUL, t_1mb1, grad));

        // v_new = beta2 * v + (1 - beta2) * grad^2
        Term v_new = thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MUL, t_beta2, v),
            thvm_op(ctx, UOP_MUL, t_1mb2, thvm_op(ctx, UOP_MUL, grad, grad)));

        // param_new = param - lr * (m_new / bc1) / (sqrt(v_new / bc2) + eps)
        Term m_hat = thvm_op(ctx, UOP_DIV, m_new, t_bc1);
        Term v_hat = thvm_op(ctx, UOP_DIV, v_new, t_bc2);
        Term denom = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_SQRT, v_hat, term_era()), t_eps);
        Term step  = thvm_op(ctx, UOP_MUL, t_lr, thvm_op(ctx, UOP_DIV, m_hat, denom));
        Term p_new = thvm_op(ctx, UOP_SUB, param, step);

        // ASSIGN blits computed values back into original buffers
        thvm_reduce(ctx, thvm_op(ctx, UOP_ASSIGN, m, m_new));
        thvm_reduce(ctx, thvm_op(ctx, UOP_ASSIGN, v, v_new));
        thvm_reduce(ctx, thvm_op(ctx, UOP_ASSIGN, param, p_new));
    }
}

static void adam_free(Adam *opt) {
    free(opt->param_ids); free(opt->param_sizes);
    free(opt->m_bufs); free(opt->v_bufs);
}
