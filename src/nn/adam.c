// nn/adam.c — Adam optimizer (via backend vtable)

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
        if (grad_ids[i]) ENSURE(ctx, grad_ids[i]);
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
