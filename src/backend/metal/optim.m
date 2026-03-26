// metal/optim.m — Optimizer and pooling GPU kernels

static void metal_op_maxpool_fwd(u32 out, u32 mask, u32 src, u32 B, u32 C, u32 H, u32 W) {
    u32 OH = H / 2, OW = W / 2;
    LayoutParams lp = {B, C, H, W};
    id<MTLBuffer> bufs[] = { metal_pool.bufs[out], metal_pool.bufs[mask], metal_pool.bufs[src] };
    const void *params[] = { &lp };
    u64 psizes[] = { sizeof(LayoutParams) };
    dc_tag=DC_MAXPOOL; dispatch_1d(pipe_maxpool2d_fwd, bufs, 3, params, psizes, 1, B * C * OH * OW);
}

static void metal_op_maxpool_bwd(u32 dx, u32 dout, u32 mask, u32 B, u32 C, u32 H, u32 W) {
    u32 OH = H / 2, OW = W / 2;
    LayoutParams lp = {B, C, H, W};
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dx], metal_pool.bufs[dout], metal_pool.bufs[mask] };
    const void *params[] = { &lp };
    u64 psizes[] = { sizeof(LayoutParams) };
    dc_tag=DC_MAXPOOL; dispatch_1d(pipe_maxpool2d_bwd, bufs, 3, params, psizes, 1, B * C * OH * OW);
}

static void metal_op_relu_bwd(u32 dx, u32 dout, u32 x, u32 n) {
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dx], metal_pool.bufs[dout], metal_pool.bufs[x] };
    dispatch_1d(pipe_relu_bwd, bufs, 3, NULL, NULL, 0, n);
}

static void metal_op_zero_fill(u32 buf, u32 n) {
    id<MTLBuffer> bufs_[] = { metal_pool.bufs[buf] };
    dispatch_1d(pipe_zero_fill, bufs_, 1, NULL, NULL, 0, n);
}

static void metal_op_adam_step(u32 param, u32 grad, u32 m, u32 v,
                                f32 lr, f32 beta1, f32 beta2, f32 eps,
                                f32 bc1, f32 bc2, u32 n) {
    typedef struct { f32 lr, beta1, beta2, eps, bc1, bc2; } AP;
    AP ap = {lr, beta1, beta2, eps, bc1, bc2};
    id<MTLBuffer> bufs[] = { metal_pool.bufs[param], metal_pool.bufs[grad],
                              metal_pool.bufs[m], metal_pool.bufs[v] };
    const void *params[] = { &ap };
    u64 psizes[] = { sizeof(AP) };
    if (n % 4 == 0) { dc_tag=DC_ADAM; dispatch_1d(pipe_adam_step_f4, bufs, 4, params, psizes, 1, n / 4); } else { dc_tag=DC_ADAM; dispatch_1d(pipe_adam_step, bufs, 4, params, psizes, 1, n); }
}
