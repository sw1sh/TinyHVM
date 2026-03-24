// metal/conv.m — CNN/layout backend ops

static void metal_op_im2col(u32 dst, u32 src, Conv2dParams p) {
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &p };
    u64 psizes[] = { sizeof(Conv2dParams) };
    dispatch_1d(pipe_im2col, bufs, 2, params, psizes, 1, p.n_patches * p.patch_size);
}

static void metal_op_col2im(u32 dst, u32 src, Conv2dParams p) {
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &p };
    u64 psizes[] = { sizeof(Conv2dParams) };
    dispatch_1d(pipe_col2im, bufs, 2, params, psizes, 1, p.B * p.Cin * p.H * p.W);
}

static void metal_op_nhwc_to_nchw(u32 dst, u32 src, u32 B, u32 C, u32 H, u32 W) {
    LayoutParams lp = {B, C, H, W};
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &lp };
    u64 psizes[] = { sizeof(LayoutParams) };
    dispatch_1d(pipe_nhwc_to_nchw, bufs, 2, params, psizes, 1, B * C * H * W);
}

static void metal_op_nchw_to_nhwc(u32 dst, u32 src, u32 B, u32 C, u32 H, u32 W) {
    LayoutParams lp = {B, C, H, W};
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &lp };
    u64 psizes[] = { sizeof(LayoutParams) };
    dispatch_1d(pipe_nchw_to_nhwc, bufs, 2, params, psizes, 1, B * C * H * W);
}

static void metal_op_bias_add(u32 buf, u32 bias, u32 C, u32 n) {
    id<MTLBuffer> bufs[] = { metal_pool.bufs[buf], metal_pool.bufs[bias] };
    const void *params[] = { &C };
    u64 psizes[] = { sizeof(u32) };
    dispatch_1d(pipe_bias_add, bufs, 2, params, psizes, 1, n);
}

static void metal_op_col_sum(u32 dst, u32 src, u32 N, u32 C) {
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &N, &C };
    u64 psizes[] = { sizeof(u32), sizeof(u32) };
    dispatch_1d(pipe_col_sum, bufs, 2, params, psizes, 2, C);
}

static void metal_op_transpose(u32 dst, u32 src, u32 M, u32 N) {
    typedef struct { u32 M, N; } TP;
    TP tp = {M, N};
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &tp };
    u64 psizes[] = { sizeof(TP) };
    dispatch_1d(pipe_matrix_transpose, bufs, 2, params, psizes, 1, M * N);
}
