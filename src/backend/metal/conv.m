// metal/conv.m — CNN/layout backend ops (using buf_ids for JIT correctness)

static void metal_op_im2col(u32 dst, u32 src, Conv2dParams p) {
    u32 ids[] = {dst, src};
    const void *params[] = { &p };
    u64 psizes[] = { sizeof(Conv2dParams) };
    dc_tag=DC_IM2COL; dispatch_1d_ids(pipe_im2col, ids, 2, params, psizes, 1, p.n_patches * p.patch_size);
}

static void metal_op_col2im(u32 dst, u32 src, Conv2dParams p) {
    u32 ids[] = {dst, src};
    const void *params[] = { &p };
    u64 psizes[] = { sizeof(Conv2dParams) };
    dispatch_1d_ids(pipe_col2im, ids, 2, params, psizes, 1, p.B * p.Cin * p.H * p.W);
}

static void metal_op_nhwc_to_nchw(u32 dst, u32 src, u32 B, u32 C, u32 H, u32 W) {
    LayoutParams lp = {B, C, H, W};
    u32 ids[] = {dst, src};
    const void *params[] = { &lp };
    u64 psizes[] = { sizeof(LayoutParams) };
    dispatch_1d_ids(pipe_nhwc_to_nchw, ids, 2, params, psizes, 1, B * C * H * W);
}

static void metal_op_nchw_to_nhwc(u32 dst, u32 src, u32 B, u32 C, u32 H, u32 W) {
    LayoutParams lp = {B, C, H, W};
    u32 ids[] = {dst, src};
    const void *params[] = { &lp };
    u64 psizes[] = { sizeof(LayoutParams) };
    dispatch_1d_ids(pipe_nchw_to_nhwc, ids, 2, params, psizes, 1, B * C * H * W);
}

static void metal_op_bias_add(u32 buf, u32 bias, u32 C, u32 n) {
    u32 ids[] = {buf, bias};
    const void *params[] = { &C };
    u64 psizes[] = { sizeof(u32) };
    dispatch_1d_ids(pipe_bias_add, ids, 2, params, psizes, 1, n);
}

static void metal_op_col_sum(u32 dst, u32 src, u32 N, u32 C) {
    u32 ids[] = {dst, src};
    const void *params[] = { &N, &C };
    u64 psizes[] = { sizeof(u32), sizeof(u32) };
    dispatch_1d_ids(pipe_col_sum, ids, 2, params, psizes, 2, C);
}

static void metal_op_transpose(u32 dst, u32 src, u32 M, u32 N) {
    typedef struct { u32 M, N; } TP;
    TP tp = {M, N};
    u32 ids[] = {dst, src};
    const void *params[] = { &tp };
    u64 psizes[] = { sizeof(TP) };
    dispatch_1d_ids(pipe_matrix_transpose, ids, 2, params, psizes, 1, M * N);
}
