// metal/dispatch.m — GPU dispatch helper

static u32 total_dispatches = 0;

// Per-category dispatch counter (for profiling)
enum { DC_FAST_UN=0, DC_FAST_BIN, DC_F4_UN, DC_F4_BIN, DC_BC2D, DC_MDIM,
       DC_SLOW_UN, DC_SLOW_BIN, DC_MM, DC_REDUCE, DC_FUSED, DC_CONTIGUIFY,
       DC_MRS, DC_IM2COL, DC_CONV, DC_ADAM, DC_MAXPOOL, DC_OTHER, DC_MAX };
static u32 dc[DC_MAX] = {0};
static u32 dc_tag = DC_OTHER;  // set before dispatch

// Core dispatch using buf_ids (preferred — avoids MTLBuffer pointer aliasing)
static void dispatch_1d_ids(id<MTLComputePipelineState> pipe,
                            u32 *buf_ids, u32 n_bufs,
                            const void **params, u64 *param_sizes, u32 n_params,
                            u32 numel) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:pipe];
    for (u32 i = 0; i < n_bufs; i++)
        [enc setBuffer:metal_pool.bufs[buf_ids[i]] offset:BUF_OFFSET(buf_ids[i]) atIndex:i];
    for (u32 i = 0; i < n_params; i++)
        [enc setBytes:params[i] length:param_sizes[i] atIndex:n_bufs + i];
    NSUInteger tpg = MIN(pipe.maxTotalThreadsPerThreadgroup, (NSUInteger)numel);
    [enc dispatchThreads:MTLSizeMake(numel, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    batch_dirty = 1;
    if (n_bufs > 0) buf_cpu_only[buf_ids[0]] = 0; // GPU writes to output buf
    total_dispatches++;
    dc[dc_tag]++; dc_tag = DC_OTHER;
    if (jit.state == JIT_CAPTURE)
        jit_record_dispatch_ids(pipe, buf_ids, n_bufs, params, param_sizes, n_params,
                                numel, 1, 1, (u32)tpg, 1, 1);
}

// Legacy dispatch using MTLBuffer pointers (for callers not yet converted)
static void dispatch_1d(id<MTLComputePipelineState> pipe,
                        id<MTLBuffer> *bufs, u32 n_bufs,
                        const void **params, u64 *param_sizes, u32 n_params,
                        u32 numel) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:pipe];
    for (u32 i = 0; i < n_bufs; i++)
        [enc setBuffer:bufs[i] offset:0 atIndex:i];
    for (u32 i = 0; i < n_params; i++)
        [enc setBytes:params[i] length:param_sizes[i] atIndex:n_bufs + i];
    NSUInteger tpg = MIN(pipe.maxTotalThreadsPerThreadgroup, (NSUInteger)numel);
    [enc dispatchThreads:MTLSizeMake(numel, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    batch_dirty = 1;
    total_dispatches++;
    dc[dc_tag]++; dc_tag = DC_OTHER;
    if (jit.state == JIT_CAPTURE)
        jit_record_dispatch_1d(pipe, bufs, n_bufs, params, param_sizes, n_params,
                                numel, 1, 1, (u32)tpg, 1, 1);
}

static void dispatch_2d(id<MTLComputePipelineState> pipe,
                        id<MTLBuffer> *bufs, u32 n_bufs,
                        const void **params, u64 *param_sizes, u32 n_params,
                        u32 width, u32 height) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:pipe];
    for (u32 i = 0; i < n_bufs; i++)
        [enc setBuffer:bufs[i] offset:0 atIndex:i];
    for (u32 i = 0; i < n_params; i++)
        [enc setBytes:params[i] length:param_sizes[i] atIndex:n_bufs + i];
    NSUInteger tpg = pipe.maxTotalThreadsPerThreadgroup;
    NSUInteger tw = MIN(32, width);
    NSUInteger th = MIN(tpg / tw, (NSUInteger)height);
    [enc dispatchThreads:MTLSizeMake(width, height, 1)
       threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    batch_dirty = 1;
    total_dispatches++;
    dc[dc_tag]++; dc_tag = DC_OTHER;
    if (jit.state == JIT_CAPTURE)
        jit_record_dispatch_1d(pipe, bufs, n_bufs, params, param_sizes, n_params,
                                width, height, 1, (u32)tw, (u32)th, 1);
}

void print_dispatch_breakdown(void) {
    const char *names[] = {"fast_un","fast_bin","f4_un","f4_bin","bc2d","mdim",
        "slow_un","slow_bin","mm","reduce","fused","contiguify",
        "mrs","im2col","conv","adam","maxpool","other"};
    fprintf(stderr, "Dispatch breakdown:");
    for (int i = 0; i < DC_MAX; i++)
        if (dc[i]) fprintf(stderr, " %s=%u", names[i], dc[i]);
    extern u64 metal_bytes_inuse, metal_bytes_total;
    fprintf(stderr, " total=%u mem=%.0fMB/%.0fMB\n", total_dispatches,
        (double)metal_bytes_inuse/1e6, (double)metal_bytes_total/1e6);
    memset(dc, 0, sizeof(dc));
}
