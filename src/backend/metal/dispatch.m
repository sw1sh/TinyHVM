// metal/dispatch.m — GPU dispatch helper

static u32 total_dispatches = 0;

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
}
