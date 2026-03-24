// metal/ops.m — Core Metal compute ops: unary, binary, matmul, reduce

static void metal_op_unary(u32 uop, u32 dst, const View *dv,
                            u32 src, const View *sv) {
    u64 t0 = thvm_prof_tick();
    id<MTLComputePipelineState> pipe = nil;
    switch (uop) {
        case UOP_NEG:  pipe = pipe_neg;  break;
        case UOP_RELU: pipe = pipe_relu; break;
        case UOP_EXP:  pipe = pipe_exp;  break;
        case UOP_LOG:  pipe = pipe_log;  break;
        case UOP_SQRT: pipe = pipe_sqrt; break;
        default: return;
    }

    ViewParams dvp = view_to_params(dv);
    ViewParams svp = view_to_params(sv);

    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &dvp, &svp };
    u64 psizes[] = { sizeof(ViewParams), sizeof(ViewParams) };
    dispatch_1d(pipe, bufs, 2, params, psizes, 2, dv->numel);
    thvm_prof_record(uop, t0);
}

static void metal_op_binary(u32 uop, u32 dst, const View *dv,
                             u32 a, const View *av, u32 b, const View *bv) {
    u64 t0 = thvm_prof_tick();
    id<MTLComputePipelineState> pipe = nil;
    switch (uop) {
        case UOP_ADD: pipe = pipe_add; break;
        case UOP_MUL: pipe = pipe_mul; break;
        case UOP_SUB: pipe = pipe_sub; break;
        case UOP_DIV: pipe = pipe_div; break;
        case UOP_MAX: pipe = pipe_max; break;
        case UOP_CMP: pipe = pipe_cmp; break;
        default: return;
    }

    ViewParams dvp = view_to_params(dv);
    ViewParams avp = view_to_params(av);
    ViewParams bvp = view_to_params(bv);

    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[a], metal_pool.bufs[b] };
    const void *params[] = { &dvp, &avp, &bvp };
    u64 psizes[] = { sizeof(ViewParams), sizeof(ViewParams), sizeof(ViewParams) };
    dispatch_1d(pipe, bufs, 3, params, psizes, 3, dv->numel);
    thvm_prof_record(uop, t0);
}

static void metal_op_mm(u32 dst, u32 a, const View *av, u32 b, const View *bv,
                         u32 M, u32 K, u32 N) {
    u64 t0 = thvm_prof_tick();
    if (batch_dirty) metal_flush();

    id<MTLBuffer> buf_a = metal_pool.bufs[a];
    id<MTLBuffer> buf_b = metal_pool.bufs[b];
    id<MTLBuffer> tmp_a = nil, tmp_b = nil;

    if (!av->contiguous) {
        u32 n = M * K;
        tmp_a = [mtl_dev newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
        float *src = (float *)buf_a.contents;
        float *dst_ptr = (float *)tmp_a.contents;
        for (u32 i = 0; i < n; i++) {
            u32 idx = (u32)av->offset, rem = i;
            for (i32 d = (i32)av->shape.rank - 1; d >= 0; d--) {
                u32 coord = rem % av->shape.dims[d];
                rem /= av->shape.dims[d];
                idx += coord * (u32)av->strides[d];
            }
            dst_ptr[i] = src[idx];
        }
        buf_a = tmp_a;
    }

    if (!bv->contiguous) {
        u32 n = K * N;
        tmp_b = [mtl_dev newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
        float *src = (float *)buf_b.contents;
        float *dst_ptr = (float *)tmp_b.contents;
        for (u32 i = 0; i < n; i++) {
            u32 idx = (u32)bv->offset, rem = i;
            for (i32 d = (i32)bv->shape.rank - 1; d >= 0; d--) {
                u32 coord = rem % bv->shape.dims[d];
                rem /= bv->shape.dims[d];
                idx += coord * (u32)bv->strides[d];
            }
            dst_ptr[i] = src[idx];
        }
        buf_b = tmp_b;
    }

    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M columns:K rowBytes:K*sizeof(float)
        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:K columns:N rowBytes:N*sizeof(float)
        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M columns:N rowBytes:N*sizeof(float)
        dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:buf_a descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:buf_b descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[dst] descriptor:descC];

    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:mtl_dev transposeLeft:NO transposeRight:NO
        resultRows:M resultColumns:N interiorColumns:K
        alpha:1.0 beta:0.0];

    id<MTLCommandBuffer> cmd = [mtl_queue commandBuffer];
    [mm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [cmd commit];

    if (!batch_active) {
        [cmd waitUntilCompleted];
    } else {
        [cmd waitUntilCompleted];
    }

    tmp_a = nil;
    tmp_b = nil;
    thvm_prof_record(UOP_MM, t0);
}

static void metal_op_reduce(u32 uop, u32 dst, u32 dst_numel,
                             u32 src, u32 src_numel, u32 reduce_dim) {
    u64 t0 = thvm_prof_tick();
    (void)src_numel;
    id<MTLComputePipelineState> pipe = nil;
    switch (uop) {
        case UOP_SUM:  pipe = pipe_reduce_sum; break;
        case UOP_RMAX: pipe = pipe_reduce_max; break;
        default: return;
    }

    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &reduce_dim };
    u64 psizes[] = { sizeof(u32) };
    dispatch_1d(pipe, bufs, 2, params, psizes, 1, dst_numel);
    thvm_prof_record(uop, t0);
}
