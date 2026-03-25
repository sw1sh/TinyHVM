// metal/ops.m — Core Metal compute ops: unary, binary, matmul, reduce

static u32 fast_dispatch_count = 0, slow_dispatch_count = 0;

// Check if view is truly contiguous for fast-path (offset=0, standard strides, no mask)
static int is_fast_view(const View *v) {
    if (v->offset != 0 || v->has_mask) return 0;
    i32 expected = 1;
    for (int d = (int)v->shape.rank - 1; d >= 0; d--) {
        if (v->shape.dims[d] > 1) {
            if (v->strides[d] != expected) return 0;
            expected *= (i32)v->shape.dims[d];
        }
    }
    return 1;
}

static void metal_op_unary(u32 uop, u32 dst, const View *dv,
                            u32 src, const View *sv) {
    u64 t0 = thvm_prof_tick();

    // Fast path: contiguous input, direct indexing (no ViewParams overhead)
    if (is_fast_view(sv)) {
        id<MTLComputePipelineState> fpipe = nil;
        switch (uop) {
            case UOP_NEG:  fpipe = fpipe_neg;  break;
            case UOP_RELU: fpipe = fpipe_relu; break;
            case UOP_EXP:  fpipe = fpipe_exp;  break;
            case UOP_LOG:  fpipe = fpipe_log;  break;
            case UOP_SQRT: fpipe = fpipe_sqrt; break;
            default: break;
        }
        if (fpipe) {
            fast_dispatch_count++;
            u32 n = dv->numel;
            id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
            const void *params[] = { &n };
            u64 psizes[] = { sizeof(u32) };
            dispatch_1d(fpipe, bufs, 2, params, psizes, 1, n);
            thvm_prof_record(uop, t0);
            return;
        }
    }
    slow_dispatch_count++;

    // Slow path: strided ViewParams
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

    // Fast path: both inputs contiguous, no broadcast
    if (is_fast_view(av) && is_fast_view(bv) && av->numel == bv->numel) {
        id<MTLComputePipelineState> fpipe = nil;
        switch (uop) {
            case UOP_ADD: fpipe = fpipe_add; break;
            case UOP_MUL: fpipe = fpipe_mul; break;
            case UOP_SUB: fpipe = fpipe_sub; break;
            case UOP_DIV: fpipe = fpipe_div; break;
            case UOP_MAX: fpipe = fpipe_max; break;
            case UOP_CMP: fpipe = fpipe_cmp; break;
            default: break;
        }
        if (fpipe) {
            fast_dispatch_count++;
            u32 n = dv->numel;
            id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[a], metal_pool.bufs[b] };
            const void *params[] = { &n };
            u64 psizes[] = { sizeof(u32) };
            dispatch_1d(fpipe, bufs, 3, params, psizes, 1, n);
            thvm_prof_record(uop, t0);
            return;
        }
    }
    slow_dispatch_count++;

    // Slow path: strided ViewParams with broadcast support
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

    // Contiguify non-contiguous inputs on GPU (no CPU round-trip)
    u32 buf_a_id = a, buf_b_id = b;
    if (!av->contiguous) {
        u32 n = M * K;
        u32 tmp = metal_buf_alloc(n * sizeof(float));
        metal_contiguify(tmp, n, a, av);
        buf_a_id = tmp;
    }
    if (!bv->contiguous) {
        u32 n = K * N;
        u32 tmp = metal_buf_alloc(n * sizeof(float));
        metal_contiguify(tmp, n, b, bv);
        buf_b_id = tmp;
    }

    // End compute encoder (MPS needs its own encoding), keep command buffer
    if (batch_encoder) {
        [batch_encoder endEncoding];
        batch_encoder = nil;
    }

    // Ensure we have a command buffer
    if (!batch_cmd) batch_cmd = [mtl_queue commandBuffer];

    // Encode MPS matmul on the SAME command buffer — no commit, no wait
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M columns:K rowBytes:K*sizeof(float)
        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:K columns:N rowBytes:N*sizeof(float)
        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M columns:N rowBytes:N*sizeof(float)
        dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[buf_a_id] descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[buf_b_id] descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[dst] descriptor:descC];

    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:mtl_dev transposeLeft:NO transposeRight:NO
        resultRows:M resultColumns:N interiorColumns:K
        alpha:1.0 beta:0.0];

    [mm encodeToCommandBuffer:batch_cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    batch_dirty = 1;
    // Next dispatch_1d will lazily create a new compute encoder via get_encoder()

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
