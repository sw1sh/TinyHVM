// metal/ops.m — Core Metal compute ops: unary, binary, matmul, reduce

// Forward declaration (defined in fused.m, included after ops.m)
void metal_dispatch_mdim_binary(u32 uop, u32 dst, const View *dv,
                                 u32 a_buf, const View *av,
                                 u32 b_buf, const View *bv);

static u32 fast_dispatch_count = 0, slow_dispatch_count = 0;
u32 bc2d_count = 0;

// Check if view is truly contiguous (offset=0, standard strides, no mask, no stride-0)
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

// Check if view has stride-0 (broadcast) dims
static int has_broadcast(const View *v) {
    for (u32 d = 0; d < v->shape.rank; d++)
        if (v->shape.dims[d] > 1 && v->strides[d] == 0) return 1;
    return 0;
}

static void metal_op_unary(u32 uop, u32 dst, const View *dv,
                            u32 src, const View *sv) {
    u64 t0 = thvm_prof_tick();

    if (is_fast_view(sv)) {
        u32 n = dv->numel;
        // Float4 vectorized path (4x throughput)
        if ((n % 4) == 0) {
            id<MTLComputePipelineState> f4p = nil;
            switch (uop) {
                case UOP_NEG:  f4p = f4pipe_neg;  break;
                case UOP_RELU: f4p = f4pipe_relu; break;
                default: break;
            }
            if (f4p) {
                u32 n4 = n / 4;
                id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
                const void *params[] = { &n4 };
                u64 psizes[] = { sizeof(u32) };
                dispatch_1d(f4p, bufs, 2, params, psizes, 1, n4);
                thvm_prof_record(uop, t0);
                return;
            }
        }
        // Scalar fast path
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
            id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
            const void *params[] = { &n };
            u64 psizes[] = { sizeof(u32) };
            dispatch_1d(fpipe, bufs, 2, params, psizes, 1, n);
            thvm_prof_record(uop, t0);
            return;
        }
    }

    // Fallback: generic strided ViewParams
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
    id<MTLComputePipelineState> fpipe = nil;
    switch (uop) {
        case UOP_ADD: fpipe = fpipe_add; break;
        case UOP_MUL: fpipe = fpipe_mul; break;
        case UOP_SUB: fpipe = fpipe_sub; break;
        case UOP_DIV: fpipe = fpipe_div; break;
        case UOP_MAX: fpipe = fpipe_max; break;
        case UOP_CMP: fpipe = fpipe_cmp; break;
        default: return;
    }

    // Fast path: both inputs contiguous, same numel
    if (is_fast_view(av) && is_fast_view(bv) && av->numel == bv->numel) {
        u32 n = dv->numel;
        // Float4 vectorized (4x throughput)
        if ((n % 4) == 0) {
            id<MTLComputePipelineState> f4p = nil;
            switch (uop) {
                case UOP_ADD: f4p = f4pipe_add; break;
                case UOP_MUL: f4p = f4pipe_mul; break;
                case UOP_SUB: f4p = f4pipe_sub; break;
                default: break;
            }
            if (f4p) {
                u32 n4 = n / 4;
                id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[a], metal_pool.bufs[b] };
                const void *params[] = { &n4 };
                u64 psizes[] = { sizeof(u32) };
                dispatch_1d(f4p, bufs, 3, params, psizes, 1, n4);
                thvm_prof_record(uop, t0);
                return;
            }
        }
        // Scalar fast path
        id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[a], metal_pool.bufs[b] };
        const void *params[] = { &n };
        u64 psizes[] = { sizeof(u32) };
        dispatch_1d(fpipe, bufs, 3, params, psizes, 1, n);
        thvm_prof_record(uop, t0);
        return;
    }

    // 2D broadcast path: a[B,C,H,W] contiguous, b[B,C,H,W] with strides [0,s,0,0]
    // (or vice versa). Uses 2D dispatch to avoid divisions.
    {
        const View *contig_v = NULL, *bcast_v = NULL;
        u32 contig_buf = 0, bcast_buf = 0;
        int swapped = 0;

        if (is_fast_view(av) && bv->shape.rank >= 3 && !bv->has_mask && bv->offset == 0) {
            contig_v = av; contig_buf = a;
            bcast_v = bv; bcast_buf = b;
        } else if (is_fast_view(bv) && av->shape.rank >= 3 && !av->has_mask && av->offset == 0) {
            contig_v = bv; contig_buf = b;
            bcast_v = av; bcast_buf = a;
            swapped = 1;
        }

        if (contig_v && bcast_v) {
            // Check broadcast pattern: exactly one non-zero stride dim
            u32 rank = bcast_v->shape.rank;
            int bc_dim = -1;
            int bc_ok = 1;
            for (u32 d = 0; d < rank; d++) {
                if (bcast_v->strides[d] != 0 && bcast_v->shape.dims[d] > 1) {
                    if (bc_dim >= 0) { bc_ok = 0; break; } // multiple non-zero strides
                    bc_dim = (int)d;
                }
            }
            if (bc_ok && bc_dim >= 0) {
                // Compute spatial dims (everything after bc_dim)
                u32 n_spatial = 1;
                for (u32 d = (u32)bc_dim + 1; d < rank; d++) n_spatial *= bcast_v->shape.dims[d];
                u32 C_dim = bcast_v->shape.dims[bc_dim];
                u32 n_bc = dv->numel / n_spatial;

                id<MTLComputePipelineState> bc_pipe = nil;
                // For non-commutative ops with swapped inputs, adjust
                if (!swapped || uop == UOP_ADD || uop == UOP_MUL || uop == UOP_MAX) {
                    switch (uop) {
                        case UOP_ADD: bc_pipe = bc2d_add; break;
                        case UOP_MUL: bc_pipe = bc2d_mul; break;
                        case UOP_SUB: if (!swapped) bc_pipe = bc2d_sub; break;
                        case UOP_DIV: if (!swapped) bc_pipe = bc2d_div; break;
                        case UOP_CMP: if (!swapped) bc_pipe = bc2d_cmp; break;
                        default: break;
                    }
                }
                if (bc_pipe) {
                    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[contig_buf], metal_pool.bufs[bcast_buf] };
                    const void *params[] = { &n_spatial, &n_bc, &C_dim };
                    u64 psizes[] = { sizeof(u32), sizeof(u32), sizeof(u32) };
                    bc2d_count++;
                    dispatch_2d(bc_pipe, bufs, 3, params, psizes, 3, n_spatial, n_bc);
                    thvm_prof_record(uop, t0);
                    return;
                }
            }
        }
    }

    // Multi-dim JIT path: generates kernel with no integer divisions
    // Uses 3D dispatch, coordinates → multiply-only index computation
    if (av->shape.rank == bv->shape.rank && !av->has_mask && !bv->has_mask &&
        av->shape.rank <= 8) {
        metal_dispatch_mdim_binary(uop, dst, dv, a, av, b, bv);
        thvm_prof_record(uop, t0);
        return;
    }

    // Fallback: generic strided ViewParams (rank mismatch, masks)
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

    if (batch_encoder) { [batch_encoder endEncoding]; batch_encoder = nil; }
    if (!batch_cmd) batch_cmd = [mtl_queue commandBuffer];

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
