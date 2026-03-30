// metal/ops.m — Core Metal compute ops: unary, binary, matmul, reduce

// Forward declarations (defined in fused.m/codegen.m, included after ops.m)
void metal_dispatch_mdim_binary(u32 uop, u32 dst, const View *dv,
                                 u32 a_buf, const View *av,
                                 u32 b_buf, const View *bv);
void metal_dispatch_kernel(u32 out_buf, u32 out_numel,
                            u32 *leaf_bufs, const View **leaf_views, u32 n_leaves,
                            FusedOp *ops, u32 n_ops, int has_reduce, u32 reduce_dim,
                            const Shape *out_shape_hint);

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

    // Unified codegen path
    if (!sv->has_mask && sv->shape.rank <= 8 && sv->shape.rank > 0) {
        FusedOp op = { .uop = uop, .arg_a = 0, .arg_b = 0 };
        u32 leaf_bufs[] = { src };
        const View *lvs[] = { sv };
        metal_dispatch_kernel(dst, dv->numel, leaf_bufs, lvs, 1, &op, 1, 0, 0, &dv->shape);
        thvm_prof_record(uop, t0);
        return;
    }

    // Legacy fallback
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
                u32 ids[] = {dst, src};
                const void *params[] = { &n4 };
                u64 psizes[] = { sizeof(u32) };
                dispatch_1d_ids(f4p, ids, 2, params, psizes, 1, n4);
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
            u32 ids[] = {dst, src};
            const void *params[] = { &n };
            u64 psizes[] = { sizeof(u32) };
            dispatch_1d_ids(fpipe, ids, 2, params, psizes, 1, n);
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
    u32 ids[] = {dst, src};
    const void *params[] = { &dvp, &svp };
    u64 psizes[] = { sizeof(ViewParams), sizeof(ViewParams) };
    dc_tag=DC_SLOW_UN; dispatch_1d_ids(pipe, ids, 2, params, psizes, 2, dv->numel);
    thvm_prof_record(uop, t0);
}

static void metal_op_binary(u32 uop, u32 dst, const View *dv,
                             u32 a, const View *av, u32 b, const View *bv) {
    u64 t0 = thvm_prof_tick();

    // Unified codegen: JIT-generates optimal kernel for any stride pattern
    if (av->shape.rank == bv->shape.rank && !av->has_mask && !bv->has_mask &&
        av->shape.rank <= 8 && av->shape.rank > 0) {
        FusedOp op = { .uop = uop, .arg_a = 0, .arg_b = 1 };
        u32 leaf_bufs[] = { a, b };
        const View *lvs[] = { av, bv };
        metal_dispatch_kernel(dst, dv->numel, leaf_bufs, lvs, 2, &op, 1, 0, 0, &dv->shape);
        thvm_prof_record(uop, t0);
        return;
    }

    // Legacy fallback for rank mismatch or masks
    {
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
                u32 ids[] = {dst, a, b};
                const void *params[] = { &n4 };
                u64 psizes[] = { sizeof(u32) };
                dc_tag=DC_F4_BIN; dispatch_1d_ids(f4p, ids, 3, params, psizes, 1, n4);
                thvm_prof_record(uop, t0);
                return;
            }
        }
        // Scalar fast path
        u32 ids[] = {dst, a, b};
        const void *params[] = { &n };
        u64 psizes[] = { sizeof(u32) };
        dc_tag=DC_FAST_BIN; dispatch_1d_ids(fpipe, ids, 3, params, psizes, 1, n);
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
                    u32 ids[] = {dst, contig_buf, bcast_buf};
                    const void *params[] = { &n_spatial, &n_bc, &C_dim };
                    u64 psizes[] = { sizeof(u32), sizeof(u32), sizeof(u32) };
                    bc2d_count++;
                    // dispatch_2d still uses legacy path — convert inline
                    id<MTLComputeCommandEncoder> enc = get_encoder();
                    [enc setComputePipelineState:bc_pipe];
                    for(u32 ii=0;ii<3;ii++) [enc setBuffer:metal_pool.bufs[ids[ii]] offset:0 atIndex:ii];
                    for(u32 ii=0;ii<3;ii++) [enc setBytes:params[ii] length:psizes[ii] atIndex:3+ii];
                    NSUInteger tpg=bc_pipe.maxTotalThreadsPerThreadgroup;
                    NSUInteger tw=MIN(32,n_spatial),th=MIN(tpg/tw,(NSUInteger)n_bc);
                    [enc dispatchThreads:MTLSizeMake(n_spatial,n_bc,1) threadsPerThreadgroup:MTLSizeMake(tw,th,1)];
                    batch_dirty=1;total_dispatches++;dc[DC_BC2D]++;dc_tag=DC_OTHER;
                    if(jit.state==JIT_CAPTURE) jit_record_dispatch_ids(bc_pipe,ids,3,params,psizes,3,n_spatial,n_bc,1,(u32)tw,(u32)th,1);
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
    u32 ids[] = {dst, a, b};
    const void *params[] = { &dvp, &avp, &bvp };
    u64 psizes[] = { sizeof(ViewParams), sizeof(ViewParams), sizeof(ViewParams) };
    dc_tag=DC_SLOW_BIN; dispatch_1d_ids(pipe, ids, 3, params, psizes, 3, dv->numel);
    thvm_prof_record(uop, t0);
    } // end legacy fallback
}

static void metal_op_mm(u32 dst, u32 a, const View *av, u32 b, const View *bv,
                         u32 M, u32 K, u32 N) {
    u64 t0 = thvm_prof_tick();

    // Detect simple transposes: shape=[R,C] strides=[1,R] (swapped from contiguous)
    // Use MPS transpose flag instead of contiguifying.
    BOOL transposeA = NO, transposeB = NO;
    u32 buf_a_id = a, buf_b_id = b;

    if (!av->contiguous && av->shape.rank == 2 &&
        av->strides[0] == 1 && av->strides[1] == (i32)av->shape.dims[0] && av->offset == 0) {
        // A is transposed: physical layout is [K, M], MPS reads as [K, M]^T = [M, K]
        transposeA = YES;
    } else if (!av->contiguous) {
        u32 n = M * K;
        u32 tmp = metal_buf_alloc(n * sizeof(float));
        metal_contiguify(tmp, n, a, av);
        buf_a_id = tmp;
    }

    if (!bv->contiguous && bv->shape.rank == 2 &&
        bv->strides[0] == 1 && bv->strides[1] == (i32)bv->shape.dims[0] && bv->offset == 0) {
        transposeB = YES;
    } else if (!bv->contiguous) {
        u32 n = K * N;
        u32 tmp = metal_buf_alloc(n * sizeof(float));
        metal_contiguify(tmp, n, b, bv);
        buf_b_id = tmp;
    }

    if (batch_encoder) { [batch_encoder endEncoding]; batch_encoder = nil; }
    if (!batch_cmd) batch_cmd = [mtl_queue commandBuffer];

    // MPS descriptors use physical (storage) layout, not logical layout
    u32 phys_a_rows = transposeA ? K : M;
    u32 phys_a_cols = transposeA ? M : K;
    u32 phys_b_rows = transposeB ? N : K;
    u32 phys_b_cols = transposeB ? K : N;

    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:phys_a_rows columns:phys_a_cols
        rowBytes:phys_a_cols*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:phys_b_rows columns:phys_b_cols
        rowBytes:phys_b_cols*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M columns:N rowBytes:N*sizeof(float)
        dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[buf_a_id] descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[buf_b_id] descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[dst] descriptor:descC];

    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:mtl_dev transposeLeft:transposeA transposeRight:transposeB
        resultRows:M resultColumns:N interiorColumns:K
        alpha:1.0 beta:0.0];

    [mm encodeToCommandBuffer:batch_cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    batch_dirty = 1; buf_cpu_only[dst] = 0; total_dispatches++; dc[DC_MM]++;
    if (jit.state == JIT_CAPTURE)
        jit_record_mps(dst, buf_a_id, buf_b_id, M, K, N,
            transposeA, transposeB, phys_a_rows, phys_a_cols, phys_b_rows, phys_b_cols);
    thvm_prof_record(UOP_MM, t0);
}

static void metal_op_reduce(u32 uop, u32 dst, u32 dst_numel,
                             u32 src, u32 src_numel, u32 reduce_dim) {
    u64 t0 = thvm_prof_tick();
    (void)src_numel;

    // Parallel path: use simd_sum/simd_max with 32 threads per output element
    if (reduce_dim >= 32) {
        id<MTLComputePipelineState> par_pipe = nil;
        switch (uop) {
            case UOP_SUM:  par_pipe = pipe_reduce_sum_par; break;
            case UOP_RMAX: par_pipe = pipe_reduce_max_par; break;
            default: break;
        }
        if (par_pipe) {
            id<MTLComputeCommandEncoder> enc = get_encoder();
            [enc setComputePipelineState:par_pipe];
            [enc setBuffer:metal_pool.bufs[dst] offset:0 atIndex:0];
            [enc setBuffer:metal_pool.bufs[src] offset:0 atIndex:1];
            [enc setBytes:&reduce_dim length:sizeof(u32) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(dst_numel, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            batch_dirty = 1;
            total_dispatches++;
            dc[DC_REDUCE]++;
            if (jit.state == JIT_CAPTURE) {
                u32 ids[] = {dst, src};
                const void *p[] = {&reduce_dim};
                u64 ps[] = {sizeof(u32)};
                // dispatchThreadgroups(N,1,1) tpg(32,1,1) = N*32 total threads
                jit_record_dispatch_ids(par_pipe, ids, 2, p, ps, 1,
                                        dst_numel * 32, 1, 1, 32, 1, 1);
            }
            thvm_prof_record(uop, t0);
            return;
        }
    }

    // Serial fallback (small reduce_dim)
    id<MTLComputePipelineState> pipe = nil;
    switch (uop) {
        case UOP_SUM:  pipe = pipe_reduce_sum; break;
        case UOP_RMAX: pipe = pipe_reduce_max; break;
        default: return;
    }

    u32 rids[] = {dst, src};
    const void *params[] = { &reduce_dim };
    u64 psizes[] = { sizeof(u32) };
    dc_tag=DC_REDUCE; dispatch_1d_ids(pipe, rids, 2, params, psizes, 1, dst_numel);
    thvm_prof_record(uop, t0);
}
