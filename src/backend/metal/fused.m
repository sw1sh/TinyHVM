// metal/fused.m — Fused kernel dispatch: MUL+SUM, and general elementwise+reduce JIT

typedef struct {
    uint32_t n_reduce;
    uint32_t reduce_numel;
    uint32_t reduce_dims[8];
    uint32_t reduce_strides_a[8];
    uint32_t reduce_strides_b[8];
} MulReduceParams;

void metal_mul_reduce_sum(u32 dst, u32 dst_numel,
                          u32 a_buf, const View *av,
                          u32 b_buf, const View *bv,
                          const View *ov,
                          u32 n_reduce,
                          const u32 *reduce_dims,
                          const u32 *reduce_strides_a,
                          const u32 *reduce_strides_b) {
    u64 t0 = thvm_prof_tick();
    ViewParams avp = view_to_params(av);
    ViewParams bvp = view_to_params(bv);
    ViewParams ovp = view_to_params(ov);
    MulReduceParams rp = {0};
    rp.n_reduce = n_reduce;
    rp.reduce_numel = 1;
    for (u32 i = 0; i < n_reduce; i++) {
        rp.reduce_dims[i] = reduce_dims[i];
        rp.reduce_strides_a[i] = reduce_strides_a[i];
        rp.reduce_strides_b[i] = reduce_strides_b[i];
        rp.reduce_numel *= reduce_dims[i];
    }
    id<MTLBuffer> mbufs[] = { metal_pool.bufs[dst], metal_pool.bufs[a_buf], metal_pool.bufs[b_buf] };
    const void *params[] = { &avp, &bvp, &ovp, &rp };
    u64 psizes[] = { sizeof(ViewParams), sizeof(ViewParams), sizeof(ViewParams), sizeof(MulReduceParams) };

    if (rp.reduce_numel >= 32) {
        // Parallel path: one threadgroup (32 threads = 1 SIMD group) per output element
        // Use dispatchThreads with total_threads = dst_numel * 32 so that exactly
        // dst_numel threadgroups are created, each with 32 threads.
        u32 total_threads = dst_numel * 32;
        id<MTLComputeCommandEncoder> enc = get_encoder();
        [enc setComputePipelineState:pipe_mul_reduce_sum_parallel];
        for (u32 i = 0; i < 3; i++)
            [enc setBuffer:mbufs[i] offset:0 atIndex:i];
        for (u32 i = 0; i < 4; i++)
            [enc setBytes:params[i] length:psizes[i] atIndex:3 + i];
        [enc dispatchThreads:MTLSizeMake(total_threads, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        batch_dirty = 1;
        total_dispatches++;
        dc[DC_MRS]++; dc_tag = DC_OTHER;
        if (jit.state == JIT_CAPTURE)
            jit_record_dispatch_1d(pipe_mul_reduce_sum_parallel, mbufs, 3, params, psizes, 4,
                                    total_threads, 1, 1, 32, 1, 1);
    } else {
        // Small reduce: serial path (original kernel)
        dc_tag=DC_MRS; dispatch_1d(pipe_mul_reduce_sum, mbufs, 3, params, psizes, 4, dst_numel);
    }
    thvm_prof_record(UOP_SUM, t0);
}

// ============================================================
// General fused elementwise kernel codegen
// ============================================================
//
// Fuses arbitrary chains of elementwise+optional reduce ops into one kernel.
// Each op in the chain references inputs by index (leaf buffer or prior temp).
// Leaf inputs are read via strided indexing (handles broadcast/permute/expand).
//
// FusedOp: { uop, arg_a, arg_b }
//   arg_a/arg_b: 0..n_leaves-1 = leaf buffer, n_leaves+ = temp var from prior op
//   For unary ops: arg_b is ignored.

// FusedOp defined in tinyhvm.c (before this file is included)

#define FUSED_CACHE_SIZE 64
static struct {
    u64 key;
    id<MTLComputePipelineState> pipe;
} fused_cache[FUSED_CACHE_SIZE];
static u32 fused_cache_count = 0;

static u64 fuse_hash_v2(const FusedOp *ops, u32 n_ops, u32 n_leaves) {
    u64 h = 0xcbf29ce484222325ULL;
    h ^= n_leaves; h *= 0x100000001b3ULL;
    for (u32 i = 0; i < n_ops; i++) {
        h ^= ops[i].uop; h *= 0x100000001b3ULL;
        h ^= ops[i].arg_a; h *= 0x100000001b3ULL;
        h ^= ops[i].arg_b; h *= 0x100000001b3ULL;
    }
    return h;
}

// strided_idx + masked_read: maps flat output index to physical buffer offset using View strides
static const char *strided_idx_helper =
    "inline uint strided_idx(uint flat, constant int *strides, constant uint *shape, int offset, uint rank) {\n"
    "  uint phys = uint(offset);\n"
    "  for (int d = int(rank) - 1; d >= 0; d--) {\n"
    "    phys += (flat % shape[d]) * uint(strides[d] > 0 ? strides[d] : 0);\n"
    "    flat /= shape[d];\n"
    "  }\n"
    "  return phys;\n"
    "}\n"
    "inline float masked_read_v(device const float *buf, uint flat, constant VP &v) {\n"
    "  if (v.has_mask) {\n"
    "    uint rem = flat;\n"
    "    for (int d = int(v.rank) - 1; d >= 0; d--) {\n"
    "      uint c = rem % v.shape[d]; rem /= v.shape[d];\n"
    "      if (c < v.mask_begin[d] || c >= v.mask_end[d]) return 0.0f;\n"
    "    }\n"
    "  }\n"
    "  return buf[strided_idx(flat, v.strides, v.shape, v.offset, v.rank)];\n"
    "}\n";

static NSString *codegen_fused_v2(const FusedOp *ops, u32 n_ops, u32 n_leaves, int has_reduce) {
    NSMutableString *src = [NSMutableString stringWithCapacity:4096];
    [src appendString:@"#include <metal_stdlib>\nusing namespace metal;\n"];

    // ViewParams struct (must match C side)
    [src appendString:@"struct VP { int strides[8]; uint shape[8]; int offset; uint rank; uint numel; uint has_mask; uint mask_begin[8]; uint mask_end[8]; };\n"];
    [src appendFormat:@"%s\n", strided_idx_helper];

    [src appendString:@"kernel void fused_v2(\n"];
    [src appendString:@"  device float *out [[buffer(0)]],\n"];
    for (u32 i = 0; i < n_leaves; i++)
        [src appendFormat:@"  device const float *in%u [[buffer(%u)]],\n", i, i + 1];
    for (u32 i = 0; i < n_leaves; i++)
        [src appendFormat:@"  constant VP &v%u [[buffer(%u)]],\n", i, n_leaves + 1 + i];
    [src appendFormat:@"  constant uint &numel [[buffer(%u)]],\n", 2 * n_leaves + 1];
    if (has_reduce)
        [src appendFormat:@"  constant uint &reduce_dim [[buffer(%u)]],\n", 2 * n_leaves + 2];
    [src appendString:@"  uint gid [[thread_position_in_grid]])\n{\n"];
    [src appendString:@"  if (gid >= numel) return;\n"];

    // Helper macro for ops
    #define EMIT_LEAF_READS(idx_var) \
        for (u32 i = 0; i < n_leaves; i++) \
            [src appendFormat:@"    float t%u = masked_read_v(in%u, %s, v%u);\n", i, i, idx_var, i]
    #define EMIT_OPS(indent) \
        for (u32 i = 0; i < n_ops; i++) { \
            u32 tid = n_leaves + i; \
            u32 a = ops[i].arg_a, b = ops[i].arg_b; \
            switch (ops[i].uop) { \
                case UOP_ADD:  [src appendFormat:@"%@float t%u = t%u + t%u;\n", indent, tid, a, b]; break; \
                case UOP_SUB:  [src appendFormat:@"%@float t%u = t%u - t%u;\n", indent, tid, a, b]; break; \
                case UOP_MUL:  [src appendFormat:@"%@float t%u = t%u * t%u;\n", indent, tid, a, b]; break; \
                case UOP_DIV:  [src appendFormat:@"%@float t%u = t%u / t%u;\n", indent, tid, a, b]; break; \
                case UOP_MAX:  [src appendFormat:@"%@float t%u = max(t%u, t%u);\n", indent, tid, a, b]; break; \
                case UOP_CMP:  [src appendFormat:@"%@float t%u = t%u > t%u ? 1.0f : 0.0f;\n", indent, tid, a, b]; break; \
                case UOP_NEG:  [src appendFormat:@"%@float t%u = -t%u;\n", indent, tid, a]; break; \
                case UOP_RELU: [src appendFormat:@"%@float t%u = max(t%u, 0.0f);\n", indent, tid, a]; break; \
                case UOP_EXP:  [src appendFormat:@"%@float t%u = exp(t%u);\n", indent, tid, a]; break; \
                case UOP_LOG:  [src appendFormat:@"%@float t%u = log(t%u);\n", indent, tid, a]; break; \
                case UOP_SQRT: [src appendFormat:@"%@float t%u = sqrt(t%u);\n", indent, tid, a]; break; \
                default:       [src appendFormat:@"%@float t%u = t%u;\n", indent, tid, a]; break; \
            } \
        }

    if (has_reduce) {
        [src appendString:@"  float acc = 0.0f;\n"];
        [src appendString:@"  for (uint r = 0; r < reduce_dim; r++) {\n"];
        [src appendString:@"    uint idx = gid * reduce_dim + r;\n"];
        EMIT_LEAF_READS("idx");
        EMIT_OPS(@"    ");
        [src appendFormat:@"    acc += t%u;\n", n_leaves + n_ops - 1];
        [src appendString:@"  }\n"];
        [src appendString:@"  out[gid] = acc;\n"];
    } else {
        EMIT_LEAF_READS("gid");
        EMIT_OPS(@"  ");
        [src appendFormat:@"  out[gid] = t%u;\n", n_leaves + n_ops - 1];
    }
    #undef EMIT_LEAF_READS
    #undef EMIT_OPS
    [src appendString:@"}\n"];
    return src;
}

static id<MTLComputePipelineState> get_fused_pipe_v2(const FusedOp *ops, u32 n_ops, u32 n_leaves, int has_reduce) {
    u64 key = fuse_hash_v2(ops, n_ops, n_leaves) ^ ((u64)has_reduce << 63);
    for (u32 i = 0; i < fused_cache_count && i < FUSED_CACHE_SIZE; i++)
        if (fused_cache[i].key == key) return fused_cache[i].pipe;

    NSString *src = codegen_fused_v2(ops, n_ops, n_leaves, has_reduce);
    static u32 jit_compile_count = 0;
    jit_compile_count++;
    if (jit_compile_count <= 30) fprintf(stderr, "JIT compile #%u: n_ops=%u n_leaves=%u reduce=%d\n", jit_compile_count, n_ops, n_leaves, has_reduce);
    NSError *err = nil;
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [mtl_dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { NSLog(@"TinyHVM fused v2 codegen error: %@\n%@", err, src); return nil; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"fused_v2"];
    id<MTLComputePipelineState> pipe = [mtl_dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pipe) { NSLog(@"TinyHVM fused v2 pipeline: %@", err); return nil; }

    u32 slot = fused_cache_count < FUSED_CACHE_SIZE ?
               fused_cache_count++ : ((fused_cache_count++) % FUSED_CACHE_SIZE);
    fused_cache[slot].key = key;
    fused_cache[slot].pipe = pipe;
    return pipe;
}

// GPU contiguify: copy non-contiguous view to contiguous buffer.
// Uses codegen (baked coordinate decomposition) for fast index computation.
void metal_contiguify(u32 dst_buf, u32 numel, u32 src_buf, const View *src_view) {
    u32 leaf_bufs[] = { src_buf };
    const View *leaf_views[] = { src_view };
    dc_tag = DC_CONTIGUIFY;
    metal_dispatch_kernel(dst_buf, numel, leaf_bufs, leaf_views, 1,
                           NULL, 0, 0, 0, &src_view->shape);
}

// Dispatch a fused elementwise kernel.
// leaf_bufs: buffer IDs for leaf inputs. leaf_views: View pointers for each.
// ops: the fused op chain. Result goes to out_buf.
void metal_dispatch_fused_v2(u32 out_buf, u32 out_numel,
                               u32 *leaf_bufs, const View **leaf_views, u32 n_leaves,
                               FusedOp *ops, u32 n_ops,
                               int has_reduce, u32 reduce_dim,
                               const Shape *out_shape) {
    // Unified codegen for non-reduce fused chains (handles masks via codegen)
    if (!has_reduce && n_leaves <= 16 && n_ops <= 32 && out_shape) {
        if (out_shape->rank <= 8 && out_shape->rank > 0) {
            metal_dispatch_kernel(out_buf, out_numel, leaf_bufs, leaf_views, n_leaves,
                                   ops, n_ops, 0, 0, out_shape);
            return;
        }
    }

    // Legacy path for reduce, rank mismatch, or masks
    id<MTLComputePipelineState> pipe = get_fused_pipe_v2(ops, n_ops, n_leaves, has_reduce);
    if (!pipe) return;

    id<MTLBuffer> bufs[16];
    bufs[0] = metal_pool.bufs[out_buf];
    for (u32 i = 0; i < n_leaves && i < 15; i++)
        bufs[i + 1] = metal_pool.bufs[leaf_bufs[i]];

    ViewParams vps[8];
    const void *params[16];
    u64 psizes[16];
    for (u32 i = 0; i < n_leaves; i++) {
        vps[i] = view_to_params(leaf_views[i]);
        params[i] = &vps[i];
        psizes[i] = sizeof(ViewParams);
    }
    params[n_leaves] = &out_numel;
    psizes[n_leaves] = sizeof(u32);
    if (has_reduce) {
        params[n_leaves + 1] = &reduce_dim;
        psizes[n_leaves + 1] = sizeof(u32);
        dc_tag=DC_FUSED; dispatch_1d(pipe, bufs, n_leaves + 1, params, psizes, n_leaves + 2, out_numel);
    } else {
        dc_tag=DC_FUSED; dispatch_1d(pipe, bufs, n_leaves + 1, params, psizes, n_leaves + 1, out_numel);
    }
}

// ============================================================
// Legacy: old unary-only codegen (kept for compatibility)
// ============================================================

static NSString *metal_codegen_fused(const u32 *ops, u32 n_ops, u32 n_inputs,
                                      int has_reduce) {
    NSMutableString *src = [NSMutableString stringWithCapacity:2048];
    [src appendString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
    [src appendString:@"kernel void fused_kernel(\n"];
    [src appendString:@"  device float *out [[buffer(0)]],\n"];
    [src appendString:@"  device const float *in0 [[buffer(1)]],\n"];
    for (u32 i = 1; i < n_inputs; i++)
        [src appendFormat:@"  device const float *in%u [[buffer(%u)]],\n", i, i + 1];
    [src appendFormat:@"  constant uint &reduce_dim [[buffer(%u)]],\n", n_inputs + 1];
    [src appendFormat:@"  constant uint &total_numel [[buffer(%u)]],\n", n_inputs + 2];
    [src appendString:@"  uint gid [[thread_position_in_grid]])\n{\n"];

    if (has_reduce) {
        [src appendString:@"  uint out_count = total_numel / reduce_dim;\n"];
        [src appendString:@"  if (gid >= out_count) return;\n"];
        [src appendString:@"  float acc = 0.0;\n"];
        [src appendString:@"  for (uint j = 0; j < reduce_dim; j++) {\n"];
        [src appendString:@"    uint idx = gid * reduce_dim + j;\n"];
        [src appendString:@"    float v = in0[idx];\n"];
        for (int k = (int)n_ops - 1; k >= 0; k--) {
            switch (ops[k]) {
                case UOP_RELU: [src appendString:@"    v = max(v, 0.0f);\n"]; break;
                case UOP_NEG:  [src appendString:@"    v = -v;\n"]; break;
                case UOP_EXP:  [src appendString:@"    v = exp(v);\n"]; break;
                case UOP_LOG:  [src appendString:@"    v = log(v);\n"]; break;
                case UOP_SQRT: [src appendString:@"    v = sqrt(v);\n"]; break;
                case UOP_SUM: case UOP_RMAX: break;
                default: break;
            }
        }
        int is_max = (ops[0] == UOP_RMAX);
        if (is_max) [src appendString:@"    if (j == 0) acc = v; else acc = max(acc, v);\n"];
        else [src appendString:@"    acc += v;\n"];
        [src appendString:@"  }\n  out[gid] = acc;\n"];
    } else {
        [src appendString:@"  if (gid >= total_numel) return;\n"];
        [src appendString:@"  float v = in0[gid];\n"];
        for (int k = (int)n_ops - 1; k >= 0; k--) {
            switch (ops[k]) {
                case UOP_RELU: [src appendString:@"  v = max(v, 0.0f);\n"]; break;
                case UOP_NEG:  [src appendString:@"  v = -v;\n"]; break;
                case UOP_EXP:  [src appendString:@"  v = exp(v);\n"]; break;
                case UOP_LOG:  [src appendString:@"  v = log(v);\n"]; break;
                case UOP_SQRT: [src appendString:@"  v = sqrt(v);\n"]; break;
                default: break;
            }
        }
        [src appendString:@"  out[gid] = v;\n"];
    }
    [src appendString:@"}\n"];
    return src;
}

static id<MTLComputePipelineState> get_fused_pipe(const u32 *ops, u32 n_ops,
                                                    u32 n_inputs, int has_reduce) {
    u64 key = fuse_hash_v2((FusedOp[]){}, 0, 0) ^ n_ops ^ has_reduce; // legacy hash
    for (u32 i = 0; i < fused_cache_count && i < FUSED_CACHE_SIZE; i++)
        if (fused_cache[i].key == key) return fused_cache[i].pipe;
    NSString *src = metal_codegen_fused(ops, n_ops, n_inputs, has_reduce);
    NSError *err = nil;
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [mtl_dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { NSLog(@"TinyHVM fused codegen: %@\n%@", err, src); return nil; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"fused_kernel"];
    id<MTLComputePipelineState> pipe = [mtl_dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pipe) return nil;
    u32 slot = fused_cache_count < FUSED_CACHE_SIZE ?
               fused_cache_count++ : ((fused_cache_count++) % FUSED_CACHE_SIZE);
    fused_cache[slot].key = key;
    fused_cache[slot].pipe = pipe;
    return pipe;
}

void metal_dispatch_fused(u32 out_buf, u32 *input_bufs, u32 n_inputs,
                           u32 *ops, u32 n_ops, int has_reduce,
                           u32 out_numel, u32 reduce_dim, u32 total_numel) {
    id<MTLComputePipelineState> pipe = get_fused_pipe(ops, n_ops, n_inputs, has_reduce);
    if (!pipe) return;
    id<MTLBuffer> bufs[6];
    bufs[0] = metal_pool.bufs[out_buf];
    for (u32 i = 0; i < n_inputs; i++)
        bufs[i + 1] = metal_pool.bufs[input_bufs[i]];
    const void *params[] = { &reduce_dim, &total_numel };
    u64 psizes[] = { sizeof(u32), sizeof(u32) };
    dispatch_1d(pipe, bufs, n_inputs + 1, params, psizes, 2, out_numel);
}

// ============================================================
// Multi-dimensional dispatch codegen (tinygrad-style)
// No integer divisions — coordinates from hardware grid position.
// ============================================================

// Cache for multi-dim kernels (separate from fused cache)
#define MDIM_CACHE_SIZE 128
static struct { u64 key; id<MTLComputePipelineState> pipe; } mdim_cache[MDIM_CACHE_SIZE];
static u32 mdim_cache_count = 0;

// Hash for multi-dim kernel: includes op, output shape, all input strides
static u64 mdim_hash(u32 uop, const View *dv, const View *av, const View *bv) {
    u64 h = 0xcbf29ce484222325ULL;
    h ^= uop; h *= 0x100000001b3ULL;
    for (u32 d = 0; d < dv->shape.rank; d++) {
        h ^= dv->shape.dims[d]; h *= 0x100000001b3ULL;
    }
    for (u32 d = 0; d < av->shape.rank; d++) {
        h ^= (u64)(u32)av->strides[d]; h *= 0x100000001b3ULL;
    }
    h ^= (u64)av->offset; h *= 0x100000001b3ULL;
    if (bv) {
        for (u32 d = 0; d < bv->shape.rank; d++) {
            h ^= (u64)(u32)bv->strides[d]; h *= 0x100000001b3ULL;
        }
        h ^= (u64)bv->offset; h *= 0x100000001b3ULL;
    }
    return h;
}

static const char *uop_to_op(u32 uop) {
    switch(uop) {
        case UOP_ADD: return "+"; case UOP_SUB: return "-";
        case UOP_MUL: return "*"; case UOP_DIV: return "/";
        default: return "+";
    }
}

// Generate a tinygrad-style kernel for binary op with any stride pattern.
// Uses 3D grid: collapse output dims into (inner, mid, outer).
// Each input's physical index = sum(coord[d] * stride[d]) — multiplies only.
static id<MTLComputePipelineState> get_mdim_binary_pipe(
    u32 uop, const View *dv, const View *av, const View *bv,
    u32 *grid_w, u32 *grid_h, u32 *grid_d,
    u32 *tg_w, u32 *tg_h, u32 *tg_d) {

    u64 key = mdim_hash(uop, dv, av, bv);
    for (u32 i = 0; i < mdim_cache_count && i < MDIM_CACHE_SIZE; i++)
        if (mdim_cache[i].key == key) {
            // Recompute grid dims (same for same key)
            goto compute_grid;
        }

    {
        u32 rank = dv->shape.rank;
        NSMutableString *src = [NSMutableString stringWithCapacity:2048];
        [src appendString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
        [src appendFormat:@"kernel void mdim_bin(\n"
            "  device float *out [[buffer(0)]],\n"
            "  device const float *a [[buffer(1)]],\n"
            "  device const float *b [[buffer(2)]],\n"
            "  uint3 gid [[thread_position_in_grid]])\n{\n"];

        // Collapse dims into 3 groups: outer, mid, inner
        // inner = last dims, mid = middle, outer = first dims
        // Goal: each group is one gid component
        u32 inner = 1, mid = 1, outer = 1;
        u32 inner_start = rank; // dims in inner group: [inner_start, rank)

        // Inner group: rightmost dims up to ~256 elements
        for (int d = (int)rank - 1; d >= 0; d--) {
            if (inner * dv->shape.dims[d] <= 1024) {
                inner *= dv->shape.dims[d];
                inner_start = (u32)d;
            } else break;
        }
        // Mid group: next dims
        u32 mid_start = inner_start;
        for (int d = (int)inner_start - 1; d >= 0; d--) {
            if (mid * dv->shape.dims[d] <= 65535) {
                mid *= dv->shape.dims[d];
                mid_start = (u32)d;
            } else break;
        }
        // Outer: remaining
        for (u32 d = 0; d < mid_start; d++) outer *= dv->shape.dims[d];

        [src appendFormat:@"  uint inner_idx = gid.x; // [0, %u)\n", inner];
        [src appendFormat:@"  uint mid_idx = gid.y;   // [0, %u)\n", mid];
        [src appendFormat:@"  uint outer_idx = gid.z; // [0, %u)\n", outer];
        [src appendFormat:@"  if (inner_idx >= %uu || mid_idx >= %uu || outer_idx >= %uu) return;\n",
            inner, mid, outer];

        // Decompose each group into per-dim coordinates using multiplies only
        // (since the dim sizes are compile-time constants, Metal compiler optimizes)
        [src appendString:@"  // Decompose coordinates\n"];

        // For each dim, compute: coord[d] = (group_idx / divisor) % shape[d]
        // divisor = product of dims below d within the group
        // Since these are compile-time constants, compiler optimizes to shifts/masks
        for (u32 d = 0; d < rank; d++) {
            const char *group;
            u32 group_start, group_end;
            if (d < mid_start) { group = "outer_idx"; group_start = 0; group_end = mid_start; }
            else if (d < inner_start) { group = "mid_idx"; group_start = mid_start; group_end = inner_start; }
            else { group = "inner_idx"; group_start = inner_start; group_end = rank; }

            u32 divisor = 1;
            for (u32 dd = d + 1; dd < group_end; dd++) divisor *= dv->shape.dims[dd];

            if (divisor == 1 && dv->shape.dims[d] == 1) {
                [src appendFormat:@"  uint c%u = 0;\n", d];
            } else if (divisor == 1) {
                [src appendFormat:@"  uint c%u = %s %% %uu;\n", d, group, dv->shape.dims[d]];
            } else {
                [src appendFormat:@"  uint c%u = (%s / %uu) %% %uu;\n", d, group, divisor, dv->shape.dims[d]];
            }
        }

        // Compute physical index for each input: sum(c[d] * stride[d]) + offset
        // Stride=0 dims contribute nothing (broadcast)
        for (int inp = 0; inp < 2; inp++) {
            const View *v = (inp == 0) ? av : bv;
            const char *name = (inp == 0) ? "a" : "b";
            [src appendFormat:@"  uint %s_idx = %d", name, v->offset];
            for (u32 d = 0; d < rank; d++) {
                if (v->strides[d] != 0) {
                    if (v->strides[d] == 1)
                        [src appendFormat:@" + c%u", d];
                    else
                        [src appendFormat:@" + c%u * %du", d, v->strides[d]];
                }
            }
            [src appendString:@";\n"];
        }

        // Output flat index: outer * (mid*inner) + mid * inner + inner
        [src appendFormat:@"  uint out_idx = outer_idx * %uu + mid_idx * %uu + inner_idx;\n",
            mid * inner, inner];

        // The op
        if (uop == UOP_CMP) {
            [src appendString:@"  out[out_idx] = a[a_idx] > b[b_idx] ? 1.0f : 0.0f;\n"];
        } else if (uop == UOP_MAX) {
            [src appendString:@"  out[out_idx] = max(a[a_idx], b[b_idx]);\n"];
        } else {
            [src appendFormat:@"  out[out_idx] = a[a_idx] %s b[b_idx];\n", uop_to_op(uop)];
        }

        [src appendString:@"}\n"];

        NSError *err = nil;
        id<MTLLibrary> lib = [mtl_dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { NSLog(@"mdim codegen error: %@\n%@", err, src); return nil; }
        id<MTLComputePipelineState> pipe = [mtl_dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"mdim_bin"] error:&err];
        if (!pipe) { NSLog(@"mdim pipeline error: %@", err); return nil; }

        u32 slot = mdim_cache_count < MDIM_CACHE_SIZE ?
            mdim_cache_count++ : (mdim_cache_count++ % MDIM_CACHE_SIZE);
        mdim_cache[slot].key = key;
        mdim_cache[slot].pipe = pipe;
    }

compute_grid:;
    // Compute grid dimensions
    u32 rank = dv->shape.rank;
    u32 inner_s = rank, mid_s = 0;
    *grid_w = 1; *grid_h = 1; *grid_d = 1;
    for (int d = (int)rank - 1; d >= 0; d--) {
        if (*grid_w * dv->shape.dims[d] <= 1024) { *grid_w *= dv->shape.dims[d]; inner_s = (u32)d; }
        else break;
    }
    for (int d = (int)inner_s - 1; d >= 0; d--) {
        if (*grid_h * dv->shape.dims[d] <= 65535) { *grid_h *= dv->shape.dims[d]; mid_s = (u32)d; }
        else break;
    }
    for (u32 d = 0; d < mid_s; d++) *grid_d *= dv->shape.dims[d];

    // Threadgroup: use 256 threads (Metal sweet spot)
    *tg_w = MIN(*grid_w, 256u);
    *tg_h = 1; *tg_d = 1;

    // Return the cached pipe
    for (u32 i = 0; i < mdim_cache_count && i < MDIM_CACHE_SIZE; i++)
        if (mdim_cache[i].key == mdim_hash(uop, dv, av, bv))
            return mdim_cache[i].pipe;
    return nil;
}

// Try float4 mdim dispatch. Returns 1 if handled, 0 if caller should use scalar.
static int try_mdim_float4(u32 uop, u32 dst, const View *dv,
                            u32 a_buf, const View *av, u32 b_buf, const View *bv) {
    u32 rank = dv->shape.rank;
    if (rank == 0) return 0;
    i32 a_is = av->strides[rank-1], b_is = bv->strides[rank-1];
    u32 idim = dv->shape.dims[rank-1];
    // Float4: innermost dim divisible by 4, both inputs stride 1 or 0 on innermost, at least one stride 1
    if (idim % 4 != 0) return 0;
    if (a_is != 1 && a_is != 0) return 0;
    if (b_is != 1 && b_is != 0) return 0;
    if (a_is != 1 && b_is != 1) return 0;

    // Build the float4 kernel key
    u64 key = mdim_hash(uop, dv, av, bv) ^ 0xF4F4F4F4ULL;
    id<MTLComputePipelineState> pipe = nil;
    for (u32 i = 0; i < mdim_cache_count && i < MDIM_CACHE_SIZE; i++)
        if (mdim_cache[i].key == key) { pipe = mdim_cache[i].pipe; break; }

    if (!pipe) {
        // Generate float4 kernel: same as scalar mdim but with float4 loads and /4 dispatch
        NSMutableString *src = [NSMutableString stringWithCapacity:2048];
        [src appendString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
        [src appendFormat:@"kernel void mdim_f4(\n"
            "  device float *out [[buffer(0)]],\n"
            "  device const float *a [[buffer(1)]],\n"
            "  device const float *b [[buffer(2)]],\n"
            "  uint3 gid [[thread_position_in_grid]])\n{\n"];

        // Same group collapsing as scalar
        u32 inner = 1, mid = 1, outer = 1;
        u32 inner_start = rank;
        for (int d = (int)rank - 1; d >= 0; d--) {
            if (inner * dv->shape.dims[d] <= 1024) { inner *= dv->shape.dims[d]; inner_start = (u32)d; }
            else break;
        }
        u32 mid_start = inner_start;
        for (int d = (int)inner_start - 1; d >= 0; d--) {
            if (mid * dv->shape.dims[d] <= 65535) { mid *= dv->shape.dims[d]; mid_start = (u32)d; }
            else break;
        }
        for (u32 d = 0; d < mid_start; d++) outer *= dv->shape.dims[d];

        // gid.x covers inner/4, gid.y covers mid, gid.z covers outer
        [src appendFormat:@"  uint raw = gid.x; // [0, %u)\n", inner/4];
        [src appendFormat:@"  uint inner_idx = raw * 4u; // base inner coordinate\n"];
        [src appendFormat:@"  uint mid_idx = gid.y;\n  uint outer_idx = gid.z;\n"];

        // Decompose coordinates (same as scalar)
        for (u32 d = 0; d < rank; d++) {
            const char *group;
            u32 group_end;
            if (d < mid_start) { group = "outer_idx"; group_end = mid_start; }
            else if (d < inner_start) { group = "mid_idx"; group_end = inner_start; }
            else { group = "inner_idx"; group_end = rank; }
            u32 divisor = 1;
            for (u32 dd = d + 1; dd < group_end; dd++) divisor *= dv->shape.dims[dd];
            if (divisor == 1 && dv->shape.dims[d] == 1)
                [src appendFormat:@"  uint c%u = 0;\n", d];
            else if (divisor == 1)
                [src appendFormat:@"  uint c%u = %s %% %uu;\n", d, group, dv->shape.dims[d]];
            else
                [src appendFormat:@"  uint c%u = (%s / %uu) %% %uu;\n", d, group, divisor, dv->shape.dims[d]];
        }

        // Compute base indices
        for (int inp = 0; inp < 2; inp++) {
            const View *v = (inp == 0) ? av : bv;
            const char *nm = (inp == 0) ? "a" : "b";
            [src appendFormat:@"  uint %s_idx = %d", nm, v->offset];
            for (u32 d = 0; d < rank; d++)
                if (v->strides[d] != 0) {
                    if (v->strides[d] == 1) [src appendFormat:@" + c%u", d];
                    else [src appendFormat:@" + c%u * %du", d, v->strides[d]];
                }
            [src appendString:@";\n"];
        }

        // Float4 loads: stride=1 → float4 load, stride=0 → scalar broadcast
        [src appendFormat:@"  float4 va = %s;\n",
            (a_is == 1) ? "*((device const float4*)(a + a_idx))" : "float4(a[a_idx])"];
        [src appendFormat:@"  float4 vb = %s;\n",
            (b_is == 1) ? "*((device const float4*)(b + b_idx))" : "float4(b[b_idx])"];

        // Output base index
        [src appendFormat:@"  uint out_base = outer_idx * %uu + mid_idx * %uu + inner_idx;\n",
            mid * inner, inner];

        // Op
        if (uop == UOP_CMP)
            [src appendString:@"  *((device float4*)(out+out_base)) = float4(va.x>vb.x?1.f:0.f,va.y>vb.y?1.f:0.f,va.z>vb.z?1.f:0.f,va.w>vb.w?1.f:0.f);\n"];
        else if (uop == UOP_MAX)
            [src appendString:@"  *((device float4*)(out+out_base)) = max(va,vb);\n"];
        else
            [src appendFormat:@"  *((device float4*)(out+out_base)) = va %s vb;\n", uop_to_op(uop)];
        [src appendString:@"}\n"];

        NSError *err;
        id<MTLLibrary> lib = [mtl_dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { NSLog(@"mdim f4 error: %@\n%@", err, src); return 0; }
        pipe = [mtl_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mdim_f4"] error:&err];
        if (!pipe) return 0;
        u32 slot = mdim_cache_count < MDIM_CACHE_SIZE ? mdim_cache_count++ : (mdim_cache_count++ % MDIM_CACHE_SIZE);
        mdim_cache[slot].key = key;
        mdim_cache[slot].pipe = pipe;
    }

    // Recompute grid dims for dispatch
    u32 inner2 = 1, mid2 = 1, outer2 = 1;
    { u32 is2 = rank;
      for (int d = (int)rank-1; d >= 0; d--) { if (inner2*dv->shape.dims[d]<=1024) { inner2*=dv->shape.dims[d]; is2=(u32)d; } else break; }
      u32 ms2 = is2;
      for (int d = (int)is2-1; d >= 0; d--) { if (mid2*dv->shape.dims[d]<=65535) { mid2*=dv->shape.dims[d]; ms2=(u32)d; } else break; }
      for (u32 d = 0; d < ms2; d++) outer2 *= dv->shape.dims[d];
    }
    u32 gw = inner2/4, gh = mid2, gd = outer2;
    u32 tw = MIN(gw, 256u);
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:pipe];
    [enc setBuffer:metal_pool.bufs[dst] offset:0 atIndex:0];
    [enc setBuffer:metal_pool.bufs[a_buf] offset:0 atIndex:1];
    [enc setBuffer:metal_pool.bufs[b_buf] offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(gw, gh, gd) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    batch_dirty = 1;
    dc[DC_MDIM]++; total_dispatches++;
    return 1;
}

// Dispatch a multi-dim binary kernel
void metal_dispatch_mdim_binary(u32 uop, u32 dst, const View *dv,
                                 u32 a_buf, const View *av,
                                 u32 b_buf, const View *bv) {
    // Try float4 first
    if (try_mdim_float4(uop, dst, dv, a_buf, av, b_buf, bv)) return;

    // Scalar fallback
    u32 gw, gh, gd, tw, th, td;
    id<MTLComputePipelineState> pipe = get_mdim_binary_pipe(uop, dv, av, bv,
                                                              &gw, &gh, &gd, &tw, &th, &td);
    if (!pipe) return;

    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:pipe];
    [enc setBuffer:metal_pool.bufs[dst] offset:0 atIndex:0];
    [enc setBuffer:metal_pool.bufs[a_buf] offset:0 atIndex:1];
    [enc setBuffer:metal_pool.bufs[b_buf] offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(gw, gh, gd)
       threadsPerThreadgroup:MTLSizeMake(tw, th, td)];
    batch_dirty = 1;
    dc[DC_MDIM]++; total_dispatches++;
    if (jit.state == JIT_CAPTURE) {
        id<MTLBuffer> bufs[] = {metal_pool.bufs[dst], metal_pool.bufs[a_buf], metal_pool.bufs[b_buf]};
        jit_record_dispatch_1d(pipe, bufs, 3, NULL, NULL, 0, gw, gh, gd, tw, th, td);
    }
}

// Multi-dim unary kernel codegen (same approach as binary)
void metal_dispatch_mdim_unary(u32 uop, u32 dst, const View *dv,
                                u32 src_buf, const View *sv) {
    // Reuse binary codegen with b = dummy
    // Actually, generate a simpler kernel
    u64 h = 0xcbf29ce484222325ULL;
    h ^= uop; h *= 0x100000001b3ULL;
    h ^= 0xAAAA; h *= 0x100000001b3ULL; // mark as unary
    for (u32 d = 0; d < dv->shape.rank; d++) {
        h ^= dv->shape.dims[d]; h *= 0x100000001b3ULL;
    }
    for (u32 d = 0; d < sv->shape.rank; d++) {
        h ^= (u64)(u32)sv->strides[d]; h *= 0x100000001b3ULL;
    }
    h ^= (u64)sv->offset; h *= 0x100000001b3ULL;

    for (u32 i = 0; i < mdim_cache_count && i < MDIM_CACHE_SIZE; i++)
        if (mdim_cache[i].key == h) goto dispatch_unary;

    {
        u32 rank = dv->shape.rank;
        u32 inner = 1, mid = 1, outer = 1;
        u32 inner_start = rank, mid_start;
        for (int d = (int)rank - 1; d >= 0; d--) {
            if (inner * dv->shape.dims[d] <= 1024) { inner *= dv->shape.dims[d]; inner_start = (u32)d; }
            else break;
        }
        mid_start = inner_start;
        for (int d = (int)inner_start - 1; d >= 0; d--) {
            if (mid * dv->shape.dims[d] <= 65535) { mid *= dv->shape.dims[d]; mid_start = (u32)d; }
            else break;
        }
        for (u32 d = 0; d < mid_start; d++) outer *= dv->shape.dims[d];

        const char *op_expr;
        switch(uop) {
            case UOP_NEG:  op_expr = "-a[a_idx]"; break;
            case UOP_RELU: op_expr = "max(a[a_idx], 0.0f)"; break;
            case UOP_EXP:  op_expr = "exp(a[a_idx])"; break;
            case UOP_LOG:  op_expr = "log(a[a_idx])"; break;
            case UOP_SQRT: op_expr = "sqrt(a[a_idx])"; break;
            default: return;
        }

        NSMutableString *src = [NSMutableString stringWithCapacity:1024];
        [src appendString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
        [src appendFormat:@"kernel void mdim_un(device float *out[[buffer(0)]],device const float *a[[buffer(1)]],uint3 gid[[thread_position_in_grid]]){\n"];
        [src appendFormat:@"  uint inner_idx=gid.x,mid_idx=gid.y,outer_idx=gid.z;\n"];
        [src appendFormat:@"  if(inner_idx>=%uu||mid_idx>=%uu||outer_idx>=%uu)return;\n", inner, mid, outer];

        for (u32 d = 0; d < rank; d++) {
            const char *group;
            u32 group_end;
            if (d < mid_start) { group = "outer_idx"; group_end = mid_start; }
            else if (d < inner_start) { group = "mid_idx"; group_end = inner_start; }
            else { group = "inner_idx"; group_end = rank; }
            u32 divisor = 1;
            for (u32 dd = d + 1; dd < group_end; dd++) divisor *= dv->shape.dims[dd];
            if (divisor == 1 && dv->shape.dims[d] == 1)
                [src appendFormat:@"  uint c%u=0;\n", d];
            else if (divisor == 1)
                [src appendFormat:@"  uint c%u=%s%%%uu;\n", d, group, dv->shape.dims[d]];
            else
                [src appendFormat:@"  uint c%u=(%s/%uu)%%%uu;\n", d, group, divisor, dv->shape.dims[d]];
        }

        [src appendFormat:@"  uint a_idx=%d", sv->offset];
        for (u32 d = 0; d < rank; d++)
            if (sv->strides[d] != 0) {
                if (sv->strides[d] == 1) [src appendFormat:@"+c%u", d];
                else [src appendFormat:@"+c%u*%du", d, sv->strides[d]];
            }
        [src appendString:@";\n"];
        [src appendFormat:@"  uint out_idx=outer_idx*%uu+mid_idx*%uu+inner_idx;\n", mid*inner, inner];
        [src appendFormat:@"  out[out_idx]=%s;\n}\n", op_expr];

        NSError *err;
        id<MTLLibrary> lib = [mtl_dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { NSLog(@"mdim unary error: %@\n%@", err, src); return; }
        id<MTLComputePipelineState> pipe = [mtl_dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"mdim_un"] error:&err];
        if (!pipe) return;
        u32 slot = mdim_cache_count < MDIM_CACHE_SIZE ?
            mdim_cache_count++ : (mdim_cache_count++ % MDIM_CACHE_SIZE);
        mdim_cache[slot].key = h;
        mdim_cache[slot].pipe = pipe;
    }

dispatch_unary:;
    u32 rank = dv->shape.rank;
    u32 gw = 1, gh = 1, gd = 1;
    u32 is2 = rank;
    for (int d = (int)rank-1; d >= 0; d--) { if (gw*dv->shape.dims[d]<=1024) { gw*=dv->shape.dims[d]; is2=(u32)d; } else break; }
    u32 ms = is2;
    for (int d = (int)is2-1; d >= 0; d--) { if (gh*dv->shape.dims[d]<=65535) { gh*=dv->shape.dims[d]; ms=(u32)d; } else break; }
    for (u32 d = 0; d < ms; d++) gd *= dv->shape.dims[d];

    id<MTLComputePipelineState> pipe = nil;
    for (u32 i = 0; i < mdim_cache_count && i < MDIM_CACHE_SIZE; i++)
        if (mdim_cache[i].key == h) { pipe = mdim_cache[i].pipe; break; }
    if (!pipe) return;

    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:pipe];
    [enc setBuffer:metal_pool.bufs[dst] offset:0 atIndex:0];
    [enc setBuffer:metal_pool.bufs[src_buf] offset:0 atIndex:1];
    u32 tw = MIN(gw, 256u);
    [enc dispatchThreads:MTLSizeMake(gw,gh,gd) threadsPerThreadgroup:MTLSizeMake(tw,1,1)];
    batch_dirty = 1;
    dc[DC_OTHER]++; total_dispatches++;
}
