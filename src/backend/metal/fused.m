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
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[a_buf], metal_pool.bufs[b_buf] };
    const void *params[] = { &avp, &bvp, &ovp, &rp };
    u64 psizes[] = { sizeof(ViewParams), sizeof(ViewParams), sizeof(ViewParams), sizeof(MulReduceParams) };
    dispatch_1d(pipe_mul_reduce_sum, bufs, 3, params, psizes, 4, dst_numel);
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
    "  // contiguous fast path disabled for fused — leaf views may have non-zero offset\n"
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
    [src appendString:@"struct VP { int strides[8]; uint shape[8]; int offset; uint rank; uint numel; uint has_mask; uint mask_begin[8]; uint mask_end[8]; uint contiguous; };\n"];
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

// GPU contiguify: copy non-contiguous view to contiguous buffer via strided read.
// Uses fused_v2 codegen with 0 ops and 1 leaf (identity copy with ViewParams).
void metal_contiguify(u32 dst_buf, u32 numel, u32 src_buf, const View *src_view) {
    const View *views[] = { src_view };
    u32 bufs[] = { src_buf };
    FusedOp ops[1]; // unused but array must exist
    id<MTLComputePipelineState> pipe = get_fused_pipe_v2(ops, 0, 1, 0);
    if (!pipe) return;

    id<MTLBuffer> mbufs[2];
    mbufs[0] = metal_pool.bufs[dst_buf];
    mbufs[1] = metal_pool.bufs[src_buf];

    ViewParams vp = view_to_params(src_view);
    const void *params[] = { &vp, &numel };
    u64 psizes[] = { sizeof(ViewParams), sizeof(u32) };
    dispatch_1d(pipe, mbufs, 2, params, psizes, 2, numel);
}

// Dispatch a fused elementwise kernel.
// leaf_bufs: buffer IDs for leaf inputs. leaf_views: View pointers for each.
// ops: the fused op chain. Result goes to out_buf.
void metal_dispatch_fused_v2(u32 out_buf, u32 out_numel,
                               u32 *leaf_bufs, const View **leaf_views, u32 n_leaves,
                               FusedOp *ops, u32 n_ops,
                               int has_reduce, u32 reduce_dim) {
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
        dispatch_1d(pipe, bufs, n_leaves + 1, params, psizes, n_leaves + 2, out_numel);
    } else {
        dispatch_1d(pipe, bufs, n_leaves + 1, params, psizes, n_leaves + 1, out_numel);
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
