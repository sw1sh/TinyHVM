// metal/fused.m — Fused MUL+SUM and runtime kernel codegen

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

// Runtime kernel codegen cache
#define FUSED_CACHE_SIZE 8

static struct {
    u64 key;
    id<MTLComputePipelineState> pipe;
} fused_cache[FUSED_CACHE_SIZE];
static u32 fused_cache_count = 0;

static u64 fuse_hash(const u32 *ops, u32 n_ops, int has_reduce) {
    u64 h = 0xcbf29ce484222325ULL;
    for (u32 i = 0; i < n_ops; i++) {
        h ^= ops[i];
        h *= 0x100000001b3ULL;
    }
    h ^= (u64)has_reduce;
    return h;
}

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
    u64 key = fuse_hash(ops, n_ops, has_reduce);
    for (u32 i = 0; i < fused_cache_count && i < FUSED_CACHE_SIZE; i++)
        if (fused_cache[i].key == key) return fused_cache[i].pipe;

    NSString *src = metal_codegen_fused(ops, n_ops, n_inputs, has_reduce);
    NSError *err = nil;
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [mtl_dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { NSLog(@"TinyHVM fused codegen: %@\n%@", err, src); return nil; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"fused_kernel"];
    id<MTLComputePipelineState> pipe = [mtl_dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pipe) { NSLog(@"TinyHVM fused pipeline: %@", err); return nil; }

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
