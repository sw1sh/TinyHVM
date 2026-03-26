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

// GPU contiguify: strided copy via identity kernel
void metal_contiguify(u32 dst_buf, u32 numel, u32 src_buf, const View *src_view) {
    const View *views[] = { src_view };
    u32 bufs_[] = { src_buf };
    FusedOp ops_[1];
    // Use 0-op fused kernel (identity copy with ViewParams)
    // Fall through to the general dispatch below
    ViewParams vp = view_to_params(src_view);

    id<MTLBuffer> mbufs[2];
    mbufs[0] = metal_pool.bufs[dst_buf];
    mbufs[1] = metal_pool.bufs[src_buf];
    const void *params[] = { &vp, &numel };
    u64 psizes[] = { sizeof(ViewParams), sizeof(u32) };
    // Need a pipeline for identity copy — use cached fused_v2
    static id<MTLComputePipelineState> contiguify_pipe = nil;
    if (!contiguify_pipe) {
        NSMutableString *src = [NSMutableString stringWithCapacity:1024];
        [src appendString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
        [src appendString:@"struct VP { int strides[8]; uint shape[8]; int offset; uint rank; uint numel; uint has_mask; uint mask_begin[8]; uint mask_end[8]; };\n"];
        [src appendString:@"inline uint si(uint f, constant VP &v) { uint p=uint(v.offset); for(int d=int(v.rank)-1;d>=0;d--){p+=(f%v.shape[d])*uint(v.strides[d]>0?v.strides[d]:0);f/=v.shape[d];}return p; }\n"];
        [src appendString:@"kernel void contiguify(device float *out[[buffer(0)]],device const float *in[[buffer(1)]],constant VP &v[[buffer(2)]],constant uint &n[[buffer(3)]],uint i[[thread_position_in_grid]]){if(i>=n)return;out[i]=in[si(i,v)];}\n"];
        NSError *err;
        id<MTLLibrary> lib = [mtl_dev newLibraryWithSource:src options:nil error:&err];
        contiguify_pipe = [mtl_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"contiguify"] error:&err];
    }
    dispatch_1d(contiguify_pipe, mbufs, 2, params, psizes, 2, numel);
}

// ============================================================
// Specialized JIT codegen with per-leaf inline indexing
// ============================================================

#define FUSED_CACHE_SIZE 128
static struct {
    u64 key;
    id<MTLComputePipelineState> pipe;
} fused_cache[FUSED_CACHE_SIZE];
static u32 fused_cache_count = 0;

// Classify leaf access: can we use buf[flat] (gid = physical index)?
// Only valid when: contiguous, no mask, no offset, AND numel matches output.
// The out_numel is passed so broadcast leaves (numel < out) use strided indexing.
static u32 fused_out_numel_hint = 0;  // set before codegen

static int leaf_is_contiguous(const View *v) {
    if (v->offset != 0 || v->has_mask) return 0;
    if (v->numel != fused_out_numel_hint) return 0;  // broadcast mismatch
    i32 expected = 1;
    for (int d = (int)v->shape.rank - 1; d >= 0; d--) {
        if (v->shape.dims[d] > 1) {
            if (v->strides[d] != expected) return 0;
            expected *= (i32)v->shape.dims[d];
        }
    }
    return 1;
}

static u64 fuse_hash_v2(const FusedOp *ops, u32 n_ops, u32 n_leaves,
                         const View **leaf_views) {
    u64 h = 0xcbf29ce484222325ULL;
    h ^= n_leaves; h *= 0x100000001b3ULL;
    for (u32 i = 0; i < n_ops; i++) {
        h ^= ops[i].uop; h *= 0x100000001b3ULL;
        h ^= ops[i].arg_a; h *= 0x100000001b3ULL;
        h ^= ops[i].arg_b; h *= 0x100000001b3ULL;
    }
    // Include leaf view patterns in hash (shape, strides determine generated code)
    for (u32 i = 0; i < n_leaves; i++) {
        const View *v = leaf_views[i];
        h ^= (u64)v->numel; h *= 0x100000001b3ULL;
        h ^= (u64)v->offset; h *= 0x100000001b3ULL;
        for (u32 d = 0; d < v->shape.rank; d++) {
            h ^= (u64)v->shape.dims[d]; h *= 0x100000001b3ULL;
            h ^= (u64)(u32)v->strides[d]; h *= 0x100000001b3ULL;
        }
    }
    h ^= (u64)fused_out_numel_hint; h *= 0x100000001b3ULL;
    return h;
}

static NSString *codegen_fused_v2(const FusedOp *ops, u32 n_ops, u32 n_leaves,
                                   const View **leaf_views, int has_reduce) {
    NSMutableString *src = [NSMutableString stringWithCapacity:4096];
    [src appendString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
    [src appendString:@"struct VP { int strides[8]; uint shape[8]; int offset; uint rank; uint numel; uint has_mask; uint mask_begin[8]; uint mask_end[8]; };\n"];

    // Only include strided_idx helper if any leaf needs it
    int any_strided = 0;
    for (u32 i = 0; i < n_leaves; i++)
        if (!leaf_is_contiguous(leaf_views[i])) any_strided = 1;

    if (any_strided) {
        [src appendString:@"inline uint si(uint f, constant VP &v) {\n"
            "  uint p=uint(v.offset);\n"
            "  for(int d=int(v.rank)-1;d>=0;d--){p+=(f%v.shape[d])*uint(v.strides[d]>0?v.strides[d]:0);f/=v.shape[d];}\n"
            "  return p;\n"
            "}\n"];
        [src appendString:@"inline float sr(device const float *buf, uint flat, constant VP &v) {\n"
            "  if(v.has_mask){uint r=flat;for(int d=int(v.rank)-1;d>=0;d--){uint c=r%v.shape[d];r/=v.shape[d];if(c<v.mask_begin[d]||c>=v.mask_end[d])return 0.0f;}}\n"
            "  return buf[si(flat,v)];\n"
            "}\n"];
    }

    [src appendString:@"kernel void fused_v2(\n"];
    [src appendString:@"  device float *out [[buffer(0)]],\n"];
    for (u32 i = 0; i < n_leaves; i++)
        [src appendFormat:@"  device const float *in%u [[buffer(%u)]],\n", i, i + 1];
    // Only pass VP for strided leaves
    for (u32 i = 0; i < n_leaves; i++)
        if (!leaf_is_contiguous(leaf_views[i]))
            [src appendFormat:@"  constant VP &v%u [[buffer(%u)]],\n", i, n_leaves + 1 + i];
    [src appendFormat:@"  constant uint &numel [[buffer(%u)]],\n", 2 * n_leaves + 1];
    if (has_reduce)
        [src appendFormat:@"  constant uint &reduce_dim [[buffer(%u)]],\n", 2 * n_leaves + 2];
    [src appendString:@"  uint gid [[thread_position_in_grid]])\n{\n"];
    [src appendString:@"  if (gid >= numel) return;\n"];

    // Generate per-leaf inline index expression based on view pattern.
    // For each leaf, emit the most efficient read for its specific strides.
    // leaf_views are known at JIT time, so we bake strides into the code.
    #define EMIT_LEAF_READS_SPEC(idx_var) \
        for (u32 li = 0; li < n_leaves; li++) { \
            const View *lv = leaf_views[li]; \
            if (leaf_is_contiguous(lv)) { \
                [src appendFormat:@"    float t%u = in%u[%s];\n", li, li, idx_var]; \
            } else if (!lv->has_mask) { \
                /* Generate inline index: sum of (flat/divisor % dim) * stride per active dim */ \
                [src appendFormat:@"    float t%u = in%u[", li, li]; \
                u32 divisor = 1; \
                int first = 1; \
                /* Walk dims from innermost to outermost (reverse) */ \
                /* Build products from the right: divisor[d] = product of shape[d+1..rank-1] */ \
                u32 divisors[MAX_DIM]; \
                { u32 div = 1; \
                  for (int d = (int)lv->shape.rank - 1; d >= 0; d--) { \
                      divisors[d] = div; div *= lv->shape.dims[d]; \
                  } \
                } \
                for (u32 d = 0; d < lv->shape.rank; d++) { \
                    if (lv->shape.dims[d] <= 1) continue; /* skip trivial dims */ \
                    if (lv->strides[d] == 0) continue; /* skip broadcast dims */ \
                    if (!first) [src appendString:@" + "]; \
                    first = 0; \
                    /* coord[d] = (flat / divisor[d]) % shape[d] */ \
                    /* phys += coord[d] * stride[d] */ \
                    if (lv->strides[d] == 1 && divisors[d] == 1) { \
                        /* innermost contiguous dim: just flat % shape */ \
                        [src appendFormat:@"(%s %% %uu)", idx_var, lv->shape.dims[d]]; \
                    } else if (divisors[d] == 1) { \
                        [src appendFormat:@"(%s %% %uu) * %du", idx_var, lv->shape.dims[d], lv->strides[d]]; \
                    } else { \
                        [src appendFormat:@"((%s / %uu) %% %uu) * %du", idx_var, divisors[d], lv->shape.dims[d], lv->strides[d]]; \
                    } \
                } \
                if (lv->offset != 0) { \
                    [src appendFormat:@" + %d", lv->offset]; \
                } \
                if (first) [src appendString:@"0"]; /* all dims broadcast = scalar */ \
                [src appendString:@"];\n"]; \
            } else { \
                [src appendFormat:@"    float t%u = sr(in%u, %s, v%u);\n", li, li, idx_var, li]; \
            } \
        }

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
        EMIT_LEAF_READS_SPEC("idx");
        EMIT_OPS(@"    ");
        [src appendFormat:@"    acc += t%u;\n", n_leaves + n_ops - 1];
        [src appendString:@"  }\n"];
        [src appendString:@"  out[gid] = acc;\n"];
    } else {
        EMIT_LEAF_READS_SPEC("gid");
        EMIT_OPS(@"  ");
        [src appendFormat:@"  out[gid] = t%u;\n", n_leaves + n_ops - 1];
    }
    #undef EMIT_LEAF_READS_SPEC
    #undef EMIT_OPS
    [src appendString:@"}\n"];
    return src;
}

static id<MTLComputePipelineState> get_fused_pipe_v2(const FusedOp *ops, u32 n_ops,
                                                       u32 n_leaves, const View **leaf_views,
                                                       int has_reduce) {
    u64 key = fuse_hash_v2(ops, n_ops, n_leaves, leaf_views) ^ ((u64)has_reduce << 63);
    for (u32 i = 0; i < fused_cache_count && i < FUSED_CACHE_SIZE; i++)
        if (fused_cache[i].key == key) return fused_cache[i].pipe;

    NSString *src = codegen_fused_v2(ops, n_ops, n_leaves, leaf_views, has_reduce);
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

// Dispatch a fused elementwise kernel with specialized per-leaf indexing
void metal_dispatch_fused_v2(u32 out_buf, u32 out_numel,
                               u32 *leaf_bufs, const View **leaf_views, u32 n_leaves,
                               FusedOp *ops, u32 n_ops,
                               int has_reduce, u32 reduce_dim) {
    fused_out_numel_hint = out_numel;
    id<MTLComputePipelineState> pipe = get_fused_pipe_v2(ops, n_ops, n_leaves, leaf_views, has_reduce);
    if (!pipe) return;

    id<MTLBuffer> bufs[16];
    bufs[0] = metal_pool.bufs[out_buf];
    for (u32 i = 0; i < n_leaves && i < 15; i++)
        bufs[i + 1] = metal_pool.bufs[leaf_bufs[i]];

    ViewParams vps[8];
    const void *params[16];
    u64 psizes[16];
    u32 n_params = 0;
    // Only pass VP for strided (non-contiguous) leaves
    for (u32 i = 0; i < n_leaves; i++) {
        vps[i] = view_to_params(leaf_views[i]);
        params[n_params] = &vps[i];
        psizes[n_params] = sizeof(ViewParams);
        n_params++;
    }
    params[n_params] = &out_numel;
    psizes[n_params] = sizeof(u32);
    n_params++;
    if (has_reduce) {
        params[n_params] = &reduce_dim;
        psizes[n_params] = sizeof(u32);
        n_params++;
        dispatch_1d(pipe, bufs, n_leaves + 1, params, psizes, n_params, out_numel);
    } else {
        dispatch_1d(pipe, bufs, n_leaves + 1, params, psizes, n_params, out_numel);
    }
}

// ============================================================
// Legacy fused dispatch (old format, kept for compatibility)
// ============================================================

void metal_dispatch_fused(u32 out_buf, u32 *input_bufs, u32 n_inputs,
                          u32 *uops, u32 n_ops, u32 out_numel) {
    // Legacy path — not used in current code
    (void)out_buf; (void)input_bufs; (void)n_inputs;
    (void)uops; (void)n_ops; (void)out_numel;
}
