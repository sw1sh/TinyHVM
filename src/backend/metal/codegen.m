// metal/codegen.m — Unified JIT kernel codegen
// ONE function generates optimal Metal kernels for all elementwise ops.
// Replaces: fused_v2, mdim_bin, mdim_f4, mdim_un, bc2d, fast_*, slow_*.

#define CODEGEN_CACHE_SIZE 256
static struct { u64 key; id<MTLComputePipelineState> pipe; } cg_cache[CODEGEN_CACHE_SIZE];
static u32 cg_cache_count = 0;

// Hash: op chain + leaf stride patterns + output shape
static u64 cg_hash(const FusedOp *ops, u32 n_ops, u32 n_leaves,
                    const View **leaf_views, u32 out_numel) {
    u64 h = 0x1337c0de00000000ULL;
    h ^= n_ops; h *= 0x100000001b3ULL;
    h ^= n_leaves; h *= 0x100000001b3ULL;
    for (u32 i = 0; i < n_ops; i++) {
        h ^= ops[i].uop; h *= 0x100000001b3ULL;
        h ^= ops[i].arg_a; h *= 0x100000001b3ULL;
        h ^= ops[i].arg_b; h *= 0x100000001b3ULL;
    }
    for (u32 i = 0; i < n_leaves; i++) {
        const View *v = leaf_views[i];
        h ^= v->numel; h *= 0x100000001b3ULL;
        h ^= (u64)v->offset; h *= 0x100000001b3ULL;
        h ^= v->has_mask; h *= 0x100000001b3ULL;
        for (u32 d = 0; d < v->shape.rank; d++) {
            h ^= v->shape.dims[d]; h *= 0x100000001b3ULL;
            h ^= (u64)(u32)v->strides[d]; h *= 0x100000001b3ULL;
        }
    }
    h ^= out_numel; h *= 0x100000001b3ULL;
    return h;
}

static const char *cg_op_str(u32 uop) {
    switch(uop) {
        case UOP_ADD: return "+"; case UOP_SUB: return "-";
        case UOP_MUL: return "*"; case UOP_DIV: return "/";
        default: return "+";
    }
}

// Compute output shape (broadcast max of all leaf shapes)
static void cg_output_shape(const View **leaves, u32 n_leaves,
                              u32 *out_shape, u32 *out_rank) {
    *out_rank = 0;
    memset(out_shape, 0, MAX_DIM * sizeof(u32));
    for (u32 i = 0; i < n_leaves; i++) {
        if (leaves[i]->shape.rank > *out_rank) *out_rank = leaves[i]->shape.rank;
        for (u32 d = 0; d < leaves[i]->shape.rank; d++)
            if (leaves[i]->shape.dims[d] > out_shape[d])
                out_shape[d] = leaves[i]->shape.dims[d];
    }
}

// Check if leaf is flat-indexable (contiguous, no broadcast, same numel as output)
static int cg_leaf_is_flat(const View *v, u32 out_numel) {
    if (v->offset != 0 || v->has_mask || v->numel != out_numel) return 0;
    i32 exp = 1;
    for (int d = (int)v->shape.rank - 1; d >= 0; d--) {
        if (v->shape.dims[d] > 1 && v->strides[d] != exp) return 0;
        exp *= (i32)v->shape.dims[d];
    }
    return 1;
}

// Generate the unified kernel
static NSString *codegen_kernel(const FusedOp *ops, u32 n_ops, u32 n_leaves,
                                  const View **leaf_views, u32 out_numel,
                                  int has_reduce, u32 reduce_dim,
                                  const Shape *out_shape_hint) {
    u32 out_shape[MAX_DIM], out_rank;
    if (out_shape_hint && out_shape_hint->rank > 0) {
        out_rank = out_shape_hint->rank;
        for (u32 d = 0; d < out_rank; d++) out_shape[d] = out_shape_hint->dims[d];
    } else {
        cg_output_shape(leaf_views, n_leaves, out_shape, &out_rank);
    }
    if (out_rank == 0) out_rank = 1;

    // Collapse output dims into 3 groups for 3D dispatch
    u32 inner = 1, mid = 1, outer = 1;
    u32 inner_start = out_rank;
    for (int d = (int)out_rank - 1; d >= 0; d--) {
        if (inner * out_shape[d] <= 1024) { inner *= out_shape[d]; inner_start = (u32)d; }
        else break;
    }
    u32 mid_start = inner_start;
    for (int d = (int)inner_start - 1; d >= 0; d--) {
        if (mid * out_shape[d] <= 65535) { mid *= out_shape[d]; mid_start = (u32)d; }
        else break;
    }
    for (u32 d = 0; d < mid_start; d++) outer *= out_shape[d];

    // Check float4 eligibility
    int use_f4 = (inner % 4 == 0) && (out_rank > 0) && (out_shape[out_rank-1] % 4 == 0);
    if (use_f4) {
        for (u32 i = 0; i < n_leaves; i++) {
            i32 ist = (leaf_views[i]->shape.rank > 0) ? leaf_views[i]->strides[leaf_views[i]->shape.rank-1] : 1;
            if (ist != 1 && ist != 0) { use_f4 = 0; break; }
        }
    }
    if (has_reduce) use_f4 = 0;  // reduce path doesn't vectorize (yet)

    u32 inner_dispatch = use_f4 ? inner / 4 : inner;

    NSMutableString *s = [NSMutableString stringWithCapacity:4096];
    [s appendString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
    [s appendFormat:@"kernel void K(device float *out[[buffer(0)]],\n"];
    for (u32 i = 0; i < n_leaves; i++)
        [s appendFormat:@"  device const float *in%u[[buffer(%u)]],\n", i, i+1];
    [s appendFormat:@"  uint3 gid[[thread_position_in_grid]])\n{\n"];
    [s appendFormat:@"  uint ix=gid.x,iy=gid.y,iz=gid.z;\n"];
    [s appendFormat:@"  if(ix>=%uu||iy>=%uu||iz>=%uu)return;\n", inner_dispatch, mid, outer];

    // Compute per-dim coordinates from groups
    if (use_f4) [s appendFormat:@"  uint inner_base=ix*4u;\n"];
    else [s appendFormat:@"  uint inner_base=ix;\n"];

    for (u32 d = 0; d < out_rank; d++) {
        const char *grp;
        u32 grp_end;
        if (d < mid_start) { grp = "iz"; grp_end = mid_start; }
        else if (d < inner_start) { grp = "iy"; grp_end = inner_start; }
        else { grp = "inner_base"; grp_end = out_rank; }
        u32 div = 1;
        for (u32 dd = d+1; dd < grp_end; dd++) div *= out_shape[dd];
        if (div == 1 && out_shape[d] == 1) [s appendFormat:@"  uint c%u=0;\n", d];
        else if (div == 1) [s appendFormat:@"  uint c%u=%s%%%uu;\n", d, grp, out_shape[d]];
        else [s appendFormat:@"  uint c%u=(%s/%uu)%%%uu;\n", d, grp, div, out_shape[d]];
    }

    // Per-leaf index expression
    for (u32 li = 0; li < n_leaves; li++) {
        const View *lv = leaf_views[li];
        if (cg_leaf_is_flat(lv, has_reduce ? inner * mid * outer : out_numel)) {
            // Flat contiguous
            if (use_f4) [s appendFormat:@"  // leaf %u: flat (f4)\n", li];
            else [s appendFormat:@"  uint i%u=iz*%uu+iy*%uu+inner_base;\n", li, mid*inner, inner];
        } else if (lv->shape.rank == out_rank && ({
            // Same rank: check that mismatched dims have stride 0 (broadcast)
            int _ok = 1;
            for (u32 _d = 0; _d < out_rank; _d++)
                if (lv->shape.dims[_d] != out_shape[_d] && lv->strides[_d] != 0)
                    { _ok = 0; break; }
            _ok; })) {
            // Same rank, broadcast-safe: use output coordinates (no divisions)
            [s appendFormat:@"  uint i%u=%d", li, lv->offset];
            for (u32 d = 0; d < out_rank; d++) {
                if (lv->strides[d] == 0) continue;
                if (lv->strides[d] == 1) [s appendFormat:@"+c%u", d];
                else [s appendFormat:@"+c%u*%du", d, lv->strides[d]];
            }
            [s appendString:@";\n"];
        } else {
            // Different rank: decompose flat output index through LEAF's shape
            // (uses compile-time constant divisions — baked in, not runtime ViewParams)
            u32 flat_numel = 1;
            for (u32 d = 0; d < out_rank; d++) flat_numel *= out_shape[d];
            [s appendFormat:@"  uint fi%u=iz*%uu+iy*%uu+inner_base;\n", li, mid*inner, inner];
            [s appendFormat:@"  uint i%u=%d", li, lv->offset];
            u32 leaf_divisor = 1;
            for (int d = (int)lv->shape.rank - 1; d >= 0; d--) {
                if (lv->strides[d] == 0) { leaf_divisor *= lv->shape.dims[d]; continue; }
                if (leaf_divisor == 1 && lv->strides[d] == 1)
                    [s appendFormat:@"+(fi%u%%%uu)", li, lv->shape.dims[d]];
                else if (leaf_divisor == 1)
                    [s appendFormat:@"+(fi%u%%%uu)*%du", li, lv->shape.dims[d], lv->strides[d]];
                else
                    [s appendFormat:@"+((fi%u/%uu)%%%uu)*%du", li, leaf_divisor, lv->shape.dims[d], lv->strides[d]];
                leaf_divisor *= lv->shape.dims[d];
            }
            [s appendString:@";\n"];
        }
    }

    // Read leaves
    NSString *idx_var = has_reduce ? @"ridx" : nil;

    if (has_reduce) {
        [s appendFormat:@"  float acc=0.0f;\n"];
        [s appendFormat:@"  uint base=iz*%uu+iy*%uu+inner_base;\n", mid*inner, inner];
        [s appendFormat:@"  for(uint r=0;r<%uu;r++){\n", reduce_dim];
        [s appendFormat:@"    uint ridx=base*%uu+r;\n", reduce_dim];
        for (u32 li = 0; li < n_leaves; li++) {
            if (cg_leaf_is_flat(leaf_views[li], inner*mid*outer*reduce_dim))
                [s appendFormat:@"    float t%u=in%u[ridx];\n", li, li];
            else
                [s appendFormat:@"    float t%u=in%u[i%u];\n", li, li, li]; // TODO: proper reduce indexing
        }
    } else if (use_f4) {
        // Float4 reads
        u32 flat_out_numel = inner * mid * outer;
        for (u32 li = 0; li < n_leaves; li++) {
            const View *lv = leaf_views[li];
            i32 ist = (lv->shape.rank > 0) ? lv->strides[lv->shape.rank-1] : 1;
            if (cg_leaf_is_flat(lv, flat_out_numel)) {
                [s appendFormat:@"  uint fi%u=iz*%uu+iy*%uu+inner_base;\n", li, mid*inner, inner];
                [s appendFormat:@"  float4 t%u=*((device const float4*)(in%u+fi%u));\n", li, li, li];
            } else if (ist == 0 || lv->numel < 4) {
                // Broadcast: scalar load → float4 (stride=0 or buffer too small for float4)
                [s appendFormat:@"  float4 t%u=float4(in%u[i%u]);\n", li, li, li];
            } else if (lv->numel == flat_out_numel && ist == 1) {
                // Same size, contiguous innermost: safe float4 load
                [s appendFormat:@"  float4 t%u=*((device const float4*)(in%u+i%u));\n", li, li, li];
            } else {
                // Non-contiguous or mismatched size: scalar broadcast to be safe
                [s appendFormat:@"  float4 t%u=float4(in%u[i%u],in%u[i%u+1u],in%u[i%u+2u],in%u[i%u+3u]);\n",
                    li, li, li, li, li, li, li, li, li];
            }
        }
    } else {
        // Scalar reads
        for (u32 li = 0; li < n_leaves; li++) {
            if (cg_leaf_is_flat(leaf_views[li], out_numel))
                [s appendFormat:@"  uint fi%u=iz*%uu+iy*%uu+inner_base;\n  float t%u=in%u[fi%u];\n",
                    li, mid*inner, inner, li, li, li];
            else
                [s appendFormat:@"  float t%u=in%u[i%u];\n", li, li, li];
        }
    }

    // Ops
    NSString *ft = use_f4 ? @"float4" : @"float";
    for (u32 i = 0; i < n_ops; i++) {
        u32 tid = n_leaves + i, a = ops[i].arg_a, b = ops[i].arg_b;
        switch (ops[i].uop) {
            case UOP_ADD:  [s appendFormat:@"  %@ t%u=t%u+t%u;\n", ft, tid, a, b]; break;
            case UOP_SUB:  [s appendFormat:@"  %@ t%u=t%u-t%u;\n", ft, tid, a, b]; break;
            case UOP_MUL:  [s appendFormat:@"  %@ t%u=t%u*t%u;\n", ft, tid, a, b]; break;
            case UOP_DIV:  [s appendFormat:@"  %@ t%u=t%u/t%u;\n", ft, tid, a, b]; break;
            case UOP_MAX:  [s appendFormat:@"  %@ t%u=max(t%u,t%u);\n", ft, tid, a, b]; break;
            case UOP_CMP:
                if (use_f4) [s appendFormat:@"  float4 t%u=float4(t%u.x>t%u.x?1.f:0.f,t%u.y>t%u.y?1.f:0.f,t%u.z>t%u.z?1.f:0.f,t%u.w>t%u.w?1.f:0.f);\n",
                    tid, a, b, a, b, a, b, a, b];
                else [s appendFormat:@"  float t%u=t%u>t%u?1.f:0.f;\n", tid, a, b]; break;
            case UOP_NEG:  [s appendFormat:@"  %@ t%u=-t%u;\n", ft, tid, a]; break;
            case UOP_RELU:
                if (use_f4) [s appendFormat:@"  float4 t%u=max(t%u,float4(0.f));\n", tid, a];
                else [s appendFormat:@"  float t%u=max(t%u,0.f);\n", tid, a]; break;
            case UOP_EXP:  [s appendFormat:@"  %@ t%u=exp(t%u);\n", ft, tid, a]; break;
            case UOP_LOG:  [s appendFormat:@"  %@ t%u=log(t%u);\n", ft, tid, a]; break;
            case UOP_SQRT: [s appendFormat:@"  %@ t%u=sqrt(t%u);\n", ft, tid, a]; break;
            default:       [s appendFormat:@"  %@ t%u=t%u;\n", ft, tid, a]; break;
        }
        if (has_reduce) [s appendFormat:@"    acc+="];
    }

    // Write output
    u32 last = n_leaves + n_ops - 1;
    if (has_reduce) {
        [s appendFormat:@"    acc+=t%u;\n  }\n", last];
        [s appendFormat:@"  uint oi=iz*%uu+iy*%uu+inner_base;\n  out[oi]=acc;\n", mid*inner, inner];
    } else if (use_f4) {
        [s appendFormat:@"  uint oi=iz*%uu+iy*%uu+inner_base;\n", mid*inner, inner];
        [s appendFormat:@"  *((device float4*)(out+oi))=t%u;\n", last];
    } else {
        [s appendFormat:@"  uint oi=iz*%uu+iy*%uu+inner_base;\n  out[oi]=t%u;\n", mid*inner, inner, last];
    }

    [s appendString:@"}\n"];
    { static u32 _pk=0; if(n_ops>=2 && ++_pk<=2) fprintf(stderr, "KERNEL(ops=%u leaves=%u out=%u rank=%u):\n%s\n", n_ops, n_leaves, out_numel, out_rank, [s UTF8String]); }
    return s;
}

// Get or compile a cached kernel
static id<MTLComputePipelineState> cg_get_pipe(const FusedOp *ops, u32 n_ops,
                                                  u32 n_leaves, const View **leaf_views,
                                                  u32 out_numel, int has_reduce, u32 reduce_dim,
                                                  const Shape *out_shape_hint) {
    u64 key = cg_hash(ops, n_ops, n_leaves, leaf_views, out_numel) ^ ((u64)has_reduce << 63);
    for (u32 i = 0; i < cg_cache_count && i < CODEGEN_CACHE_SIZE; i++)
        if (cg_cache[i].key == key) return cg_cache[i].pipe;

    NSString *src = codegen_kernel(ops, n_ops, n_leaves, leaf_views, out_numel, has_reduce, reduce_dim, out_shape_hint);
    static int _kp = 0;
    if (_kp < 8) { fprintf(stderr, "KERNEL(ops=%u leaves=%u out=%u rank=%u):\n%s\n", n_ops, n_leaves, out_numel, out_shape_hint?out_shape_hint->rank:0, [src UTF8String]); _kp++; }
    NSError *err;
    id<MTLLibrary> lib = [mtl_dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"codegen error: %@\n%@", err, src); return nil; }
    id<MTLComputePipelineState> pipe = [mtl_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"K"] error:&err];
    if (!pipe) return nil;

    u32 slot = cg_cache_count < CODEGEN_CACHE_SIZE ?
        cg_cache_count++ : (cg_cache_count++ % CODEGEN_CACHE_SIZE);
    cg_cache[slot].key = key;
    cg_cache[slot].pipe = pipe;
    return pipe;
}

// Unified dispatch: handles any elementwise op (single or fused chain)
void metal_dispatch_kernel(u32 out_buf, u32 out_numel,
                            u32 *leaf_bufs, const View **leaf_views, u32 n_leaves,
                            FusedOp *ops, u32 n_ops,
                            int has_reduce, u32 reduce_dim,
                            const Shape *out_shape_hint) {
    id<MTLComputePipelineState> pipe = cg_get_pipe(ops, n_ops, n_leaves, leaf_views,
                                                      out_numel, has_reduce, reduce_dim, out_shape_hint);
    if (!pipe) return;

    // Compute grid dims (use hint if available)
    u32 out_shape[MAX_DIM], out_rank;
    if (out_shape_hint && out_shape_hint->rank > 0) {
        out_rank = out_shape_hint->rank;
        for (u32 d = 0; d < out_rank; d++) out_shape[d] = out_shape_hint->dims[d];
    } else {
        cg_output_shape(leaf_views, n_leaves, out_shape, &out_rank);
    }
    if (out_rank == 0) out_rank = 1;

    u32 inner = 1, mid = 1, outer = 1;
    u32 inner_start = out_rank;
    for (int d = (int)out_rank - 1; d >= 0; d--) {
        if (inner * out_shape[d] <= 1024) { inner *= out_shape[d]; inner_start = (u32)d; }
        else break;
    }
    u32 mid_start = inner_start;
    for (int d = (int)inner_start - 1; d >= 0; d--) {
        if (mid * out_shape[d] <= 65535) { mid *= out_shape[d]; mid_start = (u32)d; }
        else break;
    }
    for (u32 d = 0; d < mid_start; d++) outer *= out_shape[d];

    // Float4 check (must match codegen)
    int use_f4 = (inner % 4 == 0) && (out_rank > 0) && (out_shape[out_rank-1] % 4 == 0);
    if (use_f4) {
        for (u32 i = 0; i < n_leaves; i++) {
            i32 ist = (leaf_views[i]->shape.rank > 0) ? leaf_views[i]->strides[leaf_views[i]->shape.rank-1] : 1;
            if (ist != 1 && ist != 0) { use_f4 = 0; break; }
        }
    }
    if (has_reduce) use_f4 = 0;

    u32 gw = use_f4 ? inner / 4 : inner;
    u32 tw = MIN(gw, 256u);

    // Encode dispatch
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:pipe];
    [enc setBuffer:metal_pool.bufs[out_buf] offset:0 atIndex:0];
    for (u32 i = 0; i < n_leaves; i++)
        [enc setBuffer:metal_pool.bufs[leaf_bufs[i]] offset:0 atIndex:i+1];
    [enc dispatchThreads:MTLSizeMake(gw, mid, outer)
       threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    batch_dirty = 1;
    total_dispatches++;
    if (jit.state == JIT_CAPTURE) {
        id<MTLBuffer> bufs[17];
        bufs[0] = metal_pool.bufs[out_buf];
        for (u32 i = 0; i < n_leaves; i++) bufs[i+1] = metal_pool.bufs[leaf_bufs[i]];
        jit_record_dispatch_1d(pipe, bufs, n_leaves+1, NULL, NULL, 0, gw, mid, outer, tw, 1, 1);
    }
}
