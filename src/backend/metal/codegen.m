// metal/codegen.m — General JIT kernel codegen for elementwise + reduce
// Generates optimal Metal kernels for any fused subgraph.
// Handles: elementwise chains, fused reduce (any axis config), masks, broadcasting.

// ReduceSpec: defined in tinyhvm.c (included before this file)
// typedef struct { u8 is_reduce[MAX_DIM]; u32 reduce_type; } ReduceSpec;

#define CODEGEN_CACHE_SIZE 256
static struct { u64 key; id<MTLComputePipelineState> pipe; } cg_cache[CODEGEN_CACHE_SIZE];
static u32 cg_cache_count = 0;

// ── Hash: op chain + leaf patterns + full shape + reduce spec ──────
static u64 cg_hash_rs(const FusedOp *ops, u32 n_ops, u32 n_leaves,
                       const View **leaf_views, const Shape *full_shape,
                       const ReduceSpec *reduce) {
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
    for (u32 d = 0; d < full_shape->rank; d++) {
        h ^= full_shape->dims[d]; h *= 0x100000001b3ULL;
    }
    if (reduce && reduce->reduce_type) {
        h ^= (u64)reduce->reduce_type << 32; h *= 0x100000001b3ULL;
        for (u32 d = 0; d < full_shape->rank; d++) {
            h ^= (u64)reduce->is_reduce[d] << d; h *= 0x100000001b3ULL;
        }
    }
    return h;
}

static const char *cg_op_str(u32 uop) {
    switch(uop) {
        case UOP_ADD: return "+"; case UOP_SUB: return "-";
        case UOP_MUL: return "*"; case UOP_DIV: return "/";
        default: return "+";
    }
}

// Compute output shape (broadcast max of all leaf shapes) — used by old interface
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

// Check if leaf is flat-indexable (contiguous, no broadcast, same numel)
static int cg_leaf_is_flat(const View *v, u32 target_numel) {
    if (v->offset != 0 || v->has_mask || v->numel != target_numel) return 0;
    i32 exp = 1;
    for (int d = (int)v->shape.rank - 1; d >= 0; d--) {
        if (v->shape.dims[d] > 1 && v->strides[d] != exp) return 0;
        exp *= (i32)v->shape.dims[d];
    }
    return 1;
}

// ════════════════════════════════════════════════════════════════════
// codegen_kernel_rs — General codegen with ReduceSpec
// ════════════════════════════════════════════════════════════════════
// full_shape: pre-reduction shape (all axes including reduce).
// reduce: NULL or ReduceSpec with is_reduce[] and reduce_type.
// Kernel iterates output axes via dispatch grid, reduce axes via inner loop.
// Leaf indexing uses coordinates for ALL axes.

static NSString *codegen_kernel_rs(const FusedOp *ops, u32 n_ops, u32 n_leaves,
                                    const View **leaf_views,
                                    const Shape *full_shape,
                                    const ReduceSpec *reduce,
                                    const u32 *side_op_indices, u32 n_side_outputs) {
    u32 rank = full_shape->rank;
    if (rank == 0) rank = 1;

    // ── Classify axes ──────────────────────────────────────────
    u32 out_dims[MAX_DIM], n_out = 0; // output axis indices (in full shape)
    u32 red_dims[MAX_DIM], n_red = 0; // reduce axis indices
    u32 out_shape[MAX_DIM];           // output-axis-only dims (for dispatch grid)
    u32 out_numel = 1, reduce_numel = 1;
    int has_reduce = reduce && reduce->reduce_type;

    for (u32 d = 0; d < rank; d++) {
        if (has_reduce && reduce->is_reduce[d]) {
            red_dims[n_red++] = d;
            reduce_numel *= full_shape->dims[d];
        } else {
            out_shape[n_out] = full_shape->dims[d];
            out_dims[n_out] = d;
            n_out++;
            out_numel *= full_shape->dims[d];
        }
    }
    if (n_out == 0) { n_out = 1; out_shape[0] = 1; out_dims[0] = rank; }

    // ── Collapse output dims into 3 groups for 3D dispatch ──
    u32 inner = 1, mid = 1, outer = 1;
    u32 inner_start = n_out;
    for (int i = (int)n_out - 1; i >= 0; i--) {
        if (inner * out_shape[i] <= 1024) { inner *= out_shape[i]; inner_start = (u32)i; }
        else break;
    }
    u32 mid_start = inner_start;
    for (int i = (int)inner_start - 1; i >= 0; i--) {
        if (mid * out_shape[i] <= 65535) { mid *= out_shape[i]; mid_start = (u32)i; }
        else break;
    }
    for (u32 i = 0; i < mid_start; i++) outer *= out_shape[i];

    // ── Float4 eligibility ─────────────────────────────────────
    int use_f4 = !has_reduce && (inner % 4 == 0) && n_out > 0;
    if (use_f4 && out_dims[n_out - 1] < rank) {
        if (full_shape->dims[out_dims[n_out - 1]] % 4 != 0) use_f4 = 0;
    }
    if (use_f4) {
        for (u32 i = 0; i < n_leaves; i++) {
            u32 last_d = out_dims[n_out - 1];
            i32 ist = (last_d < leaf_views[i]->shape.rank)
                ? leaf_views[i]->strides[last_d] : 1;
            if (ist != 1 && ist != 0) { use_f4 = 0; break; }
        }
    }
    for (u32 i = 0; i < n_leaves; i++)
        if (leaf_views[i]->has_mask) { use_f4 = 0; break; }

    u32 inner_dispatch = use_f4 ? inner / 4 : inner;

    // ── Kernel header ──────────────────────────────────────────
    NSMutableString *s = [NSMutableString stringWithCapacity:4096];
    [s appendString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
    [s appendFormat:@"kernel void K(device float *out[[buffer(0)]],\n"];
    // Side output buffers (shared intermediates)
    for (u32 si = 0; si < n_side_outputs; si++)
        [s appendFormat:@"  device float *side%u[[buffer(%u)]],\n", si, si + 1];
    u32 buf_offset = 1 + n_side_outputs;
    for (u32 i = 0; i < n_leaves; i++)
        [s appendFormat:@"  device const float *in%u[[buffer(%u)]],\n", i, buf_offset + i];
    [s appendFormat:@"  uint3 gid[[thread_position_in_grid]])\n{\n"];
    [s appendFormat:@"  uint ix=gid.x,iy=gid.y,iz=gid.z;\n"];
    [s appendFormat:@"  if(ix>=%uu||iy>=%uu||iz>=%uu)return;\n",
        inner_dispatch, mid, outer];

    if (use_f4) [s appendFormat:@"  uint inner_base=ix*4u;\n"];
    else [s appendFormat:@"  uint inner_base=ix;\n"];

    // ── Output axis coordinates from grid ──────────────────────
    for (u32 oi = 0; oi < n_out; oi++) {
        u32 d = out_dims[oi];
        if (d >= rank) { [s appendFormat:@"  uint c%u=0;\n", oi]; continue; }
        const char *grp; u32 grp_end;
        if (oi < mid_start)        { grp = "iz";         grp_end = mid_start; }
        else if (oi < inner_start) { grp = "iy";         grp_end = inner_start; }
        else                       { grp = "inner_base"; grp_end = n_out; }
        u32 div = 1;
        for (u32 dd = oi + 1; dd < grp_end; dd++) div *= out_shape[dd];
        if (div == 1 && out_shape[oi] == 1)
            [s appendFormat:@"  uint c%u=0;\n", d];
        else if (div == 1)
            [s appendFormat:@"  uint c%u=%s%%%uu;\n", d, grp, out_shape[oi]];
        else
            [s appendFormat:@"  uint c%u=(%s/%uu)%%%uu;\n", d, grp, div, out_shape[oi]];
    }

    // ── Reduce: accumulator + loop + reduce axis coordinates ──
    NSString *indent = has_reduce ? @"    " : @"  ";
    if (has_reduce) {
        [s appendFormat:@"  float acc=%s;\n",
            reduce->reduce_type == UOP_RMAX ? "-1e30f" : "0.0f"];
        [s appendFormat:@"  for(uint r=0;r<%uu;r++){\n", reduce_numel];
        // Decompose r into per-reduce-axis coordinates (innermost last)
        u32 rdiv = 1;
        for (int ri = (int)n_red - 1; ri >= 0; ri--) {
            u32 d = red_dims[ri];
            u32 dim = full_shape->dims[d];
            if (rdiv == 1)
                [s appendFormat:@"    uint c%u=r%%%uu;\n", d, dim];
            else
                [s appendFormat:@"    uint c%u=(r/%uu)%%%uu;\n", d, rdiv, dim];
            rdiv *= dim;
        }
    }

    // ── Per-leaf index expressions ─────────────────────────────
    // Three paths:
    //  A: flat contiguous (non-reduce only, same numel, standard strides)
    //  B: coordinate-based (leaf rank == full rank, broadcast-safe)
    //  C: flat-index decomposition (different rank fallback)
    for (u32 li = 0; li < n_leaves; li++) {
        const View *lv = leaf_views[li];

        // Path A: flat contiguous
        if (!has_reduce && cg_leaf_is_flat(lv, out_numel)) {
            if (use_f4) [s appendFormat:@"  // leaf %u: flat (f4)\n", li];
            else [s appendFormat:@"  uint i%u=iz*%uu+iy*%uu+inner_base;\n",
                li, mid*inner, inner];
            continue;
        }

        // Path B: coordinate-based (leaf rank matches full rank)
        if (lv->shape.rank == rank) {
            int ok = 1;
            for (u32 d = 0; d < rank; d++) {
                if (lv->shape.dims[d] != full_shape->dims[d] &&
                    lv->strides[d] != 0 && lv->shape.dims[d] != 1)
                    { ok = 0; break; }
            }
            if (ok) {
                [s appendFormat:@"%@uint i%u=%d", indent, li, lv->offset];
                for (u32 d = 0; d < rank; d++) {
                    if (lv->strides[d] == 0 || lv->shape.dims[d] == 1) continue;
                    if (lv->strides[d] == 1) [s appendFormat:@"+c%u", d];
                    else [s appendFormat:@"+c%u*%du", d, lv->strides[d]];
                }
                [s appendString:@";\n"];
                continue;
            }
        }

        // Path C: flat-index decomposition through leaf's own shape
        // Compute flat full-shape index from all coordinates
        u32 fs[MAX_DIM]; // full-shape row-major strides
        if (rank > 0) {
            fs[rank - 1] = 1;
            for (int d = (int)rank - 2; d >= 0; d--)
                fs[d] = fs[d + 1] * full_shape->dims[d + 1];
        }
        [s appendFormat:@"%@uint fi%u=", indent, li];
        for (u32 d = 0; d < rank; d++) {
            if (d > 0) [s appendString:@"+"];
            if (fs[d] == 1) [s appendFormat:@"c%u", d];
            else [s appendFormat:@"c%u*%uu", d, fs[d]];
        }
        [s appendString:@";\n"];
        // Decompose fi through leaf's shape
        [s appendFormat:@"%@uint i%u=%d", indent, li, lv->offset];
        u32 leaf_div = 1;
        for (int d = (int)lv->shape.rank - 1; d >= 0; d--) {
            if (lv->strides[d] == 0) { leaf_div *= lv->shape.dims[d]; continue; }
            if (leaf_div == 1 && lv->strides[d] == 1)
                [s appendFormat:@"+(fi%u%%%uu)", li, lv->shape.dims[d]];
            else if (leaf_div == 1)
                [s appendFormat:@"+(fi%u%%%uu)*%du", li, lv->shape.dims[d], lv->strides[d]];
            else
                [s appendFormat:@"+((fi%u/%uu)%%%uu)*%du", li, leaf_div,
                    lv->shape.dims[d], lv->strides[d]];
            leaf_div *= lv->shape.dims[d];
        }
        [s appendString:@";\n"];
    }

    // ── Mask conditions ────────────────────────────────────────
    for (u32 li = 0; li < n_leaves; li++) {
        const View *lv = leaf_views[li];
        if (!lv->has_mask) continue;
        [s appendFormat:@"%@bool m%u=", indent, li];
        int first = 1;
        for (u32 d = 0; d < lv->shape.rank && d < rank; d++) {
            if (lv->mask_begin[d] == 0 && lv->mask_end[d] >= lv->shape.dims[d])
                continue;
            if (!first) [s appendString:@"&&"];
            first = 0;
            if (lv->mask_begin[d] > 0)
                [s appendFormat:@"c%u>=%uu", d, lv->mask_begin[d]];
            if (lv->mask_begin[d] > 0 && lv->mask_end[d] < lv->shape.dims[d])
                [s appendString:@"&&"];
            if (lv->mask_end[d] < lv->shape.dims[d])
                [s appendFormat:@"c%u<%uu", d, lv->mask_end[d]];
        }
        if (first) [s appendString:@"true"];
        [s appendString:@";\n"];
    }

    // ── Leaf reads ─────────────────────────────────────────────
    if (use_f4) {
        u32 flat_out_numel = inner * mid * outer;
        for (u32 li = 0; li < n_leaves; li++) {
            const View *lv = leaf_views[li];
            u32 last_d = out_dims[n_out - 1];
            i32 ist = (last_d < lv->shape.rank) ? lv->strides[last_d] : 1;
            if (cg_leaf_is_flat(lv, flat_out_numel)) {
                [s appendFormat:@"  uint fi%u=iz*%uu+iy*%uu+inner_base;\n",
                    li, mid*inner, inner];
                [s appendFormat:@"  float4 t%u=*((device const float4*)(in%u+fi%u));\n",
                    li, li, li];
            } else if (ist == 0 || lv->numel < 4) {
                [s appendFormat:@"  float4 t%u=float4(in%u[i%u]);\n", li, li, li];
            } else if (lv->numel == flat_out_numel && ist == 1) {
                [s appendFormat:@"  float4 t%u=*((device const float4*)(in%u+i%u));\n",
                    li, li, li];
            } else {
                [s appendFormat:@"  float4 t%u=float4(in%u[i%u],in%u[i%u+1u],in%u[i%u+2u],in%u[i%u+3u]);\n",
                    li, li, li, li, li, li, li, li, li];
            }
        }
    } else {
        for (u32 li = 0; li < n_leaves; li++) {
            const View *lv = leaf_views[li];
            if (!has_reduce && cg_leaf_is_flat(lv, out_numel))
                [s appendFormat:@"  uint fi%u=iz*%uu+iy*%uu+inner_base;\n  float t%u=in%u[fi%u];\n",
                    li, mid*inner, inner, li, li, li];
            else if (lv->has_mask)
                [s appendFormat:@"%@float t%u=m%u?in%u[i%u]:0.f;\n", indent, li, li, li, li];
            else
                [s appendFormat:@"%@float t%u=in%u[i%u];\n", indent, li, li, li];
        }
    }

    // ── Op chain ───────────────────────────────────────────────
    NSString *ft = use_f4 ? @"float4" : @"float";
    for (u32 i = 0; i < n_ops; i++) {
        u32 tid = n_leaves + i, a = ops[i].arg_a, b = ops[i].arg_b;
        switch (ops[i].uop) {
            case UOP_ADD:  [s appendFormat:@"%@%@ t%u=t%u+t%u;\n", indent, ft, tid, a, b]; break;
            case UOP_SUB:  [s appendFormat:@"%@%@ t%u=t%u-t%u;\n", indent, ft, tid, a, b]; break;
            case UOP_MUL:  [s appendFormat:@"%@%@ t%u=t%u*t%u;\n", indent, ft, tid, a, b]; break;
            case UOP_DIV:  [s appendFormat:@"%@%@ t%u=t%u/t%u;\n", indent, ft, tid, a, b]; break;
            case UOP_MAX:  [s appendFormat:@"%@%@ t%u=max(t%u,t%u);\n", indent, ft, tid, a, b]; break;
            case UOP_CMP:
                if (use_f4) [s appendFormat:@"  float4 t%u=float4(t%u.x>t%u.x?1.f:0.f,t%u.y>t%u.y?1.f:0.f,t%u.z>t%u.z?1.f:0.f,t%u.w>t%u.w?1.f:0.f);\n",
                    tid, a, b, a, b, a, b, a, b];
                else [s appendFormat:@"%@float t%u=t%u>t%u?1.f:0.f;\n", indent, tid, a, b]; break;
            case UOP_NEG:  [s appendFormat:@"%@%@ t%u=-t%u;\n", indent, ft, tid, a]; break;
            case UOP_RELU:
                if (use_f4) [s appendFormat:@"  float4 t%u=max(t%u,float4(0.f));\n", tid, a];
                else [s appendFormat:@"%@float t%u=max(t%u,0.f);\n", indent, tid, a]; break;
            case UOP_EXP:  [s appendFormat:@"%@%@ t%u=exp(t%u);\n", indent, ft, tid, a]; break;
            case UOP_LOG:  [s appendFormat:@"%@%@ t%u=log(t%u);\n", indent, ft, tid, a]; break;
            case UOP_SQRT: [s appendFormat:@"%@%@ t%u=sqrt(t%u);\n", indent, ft, tid, a]; break;
            default:       [s appendFormat:@"%@%@ t%u=t%u;\n", indent, ft, tid, a]; break;
        }
    }

    // ── Output write ───────────────────────────────────────────
    u32 last = n_leaves + n_ops - 1;
    if (has_reduce) {
        if (reduce->reduce_type == UOP_RMAX)
            [s appendFormat:@"    acc=max(acc,t%u);\n  }\n", last];
        else
            [s appendFormat:@"    acc+=t%u;\n  }\n", last];
        [s appendFormat:@"  uint oi=iz*%uu+iy*%uu+inner_base;\n  out[oi]=acc;\n",
            mid*inner, inner];
    } else if (use_f4) {
        [s appendFormat:@"  uint oi=iz*%uu+iy*%uu+inner_base;\n", mid*inner, inner];
        [s appendFormat:@"  *((device float4*)(out+oi))=t%u;\n", last];
    } else {
        [s appendFormat:@"  uint oi=iz*%uu+iy*%uu+inner_base;\n  out[oi]=t%u;\n",
            mid*inner, inner, last];
    }

    // Side output writes (shared intermediates)
    if (!has_reduce) { // side outputs only for non-reduce kernels
        for (u32 si = 0; si < n_side_outputs; si++) {
            u32 sid = side_op_indices[si]; // remapped op index = n_leaves + original_op_idx
            [s appendFormat:@"  side%u[oi]=t%u;\n", si, sid];
        }
    }

    [s appendString:@"}\n"];
    return s;
}

// ── Get or compile a cached kernel (new interface) ─────────────────
static id<MTLComputePipelineState> cg_get_pipe_rs(
        const FusedOp *ops, u32 n_ops, u32 n_leaves, const View **leaf_views,
        const Shape *full_shape, const ReduceSpec *reduce,
        const u32 *side_op_indices, u32 n_side_outputs) {
    u64 key = cg_hash_rs(ops, n_ops, n_leaves, leaf_views, full_shape, reduce);
    for (u32 i = 0; i < n_side_outputs; i++) { key ^= side_op_indices[i]; key *= 0x100000001b3ULL; }
    key ^= n_side_outputs; key *= 0x100000001b3ULL;
    for (u32 i = 0; i < cg_cache_count && i < CODEGEN_CACHE_SIZE; i++)
        if (cg_cache[i].key == key) return cg_cache[i].pipe;

    NSString *src = codegen_kernel_rs(ops, n_ops, n_leaves, leaf_views,
                                       full_shape, reduce, side_op_indices, n_side_outputs);
    NSError *err;
    id<MTLLibrary> lib = [mtl_dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"codegen error: %@\n%@", err, src); return nil; }
    id<MTLComputePipelineState> pipe =
        [mtl_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"K"]
                                               error:&err];
    if (!pipe) return nil;

    u32 slot = cg_cache_count < CODEGEN_CACHE_SIZE ?
        cg_cache_count++ : (cg_cache_count++ % CODEGEN_CACHE_SIZE);
    cg_cache[slot].key = key;
    cg_cache[slot].pipe = pipe;
    return pipe;
}

// ── Dispatch (new interface with ReduceSpec) ───────────────────────
void metal_dispatch_kernel_rs(u32 out_buf,
                               u32 *leaf_bufs, const View **leaf_views, u32 n_leaves,
                               FusedOp *ops, u32 n_ops,
                               const Shape *full_shape,
                               const ReduceSpec *reduce,
                               u32 *side_bufs, const u32 *side_op_indices, u32 n_side_outputs) {
    int has_reduce = reduce && reduce->reduce_type;
    u32 rank = full_shape->rank;
    if (rank == 0) rank = 1;

    // Classify axes (MUST match codegen exactly)
    u32 out_shape[MAX_DIM], n_out = 0;
    u32 out_dims[MAX_DIM];
    u32 out_numel = 1;
    for (u32 d = 0; d < rank; d++) {
        if (has_reduce && reduce->is_reduce[d]) continue;
        out_shape[n_out] = full_shape->dims[d];
        out_dims[n_out] = d;
        n_out++;
        out_numel *= full_shape->dims[d];
    }
    if (n_out == 0) { n_out = 1; out_shape[0] = 1; out_dims[0] = rank; }

    id<MTLComputePipelineState> pipe = cg_get_pipe_rs(ops, n_ops, n_leaves,
                                                       leaf_views, full_shape, reduce,
                                                       side_op_indices, n_side_outputs);
    if (!pipe) return;

    // Collapse output dims into 3 groups (MUST match codegen)
    u32 inner = 1, mid = 1, outer = 1;
    u32 inner_start = n_out;
    for (int i = (int)n_out - 1; i >= 0; i--) {
        if (inner * out_shape[i] <= 1024) { inner *= out_shape[i]; inner_start = (u32)i; }
        else break;
    }
    u32 mid_start = inner_start;
    for (int i = (int)inner_start - 1; i >= 0; i--) {
        if (mid * out_shape[i] <= 65535) { mid *= out_shape[i]; mid_start = (u32)i; }
        else break;
    }
    for (u32 i = 0; i < mid_start; i++) outer *= out_shape[i];

    // Float4 check (MUST match codegen)
    int use_f4 = !has_reduce && (inner % 4 == 0) && n_out > 0;
    if (use_f4 && out_dims[n_out - 1] < rank) {
        if (full_shape->dims[out_dims[n_out - 1]] % 4 != 0) use_f4 = 0;
    }
    if (use_f4) {
        for (u32 i = 0; i < n_leaves; i++) {
            u32 last_d = out_dims[n_out - 1];
            i32 ist = (last_d < leaf_views[i]->shape.rank)
                ? leaf_views[i]->strides[last_d] : 1;
            if (ist != 1 && ist != 0) { use_f4 = 0; break; }
        }
    }
    for (u32 i = 0; i < n_leaves; i++)
        if (leaf_views[i]->has_mask) { use_f4 = 0; break; }

    u32 gw = use_f4 ? inner / 4 : inner;
    u32 tw = MIN(gw, 256u);

    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:pipe];
    [enc setBuffer:metal_pool.bufs[out_buf] offset:0 atIndex:0];
    for (u32 si = 0; si < n_side_outputs; si++)
        [enc setBuffer:metal_pool.bufs[side_bufs[si]] offset:0 atIndex:si + 1];
    u32 buf_off = 1 + n_side_outputs;
    for (u32 i = 0; i < n_leaves; i++)
        [enc setBuffer:metal_pool.bufs[leaf_bufs[i]] offset:0 atIndex:buf_off + i];
    [enc dispatchThreads:MTLSizeMake(gw, mid, outer)
       threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    batch_dirty = 1;
    total_dispatches++;

    if (jit.state == JIT_CAPTURE) {
        id<MTLBuffer> bufs[17];
        bufs[0] = metal_pool.bufs[out_buf];
        for (u32 i = 0; i < n_leaves; i++) bufs[i+1] = metal_pool.bufs[leaf_bufs[i]];
        jit_record_dispatch_1d(pipe, bufs, n_leaves+1, NULL, NULL, 0,
                                gw, mid, outer, tw, 1, 1);
    }
}

// ════════════════════════════════════════════════════════════════════
// Old interface wrappers (for ops.m callers)
// ════════════════════════════════════════════════════════════════════

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

// Legacy wrapper — used by ops.m (unary/binary fast path) and metal_contiguify.
// New code should use metal_dispatch_kernel_rs or metal_dispatch_fused_rs directly.
static NSString *codegen_kernel(const FusedOp *ops, u32 n_ops, u32 n_leaves,
                                  const View **leaf_views,
                                  u32 out_numel __attribute__((unused)),
                                  int has_reduce, u32 reduce_dim,
                                  const Shape *out_shape_hint) {
    // Old callers: out_shape_hint is the full EW shape (including reduce dims).
    // has_reduce is 0 or UOP_SUM/UOP_RMAX. reduce_dim is product of trailing reduce dims.
    Shape full = {0};
    if (out_shape_hint && out_shape_hint->rank > 0) {
        full = *out_shape_hint;
    } else {
        u32 dims[MAX_DIM]; u32 rk;
        cg_output_shape(leaf_views, n_leaves, dims, &rk);
        full.rank = rk;
        for (u32 d = 0; d < rk; d++) full.dims[d] = dims[d];
    }
    if (has_reduce && reduce_dim > 1) {
        // Reconstruct ReduceSpec for trailing reduce
        ReduceSpec rs = {0};
        rs.reduce_type = (u32)has_reduce;
        u32 prod = 1;
        for (int d = (int)full.rank - 1; d >= 0; d--) {
            if (prod * full.dims[d] <= reduce_dim) {
                rs.is_reduce[d] = 1;
                prod *= full.dims[d];
                if (prod == reduce_dim) break;
            }
        }
        return codegen_kernel_rs(ops, n_ops, n_leaves, leaf_views, &full, &rs, NULL, 0);
    }
    return codegen_kernel_rs(ops, n_ops, n_leaves, leaf_views, &full, NULL, NULL, 0);
}

// Old get_pipe (wrapper)
static id<MTLComputePipelineState> cg_get_pipe(const FusedOp *ops, u32 n_ops,
                                                  u32 n_leaves, const View **leaf_views,
                                                  u32 out_numel, int has_reduce,
                                                  u32 reduce_dim,
                                                  const Shape *out_shape_hint) {
    u64 key = cg_hash(ops, n_ops, n_leaves, leaf_views, out_numel) ^ ((u64)has_reduce << 63);
    for (u32 i = 0; i < cg_cache_count && i < CODEGEN_CACHE_SIZE; i++)
        if (cg_cache[i].key == key) return cg_cache[i].pipe;

    NSString *src = codegen_kernel(ops, n_ops, n_leaves, leaf_views, out_numel,
                                     has_reduce, reduce_dim, out_shape_hint);
    NSError *err;
    id<MTLLibrary> lib = [mtl_dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"codegen error: %@\n%@", err, src); return nil; }
    id<MTLComputePipelineState> pipe =
        [mtl_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"K"]
                                               error:&err];
    if (!pipe) return nil;

    u32 slot = cg_cache_count < CODEGEN_CACHE_SIZE ?
        cg_cache_count++ : (cg_cache_count++ % CODEGEN_CACHE_SIZE);
    cg_cache[slot].key = key;
    cg_cache[slot].pipe = pipe;
    return pipe;
}

// Old dispatch (wrapper — used by ops.m, metal_contiguify, old metal_dispatch_fused_v2)
void metal_dispatch_kernel(u32 out_buf,
                            u32 out_numel __attribute__((unused)),
                            u32 *leaf_bufs, const View **leaf_views, u32 n_leaves,
                            FusedOp *ops, u32 n_ops,
                            int has_reduce, u32 reduce_dim,
                            const Shape *out_shape_hint) {
    // Construct full_shape + ReduceSpec from old params
    Shape full = {0};
    if (out_shape_hint && out_shape_hint->rank > 0) {
        full = *out_shape_hint;
    } else {
        u32 dims[MAX_DIM]; u32 rk;
        cg_output_shape(leaf_views, n_leaves, dims, &rk);
        full.rank = rk;
        for (u32 d = 0; d < rk; d++) full.dims[d] = dims[d];
    }
    if (has_reduce && reduce_dim > 1) {
        ReduceSpec rs = {0};
        rs.reduce_type = (u32)has_reduce;
        u32 prod = 1;
        for (int d = (int)full.rank - 1; d >= 0; d--) {
            if (prod * full.dims[d] <= reduce_dim) {
                rs.is_reduce[d] = 1;
                prod *= full.dims[d];
                if (prod == reduce_dim) break;
            }
        }
        metal_dispatch_kernel_rs(out_buf, leaf_bufs, leaf_views, n_leaves,
                                  ops, n_ops, &full, &rs, NULL, NULL, 0);
    } else {
        metal_dispatch_kernel_rs(out_buf, leaf_bufs, leaf_views, n_leaves,
                                  ops, n_ops, &full, NULL, NULL, NULL, 0);
    }
}
