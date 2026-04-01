// metal/codegen.m — General JIT kernel codegen for elementwise + reduce
// Generates optimal Metal kernels for any fused subgraph.
// Handles: elementwise chains, fused reduce (any axis config), masks, broadcasting.

// ReduceSpec: defined in tinyhvm.c (included before this file)
// typedef struct { u8 is_reduce[MAX_DIM]; u32 reduce_type; } ReduceSpec;

#define CODEGEN_CACHE_SIZE 256
static struct { u64 key; id<MTLComputePipelineState> pipe; u8 is_uop; u8 group_reduce; u32 local_size; } cg_cache[CODEGEN_CACHE_SIZE];
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
        if (reduce->reduce2_type) {
            h ^= (u64)reduce->reduce2_type << 48; h *= 0x100000001b3ULL;
            h ^= (u64)reduce->reduce2_start << 16; h *= 0x100000001b3ULL;
            for (u32 d = 0; d < full_shape->rank; d++)
                h ^= (u64)reduce->is_reduce2[d] << (d+8); h *= 0x100000001b3ULL;
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

static int _codegen_group_reduce = 0; // set by codegen for group reduce kernels

static NSString *codegen_kernel_rs(const FusedOp *ops, u32 n_ops, u32 n_leaves,
                                    const View **leaf_views,
                                    const ShapeTracker *const *leaf_sts, // NULL or per-leaf ST
                                    const Shape *full_shape,
                                    const ReduceSpec *reduce,
                                    const u32 *side_op_indices, u32 n_side_outputs) {
    _codegen_group_reduce = 0;
    u32 rank = full_shape->rank;
    if (rank == 0) rank = 1;

    // ── Classify axes ──────────────────────────────────────────
    u32 out_dims[MAX_DIM], n_out = 0; // output axis indices (in full shape)
    u32 red_dims[MAX_DIM], n_red = 0; // reduce axis indices
    u32 out_shape[MAX_DIM];           // output-axis-only dims (for dispatch grid)
    u32 out_numel = 1, reduce_numel = 1;
    int has_reduce = reduce && reduce->reduce_type;

    // Pre-compute reduce_numel for group_reduce decision
    if (has_reduce) {
        for (u32 d = 0; d < rank; d++)
            if (reduce->is_reduce[d]) reduce_numel *= full_shape->dims[d];
    }
    int group_reduce = has_reduce && reduce_numel >= 1024 && reduce->reduce_type == UOP_SUM;
    reduce_numel = 1; // reset — recomputed in the axis classification loop

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
    if (group_reduce) {
        // Group reduce: 1D dispatch, split gid.x into lane + flat output index
        [s appendFormat:@"  uint _lane=gid.x%%256u;\n"];
        [s appendFormat:@"  uint _oidx=gid.x/256u;\n"];
        [s appendFormat:@"  if(_oidx>=%uu)return;\n", out_numel];
        // Decompose flat output index into 3D (iz, iy, ix)
        [s appendFormat:@"  uint iz=_oidx/%uu,iy=(_oidx/%uu)%%%uu,ix=_oidx%%%uu;\n",
            mid * inner, inner, mid, inner];
        [s appendFormat:@"  uint inner_base=ix;\n"];
    } else {
        [s appendFormat:@"  uint ix=gid.x,iy=gid.y,iz=gid.z;\n"];
        [s appendFormat:@"  if(ix>=%uu||iy>=%uu||iz>=%uu)return;\n",
            inner_dispatch, mid, outer];
        if (use_f4) [s appendFormat:@"  uint inner_base=ix*4u;\n"];
        else [s appendFormat:@"  uint inner_base=ix;\n"];
    }

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
        if (group_reduce) {
            // Parallel reduce: contiguous block per thread (preserves serial sum order)
            [s appendFormat:@"  uint _blk=%uu/256u,_r0=_lane*_blk,_r1=min(_r0+_blk,%uu);\n",
                reduce_numel, reduce_numel];
            // Last thread handles remainder
            [s appendFormat:@"  if(_lane==255u)_r1=%uu;\n", reduce_numel];
            [s appendFormat:@"  for(uint r=_r0;r<_r1;r++){\n"];
        } else {
            [s appendFormat:@"  for(uint r=0;r<%uu;r++){\n", reduce_numel];
        }
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
                    if (lv->mod_size[d] > 0 && lv->mod_size[d] < lv->shape.dims[d]) {
                        // Modular indexing: (coord % mod_size) * stride
                        [s appendFormat:@"+(c%u%%%uu)*%du", d, lv->mod_size[d], lv->strides[d]];
                    } else if (lv->strides[d] == 1) {
                        [s appendFormat:@"+c%u", d];
                    } else {
                        [s appendFormat:@"+c%u*%du", d, lv->strides[d]];
                    }
                }
                [s appendString:@";\n"];
                continue;
            }
        }

        // Path D: ShapeTracker composition (views_to_indexed_uops port)
        // Process views right-to-left: compute flat idx from outer view,
        // unravel through each inner view, apply strides.
        if (leaf_sts && leaf_sts[li] && leaf_sts[li]->n_views >= 2) {
            const ShapeTracker *st = leaf_sts[li];
            // Step 1: compute flat index from kernel coords (= views[n-1] space)
            u32 fs_d[MAX_DIM];
            if (rank > 0) {
                fs_d[rank - 1] = 1;
                for (int d = (int)rank - 2; d >= 0; d--)
                    fs_d[d] = fs_d[d + 1] * full_shape->dims[d + 1];
            }
            [s appendFormat:@"%@uint si%u=", indent, li];
            for (u32 d = 0; d < rank; d++) {
                if (d > 0) [s appendString:@"+"];
                if (fs_d[d] == 1) [s appendFormat:@"c%u", d];
                else [s appendFormat:@"c%u*%uu", d, fs_d[d]];
            }
            [s appendString:@";\n"];

            // Step 2: for each view from n-2 down to 0, unravel + apply strides
            for (int vi = (int)st->n_views - 2; vi >= 0; vi--) {
                const View *vw = &st->views[vi];
                // Unravel si through views[vi+1].shape → multi-dim coords → apply vw strides
                const View *outer = &st->views[vi + 1];
                // Compute unravel divisors for outer shape
                u32 udiv = 1;
                [s appendFormat:@"%@si%u=%d", indent, li, vw->offset];
                for (int d = (int)outer->shape.rank - 1; d >= 0; d--) {
                    u32 dim = outer->shape.dims[d];
                    if (dim <= 1) { udiv *= dim; continue; }
                    // coord_d = (si / udiv) % dim
                    // Then apply vw stride if vw has this dim
                    if ((u32)d < vw->shape.rank && vw->strides[d] != 0) {
                        if (udiv == 1)
                            [s appendFormat:@"+(si%u%%%uu)*%du", li, dim, vw->strides[d]];
                        else
                            [s appendFormat:@"+((si%u/%uu)%%%uu)*%du", li, udiv, dim, vw->strides[d]];
                    }
                    udiv *= dim;
                }
                [s appendString:@";\n"];
                // Handle mask on this view
                if (vw->has_mask) {
                    u32 mdiv = 1;
                    [s appendFormat:@"%@bool vm%u_%u=true", indent, li, vi];
                    for (int d = (int)outer->shape.rank - 1; d >= 0; d--) {
                        u32 dim = outer->shape.dims[d];
                        if (dim <= 1) { mdiv *= dim; continue; }
                        if ((u32)d < vw->shape.rank &&
                            (vw->mask_begin[d] > 0 || vw->mask_end[d] < vw->shape.dims[d])) {
                            if (mdiv == 1) {
                                if (vw->mask_begin[d] > 0)
                                    [s appendFormat:@"&&(si%u%%%uu)>=%uu", li, dim, vw->mask_begin[d]];
                                if (vw->mask_end[d] < vw->shape.dims[d])
                                    [s appendFormat:@"&&(si%u%%%uu)<%uu", li, dim, vw->mask_end[d]];
                            } else {
                                if (vw->mask_begin[d] > 0)
                                    [s appendFormat:@"&&((si%u/%uu)%%%uu)>=%uu", li, mdiv, dim, vw->mask_begin[d]];
                                if (vw->mask_end[d] < vw->shape.dims[d])
                                    [s appendFormat:@"&&((si%u/%uu)%%%uu)<%uu", li, mdiv, dim, vw->mask_end[d]];
                            }
                        }
                        mdiv *= dim;
                    }
                    [s appendString:@";\n"];
                }
            }
            [s appendFormat:@"%@uint i%u=si%u;\n", indent, li, li];
            continue;
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
            u32 dim = lv->shape.dims[d];
            u32 mod = (lv->mod_size[d] > 0 && lv->mod_size[d] < dim) ? lv->mod_size[d] : dim;
            if (leaf_div == 1 && lv->strides[d] == 1)
                [s appendFormat:@"+(fi%u%%%uu)", li, mod];
            else if (leaf_div == 1)
                [s appendFormat:@"+(fi%u%%%uu)*%du", li, mod, lv->strides[d]];
            else
                [s appendFormat:@"+((fi%u/%uu)%%%uu)*%du", li, leaf_div,
                    mod, lv->strides[d]];
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
        // Compound masks: begin <= c_a * stride_a + c_b < end
        for (u32 cm = 0; cm < lv->n_compound_masks; cm++) {
            u32 da = lv->compound_masks[cm].dim_a;
            u32 db = lv->compound_masks[cm].dim_b;
            i32 sa = lv->compound_masks[cm].stride_a;
            u32 mb = lv->compound_masks[cm].begin;
            u32 me = lv->compound_masks[cm].end;
            if (!first) [s appendString:@"&&"];
            first = 0;
            [s appendFormat:@"(c%u*%d+c%u)>=%uu&&(c%u*%d+c%u)<%uu",
                da, sa, db, mb, da, sa, db, me];
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
        // Only read pre-reduce leaves inside the reduce loop; post-reduce leaves read after
        u32 n_pre_leaves = (has_reduce && reduce->post_reduce_start > 0)
            ? (n_leaves - reduce->n_post_leaves) : n_leaves;
        for (u32 li = 0; li < n_pre_leaves; li++) {
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

    // ── Op chain (pre-reduce only; post-reduce emitted after the loop) ──
    NSString *ft = use_f4 ? @"float4" : @"float";
    u32 n_pre_ops = (has_reduce && reduce->post_reduce_start > 0) ? reduce->post_reduce_start : n_ops;
    for (u32 i = 0; i < n_pre_ops; i++) {
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
    u32 last = n_leaves + n_pre_ops - 1;
    if (has_reduce) {
        // Side outputs INSIDE reduce loop (before acc, while intermediates are live)
        if (n_side_outputs > 0) {
            [s appendString:@"    uint soi="];
            u32 stride = 1;
            for (int d = (int)rank - 1; d >= 0; d--) {
                if (d < (int)rank - 1) [s appendString:@"+"];
                [s appendFormat:@"c%u*%uu", d, stride];
                stride *= full_shape->dims[d];
            }
            [s appendString:@";\n"];
            for (u32 si = 0; si < n_side_outputs; si++) {
                u32 sid = side_op_indices[si];
                [s appendFormat:@"    side%u[soi]=t%u;\n", si, sid];
            }
        }
        if (reduce->reduce_type == UOP_RMAX)
            [s appendFormat:@"    acc=max(acc,t%u);\n  }\n", last];
        else
            [s appendFormat:@"    acc+=t%u;\n  }\n", last];
        if (group_reduce) {
            // Serial cross-thread reduction: preserves summation order
            [s appendString:@"  threadgroup float _sh[256];\n"];
            [s appendString:@"  _sh[_lane]=acc;\n"];
            [s appendString:@"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"];
            [s appendString:@"  if(_lane!=0u)return;\n"];
            [s appendString:@"  acc=0.f;for(uint _i=0;_i<256u;_i++)acc+=_sh[_i];\n"];
        }
        [s appendFormat:@"  uint oi=iz*%uu+iy*%uu+inner_base;\n",
            mid*inner, inner];

        // Post-reduce ops: applied to acc using output (non-reduce) coordinates.
        // Uses a separate variable namespace (p0, p1, ...) to avoid conflicts with
        // pre-reduce t-variables that are scoped inside the reduce loop.
        u32 prs = reduce->post_reduce_start;
        if (prs > 0 && prs < n_ops) {
            // Read post-reduce-only leaves using output coordinates
            u32 post_leaf_base = n_leaves - reduce->n_post_leaves;
            for (u32 pli = post_leaf_base; pli < n_leaves; pli++) {
                const View *plv = leaf_views[pli];
                [s appendString:@"  float pl"];
                [s appendFormat:@"%u=in%u[", pli - post_leaf_base, pli];
                // Coordinate-based index using non-reduce dims
                int first = 1;
                for (u32 oi2 = 0; oi2 < n_out; oi2++) {
                    u32 dim = out_dims[oi2];
                    if (dim < plv->shape.rank && plv->strides[dim] > 0) {
                        if (!first) [s appendString:@"+"];
                        [s appendFormat:@"c%u*%du", dim, plv->strides[dim]];
                        first = 0;
                    }
                }
                if (first) [s appendString:@"0"];
                [s appendString:@"];\n"];
            }
            // Emit post-reduce chain: p0 = acc, p1 = f(p0, leaf), ...
            for (u32 pi = prs; pi < n_ops; pi++) {
                u32 pidx = pi - prs;
                NSString *a_name = (pidx == 0) ? @"acc" : [NSString stringWithFormat:@"p%u", pidx - 1];
                NSString *b_name = nil;
                if (is_binary(ops[pi].uop)) {
                    // arg_b should reference a post-reduce leaf
                    u32 b = ops[pi].arg_b;
                    if (b >= post_leaf_base && b < n_leaves)
                        b_name = [NSString stringWithFormat:@"pl%u", b - post_leaf_base];
                    else
                        b_name = [NSString stringWithFormat:@"t%u", b]; // fallback
                }
                switch (ops[pi].uop) {
                    case UOP_ADD:  [s appendFormat:@"  float p%u=%@+%@;\n", pidx, a_name, b_name]; break;
                    case UOP_SUB:  [s appendFormat:@"  float p%u=%@-%@;\n", pidx, a_name, b_name]; break;
                    case UOP_MUL:  [s appendFormat:@"  float p%u=%@*%@;\n", pidx, a_name, b_name]; break;
                    case UOP_DIV:  [s appendFormat:@"  float p%u=%@/%@;\n", pidx, a_name, b_name]; break;
                    case UOP_NEG:  [s appendFormat:@"  float p%u=-%@;\n", pidx, a_name]; break;
                    case UOP_RELU: [s appendFormat:@"  float p%u=max(%@,0.f);\n", pidx, a_name]; break;
                    case UOP_EXP:  [s appendFormat:@"  float p%u=exp(%@);\n", pidx, a_name]; break;
                    case UOP_LOG:  [s appendFormat:@"  float p%u=log(%@);\n", pidx, a_name]; break;
                    default:       [s appendFormat:@"  float p%u=%@;\n", pidx, a_name]; break;
                }
            }

            // ── Second reduce phase (multi-reduce) ────────────────
            if (reduce->reduce2_type && reduce->reduce2_start > 0 && reduce->reduce2_start < n_ops) {
                u32 r2s = reduce->reduce2_start;
                // The last between-reduce op result is the input to reduce2
                u32 between_last = (r2s > prs + 1) ? r2s - prs - 1 : 0;
                NSString *r2_input = (between_last > 0)
                    ? [NSString stringWithFormat:@"p%u", between_last]
                    : @"acc";

                // Compute reduce2 axes (may be same as reduce1)
                u32 r2_numel = 1;
                u32 r2_dims[MAX_DIM]; u32 n_r2 = 0;
                for (u32 d = 0; d < rank; d++) {
                    if (reduce->is_reduce2[d]) {
                        r2_dims[n_r2++] = d;
                        r2_numel *= full_shape->dims[d];
                    }
                }
                if (n_r2 == 0) { // same axes as reduce1
                    n_r2 = n_red; r2_numel = reduce_numel;
                    for (u32 i = 0; i < n_red; i++) r2_dims[i] = red_dims[i];
                }

                [s appendFormat:@"  float acc2=%s;\n",
                    reduce->reduce2_type == UOP_RMAX ? "-1e30f" : "0.0f"];
                [s appendFormat:@"  for(uint r2=0;r2<%uu;r2++){\n", r2_numel];
                // Reduce2 axis coordinates
                u32 r2div = 1;
                for (int ri = (int)n_r2 - 1; ri >= 0; ri--) {
                    u32 d = r2_dims[ri];
                    u32 dim = full_shape->dims[d];
                    if (r2div == 1)
                        [s appendFormat:@"    uint c%u=r2%%%uu;\n", d, dim];
                    else
                        [s appendFormat:@"    uint c%u=(r2/%uu)%%%uu;\n", d, r2div, dim];
                    r2div *= dim;
                }
                // Re-compute leaf indices inside reduce2 loop (c%u changed by r2 decomp)
                u32 n_pre2_leaves = n_leaves - reduce->n_post_leaves;
                for (u32 li = 0; li < n_pre2_leaves; li++) {
                    const View *lv = leaf_views[li];
                    // Coordinate-based index (same as path B above)
                    if (lv->shape.rank == rank) {
                        [s appendFormat:@"    uint r2i%u=%d", li, lv->offset];
                        for (u32 d = 0; d < rank; d++) {
                            if (lv->strides[d] == 0 || lv->shape.dims[d] == 1) continue;
                            if (lv->strides[d] == 1) [s appendFormat:@"+c%u", d];
                            else [s appendFormat:@"+c%u*%du", d, lv->strides[d]];
                        }
                        [s appendString:@";\n"];
                    } else {
                        // Fallback: flat index decomposition
                        u32 fss[MAX_DIM];
                        if (rank > 0) { fss[rank-1]=1; for(int d=(int)rank-2;d>=0;d--) fss[d]=fss[d+1]*full_shape->dims[d+1]; }
                        [s appendFormat:@"    uint r2fi%u=", li];
                        for (u32 d=0;d<rank;d++) { if(d>0)[s appendString:@"+"]; [s appendFormat:@"c%u*%uu",d,fss[d]]; }
                        [s appendString:@";\n"];
                        [s appendFormat:@"    uint r2i%u=%d", li, lv->offset];
                        u32 ld=1;
                        for(int d=(int)lv->shape.rank-1;d>=0;d--) {
                            if(lv->strides[d]==0){ld*=lv->shape.dims[d];continue;}
                            if(ld==1&&lv->strides[d]==1) [s appendFormat:@"+(r2fi%u%%%uu)",li,lv->shape.dims[d]];
                            else if(ld==1) [s appendFormat:@"+(r2fi%u%%%uu)*%du",li,lv->shape.dims[d],lv->strides[d]];
                            else [s appendFormat:@"+((r2fi%u/%uu)%%%uu)*%du",li,ld,lv->shape.dims[d],lv->strides[d]];
                            ld*=lv->shape.dims[d];
                        }
                        [s appendString:@";\n"];
                    }
                    if (lv->has_mask)
                        [s appendFormat:@"    float r2t%u=m%u?in%u[r2i%u]:0.f;\n", li, li, li, li];
                    else
                        [s appendFormat:@"    float r2t%u=in%u[r2i%u];\n", li, li, li];
                }

                // Emit reduce2 phase ops
                for (u32 pi = r2s; pi < n_ops; pi++) {
                    u32 pidx = pi - r2s;
                    // In reduce2 phase, references to leaves use r2t%u, references to
                    // between-reduce results use the p%u namespace
                    NSString *a_name, *b_name = nil;
                    u32 a = ops[pi].arg_a, b = ops[pi].arg_b;

                    // Map arg references: leaves → r2t, between-reduce → p, acc → acc
                    if (a < n_pre2_leaves) a_name = [NSString stringWithFormat:@"r2t%u", a];
                    else if (a >= n_leaves && a < n_leaves + prs) a_name = @"acc"; // reduce1 result
                    else if (a >= n_leaves + prs) a_name = [NSString stringWithFormat:@"p%u", a - n_leaves - prs];
                    else a_name = [NSString stringWithFormat:@"r2t%u", a];

                    if (is_binary(ops[pi].uop)) {
                        if (b < n_pre2_leaves) b_name = [NSString stringWithFormat:@"r2t%u", b];
                        else if (b >= n_leaves && b < n_leaves + prs) b_name = @"acc";
                        else if (b >= n_leaves + prs) b_name = [NSString stringWithFormat:@"p%u", b - n_leaves - prs];
                        else b_name = [NSString stringWithFormat:@"r2t%u", b];
                    }

                    switch (ops[pi].uop) {
                        case UOP_ADD:  [s appendFormat:@"    float r2v%u=%@+%@;\n", pidx, a_name, b_name]; break;
                        case UOP_SUB:  [s appendFormat:@"    float r2v%u=%@-%@;\n", pidx, a_name, b_name]; break;
                        case UOP_MUL:  [s appendFormat:@"    float r2v%u=%@*%@;\n", pidx, a_name, b_name]; break;
                        case UOP_DIV:  [s appendFormat:@"    float r2v%u=%@/%@;\n", pidx, a_name, b_name]; break;
                        case UOP_NEG:  [s appendFormat:@"    float r2v%u=-%@;\n", pidx, a_name]; break;
                        case UOP_RELU: [s appendFormat:@"    float r2v%u=max(%@,0.f);\n", pidx, a_name]; break;
                        case UOP_EXP:  [s appendFormat:@"    float r2v%u=exp(%@);\n", pidx, a_name]; break;
                        case UOP_LOG:  [s appendFormat:@"    float r2v%u=log(%@);\n", pidx, a_name]; break;
                        case UOP_SQRT: [s appendFormat:@"    float r2v%u=sqrt(%@);\n", pidx, a_name]; break;
                        default:       [s appendFormat:@"    float r2v%u=%@;\n", pidx, a_name]; break;
                    }
                }
                u32 r2_last = n_ops - r2s - 1;
                if (reduce->reduce2_type == UOP_RMAX)
                    [s appendFormat:@"    acc2=max(acc2,r2v%u);\n  }\n", r2_last];
                else
                    [s appendFormat:@"    acc2+=r2v%u;\n  }\n", r2_last];
                [s appendFormat:@"  out[oi]=acc2;\n"];
            } else {
                [s appendFormat:@"  out[oi]=p%u;\n", n_ops - prs - 1];
            }
        } else if (reduce->reduce2_type && reduce->reduce2_start < n_ops) {
            // reduce1 → reduce2 directly (no between-reduce ew ops)
            // All ops are inside reduce2 loop, referencing acc and leaves
            u32 r2s = reduce->reduce2_start;
            u32 r2_numel = 1;
            u32 r2_dims[MAX_DIM]; u32 n_r2 = 0;
            for (u32 d = 0; d < rank; d++) {
                if (reduce->is_reduce2[d]) { r2_dims[n_r2++] = d; r2_numel *= full_shape->dims[d]; }
            }
            if (n_r2 == 0) { n_r2 = n_red; r2_numel = reduce_numel;
                for (u32 i = 0; i < n_red; i++) r2_dims[i] = red_dims[i]; }

            [s appendFormat:@"  float acc2=%s;\n",
                reduce->reduce2_type == UOP_RMAX ? "-1e30f" : "0.0f"];
            [s appendFormat:@"  for(uint r2=0;r2<%uu;r2++){\n", r2_numel];
            u32 r2div = 1;
            for (int ri = (int)n_r2 - 1; ri >= 0; ri--) {
                u32 d = r2_dims[ri]; u32 dim = full_shape->dims[d];
                if (r2div == 1) [s appendFormat:@"    uint c%u=r2%%%uu;\n", d, dim];
                else [s appendFormat:@"    uint c%u=(r2/%uu)%%%uu;\n", d, r2div, dim];
                r2div *= dim;
            }
            // Re-compute leaf indices + read leaves
            u32 n_pre2_leaves = n_leaves - reduce->n_post_leaves;
            for (u32 li = 0; li < n_pre2_leaves; li++) {
                const View *lv = leaf_views[li];
                if (lv->shape.rank == rank) {
                    [s appendFormat:@"    uint r2i%u=%d", li, lv->offset];
                    for (u32 d = 0; d < rank; d++) {
                        if (lv->strides[d] == 0 || lv->shape.dims[d] == 1) continue;
                        if (lv->strides[d] == 1) [s appendFormat:@"+c%u", d];
                        else [s appendFormat:@"+c%u*%du", d, lv->strides[d]];
                    }
                    [s appendString:@";\n"];
                } else {
                    [s appendFormat:@"    uint r2i%u=0;\n", li]; // fallback
                }
                [s appendFormat:@"    float r2t%u=in%u[r2i%u];\n", li, li, li];
            }
            // Emit reduce2 ops
            for (u32 pi = r2s; pi < n_ops; pi++) {
                u32 pidx = pi - r2s, a = ops[pi].arg_a, b = ops[pi].arg_b;
                NSString *a_name = (a < n_pre2_leaves) ? [NSString stringWithFormat:@"r2t%u", a]
                    : (a == n_leaves) ? @"acc" : [NSString stringWithFormat:@"r2v%u", a - n_leaves - 1];
                NSString *b_name = nil;
                if (is_binary(ops[pi].uop)) {
                    b_name = (b < n_pre2_leaves) ? [NSString stringWithFormat:@"r2t%u", b]
                        : (b == n_leaves) ? @"acc" : [NSString stringWithFormat:@"r2v%u", b - n_leaves - 1];
                }
                switch (ops[pi].uop) {
                    case UOP_ADD:  [s appendFormat:@"    float r2v%u=%@+%@;\n", pidx, a_name, b_name]; break;
                    case UOP_SUB:  [s appendFormat:@"    float r2v%u=%@-%@;\n", pidx, a_name, b_name]; break;
                    case UOP_MUL:  [s appendFormat:@"    float r2v%u=%@*%@;\n", pidx, a_name, b_name]; break;
                    case UOP_DIV:  [s appendFormat:@"    float r2v%u=%@/%@;\n", pidx, a_name, b_name]; break;
                    case UOP_NEG:  [s appendFormat:@"    float r2v%u=-%@;\n", pidx, a_name]; break;
                    case UOP_EXP:  [s appendFormat:@"    float r2v%u=exp(%@);\n", pidx, a_name]; break;
                    case UOP_LOG:  [s appendFormat:@"    float r2v%u=log(%@);\n", pidx, a_name]; break;
                    case UOP_SQRT: [s appendFormat:@"    float r2v%u=sqrt(%@);\n", pidx, a_name]; break;
                    default:       [s appendFormat:@"    float r2v%u=%@;\n", pidx, a_name]; break;
                }
            }
            u32 r2_last = n_ops - r2s - 1;
            if (reduce->reduce2_type == UOP_RMAX)
                [s appendFormat:@"    acc2=max(acc2,r2v%u);\n  }\n", r2_last];
            else
                [s appendFormat:@"    acc2+=r2v%u;\n  }\n", r2_last];
            [s appendFormat:@"  out[oi]=acc2;\n"];
        } else {
            [s appendFormat:@"  out[oi]=acc;\n"];
        }
    } else if (use_f4) {
        [s appendFormat:@"  uint oi=iz*%uu+iy*%uu+inner_base;\n", mid*inner, inner];
        [s appendFormat:@"  *((device float4*)(out+oi))=t%u;\n", last];
    } else {
        [s appendFormat:@"  uint oi=iz*%uu+iy*%uu+inner_base;\n  out[oi]=t%u;\n",
            mid*inner, inner, last];
    }

    // Side output writes for non-reduce kernels
    if (!has_reduce) {
        for (u32 si = 0; si < n_side_outputs; si++) {
            u32 sid = side_op_indices[si];
            if (use_f4)
                [s appendFormat:@"  *((device float4*)(side%u+oi))=t%u;\n", si, sid];
            else
                [s appendFormat:@"  side%u[oi]=t%u;\n", si, sid];
        }
    }

    [s appendString:@"}\n"];
    _codegen_group_reduce = group_reduce;
    return s;
}

// ── Get or compile a cached kernel (new interface) ─────────────────
static id<MTLComputePipelineState> cg_get_pipe_rs(
        const FusedOp *ops, u32 n_ops, u32 n_leaves, const View **leaf_views,
        const ShapeTracker *const *leaf_sts,
        const Shape *full_shape, const ReduceSpec *reduce,
        const u32 *side_op_indices, u32 n_side_outputs) {
    u64 key = cg_hash_rs(ops, n_ops, n_leaves, leaf_views, full_shape, reduce);
    for (u32 i = 0; i < n_side_outputs; i++) { key ^= side_op_indices[i]; key *= 0x100000001b3ULL; }
    key ^= n_side_outputs; key *= 0x100000001b3ULL;
    for (u32 i = 0; i < cg_cache_count && i < CODEGEN_CACHE_SIZE; i++)
        if (cg_cache[i].key == key) {
            _last_compiled_uop = cg_cache[i].is_uop;
            _last_local_size = cg_cache[i].group_reduce ? (int)cg_cache[i].local_size : 0;
            return cg_cache[i].pipe;
        }

    NSString *src;
    // UOp IR path: build IR → render MSL (when THVM_UOP=1 and no side outputs)
    static int _use_uop = -1;
    if (_use_uop < 0) _use_uop = getenv("THVM_NO_UOP") == NULL; // UOp on by default
    int has_reduce_r = reduce && reduce->reduce_type;
    _last_compiled_uop = 0;
    _last_local_size = 0;
    if (0 && _use_uop && n_side_outputs == 0 && !has_reduce_r) {
        static UOpKernel uk; // static: avoid 10KB stack allocation in recursive calls
        if (uop_from_fused(&uk, ops, n_ops, n_leaves, leaf_views, full_shape, reduce)) {
            src = uop_render_msl(&uk);
            _last_compiled_uop = 1;
            _last_local_size = (int)uk.local_size;
        } else
            src = codegen_kernel_rs(ops, n_ops, n_leaves, leaf_views, leaf_sts,
                                     full_shape, reduce, side_op_indices, n_side_outputs);
            if (_codegen_group_reduce) _last_local_size = 256;
    } else {
        src = codegen_kernel_rs(ops, n_ops, n_leaves, leaf_views, leaf_sts,
                                 full_shape, reduce, side_op_indices, n_side_outputs);
        if (_codegen_group_reduce) _last_local_size = 256;
    }
    if (getenv("THVM_DUMP_CODEGEN")) fprintf(stderr, "--- codegen (n_ops=%u n_leaves=%u n_side=%u cmd=%u) ---\n%s\n---\n", n_ops, n_leaves, n_side_outputs, jit.n_cmds, [src UTF8String]);
    NSError *err;
    id<MTLLibrary> lib = [mtl_dev newLibraryWithSource:src options:nil error:&err];
    if (!lib && _use_uop && n_side_outputs == 0) {
        // UOp MSL failed to compile — fall back to old codegen
        src = codegen_kernel_rs(ops, n_ops, n_leaves, leaf_views, leaf_sts,
                                 full_shape, reduce, side_op_indices, n_side_outputs);
        lib = [mtl_dev newLibraryWithSource:src options:nil error:&err];
    }
    if (!lib) { NSLog(@"codegen error: %@\n%@", err, src); return nil; }
    id<MTLComputePipelineState> pipe =
        [mtl_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"K"]
                                               error:&err];
    if (!pipe) return nil;

    u32 slot = cg_cache_count < CODEGEN_CACHE_SIZE ?
        cg_cache_count++ : (cg_cache_count++ % CODEGEN_CACHE_SIZE);
    cg_cache[slot].key = key;
    cg_cache[slot].pipe = pipe;
    cg_cache[slot].is_uop = _last_compiled_uop;
    cg_cache[slot].group_reduce = (_last_local_size > 0);
    cg_cache[slot].local_size = (u32)_last_local_size;
    return pipe;
}

// ── Dispatch (new interface with ReduceSpec) ───────────────────────
static void metal_dispatch_kernel_rs_st(u32 out_buf,
    u32 *leaf_bufs, const View **leaf_views,
    const ShapeTracker *const *leaf_sts, u32 n_leaves,
    FusedOp *ops, u32 n_ops, const Shape *full_shape, const ReduceSpec *reduce,
    u32 *side_bufs, const u32 *side_op_indices, u32 n_side_outputs);

void metal_dispatch_kernel_rs(u32 out_buf,
                               u32 *leaf_bufs, const View **leaf_views, u32 n_leaves,
                               FusedOp *ops, u32 n_ops,
                               const Shape *full_shape,
                               const ReduceSpec *reduce,
                               u32 *side_bufs, const u32 *side_op_indices, u32 n_side_outputs) {
    // No ShapeTracker — delegate to the full version
    const ShapeTracker *null_sts[FUSE_MAX_LEAVES];
    memset(null_sts, 0, sizeof(null_sts));
    metal_dispatch_kernel_rs_st(out_buf, leaf_bufs, leaf_views, null_sts, n_leaves,
                                 ops, n_ops, full_shape, reduce,
                                 side_bufs, side_op_indices, n_side_outputs);
}

void metal_dispatch_kernel_rs_st(u32 out_buf,
                               u32 *leaf_bufs, const View **leaf_views,
                               const ShapeTracker *const *leaf_sts, u32 n_leaves,
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
                                                       leaf_views, leaf_sts,
                                                       full_shape, reduce,
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

    // Float4 check (MUST match codegen).
    // UOp kernels are scalar — skip float4 grid reduction for them.
    int use_f4 = !_last_compiled_uop && !has_reduce && (inner % 4 == 0) && n_out > 0;
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
    [enc setBuffer:metal_pool.bufs[out_buf] offset:BUF_OFFSET(out_buf) atIndex:0];
    for (u32 si = 0; si < n_side_outputs; si++)
        [enc setBuffer:metal_pool.bufs[side_bufs[si]] offset:BUF_OFFSET(side_bufs[si]) atIndex:si + 1];
    u32 buf_off = 1 + n_side_outputs;
    for (u32 i = 0; i < n_leaves; i++)
        [enc setBuffer:metal_pool.bufs[leaf_bufs[i]] offset:BUF_OFFSET(leaf_bufs[i]) atIndex:buf_off + i];
    if (_last_local_size > 0) {
        // GROUP_REDUCE: dispatch threadgroups with explicit local size
        [enc dispatchThreadgroups:MTLSizeMake(out_numel, 1, 1)
           threadsPerThreadgroup:MTLSizeMake((u32)_last_local_size, 1, 1)];
    } else {
        [enc dispatchThreads:MTLSizeMake(gw, mid, outer)
           threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    }
    batch_dirty = 1;
    buf_cpu_only[out_buf] = 0;
    for (u32 si = 0; si < n_side_outputs; si++) buf_cpu_only[side_bufs[si]] = 0;
    total_dispatches++;
    dc[has_reduce ? DC_REDUCE : DC_FUSED]++;

    // Per-kernel profiling (THVM_DEBUG>=2)
    { u64 ops_est = (u64)out_numel * (n_ops > 0 ? n_ops : 1);
      u64 mem_est = (u64)(out_numel + n_leaves * out_numel) * 4;
      if (has_reduce) { u32 rn=1; for(u32 d=0;d<rank;d++) if(reduce->is_reduce[d]) rn*=full_shape->dims[d];
          ops_est = (u64)out_numel * rn * (n_ops > 0 ? n_ops : 1);
          mem_est = (u64)(out_numel * rn * n_leaves + out_numel) * 4; }
      profiled_dispatch(out_numel, ops_est, mem_est,
          has_reduce ? "fused_reduce" : (n_ops > 0 ? "fused_ew" : "copy")); }

    if (jit.state == JIT_CAPTURE) {
        u32 ids[24];
        ids[0] = out_buf;
        for (u32 si = 0; si < n_side_outputs; si++)
            ids[si + 1] = side_bufs[si];
        u32 jit_buf_off = 1 + n_side_outputs;
        for (u32 i = 0; i < n_leaves; i++)
            ids[jit_buf_off + i] = leaf_bufs[i];
        jit_record_dispatch_ids(pipe, ids, jit_buf_off + n_leaves, NULL, NULL, 0,
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
        return codegen_kernel_rs(ops, n_ops, n_leaves, leaf_views, NULL, &full, &rs, NULL, 0);
    }
    return codegen_kernel_rs(ops, n_ops, n_leaves, leaf_views, NULL, &full, NULL, NULL, 0);
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
