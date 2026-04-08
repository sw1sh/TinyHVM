// cpu/dispatch.c — CPU interpreter for fused kernel specs.
// Same interface as Metal's dispatch_kernel_rs: evaluates FusedOp DAG
// over leaf views with optional reduction.

static inline f32 eval_uop(u32 uop, f32 a, f32 b) {
    switch (uop) {
        case UOP_ADD:  return a + b;
        case UOP_MUL:  return a * b;
        case UOP_DIV:  return b != 0.f ? a / b : 0.f;
        case UOP_SUB:  return a - b;
        case UOP_MAX:  return a > b ? a : b;
        case UOP_CMP:  return a > b ? 1.f : 0.f;
        case UOP_NEG:  return -a;
        case UOP_RELU: return a > 0.f ? a : 0.f;
        case UOP_EXP:  return __builtin_expf(a);
        case UOP_LOG:  return __builtin_logf(a);
        case UOP_SQRT: return __builtin_sqrtf(a);
        default:       return a;
    }
}

// Read a single element via multi-view ShapeTracker.
// Handles rank-changing reshapes: computes flat index through outer shape,
// then decomposes through inner shape to apply inner strides.
static inline f32 leaf_read_st(const f32 *buf, const ShapeTracker *st,
                                const u32 *coords, u32 rank,
                                const Shape *full_shape) {
    if (!st || st->n_views <= 1) return 0.f;
    // Step 1: compute flat index from kernel coords in full_shape space
    u32 si = 0;
    { u32 stride = 1;
      for (i32 d = (i32)rank - 1; d >= 0; d--) {
        si += coords[d] * stride;
        stride *= full_shape->dims[d];
      }
    }
    // Step 2: for each view pair, convert flat index through reshape
    for (i32 vi = (i32)st->n_views - 2; vi >= 0; vi--) {
        const View *vw = &st->views[vi];
        const View *outer = &st->views[vi + 1];
        // Pass 1: extract flat index in outer's contiguous space from si.
        // For broadcast dims (outer dim=1), coordinate is 0 (no contribution).
        u32 outer_flat = 0;
        { u32 udiv = 1;
          u32 ostrides[MAX_DIM];
          if (outer->shape.rank > 0) {
            ostrides[outer->shape.rank - 1] = 1;
            for (i32 d = (i32)outer->shape.rank - 2; d >= 0; d--)
                ostrides[d] = ostrides[d + 1] * outer->shape.dims[d + 1];
          }
          for (i32 d = (i32)outer->shape.rank - 1; d >= 0; d--) {
            u32 dim = outer->shape.dims[d];
            if (dim <= 1) { udiv *= dim; continue; }
            u32 c = (si / udiv) % dim;
            outer_flat += c * ostrides[d];
            udiv *= dim;
          }
        }
        // Pass 2: decompose outer_flat through inner (vw) shape, apply strides.
        // The flat index is preserved across reshape (same numel).
        i32 idx = vw->offset;
        int masked = 0;
        { u32 rem = outer_flat;
          u32 istrides[MAX_DIM];
          if (vw->shape.rank > 0) {
            istrides[vw->shape.rank - 1] = 1;
            for (i32 d = (i32)vw->shape.rank - 2; d >= 0; d--)
                istrides[d] = istrides[d + 1] * vw->shape.dims[d + 1];
          }
          for (u32 d = 0; d < vw->shape.rank; d++) {
            u32 c = rem / istrides[d];
            rem %= istrides[d];
            if (vw->has_mask && (c < vw->mask_begin[d] || c >= vw->mask_end[d]))
                { masked = 1; break; }
            if (vw->strides[d] != 0)
                idx += (i32)c * vw->strides[d];
          }
        }
        if (masked) return 0.f;
        if (idx < 0) return 0.f;
        si = (u32)idx;
    }
    return buf[si];
}

// Read a single element from a leaf buffer using its view.
// coords[] are in the full_shape coordinate space.
static inline f32 leaf_read(const f32 *buf, const View *v,
                            const u32 *coords, u32 rank) {
    // Mask check: if coordinate is outside valid range, return 0
    if (v->has_mask) {
        for (u32 d = 0; d < rank && d < v->shape.rank; d++) {
            if (coords[d] < v->mask_begin[d] || coords[d] >= v->mask_end[d])
                return 0.f;
        }
        // Compound mask check (pool views: validity = begin <= c_a*s + c_b < end)
        for (u32 cm = 0; cm < v->n_compound_masks; cm++) {
            u32 da = v->compound_masks[cm].dim_a;
            u32 db = v->compound_masks[cm].dim_b;
            i32 sa = v->compound_masks[cm].stride_a;
            i32 composed = (i32)coords[da] * sa + (i32)coords[db];
            if (composed < (i32)v->compound_masks[cm].begin ||
                composed >= (i32)v->compound_masks[cm].end)
                return 0.f;
        }
    }
    // Compute physical index via strides.
    // Modulo coords by leaf dim to handle broadcast (leaf_dim=1 → coord%1=0).
    i32 idx = v->offset;
    for (u32 d = 0; d < rank && d < v->shape.rank; d++) {
        u32 c = coords[d] % v->shape.dims[d];
        idx += (i32)c * v->strides[d];
    }
    if (idx < 0) return 0.f;
    return buf[(u32)idx];
}

void cpu_dispatch_kernel_rs(
    u32 out_buf, u32 *leaf_bufs, const View **leaf_views,
    const ShapeTracker *const *leaf_sts, u32 n_leaves,
    FusedOp *ops, u32 n_ops, const Shape *full_shape, const ReduceSpec *reduce,
    u32 *side_bufs, const u32 *side_op_indices, u32 n_side_outputs)
{
    (void)side_bufs; (void)side_op_indices; (void)n_side_outputs;
    u32 rank = full_shape->rank;
    if (rank == 0) rank = 1;

    // Determine which dims are reduced
    int has_reduce = reduce && reduce->reduce_type;
    u32 out_numel = 1, full_numel = 1;
    u32 out_shape[MAX_DIM];
    for (u32 d = 0; d < rank; d++) {
        full_numel *= full_shape->dims[d];
        out_shape[d] = (has_reduce && reduce->is_reduce[d]) ? 1 : full_shape->dims[d];
        out_numel *= out_shape[d];
    }

    // Read all leaf buffers into CPU pointers
    f32 *leaf_ptrs[FUSE_MAX_LEAVES];
    for (u32 i = 0; i < n_leaves; i++)
        leaf_ptrs[i] = cpu_pool.bufs[leaf_bufs[i]];

    // Allocate output
    f32 *out = cpu_pool.bufs[out_buf];
    // Initialize output (reduce needs accumulator init)
    if (has_reduce) {
        f32 init = (reduce->reduce_type == UOP_RMAX) ? -1e30f : 0.f;
        for (u32 i = 0; i < out_numel; i++) out[i] = init;
    }

    // Compute strides for full_shape (row-major) for coord decomposition
    u32 full_strides[MAX_DIM];
    full_strides[rank - 1] = 1;
    for (i32 d = (i32)rank - 2; d >= 0; d--)
        full_strides[d] = full_strides[d + 1] * full_shape->dims[d + 1];

    // Compute strides for out_shape (for mapping full coords → output index)
    u32 out_strides[MAX_DIM];
    out_strides[rank - 1] = 1;
    for (i32 d = (i32)rank - 2; d >= 0; d--)
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];

    // Temporary for evaluating the FusedOp DAG
    f32 vals[FUSE_MAX_LEAVES + FUSE_MAX_OPS];

    // Main loop: iterate over every element in full_shape
    for (u32 flat = 0; flat < full_numel; flat++) {
        // Decompose flat index → coordinates
        u32 coords[MAX_DIM], rem = flat;
        for (u32 d = 0; d < rank; d++) {
            coords[d] = rem / full_strides[d];
            rem %= full_strides[d];
        }

        // Read leaf values (use multi-view ST when available)
        for (u32 i = 0; i < n_leaves; i++) {
            if (leaf_sts && leaf_sts[i] && leaf_sts[i]->n_views >= 2)
                vals[i] = leaf_read_st(leaf_ptrs[i], leaf_sts[i], coords, rank, full_shape);
            else
                vals[i] = leaf_read(leaf_ptrs[i], leaf_views[i], coords, rank);
        }

        // Evaluate FusedOp DAG (pre-reduce ops only if post_reduce_start set)
        u32 eval_n_ops = (has_reduce && reduce->post_reduce_start) ? reduce->post_reduce_start : n_ops;
        for (u32 i = 0; i < eval_n_ops; i++) {
            u32 aa = ops[i].arg_a, bb = ops[i].arg_b;
            if (aa >= n_leaves + eval_n_ops || bb >= n_leaves + eval_n_ops) {
                if (flat == 0) fprintf(stderr, "OOB: op[%u] a=%u b=%u max=%u\n", i, aa, bb, n_leaves+eval_n_ops-1);
                break;
            }
            f32 a = vals[aa];
            f32 b = vals[bb];
            vals[n_leaves + i] = eval_uop(ops[i].uop, a, b);
        }

        f32 result = (eval_n_ops > 0) ? vals[n_leaves + eval_n_ops - 1] : vals[0];

        // Write or accumulate
        if (has_reduce) {
            // Compute output index (reduce dims contribute 0)
            u32 oi = 0;
            for (u32 d = 0; d < rank; d++) {
                if (!reduce->is_reduce[d])
                    oi += coords[d] * out_strides[d];
            }
            if (reduce->reduce_type == UOP_SUM)
                out[oi] += result;
            else if (reduce->reduce_type == UOP_RMAX)
                out[oi] = result > out[oi] ? result : out[oi];
        } else {
            out[flat] = result;
        }
    }

    // Post-reduce ew ops: iterate over the REDUCED output shape,
    // evaluate post-reduce ops that read the reduce result + post-reduce leaves.
    if (has_reduce && reduce->post_reduce_start && reduce->post_reduce_start < n_ops) {
        u32 post_start = reduce->post_reduce_start;
        // The reduce result for each output element is in out[oi].
        // Post-reduce ops can reference it via the pre-reduce result index.
        // Pre-reduce leaves: [0..n_leaves-n_post_leaves-1]
        // Post-reduce leaves: [n_leaves-n_post_leaves..n_leaves-1]
        // The reduce result index in vals: n_leaves + post_start - 1 (last pre-reduce op)
        // BUT post-reduce ops were remapped to reference this index.

        // Compute out_strides for output iteration
        u32 out_strides2[MAX_DIM];
        out_strides2[rank - 1] = 1;
        for (i32 d = (i32)rank - 2; d >= 0; d--)
            out_strides2[d] = out_strides2[d + 1] * out_shape[d + 1];

        for (u32 oi = 0; oi < out_numel; oi++) {
            // Decompose output index → coordinates in out_shape
            u32 coords[MAX_DIM], rem = oi;
            for (u32 d = 0; d < rank; d++) {
                coords[d] = rem / out_strides2[d];
                rem %= out_strides2[d];
            }

            // Read the reduce result for this output position
            f32 reduce_result = out[oi];

            // Read post-reduce leaf values
            // Post-reduce leaves are at the END of the leaf array
            u32 n_pre_leaves = n_leaves - reduce->n_post_leaves;
            for (u32 i = 0; i < n_leaves; i++) {
                if (i < n_pre_leaves) {
                    // Pre-reduce leaf: the reduce result maps to the last pre-reduce op
                    // Post-reduce ops reference this via their remapped arg indices
                    vals[i] = 0; // not used by post-reduce ops (they use reduce result)
                } else {
                    // Post-reduce leaf: read from buffer
                    if (leaf_sts && leaf_sts[i] && leaf_sts[i]->n_views >= 2)
                        vals[i] = leaf_read_st(leaf_ptrs[i], leaf_sts[i], coords, rank, &(Shape){.rank=rank, .dims={0}});
                    else
                        vals[i] = leaf_read(leaf_ptrs[i], leaf_views[i], coords, rank);
                }
            }

            // The reduce result is at index (n_pre_leaves + post_start - 1) in vals
            // (that's where post-reduce ops expect to find it based on merge remapping)
            if (post_start > 0)
                vals[n_leaves + post_start - 1] = reduce_result;

            // Evaluate post-reduce ops
            for (u32 i = post_start; i < n_ops; i++) {
                f32 a = vals[ops[i].arg_a];
                f32 b = vals[ops[i].arg_b];
                vals[n_leaves + i] = eval_uop(ops[i].uop, a, b);
            }

            f32 post_result = vals[n_leaves + n_ops - 1];
            out[oi] = post_result;
        }
    }
}
