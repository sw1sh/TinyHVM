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
// Port of Metal codegen Path D: unravel right-to-left through view stack.
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
    // Step 2: for each view from n-2 down to 0, unravel si through
    // views[vi+1].shape, apply views[vi] strides (matching Metal exactly)
    for (i32 vi = (i32)st->n_views - 2; vi >= 0; vi--) {
        const View *vw = &st->views[vi];
        const View *outer = &st->views[vi + 1];
        i32 idx = vw->offset;
        u32 udiv = 1;
        int masked = 0;
        for (i32 d = (i32)outer->shape.rank - 1; d >= 0; d--) {
            u32 dim = outer->shape.dims[d];
            if (dim <= 1) { udiv *= dim; continue; }
            u32 c = (si / udiv) % dim;
            // Mask check on vw
            if (vw->has_mask && (u32)d < vw->shape.rank) {
                if (c < vw->mask_begin[d] || c >= vw->mask_end[d])
                    { masked = 1; break; }
            }
            if ((u32)d < vw->shape.rank && vw->strides[d] != 0)
                idx += (i32)c * vw->strides[d];
            udiv *= dim;
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

        // Evaluate FusedOp DAG
        for (u32 i = 0; i < n_ops; i++) {
            f32 a = vals[ops[i].arg_a];
            f32 b = vals[ops[i].arg_b];
            vals[n_leaves + i] = eval_uop(ops[i].uop, a, b);
        }

        f32 result = (n_ops > 0) ? vals[n_leaves + n_ops - 1] : vals[0];

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
}
