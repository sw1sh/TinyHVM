// metal/uop.m — Micro-operation IR for kernel codegen
static int _last_compiled_uop = 0; // set by cg_get_pipe_rs when UOp path used
// Backend-agnostic kernel representation, rendered to MSL.
// Replaces FusedOp[] + ReduceSpec with a single linear SSA IR
// that naturally supports multi-reduce, multi-output, and complex indexing.

#define KOP_MAX 512

typedef enum {
    KOP_NOOP = 0,
    KOP_GID,     // grid position: imm.u = axis (0=x, 1=y, 2=z)
    KOP_CONST_F, // float constant: imm.f = value
    KOP_CONST_U, // uint constant: imm.u = value
    KOP_RANGE,   // reduce loop: imm.u = trip count. arg[0] = body end marker
    KOP_ENDRANGE,// close reduce loop
    KOP_LOAD,    // load: imm.u = buf_idx, arg[0] = index
    KOP_STORE,   // store: imm.u = buf_idx, arg[0] = index, arg[1] = value
    KOP_ALU,     // arithmetic: imm.u = UOP_ADD etc, arg[0..1] = inputs
    KOP_ACC_INIT,// accumulator init: imm.f = initial value (0 for sum, -inf for max)
    KOP_ACC,     // accumulate: imm.u = reduce type (UOP_SUM/UOP_RMAX), arg[0] = acc, arg[1] = value
    KOP_IDX,     // index arithmetic: result = arg[0]*imm.u + arg[1]  (multiply-add)
    KOP_MOD,     // modulo: result = arg[0] % imm.u
    KOP_DIV,     // integer div: result = arg[0] / imm.u
    KOP_MASK,    // conditional: result = arg[0] ? arg[1] : 0.0f  (mask for PAD views)
} KOpType;

typedef struct {
    KOpType type;
    u32     arg[3];  // references to other UOps (SSA)
    union {
        f32 f;
        u32 u;
    } imm;
} UOp;

typedef struct {
    UOp ops[KOP_MAX];
    u32 n_ops;
    u32 n_bufs;
    u32 n_leaves;
    const View *leaf_views[32];
    u32 coord_uop[MAX_DIM]; // UOp id for coordinate of each dim (for mask rendering)
    u32 rank;
    u32 grid[3];
    u32 tg[3];
} UOpKernel;

// ── Build UOp kernel from FusedOp[] + ReduceSpec (compatibility layer) ──

static u32 uop_emit(UOpKernel *k, KOpType type, u32 a0, u32 a1, u32 a2, u32 imm_u) {
    assert(k->n_ops < KOP_MAX);
    u32 id = k->n_ops++;
    k->ops[id] = (UOp){ .type = type, .arg = {a0, a1, a2}, .imm.u = imm_u };
    return id;
}

static u32 uop_emit_f(UOpKernel *k, KOpType type, u32 a0, u32 a1, u32 a2, f32 imm_f) {
    assert(k->n_ops < KOP_MAX);
    u32 id = k->n_ops++;
    k->ops[id] = (UOp){ .type = type, .arg = {a0, a1, a2} };
    k->ops[id].imm.f = imm_f;
    return id;
}

// Build mask condition for a leaf view (returns 0 if no mask needed, else UOp id for bool).
// The mask checks if coordinates are within [mask_begin, mask_end) for each dim.
static u32 uop_build_mask(UOpKernel *k, const View *v, u32 *coords, u32 rank) {
    if (!v->has_mask) return 0;
    u32 cond = 0; // 0 = no mask ops emitted yet
    for (u32 d = 0; d < v->shape.rank && d < rank; d++) {
        if (v->mask_begin[d] == 0 && v->mask_end[d] >= v->shape.dims[d]) continue;
        // Check: coords[d] >= begin && coords[d] < end
        // Emit as: (coords[d] - begin) < (end - begin) [unsigned comparison trick]
        // Simpler: emit explicit comparisons and AND them
        // For MSL we just emit inline — rendered as ternary in KOP_MASK
        // Store mask params in the imm field (pack begin/end)
        // Actually, let's just generate the mask in the renderer by examining the view
        cond = 1; // mark that mask is needed
    }
    return cond;
}

// Build index expression for a leaf view using coordinates.
// coords[d] = UOp id for coordinate d (uint). Returns UOp id for the flat index (uint).
static u32 uop_build_index(UOpKernel *k, const View *v, u32 *coords, u32 rank,
                            const Shape *full_shape) {
    // Offset may be negative (e.g., shrink views). Use signed→uint wrapping.
    u32 idx = uop_emit(k, KOP_CONST_U, 0, 0, 0, (u32)(i32)v->offset);

    // Check if coordinate-based indexing is safe (Path B check from old codegen)
    int coord_ok = (v->shape.rank == rank);
    if (coord_ok) {
        for (u32 d = 0; d < rank; d++) {
            if (v->shape.dims[d] != full_shape->dims[d] &&
                v->strides[d] != 0 && v->shape.dims[d] != 1)
                { coord_ok = 0; break; }
        }
    }

    // Path C: need flat-index decomposition (rank mismatch or incompatible dims)
    if (!coord_ok) {
        // Compute flat full-shape index from all coordinates
        u32 flat_strides[MAX_DIM];
        if (rank > 0) {
            flat_strides[rank - 1] = 1;
            for (int d = (int)rank - 2; d >= 0; d--)
                flat_strides[d] = flat_strides[d + 1] * full_shape->dims[d + 1];
        }
        // fi = sum(coords[d] * flat_strides[d])
        u32 fi = uop_emit(k, KOP_CONST_U, 0, 0, 0, 0);
        for (u32 d = 0; d < rank; d++) {
            if (flat_strides[d] == 1)
                fi = uop_emit(k, KOP_IDX, fi, coords[d], 0, 1); // fi + coords[d]
            else {
                u32 term = uop_emit(k, KOP_IDX, coords[d], 0, 0, flat_strides[d]);
                fi = uop_emit(k, KOP_IDX, fi, term, 0, 1);
            }
        }
        // Decompose fi through leaf's own shape
        u32 leaf_idx = uop_emit(k, KOP_CONST_U, 0, 0, 0, (u32)(i32)v->offset);
        u32 leaf_div = 1;
        for (int d = (int)v->shape.rank - 1; d >= 0; d--) {
            if (v->strides[d] == 0) { leaf_div *= v->shape.dims[d]; continue; }
            // coord_d = (fi / leaf_div) % dim
            u32 cd;
            if (leaf_div == 1) cd = uop_emit(k, KOP_MOD, fi, 0, 0, v->shape.dims[d]);
            else {
                u32 div = uop_emit(k, KOP_DIV, fi, 0, 0, leaf_div);
                cd = uop_emit(k, KOP_MOD, div, 0, 0, v->shape.dims[d]);
            }
            // leaf_idx += cd * stride
            if (v->strides[d] == 1)
                leaf_idx = uop_emit(k, KOP_IDX, leaf_idx, cd, 0, 1);
            else {
                u32 term = uop_emit(k, KOP_IDX, cd, 0, 0, (u32)v->strides[d]);
                leaf_idx = uop_emit(k, KOP_IDX, leaf_idx, term, 0, 1);
            }
            leaf_div *= v->shape.dims[d];
        }
        return leaf_idx;
    }

    for (u32 d = 0; d < rank && d < v->shape.rank; d++) {
        if (v->strides[d] == 0 || v->shape.dims[d] == 1) continue;
        // idx = idx + coords[d] * stride
        u32 term;
        if (v->strides[d] == 1) {
            term = coords[d];
        } else {
            term = uop_emit(k, KOP_IDX, coords[d], 0, 0, (u32)v->strides[d]);
        }
        idx = uop_emit(k, KOP_IDX, idx, term, 0, 1); // idx*1 + term = idx + term
    }
    return idx;
}

// Convert FusedOp[] + ReduceSpec to UOp kernel
static int uop_from_fused(UOpKernel *k, const FusedOp *ops, u32 n_ops,
                           u32 n_leaves, const View **leaf_views,
                           const Shape *full_shape, const ReduceSpec *reduce) {
    memset(k, 0, sizeof(*k));
    u32 rank = full_shape->rank;
    if (rank == 0) rank = 1;

    int has_reduce = reduce && reduce->reduce_type;

    // Classify axes
    u32 out_dims[MAX_DIM], n_out = 0;
    u32 red_dims[MAX_DIM], n_red = 0;
    u32 out_numel = 1, reduce_numel = 1;
    for (u32 d = 0; d < rank; d++) {
        if (has_reduce && reduce->is_reduce[d]) {
            red_dims[n_red++] = d;
            reduce_numel *= full_shape->dims[d];
        } else {
            out_dims[n_out++] = d;
            out_numel *= full_shape->dims[d];
        }
    }

    // Compute dispatch grid (must match metal_dispatch_kernel_rs)
    u32 out_shape[MAX_DIM];
    for (u32 i = 0; i < n_out; i++) out_shape[i] = full_shape->dims[out_dims[i]];
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
    // UOp kernel is scalar — dispatch must NOT use float4 grid reduction.
    // Grid: (inner, mid, outer) without float4 division.
    k->grid[0] = inner; k->grid[1] = mid; k->grid[2] = outer;
    k->tg[2] = 1;
    k->n_bufs = 1 + n_leaves;
    k->n_leaves = n_leaves;
    for (u32 i = 0; i < n_leaves && i < 32; i++) k->leaf_views[i] = leaf_views[i];

    // GID for output position
    u32 gid = uop_emit(k, KOP_GID, 0, 0, 0, 0);

    // Decompose gid into output coordinates
    u32 coords[MAX_DIM]; // coords[d] = UOp for coordinate d
    memset(coords, 0, sizeof(coords));
    {
        u32 rem = gid;
        for (int oi = (int)n_out - 1; oi >= 0; oi--) {
            u32 d = out_dims[oi];
            u32 dim = full_shape->dims[d];
            if (dim == 1) { coords[d] = uop_emit(k, KOP_CONST_U, 0,0,0, 0); continue; }
            coords[d] = uop_emit(k, KOP_MOD, rem, 0, 0, dim);
            if (oi > 0) rem = uop_emit(k, KOP_DIV, rem, 0, 0, dim);
        }
    }

    // Store coord UOp ids + rank for mask rendering
    k->rank = rank;
    memcpy(k->coord_uop, coords, sizeof(coords));

    if (!has_reduce) {
        // Pure elementwise: load leaves, apply ops, store
        u32 vals[64]; // FusedOp arg indices: max n_leaves(32) + n_ops(32) // val[i] = UOp for FusedOp output i
        for (u32 li = 0; li < n_leaves; li++) {
            u32 idx = uop_build_index(k, leaf_views[li], coords, rank, full_shape);
            vals[li] = uop_emit(k, KOP_LOAD, idx, 0, 0, 1 + li); // buf 0=out, 1..=leaves
        }
        for (u32 i = 0; i < n_ops; i++) {
            u32 a = vals[ops[i].arg_a], b = vals[ops[i].arg_b];
            vals[n_leaves + i] = uop_emit(k, KOP_ALU, a, b, 0, ops[i].uop);
        }
        u32 out_idx = uop_emit(k, KOP_IDX, gid, 0, 0, 1); // flat output index
        uop_emit(k, KOP_STORE, out_idx, vals[n_leaves + n_ops - 1], 0, 0);
        return 1;
    }

    // Reduce kernel: accumulate over reduce dims
    f32 init = (reduce->reduce_type == UOP_RMAX) ? -1e30f : 0.0f;
    u32 acc = uop_emit_f(k, KOP_ACC_INIT, 0, 0, 0, init);
    u32 range = uop_emit(k, KOP_RANGE, 0, 0, 0, reduce_numel);

    // Decompose range var into reduce coordinates
    u32 range_rem = range;
    for (int ri = (int)n_red - 1; ri >= 0; ri--) {
        u32 d = red_dims[ri], dim = full_shape->dims[d];
        coords[d] = uop_emit(k, KOP_MOD, range_rem, 0, 0, dim);
        if (ri > 0) range_rem = uop_emit(k, KOP_DIV, range_rem, 0, 0, dim);
    }

    // Load leaves + apply pre-reduce ops (inside loop)
    u32 vals[64]; // FusedOp arg indices: max n_leaves(32) + n_ops(32)
    for (u32 li = 0; li < n_leaves; li++) {
        u32 idx = uop_build_index(k, leaf_views[li], coords, rank, full_shape);
        vals[li] = uop_emit(k, KOP_LOAD, idx, 0, 0, 1 + li);
    }
    u32 pre_ops = (reduce->post_reduce_start > 0) ? reduce->post_reduce_start : n_ops;
    for (u32 i = 0; i < pre_ops; i++) {
        u32 a = vals[ops[i].arg_a], b = vals[ops[i].arg_b];
        vals[n_leaves + i] = uop_emit(k, KOP_ALU, a, b, 0, ops[i].uop);
    }
    u32 acc_init = acc; // save ACC_INIT id — this is the variable that persists
    uop_emit(k, KOP_ACC, acc, vals[n_leaves + pre_ops - 1], 0,
                   reduce->reduce_type);
    uop_emit(k, KOP_ENDRANGE, range, 0, 0, 0);

    // Post-reduce ops: mirrors old codegen exactly.
    // p0 = acc op leaf, p1 = p0 op leaf, ...
    u32 result = acc_init;
    if (reduce->post_reduce_start > 0 && reduce->post_reduce_start < n_ops) {
        u32 prs = reduce->post_reduce_start;
        u32 post_leaf_base = n_leaves - reduce->n_post_leaves;

        // Re-load post-reduce-only leaves with output coordinates
        for (u32 pli = post_leaf_base; pli < n_leaves; pli++) {
            u32 pidx = uop_build_index(k, leaf_views[pli], coords, rank, full_shape);
            vals[pli] = uop_emit(k, KOP_LOAD, pidx, 0, 0, 1 + pli);
        }

        // Emit post-reduce op chain (same as old codegen p0, p1, ...)
        u32 prev = acc_init; // first op chains from accumulator
        for (u32 i = prs; i < n_ops; i++) {
            u32 a = prev; // chain from previous result (or acc for first)
            u32 b = 0;
            if (is_binary(ops[i].uop)) {
                u32 ob = ops[i].arg_b;
                if (ob >= post_leaf_base && ob < n_leaves)
                    b = vals[ob]; // post-reduce leaf
                else
                    b = acc_init; // fallback
            }
            prev = uop_emit(k, KOP_ALU, a, b, 0, ops[i].uop);
            vals[n_leaves + i] = prev;
        }
        result = prev;
    }

    // Second reduce phase (multi-reduce)
    if (reduce->reduce2_type && reduce->reduce2_start < n_ops) {
        f32 init2 = (reduce->reduce2_type == UOP_RMAX) ? -1e30f : 0.0f;
        u32 acc2 = uop_emit_f(k, KOP_ACC_INIT, 0, 0, 0, init2);
        u32 range2 = uop_emit(k, KOP_RANGE, 0, 0, 0, reduce_numel); // same dims

        // Recompute reduce coords from range2
        u32 range2_rem = range2;
        for (int ri = (int)n_red - 1; ri >= 0; ri--) {
            u32 d = red_dims[ri], dim = full_shape->dims[d];
            coords[d] = uop_emit(k, KOP_MOD, range2_rem, 0, 0, dim);
            if (ri > 0) range2_rem = uop_emit(k, KOP_DIV, range2_rem, 0, 0, dim);
        }

        // Re-load leaves inside loop
        for (u32 li = 0; li < n_leaves; li++) {
            u32 idx = uop_build_index(k, leaf_views[li], coords, rank, full_shape);
            vals[li] = uop_emit(k, KOP_LOAD, idx, 0, 0, 1 + li);
        }
        // Phase 2 ops (reference acc via result variable)
        for (u32 i = reduce->reduce2_start; i < n_ops; i++) {
            u32 a = ops[i].arg_a, b = ops[i].arg_b;
            u32 va = (a < n_leaves) ? vals[a] : (a == n_leaves) ? result : vals[a];
            u32 vb = (b < n_leaves) ? vals[b] : (b == n_leaves) ? result : vals[b];
            vals[n_leaves + i] = uop_emit(k, KOP_ALU, va, vb, 0, ops[i].uop);
        }
        u32 acc2_init = acc2;
        uop_emit(k, KOP_ACC, acc2, vals[n_leaves + n_ops - 1], 0,
                        reduce->reduce2_type);
        uop_emit(k, KOP_ENDRANGE, range2, 0, 0, 0);
        result = acc2_init;
    }

    u32 out_idx = uop_emit(k, KOP_IDX, gid, 0, 0, 1);
    uop_emit(k, KOP_STORE, out_idx, result, 0, 0);
    return 1;
}

// ── Render UOp kernel to MSL ──────────────────────────────────────

static NSString *uop_render_msl(const UOpKernel *k) {
    NSMutableString *s = [NSMutableString stringWithCapacity:4096];
    [s appendString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
    [s appendFormat:@"kernel void K(device float *buf0[[buffer(0)]]"];
    for (u32 i = 1; i < k->n_bufs; i++)
        [s appendFormat:@",\n  device const float *buf%u[[buffer(%u)]]", i, i];
    [s appendString:@",\n  uint3 _gid[[thread_position_in_grid]])\n{\n"];
    u32 total = k->grid[0] * k->grid[1] * k->grid[2];
    [s appendFormat:@"  uint gid_x=_gid.z*%uu+_gid.y*%uu+_gid.x;\n",
        k->grid[1] * k->grid[0], k->grid[0]];
    [s appendFormat:@"  if(gid_x>=%uu)return;\n", total];

    int depth = 1;

    for (u32 i = 0; i < k->n_ops; i++) {
        const UOp *op = &k->ops[i];
        NSString *indent = (depth <= 1) ? @"  " : (depth == 2) ? @"    " : @"      ";
        switch (op->type) {
            case KOP_GID:
                [s appendFormat:@"%@uint v%u=gid_x;\n", indent, i]; break;
            case KOP_CONST_U:
                [s appendFormat:@"%@uint v%u=%uu;\n", indent, i, op->imm.u]; break;
            case KOP_CONST_F:
                [s appendFormat:@"%@float v%u=%.8ef;\n", indent, i, op->imm.f]; break;
            case KOP_MOD:
                [s appendFormat:@"%@uint v%u=v%u%%%uu;\n", indent, i, op->arg[0], op->imm.u]; break;
            case KOP_DIV:
                [s appendFormat:@"%@uint v%u=v%u/%uu;\n", indent, i, op->arg[0], op->imm.u]; break;
            case KOP_IDX:
                if (op->imm.u == 1 && op->arg[1] == 0)
                    [s appendFormat:@"%@uint v%u=v%u;\n", indent, i, op->arg[0]];
                else if (op->imm.u == 1)
                    [s appendFormat:@"%@uint v%u=v%u+v%u;\n", indent, i, op->arg[0], op->arg[1]];
                else if (op->arg[1] == 0)
                    [s appendFormat:@"%@uint v%u=v%u*%uu;\n", indent, i, op->arg[0], op->imm.u];
                else
                    [s appendFormat:@"%@uint v%u=v%u*%uu+v%u;\n", indent, i,
                        op->arg[0], op->imm.u, op->arg[1]]; break;
            case KOP_LOAD: {
                u32 leaf_idx = op->imm.u - 1; // buf 0=out, 1+=leaves
                if (leaf_idx < k->n_leaves && k->leaf_views[leaf_idx]->has_mask) {
                    const View *lv = k->leaf_views[leaf_idx];
                    [s appendFormat:@"%@float v%u=(", indent, i];
                    int first = 1;
                    for (u32 d = 0; d < lv->shape.rank && d < k->rank; d++) {
                        if (lv->mask_begin[d] == 0 && lv->mask_end[d] >= lv->shape.dims[d]) continue;
                        u32 cv = k->coord_uop[d]; // UOp id for coordinate d
                        if (!first) [s appendString:@"&&"];
                        first = 0;
                        if (lv->mask_begin[d] > 0)
                            [s appendFormat:@"v%u>=%uu", cv, lv->mask_begin[d]];
                        if (lv->mask_begin[d] > 0 && lv->mask_end[d] < lv->shape.dims[d])
                            [s appendString:@"&&"];
                        if (lv->mask_end[d] < lv->shape.dims[d])
                            [s appendFormat:@"v%u<%uu", cv, lv->mask_end[d]];
                    }
                    if (first) [s appendString:@"true"];
                    [s appendFormat:@")?buf%u[v%u]:0.f;\n", op->imm.u, op->arg[0]];
                } else {
                    [s appendFormat:@"%@float v%u=buf%u[v%u];\n", indent, i, op->imm.u, op->arg[0]];
                }
            } break;
            case KOP_STORE:
                [s appendFormat:@"%@buf0[v%u]=v%u;\n", indent, op->arg[0], op->arg[1]]; break;
            case KOP_ALU: {
                u32 a = op->arg[0], b = op->arg[1];
                switch (op->imm.u) {
                    case UOP_ADD:  [s appendFormat:@"%@float v%u=v%u+v%u;\n", indent, i, a, b]; break;
                    case UOP_SUB:  [s appendFormat:@"%@float v%u=v%u-v%u;\n", indent, i, a, b]; break;
                    case UOP_MUL:  [s appendFormat:@"%@float v%u=v%u*v%u;\n", indent, i, a, b]; break;
                    case UOP_DIV:  [s appendFormat:@"%@float v%u=v%u/v%u;\n", indent, i, a, b]; break;
                    case UOP_NEG:  [s appendFormat:@"%@float v%u=-v%u;\n", indent, i, a]; break;
                    case UOP_RELU: [s appendFormat:@"%@float v%u=max(v%u,0.f);\n", indent, i, a]; break;
                    case UOP_EXP:  [s appendFormat:@"%@float v%u=exp(v%u);\n", indent, i, a]; break;
                    case UOP_LOG:  [s appendFormat:@"%@float v%u=log(v%u);\n", indent, i, a]; break;
                    case UOP_SQRT: [s appendFormat:@"%@float v%u=sqrt(v%u);\n", indent, i, a]; break;
                    case UOP_MAX:  [s appendFormat:@"%@float v%u=max(v%u,v%u);\n", indent, i, a, b]; break;
                    case UOP_CMP:  [s appendFormat:@"%@float v%u=v%u>v%u?1.f:0.f;\n", indent, i, a, b]; break;
                    default:       [s appendFormat:@"%@float v%u=v%u;\n", indent, i, a]; break;
                }
            } break;
            case KOP_ACC_INIT:
                [s appendFormat:@"%@float v%u=%.1ff;\n", indent, i, op->imm.f]; break;
            case KOP_RANGE:
                [s appendFormat:@"%@for(uint v%u=0;v%u<%uu;v%u++){\n",
                    indent, i, i, op->imm.u, i];
                depth++; break;
            case KOP_ENDRANGE:
                depth--;
                indent = (depth <= 1) ? @"  " : (depth == 2) ? @"    " : @"      ";
                [s appendFormat:@"%@}\n", indent]; break;
            case KOP_ACC: {
                u32 acc_var = op->arg[0], val = op->arg[1];
                if (op->imm.u == UOP_RMAX)
                    [s appendFormat:@"%@v%u=max(v%u,v%u);\n", indent, acc_var, acc_var, val];
                else
                    [s appendFormat:@"%@v%u+=v%u;\n", indent, acc_var, val];
                // No SSA alias — references to this ACC resolve to acc_var
            } break;
            default: break;
        }
    }
    [s appendString:@"}\n"];
    return s;
}
