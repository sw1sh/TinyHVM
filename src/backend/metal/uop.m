// metal/uop.m — Micro-operation IR for kernel codegen
// Backend-agnostic kernel representation, rendered to MSL.
// Replaces FusedOp[] + ReduceSpec with a single linear SSA IR
// that naturally supports multi-reduce, multi-output, and complex indexing.

#define KOP_MAX 128

typedef enum {
    KOP_NOOP = 0,
    KOP_GID,     // grid position: imm.u = axis (0=x, 1=y, 2=z)
    KOP_CONST,   // constant: imm.f = value
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
    u32 n_bufs;      // number of buffers (out + side_outputs + leaves)
    u32 grid[3];     // dispatch grid
    u32 tg[3];       // threadgroup size
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

// Build index expression for a leaf view using coordinates
// coords[d] = UOp id for coordinate d. Returns UOp id for the flat index.
static u32 uop_build_index(UOpKernel *k, const View *v, u32 *coords, u32 rank) {
    // Start with offset
    u32 idx = uop_emit_f(k, KOP_CONST, 0, 0, 0, (f32)v->offset);
    for (u32 d = 0; d < rank && d < v->shape.rank; d++) {
        if (v->strides[d] == 0 || v->shape.dims[d] == 1) continue;
        if (v->strides[d] == 1) {
            // idx += coord
            idx = uop_emit(k, KOP_ALU, idx, coords[d], 0, UOP_ADD);
        } else {
            // idx += coord * stride
            u32 scaled = uop_emit(k, KOP_IDX, coords[d], 0, 0, (u32)v->strides[d]);
            idx = uop_emit(k, KOP_ALU, idx, scaled, 0, UOP_ADD);
        }
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

    // Grid: output elements
    k->grid[0] = out_numel; k->grid[1] = 1; k->grid[2] = 1;
    k->tg[0] = MIN(256u, out_numel); k->tg[1] = 1; k->tg[2] = 1;
    k->n_bufs = 1 + n_leaves; // out + leaves

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
            if (dim == 1) { coords[d] = uop_emit_f(k, KOP_CONST, 0,0,0, 0.f); continue; }
            coords[d] = uop_emit(k, KOP_MOD, rem, 0, 0, dim);
            if (oi > 0) rem = uop_emit(k, KOP_DIV, rem, 0, 0, dim);
        }
    }

    if (!has_reduce) {
        // Pure elementwise: load leaves, apply ops, store
        u32 vals[UOP_MAX]; // val[i] = UOp for FusedOp output i
        for (u32 li = 0; li < n_leaves; li++) {
            u32 idx = uop_build_index(k, leaf_views[li], coords, rank);
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
    u32 vals[UOP_MAX];
    for (u32 li = 0; li < n_leaves; li++) {
        u32 idx = uop_build_index(k, leaf_views[li], coords, rank);
        vals[li] = uop_emit(k, KOP_LOAD, idx, 0, 0, 1 + li);
    }
    u32 pre_ops = (reduce->post_reduce_start > 0) ? reduce->post_reduce_start : n_ops;
    for (u32 i = 0; i < pre_ops; i++) {
        u32 a = vals[ops[i].arg_a], b = vals[ops[i].arg_b];
        vals[n_leaves + i] = uop_emit(k, KOP_ALU, a, b, 0, ops[i].uop);
    }
    acc = uop_emit(k, KOP_ACC, acc, vals[n_leaves + pre_ops - 1], 0,
                   reduce->reduce_type);
    uop_emit(k, KOP_ENDRANGE, range, 0, 0, 0);

    // Post-reduce ops (if any)
    u32 result = acc;
    if (reduce->post_reduce_start > 0 && reduce->post_reduce_start < n_ops) {
        u32 prs = reduce->post_reduce_start;
        vals[n_leaves + prs - 1] = acc; // post-reduce chain starts from acc
        for (u32 i = prs; i < n_ops; i++) {
            u32 a = vals[ops[i].arg_a], b = vals[ops[i].arg_b];
            vals[n_leaves + i] = uop_emit(k, KOP_ALU, a, b, 0, ops[i].uop);
        }
        result = vals[n_leaves + n_ops - 1];
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
            u32 idx = uop_build_index(k, leaf_views[li], coords, rank);
            vals[li] = uop_emit(k, KOP_LOAD, idx, 0, 0, 1 + li);
        }
        // Phase 2 ops (reference acc via result variable)
        for (u32 i = reduce->reduce2_start; i < n_ops; i++) {
            u32 a = ops[i].arg_a, b = ops[i].arg_b;
            u32 va = (a < n_leaves) ? vals[a] : (a == n_leaves) ? result : vals[a];
            u32 vb = (b < n_leaves) ? vals[b] : (b == n_leaves) ? result : vals[b];
            vals[n_leaves + i] = uop_emit(k, KOP_ALU, va, vb, 0, ops[i].uop);
        }
        acc2 = uop_emit(k, KOP_ACC, acc2, vals[n_leaves + n_ops - 1], 0,
                        reduce->reduce2_type);
        uop_emit(k, KOP_ENDRANGE, range2, 0, 0, 0);
        result = acc2;
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
    [s appendString:@",\n  uint gid_x[[thread_position_in_grid]])\n{\n"];
    [s appendFormat:@"  if(gid_x>=%uu)return;\n", k->grid[0]];

    int depth = 1;

    for (u32 i = 0; i < k->n_ops; i++) {
        const UOp *op = &k->ops[i];
        NSString *indent = (depth <= 1) ? @"  " : (depth == 2) ? @"    " : @"      ";
        switch (op->type) {
            case KOP_GID:
                [s appendFormat:@"%@uint v%u=gid_x;\n", indent, i]; break;
            case KOP_CONST:
                [s appendFormat:@"%@float v%u=%.8ef;\n", indent, i, op->imm.f]; break;
            case KOP_MOD:
                [s appendFormat:@"%@uint v%u=v%u%%%uu;\n", indent, i, op->arg[0], op->imm.u]; break;
            case KOP_DIV:
                [s appendFormat:@"%@uint v%u=v%u/%uu;\n", indent, i, op->arg[0], op->imm.u]; break;
            case KOP_IDX:
                if (op->imm.u == 1)
                    [s appendFormat:@"%@uint v%u=v%u;\n", indent, i, op->arg[0]];
                else
                    [s appendFormat:@"%@uint v%u=v%u*%uu+v%u;\n", indent, i,
                        op->arg[0], op->imm.u, op->arg[1]]; break;
            case KOP_LOAD:
                [s appendFormat:@"%@float v%u=buf%u[v%u];\n", indent, i, op->imm.u, op->arg[0]]; break;
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
                [s appendFormat:@"%@float v%u=v%u;\n", indent, i, acc_var]; // SSA alias
            } break;
            default: break;
        }
    }
    [s appendString:@"}\n"];
    return s;
}
