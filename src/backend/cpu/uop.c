// cpu/uop.c — CPU interpreter for lowered UOp kernels

static inline int cpu_uop_is_float_type(const UOpKernel *k, u32 id) {
    if (!k || id >= k->n_ops) return 0;
    switch (k->ops[id].type) {
        case KOP_CONST_F:
        case KOP_LOAD:
        case KOP_ALU:
        case KOP_ACC_INIT:
        case KOP_MASK:
            return 1;
        default:
            return 0;
    }
}

static inline u32 cpu_uop_as_u32(const UOpKernel *k, u32 id,
                                 const u32 *uv, const f32 *fv) {
    if (!k || id >= k->n_ops) return 0;
    return cpu_uop_is_float_type(k, id) ? (u32)fv[id] : uv[id];
}

static inline f32 cpu_uop_as_f32(const UOpKernel *k, u32 id,
                                 const u32 *uv, const f32 *fv) {
    if (!k || id >= k->n_ops) return 0.0f;
    return cpu_uop_is_float_type(k, id) ? fv[id] : (f32)uv[id];
}

static int cpu_exec_uop_block(const UOpKernel *k, const u32 *range_end,
                              u32 start, u32 stop, u32 gid,
                              void *out_ptr, const void **leaf_ptrs,
                              u32 *uv, f32 *fv) {
    for (u32 i = start; i < stop; i++) {
        const KOp *op = &k->ops[i];
        switch (op->type) {
            case KOP_GID:
                uv[i] = gid;
                break;
            case KOP_CONST_U:
                uv[i] = op->imm.u;
                break;
            case KOP_CONST_F:
                fv[i] = op->imm.f;
                break;
            case KOP_MOD:
                uv[i] = op->imm.u ? (cpu_uop_as_u32(k, op->arg[0], uv, fv) % op->imm.u) : 0;
                break;
            case KOP_DIV:
                uv[i] = op->imm.u ? (cpu_uop_as_u32(k, op->arg[0], uv, fv) / op->imm.u) : 0;
                break;
            case KOP_IDX:
                uv[i] = cpu_uop_as_u32(k, op->arg[0], uv, fv) * op->imm.u +
                        cpu_uop_as_u32(k, op->arg[1], uv, fv);
                break;
            case KOP_LOAD: {
                u32 buf_idx = op->imm.u;
                const void *buf = NULL;
                u32 dtype = DTYPE_F32;
                if (buf_idx == 0) {
                    buf = out_ptr;
                    dtype = k->out_dtype;
                } else if (buf_idx - 1 < k->n_leaves) {
                    buf = leaf_ptrs[buf_idx - 1];
                    dtype = k->leaf_dtypes[buf_idx - 1];
                }
                fv[i] = buf ? dtype_load_as_f32(buf, dtype, cpu_uop_as_u32(k, op->arg[0], uv, fv))
                            : 0.0f;
                break;
            }
            case KOP_MASK:
                fv[i] = cpu_uop_as_f32(k, op->arg[0], uv, fv) != 0.0f
                      ? cpu_uop_as_f32(k, op->arg[1], uv, fv)
                      : 0.0f;
                break;
            case KOP_ALU: {
                f32 a = cpu_uop_as_f32(k, op->arg[0], uv, fv);
                f32 b = cpu_uop_as_f32(k, op->arg[1], uv, fv);
                fv[i] = eval_uop(op->imm.u, op->arg[2], a, b);
                break;
            }
            case KOP_ACC_INIT:
                fv[i] = op->imm.f;
                break;
            case KOP_ACC: {
                u32 acc = op->arg[0];
                fv[acc] = (op->imm.u == UOP_RMAX)
                        ? (fv[acc] > cpu_uop_as_f32(k, op->arg[1], uv, fv)
                                ? fv[acc]
                                : cpu_uop_as_f32(k, op->arg[1], uv, fv))
                        : (fv[acc] + cpu_uop_as_f32(k, op->arg[1], uv, fv));
                break;
            }
            case KOP_RANGE: {
                u32 end = range_end ? range_end[i] : 0xFFFFFFFFu;
                if (end == 0xFFFFFFFFu || end <= i) return 0;
                for (u32 r = 0; r < op->imm.u; r++) {
                    uv[i] = r;
                    if (!cpu_exec_uop_block(k, range_end, i + 1, end, gid, out_ptr, leaf_ptrs, uv, fv))
                        return 0;
                }
                i = end;
                break;
            }
            case KOP_ENDRANGE:
                break;
            case KOP_STORE:
                dtype_store_from_f32(out_ptr, k->out_dtype,
                                     cpu_uop_as_u32(k, op->arg[0], uv, fv),
                                     cpu_uop_as_f32(k, op->arg[1], uv, fv));
                break;
            default:
                return 0;
        }
    }
    return 1;
}

void cpu_dispatch_uop_kernel(u32 out_buf, const u32 *leaf_bufs, u32 n_leaves,
                             const UOpKernel *kernel, u64 cache_key) {
    (void)cache_key;
    if (!kernel) return;
    void *out_ptr = cpu_pool.bufs[out_buf];
    if (!out_ptr) return;

    const void *leaf_ptrs[FUSE_MAX_LEAVES] = {0};
    for (u32 i = 0; i < n_leaves && i < FUSE_MAX_LEAVES; i++)
        leaf_ptrs[i] = cpu_pool.bufs[leaf_bufs[i]];

    u32 range_end[KOP_MAX];
    for (u32 i = 0; i < KOP_MAX; i++) range_end[i] = 0xFFFFFFFFu;
    u32 stack[KOP_MAX];
    u32 sp = 0;
    for (u32 i = 0; i < kernel->n_ops; i++) {
        if (kernel->ops[i].type == KOP_RANGE) {
            if (sp < KOP_MAX) stack[sp++] = i;
        } else if (kernel->ops[i].type == KOP_ENDRANGE) {
            if (sp == 0) return;
            u32 start = stack[--sp];
            range_end[start] = i;
        }
    }
    if (sp != 0) return;

    u32 total = kernel->local_size > 0
              ? kernel->grid[0]
              : (kernel->grid[0] * kernel->grid[1] * kernel->grid[2]);
    for (u32 gid = 0; gid < total; gid++) {
        u32 uv[KOP_MAX];
        f32 fv[KOP_MAX];
        memset(uv, 0, sizeof(uv));
        memset(fv, 0, sizeof(fv));
        if (!cpu_exec_uop_block(kernel, range_end, 0, kernel->n_ops, gid, out_ptr, leaf_ptrs, uv, fv))
            return;
    }
}
