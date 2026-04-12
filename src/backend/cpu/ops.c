// cpu/ops.c — CPU compute kernels: strided unary, binary, matmul, reduce

// Convert flat output index → strided input index
static inline u32 strided_index(u32 flat, const View *v) {
    u32 idx = (u32)v->offset;
    u32 rem = flat;
    for (i32 d = (i32)v->shape.rank - 1; d >= 0; d--) {
        u32 coord = rem % v->shape.dims[d];
        rem /= v->shape.dims[d];
        idx += coord * (u32)v->strides[d];
    }
    return idx;
}

static void cpu_op_unary(u32 uop, u32 dst, const View *dv, u32 dst_dtype,
                         u32 src, const View *sv, u32 src_dtype) {
    void *pd = cpu_pool.bufs[dst];
    const void *ps = cpu_pool.bufs[src];
    u32 n = dv->numel;

    for (u32 i = 0; i < n; i++) {
        u32 si = strided_index(i, sv);
        f32 val = dtype_load_as_f32(ps, src_dtype, si);
        switch (uop) {
            case UOP_NEG:  val = -val; break;
            case UOP_RELU: val = val > 0.0f ? val : 0.0f; break;
            case UOP_EXP:  val = __builtin_expf(val); break;
            case UOP_LOG:  val = __builtin_logf(val); break;
            case UOP_SQRT: val = __builtin_sqrtf(val); break;
            default: break;
        }
        dtype_store_from_f32(pd, dst_dtype, i, val);
    }
}

static void cpu_op_binary(u32 uop, u32 dst, const View *dv, u32 dst_dtype,
                          u32 a, const View *av, u32 a_dtype, u32 b, const View *bv, u32 b_dtype) {
    void *pd = cpu_pool.bufs[dst];
    const void *pa = cpu_pool.bufs[a];
    const void *pb = cpu_pool.bufs[b];
    u32 n = dv->numel;

    for (u32 i = 0; i < n; i++) {
        u32 ai = strided_index(i, av);
        u32 bi = strided_index(i, bv);
        f32 va = dtype_load_as_f32(pa, a_dtype, ai);
        f32 vb = dtype_load_as_f32(pb, b_dtype, bi);
        f32 vr;
        switch (uop) {
            case UOP_ADD: vr = va + vb; break;
            case UOP_MUL: vr = va * vb; break;
            case UOP_DIV: vr = vb != 0.0f ? va / vb : 0.0f; break;
            case UOP_SUB: vr = va - vb; break;
            case UOP_MAX: vr = va > vb ? va : vb; break;
            case UOP_CMP: vr = va > vb ? 1.0f : 0.0f; break;
            default:      vr = va; break;
        }
        dtype_store_from_f32(pd, dst_dtype, i, vr);
    }
}

// Check if View is a simple transpose (strides swapped from row-major)
static int is_transposed_2d(const View *v) {
    if (v->shape.rank != 2) return 0;
    u32 M = v->shape.dims[0], N = v->shape.dims[1];
    return (v->strides[0] == 1 && v->strides[1] == (i32)M) ||
           (v->strides[0] == 1 && v->strides[1] == (i32)N && !v->contiguous);
    (void)N;
}

// Materialize a non-contiguous 2D View into a contiguous buffer
static f32 *materialize_2d(u32 buf, u32 dtype, const View *v, u32 rows, u32 cols) {
    const void *src = cpu_pool.bufs[buf];
    u32 n = rows * cols;
    f32 *out = malloc(n * sizeof(f32));
    for (u32 i = 0; i < n; i++)
        out[i] = dtype_load_as_f32(src, dtype, strided_index(i, v));
    return out;
}

static void cpu_op_mm(u32 dst, u32 dst_dtype, u32 a, const View *av, u32 a_dtype, u32 b, const View *bv, u32 b_dtype,
                      u32 M, u32 K, u32 N) {
#if HAS_BLAS
    if (a_dtype == DTYPE_F32 && b_dtype == DTYPE_F32 && dst_dtype == DTYPE_F32) {
        f32 *pa = !av->contiguous ? materialize_2d(a, a_dtype, av, M, K) : (f32 *)cpu_pool.bufs[a];
        f32 *pb = !bv->contiguous ? materialize_2d(b, b_dtype, bv, K, N) : (f32 *)cpu_pool.bufs[b];

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)M, (int)N, (int)K, 1.0f,
                    pa, (int)K, pb, (int)N,
                    0.0f, (f32 *)cpu_pool.bufs[dst], (int)N);

        if (!av->contiguous) free(pa);
        if (!bv->contiguous) free(pb);
        return;
    }
#else
#endif
    void *pd = cpu_pool.bufs[dst];
    for (u32 i = 0; i < M; i++)
        for (u32 j = 0; j < N; j++) {
            f32 s = 0;
            for (u32 k = 0; k < K; k++) {
                u32 a_flat = i * K + k;
                u32 b_flat = k * N + j;
                f32 va = dtype_load_as_f32(cpu_pool.bufs[a], a_dtype, strided_index(a_flat, av));
                f32 vb = dtype_load_as_f32(cpu_pool.bufs[b], b_dtype, strided_index(b_flat, bv));
                s += va * vb;
            }
            dtype_store_from_f32(pd, dst_dtype, i * N + j, s);
        }
}

static void cpu_op_reduce(u32 uop, u32 dst, u32 dst_numel,
                           u32 dst_dtype, u32 src, u32 src_numel, u32 src_dtype, u32 reduce_dim) {
    (void)src_numel;
    void *pd = cpu_pool.bufs[dst];
    const void *ps = cpu_pool.bufs[src];
    u32 outer = dst_numel;

    for (u32 o = 0; o < outer; o++) {
        f32 acc = (uop == UOP_RMAX) ? -1e30f : 0.0f;
        for (u32 r = 0; r < reduce_dim; r++) {
            f32 v = dtype_load_as_f32(ps, src_dtype, o * reduce_dim + r);
            if (uop == UOP_SUM)       acc += v;
            else if (uop == UOP_RMAX) acc = v > acc ? v : acc;
        }
        dtype_store_from_f32(pd, dst_dtype, o, acc);
    }
}
