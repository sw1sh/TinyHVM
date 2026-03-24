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

static void cpu_op_unary(u32 uop, u32 dst, const View *dv,
                         u32 src, const View *sv) {
    f32 *pd = cpu_pool.bufs[dst];
    f32 *ps = cpu_pool.bufs[src];
    u32 n = dv->numel;

    for (u32 i = 0; i < n; i++) {
        u32 si = strided_index(i, sv);
        f32 val = ps[si];
        switch (uop) {
            case UOP_NEG:  pd[i] = -val; break;
            case UOP_RELU: pd[i] = val > 0.0f ? val : 0.0f; break;
            case UOP_EXP:  pd[i] = __builtin_expf(val); break;
            case UOP_LOG:  pd[i] = __builtin_logf(val); break;
            case UOP_SQRT: pd[i] = __builtin_sqrtf(val); break;
            default:       pd[i] = val; break;
        }
    }
}

static void cpu_op_binary(u32 uop, u32 dst, const View *dv,
                          u32 a, const View *av, u32 b, const View *bv) {
    f32 *pd = cpu_pool.bufs[dst];
    f32 *pa = cpu_pool.bufs[a];
    f32 *pb = cpu_pool.bufs[b];
    u32 n = dv->numel;

    for (u32 i = 0; i < n; i++) {
        u32 ai = strided_index(i, av);
        u32 bi = strided_index(i, bv);
        f32 va = pa[ai], vb = pb[bi];
        switch (uop) {
            case UOP_ADD: pd[i] = va + vb; break;
            case UOP_MUL: pd[i] = va * vb; break;
            case UOP_DIV: pd[i] = vb != 0.0f ? va / vb : 0.0f; break;
            case UOP_SUB: pd[i] = va - vb; break;
            case UOP_MAX: pd[i] = va > vb ? va : vb; break;
            case UOP_CMP: pd[i] = va > vb ? 1.0f : 0.0f; break;
            default:      pd[i] = va; break;
        }
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
static f32 *materialize_2d(u32 buf, const View *v, u32 rows, u32 cols) {
    f32 *src = cpu_pool.bufs[buf];
    u32 n = rows * cols;
    f32 *out = malloc(n * sizeof(f32));
    for (u32 i = 0; i < n; i++)
        out[i] = src[strided_index(i, v)];
    return out;
}

static void cpu_op_mm(u32 dst, u32 a, const View *av, u32 b, const View *bv,
                      u32 M, u32 K, u32 N) {
#if HAS_BLAS
    f32 *pa = !av->contiguous ? materialize_2d(a, av, M, K) : (f32 *)cpu_pool.bufs[a];
    f32 *pb = !bv->contiguous ? materialize_2d(b, bv, K, N) : (f32 *)cpu_pool.bufs[b];

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K, 1.0f,
                pa, (int)K, pb, (int)N,
                0.0f, (f32 *)cpu_pool.bufs[dst], (int)N);

    if (!av->contiguous) free(pa);
    if (!bv->contiguous) free(pb);
#else
    f32 *pd = cpu_pool.bufs[dst];
    for (u32 i = 0; i < M; i++)
        for (u32 j = 0; j < N; j++) {
            f32 s = 0;
            for (u32 k = 0; k < K; k++) {
                u32 a_flat = i * K + k;
                u32 b_flat = k * N + j;
                f32 va = ((f32*)cpu_pool.bufs[a])[strided_index(a_flat, av)];
                f32 vb = ((f32*)cpu_pool.bufs[b])[strided_index(b_flat, bv)];
                s += va * vb;
            }
            pd[i*N+j] = s;
        }
#endif
}

static void cpu_op_reduce(u32 uop, u32 dst, u32 dst_numel,
                           u32 src, u32 src_numel, u32 reduce_dim) {
    (void)src_numel;
    f32 *pd = cpu_pool.bufs[dst];
    f32 *ps = cpu_pool.bufs[src];
    u32 outer = dst_numel;

    for (u32 o = 0; o < outer; o++) {
        f32 acc = (uop == UOP_RMAX) ? -1e30f : 0.0f;
        for (u32 r = 0; r < reduce_dim; r++) {
            f32 v = ps[o * reduce_dim + r];
            if (uop == UOP_SUM)       acc += v;
            else if (uop == UOP_RMAX) acc = v > acc ? v : acc;
        }
        pd[o] = acc;
    }
}
