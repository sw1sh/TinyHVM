// Forward declarations for fusion (defined in fuse/_.c)
static int is_elementwise(u32 uop);
static int is_binary(u32 uop);

static inline u32 reduce_top_arity(u32 uop) {
    return thvm_uop_storage_arity(uop);
}

static inline int reduce_term_is_era_like(TinyHVM *ctx, Term t, Term *era_out) {
    if (term_tag(t) == TAG_ERA) {
        if (era_out) *era_out = t;
        return 1;
    }
    if (term_tag(t) == TAG_VAR) {
        u64 loc = term_val(t);
        if (loc < ctx->heap_pos) {
            Term sub = heap_read(ctx, loc);
            if (term_tag(sub) == TAG_ERA) {
                if (era_out) *era_out = sub;
                return 1;
            }
        }
    }
    return 0;
}

static inline int reduce_top_has_era_arg(TinyHVM *ctx, Term t) {
    if (term_tag(t) != TAG_TOP) return 0;
    u32 uop = term_ext(t);
    if (uop == UOP_FUSE) return 0;
    u64 loc = term_val(t);
    u32 arity = reduce_top_arity(uop);
    for (u32 i = 0; i < arity; i++) {
        Term child = heap_read(ctx, loc + i);
        if (reduce_term_is_era_like(ctx, child, NULL)) return 1;
    }
    return 0;
}

static inline int reduce_top_has_add_zero_arg(TinyHVM *ctx, Term t) {
    if (term_tag(t) != TAG_TOP || term_ext(t) != UOP_ADD) return 0;
    u64 loc = term_val(t);
    Term a = heap_read(ctx, loc + 0);
    Term b = heap_read(ctx, loc + 1);
    return (term_tag(a) == TAG_NUM && term_as_f32(a) == 0.0f) ||
           (term_tag(b) == TAG_NUM && term_as_f32(b) == 0.0f);
}

/* A FUSE child is "absorbable" when thvm_fuse_public_term can either
 * bind it as a kernel leaf (TEN/KERNEL/atom) OR wrap it in FUSE so it
 * fires next. Compute-like raw TOPs (ADD/MUL/.../view ops) fall into
 * the second bucket via thvm_fuse_wrap_child. Rejecting them here
 * strands the parent FUSE forever when the outer scheduler pipeline
 * is not going to run (e.g. during pure IC step tracing). */
static inline int reduce_fuse_child_absorbable(TinyHVM *ctx, Term child) {
    if (thvm_kernel_local_child_ready(ctx, child)) return 1;
    if (term_tag(child) == TAG_TOP) {
        u32 cu = term_ext(child);
        if (is_binary(cu) || is_elementwise(cu) || cu == UOP_SUM ||
            cu == UOP_RMAX || is_view_op(cu) || cu == UOP_FUSE)
            return 1;
    }
    return 0;
}

static inline int reduce_fuse_payload_top_ready(TinyHVM *ctx, Term t) {
    if (!ctx || term_tag(t) != TAG_TOP) return 0;
    u32 uop = term_ext(t);
    if (uop == UOP_ASSIGN || uop == UOP_KERNEL || uop == UOP_FUSE)
        return 1;
    if (!(is_binary(uop) || is_elementwise(uop) ||
          uop == UOP_SUM || uop == UOP_RMAX || is_view_op(uop)))
        return 0;
    u64 loc = term_val(t);
    if (loc == 0 || loc >= ctx->heap_pos) return 0;
    if (!reduce_fuse_child_absorbable(ctx, heap_read(ctx, loc + 0)))
        return 0;
    if (!is_binary(uop) && is_elementwise(uop))
        return 1;
    if (loc + 1 >= ctx->heap_pos) return 0;
    return reduce_fuse_child_absorbable(ctx, heap_read(ctx, loc + 1));
}

static inline int reduce_top_direct_uop(u32 uop) {
    return uop == UOP_ASSIGN || uop == UOP_GRAD || uop == UOP_GRAD_FWD || uop == UOP_IFZ ||
           uop == UOP_LOG_PRINT || uop == UOP_TODEVICE || uop == UOP_CAST ||
           uop == UOP_DETACH ||
           uop == UOP_WHERE ||
           uop == UOP_EXEC ||
           uop == UOP_KERNEL ||
           uop == UOP_FUSE;
}

static inline int reduce_top_direct_uop_ctx(TinyHVM *ctx, u32 uop) {
    (void)ctx;
    return reduce_top_direct_uop(uop);
}

static inline u32 reduce_net_term_arity(Term t) {
    u8 tag = term_tag(t);
    u32 ext = term_ext(t);
    switch (tag) {
        case TAG_TOP:
            return reduce_top_arity(ext);
        case TAG_APP:
        case TAG_LAM:
        case TAG_BRI:
        case TAG_SEQ:
        case TAG_SUP:
        case TAG_USP:
        case TAG_OP2:
        case TAG_EQL:
        case TAG_AND:
        case TAG_OR:
        case TAG_MAT:
        case TAG_ANN:
            return 2;
        case TAG_DSU:
        case TAG_DDU:
            return 3;
        case TAG_CTR:
            return ext;
        case TAG_DP0:
        case TAG_DP1:
        case TAG_UDP:
        case TAG_ERA:
        case TAG_VAR:
        case TAG_INC:
            return 1;
        default:
            return 0;
    }
}

// ============================================================
// print.c — Debug printer
// ============================================================

static const char *tag_names[] = {
    "APP", "LAM", "VAR", "SUP", "DP0", "DP1", "ERA",
    "NUM", "REF", "OP2", "TEN", "TOP", "CTR",
    "BRI", "ANN", "DSU", "DDU", "INC",
    "EQL", "AND", "OR", "MAT", "ANY", "USP", "UDP", "SEQ"
};

// uop_names now in tinyhvm.h

void thvm_print_term(TinyHVM *ctx, Term t) {
    u32 tag = term_tag(t);
    (void)ctx;
    if (tag < TAG_COUNT) printf("%s", tag_names[tag]);
    else printf("?%u", tag);

    switch (tag) {
        case TAG_NUM:
            if (term_ext(t) == NUM_F32) printf("(%.4f)", term_as_f32(t));
            else printf("(%u)", term_as_u32(t));
            break;
        case TAG_TEN: {
            u32 tid = (u32)term_val(t);
            printf("(id=%u", tid);
            if (ctx && tid < ctx->tensor_count) {
                View *v = &ctx->tensors[tid].view;
                printf(" [");
                for (u32 i = 0; i < v->shape.rank; i++) printf("%s%u", i?",":"", v->shape.dims[i]);
                printf("]");
            }
            printf(")");
            break;
        }
        case TAG_TOP:
            if (term_ext(t) < UOP_COUNT) printf("(%s @%llu)", uop_names[term_ext(t)], (unsigned long long)term_val(t));
            else printf("(uop=%u @%llu)", term_ext(t), (unsigned long long)term_val(t));
            break;
        case TAG_SUP:
            printf("(lab=%u @%llu)", term_ext(t), (unsigned long long)term_val(t));
            break;
        case TAG_BRI:
            printf("(θ @%llu)", (unsigned long long)term_val(t));
            break;
        case TAG_ANN:
            printf("({:} @%llu)", (unsigned long long)term_val(t));
            break;
        case TAG_DSU:
            printf("(&dyn @%llu)", (unsigned long long)term_val(t));
            break;
        case TAG_DDU:
            printf("(!&dyn @%llu)", (unsigned long long)term_val(t));
            break;
        case TAG_INC:
            printf("(↓ @%llu)", (unsigned long long)term_val(t));
            break;
        default:
            if (term_val(t) || term_ext(t))
                printf("(ext=%u, val=%llu)", term_ext(t), (unsigned long long)term_val(t));
            break;
    }
}

// ============================================================
// api.c — High-level tensor API
// ============================================================

// Device registry
extern Backend cpu_backend;
