// nn/tg_tensor.c — tinygrad-style tensor helpers (Tensor.dot / .linear / .relu / …)
//
// Conventions (see sibling tinygrad repo, e.g. tinygrad/tensor.py `Tensor.linear`,
// tinygrad/nn/__init__.py `Linear`):
//   - `thvm_tg_matmul(a, b)` is `a @ b` / `Tensor.dot`: last dim of `a` matches first of `b`.
//   - `thvm_tg_linear(x, weight, bias)` matches `x.linear(weight, bias)` when `weight`
//     has shape `[in_features, out_features]` (what nn.Linear passes as `weight.T`).
//   - Elementwise ops use numpy-style broadcasting via the existing lazy UOp rules.
//   - `x` for `thvm_tg_linear` must be `TAG_TEN` so batch size can be inferred from shape.

#include <stdio.h>

static int tg_term_ten_shape(TinyHVM *ctx, Term t, Shape *out) {
    if (term_tag(t) != TAG_TEN) return 0;
    u32 id = (u32)term_val(t);
    if (id >= ctx->tensor_count) return 0;
    *out = ctx->tensors[id].view.shape;
    return 1;
}

Term thvm_tg_matmul(TinyHVM *ctx, Term a, Term b) { return thvm_mm(ctx, a, b); }

Term thvm_tg_add(TinyHVM *ctx, Term a, Term b) { return thvm_op(ctx, UOP_ADD, a, b); }

Term thvm_tg_mul(TinyHVM *ctx, Term a, Term b) { return thvm_op(ctx, UOP_MUL, a, b); }

Term thvm_tg_relu(TinyHVM *ctx, Term x) { return thvm_op(ctx, UOP_RELU, x, term_era()); }

Term thvm_tg_sum(TinyHVM *ctx, Term x, const u32 *axes, u32 n_axes) {
    return thvm_sum_axes(ctx, x, axes, n_axes);
}

// y = x @ weight  (+ bias), same as tinygrad Tensor.linear(weight, bias) for 2D weight.
Term thvm_tg_linear(TinyHVM *ctx, Term x, Term weight, const Term *bias_opt) {
    Shape sx, sw;
    if (!tg_term_ten_shape(ctx, x, &sx) || !tg_term_ten_shape(ctx, weight, &sw)) {
        fprintf(stderr,
                "thvm_tg_linear: x and weight must be realized tensors (TAG_TEN) for shapes\n");
        return term_era();
    }
    if (sw.rank != 2) {
        fprintf(stderr, "thvm_tg_linear: weight must be rank-2 [in_features, out_features]\n");
        return term_era();
    }
    if (sx.rank < 1) {
        fprintf(stderr, "thvm_tg_linear: x rank must be >= 1\n");
        return term_era();
    }
    u32 in_x = sx.dims[sx.rank - 1];
    u32 in_w = sw.dims[0];
    u32 out_f = sw.dims[1];
    if (in_x != in_w) {
        fprintf(stderr, "thvm_tg_linear: shape mismatch x last=%u vs weight[0]=%u\n", in_x, in_w);
        return term_era();
    }

    u32 B = 1;
    for (u32 i = 0; i + 1 < sx.rank; i++) B *= sx.dims[i];

    Term y = thvm_mm(ctx, x, weight);
    if (!bias_opt || term_tag(*bias_opt) == TAG_ERA) return y;

    Term b = *bias_opt;
    Shape sb;
    if (!tg_term_ten_shape(ctx, b, &sb)) {
        fprintf(stderr, "thvm_tg_linear: bias must be TAG_TEN\n");
        return term_era();
    }

    // Accept [out_f] or [1, out_f]; broadcast to [B, out_f] like nn/linear.c + tinygrad.
    if (sb.rank == 1) {
        if (sb.dims[0] != out_f) {
            fprintf(stderr, "thvm_tg_linear: bias length %u != out_features %u\n", sb.dims[0],
                    out_f);
            return term_era();
        }
        Term br = thvm_reshape(ctx, b, SHAPE(1, out_f));
        Term bc = thvm_expand(ctx, br, SHAPE(B, out_f));
        return thvm_tg_add(ctx, y, bc);
    }
    if (sb.rank == 2 && sb.dims[0] == 1 && sb.dims[1] == out_f) {
        Term bc = thvm_expand(ctx, b, SHAPE(B, out_f));
        return thvm_tg_add(ctx, y, bc);
    }

    fprintf(stderr, "thvm_tg_linear: bias shape must be [out_features] or [1,out_features]\n");
    return term_era();
}
