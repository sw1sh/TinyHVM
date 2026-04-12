// nn/tensor_api.c — concise high-level C API: Tensor + Linear
// Thin wrappers over thvm_tg_* and existing gradient bundle helpers.

#include <math.h>
#include <stdlib.h>

Tensor tensor_from_term(TinyHVM *ctx, Term t) {
    Tensor x;
    x.ctx = ctx;
    x.term = t;
    return x;
}

Tensor tensor_from_f32(TinyHVM *ctx, const f32 *data, Shape s) {
    return tensor_from_term(ctx, thvm_tensor(ctx, data, s));
}

Tensor tensor_realize(Tensor x) {
    x.term = thvm_eval(x.ctx, x.term);
    if (term_tag(x.term) == TAG_TOP) {
        x.term = thvm_force_tensor_term(x.ctx, x.term);
    }
    return x;
}

f32 *tensor_to_host_f32(Tensor x) {
    x = tensor_realize(x);
    return thvm_to_host(x.ctx, x.term);
}

Tensor tensor_matmul(Tensor a, Tensor b) {
    return tensor_from_term(a.ctx, thvm_tg_matmul(a.ctx, a.term, b.term));
}

Tensor tensor_add(Tensor a, Tensor b) {
    return tensor_from_term(a.ctx, thvm_tg_add(a.ctx, a.term, b.term));
}

Tensor tensor_mul(Tensor a, Tensor b) {
    return tensor_from_term(a.ctx, thvm_tg_mul(a.ctx, a.term, b.term));
}

Tensor tensor_relu(Tensor x) {
    return tensor_from_term(x.ctx, thvm_tg_relu(x.ctx, x.term));
}

Tensor tensor_sum_axes(Tensor x, const u32 *axes, u32 n_axes) {
    return tensor_from_term(x.ctx, thvm_tg_sum(x.ctx, x.term, axes, n_axes));
}

// Backward helper: returns the keep-bundle wrapped as Tensor.
Tensor tensor_backward_keep(Tensor y, Tensor x) {
    return tensor_from_term(y.ctx, thvm_grad_keep(y.ctx, y.term, x.term));
}

u32 tensor_bundle_count(Tensor bundle) {
    return thvm_grad_bundle_count(bundle.ctx, bundle.term);
}

Tensor tensor_bundle_get(Tensor bundle, u32 index) {
    return tensor_from_term(bundle.ctx, thvm_grad_bundle_get(bundle.ctx, bundle.term, index));
}

// We store weight as [in_features, out_features] to match thvm_tg_linear.
void linear_init_uniform(Linear *layer, TinyHVM *ctx, u32 in_features,
                         u32 out_features, int has_bias) {
    u32 n_w = in_features * out_features;
    f32 *w = (f32 *)malloc((size_t)n_w * sizeof(f32));
    f32 bound = 1.0f / sqrtf((f32)in_features);
    for (u32 i = 0; i < n_w; i++) {
        f32 u = (f32)rand() / (f32)RAND_MAX;
        w[i] = -bound + (2.0f * bound) * u;
    }
    layer->weight = tensor_from_f32(ctx, w, SHAPE(in_features, out_features));
    free(w);

    layer->has_bias = has_bias ? 1 : 0;
    if (layer->has_bias) {
        f32 *b = (f32 *)malloc((size_t)out_features * sizeof(f32));
        for (u32 i = 0; i < out_features; i++) {
            f32 u = (f32)rand() / (f32)RAND_MAX;
            b[i] = -bound + (2.0f * bound) * u;
        }
        layer->bias = tensor_from_f32(ctx, b, SHAPE(out_features));
        free(b);
    } else {
        layer->bias = tensor_from_term(ctx, term_era());
    }
}

Tensor linear_forward(Linear *layer, Tensor x) {
    const Term *bias_opt = layer->has_bias ? &layer->bias.term : NULL;
    return tensor_from_term(x.ctx,
                            thvm_tg_linear(x.ctx, x.term, layer->weight.term, bias_opt));
}
