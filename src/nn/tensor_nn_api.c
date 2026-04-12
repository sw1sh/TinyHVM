// nn/tensor_nn_api.c — high-level Tensor wrappers for nn layers/ops.
// Included from nn/_.c after conv/batchnorm/softmax/loss implementations.

Tensor tensor_conv2d(Tensor x, Tensor w, const Tensor *bias_opt, u32 groups,
                     const u32 *stride_, const u32 *padding_) {
    Term b = (bias_opt && term_tag(bias_opt->term) != TAG_ERA) ? bias_opt->term : term_era();
    return tensor_from_term(x.ctx, thvm_conv2d(x.ctx, x.term, w.term, b, groups, stride_, padding_));
}

Tensor tensor_maxpool2d(Tensor x, const u32 *kernel, const u32 *stride_) {
    return tensor_from_term(x.ctx, thvm_maxpool2d(x.ctx, x.term, kernel, stride_));
}

Tensor tensor_batchnorm(Tensor x, Tensor gamma, Tensor beta, Tensor rmean, Tensor rvar,
                        u32 B, u32 C, u32 H, u32 W, int training) {
    return tensor_from_term(
        x.ctx, batchnorm_term(x.ctx, x.term, gamma.term, beta.term, rmean.term, rvar.term,
                              B, C, H, W, training));
}

Tensor tensor_softmax(Tensor logits, u32 B, u32 C) {
    return tensor_from_term(logits.ctx, softmax(logits.ctx, logits.term, B, C));
}

Tensor tensor_cross_entropy(Tensor logits, const u8 *labels, u32 B, u32 C) {
    return tensor_from_term(logits.ctx, cross_entropy_loss(logits.ctx, logits.term, (u8 *)labels, B, C));
}
