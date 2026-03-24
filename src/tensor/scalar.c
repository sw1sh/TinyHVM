// tensor/scalar.c — thvm_scalar(): create a [1] tensor from a single f32 value.
//
// Replaces TAG_NUM. Whenever an op needs a numeric constant (e.g. learning
// rate, axis index, loop counter), call thvm_scalar(ctx, val) to get a
// TAG_TEN term backed by a 1-element CPU buffer.

Term thvm_scalar(TinyHVM *ctx, f32 val) {
  return thvm_tensor(ctx, &val, SHAPE(1));
}

// thvm_scalar_u32: encode a u32 as f32 scalar (for axes, step counters, etc.)
Term thvm_scalar_u32(TinyHVM *ctx, u32 n) {
  f32 v = (f32)n;
  return thvm_scalar(ctx, v);
}

// Read the first f32 from a 1-element tensor (scalar read-back)
f32 thvm_scalar_val(TinyHVM *ctx, Term t) {
  assert(term_tag(t) == TAG_TEN);
  f32 v;
  ctx->backend->buf_read(ctx->tensors[(u32)term_val(t)].buf_id, &v, sizeof(f32));
  return v;
}
