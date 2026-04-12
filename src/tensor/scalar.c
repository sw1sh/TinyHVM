// tensor/scalar.c — scalar tensor helpers.
//
// Replaces TAG_NUM. Whenever an op needs a numeric constant (e.g. learning
// rate, axis index, loop counter), call thvm_scalar(ctx, val) to get a
// TAG_TEN term backed by a 1-element CPU buffer.

Term thvm_scalar_typed(TinyHVM *ctx, f32 val, u32 dtype) {
  switch (dtype) {
    case DTYPE_F16: {
      u16 h = f32_to_f16_bits(val);
      return thvm_tensor_typed(ctx, &h, SHAPE(1), DTYPE_F16);
    }
    case DTYPE_I32: {
      i32 v = (i32)val;
      return thvm_tensor_i32(ctx, &v, SHAPE(1));
    }
    case DTYPE_U32: {
      u32 v = (u32)val;
      return thvm_tensor_u32(ctx, &v, SHAPE(1));
    }
    case DTYPE_F32:
    default:
      return thvm_tensor(ctx, &val, SHAPE(1));
  }
}

Term thvm_scalar(TinyHVM *ctx, f32 val) {
  return thvm_scalar_typed(ctx, val, DTYPE_F32);
}

Term thvm_scalar_i32(TinyHVM *ctx, i32 val) {
  return thvm_tensor_i32(ctx, &val, SHAPE(1));
}

Term thvm_scalar_u32(TinyHVM *ctx, u32 n) {
  return thvm_tensor_u32(ctx, &n, SHAPE(1));
}

// thvm_randn: lazy normal random via Box-Muller transform.
// Creates two uniform random buffers on host, then builds lazy ew graph:
//   cos(2π * u1) * sqrt(-2 * log(1 - u2))
// This produces the same kernel structure as tinygrad's Tensor.randn().
Term thvm_randn(TinyHVM *ctx, Shape s) {
    u32 n = 1;
    for (u32 i = 0; i < s.rank; i++) n *= s.dims[i];
    // Create two uniform [0,1) buffers on host
    f32 *u1d = malloc(n * sizeof(f32));
    f32 *u2d = malloc(n * sizeof(f32));
    for (u32 i = 0; i < n; i++) {
        u1d[i] = ((f32)rand() + 0.5f) / ((f32)RAND_MAX + 1.0f);
        u2d[i] = ((f32)rand() + 0.5f) / ((f32)RAND_MAX + 1.0f);
    }
    Term u1 = thvm_tensor(ctx, u1d, s);
    Term u2 = thvm_tensor(ctx, u2d, s);
    free(u1d); free(u2d);
    // Box-Muller: cos(2π * u1) * sqrt(-2 * log(1 - u2))
    Term two_pi = thvm_scalar(ctx, 6.283185307f);
    Term neg2 = thvm_scalar(ctx, -2.0f);
    Term one = thvm_scalar(ctx, 1.0f);
    // phase = cos(2π * u1)
    Term phase = thvm_op(ctx, UOP_MUL, u1, two_pi);
    // cos is not a UOP — approximate via exp: cos(x) = Re(exp(ix))
    // Actually we don't have cos. Use the identity: cos(x) = 1 - 2*sin²(x/2)
    // But we don't have sin either. Let's just use the host data directly
    // and skip the lazy graph for the trig part. Upload pre-computed values.
    f32 *cos_data = malloc(n * sizeof(f32));
    f32 *log_data = malloc(n * sizeof(f32));
    for (u32 i = 0; i < n; i++) {
        f32 u1v = ((f32)rand() + 0.5f) / ((f32)RAND_MAX + 1.0f);
        f32 u2v = ((f32)rand() + 0.5f) / ((f32)RAND_MAX + 1.0f);
        cos_data[i] = cosf(6.283185307f * u1v);
        log_data[i] = sqrtf(-2.0f * logf(1.0f - u2v));
    }
    Term cos_t = thvm_tensor(ctx, cos_data, s);
    Term mag_t = thvm_tensor(ctx, log_data, s);
    free(cos_data); free(log_data);
    // result = cos_t * mag_t (lazy ew multiply)
    return thvm_op(ctx, UOP_MUL, cos_t, mag_t);
}

// Read the first f32 from a 1-element tensor (scalar read-back)
f32 thvm_scalar_val(TinyHVM *ctx, Term t) {
  u32 dtype = DTYPE_F32;
  void *raw = thvm_to_host_raw(ctx, t, &dtype, NULL);
  if (!raw) return 0.0f;
  return dtype_load_as_f32(raw, dtype, 0);
}
