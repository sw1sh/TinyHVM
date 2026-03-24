// tensor/view/reshape.c — view_reshape(): change shape, validate numel
static View view_reshape(View v, Shape new_shape) {
  u32 new_numel = 1;
  for (u32 i = 0; i < new_shape.rank; i++) {
    new_numel *= new_shape.dims[i];
  }
  if (new_numel != v.numel) {
    fprintf(stderr, "reshape: numel mismatch old=%u new=%u\n", v.numel, new_numel);
  }
  assert(new_numel == v.numel && "reshape: numel mismatch");
  View r = {0};
  r.shape  = new_shape;
  r.numel  = new_numel;
  r.offset = v.offset;
  for (u32 i = 0; i < new_shape.rank; i++) {
    i32 st = 1;
    for (u32 j = i + 1; j < new_shape.rank; j++) {
      st *= (i32)new_shape.dims[j];
    }
    r.strides[i] = st;
  }
  r.contiguous = v.contiguous;
  return r;
}
