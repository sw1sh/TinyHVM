// tensor/view/expand.c — view_expand(): set stride=0 for broadcast dims
static View view_expand(View v, Shape new_shape) {
  assert(v.shape.rank == new_shape.rank);
  View r   = v;
  r.numel  = 1;
  for (u32 i = 0; i < new_shape.rank; i++) {
    if (v.shape.dims[i] == 1 && new_shape.dims[i] > 1) {
      r.strides[i] = 0;  // broadcast!
    } else {
      assert(v.shape.dims[i] == new_shape.dims[i]);
    }
    r.shape.dims[i] = new_shape.dims[i];
    r.numel         *= new_shape.dims[i];
  }
  r.contiguous = 0;
  return r;
}
