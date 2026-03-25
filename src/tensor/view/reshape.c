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

  // Check if input is truly contiguous — stride-0 broadcasts are NOT,
  // even though expand sets contiguous=1 for the view.
  int truly_contiguous = 1;
  if (v.numel > 1) {
    for (u32 i = 0; i < v.shape.rank; i++) {
      if (v.shape.dims[i] > 1 && v.strides[i] == 0) {
        truly_contiguous = 0;
        break;
      }
    }
    // Also check stride ordering for non-broadcast non-contiguity
    if (truly_contiguous) {
      i32 expected = 1;
      for (int i = (int)v.shape.rank - 1; i >= 0; i--) {
        if (v.shape.dims[i] > 1) {
          if (v.strides[i] != expected) { truly_contiguous = 0; break; }
          expected *= (i32)v.shape.dims[i];
        }
      }
    }
  }

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
  if (v.has_mask) r.contiguous = 0;  // masked reshape requires materialization
  else r.contiguous = truly_contiguous;
  return r;
}
