// tensor/view/shrink.c — view_shrink(): slice via offset + shape adjustment
static View view_shrink(View v, const u32 *starts, const u32 *ends) {
  View r   = v;
  r.numel  = 1;
  for (u32 i = 0; i < v.shape.rank; i++) {
    r.offset        += (i32)starts[i] * v.strides[i];
    r.shape.dims[i]  = ends[i] - starts[i];
    r.numel          *= r.shape.dims[i];
  }
  r.contiguous = 0;
  view_canonicalize(&r);
  return r;
}
