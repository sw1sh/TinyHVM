// tensor/view/pad.c — view_pad(): set mask for zero-copy padding
static View view_pad(View v, const u32 *pad_before, const u32 *pad_after) {
  View r = v;
  r.has_mask = 1;
  r.numel = 1;
  for (u32 i = 0; i < v.shape.rank; i++) {
    r.mask_begin[i] = pad_before[i];
    r.mask_end[i] = pad_before[i] + v.shape.dims[i];
    r.shape.dims[i] = v.shape.dims[i] + pad_before[i] + pad_after[i];
    r.offset -= (i32)pad_before[i] * v.strides[i];
    r.numel *= r.shape.dims[i];
  }
  r.contiguous = 0;
  view_canonicalize(&r);
  return r;
}
