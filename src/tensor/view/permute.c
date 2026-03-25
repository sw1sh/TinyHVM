// tensor/view/permute.c — view_permute(): reorder axes
static View view_permute(View v, const u32 *axes) {
  View r = v;
  for (u32 i = 0; i < v.shape.rank; i++) {
    r.shape.dims[i] = v.shape.dims[axes[i]];
    r.strides[i]    = v.strides[axes[i]];
  }
  if (v.has_mask) {
    r.has_mask = 1;
    for (u32 i = 0; i < v.shape.rank; i++) {
      r.mask_begin[i] = v.mask_begin[axes[i]];
      r.mask_end[i] = v.mask_end[axes[i]];
    }
  }
  r.contiguous = 0;
  return r;
}
