// tensor/view/pad.c — view_pad(): extend shape for padding (logical only)
// Note: padding requires a physical copy at dispatch time.
static View view_pad(View v, const u32 *pad_before, const u32 *pad_after) {
  View r   = v;
  r.numel  = 1;
  for (u32 i = 0; i < v.shape.rank; i++) {
    r.shape.dims[i] = v.shape.dims[i] + pad_before[i] + pad_after[i];
    r.numel         *= r.shape.dims[i];
  }
  r.contiguous = 0;
  return r;
}
