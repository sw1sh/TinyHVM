// tensor/view/stride.c — view_stride(): dilate strides for pooling window extraction
static View view_stride(View v, const u32 *strides_mult) {
  View r   = v;
  r.numel  = 1;
  for (u32 i = 0; i < v.shape.rank; i++) {
    r.shape.dims[i]  = (v.shape.dims[i] + strides_mult[i] - 1) / strides_mult[i];
    r.strides[i]     = v.strides[i] * (i32)strides_mult[i];
    r.numel          *= r.shape.dims[i];
  }
  r.contiguous = 0;
  return r;
}
