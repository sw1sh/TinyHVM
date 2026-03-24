// tensor/view/broadcast.c — view_broadcast(): align two views to a common broadcast shape
static int view_broadcast(const View *a, const View *b,
                           View *out_a, View *out_b,
                           u32 *out_shape, u32 *out_ndim) {
  u32 ndim  = a->shape.rank > b->shape.rank ? a->shape.rank : b->shape.rank;
  *out_ndim = ndim;

  u32 sa[MAX_DIM]  = {0}, sb[MAX_DIM]  = {0};
  i32 sta[MAX_DIM] = {0}, stb[MAX_DIM] = {0};
  u32 off_a = ndim - a->shape.rank;
  u32 off_b = ndim - b->shape.rank;

  for (u32 i = 0; i < ndim; i++) {
    sa[i]  = (i >= off_a) ? a->shape.dims[i - off_a]  : 1;
    sb[i]  = (i >= off_b) ? b->shape.dims[i - off_b]  : 1;
    sta[i] = (i >= off_a) ? a->strides[i  - off_a]    : 0;
    stb[i] = (i >= off_b) ? b->strides[i  - off_b]    : 0;
  }

  u32 numel = 1;
  for (u32 i = 0; i < ndim; i++) {
    if      (sa[i] == sb[i]) { out_shape[i] = sa[i]; }
    else if (sa[i] == 1)     { out_shape[i] = sb[i]; }
    else if (sb[i] == 1)     { out_shape[i] = sa[i]; }
    else { return 0; }  // incompatible
    numel *= out_shape[i];
  }

  *out_a = (View){0};
  *out_b = (View){0};
  out_a->shape.rank = out_b->shape.rank = ndim;
  out_a->numel      = out_b->numel      = numel;
  out_a->offset     = a->offset;
  out_b->offset     = b->offset;
  for (u32 i = 0; i < ndim; i++) {
    out_a->shape.dims[i] = out_b->shape.dims[i] = out_shape[i];
    out_a->strides[i]    = (sa[i] == 1 && out_shape[i] > 1) ? 0 : sta[i];
    out_b->strides[i]    = (sb[i] == 1 && out_shape[i] > 1) ? 0 : stb[i];
  }
  return 1;
}
