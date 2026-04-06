// tensor/view/create.c — view_create(): build a row-major View from a Shape

// Canonicalize strides: size-1 dims get stride 0 (like tinygrad).
// Must be called on every View construction.
static inline void view_canonicalize(View *v) {
    for (u32 i = 0; i < v->shape.rank; i++)
        if (v->shape.dims[i] == 1) v->strides[i] = 0;
}

static View view_create(Shape s) {
  View v = {0};
  v.shape = s;
  v.numel = 1;
  for (u32 i = 0; i < s.rank; i++) {
    v.numel *= s.dims[i];
  }
  for (u32 i = 0; i < s.rank; i++) {
    i32 st = 1;
    for (u32 j = i + 1; j < s.rank; j++) {
      st *= (i32)s.dims[j];
    }
    v.strides[i] = st;
  }
  v.offset      = 0;
  v.contiguous  = 1;
  v.has_mask    = 0;
  view_canonicalize(&v);
  return v;
}
