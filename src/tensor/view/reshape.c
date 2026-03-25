// tensor/view/reshape.c — view_reshape(): change shape, compute strides
// Implements merge-split algorithm matching tinygrad's ShapeTracker.
// For non-contiguous views (expanded, permuted), computes valid strides
// when possible, avoiding materialization.

static View view_reshape(View v, Shape new_shape) {
  u32 new_numel = 1;
  for (u32 i = 0; i < new_shape.rank; i++)
    new_numel *= new_shape.dims[i];
  if (new_numel != v.numel)
    fprintf(stderr, "reshape: numel mismatch old=%u new=%u\n", v.numel, new_numel);
  assert(new_numel == v.numel && "reshape: numel mismatch");

  if (v.has_mask) {
    // Masked views need materialization — return non-contiguous placeholder
    View r = {0};
    r.shape = new_shape; r.numel = new_numel; r.offset = v.offset;
    for (u32 i = 0; i < new_shape.rank; i++) {
      i32 st = 1;
      for (u32 j = i + 1; j < new_shape.rank; j++) st *= (i32)new_shape.dims[j];
      r.strides[i] = st;
    }
    r.contiguous = 0;
    return r;
  }

  // Merge-split algorithm: walk old and new shapes simultaneously.
  // Skip dims of size 1 (they don't affect data layout).
  // For each group of merged old dims → one new dim (or split):
  //   - All stride-0 → new stride is 0
  //   - Contiguous chain → new stride is the innermost stride
  //   - Otherwise → not reshapable (need materialization)

  // Collect non-trivial (size > 1) old dims
  u32 od[MAX_DIM], os[MAX_DIM], on = 0;
  for (u32 i = 0; i < v.shape.rank; i++) {
    if (v.shape.dims[i] > 1) {
      od[on] = v.shape.dims[i];
      os[on] = (u32)v.strides[i];
      on++;
    }
  }
  // Collect non-trivial new dims
  u32 nd[MAX_DIM], nn = 0;
  u32 new_dim_map[MAX_DIM]; // index into new_shape for each non-trivial dim
  for (u32 i = 0; i < new_shape.rank; i++) {
    if (new_shape.dims[i] > 1) {
      new_dim_map[nn] = i;
      nd[nn] = new_shape.dims[i];
      nn++;
    }
  }

  // Compute strides for new shape — start with 0 for trivial dims
  i32 new_strides[MAX_DIM];
  for (u32 i = 0; i < new_shape.rank; i++) new_strides[i] = 0;

  // Walk both dimension lists
  u32 oi = 0, ni = 0;
  int reshapable = 1;

  while (oi < on && ni < nn) {
    // Match old dims to new dims by accumulating products
    u32 old_prod = od[oi];
    u32 old_start = oi;
    oi++;

    u32 new_prod = nd[ni];
    u32 new_start = ni;
    ni++;

    // Grow whichever side is smaller until products match
    while (old_prod != new_prod) {
      if (old_prod < new_prod) {
        if (oi >= on) { reshapable = 0; break; }
        old_prod *= od[oi++];
      } else {
        if (ni >= nn) { reshapable = 0; break; }
        new_prod *= nd[ni++];
      }
    }
    if (!reshapable) break;

    // Check if old dims in [old_start, oi) are compatible for merging
    int all_zero = 1, any_zero = 0;
    for (u32 k = old_start; k < oi; k++) {
      if (os[k] == 0) any_zero = 1;
      else all_zero = 0;
    }

    if (all_zero) {
      // All stride-0: new dims all get stride 0
      for (u32 k = new_start; k < ni; k++)
        new_strides[new_dim_map[k]] = 0;
    } else if (any_zero) {
      // Mix of stride-0 and non-zero: can't reshape
      reshapable = 0; break;
    } else {
      // All non-zero: check contiguity within the merged group
      for (u32 k = old_start; k + 1 < oi; k++) {
        if ((i32)os[k] != (i32)(od[k + 1] * os[k + 1])) {
          reshapable = 0; break;
        }
      }
      if (!reshapable) break;

      // Assign strides to new dims: innermost gets the innermost old stride,
      // then multiply outward
      i32 inner_stride = (i32)os[oi - 1];
      for (int k = (int)ni - 1; k >= (int)new_start; k--) {
        new_strides[new_dim_map[k]] = inner_stride;
        inner_stride *= (i32)nd[k];
      }
    }
  }

  // If any dims left over, not reshapable
  if (oi != on || ni != nn) reshapable = 0;

  View r = {0};
  r.shape = new_shape;
  r.numel = new_numel;
  r.offset = v.offset;

  if (reshapable) {
    for (u32 i = 0; i < new_shape.rank; i++)
      r.strides[i] = new_strides[i];
    // Check if the result is truly contiguous
    r.contiguous = 1;
    i32 expected = 1;
    for (int i = (int)new_shape.rank - 1; i >= 0; i--) {
      if (new_shape.dims[i] > 1) {
        if (r.strides[i] != expected) { r.contiguous = 0; break; }
        expected *= (i32)new_shape.dims[i];
      }
    }
  } else {
    // Fallback: contiguous strides, mark non-contiguous for materialization
    for (u32 i = 0; i < new_shape.rank; i++) {
      i32 st = 1;
      for (u32 j = i + 1; j < new_shape.rank; j++) st *= (i32)new_shape.dims[j];
      r.strides[i] = st;
    }
    r.contiguous = 0;
  }
  return r;
}
