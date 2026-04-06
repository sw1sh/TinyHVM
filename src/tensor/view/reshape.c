// tensor/view/reshape.c — view_reshape(): direct port of tinygrad's View.reshape()
// Uses merge_dims() + reversed walk, matching tinygrad/shape/view.py exactly.

// merge_dims: port of tinygrad's merge_dims() (view.py:45-64)
// Returns array of (merged_size, stride, real_size) tuples.
// real_size = merged_size excluding stride-0 dims (broadcast).
typedef struct { u32 merged_size; i32 stride; u32 real_size; } MergedDim;

static u32 merge_dims(const View *v, MergedDim *ret) {
  u32 rank = v->shape.rank;
  if (rank == 0) return 0;

  u32 n = 0;
  ret[0] = (MergedDim){
    .merged_size = v->shape.dims[0],
    .stride = v->strides[0],
    .real_size = (v->strides[0] != 0) ? v->shape.dims[0] : 0
  };
  n = 1;

  // merging = true if this dim has mask width 1 (or size 1 with no mask)
  int merging = v->has_mask
    ? ((i32)(v->mask_end[0] - v->mask_begin[0]) == 1)
    : (v->shape.dims[0] == 1);

  for (u32 i = 1; i < rank; i++) {
    u32 s = v->shape.dims[i];
    i32 st = v->strides[i];

    // always merge size-1 dims
    if (s == 1) continue;

    u32 last_s = ret[n-1].merged_size;
    i32 last_st = ret[n-1].stride;
    u32 last_rs = ret[n-1].real_size;

    // merge if: previous was "merging" (size-1 or mask-width-1),
    // OR strides are contiguous (last_stride == s * st)
    if (merging || last_st == (i32)(s * (u32)st)) {
      ret[n-1].merged_size = last_s * s;
      ret[n-1].stride = st;
      ret[n-1].real_size = merging ? s : last_rs * s;
    } else {
      ret[n++] = (MergedDim){ .merged_size = s, .stride = st, .real_size = s };
    }

    merging = v->has_mask
      ? ((i32)(v->mask_end[i] - v->mask_begin[i]) == 1)
      : (s == 1);
  }
  return n;
}

// _reshape_mask: port of tinygrad's _reshape_mask() (view.py:67-96)
// Returns 1 if mask was successfully reshaped, 0 if not possible.
static int reshape_mask(const View *v, Shape new_shape,
                        u32 *new_mask_begin, u32 *new_mask_end) {
  if (!v->has_mask) {
    for (u32 i = 0; i < new_shape.rank; i++) {
      new_mask_begin[i] = 0;
      new_mask_end[i] = new_shape.dims[i];
    }
    return 1;
  }

  // Process from right to left (reversed)
  u32 nm_count = 0;
  u32 nm_b[MAX_DIM], nm_e[MAX_DIM]; // reversed order

  int oi = (int)v->shape.rank - 1;   // old shape index (reversed)
  int ni = (int)new_shape.rank - 1;  // new shape index (reversed)
  int mi = (int)v->shape.rank - 1;   // mask index (reversed)

  u32 curr_stride = 1;
  u32 old_dim = (oi >= 0) ? v->shape.dims[oi--] : 1;
  u32 new_dim = (ni >= 0) ? new_shape.dims[ni--] : 1;
  u32 mask_l = (mi >= 0) ? v->mask_begin[mi] : 0;
  u32 mask_r = (mi >= 0) ? v->mask_end[mi] : 1;
  if (mi >= 0) mi--;

  while (nm_count < new_shape.rank) {
    u32 next_stride = new_dim * curr_stride;

    if (old_dim == next_stride) {
      // Simply copy mask and get next batch
      nm_b[nm_count] = mask_l / curr_stride;
      nm_e[nm_count] = (mask_r - 1) / curr_stride + 1;
      nm_count++;
      curr_stride = 1;
      old_dim = (oi >= 0) ? v->shape.dims[oi--] : 1;
      new_dim = (ni >= 0) ? new_shape.dims[ni--] : 1;
      mask_l = (mi >= 0) ? v->mask_begin[mi] : 0;
      mask_r = (mi >= 0) ? v->mask_end[mi] : 1;
      if (mi >= 0) mi--;
    } else if (old_dim > next_stride) {
      // Mask can only be split if reshape doesn't cut across the mask
      if (old_dim % next_stride != 0) return 0;
      if ((mask_l % next_stride != 0 || mask_r % next_stride != 0) &&
          mask_l / next_stride != (mask_r - 1) / next_stride) return 0;
      nm_b[nm_count] = (mask_l % next_stride) / curr_stride;
      nm_e[nm_count] = ((mask_r - 1) % next_stride) / curr_stride + 1;
      nm_count++;
      curr_stride = next_stride;
      new_dim = (ni >= 0) ? new_shape.dims[ni--] : 1;
    } else {
      // old_dim < next_stride: combine masks
      u32 next_mask_l = (mi >= 0) ? v->mask_begin[mi] : 0;
      u32 next_mask_r = (mi >= 0) ? v->mask_end[mi] : 1;
      if (mi >= 0) mi--;
      // combine if mask can unfold continuously
      if (mask_l != 0 && mask_r != 0 && mask_l != mask_r &&
          next_mask_r - next_mask_l != 1) return 0;
      u32 combined_l = next_mask_l * old_dim + mask_l;
      u32 combined_r = (next_mask_r - 1) * old_dim + mask_r;
      mask_l = combined_l;
      mask_r = combined_r;
      u32 next_old = (oi >= 0) ? v->shape.dims[oi--] : 1;
      old_dim = old_dim * next_old;
    }
  }

  // Reverse the output
  for (u32 i = 0; i < new_shape.rank; i++) {
    new_mask_begin[i] = nm_b[new_shape.rank - 1 - i];
    new_mask_end[i] = nm_e[new_shape.rank - 1 - i];
  }
  return 1;
}

static View view_reshape(View v, Shape new_shape) {
  u32 new_numel = 1;
  for (u32 i = 0; i < new_shape.rank; i++)
    new_numel *= new_shape.dims[i];
  if (new_numel != v.numel) {
    fprintf(stderr, "reshape: numel mismatch old=%u new=%u\n", v.numel, new_numel);
  }
  assert(new_numel == v.numel && "reshape: numel mismatch");

  // Same shape → identity
  if (v.shape.rank == new_shape.rank) {
    int same = 1;
    for (u32 i = 0; i < v.shape.rank; i++)
      if (v.shape.dims[i] != new_shape.dims[i]) { same = 0; break; }
    if (same) return v;
  }

  // Contiguous → just change shape
  if (v.contiguous) {
    View r = {0};
    r.shape = new_shape;
    r.numel = new_numel;
    // Compute natural strides
    if (new_shape.rank > 0) {
      r.strides[new_shape.rank - 1] = 1;
      for (int i = (int)new_shape.rank - 2; i >= 0; i--)
        r.strides[i] = r.strides[i+1] * (i32)new_shape.dims[i+1];
    }
    // Canonicalize: stride=0 for size-1 dims
    for (u32 i = 0; i < new_shape.rank; i++)
      if (new_shape.dims[i] == 1) r.strides[i] = 0;
    r.contiguous = (v.offset == 0 && !v.has_mask);
    r.offset = v.offset;
    return r;
  }

  // Port of tinygrad View.reshape() lines 325-342
  // Use merge_dims then walk reversed
  MergedDim md[MAX_DIM];
  u32 n_md = merge_dims(&v, md);

  i32 r_strides[MAX_DIM];
  u32 r_count = 0;
  int ni_rev = (int)new_shape.rank - 1; // walk new_shape reversed

  // Walk merged dims reversed
  for (int mi = (int)n_md - 1; mi >= 0; mi--) {
    u32 merged_size = md[mi].merged_size;
    i32 new_stride = md[mi].stride;
    u32 real_size = md[mi].real_size;

    u32 acc = 1;
    while (acc < merged_size && ni_rev >= 0) {
      u32 new_dim = new_shape.dims[ni_rev];
      if (new_dim == 0) break;
      r_strides[r_count++] = new_stride * (i32)acc;
      acc *= new_dim;
      ni_rev--;
      if (acc >= real_size && real_size > 0) new_stride = 0;
    }
    if (acc != merged_size) {
      // Can't reshape — return fallback with natural strides
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
  }

  // Prepend zeros for remaining new dims (size-1 dims at the front)
  u32 n_prepend = new_shape.rank - r_count;

  View r = {0};
  r.shape = new_shape;
  r.numel = new_numel;
  r.offset = v.offset;

  // Fill strides: n_prepend zeros + reversed r_strides
  for (u32 i = 0; i < n_prepend; i++)
    r.strides[i] = 0;
  for (u32 i = 0; i < r_count; i++)
    r.strides[n_prepend + i] = r_strides[r_count - 1 - i];

  // Check contiguous
  r.contiguous = 0;
  if (r.offset == 0 && !v.has_mask) {
    r.contiguous = 1;
    i32 expected = 1;
    for (int i = (int)new_shape.rank - 1; i >= 0; i--) {
      if (new_shape.dims[i] > 1) {
        if (r.strides[i] != expected) { r.contiguous = 0; break; }
        expected *= (i32)new_shape.dims[i];
      }
    }
  }

  // Reshape mask
  if (v.has_mask) {
    u32 new_mb[MAX_DIM], new_me[MAX_DIM];
    if (reshape_mask(&v, new_shape, new_mb, new_me)) {
      r.has_mask = 1;
      // Adjust offset for mask shift
      i32 old_mask_off = 0, new_mask_off = 0;
      for (u32 i = 0; i < v.shape.rank; i++)
        old_mask_off += (i32)v.mask_begin[i] * v.strides[i];
      for (u32 i = 0; i < new_shape.rank; i++)
        new_mask_off += (i32)new_mb[i] * r.strides[i];
      r.offset += old_mask_off - new_mask_off;
      for (u32 i = 0; i < new_shape.rank; i++) {
        r.mask_begin[i] = new_mb[i];
        r.mask_end[i] = new_me[i];
      }
      // Check if mask is trivial (all full range)
      int trivial = 1;
      for (u32 i = 0; i < new_shape.rank; i++)
        if (r.mask_begin[i] != 0 || r.mask_end[i] != new_shape.dims[i]) { trivial = 0; break; }
      if (trivial) r.has_mask = 0;
    } else {
      // Mask reshape failed → not reshapable
      r.has_mask = 0;
      for (u32 i = 0; i < new_shape.rank; i++) {
        i32 st = 1;
        for (u32 j = i + 1; j < new_shape.rank; j++) st *= (i32)new_shape.dims[j];
        r.strides[i] = st;
      }
      r.contiguous = 0;
      return r;
    }
  }

  return r;
}
