// tensor/view/reshape.c — view_reshape(): change shape, compute strides
// Implements merge-split algorithm matching tinygrad's ShapeTracker.
// For non-contiguous views (expanded, permuted), computes valid strides
// when possible, avoiding materialization.

static View view_reshape(View v, Shape new_shape) {
  u32 new_numel = 1;
  for (u32 i = 0; i < new_shape.rank; i++)
    new_numel *= new_shape.dims[i];
  if (new_numel != v.numel) {
    fprintf(stderr, "reshape: numel mismatch old=%u new=%u old_shape=[", v.numel, new_numel);
    for (u32 _d=0;_d<v.shape.rank;_d++) fprintf(stderr,"%u,",v.shape.dims[_d]);
    fprintf(stderr,"] new_shape=[");
    for (u32 _d=0;_d<new_shape.rank;_d++) fprintf(stderr,"%u,",new_shape.dims[_d]);
    fprintf(stderr,"]\n");
  }
  assert(new_numel == v.numel && "reshape: numel mismatch");

  if (v.has_mask) {
    // Try to propagate mask through reshape.
    // This works when both old and new shapes have the same element order
    // (both contiguous or compatible stride pattern). The mask bounds
    // transform through the same merge-split as the strides.
    // For now: only handle the case where the source is contiguous-strided
    // (the mask was applied to a contiguous buffer via PAD).
    // This avoids materialization for the common SHRINK backward → PAD → RESHAPE pattern.

    // Check if strides are contiguous (ignoring mask)
    int src_contig = 1;
    i32 exp = 1;
    for (int i = (int)v.shape.rank - 1; i >= 0; i--) {
      if (v.shape.dims[i] > 1 && v.strides[i] != exp) { src_contig = 0; break; }
      exp *= (i32)v.shape.dims[i];
    }
    if (!src_contig) {
      // Non-contiguous masked view — must materialize
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
    // Contiguous masked view: reshape strides + propagate mask bounds.
    // Flatten the mask to a 1D range [flat_begin, flat_end), then
    // decompose into new shape's mask_begin/mask_end.
    // This works because contiguous layout means flat index = physical index.
    View r = {0};
    r.shape = new_shape; r.numel = new_numel; r.offset = v.offset;
    r.has_mask = 1;
    // Compute contiguous strides for new shape
    for (u32 i = 0; i < new_shape.rank; i++) {
      i32 st = 1;
      for (u32 j = i + 1; j < new_shape.rank; j++) st *= (i32)new_shape.dims[j];
      r.strides[i] = st;
    }
    r.contiguous = 1; // contiguous with mask
    // Propagate mask: for each new dim, find the corresponding old dim's mask
    // Simple case: if old and new dims align (merge/split), propagate bounds.
    // Complex case: fall back to full range (no masking on that dim).
    // For the common pad-then-reshape case: old mask_end on last dims maps
    // to new mask_end on the corresponding flattened/split dims.
    for (u32 i = 0; i < new_shape.rank; i++) {
      r.mask_begin[i] = 0;
      r.mask_end[i] = new_shape.dims[i];
    }
    // Walk old dims and new dims simultaneously (merge-split) to map mask bounds
    u32 oi = 0, ni = 0;
    u32 old_prod = 1, new_prod = 1;
    while (oi < v.shape.rank && ni < new_shape.rank) {
      // Skip size-1 dims
      if (v.shape.dims[oi] == 1) { oi++; continue; }
      if (new_shape.dims[ni] == 1) { ni++; continue; }
      if (old_prod == 1 && new_prod == 1) {
        // Start of a new group: match dims
        old_prod = v.shape.dims[oi];
        new_prod = new_shape.dims[ni];
      }
      if (old_prod == new_prod) {
        // 1:1 mapping — propagate mask directly
        if (old_prod == v.shape.dims[oi] && new_prod == new_shape.dims[ni]) {
          r.mask_begin[ni] = v.mask_begin[oi];
          r.mask_end[ni] = v.mask_end[oi];
        }
        oi++; ni++; old_prod = 1; new_prod = 1;
      } else if (old_prod < new_prod) {
        oi++; if (oi < v.shape.rank) old_prod *= v.shape.dims[oi];
      } else {
        ni++; if (ni < new_shape.rank) new_prod *= new_shape.dims[ni];
      }
    }
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
