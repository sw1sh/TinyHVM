// tensor/view/expand.c — view_expand(): set stride=0 for broadcast dims
static View view_expand(View v, Shape new_shape) {
  if (v.shape.rank != new_shape.rank) {
    fprintf(stderr, "view_expand rank mismatch: v.rank=%u new.rank=%u\n  v=[", v.shape.rank, new_shape.rank);
    for(u32 d=0;d<v.shape.rank;d++) fprintf(stderr,"%u%s",v.shape.dims[d],d+1<v.shape.rank?",":"");
    fprintf(stderr,"]  new=[");
    for(u32 d=0;d<new_shape.rank;d++) fprintf(stderr,"%u%s",new_shape.dims[d],d+1<new_shape.rank?",":"");
    fprintf(stderr,"]\n");
  }
  assert(v.shape.rank == new_shape.rank);
  View r   = v;
  r.numel  = 1;
  for (u32 i = 0; i < new_shape.rank; i++) {
    if (v.shape.dims[i] == 1 && new_shape.dims[i] > 1) {
      r.strides[i] = 0;  // broadcast!
    } else {
      assert(v.shape.dims[i] == new_shape.dims[i]);
    }
    r.shape.dims[i] = new_shape.dims[i];
    r.numel         *= new_shape.dims[i];
  }
  if (v.has_mask) {
    r.has_mask = 1;
    for (u32 i = 0; i < new_shape.rank; i++) {
      r.mask_begin[i] = v.mask_begin[i];
      r.mask_end[i] = (v.shape.dims[i] == 1 && new_shape.dims[i] > 1) ? new_shape.dims[i] : v.mask_end[i];
    }
  }
  r.contiguous = 0;
  return r;
}
