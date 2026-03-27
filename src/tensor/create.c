// tensor/create.c — tensor_create() / tensor_view_of()

// Allocate a new tensor slot with a fresh GPU/CPU buffer.
static u32 tensor_create(TinyHVM *ctx, Shape s, u32 dtype) {
  assert(s.rank <= MAX_DIM && ctx->tensor_count < MAX_TENSORS);
  u32         id = ctx->tensor_count++;
  TensorMeta *m  = &ctx->tensors[id];
  memset(m, 0, sizeof(*m));
  m->dtype   = dtype;
  m->refcount = 1;
  m->last_use_loc = (u64)-1;
  m->view    = view_create(s);
  if (ctx->backend) {
    u64 bytes = (u64)m->view.numel * dtype_size(dtype);
    m->buf_id = ctx->backend->buf_alloc(bytes);
  }
  thvm_prof_tensor_created(0);
  thvm_prof_update_watermarks(ctx->tensor_count, ctx->heap_pos);
  return id;
}

// Create a view alias: shares the buffer, but has a different View.
static u32 tensor_view_of(TinyHVM *ctx, u32 src_id, View new_view) {
  assert(ctx->tensor_count < MAX_TENSORS);
  u32         id  = ctx->tensor_count++;
  TensorMeta *m   = &ctx->tensors[id];
  TensorMeta *ms  = &ctx->tensors[src_id];
  memset(m, 0, sizeof(*m));
  m->dtype    = ms->dtype;
  m->refcount = 1;
  m->last_use_loc = (u64)-1;
  m->buf_id   = ms->buf_id;
  m->view     = new_view;
  return id;
}

// Refcount helpers for inet GC
static inline void tensor_incref(TinyHVM *ctx, u32 id) {
    ctx->tensors[id].refcount++;
}

static inline void tensor_decref(TinyHVM *ctx, u32 id) {
    TensorMeta *m = &ctx->tensors[id];
    if (m->refcount > 0) m->refcount--;
    // Buffer freeing deferred to thvm_reset — view-shared bufs make eager free unsafe.
    (void)ctx;
}
