// tensor/create.c — tensor_create() / tensor_view_of()

// Allocate a new tensor slot with a fresh GPU/CPU buffer.
// When n_threads > 1: per-thread tensor ID range (zero contention).
static u32 tensor_create(TinyHVM *ctx, Shape s, u32 dtype) {
  u32 id;
  if (ctx->n_threads > 1) {
    ThvmThread *ts = &ctx->threads[tl_thread_id];
    id = ts->tensor_next++;
    assert(id < ts->tensor_end && "thread tensor ID overflow");
  } else {
    if (ctx->tensor_count >= MAX_TENSORS - 64) {
        fprintf(stderr, "FATAL: Tensor pool near limit (%u/%u). Aborting safely.\n", ctx->tensor_count, MAX_TENSORS);
        exit(1);
    }
    assert(s.rank <= MAX_DIM);
    id = ctx->tensor_count++;
  }
  TensorMeta *m  = &ctx->tensors[id];
  memset(m, 0, sizeof(*m));
  m->dtype   = dtype;
  m->refcount = 1;
  m->view    = view_create(s);
  m->backend = ctx_default_backend(ctx);
  if (m->backend) {
    u64 bytes = (u64)m->view.numel * dtype_size(dtype);
    m->buf_id = m->backend->buf_alloc(bytes);
  }
  thvm_prof_tensor_created(0);
  thvm_prof_update_watermarks(ctx->tensor_count, ctx->heap_pos);
  return id;
}

// Create a view alias: shares the buffer, but has a different View.
static u32 tensor_view_of(TinyHVM *ctx, u32 src_id, View new_view) {
  u32 id;
  if (ctx->n_threads > 1) {
    ThvmThread *ts = &ctx->threads[tl_thread_id];
    id = ts->tensor_next++;
    assert(id < ts->tensor_end && "thread tensor ID overflow");
  } else {
    assert(ctx->tensor_count < MAX_TENSORS);
    id = ctx->tensor_count++;
  }
  TensorMeta *m   = &ctx->tensors[id];
  TensorMeta *ms  = &ctx->tensors[src_id];
  memset(m, 0, sizeof(*m));
  m->dtype    = ms->dtype;
  m->refcount = 1;
  m->buf_id   = ms->buf_id;
  m->backend  = ms->backend;
  m->view     = new_view;
  if (m->buf_id && m->backend && m->backend->buf_incref)
      m->backend->buf_incref(m->buf_id);
  return id;
}

// Refcount helpers for inet GC.
// When n_threads > 1: use atomic operations for safe cross-thread sharing.
static inline void tensor_incref(TinyHVM *ctx, u32 id) {
    if (ctx->n_threads > 1)
        atomic_fetch_add((_Atomic(u32)*)&ctx->tensors[id].refcount, 1);
    else
        ctx->tensors[id].refcount++;
}

static inline void tensor_decref(TinyHVM *ctx, u32 id) {
    if (ctx->n_threads > 1)
        atomic_fetch_sub((_Atomic(u32)*)&ctx->tensors[id].refcount, 1);
    else if (ctx->tensors[id].refcount > 0)
        ctx->tensors[id].refcount--;
    // Buffer freeing uses buf_refcount infrastructure but is NOT triggered here.
    // Tensor metadata (buf_id, src_ids) is still accessed by ASSIGN/GRAD/materialize
    // handlers after inet refcount reaches 0. Buffer release happens in thvm_reset
    // via pool_reset (bulk) or via tensor_release at ERA-absorption safe points.
}

// ERA-safe release: decref tensor AND release buffer when refcount reaches 0.
// ONLY safe when the tensor is genuinely dead — no backward walk, no ASSIGN,
// no src_ids traversal will reach it. This is true for ERA-discarded tensors:
// when ERA absorbs a tensor in a dead gradient branch, the gradient signal is
// dead and no GRAD handler will ever traverse this tensor's metadata.
static inline void tensor_release(TinyHVM *ctx, u32 id) {
    TensorMeta *m = &ctx->tensors[id];
    u32 prev;
    if (ctx->n_threads > 1)
        prev = atomic_fetch_sub((_Atomic(u32)*)&m->refcount, 1);
    else {
        prev = m->refcount;
        if (prev > 0) m->refcount = prev - 1;
    }
    if (prev == 1 && m->buf_id && m->backend && m->backend->buf_decref)
        m->backend->buf_decref(m->buf_id);
}
