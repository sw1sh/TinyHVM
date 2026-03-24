// heap/alloc.c — heap_alloc(): bump-pointer allocator
static inline u64 heap_alloc(TinyHVM *ctx, u64 w) {
  u64 l = ctx->heap_pos;
  ctx->heap_pos += w;
  assert(ctx->heap_pos < HEAP_CAP);
  return l;
}
