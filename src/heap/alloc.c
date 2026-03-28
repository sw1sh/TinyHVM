// heap/alloc.c — heap_alloc(): bump-pointer allocator
static inline u64 heap_alloc(TinyHVM *ctx, u64 w) {
  u64 l = ctx->heap_pos;
  ctx->heap_pos += w;
  if (ctx->heap_pos >= HEAP_CAP) { fprintf(stderr, "HEAP_OVERFLOW: pos=%llu cap=%llu\n", (unsigned long long)ctx->heap_pos, (unsigned long long)HEAP_CAP); fflush(stderr); assert(0); }
  return l;
}
