// heap/read.c — heap_read(): read a term from the heap
static inline Term heap_read(TinyHVM *ctx, u64 l) { return ctx->heap[l]; }
