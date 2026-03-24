// cpu/pool.c — CPU buffer pool: ID → pointer
// Uses simple array mapping. Buffer 0 is reserved (null).

#define MAX_BUFS 524288

static struct {
    void *bufs[MAX_BUFS];
    u64   sizes[MAX_BUFS];
    u32   count;
} cpu_pool;

static int  cpu_init(void)          { memset(&cpu_pool, 0, sizeof(cpu_pool)); cpu_pool.count = 1; return 0; }
static void cpu_shutdown(void)      { for (u32 i = 1; i < cpu_pool.count; i++) free(cpu_pool.bufs[i]); memset(&cpu_pool, 0, sizeof(cpu_pool)); }
static u32  cpu_buf_alloc(u64 b)    { u32 id = cpu_pool.count++; cpu_pool.bufs[id] = calloc(1, b); cpu_pool.sizes[id] = b; return id; }
static void cpu_buf_free(u32 id)    { free(cpu_pool.bufs[id]); cpu_pool.bufs[id] = NULL; }
static void cpu_buf_write(u32 id, const void *d, u64 b) { memcpy(cpu_pool.bufs[id], d, b); }
static void cpu_buf_read(u32 id, void *o, u64 b)        { memcpy(o, cpu_pool.bufs[id], b); }

static void cpu_pool_reset(u32 keep) {
    u32 buf_keep = keep + 1;
    for (u32 i = buf_keep; i < cpu_pool.count; i++) {
        free(cpu_pool.bufs[i]);
        cpu_pool.bufs[i] = NULL;
    }
    cpu_pool.count = buf_keep;
}
