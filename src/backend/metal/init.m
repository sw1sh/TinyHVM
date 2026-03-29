// metal/init.m — Metal state declarations + init/shutdown
// All shared state is declared here. Sub-files contain function bodies only.

#define MAX_BUFS 524288

// Device state
static id<MTLDevice>       mtl_dev;
static id<MTLCommandQueue> mtl_queue;
static id<MTLLibrary>      mtl_lib;

// Shader pipelines
static id<MTLComputePipelineState> pipe_neg, pipe_relu, pipe_exp, pipe_log, pipe_sqrt;
static id<MTLComputePipelineState> pipe_add, pipe_mul, pipe_sub, pipe_div, pipe_max, pipe_cmp;
static id<MTLComputePipelineState> pipe_mm;
static id<MTLComputePipelineState> pipe_reduce_sum, pipe_reduce_max;
static id<MTLComputePipelineState> pipe_reduce_sum_par, pipe_reduce_max_par;
static id<MTLComputePipelineState> pipe_mul_reduce_sum;
static id<MTLComputePipelineState> pipe_mul_reduce_sum_parallel;
static id<MTLComputePipelineState> pipe_im2col, pipe_col2im;
static id<MTLComputePipelineState> pipe_nhwc_to_nchw, pipe_nchw_to_nhwc;
static id<MTLComputePipelineState> pipe_bias_add, pipe_col_sum;
static id<MTLComputePipelineState> pipe_adam_step, pipe_adam_step_f4;
static id<MTLComputePipelineState> pipe_maxpool2d_fwd, pipe_maxpool2d_bwd;
// Fast contiguous kernels (no ViewParams overhead)
static id<MTLComputePipelineState> fpipe_add, fpipe_sub, fpipe_mul, fpipe_div, fpipe_max, fpipe_cmp;
static id<MTLComputePipelineState> fpipe_neg, fpipe_relu, fpipe_exp, fpipe_log, fpipe_sqrt;
// Float4 vectorized (4x throughput)
static id<MTLComputePipelineState> f4pipe_add, f4pipe_sub, f4pipe_mul;
static id<MTLComputePipelineState> f4pipe_neg, f4pipe_relu;
// 2D broadcast kernels: a[B,C,H,W] op b[1,C,1,1]
static id<MTLComputePipelineState> bc2d_add, bc2d_mul, bc2d_sub, bc2d_div, bc2d_cmp;
static id<MTLComputePipelineState> pipe_relu_bwd;
static id<MTLComputePipelineState> pipe_matrix_transpose, pipe_zero_fill, pipe_pad;

// Buffer pool
static struct {
    id<MTLBuffer> bufs[MAX_BUFS];
    u64           sizes[MAX_BUFS];
    u32           count;
} metal_pool;

// Command batching state
static id<MTLCommandBuffer>         batch_cmd;
static id<MTLComputeCommandEncoder> batch_encoder;
static int                          batch_active;
static int                          batch_dirty;

// JIT capture/replay state (forward declaration — implementation in jit.m)
#define JIT_MAX_CMDS    2048
#define JIT_MAX_SLOTS   2048
#define JIT_MAX_PARAMS  2048

typedef struct {
    id<MTLComputePipelineState> pipe;
    u32 buf_slots[24]; u32 n_bufs;
    u8 params[JIT_MAX_PARAMS]; u32 param_sizes[8]; u32 n_params;
    u32 grid[3]; u32 tg[3]; u8 is_mps;
    u32 mps_m, mps_k, mps_n, mps_dst_slot, mps_a_slot, mps_b_slot;
    u8 mps_trans_a, mps_trans_b;
    u32 mps_phys_a_rows, mps_phys_a_cols, mps_phys_b_rows, mps_phys_b_cols;
    // Blit copy
    u8 is_blit; u32 blit_dst_slot, blit_src_slot; u64 blit_size;
} JITCmd;

typedef struct {
    u64 alloc_size; u32 buf_id; u8 persistent;
} JITSlot;

// Small constant data saved during capture for ephemeral buffers
#define JIT_MAX_CONST 512
#define JIT_CONST_MAX_BYTES 1024  // save buffers up to 1KB
typedef struct { u32 slot; u32 size; u8 data[1024]; } JITConst;

typedef struct {
    JITCmd cmds[JIT_MAX_CMDS]; u32 n_cmds;
    JITSlot slots[JIT_MAX_SLOTS]; u32 n_slots;
    u32 persistent_count;
    enum { JIT_OFF, JIT_CAPTURE, JIT_REPLAY } state;
    u32 loss_slot;
    u8  ephemeral_ready;  // 1 after first replay allocates ephemeral buffers
    // Constant data for ephemeral scalar buffers (CPU-written, no GPU command)
    JITConst consts[JIT_MAX_CONST]; u32 n_consts;
} JITState;

static JITState jit = {0};

// JIT function forward declarations (implementation in jit.m)
static u32  jit_slot_for_buf(u32 buf_id);
static void jit_record_dispatch_ids(id<MTLComputePipelineState>, u32*, u32,
                                      const void**, u64*, u32, u32,u32,u32, u32,u32,u32);
static void jit_record_dispatch_1d(id<MTLComputePipelineState>, id<MTLBuffer>*, u32,
                                     const void**, u64*, u32, u32,u32,u32, u32,u32,u32);
static void jit_record_mps(u32, u32, u32, u32, u32, u32, BOOL, BOOL, u32, u32, u32, u32);

// Free-list for buffer reuse
#define MAX_FREE_BUFS 512
static struct {
    id<MTLBuffer> buf;
    u64           size;
} free_list[MAX_FREE_BUFS];
static u32 free_count = 0;

// Buffer-level refcounting — tracks how many tensors share each buf_id.
// When refcount→0, the buf_id is enqueued to pending_free (not freed immediately
// because GPU may still be reading it). Drained in metal_flush after GPU sync.
static u32 buf_refcount[MAX_BUFS];
#define PENDING_FREE_CAP 4096
static u32 pending_free[PENDING_FREE_CAP];
static u32 pending_free_count = 0;

// ViewParams (must match shaders.metal)
typedef struct {
    int32_t  strides[8];
    uint32_t shape[8];
    int32_t  offset;
    uint32_t rank;
    uint32_t numel;
    uint32_t has_mask;
    uint32_t mask_begin[8];
    uint32_t mask_end[8];
} ViewParams;

static ViewParams view_to_params(const View *v) {
    ViewParams p = {0};
    p.offset = v->offset;
    p.rank   = v->shape.rank;
    p.numel  = v->numel;
    p.has_mask = v->has_mask;
    for (u32 i = 0; i < v->shape.rank; i++) {
        p.strides[i] = v->strides[i];
        p.shape[i]   = v->shape.dims[i];
        p.mask_begin[i] = v->mask_begin[i];
        p.mask_end[i] = v->mask_end[i];
    }
    return p;
}

static id<MTLComputePipelineState> make_pipe(NSString *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [mtl_lib newFunctionWithName:name];
    if (!fn) {
        NSLog(@"TinyHVM Metal: function '%@' not found", name);
        return nil;
    }
    id<MTLComputePipelineState> ps = [mtl_dev newComputePipelineStateWithFunction:fn error:&err];
    if (err) NSLog(@"TinyHVM Metal: pipeline error: %@", err);
    return ps;
}

// Forward declarations (defined in batch.m / pool.m)
static void metal_flush(void);
static void metal_buf_free(u32 id);

static int metal_init(void) {
    thvm_prof_init();
    mtl_dev = MTLCreateSystemDefaultDevice();
    if (!mtl_dev) return -1;
    mtl_queue = [mtl_dev newCommandQueue];

    NSError *err = nil;
    // Check override path (set by embedders like Python FFI)
    NSString *path = nil;
    const char *env_path = getenv("THVM_METALLIB");
    if (env_path && env_path[0]) {
        path = [NSString stringWithUTF8String:env_path];
    }
    if (!path) path = [[NSBundle mainBundle] pathForResource:@"shaders" ofType:@"metallib"];
    if (!path) {
        path = @"shaders.metallib";
    }
    mtl_lib = [mtl_dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (!mtl_lib) {
        NSString *srcPath = @"src/shaders.metal";
        NSString *src = [NSString stringWithContentsOfFile:srcPath
                                                  encoding:NSUTF8StringEncoding error:&err];
        if (src) {
            MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
            mtl_lib = [mtl_dev newLibraryWithSource:src options:opts error:&err];
        }
        if (!mtl_lib) {
            NSLog(@"TinyHVM Metal: cannot load shaders: %@", err);
            return -1;
        }
    }

    pipe_neg  = make_pipe(@"unary_neg");
    pipe_relu = make_pipe(@"unary_relu");
    pipe_exp  = make_pipe(@"unary_exp");
    pipe_log  = make_pipe(@"unary_log");
    pipe_sqrt = make_pipe(@"unary_sqrt");
    pipe_add  = make_pipe(@"binary_add");
    pipe_mul  = make_pipe(@"binary_mul");
    pipe_sub  = make_pipe(@"binary_sub");
    pipe_div  = make_pipe(@"binary_div");
    pipe_max  = make_pipe(@"binary_max");
    pipe_cmp  = make_pipe(@"binary_cmp");
    pipe_mm   = make_pipe(@"matmul_f32");
    pipe_im2col = make_pipe(@"im2col");
    pipe_col2im = make_pipe(@"col2im");
    pipe_nhwc_to_nchw = make_pipe(@"nhwc_to_nchw");
    pipe_nchw_to_nhwc = make_pipe(@"nchw_to_nhwc");
    pipe_bias_add = make_pipe(@"bias_add");
    pipe_col_sum = make_pipe(@"col_sum");
    pipe_adam_step = make_pipe(@"adam_step");
    pipe_adam_step_f4 = make_pipe(@"adam_step_f4");
    pipe_maxpool2d_fwd = make_pipe(@"maxpool2d_fwd");
    pipe_maxpool2d_bwd = make_pipe(@"maxpool2d_bwd");
    pipe_relu_bwd = make_pipe(@"relu_bwd");
    pipe_matrix_transpose = make_pipe(@"matrix_transpose");
    pipe_zero_fill = make_pipe(@"zero_fill");
    pipe_pad = make_pipe(@"pad_kernel");    pipe_reduce_sum = make_pipe(@"reduce_sum");
    pipe_reduce_max = make_pipe(@"reduce_max");
    pipe_mul_reduce_sum = make_pipe(@"mul_reduce_sum");
    pipe_mul_reduce_sum_parallel = make_pipe(@"mul_reduce_sum_parallel");
    pipe_reduce_sum_par = make_pipe(@"reduce_sum_parallel");
    pipe_reduce_max_par = make_pipe(@"reduce_max_parallel");

    // Fast contiguous kernels
    fpipe_add  = make_pipe(@"fast_add");
    fpipe_sub  = make_pipe(@"fast_sub");
    fpipe_mul  = make_pipe(@"fast_mul");
    fpipe_div  = make_pipe(@"fast_div");
    fpipe_max  = make_pipe(@"fast_max");
    fpipe_cmp  = make_pipe(@"fast_cmp");
    fpipe_neg  = make_pipe(@"fast_neg");
    fpipe_relu = make_pipe(@"fast_relu");
    fpipe_exp  = make_pipe(@"fast_exp");
    fpipe_log  = make_pipe(@"fast_log");
    fpipe_sqrt = make_pipe(@"fast_sqrt");
    f4pipe_add  = make_pipe(@"fast4_add");
    f4pipe_sub  = make_pipe(@"fast4_sub");
    f4pipe_mul  = make_pipe(@"fast4_mul");
    f4pipe_neg  = make_pipe(@"fast4_neg");
    f4pipe_relu = make_pipe(@"fast4_relu");
    bc2d_add  = make_pipe(@"add_bc_2d");
    bc2d_mul  = make_pipe(@"mul_bc_2d");
    bc2d_sub  = make_pipe(@"sub_bc_2d");
    bc2d_div  = make_pipe(@"div_bc_2d");
    bc2d_cmp  = make_pipe(@"cmp_bc_2d");

    memset(&metal_pool, 0, sizeof(metal_pool));
    metal_pool.count = 1;

    batch_cmd = nil;
    batch_encoder = nil;
    batch_active = 0;
    batch_dirty = 0;
    return 0;
}

static void metal_shutdown(void) {
    metal_flush();
    for (u32 i = 1; i < metal_pool.count; i++) {
        metal_pool.bufs[i] = nil;
    }
    mtl_lib = nil;
    mtl_queue = nil;
    mtl_dev = nil;
    memset(&metal_pool, 0, sizeof(metal_pool));
}
