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
static id<MTLComputePipelineState> pipe_mul_reduce_sum;
static id<MTLComputePipelineState> pipe_im2col, pipe_col2im;
static id<MTLComputePipelineState> pipe_nhwc_to_nchw, pipe_nchw_to_nhwc;
static id<MTLComputePipelineState> pipe_bias_add, pipe_col_sum;
static id<MTLComputePipelineState> pipe_adam_step;
static id<MTLComputePipelineState> pipe_maxpool2d_fwd, pipe_maxpool2d_bwd;
// Fast contiguous kernels (no ViewParams overhead)
static id<MTLComputePipelineState> fpipe_add, fpipe_sub, fpipe_mul, fpipe_div, fpipe_max, fpipe_cmp;
static id<MTLComputePipelineState> fpipe_neg, fpipe_relu, fpipe_exp, fpipe_log, fpipe_sqrt;
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

// Free-list for buffer reuse
#define MAX_FREE_BUFS 512
static struct {
    id<MTLBuffer> buf;
    u64           size;
} free_list[MAX_FREE_BUFS];
static u32 free_count = 0;

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

// Forward declaration (defined in batch.m)
static void metal_flush(void);

static int metal_init(void) {
    thvm_prof_init();
    mtl_dev = MTLCreateSystemDefaultDevice();
    if (!mtl_dev) return -1;
    mtl_queue = [mtl_dev newCommandQueue];

    NSError *err = nil;
    NSString *path = [[NSBundle mainBundle] pathForResource:@"shaders" ofType:@"metallib"];
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
    pipe_maxpool2d_fwd = make_pipe(@"maxpool2d_fwd");
    pipe_maxpool2d_bwd = make_pipe(@"maxpool2d_bwd");
    pipe_relu_bwd = make_pipe(@"relu_bwd");
    pipe_matrix_transpose = make_pipe(@"matrix_transpose");
    pipe_zero_fill = make_pipe(@"zero_fill");
    pipe_pad = make_pipe(@"pad_kernel");    pipe_reduce_sum = make_pipe(@"reduce_sum");
    pipe_reduce_max = make_pipe(@"reduce_max");
    pipe_mul_reduce_sum = make_pipe(@"mul_reduce_sum");

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
