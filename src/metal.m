// metal.m — Metal backend for TinyHVM
// Pre-built compute shaders + MPS matmul.
// Matches Backend vtable exactly.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "tinyhvm.h"
#include <stdlib.h>
#include <string.h>

// ============================================================
// Metal state
// ============================================================

#define MAX_BUFS 8192

static id<MTLDevice>       mtl_dev;
static id<MTLCommandQueue> mtl_queue;
static id<MTLLibrary>      mtl_lib;

// Shader pipelines
static id<MTLComputePipelineState> pipe_neg, pipe_relu, pipe_exp, pipe_log;
static id<MTLComputePipelineState> pipe_add, pipe_mul, pipe_sub, pipe_div, pipe_max, pipe_cmp;
static id<MTLComputePipelineState> pipe_mm;
static id<MTLComputePipelineState> pipe_reduce_sum, pipe_reduce_max;

// Buffer pool
static struct {
    id<MTLBuffer> bufs[MAX_BUFS];
    u64           sizes[MAX_BUFS];
    u32           count;
} metal_pool;

// ViewParams must match the struct in shaders.metal
typedef struct {
    int32_t  strides[8];
    uint32_t shape[8];
    int32_t  offset;
    uint32_t rank;
    uint32_t numel;
} ViewParams;

static ViewParams view_to_params(const View *v) {
    ViewParams p = {0};
    p.offset = v->offset;
    p.rank   = v->shape.rank;
    p.numel  = v->numel;
    for (u32 i = 0; i < v->shape.rank; i++) {
        p.strides[i] = v->strides[i];
        p.shape[i]   = v->shape.dims[i];
    }
    return p;
}

// ============================================================
// Pipeline helpers
// ============================================================

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

// ============================================================
// Backend vtable implementation
// ============================================================

static int metal_init(void) {
    mtl_dev = MTLCreateSystemDefaultDevice();
    if (!mtl_dev) return -1;
    mtl_queue = [mtl_dev newCommandQueue];

    // Load compiled metallib from same directory as executable
    NSError *err = nil;
    NSString *path = [[NSBundle mainBundle] pathForResource:@"shaders" ofType:@"metallib"];
    if (!path) {
        // Try current directory
        path = @"shaders.metallib";
    }
    mtl_lib = [mtl_dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (!mtl_lib) {
        // Fall back: compile from source at runtime
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

    // Build pipelines
    pipe_neg  = make_pipe(@"unary_neg");
    pipe_relu = make_pipe(@"unary_relu");
    pipe_exp  = make_pipe(@"unary_exp");
    pipe_log  = make_pipe(@"unary_log");
    pipe_add  = make_pipe(@"binary_add");
    pipe_mul  = make_pipe(@"binary_mul");
    pipe_sub  = make_pipe(@"binary_sub");
    pipe_div  = make_pipe(@"binary_div");
    pipe_max  = make_pipe(@"binary_max");
    pipe_cmp  = make_pipe(@"binary_cmp");
    pipe_mm   = make_pipe(@"matmul_f32");
    pipe_reduce_sum = make_pipe(@"reduce_sum");
    pipe_reduce_max = make_pipe(@"reduce_max");

    memset(&metal_pool, 0, sizeof(metal_pool));
    metal_pool.count = 1;  // 0 reserved
    return 0;
}

static void metal_shutdown(void) {
    for (u32 i = 1; i < metal_pool.count; i++) {
        metal_pool.bufs[i] = nil;
    }
    mtl_lib = nil;
    mtl_queue = nil;
    mtl_dev = nil;
    memset(&metal_pool, 0, sizeof(metal_pool));
}

static u32 metal_buf_alloc(u64 bytes) {
    u32 id = metal_pool.count++;
    assert(id < MAX_BUFS);
    metal_pool.bufs[id] = [mtl_dev newBufferWithLength:MAX(bytes, 4)
                                               options:MTLResourceStorageModeShared];
    metal_pool.sizes[id] = bytes;
    return id;
}

static void metal_buf_free(u32 id) {
    metal_pool.bufs[id] = nil;
    metal_pool.sizes[id] = 0;
}

static void metal_buf_write(u32 id, const void *data, u64 bytes) {
    memcpy(metal_pool.bufs[id].contents, data, bytes);
}

static void metal_buf_read(u32 id, void *out, u64 bytes) {
    memcpy(out, metal_pool.bufs[id].contents, bytes);
}

// ============================================================
// Dispatch helpers
// ============================================================

static void dispatch_1d(id<MTLComputePipelineState> pipe,
                        id<MTLBuffer> *bufs, u32 n_bufs,
                        const void **params, u64 *param_sizes, u32 n_params,
                        u32 numel) {
    id<MTLCommandBuffer> cmd = [mtl_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipe];

    for (u32 i = 0; i < n_bufs; i++)
        [enc setBuffer:bufs[i] offset:0 atIndex:i];
    for (u32 i = 0; i < n_params; i++)
        [enc setBytes:params[i] length:param_sizes[i] atIndex:n_bufs + i];

    NSUInteger tpg = MIN(pipe.maxTotalThreadsPerThreadgroup, (NSUInteger)numel);
    [enc dispatchThreads:MTLSizeMake(numel, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

// ============================================================
// Unary op dispatch
// ============================================================

static void metal_op_unary(u32 uop, u32 dst, const View *dv,
                            u32 src, const View *sv) {
    id<MTLComputePipelineState> pipe = nil;
    switch (uop) {
        case UOP_NEG:  pipe = pipe_neg;  break;
        case UOP_RELU: pipe = pipe_relu; break;
        case UOP_EXP:  pipe = pipe_exp;  break;
        case UOP_LOG:  pipe = pipe_log;  break;
        default: return;
    }

    ViewParams dvp = view_to_params(dv);
    ViewParams svp = view_to_params(sv);

    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &dvp, &svp };
    u64 psizes[] = { sizeof(ViewParams), sizeof(ViewParams) };
    dispatch_1d(pipe, bufs, 2, params, psizes, 2, dv->numel);
}

// ============================================================
// Binary op dispatch
// ============================================================

static void metal_op_binary(u32 uop, u32 dst, const View *dv,
                             u32 a, const View *av, u32 b, const View *bv) {
    id<MTLComputePipelineState> pipe = nil;
    switch (uop) {
        case UOP_ADD: pipe = pipe_add; break;
        case UOP_MUL: pipe = pipe_mul; break;
        case UOP_SUB: pipe = pipe_sub; break;
        case UOP_DIV: pipe = pipe_div; break;
        case UOP_MAX: pipe = pipe_max; break;
        case UOP_CMP: pipe = pipe_cmp; break;
        default: return;
    }

    ViewParams dvp = view_to_params(dv);
    ViewParams avp = view_to_params(av);
    ViewParams bvp = view_to_params(bv);

    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[a], metal_pool.bufs[b] };
    const void *params[] = { &dvp, &avp, &bvp };
    u64 psizes[] = { sizeof(ViewParams), sizeof(ViewParams), sizeof(ViewParams) };
    dispatch_1d(pipe, bufs, 3, params, psizes, 3, dv->numel);
}

// ============================================================
// Matmul — MPS fast path, compute shader fallback
// ============================================================

static void metal_op_mm(u32 dst, u32 a, const View *av, u32 b, const View *bv,
                         u32 M, u32 K, u32 N) {
    (void)av; (void)bv;

    // MPS path: float matmul
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M columns:K rowBytes:K*sizeof(float)
        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:K columns:N rowBytes:N*sizeof(float)
        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M columns:N rowBytes:N*sizeof(float)
        dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[a] descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[b] descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[dst] descriptor:descC];

    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:mtl_dev transposeLeft:NO transposeRight:NO
        resultRows:M resultColumns:N interiorColumns:K
        alpha:1.0 beta:0.0];

    id<MTLCommandBuffer> cmd = [mtl_queue commandBuffer];
    [mm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [cmd commit];
    [cmd waitUntilCompleted];
}

// ============================================================
// Reduce op dispatch
// ============================================================

static void metal_op_reduce(u32 uop, u32 dst, u32 dst_numel,
                             u32 src, u32 src_numel, u32 reduce_dim) {
    (void)src_numel;
    id<MTLComputePipelineState> pipe = nil;
    switch (uop) {
        case UOP_SUM:  pipe = pipe_reduce_sum; break;
        case UOP_RMAX: pipe = pipe_reduce_max; break;
        default: return;
    }

    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &reduce_dim };
    u64 psizes[] = { sizeof(u32) };
    dispatch_1d(pipe, bufs, 2, params, psizes, 1, dst_numel);
}

static void metal_pool_reset(u32 keep) {
    u32 buf_keep = keep + 1;
    for (u32 i = buf_keep; i < metal_pool.count; i++) {
        metal_pool.bufs[i] = nil;
        metal_pool.sizes[i] = 0;
    }
    metal_pool.count = buf_keep;
}

// ============================================================
// Export
// ============================================================

Backend metal_backend = {
    .init      = metal_init,
    .shutdown  = metal_shutdown,
    .buf_alloc = metal_buf_alloc,
    .buf_free  = metal_buf_free,
    .buf_write = metal_buf_write,
    .buf_read  = metal_buf_read,
    .op_unary  = metal_op_unary,
    .op_binary = metal_op_binary,
    .op_mm     = metal_op_mm,
    .op_reduce = metal_op_reduce,
    .pool_reset = metal_pool_reset,
};
