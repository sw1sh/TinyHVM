// metal.m — Metal backend for TinyHVM
// Pre-built compute shaders + MPS matmul.
// Command buffer batching for performance.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "tinyhvm.h"
#include <stdlib.h>
#include <string.h>

// ============================================================
// Metal state
// ============================================================

#define MAX_BUFS 16384

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
static id<MTLComputePipelineState> pipe_relu_bwd;
static id<MTLComputePipelineState> pipe_matrix_transpose, pipe_zero_fill;

// Buffer pool
static struct {
    id<MTLBuffer> bufs[MAX_BUFS];
    u64           sizes[MAX_BUFS];
    u32           count;
} metal_pool;

// ============================================================
// Command buffer batching
// ============================================================
// When batching is active, compute dispatches accumulate into a single
// command buffer instead of each creating + committing their own.
// This eliminates ~50 GPU syncs per training step.

static id<MTLCommandBuffer>         batch_cmd;      // active batch command buffer
static id<MTLComputeCommandEncoder> batch_encoder;   // shared compute encoder
static int                          batch_active;    // 1 = batching on
static int                          batch_dirty;     // 1 = encoder has pending work

// Flush (commit + wait) any pending compute work
static void metal_flush(void) {
    if (batch_encoder) {
        [batch_encoder endEncoding];
        batch_encoder = nil;
    }
    if (batch_cmd) {
        [batch_cmd commit];
        [batch_cmd waitUntilCompleted];
        batch_cmd = nil;
    }
    batch_dirty = 0;
}

// Get or create the shared compute encoder
static id<MTLComputeCommandEncoder> get_encoder(void) {
    if (!batch_cmd) {
        batch_cmd = [mtl_queue commandBuffer];
    }
    if (!batch_encoder) {
        batch_encoder = [batch_cmd computeCommandEncoder];
    }
    return batch_encoder;
}

static void metal_begin_batch(void) {
    batch_active = 1;
}

static void metal_end_batch(void) {
    if (batch_dirty) metal_flush();
    batch_active = 0;
}

// ============================================================
// ViewParams (must match shaders.metal)
// ============================================================

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
    thvm_prof_init();
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
    pipe_reduce_sum = make_pipe(@"reduce_sum");
    pipe_reduce_max = make_pipe(@"reduce_max");
    pipe_mul_reduce_sum = make_pipe(@"mul_reduce_sum");

    memset(&metal_pool, 0, sizeof(metal_pool));
    metal_pool.count = 1;  // 0 reserved

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

static u32 metal_buf_alloc(u64 bytes) {
    u32 id = metal_pool.count++;
    assert(id < MAX_BUFS);
    metal_pool.bufs[id] = [mtl_dev newBufferWithLength:MAX(bytes, 4)
                                               options:MTLResourceStorageModeShared];
    metal_pool.sizes[id] = bytes;
    thvm_prof_buf_alloc(bytes);
    return id;
}

static void metal_buf_free(u32 id) {
    metal_pool.bufs[id] = nil;
    metal_pool.sizes[id] = 0;
}

static void metal_buf_write(u32 id, const void *data, u64 bytes) {
    // Must flush pending GPU work before CPU writes to shared buffer
    if (batch_dirty) metal_flush();
    memcpy(metal_pool.bufs[id].contents, data, bytes);
    thvm_prof_buf_write(bytes);
}

static void metal_buf_read(u32 id, void *out, u64 bytes) {
    // Must flush pending GPU work before CPU reads from shared buffer
    if (batch_dirty) metal_flush();
    memcpy(out, metal_pool.bufs[id].contents, bytes);
    thvm_prof_buf_read(bytes);
}

// ============================================================
// Dispatch helpers (batched)
// ============================================================

static void dispatch_1d(id<MTLComputePipelineState> pipe,
                        id<MTLBuffer> *bufs, u32 n_bufs,
                        const void **params, u64 *param_sizes, u32 n_params,
                        u32 numel) {
    if (batch_active) {
        // Append to shared encoder
        id<MTLComputeCommandEncoder> enc = get_encoder();
        [enc setComputePipelineState:pipe];
        for (u32 i = 0; i < n_bufs; i++)
            [enc setBuffer:bufs[i] offset:0 atIndex:i];
        for (u32 i = 0; i < n_params; i++)
            [enc setBytes:params[i] length:param_sizes[i] atIndex:n_bufs + i];
        NSUInteger tpg = MIN(pipe.maxTotalThreadsPerThreadgroup, (NSUInteger)numel);
        [enc dispatchThreads:MTLSizeMake(numel, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        batch_dirty = 1;
    } else {
        // Immediate: own command buffer
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
}

// ============================================================
// Unary op dispatch
// ============================================================

static void metal_op_unary(u32 uop, u32 dst, const View *dv,
                            u32 src, const View *sv) {
    u64 t0 = thvm_prof_tick();
    id<MTLComputePipelineState> pipe = nil;
    switch (uop) {
        case UOP_NEG:  pipe = pipe_neg;  break;
        case UOP_RELU: pipe = pipe_relu; break;
        case UOP_EXP:  pipe = pipe_exp;  break;
        case UOP_LOG:  pipe = pipe_log;  break;
        case UOP_SQRT: pipe = pipe_sqrt; break;
        default: return;
    }

    ViewParams dvp = view_to_params(dv);
    ViewParams svp = view_to_params(sv);

    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &dvp, &svp };
    u64 psizes[] = { sizeof(ViewParams), sizeof(ViewParams) };
    dispatch_1d(pipe, bufs, 2, params, psizes, 2, dv->numel);
    thvm_prof_record(uop, t0);
}

// ============================================================
// Binary op dispatch
// ============================================================

static void metal_op_binary(u32 uop, u32 dst, const View *dv,
                             u32 a, const View *av, u32 b, const View *bv) {
    u64 t0 = thvm_prof_tick();
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
    thvm_prof_record(uop, t0);
}

// ============================================================
// Matmul — MPS (requires its own command buffer)
// ============================================================

static void metal_op_mm(u32 dst, u32 a, const View *av, u32 b, const View *bv,
                         u32 M, u32 K, u32 N) {
    u64 t0 = thvm_prof_tick();
    // MPS needs its own command buffer — flush any pending compute work first
    if (batch_dirty) metal_flush();

    // Materialize non-contiguous inputs to contiguous temp buffers for MPS
    id<MTLBuffer> buf_a = metal_pool.bufs[a];
    id<MTLBuffer> buf_b = metal_pool.bufs[b];
    id<MTLBuffer> tmp_a = nil, tmp_b = nil;

    if (!av->contiguous) {
        u32 n = M * K;
        tmp_a = [mtl_dev newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
        float *src = (float *)buf_a.contents;
        float *dst_ptr = (float *)tmp_a.contents;
        for (u32 i = 0; i < n; i++) {
            u32 idx = (u32)av->offset, rem = i;
            for (i32 d = (i32)av->shape.rank - 1; d >= 0; d--) {
                u32 coord = rem % av->shape.dims[d];
                rem /= av->shape.dims[d];
                idx += coord * (u32)av->strides[d];
            }
            dst_ptr[i] = src[idx];
        }
        buf_a = tmp_a;
    }

    if (!bv->contiguous) {
        u32 n = K * N;
        tmp_b = [mtl_dev newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
        float *src = (float *)buf_b.contents;
        float *dst_ptr = (float *)tmp_b.contents;
        for (u32 i = 0; i < n; i++) {
            u32 idx = (u32)bv->offset, rem = i;
            for (i32 d = (i32)bv->shape.rank - 1; d >= 0; d--) {
                u32 coord = rem % bv->shape.dims[d];
                rem /= bv->shape.dims[d];
                idx += coord * (u32)bv->strides[d];
            }
            dst_ptr[i] = src[idx];
        }
        buf_b = tmp_b;
    }

    // MPS matmul
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M columns:K rowBytes:K*sizeof(float)
        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:K columns:N rowBytes:N*sizeof(float)
        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M columns:N rowBytes:N*sizeof(float)
        dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:buf_a descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:buf_b descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[dst] descriptor:descC];

    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:mtl_dev transposeLeft:NO transposeRight:NO
        resultRows:M resultColumns:N interiorColumns:K
        alpha:1.0 beta:0.0];

    id<MTLCommandBuffer> cmd = [mtl_queue commandBuffer];
    [mm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [cmd commit];

    // If batching, we DON'T wait — the next batched compute dispatch will create
    // a new command buffer which will be serialized after this one by the queue.
    // If not batching, we must wait.
    if (!batch_active) {
        [cmd waitUntilCompleted];
    } else {
        [cmd waitUntilCompleted];  // TODO: can remove this with proper dependency tracking
    }

    // Release temporary contiguous copies (prevents GPU memory leak)
    tmp_a = nil;
    tmp_b = nil;

    thvm_prof_record(UOP_MM, t0);
}

// ============================================================
// Reduce op dispatch
// ============================================================

static void metal_op_reduce(u32 uop, u32 dst, u32 dst_numel,
                             u32 src, u32 src_numel, u32 reduce_dim) {
    u64 t0 = thvm_prof_tick();
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
    thvm_prof_record(uop, t0);
}

// ============================================================
// Fused MUL+SUM — called from thvm_reduce pattern match
// ============================================================

typedef struct {
    uint32_t reduce_dim;
    uint32_t reduce_stride_a;
    uint32_t reduce_stride_b;
} MulReduceParams;

void metal_mul_reduce_sum(u32 dst, u32 dst_numel,
                          u32 a_buf, const View *av,
                          u32 b_buf, const View *bv,
                          const View *ov,
                          u32 reduce_dim,
                          u32 reduce_stride_a,
                          u32 reduce_stride_b) {
    u64 t0 = thvm_prof_tick();
    ViewParams avp = view_to_params(av);
    ViewParams bvp = view_to_params(bv);
    ViewParams ovp = view_to_params(ov);
    MulReduceParams rp = {reduce_dim, reduce_stride_a, reduce_stride_b};

    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[a_buf], metal_pool.bufs[b_buf] };
    const void *params[] = { &avp, &bvp, &ovp, &rp };
    u64 psizes[] = { sizeof(ViewParams), sizeof(ViewParams), sizeof(ViewParams), sizeof(MulReduceParams) };
    dispatch_1d(pipe_mul_reduce_sum, bufs, 3, params, psizes, 4, dst_numel);
    thvm_prof_record(UOP_SUM, t0);
}

static void metal_pool_reset(u32 keep) {
    if (batch_dirty) metal_flush();
    u32 buf_keep = keep + 1;
    for (u32 i = buf_keep; i < metal_pool.count; i++) {
        thvm_prof_buf_free(metal_pool.sizes[i]);
        metal_pool.bufs[i] = nil;
        metal_pool.sizes[i] = 0;
    }
    metal_pool.count = buf_keep;
}

// ============================================================
// CNN backend ops — device-agnostic interface for layers.c
// ============================================================

static void metal_op_im2col(u32 dst, u32 src, Conv2dParams p) {
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &p };
    u64 psizes[] = { sizeof(Conv2dParams) };
    dispatch_1d(pipe_im2col, bufs, 2, params, psizes, 1, p.n_patches * p.patch_size);
}

static void metal_op_col2im(u32 dst, u32 src, Conv2dParams p) {
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &p };
    u64 psizes[] = { sizeof(Conv2dParams) };
    dispatch_1d(pipe_col2im, bufs, 2, params, psizes, 1, p.B * p.Cin * p.H * p.W);
}

static void metal_op_nhwc_to_nchw(u32 dst, u32 src, u32 B, u32 C, u32 H, u32 W) {
    LayoutParams lp = {B, C, H, W};
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &lp };
    u64 psizes[] = { sizeof(LayoutParams) };
    dispatch_1d(pipe_nhwc_to_nchw, bufs, 2, params, psizes, 1, B * C * H * W);
}

static void metal_op_nchw_to_nhwc(u32 dst, u32 src, u32 B, u32 C, u32 H, u32 W) {
    LayoutParams lp = {B, C, H, W};
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &lp };
    u64 psizes[] = { sizeof(LayoutParams) };
    dispatch_1d(pipe_nchw_to_nhwc, bufs, 2, params, psizes, 1, B * C * H * W);
}

static void metal_op_bias_add(u32 buf, u32 bias, u32 C, u32 n) {
    id<MTLBuffer> bufs[] = { metal_pool.bufs[buf], metal_pool.bufs[bias] };
    const void *params[] = { &C };
    u64 psizes[] = { sizeof(u32) };
    dispatch_1d(pipe_bias_add, bufs, 2, params, psizes, 1, n);
}

static void metal_op_col_sum(u32 dst, u32 src, u32 N, u32 C) {
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &N, &C };
    u64 psizes[] = { sizeof(u32), sizeof(u32) };
    dispatch_1d(pipe_col_sum, bufs, 2, params, psizes, 2, C);
}

static void metal_op_transpose(u32 dst, u32 src, u32 M, u32 N) {
    typedef struct { u32 M, N; } TP;
    TP tp = {M, N};
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dst], metal_pool.bufs[src] };
    const void *params[] = { &tp };
    u64 psizes[] = { sizeof(TP) };
    dispatch_1d(pipe_matrix_transpose, bufs, 2, params, psizes, 1, M * N);
}

static void metal_op_maxpool_fwd(u32 out, u32 mask, u32 src, u32 B, u32 C, u32 H, u32 W) {
    u32 OH = H / 2, OW = W / 2;
    LayoutParams lp = {B, C, H, W};
    id<MTLBuffer> bufs[] = { metal_pool.bufs[out], metal_pool.bufs[mask], metal_pool.bufs[src] };
    const void *params[] = { &lp };
    u64 psizes[] = { sizeof(LayoutParams) };
    dispatch_1d(pipe_maxpool2d_fwd, bufs, 3, params, psizes, 1, B * C * OH * OW);
}

static void metal_op_maxpool_bwd(u32 dx, u32 dout, u32 mask, u32 B, u32 C, u32 H, u32 W) {
    u32 OH = H / 2, OW = W / 2;
    LayoutParams lp = {B, C, H, W};
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dx], metal_pool.bufs[dout], metal_pool.bufs[mask] };
    const void *params[] = { &lp };
    u64 psizes[] = { sizeof(LayoutParams) };
    dispatch_1d(pipe_maxpool2d_bwd, bufs, 3, params, psizes, 1, B * C * OH * OW);
}

static void metal_op_relu_bwd(u32 dx, u32 dout, u32 x, u32 n) {
    id<MTLBuffer> bufs[] = { metal_pool.bufs[dx], metal_pool.bufs[dout], metal_pool.bufs[x] };
    dispatch_1d(pipe_relu_bwd, bufs, 3, NULL, NULL, 0, n);
}

static void metal_op_zero_fill(u32 buf, u32 n) {
    id<MTLBuffer> bufs_[] = { metal_pool.bufs[buf] };
    dispatch_1d(pipe_zero_fill, bufs_, 1, NULL, NULL, 0, n);
}

static void metal_op_adam_step(u32 param, u32 grad, u32 m, u32 v,
                                f32 lr, f32 beta1, f32 beta2, f32 eps,
                                f32 bc1, f32 bc2, u32 n) {
    typedef struct { f32 lr, beta1, beta2, eps, bc1, bc2; } AP;
    AP ap = {lr, beta1, beta2, eps, bc1, bc2};
    id<MTLBuffer> bufs[] = { metal_pool.bufs[param], metal_pool.bufs[grad],
                              metal_pool.bufs[m], metal_pool.bufs[v] };
    const void *params[] = { &ap };
    u64 psizes[] = { sizeof(AP) };
    dispatch_1d(pipe_adam_step, bufs, 4, params, psizes, 1, n);
}

// ============================================================
// Fused kernel codegen — runtime Metal shader generation
// ============================================================
// See resources/ic_fusion.md
// Generates MSL source from FuseState, compiles once, caches by op-chain hash.

#define FUSED_CACHE_SIZE 8

static struct {
    u64 key;
    id<MTLComputePipelineState> pipe;
} fused_cache[FUSED_CACHE_SIZE];
static u32 fused_cache_count = 0;

static u64 fuse_hash(const u32 *ops, u32 n_ops, int has_reduce) {
    u64 h = 0xcbf29ce484222325ULL;
    for (u32 i = 0; i < n_ops; i++) {
        h ^= ops[i];
        h *= 0x100000001b3ULL;
    }
    h ^= (u64)has_reduce;
    return h;
}

static NSString *metal_codegen_fused(const u32 *ops, u32 n_ops, u32 n_inputs,
                                      int has_reduce) {
    NSMutableString *src = [NSMutableString stringWithCapacity:2048];
    [src appendString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
    [src appendString:@"kernel void fused_kernel(\n"];
    [src appendString:@"  device float *out [[buffer(0)]],\n"];
    [src appendString:@"  device const float *in0 [[buffer(1)]],\n"];
    for (u32 i = 1; i < n_inputs; i++)
        [src appendFormat:@"  device const float *in%u [[buffer(%u)]],\n", i, i + 1];
    [src appendFormat:@"  constant uint &reduce_dim [[buffer(%u)]],\n", n_inputs + 1];
    [src appendFormat:@"  constant uint &total_numel [[buffer(%u)]],\n", n_inputs + 2];
    [src appendString:@"  uint gid [[thread_position_in_grid]])\n{\n"];

    if (has_reduce) {
        [src appendString:@"  uint out_count = total_numel / reduce_dim;\n"];
        [src appendString:@"  if (gid >= out_count) return;\n"];
        int is_max = (ops[0] == UOP_RMAX);
        [src appendString:@"  float acc = 0.0;\n"];
        [src appendString:@"  for (uint j = 0; j < reduce_dim; j++) {\n"];
        [src appendString:@"    uint idx = gid * reduce_dim + j;\n"];
        [src appendString:@"    float v = in0[idx];\n"];
        for (int k = (int)n_ops - 1; k >= 0; k--) {
            switch (ops[k]) {
                case UOP_RELU: [src appendString:@"    v = max(v, 0.0f);\n"]; break;
                case UOP_NEG:  [src appendString:@"    v = -v;\n"]; break;
                case UOP_EXP:  [src appendString:@"    v = exp(v);\n"]; break;
                case UOP_LOG:  [src appendString:@"    v = log(v);\n"]; break;
                case UOP_SQRT: [src appendString:@"    v = sqrt(v);\n"]; break;
                case UOP_SUM: case UOP_RMAX: break;
                default: break;
            }
        }
        if (is_max) [src appendString:@"    if (j == 0) acc = v; else acc = max(acc, v);\n"];
        else [src appendString:@"    acc += v;\n"];
        [src appendString:@"  }\n  out[gid] = acc;\n"];
    } else {
        [src appendString:@"  if (gid >= total_numel) return;\n"];
        [src appendString:@"  float v = in0[gid];\n"];
        for (int k = (int)n_ops - 1; k >= 0; k--) {
            switch (ops[k]) {
                case UOP_RELU: [src appendString:@"  v = max(v, 0.0f);\n"]; break;
                case UOP_NEG:  [src appendString:@"  v = -v;\n"]; break;
                case UOP_EXP:  [src appendString:@"  v = exp(v);\n"]; break;
                case UOP_LOG:  [src appendString:@"  v = log(v);\n"]; break;
                case UOP_SQRT: [src appendString:@"  v = sqrt(v);\n"]; break;
                default: break;
            }
        }
        [src appendString:@"  out[gid] = v;\n"];
    }
    [src appendString:@"}\n"];
    return src;
}

static id<MTLComputePipelineState> get_fused_pipe(const u32 *ops, u32 n_ops,
                                                    u32 n_inputs, int has_reduce) {
    u64 key = fuse_hash(ops, n_ops, has_reduce);
    for (u32 i = 0; i < fused_cache_count && i < FUSED_CACHE_SIZE; i++)
        if (fused_cache[i].key == key) return fused_cache[i].pipe;

    NSString *src = metal_codegen_fused(ops, n_ops, n_inputs, has_reduce);
    NSError *err = nil;
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [mtl_dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { NSLog(@"TinyHVM fused codegen: %@\n%@", err, src); return nil; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"fused_kernel"];
    id<MTLComputePipelineState> pipe = [mtl_dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pipe) { NSLog(@"TinyHVM fused pipeline: %@", err); return nil; }

    u32 slot = fused_cache_count < FUSED_CACHE_SIZE ?
               fused_cache_count++ : ((fused_cache_count++) % FUSED_CACHE_SIZE);
    fused_cache[slot].key = key;
    fused_cache[slot].pipe = pipe;
    return pipe;
}

// Public: dispatch fused kernel on Metal (called from tinyhvm.c)
void metal_dispatch_fused(u32 out_buf, u32 *input_bufs, u32 n_inputs,
                           u32 *ops, u32 n_ops, int has_reduce,
                           u32 out_numel, u32 reduce_dim, u32 total_numel) {
    id<MTLComputePipelineState> pipe = get_fused_pipe(ops, n_ops, n_inputs, has_reduce);
    if (!pipe) return;

    id<MTLBuffer> bufs[6]; // out + up to 4 inputs
    bufs[0] = metal_pool.bufs[out_buf];
    for (u32 i = 0; i < n_inputs; i++)
        bufs[i + 1] = metal_pool.bufs[input_bufs[i]];

    const void *params[] = { &reduce_dim, &total_numel };
    u64 psizes[] = { sizeof(u32), sizeof(u32) };
    dispatch_1d(pipe, bufs, n_inputs + 1, params, psizes, 2, out_numel);
}

// ============================================================
// Graph-level profile report (tinygrad-style kernel breakdown)
// ============================================================

static void metal_profile_report(void) {
    if (!thvm_prof_global.enabled) return;
    thvm_prof_phase_end();

    printf("\n  ╔═══════════════════════════════════════════════════════════╗\n");
    printf("  ║            TinyHVM Step Profile (THVM_PROFILE=1)         ║\n");
    printf("  ╠═══════════════════════════════════════════════════════════╣\n");

    // Phase timing
    printf("  ║  Phase Timing:                                           ║\n");
    u64 total_phase_ns = 0;
    for (u32 i = 0; i < PHASE_COUNT; i++) total_phase_ns += thvm_prof_global.phase_ns[i];
    for (u32 i = 0; i < PHASE_COUNT; i++) {
        if (thvm_prof_global.phase_ns[i] == 0) continue;
        f32 ms = (f32)thvm_prof_global.phase_ns[i] / 1e6f;
        f32 pct = total_phase_ns ? 100.0f * (f32)thvm_prof_global.phase_ns[i] / (f32)total_phase_ns : 0;
        printf("  ║    %-10s %8.1fms  (%5.1f%%)                        ║\n",
               phase_names[i], ms, pct);
    }
    printf("  ║    %-10s %8.1fms                                  ║\n",
           "TOTAL", (f32)total_phase_ns / 1e6f);

    // UOp dispatch breakdown
    printf("  ╠═══════════════════════════════════════════════════════════╣\n");
    printf("  ║  UOp Dispatch:                                           ║\n");
    printf("  ║  %-10s %6s %8s %8s %6s                  ║\n",
           "Op", "Count", "Total", "Avg", "Tens");
    printf("  ╠═══════════════════════════════════════════════════════════╣\n");
    u64 total_uop_ns = 0;
    u32 total_uop_cnt = 0;
    u32 total_uop_tens = 0;
    const char *ext_uop_names[] = {
        [UOP_POOL_GATHER] = "POOL_G"
    };
    for (u32 i = 0; i < PROF_UOP_MAX; i++) {
        if (thvm_prof_global.uop_cnt[i] == 0 && thvm_prof_global.uop_tensors[i] == 0) continue;
        const char *name = "?";
        if (i < UOP_COUNT && i < sizeof(uop_names)/sizeof(uop_names[0])) name = uop_names[i];
        else if (i == UOP_POOL_GATHER) name = "POOL_G";
        f32 total_ms = (f32)thvm_prof_global.uop_ns[i] / 1e6f;
        f32 avg_us = thvm_prof_global.uop_cnt[i] ?
            (f32)thvm_prof_global.uop_ns[i] / (f32)thvm_prof_global.uop_cnt[i] / 1e3f : 0;
        printf("  ║  %-10s %6u %6.1fms %6.0fμs %6u                  ║\n",
               name, thvm_prof_global.uop_cnt[i], total_ms, avg_us,
               thvm_prof_global.uop_tensors[i]);
        total_uop_ns += thvm_prof_global.uop_ns[i];
        total_uop_cnt += thvm_prof_global.uop_cnt[i];
        total_uop_tens += thvm_prof_global.uop_tensors[i];
    }
    printf("  ║  %-10s %6u %6.1fms          %6u                  ║\n",
           "TOTAL", total_uop_cnt, (f32)total_uop_ns / 1e6f, total_uop_tens);

    // Memory
    printf("  ╠═══════════════════════════════════════════════════════════╣\n");
    printf("  ║  Memory:                                                 ║\n");
    printf("  ║    buf_alloc:  %6u calls, %8.1f MB this step        ║\n",
           thvm_prof_global.buf_alloc_cnt,
           (f32)thvm_prof_global.buf_bytes_alloc / (1024.0f * 1024.0f));
    printf("  ║    live bufs:  %8.1f MB current, %8.1f MB peak      ║\n",
           (f32)thvm_prof_global.buf_bytes_current / (1024.0f * 1024.0f),
           (f32)thvm_prof_global.buf_bytes_peak / (1024.0f * 1024.0f));

    // Tensors
    printf("  ║  Tensors:                                                ║\n");
    printf("  ║    created: %5u   freed: %5u   peak: %5u             ║\n",
           thvm_prof_global.tensor_created,
           thvm_prof_global.tensor_freed,
           thvm_prof_global.tensor_peak);

    // Heap
    printf("  ║  Heap:                                                   ║\n");
    printf("  ║    peak: %8llu words (%5.1f MB)   at_reset: %8llu   ║\n",
           (unsigned long long)thvm_prof_global.heap_peak,
           (f32)thvm_prof_global.heap_peak * 8.0f / (1024.0f * 1024.0f),
           (unsigned long long)thvm_prof_global.heap_at_reset);

    // CPU↔GPU transfers
    printf("  ║  CPU↔GPU:                                                ║\n");
    printf("  ║    read:  %6u calls, %8.1f MB                      ║\n",
           thvm_prof_global.cpu_read_cnt,
           (f32)thvm_prof_global.cpu_read_bytes / (1024.0f * 1024.0f));
    printf("  ║    write: %6u calls, %8.1f MB                      ║\n",
           thvm_prof_global.cpu_write_cnt,
           (f32)thvm_prof_global.cpu_write_bytes / (1024.0f * 1024.0f));

    printf("  ╚═══════════════════════════════════════════════════════════╝\n");
    (void)ext_uop_names;
}

static void metal_profile_reset(void) {
    thvm_prof_step_reset();
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
    .op_im2col = metal_op_im2col,
    .op_col2im = metal_op_col2im,
    .op_nhwc_to_nchw = metal_op_nhwc_to_nchw,
    .op_nchw_to_nhwc = metal_op_nchw_to_nhwc,
    .op_bias_add = metal_op_bias_add,
    .op_col_sum  = metal_op_col_sum,
    .op_transpose = metal_op_transpose,
    .op_maxpool_fwd = metal_op_maxpool_fwd,
    .op_maxpool_bwd = metal_op_maxpool_bwd,
    .op_relu_bwd = metal_op_relu_bwd,
    .op_zero_fill = metal_op_zero_fill,
    .op_adam_step = metal_op_adam_step,
    .pool_reset = metal_pool_reset,
    .begin_batch = metal_begin_batch,
    .end_batch   = metal_end_batch,
    .profile_report = metal_profile_report,
    .profile_reset  = metal_profile_reset,
};

