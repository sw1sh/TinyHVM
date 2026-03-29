// metal/_.m — Metal backend hub
// Aggregates all Metal sub-files and exports the Backend struct.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "../../tinyhvm.h"
#include <stdlib.h>
#include <string.h>

// State declarations (shared across all sub-files)
#include "init.m"     // device, pipelines, ViewParams, state vars
#include "batch.m"    // command batching (uses mtl_queue from init)
#include "pool.m"     // buffer pool (uses mtl_dev, batch_dirty from above)
#include "dispatch.m" // GPU dispatch helpers
// Forward declarations for cross-file references within the Metal backend
static void metal_contiguify(u32 dst_buf, u32 numel, u32 src_buf, const View *src_view);
static void metal_dispatch_kernel(u32 dst_buf, u32 dst_numel,
    u32 *leaf_bufs, const View **leaf_views, u32 n_leaves,
    FusedOp *ops, u32 n_ops, int has_reduce, u32 reduce_dim, const Shape *full_shape);
#include "ops.m"      // unary, binary, matmul, reduce
#include "conv.m"     // CNN/layout ops
#include "optim.m"    // optimizer/pooling GPU kernels
#include "fused.m"    // fused kernel codegen (legacy — being replaced by codegen.m)
#include "codegen.m"  // unified JIT codegen
#include "jit.m"      // JIT capture/replay
#include "profile.m"  // profiling

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
    .dispatch_kernel_rs = metal_dispatch_kernel_rs,
    .contiguify         = metal_contiguify,
    .buf_copy           = metal_buf_copy,
    .buf_read_nosync    = metal_buf_read_nosync,
    .adam_step           = metal_op_adam_step,
    .pool_reset = metal_pool_reset,
    .begin_batch = metal_begin_batch,
    .end_batch   = metal_end_batch,
    .profile_report = metal_profile_report,
    .profile_reset  = metal_profile_reset,
};
