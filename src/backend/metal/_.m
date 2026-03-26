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
