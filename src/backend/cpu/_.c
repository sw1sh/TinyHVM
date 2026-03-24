// cpu/_.c — CPU backend hub
// Aggregates pool + ops and exports the Backend struct.

#include "../../tinyhvm.h"
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define HAS_BLAS 1
#else
#define HAS_BLAS 0
#endif

#include "pool.c"
#include "ops.c"

Backend cpu_backend = {
    .init      = cpu_init,
    .shutdown  = cpu_shutdown,
    .buf_alloc = cpu_buf_alloc,
    .buf_free  = cpu_buf_free,
    .buf_write = cpu_buf_write,
    .buf_read  = cpu_buf_read,
    .op_unary  = cpu_op_unary,
    .op_binary = cpu_op_binary,
    .op_mm     = cpu_op_mm,
    .op_reduce = cpu_op_reduce,
    .op_im2col = NULL,
    .op_col2im = NULL,
    .op_nhwc_to_nchw = NULL,
    .op_nchw_to_nhwc = NULL,
    .op_bias_add = NULL,
    .op_col_sum  = NULL,
    .op_transpose = NULL,
    .op_maxpool_fwd = NULL,
    .op_maxpool_bwd = NULL,
    .op_relu_bwd = NULL,
    .op_zero_fill = NULL,
    .op_adam_step = NULL,
    .pool_reset = cpu_pool_reset,
    .begin_batch = NULL,
    .end_batch   = NULL,
    .profile_report = NULL,
    .profile_reset  = NULL,
};
