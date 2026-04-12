// nn/_.c — Neural network layers hub
// Aggregates all layer sub-files.

#include "../tinyhvm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static u32 tensor_from(TinyHVM *ctx, f32 *data, Shape s) {
    Term t = thvm_tensor(ctx, data, s);
    return (u32)term_val(t);
}

#include "softmax.c"
#include "loss.c"
#include "linear.c"
#include "batchnorm.c"
#include "adam.c"
#include "conv.c"
#include "sequential.c"
#include "tensor_nn_api.c"
