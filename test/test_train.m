// test_train.c — XOR training with thvm_grad (graph-level SGD)
// 2-layer MLP: [2] → [4] → [1], relu activation
// Trains on 4 XOR examples, converges to loss < 0.05

#include "../src/tinyhvm.c"
#include "../src/gpu_cpu.c"
#ifdef __APPLE__
  #include "../src/gpu_metal.m"
#endif

#ifndef DEVICE
  #define DEVICE "cpu"
#endif

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// SGD step: param = param - lr * grad (via graph ops, reduce in-place)
static void sgd_update(TinyHVM *ctx, Term param, Term grad_term, f32 lr) {
    // Reduce gradient
    Term g = thvm_reduce(ctx, grad_term);
    if (term_tag(g) != TAG_TEN) return;

    u32 pid = (u32)term_val(param);
    u32 gid = (u32)term_val(g);
    TensorMeta *mp = &ctx->tensors[pid];
    TensorMeta *mg = &ctx->tensors[gid];

    // Read param and grad, do param -= lr * grad on CPU
    u32 n = mp->view.numel;
    u32 dsz = dtype_size(mp->dtype);
    f32 *p_data = malloc(n * dsz);
    f32 *g_data = malloc(n * dsz);
    ctx->backend->buf_read(mp->buf_id, p_data, n * dsz);
    ctx->backend->buf_read(mg->buf_id, g_data, n * dsz);

    for (u32 i = 0; i < n; i++)
        p_data[i] -= lr * g_data[i];

    ctx->backend->buf_write(mp->buf_id, p_data, n * dsz);
    free(p_data);
    free(g_data);
}

int main(void) {
    printf("=== XOR Training (thvm_grad) ===\n\n");
    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    // XOR data: 4 examples × 2 inputs
    f32 x_data[] = {0,0, 0,1, 1,0, 1,1};  // [4,2]
    f32 y_data[] = {0, 1, 1, 0};           // [4,1]

    Term X = thvm_tensor(ctx, x_data, SHAPE(4, 2));
    Term Y = thvm_tensor(ctx, y_data, SHAPE(4, 1));

    // Weights: hand-init to break symmetry
    // W1: [2,4], b1: [1,4], W2: [4,1], b2: [1,1]
    f32 w1_data[] = { 0.5f, -0.3f,  0.2f,  0.4f,
                     -0.1f,  0.6f, -0.4f,  0.3f};
    f32 b1_data[] = { 0.0f, 0.0f, 0.0f, 0.0f};
    f32 w2_data[] = { 0.5f, -0.3f, 0.2f, 0.4f};
    f32 b2_data[] = { 0.0f };

    Term W1 = thvm_tensor(ctx, w1_data, SHAPE(2, 4));
    Term B1 = thvm_tensor(ctx, b1_data, SHAPE(1, 4));
    Term W2 = thvm_tensor(ctx, w2_data, SHAPE(4, 1));
    Term B2 = thvm_tensor(ctx, b2_data, SHAPE(1, 1));

    thvm_set_requires_grad(ctx, W1);
    thvm_set_requires_grad(ctx, B1);
    thvm_set_requires_grad(ctx, W2);
    thvm_set_requires_grad(ctx, B2);

    f32 lr = 0.5f;
    int converged = 0;

    for (int epoch = 0; epoch < 5000; epoch++) {
        // Forward: h = relu(X·W1 + B1), out = h·W2 + B2
        thvm_start_recording(ctx);

        Term z1 = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, X, W1), B1);
        Term h  = thvm_op(ctx, UOP_RELU, z1, term_era());
        Term out = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, W2), B2);

        // Loss: MSE = mean((out - Y)^2) — approximate as sum((out-Y)^2)/4
        Term diff = thvm_op(ctx, UOP_SUB, out, Y);
        Term sq   = thvm_op(ctx, UOP_MUL, diff, diff);
        Term loss_term = thvm_reduce(ctx, sq);

        thvm_stop_recording(ctx);

        // Compute loss value
        f32 *loss_data = thvm_to_host(ctx, loss_term);
        u32 loss_id = (u32)term_val(loss_term);
        f32 mse = 0;
        for (u32 i = 0; i < ctx->tensors[loss_id].view.numel; i++)
            mse += loss_data[i];
        mse /= 4.0f;

        if (epoch % 500 == 0 || mse < 0.05f) {
            printf("  epoch %4d  loss = %.6f\n", epoch, mse);
        }

        if (mse < 0.05f) {
            converged = 1;
            printf("\n  Converged at epoch %d!\n", epoch);
            break;
        }

        // Gradients via thvm_grad
        Term gW1 = thvm_grad(ctx, loss_term, W1);
        Term gB1 = thvm_grad(ctx, loss_term, B1);
        Term gW2 = thvm_grad(ctx, loss_term, W2);
        Term gB2 = thvm_grad(ctx, loss_term, B2);

        // SGD update
        sgd_update(ctx, W1, gW1, lr);
        sgd_update(ctx, B1, gB1, lr);
        sgd_update(ctx, W2, gW2, lr);
        sgd_update(ctx, B2, gB2, lr);
    }

    // Final predictions
    printf("\n  Final predictions:\n");
    Term z1 = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, X, W1), B1);
    Term h  = thvm_op(ctx, UOP_RELU, z1, term_era());
    Term out = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, W2), B2);
    Term final = thvm_reduce(ctx, out);
    f32 *pred = thvm_to_host(ctx, final);
    printf("    0 XOR 0 = %.3f (expect ~0)\n", pred[0]);
    printf("    0 XOR 1 = %.3f (expect ~1)\n", pred[1]);
    printf("    1 XOR 0 = %.3f (expect ~1)\n", pred[2]);
    printf("    1 XOR 1 = %.3f (expect ~0)\n", pred[3]);

    if (converged) {
        printf("\n  PASS: XOR training converged\n");
    } else {
        printf("\n  FAIL: did not converge in 5000 epochs\n");
    }

    thvm_free(ctx);
    return converged ? 0 : 1;
}
