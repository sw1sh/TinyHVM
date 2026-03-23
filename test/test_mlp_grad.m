// test_mlp_grad.m — Simple MLP gradient correctness test
// No convs. Verifies: gradients exist, loss decreases, no memory leak.
#include "../src/tinyhvm.c"
#include "../src/cpu.c"
#include "../src/metal.m"

static void fill_rand(f32 *buf, u32 n, f32 scale) {
    for (u32 i = 0; i < n; i++)
        buf[i] = ((f32)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
}

int main() {
    srand(42);
    TinyHVM *ctx = thvm_init(thvm_device("metal"));
    u32 BS = 16, IN = 4, HID = 8, OUT = 2;

    // Weights
    f32 w1_data[4 * 8], b1_data[8], w2_data[8 * 2], b2_data[2];
    fill_rand(w1_data, IN * HID, 0.3f);
    fill_rand(b1_data, HID, 0.1f);
    fill_rand(w2_data, HID * OUT, 0.3f);
    fill_rand(b2_data, OUT, 0.1f);

    Term w1 = thvm_tensor(ctx, w1_data, shape_of((u32[]){IN, HID}, 2));
    Term b1 = thvm_tensor(ctx, b1_data, shape_of((u32[]){1, HID}, 2));
    Term w2 = thvm_tensor(ctx, w2_data, shape_of((u32[]){HID, OUT}, 2));
    Term b2 = thvm_tensor(ctx, b2_data, shape_of((u32[]){1, OUT}, 2));
    Term params[] = {w1, b1, w2, b2};
    u32 n_params = 4;
    for (u32 i = 0; i < n_params; i++) thvm_set_requires_grad(ctx, params[i]);
    u32 n_weights = ctx->tensor_count;

    // Input data
    f32 x_data[16 * 4], target_data[16 * 2];
    fill_rand(x_data, BS * IN, 1.0f);
    for (u32 i = 0; i < BS; i++) {
        f32 s = 0;
        for (u32 j = 0; j < IN; j++) s += x_data[i * IN + j];
        target_data[i * OUT] = s > 0 ? 1.0f : 0.0f;
        target_data[i * OUT + 1] = s > 0 ? 0.0f : 1.0f;
    }

    f32 lr = 0.05f;
    f32 first_loss = 0, last_loss = 0;
    printf("=== MLP Gradient Test (4→8→2, batch=%u) ===\n", BS);

    for (u32 step = 0; step < 50; step++) {
        thvm_start_recording(ctx);
        Term x = thvm_tensor(ctx, x_data, shape_of((u32[]){BS, IN}, 2));
        Term target = thvm_tensor(ctx, target_data, shape_of((u32[]){BS, OUT}, 2));

        // Forward: h = relu(x @ W1 + b1); out = h @ W2 + b2
        Term h = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, x, w1), b1);
        h = thvm_op(ctx, UOP_RELU, h, term_era());
        Term out = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, w2), b2);

        // MSE loss
        Term diff = thvm_op(ctx, UOP_SUB, out, target);
        Term sq = thvm_op(ctx, UOP_MUL, diff, diff);
        u32 axes[] = {0, 1};
        Term loss = thvm_sum_axes(ctx, sq, axes, 2);
        f32 inv = 1.0f / (f32)(BS * OUT);
        loss = thvm_op(ctx, UOP_MUL, loss, thvm_tensor(ctx, &inv, SHAPE(1)));
        loss = thvm_reshape(ctx, loss, SHAPE(1));

        // Reduce loss while recording (provenance is tracked)
        loss = thvm_reduce(ctx, loss);
        f32 *lv = thvm_to_host(ctx, loss);

        thvm_stop_recording(ctx);

        if (step == 0) first_loss = lv[0];
        last_loss = lv[0];
        if (step < 5 || step % 10 == 0)
            printf("  step %2u  loss=%.6f  tensors=%u\n", step, lv[0], ctx->tensor_count);

        // Backward
        Term grads[4];
        loss = thvm_reduce(ctx, loss);

        thvm_backward(ctx, loss, params, grads, n_params);

        // SGD update
        for (u32 p = 0; p < n_params; p++) {
            Term g = thvm_reduce(ctx, grads[p]);
            if (term_tag(g) != TAG_TEN) { printf("  ERR: param %u no grad\n", p); continue; }
            u32 pid = (u32)term_val(params[p]), gid = (u32)term_val(g);
            TensorMeta *mp = &ctx->tensors[pid];
            u32 n = mp->view.numel;
            f32 *pd = malloc(n * sizeof(f32)), *gd = malloc(n * sizeof(f32));
            ctx->backend->buf_read(mp->buf_id, pd, n * sizeof(f32));
            ctx->backend->buf_read(ctx->tensors[gid].buf_id, gd, n * sizeof(f32));

            if (step == 0) {
                f32 gn = 0; for (u32 j = 0; j < n; j++) gn += gd[j] * gd[j];
                printf("    param[%u] |∇|=%.6f shape=[", p, sqrtf(gn));
                for (u32 d = 0; d < mp->view.shape.rank; d++)
                    printf("%u%s", mp->view.shape.dims[d], d < mp->view.shape.rank - 1 ? "," : "");
                // print first few values
                printf("] vals=[");
                for (u32 j = 0; j < (n < 4 ? n : 4); j++) printf("%.4f ", gd[j]);
                printf("]\n");
            }

            for (u32 j = 0; j < n; j++) pd[j] -= lr * gd[j];
            ctx->backend->buf_write(mp->buf_id, pd, n * sizeof(f32));
            free(pd); free(gd);
        }

        thvm_reset(ctx, n_weights);
    }

    printf("\nfirst_loss=%.6f last_loss=%.6f\n", first_loss, last_loss);
    if (last_loss < first_loss * 0.5f) {
        printf("=== PASS: loss decreased by >50%% ===\n");
    } else {
        printf("=== FAIL: loss did not decrease enough ===\n");
    }

    thvm_free(ctx);
    return last_loss < first_loss * 0.5f ? 0 : 1;
}
