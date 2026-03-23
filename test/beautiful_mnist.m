// beautiful_mnist.m — CNN matching tinygrad's beautiful_mnist.py
// Architecture: Conv(1,32,5)→ReLU→Conv(32,32,5)→ReLU→BN(32)→MaxPool(2)
//             → Conv(32,64,3)→ReLU→Conv(64,64,3)→ReLU→BN(64)→MaxPool(2)
//             → Flatten→Linear(576,10)
//
// Uses IC autograd: thvm_sequential for forward, thvm_backward for gradients.
// No manual backward pass. Weights are IC Terms.

#include "../src/tinyhvm.c"
#include "../src/cpu.c"
#ifdef __APPLE__
  #include "../src/metal.m"
#endif
#include "../src/layers.c"
#include "../src/nn/datasets.c"

#ifndef DEVICE
  #define DEVICE "metal"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

static void xavier_init(f32 *data, u32 fan_in, u32 fan_out, u32 n) {
    f32 scale = sqrtf(2.0f / (f32)(fan_in + fan_out));
    for (u32 i = 0; i < n; i++)
        data[i] = scale * ((f32)rand() / (f32)RAND_MAX * 2.0f - 1.0f);
}

static Term make_weight(TinyHVM *ctx, Shape s, u32 fan_in, u32 fan_out) {
    u32 n = 1;
    for (u32 i = 0; i < s.rank; i++) n *= s.dims[i];
    f32 *data = malloc(n * sizeof(f32));
    xavier_init(data, fan_in, fan_out, n);
    Term t = thvm_tensor(ctx, data, s);
    free(data);
    return t;
}

static Term make_zeros(TinyHVM *ctx, u32 n) {
    f32 *z = calloc(n, sizeof(f32));
    Term t = thvm_tensor(ctx, z, SHAPE(n));
    free(z);
    return t;
}

static Term make_ones(TinyHVM *ctx, u32 n) {
    f32 *o = malloc(n * sizeof(f32));
    for (u32 i = 0; i < n; i++) o[i] = 1.0f;
    Term t = thvm_tensor(ctx, o, SHAPE(n));
    free(o);
    return t;
}

static Term relu_fn(TinyHVM *ctx, Term x) {
    return thvm_op(ctx, UOP_RELU, x, term_era());
}

int main(void) {
    srand(42);
    printf("=== beautiful_mnist (TinyHVM — IC autograd) ===\n\n");

    MNISTData data = mnist_load("data");
    printf("  Train: %u, Test: %u\n\n", data.n_train, data.n_test);

    TinyHVM *ctx = thvm_init(thvm_device(DEVICE));

    // Build model as Layer array — weights are IC Terms
    #define N_LAYERS 14
    Layer model[N_LAYERS] = {
        // Block 1
        {.type=LAYER_CONV2D, .conv={make_weight(ctx, (Shape){.dims={32,1,5,5},.rank=4}, 25, 32),
                                     make_zeros(ctx, 32), 1, 32, 5}},
        {.type=LAYER_FN, .fn=relu_fn},
        {.type=LAYER_CONV2D, .conv={make_weight(ctx, (Shape){.dims={32,32,5,5},.rank=4}, 800, 32),
                                     make_zeros(ctx, 32), 32, 32, 5}},
        {.type=LAYER_FN, .fn=relu_fn},
        {.type=LAYER_BN, .bn={make_ones(ctx, 32), make_zeros(ctx, 32),
                               make_zeros(ctx, 32), make_ones(ctx, 32), 32}},
        {.type=LAYER_MAXPOOL, .pool={2}},
        // Block 2
        {.type=LAYER_CONV2D, .conv={make_weight(ctx, (Shape){.dims={64,32,3,3},.rank=4}, 288, 64),
                                     make_zeros(ctx, 64), 32, 64, 3}},
        {.type=LAYER_FN, .fn=relu_fn},
        {.type=LAYER_CONV2D, .conv={make_weight(ctx, (Shape){.dims={64,64,3,3},.rank=4}, 576, 64),
                                     make_zeros(ctx, 64), 64, 64, 3}},
        {.type=LAYER_FN, .fn=relu_fn},
        {.type=LAYER_BN, .bn={make_ones(ctx, 64), make_zeros(ctx, 64),
                               make_zeros(ctx, 64), make_ones(ctx, 64), 64}},
        {.type=LAYER_MAXPOOL, .pool={2}},
        // Head
        {.type=LAYER_FLATTEN},
        {.type=LAYER_LINEAR, .lin={make_weight(ctx, SHAPE(576, 10), 576, 10),
                                    make_zeros(ctx, 10), 576, 10}},
    };

    // Auto-extract trainable parameters from the model
    Term params[N_LAYERS * 2];  // max 2 params per layer
    u32 param_sizes[N_LAYERS * 2];
    u32 n_params = 0;
    for (u32 i = 0; i < N_LAYERS; i++) {
        Layer *l = &model[i];
        switch (l->type) {
            case LAYER_CONV2D:
                params[n_params] = l->conv.w;
                param_sizes[n_params] = ctx->tensors[(u32)term_val(l->conv.w)].view.numel;
                n_params++;
                params[n_params] = l->conv.b;
                param_sizes[n_params] = ctx->tensors[(u32)term_val(l->conv.b)].view.numel;
                n_params++;
                break;
            case LAYER_BN:
                params[n_params] = l->bn.gamma;
                param_sizes[n_params] = ctx->tensors[(u32)term_val(l->bn.gamma)].view.numel;
                n_params++;
                params[n_params] = l->bn.beta;
                param_sizes[n_params] = ctx->tensors[(u32)term_val(l->bn.beta)].view.numel;
                n_params++;
                break;
            case LAYER_LINEAR:
                params[n_params] = l->lin.w;
                param_sizes[n_params] = ctx->tensors[(u32)term_val(l->lin.w)].view.numel;
                n_params++;
                params[n_params] = l->lin.b;
                param_sizes[n_params] = ctx->tensors[(u32)term_val(l->lin.b)].view.numel;
                n_params++;
                break;
            default: break;
        }
    }
    // Mark all params as requiring gradients
    for (u32 i = 0; i < n_params; i++)
        thvm_set_requires_grad(ctx, params[i]);
    printf("  %u trainable parameters extracted\n", n_params);

    // Load entire training set as a source Term BEFORE adam init
    // so train_data is preserved by thvm_reset
    u32 BS = 128;
    Term train_data = thvm_tensor(ctx, data.train_images,
        (Shape){.dims={data.n_train, 1, 28, 28}, .rank=4});

    Adam opt = adam_init(ctx, 0.001f, n_params);
    for (u32 i = 0; i < n_params; i++)
        adam_add_param(ctx, &opt, i, (u32)term_val(params[i]), param_sizes[i]);

    u32 n_weights = ctx->tensor_count;  // AFTER data + adam state = all persistent tensors
    u32 n_batches = data.n_train / BS;
    u32 n_steps = 70;
    f32 lr_max = 0.001f, lr_min = 0.0001f;
    printf("  Training %u steps, BS=%u...\n\n", n_steps, BS);

    // Interaction count limit (for profiling): THVM_MAX_ITRS env
    u64 max_itrs = 0;
    { const char *mi = getenv("THVM_MAX_ITRS");
      if (mi) max_itrs = (u64)atoll(mi); }
    if (max_itrs) printf("  ⚡ Interaction limit: %llu\n", (unsigned long long)max_itrs);

    for (u32 step = 0; step < n_steps; step++) {
      @autoreleasepool {
        if (max_itrs && ctx->itrs >= max_itrs) {
            printf("\n  ⚡ Hit interaction limit (%llu itrs) at step %u\n",
                   (unsigned long long)ctx->itrs, step);
            break;
        }

        // Per-step profile reset
        if (ctx->backend && ctx->backend->profile_reset)
            ctx->backend->profile_reset();

        f32 progress = (f32)step / (f32)n_steps;
        opt.lr = lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf(3.14159f * progress));
        clock_t t0 = clock();

        // === FORWARD PHASE ===
        thvm_prof_phase(PHASE_FORWARD);

        // Begin GPU command buffer batching — all dispatches until end_batch
        // go into one command buffer (eliminates ~50 GPU syncs per step)
        if (ctx->backend->begin_batch) ctx->backend->begin_batch();

        // Batch via thvm_shrink — source UOp, no memcpy
        u32 bi = step % n_batches;
        u32 start = bi * BS;
        Term x = thvm_shrink(ctx, train_data,
            (u32[]){start, start + BS, 0, 1, 0, 28, 0, 28}, 4);
        thvm_set_requires_grad(ctx, x);

        // Labels (u8, used by loss only — not in IC graph)
        u8 *by = &data.train_labels[start];

        // Forward — pure IC graph composition
        thvm_start_recording(ctx);
        Term logits = thvm_sequential(ctx, x, model, N_LAYERS, BS, 1);

        // Loss — cross-entropy (pure UOp composition, fully differentiable)
        Term loss = cross_entropy_loss(ctx, logits, by, BS, 10);

        // Flush GPU work before CPU reads loss value
        if (ctx->backend->end_batch) ctx->backend->end_batch();

        // Read loss value for printing
        Term loss_reduced = thvm_reduce(ctx, loss);
        f32 *loss_host = thvm_to_host(ctx, loss_reduced);
        f32 loss_val = loss_host[0];

        // Stop recording AFTER loss is reduced — all forward ops now have creator_op
        // Backward runs with recording OFF so gradient ops can use SUM(MUL) fusion
        thvm_stop_recording(ctx);

        // === BACKWARD PHASE ===
        thvm_prof_phase(PHASE_BACKWARD);

        // Re-batch for gradient computation + adam
        if (ctx->backend->begin_batch) ctx->backend->begin_batch();

        // Single-pass backward: compute ALL param gradients in one traversal
        Term grad_terms[n_params];
        clock_t tb0 = clock();
        thvm_backward(ctx, loss_reduced, params, grad_terms, n_params);
        clock_t tb1 = clock();
        u32 grad_ids[n_params];
        for (u32 i = 0; i < n_params; i++) {
            Term g = thvm_reduce(ctx, grad_terms[i]);
            grad_ids[i] = (term_tag(g) == TAG_TEN) ? (u32)term_val(g) : 0;
        }
        clock_t tb2 = clock();
        if (step < 2) {
            printf("  backward: graph=%.0fms reduce=%.0fms\n",
                1000.0*(tb1-tb0)/(double)CLOCKS_PER_SEC,
                1000.0*(tb2-tb1)/(double)CLOCKS_PER_SEC);
            // Debug: check gradient health
            int n_era = 0, n_ten = 0;
            for (u32 i = 0; i < n_params; i++) {
                if (term_tag(grad_terms[i]) == TAG_ERA) n_era++;
                else n_ten++;
            }
            printf("  grads: %d TAG_TEN, %d TAG_ERA\n", n_ten, n_era);
            if (grad_ids[0]) {
                f32 gd[32]; u32 gn = ctx->tensors[grad_ids[0]].view.numel;
                if (gn > 32) gn = 32;
                ctx->backend->buf_read(ctx->tensors[grad_ids[0]].buf_id, gd, gn*sizeof(f32));
                f32 gnorm = 0; for (u32 j=0;j<gn;j++) gnorm += gd[j]*gd[j];
                printf("  grad[0] norm=%.6f first=%.6f\n", sqrtf(gnorm), gd[0]);
            } else {
                printf("  grad[0] = NO GRADIENT\n");
            }
        }



        // === ADAM PHASE ===
        thvm_prof_phase(PHASE_ADAM);
        adam_step(ctx, &opt, grad_ids);

        // Flush all GPU work
        if (ctx->backend->end_batch) ctx->backend->end_batch();

        f32 ms = 1000.0f * (f32)(clock() - t0) / (f32)CLOCKS_PER_SEC;
        printf("  step %3u/%u  loss=%.4f  lr=%.5f  (%.0fms)  tensors=%u  itrs=%llu\n",
               step, n_steps, loss_val, opt.lr, ms, ctx->tensor_count,
               (unsigned long long)ctx->itrs);

        // === RESET PHASE ===
        thvm_prof_phase(PHASE_RESET);
        thvm_reset(ctx, n_weights);
        thvm_prof_phase_end();

        // Per-step profile report
        if (ctx->backend && ctx->backend->profile_report)
            ctx->backend->profile_report();
      } // @autoreleasepool
    }

    // Eval (skip when profiling with interaction limit)
    f32 acc = 0;
    if (!max_itrs) {
    printf("\n  Evaluating test set...\n");
    Term test_data = thvm_tensor(ctx, data.test_images,
        (Shape){.dims={data.n_test, 1, 28, 28}, .rank=4});
    // Watermark: preserve test_data across reset calls
    u32 eval_keep = ctx->tensor_count;
    u32 correct = 0, tbs = 64, tb = data.n_test / tbs;
    for (u32 b = 0; b < tb; b++) {
        Term x = thvm_shrink(ctx, test_data,
            (u32[]){b*tbs, (b+1)*tbs, 0, 1, 0, 28, 0, 28}, 4);
        Term logits = thvm_sequential(ctx, x, model, N_LAYERS, tbs, 0);
        f32 batch_acc = thvm_eval_accuracy(ctx, logits, &data.test_labels[b*tbs], tbs, 10);
        correct += (u32)(batch_acc * (f32)tbs / 100.0f);
        thvm_reset(ctx, eval_keep);
    }
    acc = 100.0f * (f32)correct / (f32)(tb * tbs);
    printf("\n  Test accuracy: %.1f%% (%u/%u)\n", acc, correct, tb * tbs);
    printf("\n  %s: CNN test accuracy %s 90%%\n", acc > 90 ? "PASS" : "FAIL",
           acc > 90 ? ">" : "<");
    } else {
        printf("\n  (eval skipped — profiling mode)\n");
    }

    adam_free(&opt);
    mnist_free(&data);
    thvm_free(ctx);
    return max_itrs ? 0 : (acc > 90 ? 0 : 1);
}
