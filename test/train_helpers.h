// test/train_helpers.h — Recursive training program (test-only)
// Hardcoded for the 2-layer MLP test case.

// ── Recursive training as inet program ───────────────────────
//
// train = λn_unused. IFZ(counter, ERA, λm. APP(aW1, APP(aB1, APP(aW2, APP(aB2, APP(REF(train), m))))))
//
// IFZ(counter, zero_case, succ_lam):
//   Scalar tensor counter. If counter==0 → zero_case (ERA).
//   If counter>0 → APP(succ_lam, TEN(counter-1)).
//   The handler creates TEN(counter-1) and returns APP(succ_lam, dec).
//
// Sequencing via APP-TEN rule (APP(TEN, x) → x):
//   Each ASSIGN in function position → trampoline enters & forces → TEN.
//   APP-TEN fires → continue to next in chain.
//
// ALO (allocate-on-unfold):
//   REF(train_id) triggers term_clone(def_body), fresh heap per step.

static Term train_program(TinyHVM *ctx,
                           Term W1, Term B1, Term W2, Term B2,
                           Term X, Term Y, Term LR, int N) {
    u32 train_id = ctx->def_count++;
    assert(train_id < 256);

    // Forward: 2-layer MLP
    Term z1  = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, X, W1),
                       thvm_expand(ctx, B1, SHAPE(4, 4)));
    Term h   = thvm_op(ctx, UOP_RELU, z1, term_era());
    Term out = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, W2),
                       thvm_expand(ctx, B2, SHAPE(4, 1)));

    // MSE Loss
    Term diff = thvm_op(ctx, UOP_SUB, out, Y);
    Term sq   = thvm_op(ctx, UOP_MUL, diff, diff);
    u32 axes[] = {0, 1};
    Term loss = thvm_sum_axes(ctx, sq, axes, 2);
    f32 inv_n_val = 1.0f / 4.0f;
    loss = thvm_op(ctx, UOP_MUL, loss, thvm_tensor(ctx, &inv_n_val, SHAPE(1)));
    loss = thvm_reshape(ctx, loss, SHAPE(1));

    // Gradients
    Term gW1 = thvm_grad(ctx, loss, W1);
    Term gB1 = thvm_grad(ctx, loss, B1);
    Term gW2 = thvm_grad(ctx, loss, W2);
    Term gB2 = thvm_grad(ctx, loss, B2);

    // Log loss (side-effect: prints scalar, returns tensor for sequencing)
    Term log_loss = thvm_log_print(ctx, loss);

    // ASSIGNs (independent)
    Term aW1 = thvm_assign(ctx, W1,
        thvm_op(ctx, UOP_SUB, W1, thvm_op(ctx, UOP_MUL, LR, gW1)));
    Term aB1 = thvm_assign(ctx, B1,
        thvm_op(ctx, UOP_SUB, B1, thvm_op(ctx, UOP_MUL, LR, gB1)));
    Term aW2 = thvm_assign(ctx, W2,
        thvm_op(ctx, UOP_SUB, W2, thvm_op(ctx, UOP_MUL, LR, gW2)));
    Term aB2 = thvm_assign(ctx, B2,
        thvm_op(ctx, UOP_SUB, B2, thvm_op(ctx, UOP_MUL, LR, gB2)));

    // succ_lam = λm. APP(log_loss, APP(aW1, APP(aB1, APP(aW2, APP(aB2, APP(REF(train), m))))))
    // log_loss forces+prints the loss, returns TEN → APP-TEN fires → continues to assigns
    Term var_m;
    Term rec = thvm_app(ctx, thvm_ref(ctx, train_id), term_era()); // placeholder
    // Build with proper var binding:
    u64 lm = heap_alloc(ctx, 2);
    var_m = term_new(TAG_VAR, 0, lm);
    heap_set(ctx, lm, term_set_sub(var_m));
    rec = thvm_app(ctx, thvm_ref(ctx, train_id), var_m);
    Term chain = thvm_app(ctx, log_loss,
                 thvm_app(ctx, aW1,
                 thvm_app(ctx, aB1,
                 thvm_app(ctx, aW2,
                 thvm_app(ctx, aB2, rec)))));
    heap_set(ctx, lm + 1, chain);
    Term succ_lam = term_new(TAG_LAM, 0, lm);

    // def body = IFZ(counter_arg, ERA, succ_lam)
    // The def takes a scalar tensor counter as its arg via APP-LAM
    u64 ln = heap_alloc(ctx, 2);
    Term var_n = term_new(TAG_VAR, 0, ln);
    heap_set(ctx, ln, term_set_sub(var_n));
    heap_set(ctx, ln + 1, thvm_ifz(ctx, var_n, term_era(), succ_lam));
    Term def_body = term_new(TAG_LAM, 0, ln);

    ctx->defs[train_id] = def_body;

    // Entry: APP(REF(train), TEN(N))
    f32 nf = (f32)N;
    Term counter = thvm_tensor(ctx, &nf, SHAPE(1));
    return thvm_app(ctx, thvm_ref(ctx, train_id), counter);
}

// ── MNIST recursive inet training ────────────────────────────
// Same pattern as train_program but with dynamic batch slicing.
// Each step computes batch offset from the counter, uses SHRINK
// with a dynamically-computed pairs tensor.
//
// Architecture: [feat] → [H] → [10], ReLU, cross-entropy loss, SGD.
//
// train = λcounter. IFZ(counter, ERA, λm.
//   step = N - counter
//   offset = (start_batch + step) * BS
//   X = SHRINK(X_all, [offset, offset+BS, 0, feat])
//   Y = SHRINK(Y_all, [offset, offset+BS, 0, 10])
//   ... forward, loss, grad, log, assign ...
//   APP(REF(train), m)
// )

static Term mnist_train_program(TinyHVM *ctx,
    Term W1, Term B1, Term W2, Term B2,
    Term X_all, Term Y_all,
    Term LR, Term INV_N,
    Term MASK_S_X, Term MASK_E_X, Term FIXED_X,
    Term MASK_S_Y, Term MASK_E_Y, Term FIXED_Y,
    Term BS_ten, Term N_ten, Term START_BATCH_ten,
    u32 batch_size, u32 hidden_dim, int n_steps)
{
    u32 train_id = ctx->def_count++;
    assert(train_id < 256);

    // outer lambda: λcounter. IFZ(counter, ERA, λm. body)
    u64 ln = heap_alloc(ctx, 2);
    Term var_n = term_new(TAG_VAR, 0, ln);
    heap_set(ctx, ln, term_set_sub(var_n));

    // inner lambda: λm. chain
    u64 lm = heap_alloc(ctx, 2);
    Term var_m = term_new(TAG_VAR, 0, lm);
    heap_set(ctx, lm, term_set_sub(var_m));

    // step = N - counter (counter counts down, step goes 0..N-1)
    Term step = thvm_op(ctx, UOP_SUB, N_ten, var_n);
    // abs_batch = start_batch + step
    Term abs_batch = thvm_op(ctx, UOP_ADD, START_BATCH_ten, step);
    // row offset = abs_batch * batch_size
    Term start = thvm_op(ctx, UOP_MUL, abs_batch, BS_ten);
    Term end_t = thvm_op(ctx, UOP_ADD, start, BS_ten);

    // Dynamic pairs: [start, end, 0, D] via broadcast arithmetic
    // pairs = start * [1,0,0,0] + end * [0,1,0,0] + [0,0,0,D]
    Term pairs_x = thvm_op(ctx, UOP_ADD,
        thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MUL, start, MASK_S_X),
            thvm_op(ctx, UOP_MUL, end_t, MASK_E_X)),
        FIXED_X);
    Term pairs_y = thvm_op(ctx, UOP_ADD,
        thvm_op(ctx, UOP_ADD,
            thvm_op(ctx, UOP_MUL, start, MASK_S_Y),
            thvm_op(ctx, UOP_MUL, end_t, MASK_E_Y)),
        FIXED_Y);

    // Dynamic SHRINK (lazy — trampoline reduces pairs before SHRINK fires)
    Term X = thvm_op(ctx, UOP_SHRINK, X_all, pairs_x);
    Term Y = thvm_op(ctx, UOP_SHRINK, Y_all, pairs_y);

    // Forward: h = relu(X @ W1 + B1), out = h @ W2 + B2
    Term z1  = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, X, W1),
                       thvm_expand(ctx, B1, SHAPE(batch_size, hidden_dim)));
    Term h   = thvm_op(ctx, UOP_RELU, z1, term_era());
    Term out = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, W2),
                       thvm_expand(ctx, B2, SHAPE(batch_size, 10)));

    // Cross-entropy loss: -mean(sum(Y * log(softmax(out)), axis=1))
    // Softmax: max → sub → exp → sum → div
    Term x_max = thvm_op(ctx, UOP_RMAX, out, term_era());         // [BS,1]
    Term shifted = thvm_op(ctx, UOP_SUB, out,
                   thvm_expand(ctx, x_max, SHAPE(batch_size, 10)));
    Term e = thvm_op(ctx, UOP_EXP, shifted, term_era());          // [BS,10]
    Term e_sum = thvm_op(ctx, UOP_SUM, e, term_era());            // [BS,1]
    Term probs = thvm_op(ctx, UOP_DIV, e,
                 thvm_expand(ctx, e_sum, SHAPE(batch_size, 10)));
    // Clamped log
    f32 eps = 1e-7f;
    Term clamped = thvm_op(ctx, UOP_MAX, probs,
                   thvm_expand(ctx, thvm_tensor(ctx, &eps, SHAPE(1,1)), SHAPE(batch_size, 10)));
    Term log_probs = thvm_op(ctx, UOP_LOG, clamped, term_era());
    // Masked sum
    Term masked = thvm_op(ctx, UOP_MUL, Y, log_probs);            // one-hot × log_probs
    Term sum_c  = thvm_op(ctx, UOP_SUM, masked, term_era());      // sum classes → [BS,1]
    Term sum_b  = thvm_op(ctx, UOP_SUM, sum_c, term_era());       // sum batch  → [1,1]
    Term loss   = thvm_op(ctx, UOP_MUL, thvm_op(ctx, UOP_NEG, sum_b, term_era()), INV_N);
    loss = thvm_reshape(ctx, loss, SHAPE(1));

    // Gradients
    Term gW1 = thvm_grad(ctx, loss, W1);
    Term gB1 = thvm_grad(ctx, loss, B1);
    Term gW2 = thvm_grad(ctx, loss, W2);
    Term gB2 = thvm_grad(ctx, loss, B2);

    // Log + Assigns
    Term log_loss = thvm_log_print(ctx, loss);
    Term aW1 = thvm_assign(ctx, W1,
        thvm_op(ctx, UOP_SUB, W1, thvm_op(ctx, UOP_MUL, LR, gW1)));
    Term aB1 = thvm_assign(ctx, B1,
        thvm_op(ctx, UOP_SUB, B1, thvm_op(ctx, UOP_MUL, LR, gB1)));
    Term aW2 = thvm_assign(ctx, W2,
        thvm_op(ctx, UOP_SUB, W2, thvm_op(ctx, UOP_MUL, LR, gW2)));
    Term aB2 = thvm_assign(ctx, B2,
        thvm_op(ctx, UOP_SUB, B2, thvm_op(ctx, UOP_MUL, LR, gB2)));

    // Recursive call + APP-TEN chain
    Term rec = thvm_app(ctx, thvm_ref(ctx, train_id), var_m);
    Term chain = thvm_app(ctx, log_loss,
                 thvm_app(ctx, aW1,
                 thvm_app(ctx, aB1,
                 thvm_app(ctx, aW2,
                 thvm_app(ctx, aB2, rec)))));

    heap_set(ctx, lm + 1, chain);
    Term succ_lam = term_new(TAG_LAM, 0, lm);

    heap_set(ctx, ln + 1, thvm_ifz(ctx, var_n, term_era(), succ_lam));
    Term def_body = term_new(TAG_LAM, 0, ln);
    ctx->defs[train_id] = def_body;

    // Entry: APP(REF(train), TEN(n_steps))
    f32 nf = (f32)n_steps;
    Term counter = thvm_tensor(ctx, &nf, SHAPE(1));
    return thvm_app(ctx, thvm_ref(ctx, train_id), counter);
}
