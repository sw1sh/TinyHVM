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
