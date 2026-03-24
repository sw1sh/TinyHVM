Term thvm_grad(TinyHVM *ctx, Term y, Term x) {
    // Fully lazy: store unreduced y. The GRAD handler will reduce y
    // when it processes this term. Seed is scalar ones — the loss is
    // always reduced to a scalar before calling thvm_grad.
    f32 one = 1.0f;
    Term seed = thvm_tensor(ctx, &one, SHAPE(1));
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     y);     // unreduced — lazy
    heap_set(ctx, loc + 1, seed);
    heap_set(ctx, loc + 2, x);
    return term_new(TAG_TOP, UOP_GRAD, loc);
}

void thvm_backward(TinyHVM *ctx, Term loss, Term *params, Term *grads, u32 n_params) {
    // DUP: create independent gradient terms for each param
    // Each grads[p] is a lazy UOP_GRAD — reduce it to compute ∂loss/∂params[p]
    for (u32 p = 0; p < n_params; p++)
        grads[p] = thvm_grad(ctx, loss, params[p]);
}





// ============================================================
// thvm_train_step — N training steps as one lazy inet program
// ============================================================
//
// Builds the full N-step training as a single term:
//   ASSIGN(W, W - LR * GRAD(loss, W))  chained N times.
// ONE thvm_reduce(program) drives everything.

void thvm_train_step(TinyHVM *ctx,
                     Term *pW1, Term *pB1, Term *pW2, Term *pB2,
                     Term X, Term Y, Term LR) {
    Term W1 = *pW1, B1 = *pB1, W2 = *pW2, B2 = *pB2;

    // -- Forward: 2-layer MLP (lazy) --
    Term z1  = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, X, W1),
                       thvm_expand(ctx, B1, SHAPE(4, 4)));
    Term h   = thvm_op(ctx, UOP_RELU, z1, term_era());
    Term out = thvm_op(ctx, UOP_ADD, thvm_op(ctx, UOP_MM, h, W2),
                       thvm_expand(ctx, B2, SHAPE(4, 1)));

    // -- MSE Loss (lazy) --
    Term diff = thvm_op(ctx, UOP_SUB, out, Y);
    Term sq   = thvm_op(ctx, UOP_MUL, diff, diff);
    u32 axes[] = {0, 1};
    Term loss = thvm_sum_axes(ctx, sq, axes, 2);
    f32 inv_n = 1.0f / 4.0f;
    loss = thvm_op(ctx, UOP_MUL, loss, thvm_tensor(ctx, &inv_n, SHAPE(1)));
    loss = thvm_reshape(ctx, loss, SHAPE(1)); // ensure scalar [1] for GRAD seed

    // -- Gradients (thvm_grad reduces loss to get seed shape, returns lazy GRAD) --
    Term gW1 = thvm_grad(ctx, loss, W1);
    Term gB1 = thvm_grad(ctx, loss, B1);
    Term gW2 = thvm_grad(ctx, loss, W2);
    Term gB2 = thvm_grad(ctx, loss, B2);

    // -- SGD: W' = ASSIGN(W, W - LR * grad) — all lazy --
    *pW1 = thvm_assign(ctx, W1, thvm_op(ctx, UOP_SUB, W1, thvm_op(ctx, UOP_MUL, LR, gW1)));
    *pB1 = thvm_assign(ctx, B1, thvm_op(ctx, UOP_SUB, B1, thvm_op(ctx, UOP_MUL, LR, gB1)));
    *pW2 = thvm_assign(ctx, W2, thvm_op(ctx, UOP_SUB, W2, thvm_op(ctx, UOP_MUL, LR, gW2)));
    *pB2 = thvm_assign(ctx, B2, thvm_op(ctx, UOP_SUB, B2, thvm_op(ctx, UOP_MUL, LR, gB2)));
}

// ============================================================
// Profiling — dispatch to backend
// ============================================================

void thvm_profile_report(TinyHVM *ctx) {
    if (ctx->backend && ctx->backend->profile_report)
        ctx->backend->profile_report();
}

void thvm_profile_reset(TinyHVM *ctx) {
    if (ctx->backend && ctx->backend->profile_reset)
        ctx->backend->profile_reset();
}


// ============================================================
// Eval helpers — argmax + accuracy
// ============================================================

Term thvm_argmax(TinyHVM *ctx, Term x, u32 rows, u32 cols) {
    // Force-evaluation point: argmax reads actual data. thvm_reduce is idempotent
    // on TAG_TEN (returns immediately via WNF check), so unconditional call is fine.
    x = thvm_reduce(ctx, x);
    f32 *data = thvm_to_host(ctx, x);

    u32 *preds = malloc(rows * sizeof(u32));
    for (u32 i = 0; i < rows; i++) {
        u32 best = 0;
        f32 mv = data[i * cols];
        for (u32 j = 1; j < cols; j++) {
            if (data[i * cols + j] > mv) {
                mv = data[i * cols + j];
                best = j;
            }
        }
        preds[i] = best;
    }

    // Store as u32 tensor
    u32 id = ctx->tensor_count++;
    u32 buf = ctx->backend->buf_alloc(rows * sizeof(u32));
    ctx->tensors[id] = (TensorMeta){
        .buf_id = buf, .dtype = DTYPE_U32,
        .view = view_create(SHAPE(rows)),
    };
    ctx->backend->buf_write(buf, preds, rows * sizeof(u32));
    free(preds);
    return term_ten(id, DTYPE_U32);
}

f32 thvm_eval_accuracy(TinyHVM *ctx, Term logits, const u8 *labels,
                       u32 n_samples, u32 n_classes) {
    logits = thvm_reduce(ctx, logits);
    f32 *data = thvm_to_host(ctx, logits);

    u32 correct = 0;
    for (u32 i = 0; i < n_samples; i++) {
        u32 best = 0;
        f32 mv = data[i * n_classes];
        for (u32 j = 1; j < n_classes; j++) {
            if (data[i * n_classes + j] > mv) {
                mv = data[i * n_classes + j];
                best = j;
            }
        }
        if (best == labels[i]) correct++;
    }
    return 100.0f * (f32)correct / (f32)n_samples;
}


// ============================================================
// inet.c — Interaction combinator API (Phase 2+3)
// ============================================================

// TAG_LAM: λx.body  where heap[loc]=var_placeholder, heap[loc+1]=body
// Returns a LAM term. *var_out is a TAG_VAR pointing to the var slot.
Term thvm_lam(TinyHVM *ctx, Term *var_out, Term body) {
    u64 loc = heap_alloc(ctx, 2);
    Term var = term_new(TAG_VAR, 0, loc);
    heap_set(ctx, loc,     term_set_sub(var));  // sub flag: unbound
    heap_set(ctx, loc + 1, body);
    if (var_out) *var_out = var;
    return term_new(TAG_LAM, 0, loc);
}

// TAG_APP: (fun arg)
Term thvm_app(TinyHVM *ctx, Term fun, Term arg) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     fun);
    heap_set(ctx, loc + 1, arg);
    return term_new(TAG_APP, 0, loc);
}

// TAG_SUP: {a, b} superposition
Term thvm_sup(TinyHVM *ctx, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     a);
    heap_set(ctx, loc + 1, b);
    return term_new(TAG_SUP, 0, loc);
}

// Register a global definition; returns its name id for TAG_REF.
u32 thvm_define(TinyHVM *ctx, Term body) {
    assert(ctx->def_count < 256);
    u32 name = ctx->def_count++;
    ctx->defs[name] = body;
    return name;
}

// TAG_REF: reference to named def `name`
Term thvm_ref(TinyHVM *ctx, u32 name) {
    return term_new(TAG_REF, name, 0);
}

// UOP_WHERE: elementwise ternary select — WHERE(cond_ten, then_ten, else_ten)
// Per tinyspec: result[i] = then[i] if cond[i] != 0 else else[i]
Term thvm_where(TinyHVM *ctx, Term cond, Term then_t, Term else_t) {
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     cond);
    heap_set(ctx, loc + 1, then_t);
    heap_set(ctx, loc + 2, else_t);
    return term_new(TAG_TOP, UOP_WHERE, loc);
}

// UOP_ASSIGN: in-place weight update
Term thvm_assign(TinyHVM *ctx, Term dst, Term src) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     dst);
    heap_set(ctx, loc + 1, src);
    return term_new(TAG_TOP, UOP_ASSIGN, loc);
}
