// test_tiny_linear_sgd_loop.m — functional recursive SGD with ASSIGN-based param updates
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Split a term into N copies via DUP chain
static void dup_n(TinyHVM *ctx, Term t, Term *out, u32 n) {
    assert(n >= 1);
    if (n == 1) { out[0] = t; return; }
    Term rest;
    thvm_dup(ctx, thvm_fresh_label(ctx), t, &out[0], &rest);
    dup_n(ctx, rest, out + 1, n - 1);
}

// Build the training loop as a recursive IC program.
// Parameters w, b are persistent tensor slots updated via ASSIGN.
// GRAD targets always match because w, b have fixed tensor IDs.
//
// train := λcounter.
//   IFZ(counter,
//       CTR(w, b),                          // base case
//       λm.
//           loss = MSE(x @ w + b, y)
//           grads = GRAD_KEEP(loss, [w, b])  // → CTR(gw, gb)
//           MAT grads λgw. λgb.
//               seq(ASSIGN(w, w - lr*gw),    // update w in place
//               seq(ASSIGN(b, b - lr*gb),    // update b in place
//               train(m)))                   // recurse

static Term tiny_linear_sgd_loop(TinyHVM *ctx,
                                 Term w, Term b,
                                 Term x, Term y,
                                 Term lr,
                                 u32 n_steps) {
    u32 train_id = ctx->def_count++;
    assert(train_id < 256);

    u64 l_counter = heap_alloc(ctx, 2);
    Term counter_var = term_new(TAG_VAR, 0, l_counter);
    heap_set(ctx, l_counter + 0, term_set_sub(counter_var));

    u64 l_next = heap_alloc(ctx, 2);
    Term next_counter = term_new(TAG_VAR, 0, l_next);
    heap_set(ctx, l_next + 0, term_set_sub(next_counter));

    // w is used in: forward, grad_target, sgd_update_src, assign_dst, base_case
    // b is used in: forward, grad_target, sgd_update_src, assign_dst, base_case
    Term ws[5]; dup_n(ctx, w, ws, 5);
    Term bs[5]; dup_n(ctx, b, bs, 5);
    // ws[0]=forward, ws[1]=grad_target, ws[2]=sgd_src, ws[3]=assign_dst, ws[4]=base

    // Forward: pred = x @ w + b
    Term pred = thvm_op(ctx, UOP_ADD,
        thvm_mm(ctx, x, ws[0]),
        thvm_expand(ctx, thvm_reshape(ctx, bs[0], SHAPE(1, 4)), SHAPE(2, 4)));

    // Mean squared error
    Term diff = thvm_op(ctx, UOP_SUB, pred, y);
    Term sq = thvm_op(ctx, UOP_MUL, diff, diff);
    Term loss = thvm_sum_axes(ctx, sq, (u32[]){0, 1}, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1, 1));
    Term inv_n = thvm_reshape(ctx, thvm_scalar(ctx, 1.0f / 8.0f), SHAPE(1, 1));
    loss = thvm_op(ctx, UOP_MUL, loss, inv_n);

    // Backward: GRAD_KEEP(loss, [w, b]) → CTR(gw, gb)
    Term params[] = {ws[1], bs[1]};
    Term grads = thvm_grad_multi_keep(ctx, loss, params, 2);

    // MAT destructs CTR(gw, gb) → apply λgw. λgb. ...
    u64 l_gw = heap_alloc(ctx, 2);
    Term gw_var = term_new(TAG_VAR, 0, l_gw);
    heap_set(ctx, l_gw + 0, term_set_sub(gw_var));
    thvm_hint_shape(ctx, gw_var, SHAPE(3, 4));

    u64 l_gb = heap_alloc(ctx, 2);
    Term gb_var = term_new(TAG_VAR, 0, l_gb);
    heap_set(ctx, l_gb + 0, term_set_sub(gb_var));
    thvm_hint_shape(ctx, gb_var, SHAPE(4));

    // SGD update via ASSIGN: ASSIGN(dst, src) returns dst tensor in phase 3
    Term lr_w = thvm_expand(ctx, thvm_reshape(ctx, lr, SHAPE(1, 1)), SHAPE(3, 4));
    Term lr_b = thvm_expand(ctx, lr, SHAPE(4));
    Term update_w = thvm_op(ctx, UOP_SUB, ws[2], thvm_op(ctx, UOP_MUL, lr_w, gw_var));
    Term update_b = thvm_op(ctx, UOP_SUB, bs[2], thvm_op(ctx, UOP_MUL, lr_b, gb_var));
    Term assign_w = thvm_assign(ctx, ws[3], update_w);
    Term assign_b = thvm_assign(ctx, bs[3], update_b);

    // Recursive call: train(m)(assign_w)(assign_b)
    // ASSIGN returns dst tensor (TAG_TEN) in phase 3.
    // Passing ASSIGN results as args forces them to reduce before the body fires.
    // In phase 1: ASSIGN is WNF (TAG_TOP) — blocks APP from fully reducing.
    // In phase 3: ASSIGN fires → TAG_TEN → APP-LAM substitutes → body continues.
    // The lambda body ignores these args (uses same w,b slots).
    Term rec = thvm_app(ctx,
               thvm_app(ctx,
               thvm_app(ctx, thvm_ref(ctx, train_id), next_counter),
                        assign_w),
                        assign_b);
    // train now takes 3 args: λcounter. λ_aw. λ_ab. IFZ(...)
    // _aw and _ab are dummy — they force ASSIGN evaluation via APP reduction.

    // Wire into gradient bundle match
    heap_set(ctx, l_gb + 1, rec);
    Term lam_gb = term_new(TAG_LAM, 0, l_gb);
    heap_set(ctx, l_gw + 1, lam_gb);
    Term lam_gw = term_new(TAG_LAM, 0, l_gw);

    Term grad_bundle_match = thvm_mat(ctx, 2, lam_gw, term_era());
    Term succ_body = thvm_app(ctx, grad_bundle_match, grads);
    heap_set(ctx, l_next + 1, succ_body);
    Term succ_lam = term_new(TAG_LAM, 0, l_next);

    // Base case: return final param values
    Term done = thvm_ctr(ctx, (Term[]){ws[4], bs[4]}, 2);
    Term ifz = thvm_ifz(ctx, counter_var, done, succ_lam);

    // train takes 3 args: λcounter. λ_aw. λ_ab. IFZ(counter, done, succ)
    // _aw, _ab are dummy args that force ASSIGN evaluation before body fires.
    u64 l_aw = heap_alloc(ctx, 2);
    heap_set(ctx, l_aw + 0, term_set_sub(term_new(TAG_VAR, 0, l_aw)));
    heap_set(ctx, l_aw + 1, ifz);
    u64 l_ab = heap_alloc(ctx, 2);
    heap_set(ctx, l_ab + 0, term_set_sub(term_new(TAG_VAR, 0, l_ab)));
    heap_set(ctx, l_ab + 1, term_new(TAG_LAM, 0, l_aw));
    heap_set(ctx, l_counter + 1, term_new(TAG_LAM, 0, l_ab));
    ctx->defs[train_id] = term_new(TAG_LAM, 0, l_counter);

    // Initial call: train(n_steps)(ERA)(ERA) — no ASSIGNs on first entry
    return thvm_app(ctx,
           thvm_app(ctx,
           thvm_app(ctx, thvm_ref(ctx, train_id), thvm_scalar_u32(ctx, n_steps)),
                    term_era()),
                    term_era());
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    const char *backend = getenv("THVM_TEST_BACKEND");
    if (!backend || !backend[0]) backend = argc > 1 ? argv[1] : "metal";
    u32 n_steps = 2;
    const char *steps_env = getenv("THVM_TRAIN_STEPS");
    if (steps_env && steps_env[0]) n_steps = (u32)strtoul(steps_env, NULL, 10);
    else if (argc > 2) n_steps = (u32)strtoul(argv[2], NULL, 10);
#ifdef __APPLE__
    if (strcmp(backend, "metal") == 0 && !MTLCreateSystemDefaultDevice()) {
        fprintf(stderr, "SKIP: metal unavailable\n");
        return 0;
    }
#endif
    TinyHVM *ctx = thvm_init(backend);

    f32 xd[] = {
        1.0f,  2.0f, -1.0f,
        0.5f, -1.5f,  2.0f,
    };
    f32 yd[] = {
        0.6f, -0.2f,  0.1f,  0.9f,
       -0.3f,  0.7f, -0.5f,  0.2f,
    };
    f32 wd[] = {
         0.10f, -0.20f,  0.30f,  0.40f,
        -0.50f,  0.60f, -0.70f,  0.80f,
         0.90f, -1.00f,  1.10f, -1.20f,
    };
    f32 bd[] = {0.05f, -0.10f, 0.15f, 0.20f};
    f32 lr_val = 0.05f;

    Term x  = thvm_tensor(ctx, xd,  (Shape){.dims={2,3}, .rank=2});
    Term y  = thvm_tensor(ctx, yd,  (Shape){.dims={2,4}, .rank=2});
    Term w  = thvm_tensor(ctx, wd,  (Shape){.dims={3,4}, .rank=2});
    Term b0 = thvm_tensor(ctx, bd,  (Shape){.dims={4},   .rank=1});
    Term lr = thvm_scalar(ctx, lr_val);

    thvm_set_requires_grad(ctx, w);
    thvm_set_requires_grad(ctx, b0);

    Term program = tiny_linear_sgd_loop(ctx, w, b0, x, y, lr, n_steps);
    Term result = thvm_eval(ctx, program);

    printf("result_tag=%u ext=%u val=%llu\n",
           (u32)term_tag(result), (u32)term_ext(result), (unsigned long long)term_val(result));

    if (term_tag(result) == TAG_CTR) {
        u64 rloc = term_val(result);
        for (u32 i = 0; i < 2; i++) {
            Term child = heap_read(ctx, rloc + i);
            printf("result[%u] tag=%u ext=%u val=%llu\n",
                   i, (u32)term_tag(child), (u32)term_ext(child), (unsigned long long)term_val(child));
        }
    }

    // Read final parameter values directly from the tensor slots
    u32 w_dtype = DTYPE_F32, b_dtype = DTYPE_F32;
    Shape w_shape = SHAPE(1), b_shape = SHAPE(1);
    void *w_raw = thvm_to_host_raw(ctx, w, &w_dtype, &w_shape);
    void *b_raw = thvm_to_host_raw(ctx, b0, &b_dtype, &b_shape);

    f32 *wf = (w_raw && w_dtype == DTYPE_F32) ? (f32 *)w_raw : NULL;
    f32 *bf = (b_raw && b_dtype == DTYPE_F32) ? (f32 *)b_raw : NULL;
    if (wf) {
        printf("final_w = [");
        for (int i = 0; i < 12; i++) printf("%.4f%s", wf[i], i == 11 ? "" : ", ");
        printf("]\n");
    }
    if (bf) {
        printf("final_b = [%.4f, %.4f, %.4f, %.4f]\n", bf[0], bf[1], bf[2], bf[3]);
    }

    thvm_free(ctx);
    return 0;
}
