// test_tiny_linear_sgd_loop.m — functional recursive SGD on a tiny linear layer
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void split3(TinyHVM *ctx, Term z, Term *a, Term *b, Term *c) {
    Term tail;
    thvm_dup(ctx, thvm_fresh_label(ctx), z, a, &tail);
    thvm_dup(ctx, thvm_fresh_label(ctx), tail, b, c);
}

static Term tiny_linear_sgd_loop(TinyHVM *ctx,
                                 Term w0, Term b0,
                                 Term x, Term y,
                                 Term lr,
                                 u32 n_steps) {
    u32 train_id = ctx->def_count++;
    assert(train_id < 256);

    // train := λcounter. λw. λb.
    //   IFZ(counter,
    //       CTR(w, b),
    //       λm. (GRAD_KEEP(loss(w,b), [w,b]) (λgw. λgb. train(m)(w-lr*gw)(b-lr*gb))))
    u64 l_counter = heap_alloc(ctx, 2);
    Term counter_var = term_new(TAG_VAR, 0, l_counter);
    heap_set(ctx, l_counter + 0, term_set_sub(counter_var));

    u64 l_w = heap_alloc(ctx, 2);
    Term w_var = term_new(TAG_VAR, 0, l_w);
    heap_set(ctx, l_w + 0, term_set_sub(w_var));
    thvm_hint_shape(ctx, w_var, SHAPE(3, 4));

    u64 l_b = heap_alloc(ctx, 2);
    Term b_var = term_new(TAG_VAR, 0, l_b);
    heap_set(ctx, l_b + 0, term_set_sub(b_var));
    thvm_hint_shape(ctx, b_var, SHAPE(4));

    u64 l_next = heap_alloc(ctx, 2);
    Term next_counter = term_new(TAG_VAR, 0, l_next);
    heap_set(ctx, l_next + 0, term_set_sub(next_counter));

    Term w_loss, w_sgd, w_done;
    Term b_loss, b_sgd, b_done;
    split3(ctx, w_var, &w_loss, &w_sgd, &w_done);
    split3(ctx, b_var, &b_loss, &b_sgd, &b_done);

    // Forward: pred = x @ w + b
    Term pred = thvm_op(ctx, UOP_ADD,
        thvm_mm(ctx, x, w_loss),
        thvm_expand(ctx, thvm_reshape(ctx, b_loss, SHAPE(1, 4)), SHAPE(2, 4)));

    // Mean squared error over all 8 outputs.
    Term diff = thvm_op(ctx, UOP_SUB, pred, y);
    Term sq = thvm_op(ctx, UOP_MUL, diff, diff);
    Term loss = thvm_sum_axes(ctx, sq, (u32[]){0, 1}, 2);
    loss = thvm_reshape(ctx, loss, SHAPE(1, 1));
    Term inv_n = thvm_reshape(ctx, thvm_scalar(ctx, 1.0f / 8.0f), SHAPE(1, 1));
    loss = thvm_op(ctx, UOP_MUL, loss, inv_n);

    // Pure kept gradients: APP(driver, CTR(grad_w, grad_b)).
    Term params[] = {w_var, b_var};
    Term grads = thvm_grad_multi_keep(ctx, loss, params, 2);

    u64 l_gw = heap_alloc(ctx, 2);
    Term gw_var = term_new(TAG_VAR, 0, l_gw);
    heap_set(ctx, l_gw + 0, term_set_sub(gw_var));
    thvm_hint_shape(ctx, gw_var, SHAPE(3, 4));

    u64 l_gb = heap_alloc(ctx, 2);
    Term gb_var = term_new(TAG_VAR, 0, l_gb);
    heap_set(ctx, l_gb + 0, term_set_sub(gb_var));
    thvm_hint_shape(ctx, gb_var, SHAPE(4));

    Term lr_w = thvm_expand(ctx, thvm_reshape(ctx, lr, SHAPE(1, 1)), SHAPE(3, 4));
    Term lr_b = thvm_expand(ctx, lr, SHAPE(4));
    Term next_w = thvm_op(ctx, UOP_SUB, w_sgd, thvm_op(ctx, UOP_MUL, lr_w, gw_var));
    Term next_b = thvm_op(ctx, UOP_SUB, b_sgd, thvm_op(ctx, UOP_MUL, lr_b, gb_var));
    Term next_w_leaf = thvm_detach(ctx, next_w);
    Term next_b_leaf = thvm_detach(ctx, next_b);

    Term rec = thvm_app(ctx,
               thvm_app(ctx,
               thvm_app(ctx, thvm_ref(ctx, train_id), next_counter),
                        next_w_leaf),
                        next_b_leaf);

    heap_set(ctx, l_gb + 1, rec);
    Term lam_gb = term_new(TAG_LAM, 0, l_gb);
    heap_set(ctx, l_gw + 1, lam_gb);
    Term lam_gw = term_new(TAG_LAM, 0, l_gw);

    Term grad_bundle_match = thvm_mat(ctx, 2, lam_gw, term_era());
    Term succ_body = thvm_app(ctx, grad_bundle_match, grads);
    heap_set(ctx, l_next + 1, succ_body);
    Term succ_lam = term_new(TAG_LAM, 0, l_next);

    Term done = thvm_ctr(ctx, (Term[]){w_done, b_done}, 2);
    heap_set(ctx, l_b + 1, thvm_ifz(ctx, counter_var, done, succ_lam));
    Term lam_b = term_new(TAG_LAM, 0, l_b);
    heap_set(ctx, l_w + 1, lam_b);
    Term lam_w = term_new(TAG_LAM, 0, l_w);
    heap_set(ctx, l_counter + 1, lam_w);
    ctx->defs[train_id] = term_new(TAG_LAM, 0, l_counter);

    return thvm_app(ctx,
           thvm_app(ctx,
           thvm_app(ctx, thvm_ref(ctx, train_id), thvm_scalar_u32(ctx, n_steps)),
                    w0),
                    b0);
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
    Term w0 = thvm_tensor(ctx, wd,  (Shape){.dims={3,4}, .rank=2});
    Term b0 = thvm_tensor(ctx, bd,  (Shape){.dims={4},   .rank=1});
    Term lr = thvm_scalar(ctx, lr_val);

    thvm_set_requires_grad(ctx, w0);
    thvm_set_requires_grad(ctx, b0);

    Term program = tiny_linear_sgd_loop(ctx, w0, b0, x, y, lr, n_steps);
    Term result = thvm_eval(ctx, program);

    if (getenv("THVM_LOOP_DIAG") && term_tag(result) == TAG_CTR) {
        u64 rloc = term_val(result);
        for (u32 i = 0; i < 2; i++) {
            Term child = heap_read(ctx, rloc + i);
            printf("bundle_child[%u]_tag=%u ext=%u val=%llu\n",
                   i, (u32)term_tag(child), (u32)term_ext(child), (unsigned long long)term_val(child));
            if (term_tag(child) == TAG_TOP && term_ext(child) == UOP_DETACH) {
                u64 dloc = term_val(child);
                Term src = heap_read(ctx, dloc + 0);
                printf("bundle_child[%u]_detach_src_tag=%u ext=%u val=%llu\n",
                       i, (u32)term_tag(src), (u32)term_ext(src), (unsigned long long)term_val(src));
            }
        }
    }

    Term final_w_t = thvm_grad_bundle_get(ctx, result, 0);
    Term final_b_t = thvm_grad_bundle_get(ctx, result, 1);
    u32 w_dtype = DTYPE_F32, b_dtype = DTYPE_F32;
    Shape w_shape = SHAPE(1), b_shape = SHAPE(1);
    void *w_raw = thvm_to_host_raw(ctx, final_w_t, &w_dtype, &w_shape);
    void *b_raw = thvm_to_host_raw(ctx, final_b_t, &b_dtype, &b_shape);
    final_w_t = thvm_reduce(ctx, final_w_t);
    final_b_t = thvm_reduce(ctx, final_b_t);

    printf("result_tag=%u ext=%u val=%llu\n",
           (u32)term_tag(result), (u32)term_ext(result), (unsigned long long)term_val(result));
    printf("final_w_tag=%u ext=%u val=%llu\n",
           (u32)term_tag(final_w_t), (u32)term_ext(final_w_t), (unsigned long long)term_val(final_w_t));
    printf("final_b_tag=%u ext=%u val=%llu\n",
           (u32)term_tag(final_b_t), (u32)term_ext(final_b_t), (unsigned long long)term_val(final_b_t));
    f32 *wf = (w_raw && w_dtype == DTYPE_F32) ? (f32 *)w_raw : NULL;
    f32 *bf = (b_raw && b_dtype == DTYPE_F32) ? (f32 *)b_raw : NULL;
    if (wf && bf) {
        printf("final_w = [");
        for (int i = 0; i < 12; i++) printf("%.4f%s", wf[i], i == 11 ? "" : ", ");
        printf("]\n");
        printf("final_b = [%.4f, %.4f, %.4f, %.4f]\n", bf[0], bf[1], bf[2], bf[3]);
    }

    if (term_tag(result) == TAG_CTR) {
        printf("result_bundle = %u\n", (u32)term_ext(result));
    }

    thvm_free(ctx);
    return 0;
}
