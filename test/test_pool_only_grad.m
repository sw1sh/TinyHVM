// test_pool_only_grad.m — Minimal pool-only gradient isolation test.
// Does NOT go through conv — just:
//   pool(x) → [1,1,2,2,2,2]  (thvm_pool)
//   RMAX    → [1,1,2,2,2,1]  reshape → [1,1,2,2,2]
//   RMAX    → [1,1,2,2,1]    reshape → [1,1,2,2]
//   sum     → loss
//   thvm_grad(loss, x) vs finite-difference

#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>

// Build the forward chain for a given x and return the scalar loss term.
// x must be [1,1,4,4].
static Term build_loss(TinyHVM *ctx, Term x) {
    u32 k[] = {2, 2};
    u32 s[] = {2, 2};

    // pool_t: [1,1,2,2,2,2]
    Term pool_t = thvm_pool(ctx, x, k, s, 2);

    // r1 = RMAX over last dim (KW=2) → [1,1,2,2,2,1], reshape → [1,1,2,2,2]
    Term r1 = thvm_op(ctx, UOP_RMAX, pool_t, term_era());
    r1 = thvm_reshape(ctx, r1, shape_of((u32[]){1, 1, 2, 2, 2}, 5));

    // r2 = RMAX over last dim (KH=2) → [1,1,2,2,1], reshape → [1,1,2,2]
    Term r2 = thvm_op(ctx, UOP_RMAX, r1, term_era());
    r2 = thvm_reshape(ctx, r2, shape_of((u32[]){1, 1, 2, 2}, 4));

    // loss = sum of all elements
    u32 axes[] = {0, 1, 2, 3};
    Term loss = thvm_reshape(ctx, thvm_sum_axes(ctx, r2, axes, 4), SHAPE(1));
    return loss;
}

int main(void) {
    TinyHVM *ctx = thvm_init(thvm_device("metal"));

    // Input: [1,1,4,4] = 16 elements with values 1..16
    f32 x_data[16];
    for (int i = 0; i < 16; i++) x_data[i] = (f32)(i + 1);

    // ---- Analytic gradient ----
    Term x = thvm_tensor(ctx, x_data, (Shape){.dims = {1, 1, 4, 4}, .rank = 4});
    thvm_set_requires_grad(ctx, x);

    Term loss = build_loss(ctx, x);

    // Forward value check
    f32 loss_fwd = thvm_to_host(ctx, thvm_reduce(ctx, loss))[0];
    printf("forward loss = %.6f\n", loss_fwd);

    // Pool output (materialised for inspection)
    {
        // Re-derive pool_t from scratch for printing — ctx still has the terms
        // but we just print a few values from the loss we already reduced.
        // The maxpool of x[1,1,4,4] with k=s=2:
        // top-left 2x2: max(1,2,5,6)=6
        // top-right 2x2: max(3,4,7,8)=8
        // bot-left 2x2: max(9,10,13,14)=14
        // bot-right 2x2: max(11,12,15,16)=16
        // Expected loss = 6+8+14+16 = 44
        printf("expected loss = 44.0 (maxpool 2x2 of 1..16)\n");
    }

    printf("computing analytic grad...\n"); fflush(stdout);
    Term grad_x = thvm_grad(ctx, loss, x);
    Term gx_r   = thvm_reduce(ctx, grad_x);

    f32 *gx_raw = thvm_to_host(ctx, gx_r);
    f32 gx[16];
    if (gx_raw) {
        memcpy(gx, gx_raw, 16 * sizeof(f32));
    } else {
        printf("ERROR: gx_raw is NULL\n");
        memset(gx, 0, sizeof(gx));
    }

    printf("analytic grad (row-major [1,1,4,4]):\n");
    for (int i = 0; i < 16; i++) {
        printf("  gx[%2d] = %8.5f   (x=%2.0f)\n", i, gx[i], x_data[i]);
    }

    // ---- Numerical gradient (finite differences) ----
    printf("\nnumerical gradient (eps=1e-3):\n");
    f32 eps = 1e-3f;
    f32 gx_num[16];
    f32 max_abs_diff = 0.0f;
    int n_wrong = 0;

    for (int i = 0; i < 16; i++) {
        // f(x + eps)
        f32 saved = x_data[i];
        x_data[i] = saved + eps;
        thvm_reset(ctx, 0);
        Term xp = thvm_tensor(ctx, x_data, (Shape){.dims = {1, 1, 4, 4}, .rank = 4});
        Term lp_t = thvm_reduce(ctx, build_loss(ctx, xp));
        f32 lp = thvm_to_host(ctx, lp_t)[0];

        // f(x - eps)
        x_data[i] = saved - eps;
        thvm_reset(ctx, 0);
        Term xm = thvm_tensor(ctx, x_data, (Shape){.dims = {1, 1, 4, 4}, .rank = 4});
        Term lm_t = thvm_reduce(ctx, build_loss(ctx, xm));
        f32 lm = thvm_to_host(ctx, lm_t)[0];

        x_data[i] = saved;
        gx_num[i] = (lp - lm) / (2.0f * eps);

        f32 diff = fabsf(gx[i] - gx_num[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        int wrong = (diff > 0.1f);
        if (wrong) n_wrong++;
        printf("  gx[%2d]: analytic=%8.5f  numerical=%8.5f  diff=%.2e%s\n",
               i, gx[i], gx_num[i], diff, wrong ? " *** WRONG" : "");
    }

    printf("\nmax_abs_diff = %.6e\n", max_abs_diff);
    printf("wrong elements: %d / 16\n", n_wrong);

    // Provide expected gradient for reference:
    // For maxpool 2x2 of 1..16:
    //   Pool windows (values → winner position):
    //     top-left [1,2,5,6] → winner=6 at flat index 5 (row1,col1) → gx[5]=1
    //     top-right [3,4,7,8] → winner=8 at flat index 7 (row1,col3) → gx[7]=1
    //     bot-left [9,10,13,14] → winner=14 at flat index 13 (row3,col1) → gx[13]=1
    //     bot-right [11,12,15,16] → winner=16 at flat index 15 (row3,col3) → gx[15]=1
    // All other elements should have gradient 0.
    printf("\nexpected: gx[5]=1 gx[7]=1 gx[13]=1 gx[15]=1, all others 0\n");

    int pass = (max_abs_diff < 0.05f);
    printf("\n%s\n", pass ? "PASS" : "FAIL");

    thvm_free(ctx);
    return pass ? 0 : 1;
}
