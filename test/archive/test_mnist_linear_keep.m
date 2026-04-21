// test_mnist_linear_keep.m — MNIST linear using keep-mode grad (bundle CTR
// + MAT-CTR destructure), avoiding slot-mode grad bugs.
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include "../src/nn/datasets.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

int main(int argc, char **argv) {
    srand(42);
    const char *backend = (argc > 1) ? argv[1] : "metal";
    MNISTData data = mnist_load("data");
    TinyHVM *ctx = thvm_init(backend);
    u32 BS = 256, nb = data.n_train / BS, n_in = 28*28;

    f32 *wd = malloc(n_in*10*4);
    f32 k = 1.f / sqrtf((f32)n_in);
    for (u32 i = 0; i < n_in*10; i++) wd[i] = k*((f32)rand()/(f32)RAND_MAX*2-1);
    Term W = thvm_tensor(ctx, wd, SHAPE(n_in, 10));
    free(wd);
    f32 bz[10] = {0};
    Term B = thvm_tensor(ctx, bz, SHAPE(10));
    thvm_set_requires_grad(ctx, W);
    thvm_set_requires_grad(ctx, B);

    f32 inv255 = 1.f/255.f;
    Term norm = thvm_tensor(ctx, &inv255, SHAPE(1,1));
    Term td = thvm_tensor(ctx, data.train_images,
        (Shape){.dims={data.n_train,1,28,28}, .rank=4});
    u32 nw = ctx->tensor_count;
    printf("=== MNIST Linear keep-mode (BS=%u) ===\n\n", BS);
    double t0 = now_s();
    u32 step = 0;

    for (u32 ep = 0; ep < 3; ep++) {
        f32 lr = 0.1f * powf(0.7f, (f32)ep);
        for (u32 bi = 0; bi < nb; bi++) {
          @autoreleasepool {
            // DUP W and B for (fwd + grad_target + assign_dst).
            Term W_fwd, W_g, W_ass;
            { Term t1;
              thvm_dup(ctx, thvm_fresh_label(ctx), W, &W_fwd, &t1);
              thvm_dup(ctx, thvm_fresh_label(ctx), t1, &W_g, &W_ass);
            }
            Term B_fwd, B_g, B_ass;
            { Term t1;
              thvm_dup(ctx, thvm_fresh_label(ctx), B, &B_fwd, &t1);
              thvm_dup(ctx, thvm_fresh_label(ctx), t1, &B_g, &B_ass);
            }

            Term x = thvm_shrink(ctx, td, (u32[]){bi*BS,(bi+1)*BS,0,1,0,28,0,28}, 4);
            x = thvm_reshape(ctx, x, SHAPE(BS, n_in));
            x = thvm_op(ctx, UOP_MUL, x, thvm_expand(ctx, norm, SHAPE(BS, n_in)));
            Term lo = thvm_op(ctx, UOP_ADD,
                thvm_mm(ctx, x, W_fwd),
                thvm_expand(ctx, thvm_reshape(ctx, B_fwd, SHAPE(1,10)), SHAPE(BS,10)));
            Term loss = cross_entropy_loss(ctx, lo, &data.train_labels[bi*BS], BS, 10);

            Term params[] = {W_g, B_g};
            Term grads = thvm_grad_multi_keep(ctx, loss, params, 2);

            // MAT-CTR: bind gW, gB, then SGD update via ASSIGN.
            u64 l_gW = heap_alloc(ctx, 2);
            Term gW_var = term_new(TAG_VAR, 0, l_gW);
            heap_set(ctx, l_gW + 0, term_set_sub(gW_var));
            thvm_hint_shape(ctx, gW_var, SHAPE(n_in, 10));
            u64 l_gB = heap_alloc(ctx, 2);
            Term gB_var = term_new(TAG_VAR, 0, l_gB);
            heap_set(ctx, l_gB + 0, term_set_sub(gB_var));
            thvm_hint_shape(ctx, gB_var, SHAPE(10));

            Term lrt = thvm_tensor(ctx, &lr, SHAPE(1));
            Term lrt0, lrt1; thvm_dup(ctx, thvm_fresh_label(ctx), lrt, &lrt0, &lrt1);
            Term lr_W = thvm_expand(ctx, thvm_reshape(ctx, lrt0, SHAPE(1,1)), SHAPE(n_in, 10));
            Term lr_B = thvm_expand(ctx, lrt1, SHAPE(10));
            Term new_W = thvm_op(ctx, UOP_SUB, W_ass, thvm_op(ctx, UOP_MUL, lr_W, gW_var));
            Term new_B = thvm_op(ctx, UOP_SUB, B_ass, thvm_op(ctx, UOP_MUL, lr_B, gB_var));
            Term aW = thvm_assign(ctx, W_ass, new_W);
            Term aB = thvm_assign(ctx, B_ass, new_B);
            // SEQ fires `effect` as side-effect, returns `continuation`.
            // For both ASSIGNs to fire as effects, chain them so each is
            // the `effect` slot of a SEQ.
            Term body = thvm_seq(ctx, aW, thvm_seq(ctx, aB, term_era()));

            heap_set(ctx, l_gB + 1, body);
            Term lam_gB = term_new(TAG_LAM, 0, l_gB);
            heap_set(ctx, l_gW + 1, lam_gB);
            Term lam_gW = term_new(TAG_LAM, 0, l_gW);
            Term mat = thvm_mat(ctx, 2, lam_gW, term_era());
            Term prog = thvm_app(ctx, mat, grads);

            thvm_eval(ctx, prog);

            extern u32 total_dispatches; total_dispatches = 0;
            // Free host caches so next readback pulls fresh post-ASSIGN.
            for (Term p = W; ; p = B) {
                u32 pid = (u32)term_val(p);
                if (ctx->tensors[pid].host_ptr) { free(ctx->tensors[pid].host_ptr); ctx->tensors[pid].host_ptr = NULL; }
                if (p == B) break;
            }
            if (getenv("MNK_TRACE") && (step < 5 || step == 50 || step == 100 || step == 233)) {
                u32 wid = (u32)term_val(W);
                u32 bid = (u32)term_val(B);
                TensorMeta *mW = &ctx->tensors[wid];
                TensorMeta *mB = &ctx->tensors[bid];
                f32 wbuf[6], bbuf[10];
                if (mW->backend && mW->buf_id) mW->backend->buf_read(mW->buf_id, wbuf, sizeof(wbuf));
                if (mB->backend && mB->buf_id) mB->backend->buf_read(mB->buf_id, bbuf, sizeof(bbuf));
                fprintf(stderr, "  step %u W[0..2]=%.9f,%.9f,%.9f B[0..2]=%.9f,%.9f,%.9f\n",
                    step, wbuf[0], wbuf[1], wbuf[2], bbuf[0], bbuf[1], bbuf[2]);
            }
            thvm_reset(ctx, nw);
            step++;
          }
        }

        // Eval
        Term test_data = thvm_tensor(ctx, data.test_images,
            (Shape){.dims={data.n_test,1,28,28}, .rank=4});
        u32 ek = ctx->tensor_count;
        u32 cor = 0, tbs = 256, tb = data.n_test / tbs;
        for (u32 b2 = 0; b2 < tb; b2++) {
            Term x = thvm_shrink(ctx, test_data, (u32[]){b2*tbs,(b2+1)*tbs,0,1,0,28,0,28}, 4);
            x = thvm_reshape(ctx, x, SHAPE(tbs, n_in));
            x = thvm_op(ctx, UOP_MUL, x, thvm_expand(ctx, norm, SHAPE(tbs, n_in)));
            Term lo = thvm_op(ctx, UOP_ADD,
                thvm_mm(ctx, x, W),
                thvm_expand(ctx, thvm_reshape(ctx, B, SHAPE(1,10)), SHAPE(tbs,10)));
            f32 acc = thvm_eval_accuracy(ctx, lo, &data.test_labels[b2*tbs], tbs, 10);
            cor += (u32)(acc*(f32)tbs/100.f);
            thvm_reset(ctx, ek);
        }
        f32 acc = 100.f*(f32)cor/(f32)(tb*tbs);
        double el = now_s() - t0;
        printf("  Epoch %u: %.1f%% in %.1fs (lr=%.4f, %.1f ms/step)\n",
            ep, acc, el, lr, el*1000/step);
    }
    thvm_free(ctx);
    return 0;
}
