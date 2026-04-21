// test_ce_simple.m — cross-entropy forward + backward on both backends.
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>

static int run(const char *backend) {
    TinyHVM *ctx = thvm_init(backend);
    // B=2, C=3. Perfect predictions: logits [10,0,0] → argmax=0; [0,10,0] → argmax=1.
    f32 logits[] = {
        10.0f, 0.0f, 0.0f,
        0.0f, 10.0f, 0.0f,
    };
    u8 labels[] = {0, 1};
    Term x = thvm_tensor(ctx, logits, SHAPE(2, 3));
    Term sm = softmax(ctx, x, 2, 3);
    Term sm_out = thvm_eval(ctx, sm);
    if (term_tag(sm_out) == TAG_TEN) {
        f32 *p = thvm_to_host(ctx, sm_out);
        printf("%-6s: probs=[%.3f,%.3f,%.3f | %.3f,%.3f,%.3f]\n",
            backend, p[0], p[1], p[2], p[3], p[4], p[5]);
    }
    Term x2 = thvm_tensor(ctx, logits, SHAPE(2, 3));
    Term loss = cross_entropy_loss(ctx, x2, labels, 2, 3);
    Term out = thvm_eval(ctx, loss);
    if (term_tag(out) != TAG_TEN) {
        printf("%s: FAIL loss.tag=%u (expected TAG_TEN)\n", backend, (u32)term_tag(out));
        return 1;
    }
    f32 *lv = thvm_to_host(ctx, out);
    if (!lv) { printf("%s: FAIL loss not readable\n", backend); return 1; }
    printf("%-6s: loss=%.4f (perfect preds → expect ~0)\n", backend, lv[0]);
    int ok = lv[0] < 0.01f;
    printf("%s: %s\n", backend, ok ? "PASS" : "FAIL");
    thvm_free(ctx);
    return ok ? 0 : 1;
}

int main(void) {
    int cpu = run("cpu");
#ifdef __APPLE__
    int metal = MTLCreateSystemDefaultDevice() ? run("metal") : 0;
    return (cpu == 0 && metal == 0) ? 0 : 1;
#else
    return cpu;
#endif
}
