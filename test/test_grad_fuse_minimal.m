// test_grad_fuse_minimal.m — tiny repro with keep-mode GRAD and FUSE branch
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *backend = getenv("THVM_TEST_BACKEND");
    if (!backend || !backend[0]) backend = argc > 1 ? argv[1] : "cpu";
#ifdef __APPLE__
    if (strcmp(backend, "metal") == 0 && !MTLCreateSystemDefaultDevice()) {
        fprintf(stderr, "SKIP: metal unavailable\n");
        return 0;
    }
#endif

    TinyHVM *ctx = thvm_init(backend);

    f32 xd[] = {1.0f, 2.0f, 3.0f};
    f32 wd[] = {0.5f, -1.0f, 2.0f};
    f32 bd[] = {2.0f, 2.0f, 2.0f};
    Shape s = {.dims = {3}, .rank = 1};

    Term x = thvm_tensor(ctx, xd, s);
    Term w = thvm_tensor(ctx, wd, s);
    Term b = thvm_tensor(ctx, bd, s);
    thvm_set_requires_grad(ctx, w);

    Term loss_mul = thvm_op(ctx, UOP_MUL, x, w);
    Term loss = thvm_sum_axes(ctx, loss_mul, (u32[]){0}, 1);
    Term grad_out = thvm_grad_multi_keep(ctx, loss, (Term[]){w}, 1);

    Term fuse_mul = thvm_op(ctx, UOP_MUL, x, w);
    Term fused = thvm_op(ctx, UOP_ADD, fuse_mul, b);

    Term program = thvm_ctr(ctx, (Term[]){grad_out, fused}, 2);
    Term result = thvm_eval(ctx, program);
    if (getenv("THVM_LOWER_GRAPH") && term_tag(result) == TAG_CTR) {
        u64 loc = term_val(result);
        u32 arity = term_ext(result);
        for (u32 i = 0; i < arity; i++) {
            Term child = heap_read(ctx, loc + i);
            if (term_tag(child) != TAG_TOP) continue;
            Term forced = thvm_force_tensor_term(ctx, child);
            if (term_tag(forced) == TAG_TEN) (void)thvm_to_host(ctx, forced);
        }
    }
    (void)result;

    thvm_free(ctx);
    return 0;
}
