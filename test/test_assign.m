#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
int main(void) {
    TinyHVM *ctx = thvm_init("metal");
    f32 z[4] = {0,0,0,0};
    Term slot = thvm_tensor(ctx, z, SHAPE(4));
    f32 data[4] = {1,2,3,4};
    Term src = thvm_tensor(ctx, data, SHAPE(4));
    // ASSIGN(slot, ADD(slot, src))
    Term accum = thvm_op(ctx, UOP_ADD, slot, src);
    Term asgn = thvm_assign(ctx, slot, accum);
    thvm_reduce(ctx, thvm_app(ctx, asgn, term_era()));
    f32 *out = thvm_to_host(ctx, slot);
    printf("slot: [%.1f, %.1f, %.1f, %.1f]\n", out[0], out[1], out[2], out[3]);
    printf("%s\n", (out[0]==1&&out[1]==2&&out[2]==3&&out[3]==4) ? "PASS" : "FAIL");
    return 0;
}
