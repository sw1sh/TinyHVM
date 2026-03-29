#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
int main(void) {
    TinyHVM *ctx = thvm_init("metal");
    // Simulate grad_multi pattern: slot=zeros, gy=nonzero, deposit via ASSIGN
    f32 z[4] = {0,0,0,0};
    Term slot = thvm_tensor(ctx, z, SHAPE(4));
    f32 data[4] = {1,2,3,4};
    Term gy = thvm_tensor(ctx, data, SHAPE(4));
    // Wrap in APP(ASSIGN(slot, ADD(slot, gy)), ERA) — like the GRAD handler does
    Term accum = thvm_op(ctx, UOP_ADD, slot, gy);
    Term chain = thvm_app(ctx, thvm_assign(ctx, slot, accum), term_era());
    thvm_reduce(ctx, chain);
    f32 *out = thvm_to_host(ctx, slot);
    printf("slot: [%.1f, %.1f, %.1f, %.1f]\n", out[0], out[1], out[2], out[3]);
    printf("%s\n", (out[0]==1&&out[1]==2&&out[2]==3&&out[3]==4) ? "PASS" : "FAIL");
    // Now do a SECOND deposit (accumulative)
    f32 data2[4] = {10,20,30,40};
    Term gy2 = thvm_tensor(ctx, data2, SHAPE(4));
    Term accum2 = thvm_op(ctx, UOP_ADD, slot, gy2);
    Term chain2 = thvm_app(ctx, thvm_assign(ctx, slot, accum2), term_era());
    thvm_reduce(ctx, chain2);
    out = thvm_to_host(ctx, slot);
    printf("slot after 2nd deposit: [%.1f, %.1f, %.1f, %.1f]\n", out[0], out[1], out[2], out[3]);
    printf("%s\n", (out[0]==11&&out[1]==22&&out[2]==33&&out[3]==44) ? "PASS" : "FAIL");
    return 0;
}
