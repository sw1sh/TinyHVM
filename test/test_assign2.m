#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include <stdio.h>
int main(void) {
    TinyHVM *ctx = thvm_init("metal");
    // Simulate slot-mode GRAD leaf accumulation: each deposit adds into the
    // existing slot buffer and returns zero to the visible backward graph.
    f32 z[4] = {0,0,0,0};
    Term slot = thvm_tensor(ctx, z, SHAPE(4));
    f32 data[4] = {1,2,3,4};
    Term gy = thvm_tensor(ctx, data, SHAPE(4));
    thvm_grad_slot_accum(ctx, slot, gy);
    f32 *out = thvm_to_host(ctx, slot);
    printf("slot: [%.1f, %.1f, %.1f, %.1f]\n", out[0], out[1], out[2], out[3]);
    printf("%s\n", (out[0]==1&&out[1]==2&&out[2]==3&&out[3]==4) ? "PASS" : "FAIL");
    // Now do a SECOND deposit (accumulative)
    f32 data2[4] = {10,20,30,40};
    Term gy2 = thvm_tensor(ctx, data2, SHAPE(4));
    thvm_grad_slot_accum(ctx, slot, gy2);
    out = thvm_to_host(ctx, slot);
    printf("slot after 2nd deposit: [%.1f, %.1f, %.1f, %.1f]\n", out[0], out[1], out[2], out[3]);
    printf("%s\n", (out[0]==11&&out[1]==22&&out[2]==33&&out[3]==44) ? "PASS" : "FAIL");
    return 0;
}
