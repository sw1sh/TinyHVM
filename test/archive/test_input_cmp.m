#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <math.h>

static void check(const char *dev) {
    srand(42);
    TinyHVM *ctx = thvm_init(dev);
    u32 BS=4,Cin=1,H=14,W=14,Cout=32,K=3;
    u32 xn=BS*Cin*H*W;
    f32 *xd=malloc(xn*4);for(u32 i=0;i<xn;i++)xd[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    // Print first 5 values of x
    printf("%-6s x[0..4]: %.6f %.6f %.6f %.6f %.6f\n", dev, xd[0],xd[1],xd[2],xd[3],xd[4]);
    free(xd);
    thvm_free(ctx);
}
int main(void) { check("cpu"); check("metal"); return 0; }
