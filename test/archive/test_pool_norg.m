#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static void test(const char *dev) {
    srand(42);
    TinyHVM *ctx = thvm_init(dev);
    u32 n=4*4*12*12;
    f32 xd[2304]; for(u32 i=0;i<n;i++)xd[i]=(f32)rand()/(f32)RAND_MAX*2-1;
    Term x=thvm_tensor(ctx,xd,(Shape){.dims={4,4,12,12},.rank=4});
    // NO requires_grad
    Term h=thvm_maxpool2d(ctx,x,(u32[]){2,2},(u32[]){2,2});
    thvm_reduce(ctx,h);
    f32*out=thvm_to_host(ctx,h);
    u32 on=4*4*6*6;
    double norm=0;for(u32 i=0;i<on;i++)norm+=out[i]*out[i];
    printf("%-6s: pool norm=%.6f first=[%.6f,%.6f,%.6f,%.6f]\n",
        dev,sqrt(norm),out[0],out[1],out[2],out[3]);
    thvm_free(ctx);
}
int main(void){test("cpu");test("metal");return 0;}
