// test_reduce_axes.m — Test multi-axis reduce on Metal vs CPU
// This tests the specific pattern that fails in conv backward:
// SUM over non-trailing axes on a rank-8 tensor
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    // Test: SUM over axes 4,6 of [1,4,1,4,4,12,4,12]
    // This is the EXPAND backward pattern that zeros gradients

    u32 shape[] = {1,4,1,4,4,12,4,12};
    u32 rank = 8;
    u32 numel = 1;
    for (u32 i=0;i<rank;i++) numel *= shape[i];

    f32 *data = malloc(numel * 4);
    srand(42);
    for (u32 i=0;i<numel;i++) data[i] = (f32)rand()/(f32)RAND_MAX * 2 - 1;

    // CPU reference
    {
        TinyHVM *ctx = thvm_init("cpu");
        Term x = thvm_tensor(ctx, data, (Shape){.dims={1,4,1,4,4,12,4,12},.rank=8});
        extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
        Term r = thvm_sum_axes(ctx, x, (u32[]){4,6}, 2);
        thvm_reduce(ctx, r);
        f32 *rd = thvm_to_host(ctx, r);
        u32 rn = ctx->tensors[(u32)term_val(r)].view.numel;
        double norm = 0;
        for (u32 i=0;i<rn;i++) norm += rd[i]*rd[i];
        printf("CPU  SUM(axes 4,6): numel=%u norm=%.6f first=[%.4f,%.4f,%.4f,%.4f]\n",
            rn, sqrt(norm), rd[0], rd[1], rd[2], rd[3]);
        thvm_free(ctx);
    }

    // Metal
    {
        TinyHVM *ctx = thvm_init("metal");
        Term x = thvm_tensor(ctx, data, (Shape){.dims={1,4,1,4,4,12,4,12},.rank=8});
        extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
        Term r = thvm_sum_axes(ctx, x, (u32[]){4,6}, 2);
        thvm_reduce(ctx, r);
        f32 *rd = thvm_to_host(ctx, r);
        u32 rn = ctx->tensors[(u32)term_val(r)].view.numel;
        double norm = 0;
        for (u32 i=0;i<rn;i++) norm += rd[i]*rd[i];
        printf("Metal SUM(axes 4,6): numel=%u norm=%.6f first=[%.4f,%.4f,%.4f,%.4f]\n",
            rn, sqrt(norm), rd[0], rd[1], rd[2], rd[3]);
        thvm_free(ctx);
    }

    // Also test simpler case: SUM over axis 2 of [4,3,5]
    {
        u32 n2 = 4*3*5;
        f32 *d2 = malloc(n2*4);
        for (u32 i=0;i<n2;i++) d2[i] = (f32)(i+1);

        TinyHVM *cpu = thvm_init("cpu");
        Term xc = thvm_tensor(cpu, d2, (Shape){.dims={4,3,5},.rank=3});
        extern Term thvm_sum_axes(TinyHVM*,Term,const u32*,u32);
        Term rc = thvm_sum_axes(cpu, xc, (u32[]){1}, 1);
        thvm_reduce(cpu, rc);
        f32 *rdc = thvm_to_host(cpu, rc);
        printf("\nCPU  SUM(axis 1) [4,3,5]: first=[%.1f,%.1f,%.1f,%.1f,%.1f]\n",
            rdc[0],rdc[1],rdc[2],rdc[3],rdc[4]);
        thvm_free(cpu);

        TinyHVM *gpu = thvm_init("metal");
        Term xg = thvm_tensor(gpu, d2, (Shape){.dims={4,3,5},.rank=3});
        Term rg = thvm_sum_axes(gpu, xg, (u32[]){1}, 1);
        thvm_reduce(gpu, rg);
        f32 *rdg = thvm_to_host(gpu, rg);
        printf("Metal SUM(axis 1) [4,3,5]: first=[%.1f,%.1f,%.1f,%.1f,%.1f]\n",
            rdg[0],rdg[1],rdg[2],rdg[3],rdg[4]);
        thvm_free(gpu);
        free(d2);
    }

    free(data);
    return 0;
}
