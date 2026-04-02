// test_conv_grad.m — Verify conv backward dW via matmul vs generic
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
    srand(42);
    TinyHVM *ctx = thvm_init("metal");
    u32 BS=4, cin=2, cout=3, KH=3, KW=3, H=5, W=5;
    u32 OY = H-KH+1, OX = W-KW+1; // 3,3
    u32 M = BS*OY*OX, K = cin*KH*KW, N = cout;

    // Random input and weight
    u32 n_inp = BS*cin*H*W, n_w = cout*cin*KH*KW;
    f32 *inp_d = malloc(n_inp*4), *w_d = malloc(n_w*4);
    for (u32 i=0;i<n_inp;i++) inp_d[i] = (f32)rand()/RAND_MAX*2-1;
    for (u32 i=0;i<n_w;i++) w_d[i] = (f32)rand()/RAND_MAX*2-1;
    Term inp = thvm_tensor(ctx, inp_d, (Shape){.dims={BS,cin,H,W},.rank=4});
    Term w = thvm_tensor(ctx, w_d, (Shape){.dims={cout,cin,KH,KW},.rank=4});

    // Forward conv (using standard ops)
    u32 p0[]={0,0,0,0}, s1[]={1,1};
    Term out = thvm_conv2d(ctx, inp, w, term_era(), 1, s1, p0);

    // Random grad_output
    f32 *gy_d = malloc(BS*cout*OY*OX*4);
    for (u32 i=0;i<BS*cout*OY*OX;i++) gy_d[i] = (f32)rand()/RAND_MAX*2-1;

    // === Method 1: matmul dW ===
    // im2col = pool(inp, {KH,KW}, {1,1}) → [BS,cin,OY,OX,KH,KW]
    Term im2col = thvm_pool(ctx, inp, (u32[]){KH,KW}, (u32[]){1,1}, 2);
    // permute to [BS,OY,OX,cin,KH,KW]
    Term im_perm = thvm_permute(ctx, im2col, (u32[]){0,2,3,1,4,5}, 6);
    // reshape to [M,K]
    Term im_flat = thvm_reshape(ctx, im_perm, SHAPE(M, K));
    // im_T = [K,M]
    Term im_T = thvm_permute(ctx, im_flat, (u32[]){1,0}, 2);

    // gy → [M,N]
    Term gy_t = thvm_tensor(ctx, gy_d, (Shape){.dims={BS,cout,OY,OX},.rank=4});
    Term gy_perm = thvm_permute(ctx, gy_t, (u32[]){0,2,3,1}, 4);
    Term gy_flat = thvm_reshape(ctx, gy_perm, SHAPE(M, N));

    // dW_flat = im_T @ gy_flat = [K,M] @ [M,N] = [K,N]
    Term dW_flat = thvm_op(ctx, UOP_MM, im_T, gy_flat);
    // reshape [K,N] → [cin,KH,KW,cout]
    Term dW_rs = thvm_reshape(ctx, dW_flat, shape_of((u32[]){cin,KH,KW,cout}, 4));
    // permute → [cout,cin,KH,KW]
    Term dW_mm = thvm_permute(ctx, dW_rs, (u32[]){3,0,1,2}, 4);
    thvm_reduce(ctx, dW_mm);

    // === Method 2: CPU reference ===
    // dW[n,c,kh,kw] = sum_{b,oy,ox} gy[b,n,oy,ox] * inp[b,c,oy+kh,ox+kw]
    f32 *dW_ref = calloc(n_w, 4);
    for (u32 n=0;n<cout;n++)
      for (u32 c=0;c<cin;c++)
        for (u32 kh=0;kh<KH;kh++)
          for (u32 kw=0;kw<KW;kw++) {
            f32 acc = 0;
            for (u32 b=0;b<BS;b++)
              for (u32 oy=0;oy<OY;oy++)
                for (u32 ox=0;ox<OX;ox++)
                  acc += gy_d[b*cout*OY*OX + n*OY*OX + oy*OX + ox] *
                         inp_d[b*cin*H*W + c*H*W + (oy+kh)*W + (ox+kw)];
            dW_ref[n*cin*KH*KW + c*KH*KW + kh*KW + kw] = acc;
          }

    // Compare
    f32 *dW_mm_h = thvm_to_host(ctx, dW_mm);
    f32 max_diff = 0, max_val = 0;
    for (u32 i=0;i<n_w;i++) {
        f32 d = fabsf(dW_mm_h[i] - dW_ref[i]);
        if (d > max_diff) max_diff = d;
        if (fabsf(dW_ref[i]) > max_val) max_val = fabsf(dW_ref[i]);
    }
    printf("dW matmul vs CPU ref: max_diff=%.6f max_val=%.6f rel=%.2e\n",
        max_diff, max_val, max_val > 0 ? max_diff/max_val : 0);
    if (max_diff < 1e-3) printf("PASS\n");
    else {
        printf("FAIL — first 10 elements:\n");
        for (u32 i=0;i<10&&i<n_w;i++)
            printf("  [%u] mm=%.6f ref=%.6f\n", i, dW_mm_h[i], dW_ref[i]);
    }

    // === Test dX via matmul + col2im ===
    // dX_flat = gy_flat @ w_flat = [M,N] @ [N,K] = [M,K]
    Term w_flat2 = thvm_reshape(ctx, w, SHAPE(N, K));
    Term dX_flat2 = thvm_op(ctx, UOP_MM, gy_flat, w_flat2);
    // reshape [M,K] → [BS,OY,OX,cin,KH,KW]
    Term dX_6d = thvm_reshape(ctx, dX_flat2, shape_of((u32[]){BS,OY,OX,cin,KH,KW}, 6));
    // permute to [BS,cin,OY,OX,KH,KW]
    Term dX_perm2 = thvm_permute(ctx, dX_6d, (u32[]){0,3,1,2,4,5}, 6);
    // col2im via pad+sum
    Term dX_acc = term_era();
    for (u32 kh=0;kh<KH;kh++) for (u32 kw=0;kw<KW;kw++) {
        u32 sh6[12];
        for (u32 j=0;j<6;j++) { sh6[j*2]=0; sh6[j*2+1]=(u32[]){BS,cin,OY,OX,KH,KW}[j]; }
        sh6[4*2]=kh; sh6[4*2+1]=kh+1;
        sh6[5*2]=kw; sh6[5*2+1]=kw+1;
        Term sl = thvm_shrink(ctx, dX_perm2, sh6, 6);
        sl = thvm_reshape(ctx, sl, SHAPE(BS, cin, OY, OX));
        u32 pp[8]; memset(pp,0,sizeof(pp));
        pp[2*2]=kh; pp[2*2+1]=H-OY-kh;
        pp[3*2]=kw; pp[3*2+1]=W-OX-kw;
        sl = thvm_pad(ctx, sl, pp, 4);
        if (term_tag(dX_acc)==TAG_ERA) dX_acc=sl;
        else dX_acc=thvm_op(ctx,UOP_ADD,dX_acc,sl);
    }
    thvm_reduce(ctx, dX_acc);
    f32 *dX_h = thvm_to_host(ctx, dX_acc);

    // CPU reference: dX[b,c,h,w] = sum_{n,kh,kw} gy[b,n,h-kh,w-kw] * w[n,c,kh,kw]
    f32 *dX_ref = calloc(n_inp, 4);
    for (u32 b=0;b<BS;b++)
      for (u32 c=0;c<cin;c++)
        for (u32 h=0;h<H;h++)
          for (u32 ww=0;ww<W;ww++) {
            f32 acc = 0;
            for (u32 n=0;n<cout;n++)
              for (u32 kh=0;kh<KH;kh++)
                for (u32 kw=0;kw<KW;kw++) {
                  int oy = (int)h-(int)kh, ox = (int)ww-(int)kw;
                  if (oy>=0 && oy<(int)OY && ox>=0 && ox<(int)OX)
                    acc += gy_d[b*cout*OY*OX + n*OY*OX + oy*OX + ox] *
                           w_d[n*cin*KH*KW + c*KH*KW + kh*KW + kw];
                }
            dX_ref[b*cin*H*W + c*H*W + h*W + ww] = acc;
          }

    max_diff = 0; max_val = 0;
    for (u32 i=0;i<n_inp;i++) {
        f32 d = fabsf(dX_h[i] - dX_ref[i]);
        if (d > max_diff) max_diff = d;
        if (fabsf(dX_ref[i]) > max_val) max_val = fabsf(dX_ref[i]);
    }
    printf("dX col2im vs CPU ref: max_diff=%.6f max_val=%.6f rel=%.2e\n",
        max_diff, max_val, max_val > 0 ? max_diff/max_val : 0);
    if (max_diff < 1e-3) printf("PASS\n");
    else {
        printf("FAIL — first 10 elements:\n");
        for (u32 i=0;i<10&&i<n_inp;i++)
            printf("  [%u] col2im=%.6f ref=%.6f\n", i, dX_h[i], dX_ref[i]);
    }

    free(inp_d); free(w_d); free(gy_d); free(dW_ref); free(dX_ref);
    thvm_free(ctx);
    return 0;
}
