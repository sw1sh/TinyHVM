#!/usr/bin/env python3
"""Compare 2-layer CNN grads: Conv(1,4,3)→ReLU→Conv(4,8,3)→ReLU→Flatten→Linear"""
import numpy as np

BS, C1, C2 = 4, 4, 8

def conv2d_forward(x, w, b):
    """x:[B,Cin,H,W] w:[Cout,Cin,KH,KW] b:[Cout] → [B,Cout,OH,OW]"""
    B,Cin,H,W = x.shape
    Cout,_,KH,KW = w.shape
    OH, OW = H-KH+1, W-KW+1
    # im2col
    col = np.zeros((B, Cin, KH, KW, OH, OW), dtype=np.float32)
    for kh in range(KH):
        for kw in range(KW):
            col[:,:,kh,kw,:,:] = x[:,:,kh:kh+OH,kw:kw+OW]
    col_flat = col.reshape(B, Cin*KH*KW, OH*OW)        # [B, Cin*K*K, OH*OW]
    w_flat = w.reshape(Cout, Cin*KH*KW)                  # [Cout, Cin*K*K]
    out = np.einsum('ck,bkp->bcp', w_flat, col_flat)     # [B, Cout, OH*OW]
    out = out.reshape(B, Cout, OH, OW)
    out += b.reshape(1, Cout, 1, 1)
    return out, col

def conv2d_backward(dy, x, w, col):
    """dy:[B,Cout,OH,OW] → dw:[Cout,Cin,KH,KW], db:[Cout], dx:[B,Cin,H,W]"""
    B,Cin,H,W = x.shape
    Cout,_,KH,KW = w.shape
    OH, OW = H-KH+1, W-KW+1
    db = dy.sum(axis=(0,2,3))
    dy_flat = dy.reshape(B, Cout, OH*OW)
    col_flat = col.reshape(B, Cin*KH*KW, OH*OW)
    w_flat = w.reshape(Cout, Cin*KH*KW)
    # dw = sum_b dy @ col^T
    dw = np.einsum('bcp,bkp->ck', dy_flat, col_flat).reshape(w.shape)
    # dcol = w^T @ dy
    dcol_flat = np.einsum('ck,bcp->bkp', w_flat, dy_flat)
    dcol = dcol_flat.reshape(B, Cin, KH, KW, OH, OW)
    dx = np.zeros_like(x)
    for kh in range(KH):
        for kw in range(KW):
            dx[:,:,kh:kh+OH,kw:kw+OW] += dcol[:,:,kh,kw,:,:]
    return dw, db, dx

# Load data
cw1 = np.fromfile('/tmp/c2_cw1.bin', dtype=np.float32).reshape(C1,1,3,3)
cb1 = np.fromfile('/tmp/c2_cb1.bin', dtype=np.float32)
cw2 = np.fromfile('/tmp/c2_cw2.bin', dtype=np.float32).reshape(C2,C1,3,3)
cb2 = np.fromfile('/tmp/c2_cb2.bin', dtype=np.float32)
flat_f = C2*24*24
lw = np.fromfile('/tmp/c2_lw.bin', dtype=np.float32).reshape(flat_f, 10)
lb = np.fromfile('/tmp/c2_lb.bin', dtype=np.float32)
X = np.fromfile('/tmp/c2_X.bin', dtype=np.float32).reshape(BS, 1, 28, 28)

# Forward
h1_pre, col1 = conv2d_forward(X, cw1, cb1)
h1 = np.maximum(h1_pre, 0)
h2_pre, col2 = conv2d_forward(h1, cw2, cb2)
h2 = np.maximum(h2_pre, 0)
flat = h2.reshape(BS, flat_f)
logits = flat @ lw + lb
loss = logits.sum()
print(f"numpy loss = {loss:.6f}")

# Backward: loss = sum(logits)
d_logits = np.ones_like(logits)
d_lw = flat.T @ d_logits
d_lb = d_logits.sum(axis=0)
d_flat = d_logits @ lw.T
d_h2 = d_flat.reshape(BS, C2, 24, 24)
d_h2_pre = d_h2 * (h2_pre > 0).astype(np.float32)
d_cw2, d_cb2, d_h1 = conv2d_backward(d_h2_pre, h1, cw2, col2)
d_h1_pre = d_h1 * (h1_pre > 0).astype(np.float32)
d_cw1, d_cb1, _ = conv2d_backward(d_h1_pre, X, cw1, col1)

names = ['g_cw1','g_cb1','g_cw2','g_cb2','g_lw','g_lb']
sizes = [C1*1*3*3, C1, C2*C1*3*3, C2, flat_f*10, 10]
refs = [d_cw1, d_cb1, d_cw2, d_cb2, d_lw, d_lb]
shapes = [cw1.shape, cb1.shape, cw2.shape, cb2.shape, lw.shape, lb.shape]

ok = True
for name, sz, ref, sh in zip(names, sizes, refs, shapes):
    got = np.fromfile(f'/tmp/c2_{name}.bin', dtype=np.float32).reshape(sh)
    diff = np.abs(got - ref)
    maxd, meand = diff.max(), diff.mean()
    reld = (diff / (np.abs(ref) + 1e-12)).max()
    status = "OK" if maxd < 1e-3 else "FAIL"
    if maxd > 1e-3: ok = False
    print(f"{name}: max_abs={maxd:.2e} mean={meand:.2e} max_rel={reld:.2e} {status}  got[0:4]={got.flatten()[:4]}")

print(f"\n{'PASS' if ok else 'FAIL'}")
