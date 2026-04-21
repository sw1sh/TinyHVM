#!/usr/bin/env python3
"""CNN + CE loss grad check: Conv(1,8,3)→ReLU→Flatten→Linear(5408,10), cross-entropy"""
import numpy as np

BS = 4
flat_f = 8*26*26

cw = np.fromfile('/tmp/ce_cw.bin', dtype=np.float32).reshape(8,1,3,3)
cb = np.zeros(8, dtype=np.float32)
lw = np.fromfile('/tmp/ce_lw.bin', dtype=np.float32).reshape(flat_f, 10)
lb = np.zeros(10, dtype=np.float32)
X = np.fromfile('/tmp/ce_X.bin', dtype=np.float32).reshape(BS, 1, 28, 28)
Y_oh = np.fromfile('/tmp/ce_Y.bin', dtype=np.float32).reshape(BS, 10)

# Forward: conv + relu + flatten + linear
def conv2d_fwd(x, w, b):
    B,Cin,H,W = x.shape; Cout,_,KH,KW = w.shape; OH,OW = H-KH+1, W-KW+1
    col = np.zeros((B,Cin,KH,KW,OH,OW), dtype=np.float32)
    for kh in range(KH):
        for kw in range(KW):
            col[:,:,kh,kw,:,:] = x[:,:,kh:kh+OH,kw:kw+OW]
    w_flat = w.reshape(Cout, Cin*KH*KW)
    col_flat = col.reshape(B, Cin*KH*KW, OH*OW)
    out = np.einsum('ck,bkp->bcp', w_flat, col_flat).reshape(B, Cout, OH, OW) + b.reshape(1,Cout,1,1)
    return out, col

def conv2d_bwd(dy, x, w, col):
    B,Cin,H,W = x.shape; Cout,_,KH,KW = w.shape; OH,OW = H-KH+1, W-KW+1
    db = dy.sum(axis=(0,2,3))
    dy_flat = dy.reshape(B, Cout, OH*OW)
    col_flat = col.reshape(B, Cin*KH*KW, OH*OW)
    w_flat = w.reshape(Cout, Cin*KH*KW)
    dw = np.einsum('bcp,bkp->ck', dy_flat, col_flat).reshape(w.shape)
    return dw, db

h_pre, col = conv2d_fwd(X, cw, cb)
h = np.maximum(h_pre, 0)
flat = h.reshape(BS, flat_f)
logits = flat @ lw + lb

# CE loss
x_max = logits.max(axis=1, keepdims=True)
shifted = logits - x_max
e = np.exp(shifted)
e_sum = e.sum(axis=1, keepdims=True)
probs = e / e_sum
clamped = np.maximum(probs, 1e-7)
log_probs = np.log(clamped)
loss = -(Y_oh * log_probs).sum() / BS
print(f"numpy loss = {loss:.6f}")

# Backward
d_logits = (probs - Y_oh) / BS
d_lw = flat.T @ d_logits
d_lb = d_logits.sum(axis=0)
d_flat = d_logits @ lw.T
d_h = d_flat.reshape(BS, 8, 26, 26)
d_h_pre = d_h * (h_pre > 0).astype(np.float32)
d_cw, d_cb = conv2d_bwd(d_h_pre, X, cw, col)

names = ['g_cw','g_cb','g_lw','g_lb']
sizes = [8*1*3*3, 8, flat_f*10, 10]
refs = [d_cw, d_cb, d_lw, d_lb]
shapes = [cw.shape, cb.shape, lw.shape, lb.shape]

ok = True
for name, sz, ref, sh in zip(names, sizes, refs, shapes):
    got = np.fromfile(f'/tmp/ce_{name}.bin', dtype=np.float32).reshape(sh)
    diff = np.abs(got - ref)
    maxd = diff.max(); meand = diff.mean()
    status = "OK" if maxd < 1e-3 else "FAIL"
    if maxd > 1e-3: ok = False
    print(f"{name}: max_abs={maxd:.2e} mean={meand:.2e} {status}  got[0:4]={got.flatten()[:4]}")

print(f"\n{'PASS' if ok else 'FAIL'}")
