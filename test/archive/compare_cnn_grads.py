#!/usr/bin/env python3
"""Compare TinyHVM CNN grads against numpy reference.

Model: Conv(1,8,3) -> ReLU -> Flatten -> Linear(5408,10) -> CrossEntropy
Loads weights, input, labels, and gradients from /tmp/cnn_*.bin.
Pure numpy — no tinygrad dependency.
"""
import numpy as np

BS = 32
Cin, Cout = 1, 8
KH, KW = 3, 3
IH, IW = 28, 28
OH, OW = 26, 26  # valid padding: 28-3+1=26
flat_f = Cout * OH * OW  # 8*26*26 = 5408
NC = 10

# ── Load data ─────────────────────────────────────────────────
conv_w = np.fromfile('/tmp/cnn_conv_w.bin', dtype=np.float32).reshape(Cout, Cin, KH, KW)
conv_b = np.fromfile('/tmp/cnn_conv_b.bin', dtype=np.float32).reshape(Cout)
lin_w  = np.fromfile('/tmp/cnn_lin_w.bin',  dtype=np.float32).reshape(flat_f, NC)
lin_b  = np.fromfile('/tmp/cnn_lin_b.bin',  dtype=np.float32).reshape(NC)
X      = np.fromfile('/tmp/cnn_X.bin',      dtype=np.float32).reshape(BS, Cin, IH, IW)
Y_oh   = np.fromfile('/tmp/cnn_Y.bin',      dtype=np.float32).reshape(BS, NC)

# ── im2col helpers ────────────────────────────────────────────
def im2col(x, kh, kw, sh=1, sw=1):
    """x: (B, C, H, W) -> cols: (B, C*kh*kw, oh*ow)"""
    B, C, H, W = x.shape
    oh = (H - kh) // sh + 1
    ow = (W - kw) // sw + 1
    cols = np.zeros((B, C * kh * kw, oh * ow), dtype=x.dtype)
    idx = 0
    for i in range(oh):
        for j in range(ow):
            patch = x[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw]  # (B, C, kh, kw)
            cols[:, :, idx] = patch.reshape(B, -1)
            idx += 1
    return cols

def col2im(dcols, x_shape, kh, kw, sh=1, sw=1):
    """dcols: (B, C*kh*kw, oh*ow) -> dx: (B, C, H, W)"""
    B, C, H, W = x_shape
    oh = (H - kh) // sh + 1
    ow = (W - kw) // sw + 1
    dx = np.zeros(x_shape, dtype=dcols.dtype)
    idx = 0
    for i in range(oh):
        for j in range(ow):
            patch = dcols[:, :, idx].reshape(B, C, kh, kw)
            dx[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw] += patch
            idx += 1
    return dx

# ── Forward ───────────────────────────────────────────────────
# Conv2d: valid padding, stride 1
# im2col: X -> (BS, Cin*KH*KW, OH*OW)
X_col = im2col(X, KH, KW)  # (BS, 9, 676)
W_col = conv_w.reshape(Cout, -1)  # (8, 9)

# conv_out[b] = W_col @ X_col[b] + bias
conv_out = np.zeros((BS, Cout, OH * OW), dtype=np.float32)
for b in range(BS):
    conv_out[b] = W_col @ X_col[b] + conv_b.reshape(Cout, 1)
conv_out_4d = conv_out.reshape(BS, Cout, OH, OW)

# ReLU
relu_out = np.maximum(conv_out_4d, 0.0)
relu_mask = (conv_out_4d > 0).astype(np.float32)

# Flatten: (BS, Cout, OH, OW) -> (BS, flat_f)
flat = relu_out.reshape(BS, flat_f)

# Linear: logits = flat @ lin_w + lin_b
logits = flat @ lin_w + lin_b.reshape(1, NC)

# ── Cross-entropy loss ────────────────────────────────────────
# softmax
x_max = logits.max(axis=1, keepdims=True)
shifted = logits - x_max
e = np.exp(shifted)
e_sum = e.sum(axis=1, keepdims=True)
probs = e / e_sum

# clamp + log
clamped = np.maximum(probs, 1e-7)
log_probs = np.log(clamped)

# masked sum
masked = Y_oh * log_probs
loss = -(masked.sum() / BS)
print(f"numpy loss = {loss:.6f}")

# ── Backward ─────────────────────────────────────────────────
# d_logits from cross-entropy + softmax
d_logits = (probs - Y_oh) / BS  # (BS, NC)

# Linear backward
d_lin_w = flat.T @ d_logits          # (flat_f, NC)
d_lin_b = d_logits.sum(axis=0)       # (NC,)
d_flat = d_logits @ lin_w.T          # (BS, flat_f)

# Unflatten: (BS, flat_f) -> (BS, Cout, OH, OW)
d_relu_out = d_flat.reshape(BS, Cout, OH, OW)

# ReLU backward
d_conv_out_4d = d_relu_out * relu_mask

# Conv2d backward via im2col
d_conv_out = d_conv_out_4d.reshape(BS, Cout, OH * OW)  # (BS, 8, 676)

# d_conv_w: sum over batch of d_conv_out @ X_col^T
d_W_col = np.zeros_like(W_col)  # (8, 9)
d_X_col = np.zeros_like(X_col)  # (BS, 9, 676)
d_conv_b_acc = np.zeros(Cout, dtype=np.float32)

for b in range(BS):
    d_W_col += d_conv_out[b] @ X_col[b].T    # (8, 9)
    d_X_col[b] = W_col.T @ d_conv_out[b]     # (9, 676)
    d_conv_b_acc += d_conv_out[b].sum(axis=1) # (8,)

d_conv_w = d_W_col.reshape(Cout, Cin, KH, KW)
d_conv_b = d_conv_b_acc

# col2im for input gradient (not needed for comparison, but validates correctness)
# d_X = col2im(d_X_col, X.shape, KH, KW)

# ── Print numpy reference ────────────────────────────────────
print(f"numpy d_conv_w[0:4] = {d_conv_w.flatten()[:4]}")
print(f"numpy d_conv_b[0:4] = {d_conv_b.flatten()[:4]}")
print(f"numpy d_lin_w[0:4]  = {d_lin_w.flatten()[:4]}")
print(f"numpy d_lin_b[0:4]  = {d_lin_b.flatten()[:4]}")

# ── Compare against TinyHVM ──────────────────────────────────
names  = ['g_conv_w', 'g_conv_b', 'g_lin_w', 'g_lin_b']
shapes = [(Cout, Cin, KH, KW), (Cout,), (flat_f, NC), (NC,)]
refs   = [d_conv_w, d_conv_b, d_lin_w, d_lin_b]
sizes  = [np.prod(s) for s in shapes]

ok = True
for name, sz, shape, ref in zip(names, sizes, shapes, refs):
    path = f'/tmp/cnn_{name}.bin'
    try:
        got = np.fromfile(path, dtype=np.float32)
    except FileNotFoundError:
        print(f"{name}: FILE MISSING")
        ok = False
        continue
    if got.size != sz:
        print(f"{name}: SIZE MISMATCH got={got.size} expect={sz}")
        ok = False
        continue
    got = got.reshape(shape)
    diff = np.abs(got - ref)
    maxd = diff.max()
    meand = diff.mean()
    reld = (diff / (np.abs(ref) + 1e-12)).max()
    print(f"{name}: max_abs={maxd:.2e}  mean={meand:.2e}  max_rel={reld:.2e}  "
          f"got[0:4]={got.flatten()[:4]}  ref[0:4]={ref.flatten()[:4]}")
    if maxd > 1e-4:
        ok = False
        print(f"  FAIL: diff too large")

print(f"\n{'PASS' if ok else 'FAIL'}")
