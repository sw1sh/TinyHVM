#!/usr/bin/env python3
"""Compare TinyHVM grads against numpy — simple MLP, NO skip connection."""
import numpy as np

BS, H = 32, 64

W1 = np.fromfile('/tmp/gc_W1.bin', dtype=np.float32).reshape(784, H)
W2 = np.fromfile('/tmp/gc_W2.bin', dtype=np.float32).reshape(H, 10)
B1 = np.zeros((1, H), dtype=np.float32)
B2 = np.zeros((1, 10), dtype=np.float32)
X = np.fromfile('/tmp/gc_X.bin', dtype=np.float32).reshape(BS, 784)
Y_oh = np.fromfile('/tmp/gc_Y.bin', dtype=np.float32).reshape(BS, 10)

# Forward: simple MLP, no skip
z1 = X @ W1 + B1
h = np.maximum(z1, 0)
out = h @ W2 + B2

# Cross-entropy
x_max = out.max(axis=1, keepdims=True)
shifted = out - x_max
e = np.exp(shifted)
e_sum = e.sum(axis=1, keepdims=True)
probs = e / e_sum
clamped = np.maximum(probs, 1e-7)
log_probs = np.log(clamped)
masked = Y_oh * log_probs
loss = -(masked.sum() / BS)
print(f"numpy loss = {loss:.6f}")

# Backward
d_out = (probs - Y_oh) / BS
d_W2 = h.T @ d_out
d_B2 = d_out.sum(axis=0, keepdims=True)
d_h = d_out @ W2.T
d_z1 = d_h * (z1 > 0).astype(np.float32)
d_W1 = X.T @ d_z1
d_B1 = d_z1.sum(axis=0, keepdims=True)

names = ['gW1', 'gB1', 'gW2', 'gB2']
sizes = [784*H, H, H*10, 10]
refs = [d_W1, d_B1, d_W2, d_B2]

ok = True
for name, sz, ref in zip(names, sizes, refs):
    got = np.fromfile(f'/tmp/gc_{name}.bin', dtype=np.float32).reshape(ref.shape)
    diff = np.abs(got - ref)
    maxd = diff.max()
    meand = diff.mean()
    print(f"{name}: max_abs={maxd:.2e}  mean={meand:.2e}  got[0:4]={got.flatten()[:4]}")
    if maxd > 1e-5:
        ok = False
        print(f"  FAIL")
print(f"\n{'ALL ZERO — PASS' if ok else 'FAIL'}")
