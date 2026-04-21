#!/usr/bin/env python3
"""Compare TinyHVM gradients against numpy reference.
Loads init weights + data from /tmp/gc_*.bin (written by test_grad_check).
"""
import numpy as np, struct, sys

BS, H = 128, 128

# Load binary data written by test_grad_check
W1 = np.fromfile('/tmp/gc_W1.bin', dtype=np.float32).reshape(784, H)
W2 = np.fromfile('/tmp/gc_W2.bin', dtype=np.float32).reshape(H, 10)
B1 = np.zeros((1, H), dtype=np.float32)
B2 = np.zeros((1, 10), dtype=np.float32)
X = np.fromfile('/tmp/gc_X.bin', dtype=np.float32).reshape(BS, 784)
Y_oh = np.fromfile('/tmp/gc_Y.bin', dtype=np.float32).reshape(BS, 10)

# Forward: skip MLP
z1 = X @ W1 + B1
h = np.maximum(z1, 0)
h_sum = h.reshape(BS * H).sum().reshape(1, 1) * np.ones((BS, 10), dtype=np.float32)
out = h @ W2 + B2 + h_sum

# Cross-entropy (matching TinyHVM's cross_entropy_loss)
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

# Manual backward
d_out = (probs - Y_oh) / BS

# h_skip path
d_sum_scalar = d_out.sum()
d_h_skip = np.ones((BS, H), dtype=np.float32) * d_sum_scalar

# MM path
d_h_mm = d_out @ W2.T
d_W2 = h.T @ d_out
d_B2 = d_out.sum(axis=0, keepdims=True)

d_h = d_h_mm + d_h_skip

# relu backward
d_z1 = d_h * (z1 > 0).astype(np.float32)
d_W1 = X.T @ d_z1
d_B1 = d_z1.sum(axis=0, keepdims=True)

print(f"numpy gW1[0:4] = {d_W1.flatten()[:4]}")
print(f"numpy gB1[0:4] = {d_B1.flatten()[:4]}")
print(f"numpy gW2[0:4] = {d_W2.flatten()[:4]}")
print(f"numpy gB2[0:4] = {d_B2.flatten()[:4]}")

# Load TinyHVM grads and compare
names = ['gW1', 'gB1', 'gW2', 'gB2']
sizes = [784*H, H, H*10, 10]
refs = [d_W1, d_B1, d_W2, d_B2]

ok = True
for name, sz, ref in zip(names, sizes, refs):
    path = f'/tmp/gc_{name}.bin'
    try:
        got = np.fromfile(path, dtype=np.float32)
    except:
        print(f"{name}: FILE MISSING"); ok = False; continue
    if got.size != sz:
        print(f"{name}: SIZE MISMATCH got={got.size} expect={sz}"); ok = False; continue
    got = got.reshape(ref.shape)
    diff = np.abs(got - ref)
    maxdiff = diff.max()
    meandiff = diff.mean()
    reldiff = (diff / (np.abs(ref) + 1e-12)).max()
    print(f"{name}: max_abs_diff={maxdiff:.2e}  mean_diff={meandiff:.2e}  max_rel_diff={reldiff:.2e}  got[0:4]={got.flatten()[:4]}")
    if maxdiff > 1e-4:
        ok = False
        print(f"  FAIL: diff too large")

print(f"\n{'PASS' if ok else 'FAIL'}")
