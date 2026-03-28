#!/usr/bin/env python3
"""bench_conv.py — Tinygrad conv gradient benchmark (same arch as bench_conv.m)

Architecture: Conv(1→8, 3x3, pad=1) → ReLU → Conv(8→16, 3x3) → ReLU → flatten → Linear → MSE
Input: [BS, 1, 8, 8]
"""
import time, sys
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.nn import Conv2d, Linear

BS     = 16
CIN    = 1
H, W   = 8, 8
C1     = 8
C2     = 16
KH, KW = 3, 3
NCLASS = 10
N_STEPS = 10
WARMUP  = 3
LR      = 0.001

dev = sys.argv[1] if len(sys.argv) > 1 else Device.DEFAULT
print(f"=== Conv Gradient Benchmark (tinygrad, {dev}) ===")
print(f"    BS={BS}, Conv({CIN}→{C1},3x3)→ReLU→Conv({C1}→{C2},3x3)→ReLU→FC→MSE")
print(f"    Input: [{BS},{CIN},{H},{W}], {N_STEPS} steps\n")

np.random.seed(42)

# Init weights to match TinyHVM
scale1 = np.sqrt(2.0 / (CIN * KH * KW))
w1_np = (np.random.rand(C1, CIN, KH, KW).astype(np.float32) * 2 - 1) * scale1
scale2 = np.sqrt(2.0 / (C1 * KH * KW))
w2_np = (np.random.rand(C2, C1, KH, KW).astype(np.float32) * 2 - 1) * scale2
flat_dim = C2 * 6 * 6  # after conv2(no pad): 8-3+1=6
scale_fc = np.sqrt(2.0 / (flat_dim + NCLASS))
fc_np = (np.random.rand(flat_dim, NCLASS).astype(np.float32) * 2 - 1) * scale_fc

x_np = (np.random.rand(BS, CIN, H, W).astype(np.float32) * 2 - 1) * 0.5
y_np = np.zeros((BS, NCLASS), dtype=np.float32)
for i in range(BS):
    y_np[i, i % NCLASS] = 1.0

# Create tinygrad params
w1 = Tensor(w1_np, requires_grad=True)
w2 = Tensor(w2_np, requires_grad=True)
fc = Tensor(fc_np, requires_grad=True)

times = []
for step in range(N_STEPS + WARMUP):
    t0 = time.perf_counter()

    x = Tensor(x_np)
    y = Tensor(y_np)

    # Forward
    h1 = x.conv2d(w1, padding=1).relu()       # [BS, 8, 8, 8]
    h2 = h1.conv2d(w2, padding=0).relu()      # [BS, 16, 6, 6]
    flat = h2.reshape(BS, flat_dim)            # [BS, 576]
    logits = flat.dot(fc)                      # [BS, 10]

    # MSE loss
    diff = logits - y
    loss = (diff * diff).sum() / BS

    # Backward
    loss.backward()

    # Force realize all grads and loss
    lv = loss.numpy().item()

    # SGD update
    w1_np_new = w1.numpy() - LR * w1.grad.numpy()
    w2_np_new = w2.numpy() - LR * w2.grad.numpy()
    fc_np_new = fc.numpy() - LR * fc.grad.numpy()
    w1 = Tensor(w1_np_new, requires_grad=True)
    w2 = Tensor(w2_np_new, requires_grad=True)
    fc = Tensor(fc_np_new, requires_grad=True)

    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000

    tag = " (warmup)" if step < WARMUP else ""
    if step >= WARMUP:
        times.append(ms)
    print(f"  step {step:2d}: loss={lv:.4f}  {ms:.2f}ms{tag}")

avg = sum(times) / len(times)
print(f"\n  Average: {avg:.2f} ms/step ({N_STEPS} steps)")
print(f"  Throughput: {BS * 1000 / avg:.0f} samples/sec")

