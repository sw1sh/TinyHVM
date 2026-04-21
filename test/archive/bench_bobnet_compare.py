#!/usr/bin/env python3
"""bench_bobnet_compare.py — BobNet benchmark: PyTorch vs NumPy vs TinyHVM summary

Same architecture: x @ W1 → relu → @ W2, BS=69, 784→128→10
MSE loss, SGD lr=0.001, 20 timed steps (3 warmup)
"""
import time
import numpy as np

BS, IN, H, OUT = 69, 784, 128, 10
N_STEPS, WARMUP = 20, 3

def make_weights(seed=1337):
    np.random.seed(seed)
    w1 = (np.random.randn(IN, H) * np.sqrt(2.0/(IN+H))).astype(np.float32)
    w2 = (np.random.randn(H, OUT) * np.sqrt(2.0/(H+OUT))).astype(np.float32)
    x = np.random.randn(BS, IN).astype(np.float32)
    y = np.zeros((BS, OUT), dtype=np.float32)
    for i in range(BS): y[i, i % OUT] = 1.0
    return w1, w2, x, y

# ── PyTorch ─────────────────────────────────────────────────────
def bench_pytorch():
    import torch
    torch.set_num_threads(1)  # single-thread for fair comparison
    w1_np, w2_np, x_np, y_np = make_weights()
    w1 = torch.tensor(w1_np, requires_grad=True)
    w2 = torch.tensor(w2_np, requires_grad=True)
    x_t = torch.tensor(x_np)
    y_t = torch.tensor(y_np)

    times, losses = [], []
    for step in range(N_STEPS + WARMUP):
        t0 = time.perf_counter()
        logits = (x_t @ w1).relu() @ w2
        diff = logits - y_t
        loss = (diff * diff).sum() / BS
        loss.backward()
        with torch.no_grad():
            w1_new = w1 - 0.001 * w1.grad
            w2_new = w2 - 0.001 * w2.grad
        w1 = w1_new.detach().requires_grad_(True)
        w2 = w2_new.detach().requires_grad_(True)
        ms = (time.perf_counter() - t0) * 1000
        if step >= WARMUP: times.append(ms)
        losses.append(loss.item())
    return times, losses

# ── PyTorch (multi-thread) ──────────────────────────────────────
def bench_pytorch_mt():
    import torch
    import os
    torch.set_num_threads(int(os.cpu_count()))
    w1_np, w2_np, x_np, y_np = make_weights()
    w1 = torch.tensor(w1_np, requires_grad=True)
    w2 = torch.tensor(w2_np, requires_grad=True)
    x_t = torch.tensor(x_np)
    y_t = torch.tensor(y_np)

    times, losses = [], []
    for step in range(N_STEPS + WARMUP):
        t0 = time.perf_counter()
        logits = (x_t @ w1).relu() @ w2
        diff = logits - y_t
        loss = (diff * diff).sum() / BS
        loss.backward()
        with torch.no_grad():
            w1_new = w1 - 0.001 * w1.grad
            w2_new = w2 - 0.001 * w2.grad
        w1 = w1_new.detach().requires_grad_(True)
        w2 = w2_new.detach().requires_grad_(True)
        ms = (time.perf_counter() - t0) * 1000
        if step >= WARMUP: times.append(ms)
        losses.append(loss.item())
    return times, losses

# ── Raw NumPy (manual backward) ─────────────────────────────────
def bench_numpy():
    w1, w2, x, y = make_weights()
    times, losses = [], []
    for step in range(N_STEPS + WARMUP):
        t0 = time.perf_counter()
        # Forward
        z1 = x @ w1
        h = np.maximum(z1, 0)       # relu
        logits = h @ w2
        diff = logits - y
        loss_val = (diff * diff).sum() / BS
        # Backward (manual)
        d_logits = 2 * diff / BS     # d(MSE)/d(logits)
        dw2 = h.T @ d_logits         # d/dW2
        dh = d_logits @ w2.T         # backprop through matmul
        dz1 = dh * (z1 > 0)          # backprop through relu
        dw1 = x.T @ dz1              # d/dW1
        # SGD
        w1 -= 0.001 * dw1
        w2 -= 0.001 * dw2
        ms = (time.perf_counter() - t0) * 1000
        if step >= WARMUP: times.append(ms)
        losses.append(float(loss_val))
    return times, losses

# ── Run all ─────────────────────────────────────────────────────
print("=" * 60)
print("  BobNet Benchmark: BS=69, 784→128→10, MSE+SGD")
print("=" * 60)

results = {}

# NumPy
print("\n── NumPy (manual backward) ──")
t, l = bench_numpy()
avg = sum(t) / len(t)
results["NumPy"] = (avg, l)
print(f"  Average: {avg:.2f} ms/step")
print(f"  Loss: {l[WARMUP]:.4f} → {l[-1]:.4f}")

# PyTorch single-thread
try:
    print("\n── PyTorch (1 thread) ──")
    t, l = bench_pytorch()
    avg = sum(t) / len(t)
    results["PyTorch-1T"] = (avg, l)
    print(f"  Average: {avg:.2f} ms/step")
    print(f"  Loss: {l[WARMUP]:.4f} → {l[-1]:.4f}")
except ImportError:
    print("  (skipped — torch not installed)")

# PyTorch multi-thread
try:
    print("\n── PyTorch (multi-thread) ──")
    t, l = bench_pytorch_mt()
    avg = sum(t) / len(t)
    results["PyTorch-MT"] = (avg, l)
    print(f"  Average: {avg:.2f} ms/step")
    print(f"  Loss: {l[WARMUP]:.4f} → {l[-1]:.4f}")
except ImportError:
    print("  (skipped — torch not installed)")

# Summary
print("\n" + "=" * 60)
print("  Summary (compare with TinyHVM CPU: ~2.2 ms/step)")
print("=" * 60)
for name, (avg, _) in sorted(results.items(), key=lambda x: x[1][0]):
    ratio = avg / 2.24  # TinyHVM baseline
    print(f"  {name:15s}: {avg:6.2f} ms/step  ({ratio:.1f}× vs TinyHVM)")
print(f"  {'TinyHVM CPU':15s}: {'2.24':>6s} ms/step  (1.0× baseline)")
