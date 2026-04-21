#!/usr/bin/env python3
"""bench_bobnet.py — Tinygrad TinyBobNet benchmark (same dimensions as TinyHVM)"""
import time
import numpy as np
from tinygrad import Tensor, Device

BS, IN, H, OUT = 69, 784, 128, 10
N_STEPS, WARMUP = 20, 3

print(f"=== BobNet Benchmark (tinygrad, {Device.DEFAULT}) ===")
print(f"    BS={BS}, {IN}→{H}→{OUT}, {N_STEPS} steps\n")

np.random.seed(1337)
l1 = Tensor(np.random.randn(IN, H).astype(np.float32) * np.sqrt(2.0/(IN+H)), requires_grad=True)
l2 = Tensor(np.random.randn(H, OUT).astype(np.float32) * np.sqrt(2.0/(H+OUT)), requires_grad=True)

x_np = np.random.randn(BS, IN).astype(np.float32)
y_np = np.zeros((BS, OUT), dtype=np.float32)
for i in range(BS): y_np[i, i % OUT] = 1.0

times = []
for step in range(N_STEPS + WARMUP):
    t0 = time.perf_counter()

    x = Tensor(x_np)
    y = Tensor(y_np)

    # Forward: x.dot(l1).relu().dot(l2)
    logits = x.dot(l1).relu().dot(l2)

    # MSE loss
    diff = logits - y
    loss = (diff * diff).sum() / BS

    # Backward
    loss.backward()

    # SGD update
    l1_np = l1.numpy() - 0.001 * l1.grad.numpy()
    l2_np = l2.numpy() - 0.001 * l2.grad.numpy()
    l1 = Tensor(l1_np, requires_grad=True)
    l2 = Tensor(l2_np, requires_grad=True)

    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000
    lv = loss.item()

    if step >= WARMUP:
        times.append(ms)

    tag = " (warmup)" if step < WARMUP else ""
    print(f"  step {step:2d}: loss={lv:.4f}  {ms:.2f}ms{tag}")

avg = sum(times) / len(times)
print(f"\n  Average: {avg:.2f} ms/step ({N_STEPS} steps)")
print(f"  Throughput: {BS * 1000 / avg:.0f} samples/sec")
