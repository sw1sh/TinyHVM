#!/usr/bin/env python3
"""bench_ops.py — Side-by-side kernel benchmarks (tinygrad)
Run with: DEBUG=2 python3 test/bench_ops.py
Compare output with: THVM_DEBUG=2 bin/bench_ops
"""
from tinygrad import Tensor, GlobalCounters, Device
import time

def bench(label, n_warmup, n_iter, fn):
    # Warmup
    for _ in range(n_warmup):
        fn().realize()

    GlobalCounters.reset()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn().realize()
    t1 = time.perf_counter()
    total_ms = (t1-t0)*1000
    print(f"{label:30s} {total_ms:7.2f}ms/{n_iter}  ({total_ms/n_iter*1000:.2f}us/iter)"
          f"  kernels={GlobalCounters.kernel_count}  GPU={GlobalCounters.time_sum_s*1e3:.2f}ms")
    GlobalCounters.reset()

print("=== tinygrad Op Benchmarks ===\n")

# 1. Elementwise ADD [4096]
a = Tensor.rand(4096); b = Tensor.rand(4096)
bench("ew_add [4096]", 3, 100, lambda: a+b)

# 2. Elementwise MUL [512,512]
a = Tensor.rand(512,512); b = Tensor.rand(512,512)
bench("ew_mul [512,512]", 3, 50, lambda: a*b)

# 3. Reduce SUM [512,512] → [512]
a = Tensor.rand(512,512)
bench("reduce_sum [512→1]", 3, 50, lambda: a.sum(axis=1))

# 4. Matmul [64,784]×[784,128]
a = Tensor.rand(64,784); b = Tensor.rand(784,128)
bench("matmul [64,784]×[784,128]", 3, 50, lambda: a@b)

# 5. Fused MUL+ADD [65536] (x*w+b)
x = Tensor.rand(65536); w = Tensor.rand(65536); b = Tensor.rand(65536)
bench("fused_mul_add [65536]", 3, 50, lambda: x*w+b)

print()
