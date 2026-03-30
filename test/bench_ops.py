#!/usr/bin/env python3
"""bench_ops.py — Side-by-side kernel benchmarks (tinygrad)
Run with: DEBUG=2 python3 test/bench_ops.py
Compare output with: THVM_DEBUG=2 bin/bench_ops

For beautiful_mnist kernel breakdown:
  DEBUG=2 python3 test/bench_ops.py --beautiful
"""
from tinygrad import Tensor, GlobalCounters, Device
import time, sys

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

if "--beautiful" in sys.argv:
    # beautiful_mnist kernel breakdown (warm step)
    from tinygrad import nn, TinyJit
    from tinygrad.nn.datasets import mnist
    class Model:
      def __init__(self):
        self.layers = [nn.Conv2d(1,32,5),Tensor.relu,nn.Conv2d(32,32,5),Tensor.relu,
          nn.BatchNorm(32),Tensor.max_pool2d,nn.Conv2d(32,64,3),Tensor.relu,
          nn.Conv2d(64,64,3),Tensor.relu,nn.BatchNorm(64),Tensor.max_pool2d,
          lambda x:x.flatten(1),nn.Linear(576,10)]
      def __call__(self,x): return x.sequential(self.layers)
    X_train,Y_train,_,_=mnist()
    model=Model(); opt=nn.optim.Adam(nn.state.get_parameters(model))
    @Tensor.train()
    def step():
        opt.zero_grad()
        s=Tensor.randint(512,high=X_train.shape[0])
        loss=model(X_train[s]).sparse_categorical_crossentropy(Y_train[s]).backward()
        return loss.realize(*opt.schedule_step())
    # warmup 2 steps
    for _ in range(2): step()
    # profile step 2
    GlobalCounters.reset()
    t0=time.perf_counter()
    loss=step()
    t1=time.perf_counter()
    lv=loss.item()
    t2=time.perf_counter()
    print(f"=== tinygrad beautiful_mnist (BS=512, warm step) ===")
    print(f"  Step:    {(t1-t0)*1e3:.1f}ms")
    print(f"  Read:    {(t2-t1)*1e3:.1f}ms")
    print(f"  Total:   {(t2-t0)*1e3:.1f}ms")
    print(f"  Kernels: {GlobalCounters.kernel_count}")
    print(f"  GPU:     {GlobalCounters.time_sum_s*1e3:.1f}ms")
    print(f"  Params:  {sum(p.numel() for p in nn.state.get_parameters(model))}")
    sys.exit(0)

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


# === FUSION COUNTING ===
print("\n--- Fusion Tests (kernel count per pattern) ---")
print(f"{'Pattern':<40s} {'Kernels':>8s}")
print("─"*50)

def count_kernels(setup_fn, run_fn):
    """Count kernels for run_fn only, not setup."""
    tensors = setup_fn()
    for t in tensors:
        if isinstance(t, Tensor): t.realize()
    GlobalCounters.reset()
    run_fn(*tensors).realize()
    return GlobalCounters.kernel_count

# F1: x+y
k = count_kernels(lambda: (Tensor.rand(1024).realize(), Tensor.rand(1024).realize()),
                  lambda a,b: a+b)
print(f"{'x + y [1024]':<40s} {k:8d}")

# F2: relu(x*w+b)
k = count_kernels(lambda: (Tensor.rand(1024).realize(), Tensor.rand(1024).realize(), Tensor.rand(1024).realize()),
                  lambda x,w,b: (x*w+b).relu())
print(f"{'relu(x*w+b) [1024]':<40s} {k:8d}")

# F3: sum(x*w+b)
k = count_kernels(lambda: (Tensor.rand(1024).realize(), Tensor.rand(1024).realize(), Tensor.rand(1024).realize()),
                  lambda x,w,b: (x*w+b).sum())
print(f"{'sum(x*w+b) [1024→1]':<40s} {k:8d}")

# F4: relu(mm(x,w)+b)
k = count_kernels(lambda: (Tensor.rand(64,32).realize(), Tensor.rand(32,16).realize(), Tensor.rand(16).realize()),
                  lambda x,w,b: (x@w+b).relu())
print(f"{'relu(mm(x,w)+b)':<40s} {k:8d}")

# F5: sum(relu(mm(x,w)+b))
k = count_kernels(lambda: (Tensor.rand(64,32).realize(), Tensor.rand(32,16).realize(), Tensor.rand(16).realize()),
                  lambda x,w,b: (x@w+b).relu().sum())
print(f"{'sum(relu(mm(x,w)+b))':<40s} {k:8d}")

# F6: backward through sum(relu(mm+b))
x = Tensor.rand(64,32, requires_grad=True).realize()
w = Tensor.rand(32,16, requires_grad=True).realize()
b = Tensor.rand(16, requires_grad=True).realize()
GlobalCounters.reset()
loss = (x@w+b).relu().sum()
loss.backward()
Tensor.realize(x.grad, w.grad, b.grad)
k = GlobalCounters.kernel_count
print(f"{'backward(sum(relu(mm+b)))':<40s} {k:8d}")

print()
