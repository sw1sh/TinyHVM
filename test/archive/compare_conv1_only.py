"""Minimal: single conv + relu + sum. Compare TinyHVM vs numpy."""
import ctypes, math
import numpy as np

libc = ctypes.CDLL(None)
libc.srand(42)
def cr(): return libc.rand() / 2147483647.0 * 2 - 1
def cra(n): return np.array([cr() for _ in range(n)], dtype=np.float32)

BS, Cin, H, W, Cout, K = 4, 1, 14, 14, 32, 3
x = cra(BS*Cin*H*W).reshape(BS, Cin, H, W)
b = 1.0 / math.sqrt(Cin*K*K)
w = (b * cra(Cout*Cin*K*K)).reshape(Cout, Cin, K, K)
bias = np.zeros(Cout, dtype=np.float32)

# Forward conv
OH = H - K + 1
out = np.zeros((BS, Cout, OH, OH), dtype=np.float32)
for n in range(BS):
    for co in range(Cout):
        for oh in range(OH):
            for ow in range(OH):
                v = bias[co]
                for ci in range(Cin):
                    for kh in range(K):
                        for kw in range(K):
                            v += x[n,ci,oh+kh,ow+kw] * w[co,ci,kh,kw]
                out[n,co,oh,ow] = v
# ReLU
h = np.maximum(out, 0)
loss = h.sum()
# Backward
dh = (out > 0).astype(np.float32)  # relu grad
# dw
dw = np.zeros_like(w)
for n in range(BS):
    for co in range(Cout):
        for ci in range(Cin):
            for kh in range(K):
                for kw in range(K):
                    for oh in range(OH):
                        for ow in range(OH):
                            dw[co,ci,kh,kw] += x[n,ci,oh+kh,ow+kw] * dh[n,co,oh,ow]
print(f"numpy conv1 dw norm: {math.sqrt((dw**2).sum()):.2f}")
print(f"numpy conv1 loss:    {loss:.4f}")
