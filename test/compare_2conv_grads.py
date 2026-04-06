"""Compare TinyHVM 2-conv+BN+maxpool gradients against NumPy reference.

Reads the same random seed as test_ch32.m (srand(42), uniform [-1,1]).
The C rand() sequence must match — this script uses ctypes to call the
system's rand() with srand(42) for exact parity.
"""
import ctypes, struct, math
import numpy as np

libc = ctypes.CDLL(None)
libc.srand(42)

def crand_f32():
    return libc.rand() / 2147483647.0 * 2 - 1

def crand_array(n):
    return np.array([crand_f32() for _ in range(n)], dtype=np.float32)

BS, Cin, H, W, Cout, K = 4, 1, 14, 14, 32, 3

# Input
x = crand_array(BS*Cin*H*W).reshape(BS, Cin, H, W)

# Weights: b * crand where b = 1/sqrt(fan_in)
def mkw(shape, fan_in):
    n = int(np.prod(shape))
    b = 1.0 / math.sqrt(fan_in)
    return (b * crand_array(n)).reshape(shape)

w1 = mkw((Cout, Cin, K, K), Cin*K*K)
b1 = np.zeros(Cout, dtype=np.float32)
# consume rand for b1 tensor creation (test uses mkz which calls calloc, no rand)

w2 = mkw((Cout, Cout, K, K), Cout*K*K)
b2 = np.zeros(Cout, dtype=np.float32)

# BN params
bn_g = np.ones(Cout, dtype=np.float32)
bn_b = np.zeros(Cout, dtype=np.float32)
bn_rm = np.zeros(Cout, dtype=np.float32)
bn_rv = np.ones(Cout, dtype=np.float32)

# Forward: conv1
def conv2d_forward(x, w, b, stride=1):
    BS, Cin, H, W = x.shape
    Cout, _, K, _ = w.shape
    OH = (H - K) // stride + 1
    OW = (W - K) // stride + 1
    out = np.zeros((BS, Cout, OH, OW), dtype=np.float32)
    for n in range(BS):
        for co in range(Cout):
            for oh in range(OH):
                for ow in range(OW):
                    val = b[co]
                    for ci in range(Cin):
                        for kh in range(K):
                            for kw in range(K):
                                val += x[n, ci, oh*stride+kh, ow*stride+kw] * w[co, ci, kh, kw]
                    out[n, co, oh, ow] = val
    return out

h1 = conv2d_forward(x, w1, b1)
h1r = np.maximum(h1, 0)  # ReLU

h2 = conv2d_forward(h1r, w2, b2)
h2r = np.maximum(h2, 0)  # ReLU

# BN forward (training mode)
eps = 1e-5
OH2 = H - K + 1 - K + 1  # 10
mean = h2r.mean(axis=(0, 2, 3))  # (Cout,)
var = h2r.var(axis=(0, 2, 3))    # (Cout,)
h_bn = bn_g.reshape(1,-1,1,1) * (h2r - mean.reshape(1,-1,1,1)) / np.sqrt(var.reshape(1,-1,1,1) + eps) + bn_b.reshape(1,-1,1,1)

# Maxpool 2x2
PH = OH2 // 2
h_pool = np.zeros((BS, Cout, PH, PH), dtype=np.float32)
for n in range(BS):
    for c in range(Cout):
        for ph in range(PH):
            for pw in range(PH):
                h_pool[n,c,ph,pw] = h_bn[n,c,ph*2:ph*2+2,pw*2:pw*2+2].max()

# Flatten + sum = loss
h_flat = h_pool.reshape(BS, -1)
loss = h_flat.sum()

# Backward: d(loss)/d(h_flat) = 1
dh_flat = np.ones_like(h_flat)
dh_pool = dh_flat.reshape(BS, Cout, PH, PH)

# Maxpool backward
dh_bn = np.zeros_like(h_bn)
for n in range(BS):
    for c in range(Cout):
        for ph in range(PH):
            for pw in range(PH):
                window = h_bn[n,c,ph*2:ph*2+2,pw*2:pw*2+2]
                idx = np.unravel_index(window.argmax(), window.shape)
                dh_bn[n,c,ph*2+idx[0],pw*2+idx[1]] += dh_pool[n,c,ph,pw]

# BN backward
N_bn = BS * OH2 * OH2
x_hat = (h2r - mean.reshape(1,-1,1,1)) / np.sqrt(var.reshape(1,-1,1,1) + eps)
dh2r = bn_g.reshape(1,-1,1,1) / np.sqrt(var.reshape(1,-1,1,1) + eps) / N_bn * (
    N_bn * dh_bn - dh_bn.sum(axis=(0,2,3)).reshape(1,-1,1,1) - x_hat * (dh_bn * x_hat).sum(axis=(0,2,3)).reshape(1,-1,1,1)
)

# ReLU backward
dh2 = dh2r * (h2 > 0)

# Conv2 backward: dw2
def conv2d_backward_w(x, dy, K):
    BS, Cin, H, W = x.shape
    _, Cout, OH, OW = dy.shape
    dw = np.zeros((Cout, Cin, K, K), dtype=np.float32)
    for n in range(BS):
        for co in range(Cout):
            for ci in range(Cin):
                for kh in range(K):
                    for kw in range(K):
                        for oh in range(OH):
                            for ow in range(OW):
                                dw[co, ci, kh, kw] += x[n, ci, oh+kh, ow+kw] * dy[n, co, oh, ow]
    return dw

dw2 = conv2d_backward_w(h1r, dh2, K)

# Conv2 backward: dx (for conv1 backward)
def conv2d_backward_x(w, dy, input_shape):
    BS, Cin, H, W = input_shape
    Cout, _, K, _ = w.shape
    _, _, OH, OW = dy.shape
    dx = np.zeros((BS, Cin, H, W), dtype=np.float32)
    for n in range(BS):
        for ci in range(Cin):
            for h in range(H):
                for w_ in range(W):
                    for co in range(Cout):
                        for kh in range(K):
                            for kw in range(K):
                                oh = h - kh
                                ow = w_ - kw
                                if 0 <= oh < OH and 0 <= ow < OW:
                                    dx[n, ci, h, w_] += w[co, ci, kh, kw] * dy[n, co, oh, ow]
    return dx

dh1r = conv2d_backward_x(w2, dh2, h1r.shape)

# ReLU backward
dh1 = dh1r * (h1 > 0)

# Conv1 backward: dw1
dw1 = conv2d_backward_w(x, dh1, K)

n1 = math.sqrt((dw1**2).sum())
n2 = math.sqrt((dw2**2).sum())
print(f"numpy : w1={n1:.2f} w2={n2:.2f}")
