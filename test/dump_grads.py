#!/usr/bin/env python3
"""Dump weights and gradients from tinygrad for the skip MLP.
Model: h = relu(X@W1+B1), out = h@W2+B2 + expand(sum(h.reshape(BS*H)))
One step, BS=128, H=128, cross-entropy loss.
Dumps: init weights, gradients after 1 step.
"""
import numpy as np
np.random.seed(42)

BS, H = 128, 128

# Same Kaiming init as TinyHVM
def kaiming(fan_in, shape):
    bound = 1.0 / np.sqrt(fan_in)
    return (np.random.rand(*shape).astype(np.float32) * 2 - 1) * bound

W1 = kaiming(784, (784, H))
B1 = np.zeros((1, H), dtype=np.float32)
W2 = kaiming(H, (H, 10))
B2 = np.zeros((1, 10), dtype=np.float32)

# Load MNIST
import struct, os
def load_mnist_images(path):
    with open(path, 'rb') as f:
        _, n, r, c = struct.unpack('>4I', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r*c).astype(np.float32) / 255.0

def load_mnist_labels(path):
    with open(path, 'rb') as f:
        _, n = struct.unpack('>2I', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

X_all = load_mnist_images('data/train-images-idx3-ubyte')
Y_labels = load_mnist_labels('data/train-labels-idx1-ubyte')

X = X_all[:BS]
Y_oh = np.zeros((BS, 10), dtype=np.float32)
for i in range(BS): Y_oh[i, Y_labels[i]] = 1.0

try:
    from tinygrad import Tensor
    Tensor.manual_seed(42)

    tW1 = Tensor(W1, requires_grad=True)
    tB1 = Tensor(B1, requires_grad=True)
    tW2 = Tensor(W2, requires_grad=True)
    tB2 = Tensor(B2, requires_grad=True)
    tX = Tensor(X)
    tY = Tensor(Y_oh)

    # Forward with skip
    z1 = tX @ tW1 + tB1
    h = z1.relu()
    h_skip = h.reshape(BS * H).sum().reshape(1, 1).expand(BS, 10)
    out = h @ tW2 + tB2 + h_skip

    # Cross-entropy
    x_max = out.max(axis=1, keepdim=True)
    shifted = out - x_max
    e = shifted.exp()
    e_sum = e.sum(axis=1, keepdim=True)
    probs = e / e_sum
    clamped = probs.maximum(Tensor(1e-7))
    log_probs = clamped.log()
    masked = tY * log_probs
    loss = -(masked.sum() / BS)

    loss.backward()

    print(f"loss = {loss.item():.6f}")
    print(f"gW1[0:4] = {tW1.grad.numpy().flatten()[:4]}")
    print(f"gB1[0:4] = {tB1.grad.numpy().flatten()[:4]}")
    print(f"gW2[0:4] = {tW2.grad.numpy().flatten()[:4]}")
    print(f"gB2[0:4] = {tB2.grad.numpy().flatten()[:4]}")

    # Dump binary for TinyHVM comparison
    os.makedirs('/tmp/tinygrad_grads', exist_ok=True)
    np.save('/tmp/tinygrad_grads/gW1.npy', tW1.grad.numpy())
    np.save('/tmp/tinygrad_grads/gB1.npy', tB1.grad.numpy())
    np.save('/tmp/tinygrad_grads/gW2.npy', tW2.grad.numpy())
    np.save('/tmp/tinygrad_grads/gB2.npy', tB2.grad.numpy())
    np.save('/tmp/tinygrad_grads/W1.npy', W1)
    np.save('/tmp/tinygrad_grads/loss.npy', np.array([loss.item()]))
    print("Saved to /tmp/tinygrad_grads/")

except ImportError:
    # Fallback: numpy manual computation
    print("tinygrad not available, computing with numpy...")

    z1 = X @ W1 + B1
    h = np.maximum(z1, 0)
    h_skip = h.reshape(BS * H).sum().reshape(1, 1) * np.ones((BS, 10), dtype=np.float32)
    out = h @ W2 + B2 + h_skip

    # Softmax
    x_max = out.max(axis=1, keepdims=True)
    shifted = out - x_max
    e = np.exp(shifted)
    e_sum = e.sum(axis=1, keepdims=True)
    probs = e / e_sum
    clamped = np.maximum(probs, 1e-7)
    log_probs = np.log(clamped)
    masked = Y_oh * log_probs
    loss = -(masked.sum() / BS)

    print(f"loss = {loss:.6f}")

    # Manual backward
    # d_loss/d_out = (probs - Y_oh) / BS  (standard softmax+CE gradient)
    d_out = (probs - Y_oh) / BS

    # out = h@W2 + B2 + h_skip
    # d_h_from_mm = d_out @ W2.T
    # d_W2 = h.T @ d_out
    # d_B2 = d_out.sum(axis=0, keepdims=True)
    # d_h_skip = d_out (broadcast sum)
    # d_h_from_skip = d_h_skip.sum() * ones(BS, H) / (BS*H... no, chain through reshape+sum)
    # h_skip = sum(h.reshape(BS*H)).reshape(1,1).expand(BS,10)
    # d_sum = d_h_skip.sum() = d_out.sum()
    # d_h_flat = d_sum * ones(BS*H)
    # d_h_from_skip = d_h_flat.reshape(BS, H)

    d_h_mm = d_out @ W2.T
    d_W2 = h.T @ d_out
    d_B2 = d_out.sum(axis=0, keepdims=True)

    d_sum_scalar = d_out.sum()
    d_h_skip_back = np.ones((BS, H), dtype=np.float32) * d_sum_scalar

    d_h = d_h_mm + d_h_skip_back

    # relu backward
    d_z1 = d_h * (z1 > 0).astype(np.float32)

    d_W1 = X.T @ d_z1
    d_B1 = d_z1.sum(axis=0, keepdims=True)

    print(f"gW1[0:4] = {d_W1.flatten()[:4]}")
    print(f"gB1[0:4] = {d_B1.flatten()[:4]}")
    print(f"gW2[0:4] = {d_W2.flatten()[:4]}")
    print(f"gB2[0:4] = {d_B2.flatten()[:4]}")

    os.makedirs('/tmp/tinygrad_grads', exist_ok=True)
    np.save('/tmp/tinygrad_grads/gW1.npy', d_W1)
    np.save('/tmp/tinygrad_grads/gB1.npy', d_B1)
    np.save('/tmp/tinygrad_grads/gW2.npy', d_W2)
    np.save('/tmp/tinygrad_grads/gB2.npy', d_B2)
    np.save('/tmp/tinygrad_grads/W1.npy', W1)
    np.save('/tmp/tinygrad_grads/loss.npy', np.array([loss]))
    print("Saved to /tmp/tinygrad_grads/")
