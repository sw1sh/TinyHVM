#!/usr/bin/env python3
"""Compare TinyHVM CNN+Pool against tinygrad — same architecture, same init."""
import time, os, struct
import numpy as np

# Check if tinygrad is available
try:
    os.environ['DEBUG'] = '0'
    from tinygrad import Tensor, dtypes
    from tinygrad.nn import Conv2d, Linear
    from tinygrad.nn.optim import SGD
    HAS_TINYGRAD = True
except ImportError:
    HAS_TINYGRAD = False
    print("tinygrad not available — skipping tinygrad comparison")

# Load MNIST
def load_mnist(path="data"):
    def load(f):
        with open(f, 'rb') as g:
            magic = struct.unpack('>I', g.read(4))[0]
            n = struct.unpack('>I', g.read(4))[0]
            if magic == 2051:
                rows, cols = struct.unpack('>II', g.read(8))
                return np.frombuffer(g.read(), dtype=np.uint8).reshape(n, rows, cols)
            else:
                return np.frombuffer(g.read(), dtype=np.uint8)
    X_train = load(f"{path}/train-images-idx3-ubyte").astype(np.float32) / 255.0
    Y_train = load(f"{path}/train-labels-idx1-ubyte")
    X_test = load(f"{path}/t10k-images-idx3-ubyte").astype(np.float32) / 255.0
    Y_test = load(f"{path}/t10k-labels-idx1-ubyte")
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = load_mnist()
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

BS = 64
n_steps = 100
lr = 0.001

if HAS_TINYGRAD:
    Tensor.manual_seed(42)
    np.random.seed(42)

    # Architecture: Conv(1,8,3)→ReLU→Pool→Conv(8,16,3)→ReLU→Pool→Conv(16,32,3,p=1)→ReLU→Flatten→Linear(800,10)
    class CNN:
        def __init__(self):
            self.conv1 = Conv2d(1, 8, 3, bias=True)
            self.conv2 = Conv2d(8, 16, 3, bias=True)
            self.conv3 = Conv2d(16, 32, 3, padding=1, bias=True)
            self.fc = Linear(32*5*5, 10)

        def __call__(self, x):
            x = self.conv1(x).relu().max_pool2d((2,2))
            x = self.conv2(x).relu().max_pool2d((2,2))
            x = self.conv3(x).relu()
            x = x.reshape(x.shape[0], -1)
            return self.fc(x)

        def parameters(self):
            return [self.conv1.weight, self.conv1.bias,
                    self.conv2.weight, self.conv2.bias,
                    self.conv3.weight, self.conv3.bias,
                    self.fc.weight, self.fc.bias]

    Tensor.training = True
    model = CNN()
    opt = SGD(model.parameters(), lr=lr)

    print(f"tinygrad: Conv(1,8,3)→Pool→Conv(8,16,3)→Pool→Conv(16,32,3,p=1)→FC(800,10)")
    print(f"BS={BS}, lr={lr}, steps={n_steps}")

    t0 = time.time()
    for step in range(n_steps):
        bi = step % (len(X_train) // BS)
        xb = Tensor(X_train[bi*BS:(bi+1)*BS])
        yb = Y_train[bi*BS:(bi+1)*BS]

        logits = model(xb)
        loss = logits.cross_entropy(Tensor(yb.astype(np.int32)))

        opt.zero_grad()
        loss.backward()
        opt.step()

        lv = loss.item()
        if step < 3 or step % 20 == 0 or step == n_steps - 1:
            print(f"  step {step:3d}: loss={lv:.4f}")

    total_s = time.time() - t0
    print(f"\n  {total_s*1000/n_steps:.0f}ms/step avg")

    # Eval
    correct = 0
    for b in range(len(X_test) // BS):
        xb = Tensor(X_test[b*BS:(b+1)*BS])
        logits = model(xb)
        preds = logits.argmax(axis=1).numpy()
        correct += (preds == Y_test[b*BS:(b+1)*BS]).sum()
    total = (len(X_test) // BS) * BS
    print(f"  Test accuracy: {100*correct/total:.1f}% ({correct}/{total})")

    # Count dispatches
    try:
        from tinygrad.engine.jit import TinyJit
        from tinygrad.helpers import GlobalCounters
        GlobalCounters.reset()
        xb = Tensor(X_train[:BS])
        yb = Y_train[:BS]
        logits = model(xb)
        loss = logits.cross_entropy(Tensor(yb.astype(np.int32)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss.item()
        print(f"  Dispatches (last step): {GlobalCounters.kernel_count}")
    except Exception as e:
        print(f"  (dispatch count not available: {e})")
