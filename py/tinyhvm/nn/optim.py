"""Optimizers — tinygrad-compatible API."""
from __future__ import annotations
from typing import List
from tinyhvm.tensor import Tensor
from tinyhvm import _ffi as F


class SGD:
    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = [None] * len(params) if momentum > 0 else None

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, grads: List[Tensor]):
        """Apply gradients. Pass pre-computed grads (one per param)."""
        for i, (p, g) in enumerate(zip(self.params, grads)):
            g = g.realize()
            if self.momentum > 0:
                if self.velocity[i] is None:
                    self.velocity[i] = g * self.momentum
                else:
                    self.velocity[i] = self.velocity[i] * self.momentum + g
                update = self.velocity[i]
            else:
                update = g
            # p = p - lr * grad  (via assign for in-place update)
            new_val = p - update * self.lr
            new_val.realize()
            p._t = F.lib.py_assign(F.get_ctx(), p._t, new_val._t)


class Adam:
    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [Tensor.zeros(*p.shape) for p in params]
        self.v = [Tensor.zeros(*p.shape) for p in params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, grads: List[Tensor]):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            g = g.realize()
            self.m[i] = self.m[i] * self.b1 + g * (1 - self.b1)
            self.v[i] = self.v[i] * self.b2 + (g * g) * (1 - self.b2)
            self.m[i].realize()
            self.v[i].realize()
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            new_val = p - m_hat * self.lr / (v_hat.sqrt() + self.eps)
            new_val.realize()
            p._t = F.lib.py_assign(F.get_ctx(), p._t, new_val._t)
