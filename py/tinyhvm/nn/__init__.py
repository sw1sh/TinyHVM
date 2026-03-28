"""Neural network layers — tinygrad-compatible API."""
from __future__ import annotations
import math
from typing import Optional
from tinyhvm.tensor import Tensor


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        bound = 1.0 / math.sqrt(in_features)
        self.weight = Tensor.rand(in_features, out_features, requires_grad=True) * (2 * bound) - bound
        self.bias = Tensor.rand(out_features, requires_grad=True) * (2 * bound) - bound if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class Conv2d:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, groups: int = 1, bias: bool = True):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        bound = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = Tensor.rand(out_channels, in_channels // groups, kernel_size, kernel_size,
                                  requires_grad=True) * (2 * bound) - bound
        self.bias = Tensor.rand(out_channels, requires_grad=True) * (2 * bound) - bound if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        return x.conv2d(self.weight, self.bias,
                        stride=self.stride, padding=self.padding, groups=self.groups)


class BatchNorm2d:
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Tensor.ones(num_features, requires_grad=True)
        self.bias = Tensor.zeros(num_features, requires_grad=True)
        self.running_mean = Tensor.zeros(num_features)
        self.running_var = Tensor.ones(num_features)
        self.training = True

    def __call__(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W]
        if self.training:
            mean = x.mean(axis=(0, 2, 3), keepdim=True)
            var = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdim=True)
        else:
            mean = self.running_mean.reshape(1, self.num_features, 1, 1)
            var = self.running_var.reshape(1, self.num_features, 1, 1)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        w = self.weight.reshape(1, self.num_features, 1, 1)
        b = self.bias.reshape(1, self.num_features, 1, 1)
        return x_norm * w + b
