"""Tensor — tinygrad-compatible API backed by TinyHVM interaction net runtime."""
from __future__ import annotations
import ctypes, math, random
from typing import Optional, Tuple
from tinyhvm import _ffi as F
from tinyhvm.dtype import DType, dtypes

# ── Helpers for pure-Python nested list → flat float array ────────────────────

def _shape_of(data) -> Tuple[int, ...]:
    """Infer shape from nested lists/tuples."""
    if isinstance(data, (int, float)): return ()
    if isinstance(data, (list, tuple)):
        if len(data) == 0: return (0,)
        inner = _shape_of(data[0])
        return (len(data),) + inner
    raise TypeError(f"Cannot infer shape from {type(data)}")

def _flatten(data, out: list):
    """Flatten nested list/tuple into flat list of floats."""
    if isinstance(data, (int, float)):
        out.append(float(data))
    elif isinstance(data, (list, tuple)):
        for item in data:
            _flatten(item, out)
    else:
        raise TypeError(f"Cannot flatten {type(data)}")

def _unflatten(flat: list, shape: Tuple[int, ...], offset: list):
    """Reconstruct nested list from flat data + shape."""
    if len(shape) == 0:
        val = flat[offset[0]]; offset[0] += 1
        return val
    if len(shape) == 1:
        result = flat[offset[0]:offset[0] + shape[0]]
        offset[0] += shape[0]
        return result
    return [_unflatten(flat, shape[1:], offset) for _ in range(shape[0])]


class Tensor:
    """Lazy tensor backed by a TinyHVM Term (u64)."""

    training = False

    def __init__(self, data=None, *, requires_grad=False, _term: int = 0):
        if _term:
            self._t = _term
        elif data is not None:
            shape = _shape_of(data)
            flat: list = []
            _flatten(data, flat)
            if len(flat) == 0:
                raise ValueError("Cannot create empty tensor")
            c_data = F.make_f32_array(flat)
            c_dims = F.make_u32_array(shape) if shape else F.make_u32_array([1])
            rank = len(shape) if shape else 1
            if not shape:
                # scalar → shape (1,)
                shape = (1,)
            self._t = F.lib.py_tensor(F.get_ctx(), c_data, c_dims, rank)
        else:
            raise ValueError("Must provide data or _term")

        self.grad: Optional[Tensor] = None
        self._requires_grad = requires_grad
        if requires_grad:
            F.lib.py_set_requires_grad(F.get_ctx(), self._t)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def shape(self) -> Tuple[int, ...]:
        dims = (F.c_u32 * 8)()
        rank = F.c_u32()
        F.lib.py_shape(F.get_ctx(), self._t, dims, ctypes.byref(rank))
        return tuple(int(dims[i]) for i in range(rank.value))

    @property
    def ndim(self) -> int: return len(self.shape)

    @property
    def numel(self) -> int:
        return int(F.lib.py_numel(F.get_ctx(), self._t))

    @property
    def dtype(self) -> DType: return dtypes.float32  # TODO: multi-dtype

    @property
    def requires_grad(self) -> bool: return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._requires_grad = val
        if val:
            F.lib.py_set_requires_grad(F.get_ctx(), self._t)

    # ── Realization & data access ─────────────────────────────────────────────

    def realize(self) -> Tensor:
        self._t = F.lib.py_reduce(F.get_ctx(), self._t)
        return self

    def _host_floats(self) -> list:
        """Read materialized tensor data as flat list of Python floats."""
        self.realize()
        ptr = F.lib.py_to_host(F.get_ctx(), self._t)
        n = self.numel
        addr = ctypes.cast(ptr, ctypes.c_void_p).value
        buf = (ctypes.c_float * n).from_address(addr)
        return [float(buf[i]) for i in range(n)]

    def tolist(self):
        """Convert to nested Python list (like tinygrad)."""
        flat = self._host_floats()
        shape = self.shape
        if len(shape) == 0 or shape == (1,):
            return flat[0] if len(flat) == 1 else flat
        return _unflatten(flat, shape, [0])

    def numpy(self):
        """Convert to numpy array (optional numpy dependency)."""
        import numpy as np
        flat = self._host_floats()
        return np.array(flat, dtype=np.float32).reshape(self.shape)

    def item(self) -> float:
        flat = self._host_floats()
        if len(flat) != 1:
            raise ValueError(f"item() requires exactly 1 element, got {len(flat)}")
        return flat[0]

    # ── Factory methods ───────────────────────────────────────────────────────

    @staticmethod
    def _from_term(t: int) -> Tensor:
        return Tensor(_term=t)

    @staticmethod
    def _scalar(val: float) -> Tensor:
        return Tensor._from_term(F.lib.py_scalar(F.get_ctx(), val))

    @staticmethod
    def zeros(*shape: int, requires_grad=False, **_kw) -> Tensor:
        return Tensor._make_full(shape or (1,), 0.0, requires_grad)

    @staticmethod
    def ones(*shape: int, requires_grad=False, **_kw) -> Tensor:
        return Tensor._make_full(shape or (1,), 1.0, requires_grad)

    @staticmethod
    def full(shape, fill_value: float, requires_grad=False, **_kw) -> Tensor:
        if isinstance(shape, int): shape = (shape,)
        return Tensor._make_full(shape, fill_value, requires_grad)

    @staticmethod
    def _make_full(shape, val: float, requires_grad: bool) -> Tensor:
        n = math.prod(shape)
        data = [val] * n
        c_data = F.make_f32_array(data)
        c_dims = F.make_u32_array(shape)
        t = F.lib.py_tensor(F.get_ctx(), c_data, c_dims, len(shape))
        return Tensor(_term=t, requires_grad=requires_grad)

    @staticmethod
    def eye(n: int, requires_grad=False, **_kw) -> Tensor:
        data = [0.0] * (n * n)
        for i in range(n): data[i * n + i] = 1.0
        c_data = F.make_f32_array(data)
        c_dims = F.make_u32_array([n, n])
        t = F.lib.py_tensor(F.get_ctx(), c_data, c_dims, 2)
        return Tensor(_term=t, requires_grad=requires_grad)

    @staticmethod
    def rand(*shape: int, requires_grad=False, **_kw) -> Tensor:
        n = math.prod(shape) if shape else 1
        data = [random.random() for _ in range(n)]
        c_data = F.make_f32_array(data)
        c_dims = F.make_u32_array(shape) if shape else F.make_u32_array([1])
        t = F.lib.py_tensor(F.get_ctx(), c_data, c_dims, len(shape) or 1)
        return Tensor(_term=t, requires_grad=requires_grad)

    @staticmethod
    def randn(*shape: int, requires_grad=False, **_kw) -> Tensor:
        n = math.prod(shape) if shape else 1
        data = [random.gauss(0.0, 1.0) for _ in range(n)]
        c_data = F.make_f32_array(data)
        c_dims = F.make_u32_array(shape) if shape else F.make_u32_array([1])
        t = F.lib.py_tensor(F.get_ctx(), c_data, c_dims, len(shape) or 1)
        return Tensor(_term=t, requires_grad=requires_grad)

    @staticmethod
    def arange(start, stop=None, step=1, **_kw) -> Tensor:
        if stop is None: start, stop = 0, start
        data = []
        v = start
        while (step > 0 and v < stop) or (step < 0 and v > stop):
            data.append(float(v))
            v += step
        if not data: data = [0.0]
        return Tensor(data)

    @staticmethod
    def manual_seed(seed: int):
        random.seed(seed)

    def contiguous(self) -> Tensor:
        """No-op for now — TinyHVM handles contiguity internally."""
        return self

    # ── _like methods ─────────────────────────────────────────────────────────

    def zeros_like(self, **kw) -> Tensor: return Tensor.zeros(*self.shape, **kw)
    def ones_like(self, **kw) -> Tensor: return Tensor.ones(*self.shape, **kw)

    # ── Unary ops ─────────────────────────────────────────────────────────────

    def _unary(self, uop: int) -> Tensor:
        era = F.lib.py_era()
        return Tensor._from_term(F.lib.py_op(F.get_ctx(), uop, self._t, era))

    def neg(self) -> Tensor: return self._unary(F.UOP_NEG)
    def exp(self) -> Tensor: return self._unary(F.UOP_EXP)
    def log(self) -> Tensor: return self._unary(F.UOP_LOG)
    def relu(self) -> Tensor: return self._unary(F.UOP_RELU)
    def sqrt(self) -> Tensor: return self._unary(F.UOP_SQRT)

    def __neg__(self) -> Tensor: return self.neg()

    def abs(self) -> Tensor: return self.relu() + (-self).relu()

    def sigmoid(self) -> Tensor:
        return (self.neg().exp() + Tensor._scalar(1.0)).reciprocal()

    def reciprocal(self) -> Tensor:
        return Tensor._scalar(1.0) / self

    def tanh(self) -> Tensor:
        x2 = self * Tensor._scalar(2.0)
        return (x2.sigmoid() * Tensor._scalar(2.0)) - Tensor._scalar(1.0)

    def softmax(self, axis: int = -1) -> Tensor:
        m = self.max(axis=axis, keepdim=True)
        e = (self - m).exp()
        return e / e.sum(axis=axis, keepdim=True)

    def log_softmax(self, axis: int = -1) -> Tensor:
        m = self.max(axis=axis, keepdim=True)
        e = (self - m).exp()
        return (self - m) - e.sum(axis=axis, keepdim=True).log()

    # ── Binary ops ────────────────────────────────────────────────────────────

    def _binary(self, uop: int, other) -> Tensor:
        other = self._ensure_tensor(other)
        return Tensor._from_term(F.lib.py_op(F.get_ctx(), uop, self._t, other._t))

    @staticmethod
    def _ensure_tensor(x) -> Tensor:
        if isinstance(x, Tensor): return x
        if isinstance(x, (int, float)): return Tensor._scalar(float(x))
        raise TypeError(f"Cannot convert {type(x)} to Tensor")

    def add(self, other) -> Tensor: return self._binary(F.UOP_ADD, other)
    def sub(self, other) -> Tensor: return self._binary(F.UOP_SUB, other)
    def mul(self, other) -> Tensor: return self._binary(F.UOP_MUL, other)
    def div(self, other) -> Tensor: return self._binary(F.UOP_DIV, other)
    def maximum(self, other) -> Tensor: return self._binary(F.UOP_MAX, other)

    def __add__(self, other): return self.add(other)
    def __radd__(self, other): return self._ensure_tensor(other).add(self)
    def __sub__(self, other): return self.sub(other)
    def __rsub__(self, other): return self._ensure_tensor(other).sub(self)
    def __mul__(self, other): return self.mul(other)
    def __rmul__(self, other): return self._ensure_tensor(other).mul(self)
    def __truediv__(self, other): return self.div(other)
    def __rtruediv__(self, other): return self._ensure_tensor(other).div(self)

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            if other == 2.0 or other == 2: return self * self
            if other == 0.5: return self.sqrt()
            return (self.log() * Tensor._scalar(float(other))).exp()
        return (self.log() * other).exp()

    # ── Matmul ────────────────────────────────────────────────────────────────

    def matmul(self, other: Tensor) -> Tensor:
        other = self._ensure_tensor(other)
        return Tensor._from_term(F.lib.py_op(F.get_ctx(), F.UOP_MM, self._t, other._t))

    def dot(self, other: Tensor) -> Tensor: return self.matmul(other)
    def __matmul__(self, other): return self.matmul(other)

    def linear(self, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
        out = self.matmul(weight)
        if bias is not None: out = out + bias
        return out

    # ── Reduction ops ─────────────────────────────────────────────────────────

    def _reduce(self, uop: int, axis=None, keepdim: bool = False) -> Tensor:
        shape = self.shape
        if axis is None:
            axes = list(range(len(shape)))
        elif isinstance(axis, int):
            axes = [axis % len(shape)]
        else:
            axes = [a % len(shape) for a in axis]

        result = self
        for ax in sorted(axes, reverse=True):
            ax_ten = Tensor._scalar(float(ax))
            result = Tensor._from_term(F.lib.py_op(F.get_ctx(), uop, result._t, ax_ten._t))

        # TinyHVM keeps reduced dims as size-1; match tinygrad behavior
        if keepdim:
            # Ensure shape matches original with 1s in reduced dims
            new_shape = list(shape)
            for ax in axes:
                new_shape[ax] = 1
            result = result.reshape(*new_shape)
        else:
            # Squeeze out the reduced dimensions
            new_shape = [s for i, s in enumerate(shape) if i not in axes]
            if new_shape:
                result = result.reshape(*new_shape)
            # else: scalar result, leave as (1,)
        return result

    def sum(self, axis=None, keepdim=False) -> Tensor:
        return self._reduce(F.UOP_SUM, axis, keepdim)

    def max(self, axis=None, keepdim=False) -> Tensor:
        return self._reduce(F.UOP_RMAX, axis, keepdim)

    def mean(self, axis=None, keepdim=False) -> Tensor:
        s = self.sum(axis=axis, keepdim=keepdim)
        shape = self.shape
        if axis is None:
            n = self.numel
        elif isinstance(axis, int):
            n = shape[axis % len(shape)]
        else:
            n = 1
            for a in axis:
                n *= shape[a % len(shape)]
        return s / float(n)

    # ── Movement ops ──────────────────────────────────────────────────────────

    def reshape(self, *shape: int) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if -1 in shape:
            known = 1
            neg_idx = -1
            for i, s in enumerate(shape):
                if s == -1: neg_idx = i
                else: known *= s
            shape = list(shape)
            shape[neg_idx] = self.numel // known
            shape = tuple(shape)
        c_dims = F.make_u32_array(shape)
        return Tensor._from_term(F.lib.py_reshape(F.get_ctx(), self._t, c_dims, len(shape)))

    def expand(self, *shape: int) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        c_dims = F.make_u32_array(shape)
        return Tensor._from_term(F.lib.py_expand(F.get_ctx(), self._t, c_dims, len(shape)))

    def permute(self, *axes: int) -> Tensor:
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        c_axes = F.make_u32_array(axes)
        return Tensor._from_term(F.lib.py_permute(F.get_ctx(), self._t, c_axes, len(axes)))

    @property
    def T(self) -> Tensor:
        return self.permute(*reversed(range(self.ndim)))

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return self.permute(*axes)

    def flatten(self, start_dim: int = 0) -> Tensor:
        shape = self.shape
        new_shape = shape[:start_dim] + (math.prod(shape[start_dim:]),)
        return self.reshape(*new_shape)

    def unsqueeze(self, dim: int) -> Tensor:
        shape = list(self.shape)
        if dim < 0: dim = len(shape) + 1 + dim
        shape.insert(dim, 1)
        return self.reshape(*shape)

    def squeeze(self, dim: Optional[int] = None) -> Tensor:
        shape = self.shape
        if dim is not None:
            if shape[dim] == 1:
                new_shape = list(shape)
                new_shape.pop(dim)
                return self.reshape(*new_shape) if new_shape else self.reshape(1)
            return self
        new_shape = [s for s in shape if s != 1]
        return self.reshape(*(new_shape or [1]))

    # ── Slicing ───────────────────────────────────────────────────────────────

    def shrink(self, pairs) -> Tensor:
        flat = []
        for start, end in pairs:
            flat.extend([start, end])
        c_pairs = F.make_u32_array(flat)
        return Tensor._from_term(F.lib.py_shrink(F.get_ctx(), self._t, c_pairs, len(pairs)))

    def pad(self, pairs) -> Tensor:
        flat = []
        for lo, hi in pairs:
            flat.extend([lo, hi])
        c_pairs = F.make_u32_array(flat)
        return Tensor._from_term(F.lib.py_pad(F.get_ctx(), self._t, c_pairs, len(pairs)))

    def __getitem__(self, idx):
        if isinstance(idx, int): idx = (slice(idx, idx + 1),)
        elif isinstance(idx, slice): idx = (idx,)
        if not isinstance(idx, tuple):
            raise TypeError(f"Unsupported index type: {type(idx)}")
        shape = self.shape
        pairs = []
        squeeze_dims = []
        for i, sl in enumerate(idx):
            if isinstance(sl, int):
                s = sl if sl >= 0 else shape[i] + sl
                pairs.append((s, s + 1))
                squeeze_dims.append(i)
            elif isinstance(sl, slice):
                start = sl.start or 0
                stop = sl.stop if sl.stop is not None else shape[i]
                if start < 0: start += shape[i]
                if stop < 0: stop += shape[i]
                pairs.append((start, stop))
            else:
                raise TypeError(f"Unsupported index element: {type(sl)}")
        for i in range(len(pairs), len(shape)):
            pairs.append((0, shape[i]))
        result = self.shrink(tuple(pairs))
        if squeeze_dims:
            new_shape = [s for i, s in enumerate(result.shape) if i not in squeeze_dims]
            if new_shape:
                result = result.reshape(*new_shape)
        return result

    # ── Autograd ──────────────────────────────────────────────────────────────

    def backward(self):
        raise NotImplementedError(
            "Use Tensor.grad_fn(loss, params) for explicit gradient computation. "
            "Implicit backward() requires a global tape — coming soon."
        )

    @staticmethod
    def grad_fn(y: Tensor, x: Tensor) -> Tensor:
        """Compute dy/dx (lazy — reduce to materialize)."""
        return Tensor._from_term(F.lib.py_grad(F.get_ctx(), y._t, x._t))

    # ── Where ─────────────────────────────────────────────────────────────────

    @staticmethod
    def where(cond: Tensor, x, y) -> Tensor:
        x = Tensor._ensure_tensor(x)
        y = Tensor._ensure_tensor(y)
        return Tensor._from_term(F.lib.py_where(F.get_ctx(), cond._t, x._t, y._t))

    # ── Comparison ────────────────────────────────────────────────────────────

    def __lt__(self, other): return self._binary(F.UOP_CMP, other)
    def __gt__(self, other): return Tensor._ensure_tensor(other)._binary(F.UOP_CMP, self)

    # ── Loss functions ────────────────────────────────────────────────────────

    def cross_entropy(self, targets: Tensor) -> Tensor:
        log_probs = self.log_softmax(axis=-1)
        return -(targets * log_probs).sum(axis=-1).mean()

    def mse_loss(self, targets: Tensor) -> Tensor:
        diff = self - targets
        return (diff * diff).mean()

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self):
        return f"<Tensor shape={self.shape} dtype={self.dtype}>"

    def __len__(self) -> int:
        return self.shape[0]

    # ── Conv2d / Pool ─────────────────────────────────────────────────────────

    def conv2d(self, weight: Tensor, bias: Optional[Tensor] = None,
               stride: int = 1, padding: int = 0, groups: int = 1) -> Tensor:
        c_stride = F.make_u32_array([stride, stride])
        c_padding = F.make_u32_array([padding, padding])
        bias_t = bias._t if bias is not None else F.lib.py_era()
        return Tensor._from_term(F.lib.py_conv2d(
            F.get_ctx(), self._t, weight._t, bias_t, groups, c_stride, c_padding))

    def max_pool2d(self, kernel_size: int = 2, stride: Optional[int] = None) -> Tensor:
        if stride is None: stride = kernel_size
        c_kernel = F.make_u32_array([kernel_size, kernel_size])
        c_stride = F.make_u32_array([stride, stride])
        return Tensor._from_term(F.lib.py_maxpool2d(
            F.get_ctx(), self._t, c_kernel, c_stride))

    # ── Sequential ────────────────────────────────────────────────────────────

    def sequential(self, layers) -> Tensor:
        x = self
        for layer in layers:
            x = layer(x)
        return x
