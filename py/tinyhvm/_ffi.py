"""ctypes bindings to libthvm — the TinyHVM C runtime."""
import ctypes, ctypes.util, os, sys
from pathlib import Path

c_u32 = ctypes.c_uint32
c_u64 = ctypes.c_uint64
c_f32 = ctypes.c_float
c_u32_p = ctypes.POINTER(c_u32)
c_f32_p = ctypes.POINTER(c_f32)

def _find_lib() -> str:
    """Locate libthvm.dylib/.so relative to this package or in standard paths."""
    # 1. Next to this file (editable install / dev)
    here = Path(__file__).parent
    for candidate in [
        here / "libthvm.dylib",
        here / "libthvm.so",
        here.parent / "csrc" / "libthvm.dylib",
        here.parent / "csrc" / "libthvm.so",
        # repo root build output
        here.parent.parent.parent / "lib" / "libthvm.dylib",
        here.parent.parent.parent / "lib" / "libthvm.so",
    ]:
        if candidate.exists():
            return str(candidate)
    # 2. System path
    found = ctypes.util.find_library("thvm")
    if found:
        return found
    raise RuntimeError(
        "Cannot find libthvm. Build it with: ./build.sh --python"
    )

_lib = ctypes.CDLL(_find_lib())

# ── Typedefs ──────────────────────────────────────────────────────────────────
_ctx_p = ctypes.c_void_p  # TinyHVM*

# ── Function signatures ───────────────────────────────────────────────────────

# Lifecycle
_lib.py_set_metallib_path.argtypes = [ctypes.c_char_p]
_lib.py_set_metallib_path.restype = None

_lib.py_init.argtypes = [ctypes.c_char_p]
_lib.py_init.restype = _ctx_p

_lib.py_free.argtypes = [_ctx_p]
_lib.py_free.restype = None

_lib.py_reset.argtypes = [_ctx_p, c_u32]
_lib.py_reset.restype = None

# Tensor creation
_lib.py_tensor.argtypes = [_ctx_p, c_f32_p, c_u32_p, c_u32]
_lib.py_tensor.restype = c_u64

_lib.py_scalar.argtypes = [_ctx_p, c_f32]
_lib.py_scalar.restype = c_u64

# Ops
_lib.py_op.argtypes = [_ctx_p, c_u32, c_u64, c_u64]
_lib.py_op.restype = c_u64

# Reduce / realize
_lib.py_reduce.argtypes = [_ctx_p, c_u64]
_lib.py_reduce.restype = c_u64

_lib.py_realize.argtypes = [_ctx_p, c_u64]
_lib.py_realize.restype = None

# Read back
_lib.py_to_host.argtypes = [_ctx_p, c_u64]
_lib.py_to_host.restype = c_f32_p

# Shape introspection
_lib.py_ndim.argtypes = [_ctx_p, c_u64]
_lib.py_ndim.restype = c_u32

_lib.py_shape.argtypes = [_ctx_p, c_u64, c_u32_p, c_u32_p]
_lib.py_shape.restype = None

_lib.py_numel.argtypes = [_ctx_p, c_u64]
_lib.py_numel.restype = c_u32

# Movement
_lib.py_reshape.argtypes = [_ctx_p, c_u64, c_u32_p, c_u32]
_lib.py_reshape.restype = c_u64

_lib.py_expand.argtypes = [_ctx_p, c_u64, c_u32_p, c_u32]
_lib.py_expand.restype = c_u64

_lib.py_permute.argtypes = [_ctx_p, c_u64, c_u32_p, c_u32]
_lib.py_permute.restype = c_u64

_lib.py_pad.argtypes = [_ctx_p, c_u64, c_u32_p, c_u32]
_lib.py_pad.restype = c_u64

_lib.py_shrink.argtypes = [_ctx_p, c_u64, c_u32_p, c_u32]
_lib.py_shrink.restype = c_u64

# Autograd
_lib.py_set_requires_grad.argtypes = [_ctx_p, c_u64]
_lib.py_set_requires_grad.restype = None

_lib.py_grad.argtypes = [_ctx_p, c_u64, c_u64]
_lib.py_grad.restype = c_u64

# Where / assign
_lib.py_where.argtypes = [_ctx_p, c_u64, c_u64, c_u64]
_lib.py_where.restype = c_u64

_lib.py_assign.argtypes = [_ctx_p, c_u64, c_u64]
_lib.py_assign.restype = c_u64

# Conv / pool
_lib.py_conv2d.argtypes = [_ctx_p, c_u64, c_u64, c_u64, c_u32, c_u32_p, c_u32_p]
_lib.py_conv2d.restype = c_u64

_lib.py_maxpool2d.argtypes = [_ctx_p, c_u64, c_u32_p, c_u32_p]
_lib.py_maxpool2d.restype = c_u64

# ERA
_lib.py_era.argtypes = []
_lib.py_era.restype = c_u64

# Term introspection
_lib.py_term_tag.argtypes = [c_u64]
_lib.py_term_tag.restype = c_u32

_lib.py_term_ext.argtypes = [c_u64]
_lib.py_term_ext.restype = c_u32

# Profiling
_lib.py_profile_report.argtypes = [_ctx_p]
_lib.py_profile_report.restype = None

_lib.py_profile_reset.argtypes = [_ctx_p]
_lib.py_profile_reset.restype = None

# ── UOp constants (mirror tinyhvm.h) ──────────────────────────────────────────
UOP_LOAD = 0; UOP_STORE = 1; UOP_COPY = 2
UOP_NEG = 3; UOP_EXP = 4; UOP_LOG = 5; UOP_RELU = 6; UOP_CAST = 7; UOP_SQRT = 8
UOP_ADD = 9; UOP_MUL = 10; UOP_DIV = 11; UOP_MAX = 12; UOP_CMP = 13; UOP_SUB = 14
UOP_SUM = 15; UOP_RMAX = 16; UOP_MM = 17
UOP_RESHAPE = 18; UOP_PERMUTE = 19; UOP_EXPAND = 20; UOP_SHRINK = 21; UOP_PAD = 22

TAG_TEN = 10; TAG_TOP = 11; TAG_ERA = 6

# ── Singleton context ─────────────────────────────────────────────────────────

_ctx = None

def get_ctx():
    global _ctx
    if _ctx is None:
        _ctx = _lib.py_init(b"metal")
    return _ctx

def init(device: str = "metal"):
    global _ctx
    if _ctx is not None:
        _lib.py_free(_ctx)
    _ctx = _lib.py_init(device.encode())
    return _ctx

def reset(keep: int = 0):
    _lib.py_reset(get_ctx(), keep)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_u32_array(vals):
    return (c_u32 * len(vals))(*vals)

def make_f32_array(vals):
    return (c_f32 * len(vals))(*vals)

# Public alias
lib = _lib
