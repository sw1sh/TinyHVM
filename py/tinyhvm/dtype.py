class DType:
    def __init__(self, name: str, code: int, itemsize: int):
        self.name = name
        self.code = code
        self.itemsize = itemsize
    def __repr__(self): return f"dtypes.{self.name}"
    def __eq__(self, other): return isinstance(other, DType) and self.code == other.code
    def __hash__(self): return self.code

class _dtypes:
    float32 = DType("float32", 0, 4)
    float16 = DType("float16", 1, 2)
    int32 = DType("int32", 2, 4)
    uint32 = DType("uint32", 3, 4)
    # tinygrad aliases
    float = float32
    half = float16
    int = int32
    default_float = float32

dtypes = _dtypes()
