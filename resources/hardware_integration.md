# Hardware Integration: How UOps Enable Multi-Backend

How tinygrad supports 17 backends with minimal code per backend, how they profile and optimize kernels, and what TinyHVM should steal.

## How Tinygrad Does It

### The Compilation Pipeline

```
Tensor ops → UOp graph → Schedule → Kernel → Renderer → Source code → Compiler → Binary → Runtime
```

Each backend only needs to implement the last three stages. Everything before the Renderer is hardware-agnostic.

### The `Compiled` Device

Every backend is a `Compiled` object with four pluggable parts:

| Part | What it does | Example (Metal) |
|---|---|---|
| **Allocator** | Buffer management (alloc, free, copyin/out) | `MTLBuffer` via objc |
| **Renderer** | UOps → source code string | UOps → Metal Shading Language |
| **Compiler** | Source code → binary | MSL → MTLB via `MTLCodeGenService` |
| **Runtime** | Load binary, dispatch kernel | `MetalProgram.__call__` (command buffer) |

That's it. To add a new backend you implement these four things. The Renderer is the interesting one — it maps each UOp to a source code snippet (`code_for_op` dict). Metal's renderer inherits from `CStyleRenderer` and mostly gets things for free.

### What the Renderer Tracks

```python
class Renderer:
    supports_float4: bool       # can we use vector types?
    has_local: bool            # does this device have local/shared memory?
    has_shared: bool           # threadgroup memory?
    global_max: (int, int, int)  # max grid dimensions
    local_max: (int, int, int)   # max workgroup dimensions
    shared_max: int              # max shared memory bytes
    tensor_cores: [TensorCore]   # HW matrix multiply units (e.g., Apple AMX)
    code_for_op: {Ops → fn}    # how to render each op as source code
```

The renderer doesn't know about buffers or allocation. It only transforms the UOp graph into a source code string. This clean separation is why adding a backend is ~200 lines.

### Metal Backend Specifics

`ops_metal.py` is 235 lines total. Highlights:

- Uses `ctypes` + `libobjc` directly — no PyObjC dependency
- `MetalCompiler` calls into Apple's private `MTLCodeGenService` for fast compilation (~8ms cached)
- `MetalProgram.__call__` creates a command buffer, encodes compute, and dispatches
- GPU timing: `GPUStartTime` / `GPUEndTime` on the command buffer (nanosecond precision)
- Buffer management: `StorageModeShared` means CPU and GPU share the same memory (on Apple Silicon)

---

## Profiling

Tinygrad has two profiling systems:

### 1. Estimates (Static Analysis)

Before running anything, `Estimates.from_uops()` walks the UOp list and counts:
- **FLOPS**: each ALU op = 1, each MULACC = 2, WMMA = 2×prod(shape)
- **Load/Store bytes**: each LOAD/STORE × dtype size × loop multiplier
- **Memory**: total buffer bytes (counted once per buffer)

This gives a roofline-style estimate without running the kernel. Used to predict which optimization will help most.

### 2. Runtime Profiling (GPU Timestamps)

Each command buffer records GPU start/end timestamps. When `PROFILE=1`:
- Every kernel execution logs `(device, name, start_time, end_time)`
- Buffer alloc/free events are tracked
- At exit, everything is pickled and optionally launched in a viz tool

Metal gives this for free via `GPUStartTime`/`GPUEndTime`. CUDA uses events. CPU uses `clock_gettime`.

---

## Kernel Optimization

### OptOps — The Optimization Primitives

Tinygrad has 7 optimization operations that transform a kernel's loop structure:

| OptOp | What it does |
|---|---|
| **UPCAST** | Unroll an axis into vector operations (e.g., process 4 elements per thread) |
| **LOCAL** | Split an axis into local/global dimensions (use workgroup threads) |
| **GROUP** | Reduce using shared memory within a workgroup |
| **TC** | Apply tensor core instructions (WMMA/AMX matrix tiles) |
| **PADTO** | Pad an axis to a multiple of 32 (alignment for SIMD) |
| **SWAP** | Swap the order of two axes |
| **UNROLL** | Fully unroll a reduce axis |

Each OptOp is a `(op, axis, amount)` tuple. A kernel's optimization is a sequence of these.

### BEAM Search — Autotuning

Instead of hand-tuned heuristics, tinygrad uses **beam search** over the space of OptOp sequences:

1. Start with the unoptimized kernel
2. Generate all legal next OptOps (UPCAST on axis 0, LOCAL on axis 1, etc.)
3. For each candidate: apply the opt, compile, **actually run on GPU**, measure time
4. Keep the top-N fastest candidates
5. Repeat until no improvement > `BEAM_MIN_PROGRESS` (default 0.01μs)

Parameters:
- `BEAM=N` — beam width (number of candidates to keep)
- `BEAM_ESTIMATE=1` — use static Estimates instead of real timing (faster but less accurate)
- `BEAM_TIMEOUT_SEC=10` — max time for compilation of a single candidate
- `BEAM_UOPS_MAX=3000` — skip kernels with too many UOps

This is expensive but gives near-optimal kernel configs for any hardware. Results are disk-cached.

---

## What TinyHVM Should Take

### Phase 2 (Now)

We don't need any of this complexity yet. Our CPU backend dispatches pre-built kernels (BLAS matmul, scalar loops). This is correct and fast enough for POC.

### Phase 3 (Metal Backend)

For Metal, we have two paths:

**Path A: Pre-built Kernels** (simpler, what we'll do first)
- Write 5-6 Metal compute shaders by hand: `add_f32`, `mul_f32`, `relu_f32`, `neg_f32`, `cmp_f32`
- Use MPS (`MPSMatrixMultiplication`) for matmul
- Compile shaders at init time, dispatch via command buffer
- Profile with GPU timestamps
- ~200 lines of Objective-C

**Path B: Code Generation** (tinygrad-style, later)
- Build a simple `MetalRenderer` that maps UOps → MSL source
- Compile at runtime using `MTLLibrary`
- Fuse adjacent operations into single kernels (e.g., `mm → add → relu` as one dispatch)
- BEAM search for optimal workgroup sizes

Path A is right for Phase 2-3. Path B is where TinyHVM becomes actually fast.

### The TinyHVM Backend Interface (Current)

Our `GpuBackend` vtable already supports the Path A approach:

```c
typedef struct {
    int   (*init)(void);
    void  (*shutdown)(void);
    u32   (*buf_alloc)(u64 bytes);
    void  (*buf_free)(u32 id);
    void  (*buf_write)(u32 id, const void *data, u64 bytes);
    void  (*buf_read)(u32 id, void *out, u64 bytes);
    void  (*op_unary)(u32 uop, u32 dst, const View *dv, u32 src, const View *sv);
    void  (*op_binary)(u32 uop, u32 dst, const View *dv, u32 a, const View *av, u32 b, const View *bv);
    void  (*op_mm)(u32 dst, u32 a, const View *av, u32 b, const View *bv, u32 M, u32 K, u32 N);
} GpuBackend;
```

For Path B (codegen), we'd extend this with:

```c
// Future: code-generation backend
typedef struct {
    // ... existing pre-built dispatch ...
    // Code generation
    void  (*compile_kernel)(const char *src, u32 *kernel_id);
    void  (*dispatch_kernel)(u32 kernel_id, u32 *bufs, u32 n_bufs,
                             u32 grid[3], u32 block[3]);
} GpuBackend;
```

### Profiling for TinyHVM

Phase 2: just `clock_gettime` around dispatches, accumulated in `ctx->itrs`.

Phase 3 (Metal):
```c
typedef struct {
    f64 gpu_start;   // from command buffer
    f64 gpu_end;
    u32 flops;       // estimated from op + shapes
    u32 bytes;       // buffer sizes accessed
    const char *name; // "mm_2x3x2", "relu_2x2", etc.
} ProfileEntry;
```

This gives us roofline analysis: `GFLOPS / (bytes/sec)` shows whether we're compute or memory bound.

---

## Summary: Why UOps Make Multi-Backend Easy

The key insight from tinygrad: **UOps are the universal IR.** Every frontend operation eventually becomes the same small set of UOps. Each backend only translates those UOps to its own language. The optimization space (UPCAST, LOCAL, etc.) is hardware-independent — BEAM search explores it per-device.

TinyHVM takes this further: our UOps are interaction net nodes. The reducer is the "frontend to UOps" compiler. The GPU backend is the "UOps to hardware" translator. Shape tracking (Views, strides) sits between them. Clean layers, minimal interface.
