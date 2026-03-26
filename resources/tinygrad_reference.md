# Tinygrad: Comprehensive Reference

Deep reference for TinyHVM development — covering tinygrad's architecture, recent
evolution, kernel optimization pipeline, ShapeTracker design, commercial direction,
and strategic positioning. Current as of March 2026.

---

## 1. Project Overview

**tinygrad** is a deep learning framework aiming to capture the full scope of ML
training and inference in minimal code. Created by George Hotz (first commit
October 17, 2020). Maintained by tiny corp (6 employees, San Diego).

**Core thesis**: the software underlying all ML computation has a simple formulation
that can be expressed in ~20K lines rather than millions.

| Metric | Value |
|--------|-------|
| LoC (core, v0.12.0) | 19,075 |
| LoC (PyTorch comparison) | ~3,300,000 |
| LoC (JAX comparison) | ~400,000 |
| Python dependencies | **0** (since v0.10.0) |
| GitHub stars | ~30,800 |
| Contributors | 300+ |
| Total commits | ~16,000 |
| Team size | 6 |
| Revenue | ~$2M/year (tinybox sales) |
| Funding | $5.1M raised (May 2023) |

**Maxim**: "If XLA is CISC, tinygrad is RISC."

---

## 2. Philosophy & Vision

### 2.1 George Hotz's Stated Goals

From "Can tinygrad win?" (July 2025):
> "There is a simple formulation of the problem underlying all of the scheduling"
> across all computational scales. If successful, tinygrad will be "the fastest NN
> framework" while remaining "under 25k lines all in."

From "Five years of tinygrad" (December 2025):
> "Only a fool begins by taping out a chip; it's expensive and not the hard part.
> Once you have a fully sovereign software stack capable of training SOTA models,
> the chip is so easy."

> "AMD, Amazon, Tesla, and Groq have taped out fine chips, but only Google and
> NVIDIA chips have ever been seriously used for training. Because they have the
> software."

Mission: **"Commoditize the petaflop."**

### 2.2 Design Principles

- **The Elon Process for software**: "Make the requirements less dumb. The best
  part is no part."
- ~98% of software lines exist merely to maintain compatibility with other
  abstractions — tinygrad eliminates those layers
- No code golf — readability trumps line count
- Speedups must be benchmarked
- All features must have regression tests
- API matches PyTorch or NumPy where possible
- Dead code removal is welcome
- Small, clear PRs preferred
- Hiring: "People get hired by contributing to the repo." ($50K+ paid in bounties,
  ~80% conversion rate from bounty solvers to internship offers)

### 2.3 Anti-Cloud Philosophy

From "anticloud hopecore" (October 2025):
> "We need the cloud to go away. The cloud is a highway to serfdom."

Vision: on-device learning, personalized weights, fully reproducible training runs
in 5000 UOps. Foundation models will become commodities.

---

## 3. Release History & Milestones

| Version | Date | Lines | Key Changes |
|---------|------|-------|-------------|
| 0.10.0 | Nov 2024 | 9,937 | **Zero Python deps** (numpy removed via threefry RNG, pyobjc→ctypes); 3 new backends (QCOM, CLOUD, DSP); Apple AMX & Intel XMX tensor cores; VIZ=1 visualization; symbolic→UOp consolidation |
| 0.10.1 | Feb 2025 | 10,941 | **LazyBuffer deleted** — "just immutable UOp + Tensor"; AM driver for AMD; llvmlite removed |
| 0.10.2 | Feb 2025 | 11,263 | CLANG→CPU rename; KERNEL UOp introduced; WebGPU switched to Dawn |
| 0.10.3 | May 2025 | 12,990 | USB3-attached GPU (RDNA3/4); MI300X; RDNA 3.5/4; Torch frontend; CLOUD→REMOTE |
| 0.11.0 | Aug 2025 | 16,671 | ONNX merged; MI350 + Blackwell; NVIDIA userspace driver; multi-host InfiniBand; Muon optimizer |
| 0.12.0 | Jan 2026 | 19,075 | **Rangeify** across codebase; Mesa NIR/NAK backend; AMD SQTT & PMC visualization; multi-output kernels |

**Line count trajectory:**
```
Nov 2024:  9,937   ██████████
Feb 2025: 10,941   ███████████
May 2025: 12,990   █████████████
Jul 2025: 14,556   ██████████████▌
Aug 2025: 16,671   ████████████████▌
Dec 2025: 18,935   ██████████████████▌
Jan 2026: 19,075   ███████████████████
Target:   25,000   █████████████████████████ (GPT-5 scale training)
```

---

## 4. Architecture (Current)

### 4.1 The UOp: Single Universal IR

The defining architectural choice: **one IR node type at every compilation stage.**

```
Tensor methods → UOp graph → Schedule → Kernel UOps → Lowered UOps → Rendered source → Binary
```

A `UOp` has: `(op, dtype, src: tuple[UOp], arg, tag)`

**Structural interning**: UOps are deduplication-cached via weak references.
Same `(op, dtype, src, arg, tag)` = same Python object. O(1) equality, automatic CSE.

**84 distinct operations** organized as:
- **Movement** (9): RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP, MULTI, VIEW, VALID
- **Compute** — Unary (7): EXP2, LOG2, SIN, SQRT, RECIP, NEG, TRUNC
- **Compute** — Binary (17): ADD, MUL, IDIV, MAX, MOD, CMPLT, CMPNE, CMPEQ, XOR, SHL, SHR, OR, AND, THREEFRY, SUB, FDIV, POW
- **Compute** — Ternary (2): WHERE, MULACC
- **Memory** (6): DEFINE_GLOBAL, DEFINE_LOCAL, DEFINE_REG, LOAD, STORE, INDEX
- **Control** (6): RANGE, ENDRANGE, IF, ENDIF, BARRIER, REDUCE_AXIS
- **Buffer** (5): BUFFER, BUFFER_VIEW, COPY, ASSIGN, CONTIGUOUS
- **Codegen** (10): KERNEL, BLOCK, BLOCKSTART, BLOCKEND, WMMA, VECTORIZE, CAT, GEP, CAST, BITCAST
- **Special/Meta** (22): SINK, CONST, BIND, SPECIAL, FUSE, CUSTOM, etc.

Key **GroupOp** classifications:
- `Commutative`: ADD, MUL, MAX, CMPNE, CMPEQ, XOR, AND, OR (9)
- `Associative`: ADD, MUL, AND, OR, MAX (5)
- `UnsafePad`: RECIP, LOG2, EXP2, IDIV, POW (5 — cannot pad with zeros)

### 4.2 PatternMatcher System

The entire optimization and lowering pipeline is declarative rule-based graph rewriting:

```python
pm_lowerer = PatternMatcher([
  (UPat(Ops.LOAD, src=(UPat.var("buf").view(),)), lower_load),
  (UPat(Ops.REDUCE_AXIS, name="x"), lower_reduce_axis),
])

# Apply until fixpoint:
graph_rewrite(ast, pm_lowerer, ctx, name="lowerer", bottom_up=True)
```

**Graph Rewrite 2.0** (PR #8488): tree automata approach — each UOp traversed
exactly once per pass. Compiles `(UPat, handler)` rules into an efficient matcher.

### 4.3 Compilation Pipeline Phases

```
AST → view_push → optimize → lower → expand → devectorize → linearize → render
```

Each phase is a PatternMatcher:
- **view_left/view_right**: push views before/after ops for fusion
- **pm_lowerer**: convert to indexed loads/stores
- **expander**: vectorize operations
- **devectorize**: split to hardware vector widths (Metal: [4,2], CUDA: [16,8,4,2])
- **linearizer**: schedule UOps with priority (loads: -1000, barriers: -1500)
- **pm_render**: generate target-specific source code

### 4.4 RANGEIFY (v0.12.0+)

Major compiler evolution — creates ranges (loops) early in the graph:
- Movement ops (RESHAPE, EXPAND, SHRINK, PAD, PERMUTE, FLIP) are expressed as
  **loop manipulations** rather than index transformations
- Op fusion is determined by **matching ranges between parents** — same ranges = fused
- Goal: specify an entire $100M training run as a 10KB graph of UOps

Located in `schedule/rangeify.py` (21,885 lines).

### 4.5 Four-Piece Architecture

The codebase is structured as:
1. **Frontend** — `tensor.py` (4,460 lines): Tensor methods construct UOp DAGs
2. **Graph Compiler** — `schedule/`, `codegen/`, `uop/`: scheduling, optimization, lowering
3. **Runtimes** — `runtime/`: device-specific execution (17 backends)
4. **Drivers** — AM driver, NV driver: userspace hardware access

---

## 5. ShapeTracker & Lazy Views

The single most important optimization: **movement ops don't move data.**

### 5.1 Core Design

```python
@dataclass(frozen=True)
class ShapeTracker:
    views: tuple[View, ...]  # Stack of composed view operations

@dataclass(frozen=True)
class View:
    shape: tuple[sint, ...]        # Logical dimensions
    strides: tuple[sint, ...]      # Step size per dimension (0 = broadcast)
    offset: sint                   # Starting position in buffer
    mask: tuple[tuple[sint,sint],...]|None  # Valid range per dim (for padding)
    contiguous: bool               # Standard row-major layout?
```

### 5.2 Movement Ops (6 Total)

| Op | View Transform | Data Movement |
|---|---|---|
| `reshape(new_shape)` | Merge/split dims. May need new View if non-contiguous | None |
| `permute(axes)` | Reorder strides: `strides = [strides[a] for a in axes]` | None |
| `expand(new_shape)` | Set stride=0 where size=1→N (broadcast) | None |
| `shrink((lo,hi),...)` | Adjust offset, narrow each dim | None |
| `pad((lo,hi),...)` | Add mask for out-of-bounds regions | None |
| `flip(axes)` | Negate strides, adjust offset | None |

### 5.3 What This Enables

- `Tensor.reshape(...)` — free (metadata only)
- `Tensor.transpose()` — free (swap strides)
- `Tensor.broadcast_to(...)` — free (set stride=0)
- `Tensor[2:5, :]` — free (adjust offset + shape)
- `mm(grad, B.T)` — no copy of B, just permuted strides
- `bias + x` with shapes `[N] + [M,N]` — expand bias with stride=0 in dim 0

Only compute ops and explicit copies touch GPU buffers.

### 5.4 Kernel Indexing with Strides

The GPU kernel uses strides to index into each input. Broadcasting, transposition,
slicing, and views all collapse into one stride-based loop:

```c
for (u32 i = 0; i < out_numel; i++) {
    u32 idx_a = 0, idx_b = 0, rem = i;
    for (int d = ndim-1; d >= 0; d--) {
        u32 coord = rem % out_shape[d];
        rem /= out_shape[d];
        idx_a += coord * stride_a[d];
        idx_b += coord * stride_b[d];
    }
    dst[i] = a[idx_a + off_a] OP b[idx_b + off_b];
}
```

When stride=0, the same element is read for every index in that dimension (broadcast).
When strides are permuted, it's a transpose. When offset is nonzero, it's a slice.

### 5.5 View Merging & Simplification

`simplify()` merges adjacent views when possible. `real_strides()` extracts actual
memory access patterns for optimization decisions. The ILP solver finds equivalent
single-view representations for complex composed views.

### 5.6 Contiguity Check

A View is contiguous when:
- `offset = 0`
- `strides = row-major strides` (i.e., `strides[i] = product(shape[i+1:])`)
- no mask

Ops that need contiguous inputs (BLAS matmul) trigger a `CONTIGUOUS` realization.

---

## 6. Kernel Optimization Pipeline

### 6.1 Two-Level Strategy

**Level 1 — Heuristics** (`codegen/opt/heuristic.py`, ~0ms):

1. **Tensor core detection**: `MUL → CAST? → REDUCE(ADD)` pattern
2. **Matvec specialization**: `MV_BLOCKSIZE=4, MV_THREADS_PER_ROW=8, MV_ROWS_PER_THREAD=4`
3. **GROUPTOP**: output ≤ 2048 elements → group size 16
4. **Image upcast**: float4 images upcasted to 4
5. **Masked axis upcast**: small dims (≤7) with masks
6. **Reduce unroll**: dim ≤32 → UNROLL; ≤3 → full unroll
7. **Local grouping**: expand axes → LOCAL, try sizes `[32,16,8,4,3,2]`
8. **Final upcast**: default upcast last dim by 4

**Level 2 — Beam Search** (`codegen/opt/search.py`, seconds–minutes):

Only runs when `BEAM > 0`. ~200+ candidate transformations per iteration.

Action space:
```
UPCAST  (axis=0..7,  amt=[0,2,3,4,5,7])       48 actions
UNROLL  (axis=0..4,  amt=[0,4,7])              15 actions
LOCAL   (axis=0..5,  amt=[2,3,4,8,13,16,29])   42 actions
GROUPTOP(axis=0..2,  amt=[13,16,28,29,32,49,64,256])  24 actions
GROUP   (axis=0..2,  amt=[0,4,8,16])           12 actions
PADTO   (axis=0..6,  amt=[32])                  7 actions
TC      (variable)                              variable
SWAP    (axis pairs 0..4)                       10 actions
```

**Beam search loop:**
1. Start with unoptimized kernel
2. Generate all valid single-step transformations
3. Compile and time each (parallel, `PARALLEL=cpu_count`)
4. Keep top `amt` candidates
5. Exit when: no improvement, improvement < 0.01us, or no valid actions
6. Filter: drop candidates using 1000x more compute than minimum
7. Early stop: kernel > 3x current best

**Thresholds:**
```
BEAM_UPCAST_MAX  = 256     max total upcast
BEAM_LOCAL_MAX   = 1024    max local elements
BEAM_UOPS_MAX    = 3000    max generated UOps
BEAM_TIMEOUT_SEC = 10      compilation timeout per kernel
```

### 6.2 Scheduling & Fusion

The scheduler (`schedule/grouper.py`, `schedule/kernelize.py`) decides fusion:

**Fusion rules:**
- Only **contiguous** operations fuse
- At most **one REDUCE_AXIS** per kernel (relaxed in v0.12.0 with multi-output)
- Shape tracker sizes must match
- Unsafe pad ops force realization
- Fusion priority: COPY ops across devices get priority (heuristic=1000)

**Multi-output kernels** (v0.12.0+):
- ReduceOps fused with elementwise children
- ~20% fewer kernels for MNIST/GPT-2
- Faster LLaMA inference
- ASSIGN enables fusing optimizer updates with gradient computation

**RANGEIFY fusion** (current): fusion determined by matching ranges between
parent operations — same ranges = same kernel.

### 6.3 Tensor Core Support

| Architecture | Dims (N×M×K) | Threads | Input → Output |
|---|---|---|---|
| NVIDIA SM80 | 8×16×16 | 32 | fp16/bf16 → fp32 |
| NVIDIA SM75 | 8×16×8 | 32 | tf32 → fp32 |
| AMD RDNA3/4 | 16×16×16 | 32 | fp16/bf16 → fp32 |
| AMD CDNA | 16×16×16 | 64 | fp16/bf16 → fp32 |
| **Apple Metal** | **8×8×8** | **32** | **fp16 → fp32** |
| Apple AMX | up to 64×64 | 1 | various |
| Intel | 8×8×16 | 8 | fp16 → fp32 |

Detection: find `MUL + CAST? + REDUCE(ADD)` in compute graph, match stride-0 axes
to TC dimensions, pad if needed (opt_level >= 2).

### 6.4 Axis Types

| Type | Color | Meaning |
|---|---|---|
| GLOBAL | blue | Work items in global dispatch |
| LOCAL | cyan | Work items in threadgroup |
| LOOP | white | Sequential loops per work item |
| UPCAST | yellow | Unrolled as separate loads/ops |
| UNROLL | magenta | Fully unrolled |
| GROUP_REDUCE | green | Grouped reduction (shared memory) |
| REDUCE | red | Pure reduction dimension |

### 6.5 Caching Architecture

Three levels:
1. **Method cache** (in-memory): `{(device, ast_key, context_flags)} → CompiledRunner`
2. **Disk cache** (SQLite): SHA256-content-addressed compiler output
3. **Beam cache** (SQLite): `{ast_key, beam_amt, device} → applied_opts`

`CACHELEVEL=2` (default): memory + disk. Survives process restarts.

---

## 7. Device Backends (17)

| Backend | File | Lines | Notes |
|---|---|---|---|
| METAL | ops_metal.py | 15,248 | Apple GPU — first-class, with tensor core + AMX support |
| NV | ops_nv.py | 39,693 | NVIDIA userspace driver — bypasses CUDA entirely |
| AMD | ops_amd.py | 56,486 | AM driver — own userspace driver, no ROCm needed |
| CUDA | ops_cuda.py | 7,122 | Traditional NVIDIA CUDA |
| CPU | ops_cpu.py | 7,549 | CPU backend (was CLANG) |
| GPU | ops_gpu.py | 8,935 | OpenCL |
| HIP | ops_hip.py | 3,680 | AMD HIP |
| LLVM | ops_llvm.py | 4,594 | LLVM CPU |
| DSP | ops_dsp.py | 18,868 | Qualcomm Hexagon DSP |
| QCOM | ops_qcom.py | 22,120 | Qualcomm Adreno HCQ |
| WebGPU | ops_webgpu.py | 13,272 | Browser (Dawn backend) |
| REMOTE | ops_remote.py | 25,116 | Remote tinygrad execution |
| PYTHON | ops_python.py | 13,371 | Pure Python fallback |
| NPY | ops_npy.py | 565 | NumPy backend |
| DISK | ops_disk.py | 6,826 | Disk/memory I/O |
| NIR | (via Mesa) | — | NVK (open-source NVIDIA Vulkan) + LLVMpipe |
| NULL | ops_null.py | 1,578 | Null device |

**Renderers** (5 code generators):
- `cstyle.py` (31,663 lines) — Metal, CUDA, OpenCL, AMD, CPU, Qualcomm
- `llvmir.py` (17,359 lines) — LLVM IR
- `ptx.py` (16,184 lines) — NVIDIA PTX assembly
- `wgsl.py` (7,751 lines) — WebGPU Shading Language
- `__init__.py` (6,105 lines) — Base Renderer class

### 7.1 Metal Backend Details

**MetalRenderer** (in `renderer/cstyle.py`):
```
kernel_typedef      = "kernel void"
buffer_prefix       = "device "
smem_prefix         = "threadgroup __attribute__((aligned(16))) "
shared_max          = 32,768 bytes
barrier             = "threadgroup_barrier(mem_flags::mem_threadgroup);"
tensor_cores        = 8×8×8 simdgroup_matrix (arm64)
bfloat16            = "bfloat" type
transcendentals     = precise::sin() for accuracy
devectorize splits  = [4, 2]
```

Workgroup builtins:
- `uint3 gid [[threadgroup_position_in_grid]]`
- `uint3 lid [[thread_position_in_threadgroup]]`

**MetalCompiler** (in `runtime/ops_metal.py`):
- Uses private `MTLCompiler.framework` via ctypes
- `REQUEST_TYPE_COMPILE = 13` (undocumented API)
- Compiles Metal source → MTLB binary library
- `METAL_FAST_MATH` env flag

**MetalAllocator** (LRU-based):
- `MTLResourceStorageModeShared` (unified memory)
- GPU timing: nanosecond precision via command buffer timestamps
- MetalGraph support for graph-mode execution
- 1024 command buffer capacity

### 7.2 Sovereign AMD Stack

Tinygrad's own full-stack for AMD GPUs — no ROCm dependency:

- **AM driver** (~userspace): binds compute queues directly to MEC (bypassing MES),
  3-level page directory supporting 512GB virtual addresses, unified virtual address
  space across all AM devices
- **Runtime**: HCQ (Hardware Command Queue) API — commands issued directly to
  hardware queues, bypassing HIP/CUDA overhead
- **Compiler**: currently uses LLVM (with active work to eliminate it)
- **Missing piece**: RDNA3 assembler ($1,000 bounty for one within 10% of LLVM)
- Total stack: ~12,000 lines excluding LLVM

### 7.3 NVIDIA Open-Source Path

Two approaches for NVIDIA without CUDA:

1. **NV backend**: tinygrad's own userspace driver (no CUDA kernel module)
2. **Mesa NIR backend** (PR #12089): targets NVK (open-source NVIDIA Vulkan) with
   NAK (Rust-based compiler). Bypasses NVPTX → goes straight to NVIDIA SASS assembly.
   Paired with Nouveau/Nova kernel driver = fully free software path.

### 7.4 USB-Attached GPUs

Novel capability unique to tinygrad:

- **AMD over USB3**: RDNA3/RDNA4 via ASM2464PD controller — works on Apple Silicon
  Macs, Linux, Windows
- **NVIDIA over USB4**: RTX 30/40/50 via ADT-UT3G dock on MacBook — `brew install
  tinymesa` for NVK compiler. AI workloads only (no display).

---

## 8. NumPy Removal

Fully removed in **v0.10.0** (November 2024). Tracked in GitHub Issue #2649.

**What was replaced:**
- Random number generation: `numpy.random` → pure-Python **threefry** (threefry2x32)
- Metal bindings: `pyobjc` → `ctypes` + `libobjc`
- Tensor creation from lists/nested lists: implemented natively

**Result**: zero Python dependencies. Only Python itself and a C compiler required.

The `ops_npy.py` backend (565 lines) still exists as an optional NumPy device,
but core tinygrad never imports numpy. All numpy references in core are behind
`try/except` or `TYPE_CHECKING` guards.

---

## 9. Key Architectural Rewrites (2024-2026)

| Rewrite | When | Impact |
|---|---|---|
| **Symbolic → UOp** | v0.10.0, Nov 2024 | Symbolic system consolidated into UOp rewrite rules |
| **LazyBuffer deletion** | v0.10.1, Feb 2025 | "No LazyBuffer, just immutable UOp + Tensor" (Issue #7697) |
| **Linearizer → Lowerer** | mid-2024 | All opts at LazyOp level; UOp opts done as graph rewrite |
| **Graph Rewrite 2.0** | PR #8488 | Tree automata approach — each UOp traversed once per pass |
| **Big Graph** | Issues #7044, #8273 | Views, metaops converted to UOps; late bufferization |
| **RANGEIFY** | v0.12.0, Jan 2026 | Loops created early; movement = range manipulation; fusion by range matching |
| **Multi-output kernels** | v0.12.0 | Multiple outputs per kernel; ~20% fewer kernels for training |
| **Scheduler rewrite** | PR #8903, Feb 2025 | Scheduling via graph_rewrite PatternMatcher |

### Evolution trajectory:
```
2020-2023: Tensor → LazyBuffer → Schedule → Linearizer → Renderer → Binary
2024:      Tensor → UOp → Schedule → Lowerer → Renderer → Binary
2025:      Tensor → UOp graph (unified) → KERNEL UOps → Rendered source → Binary
2026:      UOp graph → Rangeified → Kernelized → Rendered → Binary
```

The trend: fewer stages, more unified representation, more graph_rewrite, less
imperative transformation code.

---

## 10. Memory & Execution

### 10.1 Memory Planning

TLSF (Two-Level Segregate Fit) allocator for buffer suballocation:
- Minimum block: 4 KB (`0x1000`)
- Allocates 2× needed for ~15% fragmentation headroom
- Tracks first/last appearance of each buffer in schedule
- Reuses buffers across schedule items when lifetimes don't overlap
- Three strategies: suballocation (devices with `_offset`), buffer reuse, sub-buffer

### 10.2 JIT & Graph Execution

`TinyJit` class captures execution patterns:
- **cnt=0**: normal execution (warmup)
- **cnt=1**: JIT capture — records all operations
- **cnt≥2**: JIT replay — runs pre-compiled graph

Graph batching:
- Batch kernels by device into graph objects (MetalGraph, etc.)
- Exponential backoff on batch size (doubles on success)
- `JIT=2` (default on macOS) enables batch mode

### 10.3 Process Replay

Key testing mechanism: refactors must pass process replay (`ASSERT_PROCESS_REPLAY=1`).
Ensures code changes produce identical compilation results. Critical for verifying
that rewriting the compiler doesn't change any generated kernel.

---

## 11. Performance & Competitive Positioning

### 11.1 Claims & Benchmarks

From Hotz (mid-2025):
> "For most non-CPU non-NVIDIA platforms with BEAM=2, tinygrad is the fastest."
> "For small models, tinygrad significantly outperforms other frameworks even on NVIDIA."
> "For LLM inference, tinygrad is within 10% of theoretical maximum performance."

**MLPerf submissions:**
- Training v5.0: BERT on tinybox
- Training v6.0: LLaMA on AMD MI300X/MI350X
- Active AMD contract for MI350X benchmarking with Llama 405B

**Goal**: end-to-end performance matching PyTorch 4090 on the 7900XTX.

### 11.2 vs. Other Frameworks

| Framework | LoC | Hardware | Approach |
|---|---|---|---|
| tinygrad | ~19K | AMD, NVIDIA, Metal, QCOM, WebGPU | RISC UOps, beam search, own drivers |
| PyTorch | ~3.3M | NVIDIA (primary) | Eager + torch.compile, CUDA kernels |
| JAX | ~400K | TPU (primary), GPU | XLA compilation, functional API |
| MLX | ~50K | Apple Silicon only | Unified memory, lazy evaluation |
| MLIR | ~950K | Any (infrastructure) | Multi-level IR, dialect system |

### 11.3 vs. MLX on Apple Silicon

- MLX: optimized specifically for Apple unified memory, ~230 tok/s
- tinygrad: broader cross-platform, Metal as one of 17 backends
- MLX has deeper Apple-specific optimization; tinygrad has portability

---

## 12. Commercial Direction

### 12.1 Tiny Corp

- Founded November 5, 2022 by George Hotz
- Raised $5.1M (May 2023)
- 6 employees, San Diego
- "Deconstructed company": public Discord, public GitHub, contracts negotiated on Twitter

### 12.2 Products

**TinyBox Red v2** (shipping, $12,000):
- 4× AMD RX 9070 XT (RDNA4)
- 64 GB GPU RAM, 778 TFLOPS FP16
- 32-core AMD EPYC, 128 GB system RAM, 2 TB NVMe
- Ubuntu 24.04, ships weekly from San Diego

**TinyBox Green v2** (made to order, $65,000):
- 4× NVIDIA RTX PRO 6000 (Blackwell)
- 384 GB GPU RAM, 3,086 TFLOPS FP16
- 32-core AMD GENOA, 192 GB system RAM, 4 TB RAID
- 2× 10GbE

**Exabox** (projected 2027, ~$10M):
- ~1 EXAFLOP, 720× RDNA5 GPUs, 25,920 GB GPU RAM
- Ships in a shipping container
- "With tinygrad, the exabox will function as a single very large GPU that you
  (or your agent) can drive from a Python notebook"
- Exploring $20M raise at $200M valuation for $11.5M building + 5MW power

All products: no cloud, no accounts, no usage tracking.

### 12.3 Future Vision

From "tiny corp's product" (February 2026):
> "Your tinybox will learn. It will update the weights based on its interactions
> with you. Like living things."

Local box that continuously updates weights through user interactions. "Many years away."

---

## 13. AMD Partnership

| Date | Event |
|---|---|
| 2022-2023 | Rocky early collaboration, firmware/driver instability |
| Late 2024 | Tiny corp paused AMD tinybox shipments, threatened Intel/NVIDIA pivot |
| Mar 2025 | AMD ships MI300X systems; VP publicly commits to "commoditize the petaflop" |
| 2025 | AMD contract for MI350X MLPerf (Llama 405B training) |
| 2025 | MI355X: 2.8× faster time-to-train vs MI300X in MLPerf 5.1 |

Key Hotz quote:
> "CUDA isn't really the moat people think it is, it is just an early ecosystem.
> With good software the MI300X should outperform the H100."

Invested $250,000 in AMD stock with 5-year hold.

---

## 14. Formal Specification: tinyspec

Created March 2026 at `github.com/tinygrad/tinyspec`. Written in LaTeX by Hotz.
43 commits, sole contributor. Contains `tinyspec.tex` and `tinyspec.pdf`.

Formalizes the semantics of UOps, scheduling, and lowering. Reflects the project's
shift from "hacker framework" to "formally specified compiler."

---

## 15. Environment Variables Quick Reference

```
# Core
DEBUG=0..7           2=timing, 5=AST, 6=UOps, 7=disasm
BEAM=0..N            Beam search width (0=heuristics only)
NOOPT=0|1            Skip kernel optimization
USE_TC=0|1|2         Tensor cores (0=off, 1=on, 2=shapes only)
JIT=0|1|2            0=disabled, 1=graph, 2=batch (default 2 on macOS)
PROFILE=0|1          GPU trace events

# Beam search
BEAM_UPCAST_MAX=256  Max total upcast
BEAM_LOCAL_MAX=1024  Max local elements
BEAM_UOPS_MAX=3000   Max generated UOps
BEAM_TIMEOUT_SEC=10  Compilation timeout per kernel
BEAM_PADTO=1         Padding optimization
PARALLEL=cpu_count   Worker processes for search

# Caching
CACHELEVEL=0|1|2     0=nocache, 1=disk, 2=memory+disk (default)
IGNORE_BEAM_CACHE=0|1
DISABLE_COMPILER_CACHE=0|1

# Metal-specific
METAL_FAST_MATH=0|1  Enable -ffast-math

# Matvec heuristics
MV_BLOCKSIZE=4
MV_THREADS_PER_ROW=8
MV_ROWS_PER_THREAD=4
```

---

## 16. Key Links

### Official
- Repository: github.com/tinygrad/tinygrad
- Documentation: docs.tinygrad.org
- tinyspec: github.com/tinygrad/tinyspec
- Tinybox docs: docs.tinygrad.org/tinybox/
- HCQ docs: docs.tinygrad.org/developer/hcq/
- AM driver: docs.tinygrad.org/developer/am/

### George Hotz Blog Posts
- "Can tinygrad win?" (Jul 2025) — geohot.github.io
- "Five years of tinygrad" (Dec 2025) — geohot.github.io
- "AMD YOLO" (Mar 2025) — geohot.github.io
- "anticloud hopecore" (Oct 2025) — geohot.github.io
- "tiny corp's product" (Feb 2026) — geohot.github.io
- Blog index: geohot.github.io/blog/

### Key GitHub Issues & PRs
- Delete LazyBuffer: #7697
- Big Graph: #7044, #8273
- Graph Rewrite 2.0: PR #8488
- Scheduler rewrite: PR #8903
- Mesa NIR backend: PR #12089
- Remove numpy: #2649
- Delete early bufferization: #8253

### Community
- Discord: active, channels include #pytorch-backend, #tinybox
- Community notes: mesozoic-egg.github.io/tinygrad-notes/
- DeepWiki overview: deepwiki.com/tinygrad/tinygrad
- Exo (distributed inference): github.com/exo-explore/exo

---

## 17. Relevance to TinyHVM

### What to adopt:
1. **ShapeTracker / lazy views** — TinyHVM's View struct is inspired by this (see
   TinyHVM's `View` in `src/tensor/`). Stride-based indexing eliminates reshape copies.
2. **Beam search** — autotune kernel configs over tile sizes, unroll factors, local sizes
3. **Three-level caching** — method cache + disk cache + beam cache
4. **PatternMatcher rewrites** — declarative optimization rules (analogous to IC rewrite rules)
5. **Tensor cores** — simdgroup_matrix 8×8×8 for GEMM/conv on Apple Silicon
6. **Process replay** — verify compiler changes don't alter generated kernels
7. **Multi-output kernels** — reduce kernel count by fusing reduce + elementwise

### What to avoid:
1. **Python overhead** — tinygrad's Python dispatch adds latency; TinyHVM's C core is faster
2. **No memoization** — tinygrad doesn't have reduce_memo; TinyHVM's IC model benefits from it
3. **Growing complexity** — tinygrad went from ~10K to ~19K lines in 14 months;
   stay disciplined about what TinyHVM absorbs
4. **Backend breadth** — 17 backends is impressive but TinyHVM should focus on Metal first

### Structural alignment:
Both tinygrad and TinyHVM are graph rewriting systems at their core. Tinygrad uses
PatternMatcher/graph_rewrite; TinyHVM uses interaction combinators with enter/apply
trampoline reduction. The insight from `report.md` remains valid: IC makes the
rewriting explicit and provably optimal (confluence, strong normalization).
