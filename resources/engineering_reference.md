# Engineering Patterns: Lessons from ggml, tinygrad, and C-ML

Reference doc for TinyHVM development. Studied from source:
- ggml: `/Users/swish/src/ggml`
- C-ML: `https://github.com/jaywyawhare/C-ML` (cloned 2026-03-25)

---

## 1. Backend Abstraction (ggml)

ggml has a **4-layer vtable** architecture (from `ggml-backend-impl.h`):

```
registry  →  device  →  backend (stream)  →  buffer_type  →  buffer
```

Each layer is a struct with an `iface` (vtable of function pointers) + `context` (void pointer for impl data):

```c
struct ggml_backend {
    ggml_guid_t guid;
    struct ggml_backend_i iface;   // vtable
    ggml_backend_dev_t device;
    void *context;                 // Metal context, CUDA context, etc.
};
```

### What TinyHVM should adopt

**✅ The `iface` + `context` pattern.** Our current `GpuBackend` is already a flat vtable, which is fine for now. But as we add features (async, events, graph optimization), ggml's layered approach scales better.

**✅ `graph_compute()` over per-op dispatch.** ggml's backend interface's star method is:
```c
enum ggml_status (*graph_compute)(ggml_backend_t backend, struct ggml_cgraph *cgraph);
```
The backend receives the *entire computational graph* and decides how to execute it. This enables:
- Batch command buffer submission (one `[cmd commit]` for the whole graph)
- Kernel fusion decisions at the backend level
- Pipeline parallelism

Our current approach dispatches individual ops (`op_unary`, `op_binary`, `op_mm`), each doing its own `commit` + `waitUntilCompleted`. This is correct but slow — every op is a full GPU roundtrip.

**❌ Not yet needed: buffer_type/buffer split, device registry, multi-buffer.** These matter for multi-GPU, RPC backends, CPU+GPU splits. Overkill for now.

---

## 2. Tensor Struct (ggml)

```c
struct ggml_tensor {
    enum ggml_type type;
    struct ggml_backend_buffer *buffer;
    int64_t ne[GGML_MAX_DIMS];  // shape (always 4D, unused = 1)
    size_t  nb[GGML_MAX_DIMS];  // strides in bytes
    enum ggml_op op;
    int32_t op_params[...];     // inline op parameters
    int32_t flags;              // INPUT, OUTPUT, PARAM, LOSS
    struct ggml_tensor *src[GGML_MAX_SRC];  // compute graph parents
    struct ggml_tensor *view_src;   // source tensor for views
    size_t view_offs;               // offset into view_src
    void *data;
    char name[GGML_MAX_NAME];
    void *extra;                // backend-specific data
};
```

### Key pattern: tensors ARE the graph

In ggml, tensors double as graph nodes. `src[]` tracks the inputs, `op` tracks how to compute the output. No separate graph structure needed — `ggml_build_forward()` just traverses `src[]` pointers to build the `cgraph`.

**Compare with TinyHVM:** Our tensor metadata is separate from our IC graph. Tensors live in `TensorMeta[]`, the IC graph lives on the heap. This is actually a deeper design — the IC graph handles scheduling and reduction — but for the tensor-specific paths, ggml's approach is simpler.

### Key pattern: fixed 4D with strides in bytes

ggml always uses 4 dimensions. Shape `[M, N]` becomes `ne = {N, M, 1, 1}` (note: reversed order). Strides `nb[]` are in bytes, not elements. This simplifies every kernel: no variable-rank loops.

**Gap in TinyHVM:** Our `Shape` struct has variable rank (up to `MAX_DIM=8`). This means every kernel needs a rank-dependent loop. Consider: fixed 4D could simplify Metal shaders significantly.

### Key pattern: `view_src` + `view_offs`

Views in ggml are explicit: a view tensor has `view_src` pointing to the original tensor and `view_offs` for the offset. The backend uses this to find the correct `MTLBuffer` and offset.

**TinyHVM equivalent:** Our `View` struct already has `offset` and `strides`, which is similar. But we don't track the source tensor explicitly — we just modify the view on the same buffer ID. This works but loses the lineage.

---

## 3. Metal Backend Patterns (ggml)

From `/Users/swish/src/ggml/src/ggml-metal/`:

### Kernel argument structs

ggml defines **separate C structs for each kernel's arguments** (e.g., `ggml_metal_kargs_unary`, `ggml_metal_kargs_bin`, `ggml_metal_kargs_mul_mm`). These are shared between the ObjC host code and Metal shaders.

Our `ViewParams` is a single struct for all ops. ggml's approach is more verbose but allows per-op optimization (e.g., matmul doesn't need all the stride fields that a copy op does).

### Metal function constants

ggml uses Metal **function constants** (`FC_UNARY`, `FC_BIN`, etc.) to specialize shader functions at compile time. One generic kernel function handles multiple ops by switching on a function constant, avoiding redundant shader code.

**TinyHVM current approach:** Separate kernel functions per op (`unary_neg`, `unary_relu`, etc.). This is cleaner for small op counts but doesn't scale to 100+ ops.

### Command buffer batching

ggml's Metal `graph_compute()` encodes *all ops* in the graph into command buffers before committing. Our per-op `commit` + `waitUntilCompleted` is the main performance bottleneck — we should batch.

---

## 4. tinygrad Architecture

tinygrad's pipeline (from docs and source study):

```
Tensor API → LazyBuffer → Schedule → Linearize → Render → Compile → Execute
                              ↓
                          UOp graph
```

### Key principles

1. **RISC-like ops**: ~26 fundamental operations. All higher-level ops (conv, attention) decompose into these.
2. **Lazy evaluation**: Operations build a graph, `realize()` materializes it.
3. **ShapeTracker**: Movement ops (reshape, permute, expand, pad, shrink) are zero-cost metadata changes — no data movement.
4. **Kernel fusion**: The scheduler determines fusion boundaries (which ops can run in one kernel).
5. **<8000 lines**: Hard complexity budget forces clean design.

### Relevance to TinyHVM

We already do (1), (2), and (3) — our `View` struct is basically ShapeTracker. Our IC reduction is the equivalent of `realize()`. What we lack: (4) kernel fusion and (5) the hard LoC discipline.

---

## 5. C-ML Architecture

C-ML (`github.com/jaywyawhare/C-ML`) is a pure-C ML framework: ~117K lines across 216 source files and 180+ headers. Zero external dependencies. Covers tensors, autograd, IR compilation, multi-backend execution, 28 NN layers, 9 optimizers, and pre-built model zoo (ResNet, ViT, BERT, GPT-2, YOLO).

### Pipeline: Lazy IR with tensor facades

```
Tensor API → IRNode graph (lazy) → Optimize → Fuse → Schedule → Codegen → Execute
```

Tensors are thin facades over IR nodes. Operations build a graph — no computation happens until `tensor_data_ptr()` or `tensor_ensure_executed()` forces materialization:

```c
struct Tensor {
    int* shape;
    int ndim;
    DType dtype;                // 16+ types: f32, f16, bf16, fp8, int8, ...
    DeviceType device;          // CPU/CUDA/Metal/ROCm/OpenCL/Vulkan/WebGPU
    struct IRNode* ir_node;     // Lazy: points to computation graph node
    bool is_executed;           // False until materialized
    void* data;                 // NULL until executed
    bool requires_grad;
    struct Tensor* grad;        // Also lazy
    int ref_count;
    struct Tensor* base;        // View source
    size_t* strides;
    size_t storage_offset;
};
```

This is the same lazy pattern as tinygrad, but in C with explicit IR nodes rather than Python lazy buffers.

### What TinyHVM should study

**✅ Module/Parameter abstraction.** C-ML wraps every trainable weight in a `Parameter` struct and groups them under `Module`:

```c
struct Module {
    char* name;
    ForwardFn forward;       // Function pointer
    FreeFn free;
    Parameter** parameters;  // Collected for optimizer
    int num_parameters;
    bool training;           // Training vs eval mode
};

struct Parameter {
    Tensor* tensor;
    bool requires_grad;
    char* name;
};
```

This enables `module_collect_parameters()` → hand to optimizer → `optim_step()` updates all at once. Our current approach passes tensors directly. As we add more layers (BN, conv, etc.), a thin module wrapper would help organize parameter management and training/eval mode switching.

**✅ Loss functions as UOp compositions.** C-ML builds all 13+ loss functions from primitives:

```c
Tensor* tensor_mse_loss(Tensor* input, Tensor* target) {
    Tensor* diff = uop_sub(input, target);
    Tensor* squared = uop_mul(diff, diff);
    return uop_mean(squared, &reduce_params);
}
```

We already do this — our cross-entropy loss composes from log, exp, reduce_sum. Confirms we're on the right track. Key losses to have: MSE, cross-entropy, NLL, BCE.

**✅ Timeline memory planner.** C-ML tracks tensor allocation/deallocation timestamps across the computation graph, then computes minimum peak memory and reuses freed blocks for later tensors. This is the smart version of our "two pools" idea from the ggml study:

```
Instead of:  allocate/free per tensor (fragmentation, overhead)
Do:          scan graph → compute lifetimes → assign offsets into pre-allocated pool
```

**✅ Plugin fusion patterns.** C-ML has a registerable fusion system:

```c
struct FusionPattern {
    const char* name;
    FusionTarget target;     // Which backend
    int priority;
    FusionMatchFn match;     // Does this pattern apply?
    FusionEmitFn emit;       // Emit fused code
};
```

Users register patterns like MatMul+Bias+ReLU or Conv+BN+ReLU. The optimizer walks the IR graph matching patterns by priority. This is cleaner than hand-fusing in the backend — and something we should consider when we add kernel fusion.

**✅ LR schedulers.** C-ML has 7 schedulers (StepLR, CosineAnnealing, OneCycle, ReduceOnPlateau, etc.). We'll need at least `StepLR` and `CosineAnnealing` for real training runs. These are simple to implement — just a function that takes (base_lr, step, config) → current_lr.

**✅ Reference counting on tensors.** C-ML tracks `ref_count` on each tensor for safe sharing across views and gradient chains. Our current approach frees buffers in `thvm_reduce()` which works for now, but as we support more complex graphs (e.g., skip connections sharing tensors), ref counting prevents use-after-free.

### Pattern comparison: three approaches to the same problem

| Concern | ggml | tinygrad | C-ML | TinyHVM |
|---------|------|----------|------|---------|
| Graph repr | tensor.src[] | LazyBuffer | IRNode | IC heap |
| Lazy eval | No (eager) | Yes (realize) | Yes (ensure_executed) | Yes (thvm_reduce) |
| Fusion | Backend-level | Scheduler | Plugin registry | Not yet |
| Memory | Static alloc | Lazy alloc | TLSF + timeline | Per-op alloc |
| Shaders | One giant file | Codegen per kernel | Codegen per target | Separate functions |
| Tensor rank | Fixed 4D | Variable | Variable | Variable (max 8) |
| Total LoC | ~250K | ~8K | ~117K | ~3K |

Key insight: **all three converge on the same core ideas** (lazy graphs, composed ops, view/stride tricks, batched execution). The differences are in scale and where they put complexity. ggml puts it in the backend, tinygrad puts it in the scheduler, C-ML puts it in the IR compiler. TinyHVM should keep complexity in the IC reduction layer — that's our unique advantage.

---

## 6. Concrete Action Items for TinyHVM

### Immediate (before MNIST)
- [ ] **Batch Metal command buffers**: Accumulate ops, one `commit` per `thvm_reduce()`
- [ ] **Add softmax, log, reduce_sum ops**: Required for MNIST cross-entropy loss

### Medium-term
- [ ] **Graph-level backend interface**: Replace per-op dispatch with `graph_compute(cgraph)`
- [ ] **Fixed 4D tensors**: Simplify shaders, match ggml convention
- [ ] **Kernel arg structs**: Per-op structs shared between host and Metal shaders
- [ ] **Module/Parameter wrapper**: Thin struct to group layer params, collect for optimizer, toggle train/eval
- [ ] **Ref counting on tensors**: Prevent use-after-free in complex graphs (skip connections, weight sharing)

### Long-term
- [ ] **Metal function constants**: One generic unary/binary kernel, specialized via constants
- [ ] **Timeline memory planner**: Scan graph for tensor lifetimes, pre-allocate one pool, assign offsets (study C-ML's TLSF approach)
- [ ] **Fusion pattern registry**: Registerable match/emit functions for common patterns (MatMul+Bias+ReLU, Conv+BN)
- [ ] **LR schedulers**: At minimum StepLR and CosineAnnealing
- [ ] **Quantization support**: ggml has 30+ quant types — we need at least f16

---

## 7. What NOT to Copy

### From ggml
- **Op count bloat**: ggml has 80+ ops, many ML-specific (ROPE, flash attention, SSM). We should keep our op set minimal and compose.
- **428KB shader file**: ggml-metal.metal is enormous because every op × every quant type = separate kernel. We should aim for generic kernels.
- **Complexity**: ggml is 250K+ lines of core C. Their Metal backend alone is 800K+ lines across 13 files. We're at ~3K lines total. Keep it small.

### From C-ML
- **200+ UOps when 30 suffice**: C-ML has separate UOps for RELU6, HARD_SIGMOID, CELU, SELU, LOGSIGMOID, etc. These all compose trivially from base ops (e.g., `relu6(x) = min(max(x, 0), 6)`). Adding dedicated ops for each activation defeats the point of a composable op set.
- **117K lines for a from-scratch framework**: C-ML has massive abstraction surface area (distributed training, 7 backends, model zoo) before any single path is rock-solid. Build depth before breadth.
- **IR compiler before the basics work**: C-ML has codegen targets for CUDA PTX, SPIRV, WGSL, Metal, OpenCL, LLVM JIT — but the simpler path is to make one backend excellent first. We should make Metal fast and correct before thinking about multi-target codegen.
- **Heavyweight autograd engine**: C-ML's autograd has mutex locks, anomaly detection, nested gradient support. We should keep backward pass simple — tape-based or adjoint on the IC graph — until we actually need higher-order derivatives.
- **Pre-built model zoo**: ResNet, ViT, BERT, GPT-2, YOLO baked into the framework. Models should live in user code, not the framework. Keep the core generic.
