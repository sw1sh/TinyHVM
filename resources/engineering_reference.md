# Engineering Patterns: Lessons from ggml and tinygrad

Reference doc for TinyHVM development. Studied from source: `/Users/swish/src/ggml`.

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

## 5. Concrete Action Items for TinyHVM

### Immediate (before MNIST)
- [ ] **Batch Metal command buffers**: Accumulate ops, one `commit` per `thvm_reduce()`
- [ ] **Add softmax, log, reduce_sum ops**: Required for MNIST cross-entropy loss

### Medium-term
- [ ] **Graph-level backend interface**: Replace per-op dispatch with `graph_compute(cgraph)`
- [ ] **Fixed 4D tensors**: Simplify shaders, match ggml convention
- [ ] **Kernel arg structs**: Per-op structs shared between host and Metal shaders

### Long-term
- [ ] **Metal function constants**: One generic unary/binary kernel, specialized via constants
- [ ] **Static memory allocation**: Pre-compute buffer sizes from the graph, two pools (weights + activations)
- [ ] **Quantization support**: ggml has 30+ quant types — we need at least f16

---

## 6. What NOT to Copy from ggml

- **Op count bloat**: ggml has 80+ ops, many ML-specific (ROPE, flash attention, SSM). We should keep our op set minimal and compose.
- **428KB shader file**: ggml-metal.metal is enormous because every op × every quant type = separate kernel. We should aim for generic kernels.
- **Complexity**: ggml is 250K+ lines of core C. Their Metal backend alone is 800K+ lines across 13 files. We're at ~700 lines total for the whole framework. Keep it small.
