# Engineering Patterns: Lessons from HVM, ggml, tinygrad, and C-ML

> For comprehensive HVM/Interaction Calculus reference (history, theory, performance,
> commercial direction), see **hvm_reference.md**.

Reference doc for TinyHVM development. Studied from source:
- HVM4: `github.com/HigherOrderCO/HVM` (symlinked at `HVM4/`)
- ggml: `github.com/ggerganov/ggml` (local clone)
- tinygrad: `github.com/tinygrad/tinygrad` (symlinked at `tinygrad/`)
- C-ML: `github.com/jaywyawhare/C-ML` (cloned 2026-03-25)

---

## 1. HVM4 — The Interaction Calculus Runtime

HVM4 is the latest HigherOrderCO runtime (~10.8K LoC of C, single translation unit). It implements the **Interaction Calculus** — lambda calculus extended with superpositions and duplications — and is the theoretical foundation TinyHVM builds on.

### Term representation (64-bit word)

```
SUB (1) | TAG (7) | EXT (18) | VAL (40)
[63]    [62-56]   [55-38]    [37-0]
```

46 tags total. Hot tags at indices 0–7: `APP`, `VAR`, `LAM`, `DP0`, `DP1`, `SUP`, `DUP`, `ALO`. Plus `REF`, `ERA`, `NUM`, `MAT`, constructors `C00`–`C16`, operators `OP2`/`EQL`/`AND`/`OR`, and control nodes `INC`/`DSU`/`DDU`/`PRI`.

**TinyHVM adopts this exactly.** Our term layout is `[SUB:1 | TAG:7 | EXT:18 | VAL:38]` — 2 fewer VAL bits (38 vs 40), 256M heap slots instead of 1T. Same idea, same encoding strategy, same SUB-bit substitution mechanism.

### Reduction engine: enter/apply trampoline

HVM4's `wnf()` is a stack-based two-phase loop:

```
ENTER:  walk head position, pushing eliminators (APP, DP0, DP1, OP2, MAT) as frames
APPLY:  pop frames, dispatch interaction based on WHNF result in hand
```

No C recursion on the hot path — all state lives on an explicit `WNF_STACK`. When APP meets LAM, beta-reduce inline. When DP0 meets SUP-same-label, annihilate. When OP2-NUM meets NUM, compute.

**TinyHVM's `thvm_reduce` is a direct port of this.** Our enter phase descends into TAG_TOP arg0, our apply phase dispatches tensor ops when both args are realized. The only difference: HVM4 reduces to WNF (head normal form), we reduce TAG_TOP nodes to TAG_TEN (realized tensors).

### Core interaction rules

| Active Pair | Rule | Analog in TinyHVM |
|---|---|---|
| APP-LAM | β-reduce: substitute arg into body | Same — LAM/APP work identically |
| DUP-SUP (same label) | Annihilate: extract pair | Same — used for CSE |
| DUP-SUP (diff label) | Commute: clone both sides | Same |
| DUP-LAM | Fork lambda, entangle body | Same |
| APP-SUP | Distribute: DUP arg, apply to both | Same |
| OP2-NUM-NUM | Compute scalar op | Our OP2-TEN-TEN fires tensor kernel instead |
| ERA-any | Erase recursively | Our ERA-TOP erases tensor subgraphs (= DCE) |
| APP-ERA | Erase argument | Same |
| DUP-NOD | Clone any node | Our DUP-TOP clones tensor op trees |

### Memory model

- Bump-allocated heap, no GC
- Per-thread heap banks for lock-free parallel allocation
- Substitution via SUB bit (write new value in-place, mark as substituted)
- ALO nodes bridge static book definitions to dynamic heap

**What TinyHVM inherits:**
- Bump allocation (`heap_pos++`), no free
- SUB-bit substitution for variable binding
- `reduce_memo[]` indexed by heap loc (our addition — HVM4 doesn't memoize WNF results, we do because tensor ops are expensive)

### FFI mechanism

HVM4 exposes `HvmApi` to shared libraries: `wnf()`, `heap_alloc()`, `heap_read/set()`, `term_new_*()`, `register_prim()`. Primitives are C functions `Term fn(Term *args)` that can call back into the reducer.

**Relevance:** Our tensor ops (`metal_op_binary`, `cpu_op_mm`, etc.) are conceptually the same as HVM4 FFI primitives — opaque C functions that fire when a TAG_TOP node's args are all realized. The difference is we dispatch via `thvm_interact` switch rather than dlopen.

### What TinyHVM should adopt (not yet done)

**✅ `TAG_REF` + definition table.** HVM4 has a static `BOOK[]` of definitions that `REF(name)` unfolds via ALO cloning. We need this for recursive training loops as a single inet term (see `ic_pure_inet.md` Phase 2).

**✅ Work-stealing parallelism.** HVM4 uses per-thread work-stealing deques for parallel normalization. When we move to graph-level execution, independent tensor op subgraphs could reduce in parallel the same way.

**✅ AOT compilation to standalone C.** HVM4 can emit pure C from an HVM program. If we ever want to compile a trained model to standalone inference code (no runtime dependency), this pattern shows how.

### What NOT to copy from HVM4

- **u32-only numerics.** HVM4 has no floats. We need f32/f16/bf16 for ML. Our TAG_NUM stores f32 via bitcast, and tensor buffers are dtype-aware.
- **No memoization.** HVM4 doesn't cache WNF results — fine for cheap symbolic reduction, but tensor ops are expensive. Our `reduce_memo[]` is essential.
- **Million-interaction overhead for bulk math.** A 1024×1024 matmul would be ~2 billion HVM interactions. We dispatch one Metal/MPS kernel. The hybrid architecture (IC for graph topology, GPU for tensor arithmetic) is the right call.

---

## 2. Backend Abstraction (ggml)

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

**✅ The `iface` + `context` pattern.** Our current `Backend` is already a flat vtable, which is fine for now. But as we add features (async, events, graph optimization), ggml's layered approach scales better.

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

## 3. Tensor Struct (ggml)

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

**Gap in TinyHVM:** Our `View` struct has variable rank (up to `MAX_DIM=8`). This means every kernel needs a rank-dependent loop. Consider: fixed 4D could simplify Metal shaders significantly.

### Key pattern: `view_src` + `view_offs`

Views in ggml are explicit: a view tensor has `view_src` pointing to the original tensor and `view_offs` for the offset. The backend uses this to find the correct `MTLBuffer` and offset.

**TinyHVM equivalent:** Our `View` struct already has `offset` and `strides`, which is similar. But we don't track the source tensor explicitly — we just modify the view on the same buffer ID. This works but loses the lineage.

---

## 4. Metal Backend Patterns (ggml)

From ggml's `src/ggml-metal/`:

### Kernel argument structs

ggml defines **separate C structs for each kernel's arguments** (e.g., `ggml_metal_kargs_unary`, `ggml_metal_kargs_bin`, `ggml_metal_kargs_mul_mm`). These are shared between the ObjC host code and Metal shaders.

Our `ViewParams` is a single struct for all ops. ggml's approach is more verbose but allows per-op optimization (e.g., matmul doesn't need all the stride fields that a copy op does).

### Metal function constants

ggml uses Metal **function constants** (`FC_UNARY`, `FC_BIN`, etc.) to specialize shader functions at compile time. One generic kernel function handles multiple ops by switching on a function constant, avoiding redundant shader code.

**TinyHVM current approach:** Separate kernel functions per op (`unary_neg`, `unary_relu`, etc.). This is cleaner for small op counts but doesn't scale to 100+ ops.

### Command buffer batching

ggml's Metal `graph_compute()` encodes *all ops* in the graph into command buffers before committing. Our per-op `commit` + `waitUntilCompleted` is the main performance bottleneck — we should batch.

---

## 5. tinygrad Architecture

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

## 6. C-ML Architecture

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

---

## 7. Pattern Comparison

| Concern | HVM4 | ggml | tinygrad | C-ML | TinyHVM |
|---------|------|------|----------|------|---------|
| Graph repr | interaction net | tensor.src[] | LazyBuffer | IRNode | IC heap (TAG_TOP) |
| Lazy eval | Yes (WNF) | No (eager) | Yes (realize) | Yes (ensure_executed) | Yes (thvm_reduce) |
| Reduction | enter/apply trampoline | — | — | — | enter/apply trampoline |
| Fusion | DUP-TOP rewriting | Backend-level | Scheduler | Plugin registry | Pattern match in reducer |
| Memory | Bump alloc, no GC | Static alloc | Lazy alloc | TLSF + timeline | Bump alloc + reduce_memo |
| Autograd | DUP = differentiation | None (inference only) | Graph rewrite | IR backward nodes | GRAD node on IC heap |
| Shaders | N/A (symbolic) | One giant file | Codegen per kernel | Codegen per target | Separate functions |
| Tensor rank | N/A (scalars only) | Fixed 4D | Variable | Variable | Variable (max 8) |
| Total LoC | ~10.8K | ~250K | ~8K | ~117K | ~5.9K |

Key insight: **all converge on the same core ideas** (lazy graphs, composed ops, view/stride tricks, batched execution). The differences are in scale and where they put complexity. ggml puts it in the backend, tinygrad puts it in the scheduler, C-ML puts it in the IR compiler, HVM4 puts it in interaction rules. TinyHVM inherits HVM4's reduction engine and adds tensor semantics on top — keeping complexity in the IC reduction layer.

---

## 8. Concrete Action Items for TinyHVM

### Immediate (before MNIST)
- [ ] **Batch Metal command buffers**: Accumulate ops, one `commit` per `thvm_reduce()`
- [ ] **Add softmax, log, reduce_sum ops**: Required for MNIST cross-entropy loss

### Medium-term
- [ ] **Graph-level backend interface**: Replace per-op dispatch with `graph_compute(cgraph)`
- [ ] **Fixed 4D tensors**: Simplify shaders, match ggml convention
- [ ] **Kernel arg structs**: Per-op structs shared between host and Metal shaders
- [ ] **Module/Parameter wrapper**: Thin struct to group layer params, collect for optimizer, toggle train/eval
- [ ] **Ref counting on tensors**: Prevent use-after-free in complex graphs (skip connections, weight sharing)
- [ ] **TAG_REF + definition table**: Enable recursive training as one inet term (HVM4 pattern)

### Long-term
- [ ] **Metal function constants**: One generic unary/binary kernel, specialized via constants
- [ ] **Timeline memory planner**: Scan graph for tensor lifetimes, pre-allocate one pool, assign offsets (study C-ML's TLSF approach)
- [ ] **Fusion pattern registry**: Registerable match/emit functions for common patterns (MatMul+Bias+ReLU, Conv+BN)
- [ ] **LR schedulers**: At minimum StepLR and CosineAnnealing
- [ ] **Quantization support**: ggml has 30+ quant types — we need at least f16
- [ ] **Work-stealing parallel reduction**: HVM4 pattern for independent tensor op subgraphs

---

## 9. What NOT to Copy

### From HVM4
- **u32-only numerics**: No floats, no dtype system. ML needs f32/f16/bf16.
- **No memoization**: Fine for cheap symbolic ops, catastrophic for expensive GPU dispatches.
- **Scalar-per-interaction model**: 1024×1024 matmul = 2B interactions. We dispatch one kernel.

### From ggml
- **Op count bloat**: ggml has 80+ ops, many ML-specific (ROPE, flash attention, SSM). We should keep our op set minimal and compose.
- **428KB shader file**: ggml-metal.metal is enormous because every op × every quant type = separate kernel. We should aim for generic kernels.
- **Complexity**: ggml is 250K+ lines of core C. Their Metal backend alone is 800K+ lines across 13 files. We're at ~5.9K lines total. Keep it small.

### From C-ML
- **200+ UOps when 30 suffice**: C-ML has separate UOps for RELU6, HARD_SIGMOID, CELU, SELU, LOGSIGMOID, etc. These all compose trivially from base ops (e.g., `relu6(x) = min(max(x, 0), 6)`). Adding dedicated ops for each activation defeats the point of a composable op set.
- **117K lines for a from-scratch framework**: C-ML has massive abstraction surface area (distributed training, 7 backends, model zoo) before any single path is rock-solid. Build depth before breadth.
- **IR compiler before the basics work**: C-ML has codegen targets for CUDA PTX, SPIRV, WGSL, Metal, OpenCL, LLVM JIT — but the simpler path is to make one backend excellent first. We should make Metal fast and correct before thinking about multi-target codegen.
- **Heavyweight autograd engine**: C-ML's autograd has mutex locks, anomaly detection, nested gradient support. We should keep backward pass simple — GRAD nodes on the IC heap — until we actually need higher-order derivatives.
- **Pre-built model zoo**: ResNet, ViT, BERT, GPT-2, YOLO baked into the framework. Models should live in user code, not the framework. Keep the core generic.
