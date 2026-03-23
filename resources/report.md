# Computational Graphs on Interaction Nets: A Feasibility Report

## Executive Summary

This report explores whether tinygrad's computational graph model can be built on top of HVM's interaction net runtime, creating a deep learning framework that inherits HVM's massively parallel execution model. The core thesis: **map tinygrad's lazy UOp graph onto HVM's interaction combinator network, gaining automatic parallelism and optimal sharing for free.**

The conclusion is that this is **feasible but non-trivial**, with significant architectural alignment in some areas and fundamental impedance mismatches in others. The most promising path is a **hybrid architecture** where interaction nets manage the computational graph topology, scheduling, and symbolic differentiation, while bulk tensor arithmetic dispatches to conventional GPU kernels.

---

## 1. Architecture Comparison

### 1.1 tinygrad — The Computational Graph Side

tinygrad has four stages:

```
Frontend (Tensor) → Scheduler → Codegen/Lowering → Runtime Execution
```

**Key abstractions:**

| Concept | Role |
|---------|------|
| `UOp` | Universal operation node — the IR vertex. DAG of ~60 ops |
| `ShapeTracker` | Tracks views (reshape, permute, expand) without copying data |
| `ScheduleItem` | One GPU kernel — the scheduler splits the UOp graph here |
| `Kernel` | AST specifying what a single kernel computes |
| `ExecItem` | Compiled kernel + buffer bindings, ready to run |

**Op taxonomy (from `tinygrad/uop/__init__.py`):**

- **Movement**: RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP → all fold into `VIEW` via ShapeTracker
- **Compute**: ADD, MUL, REDUCE_AXIS, EXP2, LOG2, SIN, SQRT, etc.
- **Buffer**: LOAD, STORE, BUFFER, ASSIGN, COPY
- **Control**: RANGE, IF, ENDRANGE, BARRIER

The graph is **lazy**: ops build up a DAG until `.realize()` is called, at which point the scheduler fuses eligible ops into kernels, codegen lowers to device code, and the runtime executes.

### 1.2 HVM — The Interaction Net Side

HVM has evolved through four generations:

| Version | Language | Status | Key Innovation |
|---------|----------|--------|----------------|
| HVM1 (2022) | Rust | Archived | First practical interaction combinator evaluator |
| HVM2 (2024) | Rust/CUDA | Stable | 74B MIPS on RTX 4090; C and CUDA codegen |
| HVM3 (2026) | Haskell | WIP | Simplified core |
| **HVM4 (2026)** | **C** | **Active** | **Interaction Calculus; FFI; AOT C compilation** |

**HVM4 is the latest and most relevant.** It implements the **Interaction Calculus**, which extends lambda calculus with two dual primitives:

- **Duplications** (`! x &= v; t`) — one value in two locations
- **Superpositions** (`&{a, b}`) — two values in one location

**Four core interactions:**

| Rule | When | Effect |
|------|------|--------|
| **APP-LAM** | Application meets lambda | β-reduction (substitution) |
| **DUP-SUP** | Duplication meets superposition (same label) | Annihilation (pair extraction) |
| **APP-SUP** | Application meets superposition | Distribution (propagation) |
| **DUP-LAM** | Duplication meets lambda | Lambda cloning with shared body |

**HVM4 memory layout (64-bit term pointers):**

```
SUB (1 bit) | TAG (7 bits) | EXT (16 bits) | VAL (40 bits)
```

Tags include: APP, LAM, SUP, DUP, NUM, OP2, MAT, SWI, CTR, REF, etc.

HVM4 supports:
- Binary arithmetic operators: `+`, `-`, `*`, `/`, `%`, `<<`, `>>`, `<`, `>`, `==`, `!=`, `&&`, `||`, `^`, `~`
- Constructors and pattern matching
- AOT compilation to standalone C
- FFI for shared libraries
- Superposition collapse with configurable limits

---

## 2. Structural Alignment

### 2.1 Where the Models Converge

| tinygrad Concept | HVM Equivalent | Alignment |
|-----------------|----------------|-----------|
| UOp DAG | Interaction net | ✅ Both are directed graphs of nodes with ports/edges |
| Lazy evaluation | Lazy reduction | ✅ Both defer computation until needed |
| Graph rewriting (pattern matcher) | Interaction rules | ✅ Both transform the graph via local rules |
| Kernel fusion | Annihilation/commutation | ⚠️ Analogous but different granularity |
| `UOp.toposort()` | Redex scheduling | ✅ Both need topological ordering |
| Movement ops as zero-cost views | No data copy in net rewrites | ✅ Both avoid unnecessary data movement |

**Strongest alignment: Both are graph rewriting systems.** tinygrad's `PatternMatcher` rewrites UOp graphs using pattern/rewrite rules. HVM rewrites interaction nets using interaction rules. The mechanics are structurally similar — a pattern matches a local subgraph, and the rewrite replaces it with a new subgraph.

### 2.2 Where They Diverge

| Dimension | tinygrad | HVM |
|-----------|----------|-----|
| **Data model** | Dense tensors (multi-dimensional arrays) | Individual terms (scalars, pairs, lambdas) |
| **Parallelism** | SIMD/SIMT — same op on many data points | MIMD — independent interactions reduce in parallel |
| **Arithmetic** | Bulk vectorized (thousands of FLOPs per kernel) | Scalar (one `OP2` per interaction) |
| **Memory** | Contiguous buffers, coalesced access | Pointer-chasing graph traversal |
| **Numerics** | float32, float16, bfloat16, int8, etc. | 24-bit numerics (u24, i24, f24) |
| **Reduction** | Explicit `REDUCE_AXIS` over tensor dimensions | Implicit via recursive net structure |
| **Shapes** | Central concept (ShapeTracker) | No concept of shape — everything is a tree |

**The fundamental tension: tinygrad operates on *bulk data* (tensors), HVM operates on *individual terms*.** A matmul in tinygrad is one fused kernel touching millions of floats. In HVM, each float would be a separate `NUM` node connected by `OP2` interaction nodes — millions of nodes, millions of interactions. This is catastrophically inefficient for dense linear algebra.

---

## 3. Feasibility Analysis

### 3.1 What Interaction Nets Can Do Well for Comp Graphs

**A. Graph topology management and scheduling**

Interaction nets excel at managing complex graph topologies with automatic garbage collection (erasure nodes), optimal sharing (no redundant computation), and confluent parallel rewriting. This maps well to:

- Building and transforming the *computation graph itself* (not the data)
- Scheduling which kernels to run and in what order
- Managing dependencies between operations
- Automatic memory lifecycle (HVM's ERA nodes ≈ buffer deallocation)

**B. Symbolic differentiation (autograd)**

Backpropagation is fundamentally a graph transformation — reverse the edges, apply chain rule at each node. Interaction nets are ideal for this:

```
Forward:  z = f(g(x))     →  UOp graph
Backward: dz/dx = f'(g(x)) · g'(x)  →  reversed interaction net
```

The DUP/SUP mechanism handles the key autograd challenge: when a value is used in multiple downstream operations, its gradient must be summed. DUP naturally splits the gradient flow, and the annihilation/commutation rules handle the recombination.

**C. Dynamic/symbolic computation**

HVM's superposition mechanism could elegantly handle:
- **Symbolic shapes**: `&{batch_size, 1}` — a shape that is superposed
- **Optional operations**: `&{relu, gelu}` — architecture search as superposition
- **Conditional execution**: Pattern matching (MAT/SWI) for control flow in the graph

**D. Kernel fusion as graph rewriting**

tinygrad's scheduler already works by graph rewriting (splitting a large UOp DAG into fusable subgraphs). This is naturally expressible as interaction net rewrites. The key insight: **fusion boundaries are interaction points** — wherever two kernel subgraphs interact, that's where you decide to fuse or split.

### 3.2 What Interaction Nets Cannot Do (Directly)

**A. Dense tensor arithmetic**

HVM's MIPS (millions of interactions per second) are impressive for symbolic computation, but each interaction processes O(1) data. A 1024×1024 matmul requires ~2 billion FLOPs — that's 2 billion interactions through HVM's graph, versus one GPU kernel launch in tinygrad. This is a ~1000× overhead minimum.

**B. SIMD/SIMT execution**

GPU tensor cores process 4×4 matrices in a single clock cycle. HVM's parallelism is MIMD (different interactions on different data) not SIMD (same operation on contiguous data). These are fundamentally different parallelism models.

**C. Memory-coalesced access patterns**

Tensors in tinygrad are contiguous in memory, enabling coalesced GPU memory access (reading 128 bytes in one transaction). HVM's graph nodes are scattered in heap memory with pointer chasing — the antithesis of GPU-friendly access patterns.

**D. High-precision numerics**

HVM4 uses 24-bit numerics (u24, i24, f24). Deep learning requires at minimum float16, typically float32, and sometimes float64. The 24-bit constraint would cause unacceptable precision loss.

---

## 4. Proposed Hybrid Architecture

The optimal approach is a **two-level architecture** where interaction nets manage the *computation graph* while conventional GPU kernels handle *tensor arithmetic*:

```
┌──────────────────────────────────────────────────┐
│           Interaction Net Layer (HVM4)            │
│                                                    │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │ Tensor  │───▶│ MatMul  │───▶│  ReLU   │       │
│  │ (REF)   │    │ (REF)   │    │ (REF)   │       │
│  └─────────┘    └─────────┘    └─────────┘       │
│       │              │              │              │
│  ┌────▼──────────────▼──────────────▼────┐        │
│  │    Graph Topology, Scheduling,        │        │
│  │    Autograd, Fusion Decisions         │        │
│  └───────────────────────────────────────┘        │
├──────────────────────────────────────────────────┤
│           Tensor Kernel Layer (GPU)                │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Metal/   │  │ CUDA     │  │ OpenCL   │       │
│  │ fused    │  │ kernel   │  │ kernel   │       │
│  │ kernel   │  │          │  │          │       │
│  └──────────┘  └──────────┘  └──────────┘       │
└──────────────────────────────────────────────────┘
```

### 4.1 Mapping tinygrad Ops to HVM4

```hvm
-- Tensor as an opaque reference
@Tensor = λshape. λdtype. λbuffer_id. #Tensor{shape, dtype, buffer_id}

-- MatMul as an interaction net node
@matmul = λa. λb.
  ! shape &= (@matmul_shape (shape_of a) (shape_of b));
  @Tensor shape₀ (@float32) (@gpu_matmul (buf a) (buf b) shape₁)

-- Elementwise add — can be fused
@add = λa. λb.
  @Tensor (shape_of a) (dtype_of a) (@gpu_add (buf a) (buf b))

-- ReLU — can be fused with preceding op
@relu = λx.
  @Tensor (shape_of x) (dtype_of x) (@gpu_relu (buf x))

-- Autograd via DUP: when a tensor is used twice,
-- its gradient contributions are automatically handled
@loss = λx.
  ! x &= (@forward x);
  (@criterion x₀ (@label)) + (@regularize x₁)
```

### 4.2 Key Design Elements

**A. Tensors as opaque references (REF/CTR nodes)**

Tensors are not decomposed into individual numbers. Instead, each tensor is a constructor node (`#Tensor{shape, dtype, buffer_id}`) containing metadata and a reference to a GPU buffer. The interaction net manipulates the *graph structure*, not the *data*.

**B. FFI for GPU dispatch**

HVM4's FFI mechanism is the critical enabler. GPU kernels (matmul, conv2d, etc.) are loaded as shared libraries. When the interaction net reduces to a `@gpu_matmul` primitive, it calls out to the native kernel via FFI:

```c
// FFI bridge: HVM4 → GPU kernel
Term hvm_prim_gpu_matmul(Term args) {
    // Extract buffer IDs from HVM terms
    // Dispatch to cuBLAS/Metal/etc.
    // Return new buffer ID as HVM Term
}
```

**C. Lazy evaluation alignment**

Both systems are lazy. The interaction net builds up unreduced nodes until a result is demanded. This maps directly to tinygrad's `kernelize()`/`realize()` pattern:

- Building the net = tinygrad's lazy graph construction
- Reducing the net = tinygrad's `realize()`
- Net reduction order = kernel scheduling

**D. Autograd as graph duplication**

When the backward pass needs the computation graph, HVM's DUP mechanism creates a shared copy automatically. The forward graph and backward graph share structure via DUP/SUP mechanics — this is literally optimal sharing, which is what autograd wants.

**E. Superpositions for architecture search / dynamic batching**

```hvm
-- Try two activation functions simultaneously
@activation = &{@relu, @gelu}

-- The rest of the network processes BOTH paths in superposition
@output = (@linear (@activation (@input)))

-- Collapse to get both results
-- Result: &{output_with_relu, output_with_gelu}
```

This enables neural architecture search as a native language feature.

### 4.3 Kernel Fusion via Interaction Rules

Define custom interaction rules for fusion:

```
-- Two consecutive elementwise ops fuse:
-- @add(@relu(x)) → @fused_add_relu(x)

-- Rule: when a REF to an elementwise op meets another REF to an elementwise op,
-- they ANNIHILATE into a fused kernel
@fuse_check = λop1. λop2.
  ? ((@is_elementwise op1) .&. (@is_elementwise op2))
    { 0: @compose op1 op2    -- can't fuse, compose normally
    ; 1: @fused op1 op2 }    -- fuse into single kernel
```

---

## 5. Performance Considerations

### 5.1 Expected Wins

| Scenario | Benefit |
|----------|---------|
| Complex graph scheduling | Automatic parallel scheduling via confluent reduction |
| Autograd | Optimal sharing eliminates redundant storage of activations |
| Dynamic control flow | MAT/SWI handle conditionals without graph breaks |
| Multi-model pipelines | Superpositions run multiple models in parallel |
| Graph-level optimizations | Interaction rules = pattern-match-and-rewrite, just like tinygrad |

### 5.2 Expected Costs

| Concern | Mitigation |
|---------|------------|
| Graph manipulation overhead | HVM4's C backend is fast; graph rewrites are lightweight vs. kernel execution |
| 24-bit numerics | Use only for graph metadata; actual tensor data stays in GPU buffers |
| Pointer-chasing in net | Only for graph topology, not bulk data |
| FFI call overhead | Batch FFI calls; amortize over large kernel launches |
| Learning curve | Provide tinygrad-compatible Python API that compiles to HVM4 |

### 5.3 Where This Beats Existing Approaches

1. **vs. PyTorch**: No Python GIL for graph operations; true parallel graph manipulation
2. **vs. XLA/MLIR**: Interaction rules are simpler than MLIR dialects; composition is automatic
3. **vs. pure tinygrad**: Autograd gets optimal sharing for free; superpositions enable native NAS
4. **vs. pure HVM**: Tensors stay as GPU buffers, not decomposed into millions of nodes

---

## 6. Implementation Roadmap

### Phase 1: Core Runtime ✅

Completed. TinyHVM is a standalone C library (~4700 LoC) with:

- **Interaction calculus core**: 64-bit term encoding (SUB:1 | TAG:7 | EXT:18 | VAL:38), bump-allocated heap, weak normal form reduction engine
- **Tensor abstraction**: `Shape` struct (variable rank up to 8D), `View` with strides/offset (tinygrad-inspired ShapeTracker), dtype-aware with `dtype_size()` helper
- **Backend interface**: `Backend` vtable with `init/shutdown/buf_alloc/buf_free/buf_write/buf_read/op_unary/op_binary/op_mm` + CNN dispatch ops + profiling. Runtime selection via `thvm_device("cpu"|"metal")`
- **CPU backend** (`cpu.c`): Accelerate/vDSP matmul, strided elementwise ops
- **Metal backend** (`metal.m`): Compute shaders + MPS matmul, `StorageModeShared` for zero-copy on Apple Silicon, command buffer batching
- **23 UOps** (aligned with [tinyspec](https://github.com/tinygrad/tinyspec)): movement (reshape/permute/expand/shrink/pad), elementwise (neg/exp/log/relu/sqrt/cast + add/mul/div/sub/max/cmp), reduce (sum/rmax), matmul
- **Broadcasting**: Full numpy-style shape broadcasting with stride manipulation
- **Tests**: 127/127 unit tests passing on both CPU and Metal

```
src/tinyhvm.h      409 lines  — types, constants, API, profiling, Layer abstraction
src/tinyhvm.c     1586 lines  — reduction engine, views, autograd, UOp compositions
src/cpu.c          219 lines  — CPU backend (Accelerate)
src/metal.m        608 lines  — Metal backend (MPS + compute shaders + profiling)
src/layers.c       572 lines  — CNN layers (direct Metal dispatch), Layer sequential
src/shaders.metal  475 lines  — Metal compute kernels
test/test_term.m   409 lines  — 93 unit tests
test/test_train.m  144 lines  — XOR training end-to-end
test/beautiful_mnist.m  281 lines — CNN MNIST training (96.2% accuracy)
```

### Phase 2: Autograd & Training ✅

Completed:
- **IC-native autograd** (`thvm_grad`): JAX-style — lazy Term that reduces to ∂y/∂x. Supports `grad(grad(f))`. Removed old `thvm_backward` tape walk in favor of pure IC reduction
- **Provenance tracking**: `TensorMeta.creator_op` + `src_ids[]` — no separate tape needed
- **Gradient rules**: add, sub, mul, div, matmul (with transpose), relu, exp, log, sum, rmax, expand, reshape, permute, shrink, pad, pool_gather
- **Gradient fixes**: 6 bugs fixed — SUM reduction for chained reduce, `thvm_to_host` strided copy for non-contiguous views, `tensor_reduce_sum_to` buffer overrun, broadcast ADD/SUB gradients, SUM/RMAX gradient stride preservation
- **SGD training**: XOR 2-layer MLP converges in 7 epochs
- **CNN training (beautiful_mnist)**: Conv2d + BatchNorm + MaxPool + Linear layers via direct Metal dispatch. Loss 2.92→0.11, 96.2% test accuracy, ~46ms/step
- **Device-agnostic profiling**: `ThvmProfile`, `thvm_prof_tick/record`, `THVM_PROFILE=1`
- **Layer abstraction**: Tagged union `Layer` + `thvm_sequential`

### Phase 3: Kernel Fusion (in progress)

Prototype implemented (`THVM_FUSE=1` env var, 127/127 tests pass):
- **IC fusion via intermediate accumulator nodes**: `FuseState` absorbs unary elementwise ops pairwise into a reduce, emitting one fused CPU dispatch
- **Design doc**: `resources/ic_fusion.md` — tinygrad PatternMatcher analysis, state machine, projected 98% memory reduction for conv intermediates
- **Current scope**: unary+reduce (e.g. `SUM(RELU(x))`), binary ops excluded (needs broadcast-aware indexing)
- **Next**: binary op fusion with stride-aware indexing, Metal codegen for fused kernels, gradient-safe fusion

### Phase 4: Full Framework (planned)
1. Quantization support (f16 at minimum)
2. Static memory allocation (pre-compute buffer sizes from graph)
3. Multi-backend support (CUDA via same Backend interface)
4. Python or higher-level bindings

---

## 7. Open Questions

1. **Can HVM4's reduction order match tinygrad's BEAM-optimal kernel scheduling?** HVM reduces by priority (hi/lo redex bags in HVM2; INC wrappers in HVM4). This may not perfectly align with the cost-model-driven scheduling tinygrad uses.

2. **How does superposition collapse interact with side-effecting GPU dispatch?** If a superposition contains two different kernel calls, both execute. Is this always desirable?

3. **Memory management**: tinygrad carefully manages GPU buffer lifecycle. Can HVM's ERA (erasure) nodes reliably trigger GPU buffer deallocation?

4. **Numerical precision**: HVM4's native numerics are 24-bit. While tensor data bypasses this via FFI, shape computations and indexing arithmetic need full 32/64-bit integers.

5. **Is the graph-level overhead worth it?** For simple sequential models (e.g., MLP), the interaction net layer adds overhead without much benefit. The payoff is in complex, dynamic, multi-path computations.

---

## 8. Conclusion

Building computational graphs on interaction nets is **feasible as a hybrid architecture** where:

- **The interaction net** manages graph topology, scheduling, autograd, and fusion
- **GPU kernels** handle dense tensor arithmetic via FFI

This is *not* about computing matmuls through interaction combinators (that would be absurd). It's about using the interaction net as a **powerful, parallel, formally-grounded graph rewriting engine** that replaces the ad-hoc Python-based schedulers and pattern matchers in existing DL frameworks.

The strongest argument for this approach: **tinygrad already IS a graph rewriting system** (PatternMatcher, UPat, graph_rewrite). The Interaction Calculus is the mathematical theory that makes graph rewriting rigorous, confluent, and automatically parallel. TinyHVM would be tinygrad with its implicit graph rewriting made explicit and provably optimal.

---

## References

- [tinygrad source](https://github.com/tinygrad/tinygrad) — UOp IR, scheduler, codegen
- [HVM4 source](https://github.com/HigherOrderCO/HVM4) — Latest HVM, Interaction Calculus
- [HVM2 paper](https://github.com/HigherOrderCO/HVM2/blob/main/paper/HVM2.pdf) — Performance benchmarks
- [Lafont 1997](https://www.semanticscholar.org/paper/Interaction-Combinators-Lafont/6cfe09aa6e5da6ce98077b7a048cb1badd78cc76) — Foundational theory
- [Interaction Calculus docs](https://github.com/HigherOrderCO/HVM4/blob/main/docs/theory/interaction_calculus.md) — HVM4 theory
- [tinygrad developer docs](https://docs.tinygrad.org/) — Architecture overview
