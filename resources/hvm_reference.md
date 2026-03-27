# HVM Reference: Higher-order Virtual Machine & Interaction Calculus

Comprehensive reference on HVM by Higher Order Company (Victor Taelin).
Covers architecture evolution, Interaction Calculus theory, performance,
commercial direction, and relevance to TinyHVM.

> For TinyHVM-specific architecture that builds on HVM4, see **ic_arch.md**.
> For the tinygrad-to-HVM feasibility analysis, see **report.md**.
> For engineering patterns extracted from HVM4 source, see **engineering_reference.md**.

---

## 1. Overview

**HVM** (Higher-order Virtual Machine) is a massively parallel, optimal functional
runtime based on Interaction Combinators. Created by Victor Taelin and developed
by Higher Order Company (HOC), headquartered in Delaware with the core team in
Rio de Janeiro.

Key claim: HVM is the first practical runtime that achieves **optimal beta reduction**
(Levy's sense) with massive parallelism — every independent interaction can run on
a separate thread, scaling from CPU cores to thousands of GPU threads.

**Ecosystem:**
- **HVM** — the runtime (C/Rust/CUDA)
- **Bend** — high-level language compiling to HVM ("feels like Python, scales like CUDA")
- **Kind** — proof assistant / dependently-typed language running on HVM
- **SupGen / NeoGen** — program synthesizers powered by superposition-based search

---

## 2. Version History

| Version | Year | Language | Key Innovation | Status |
|---------|------|----------|----------------|--------|
| HVM1 | 2022 | Rust | First practical IC evaluator; lazy evaluation | Archived |
| HVM2 | 2024 | Rust/CUDA | Massively parallel GPU evaluator; eager evaluation; 74K MIPS on RTX 4090 | Stable |
| HVM3 | 2025 | Haskell→C | Optimal polarized atomic linker; lazy+eager hybrid; compiles to machine code | Active |
| HVM4 | 2025-26 | C (100%) | Interaction Calculus (not just combinators); FFI; AOT C compilation; native types | Active |

### HVM1 (2022)
- First implementation showing interaction combinators could be practical
- ~3x slower than GHC single-threaded, but with lazy cloning of lambdas
- Parallel evaluator was experimental and buggy
- Proved the concept: optimal reduction works outside academia

### HVM2 (2024)
- Major rewrite. First **correct** massively parallel evaluator
- **Eager evaluation** model (allocates entire structures)
- Near-ideal speedup: 400 MIPS (M3 Max single-thread) → 5,200 MIPS (16 threads) → 74,000 MIPS (RTX 4090, 32K threads)
- CUDA backend sustained 45 billion interactions per second
- Published at ICFP 2024 (FProPer workshop): "HVM2: A Parallel Evaluator for Interaction Combinators"
- Uses SIC (Symmetric Interaction Combinators, Mazza 2007)
- Could NOT compile functions with superpositions — interpreter only
- GitHub: 11.2K stars

### HVM3 (2025)
- Combines strengths of HVM1 (lazy) and HVM2 (parallel)
- **Optimal Polarized Atomic Linker**: exploits polarities for minimal CPU instructions
- Lazy evaluation restored: positive vars point to negative variable location, enabling graph traversal with lazy algorithms — not possible in HVM2
- **1/3 less memory** than HVM2: eliminates "subst map", reuses negative binder locations
- First long-term compiler: identifies fragments for compilation to C/CUDA, achieving **10-100x speedup** in some cases, surpassing GHC single-core
- HVM3-Nano: 216 MIPS per M4 core. Claude Code optimized to 328 MIPS (+51%)
- GitHub: 280 stars, ~400 files, ported to Haskell-based Kind language

### HVM4 (2025-2026)
- **Interaction Calculus** (not just combinators) — extends lambda calculus with SUP/DUP
- Entire codebase in C (~10.8K LoC, single translation unit)
- SupGen built-in as core feature
- Native type system running directly on interaction nets — usable as fast parallel proof verifier
- General method to compile IC functions (including those with superpositions) to **zero-overhead machine code**
- Performance: 130-190 MIPS (190 with AI-assisted optimization — absolute record for this benchmark)
- FFI: `HvmApi` exposes `wnf()`, `heap_alloc()`, `heap_read/set()`, `term_new_*()`, `register_prim()`
- AOT compilation to standalone C
- **This is the version TinyHVM's architecture is based on**

---

## 3. Theoretical Foundation

### 3.1 Interaction Nets (Lafont 1990)

Yves Lafont introduced interaction nets at POPL '90 as a graphical model of computation
derived from proof structures in linear logic.

- Undirected graphs of **agents** connected through **ports**
- Each agent has one **principal port** and zero or more **auxiliary ports**
- Two agents connected via principal ports form an **active pair** (redex)
- Reduction: replace active pair with a new subgraph according to the agent types
- **Key property**: at most one rule applies to any active pair → deterministic, strongly confluent

### 3.2 Interaction Combinators (Lafont 1997)

Lafont's seminal paper "Interaction Combinators" showed that **three symbols with six rules** form a universal computation system:

| Symbol | Name | Ports | Role |
|--------|------|-------|------|
| gamma (γ) | Constructor | 1 principal + 2 auxiliary | Build data |
| delta (δ) | Duplicator | 1 principal + 2 auxiliary | Clone/share data |
| epsilon (ε) | Eraser | 1 principal only | Garbage collect |

**Six interaction rules:**

| Active Pair | Rule Type | Effect |
|------------|-----------|--------|
| γ-γ (same) | Annihilation | Extract pair, connect aux ports |
| δ-δ (same label) | Annihilation | Extract pair, connect aux ports |
| γ-δ | Commutation | Clone constructor through duplicator |
| δ-δ (diff label) | Commutation | Clone duplicator through duplicator |
| γ-ε | Erasure | Erase constructor, propagate ε to aux ports |
| δ-ε | Erasure | Erase duplicator, propagate ε to aux ports |

**Key theorem**: This system can simulate **any other interaction net system** without increasing computational complexity. It is a universal model of computation, alternative to lambda calculus and Turing machines.

### 3.3 Interaction Calculus (Taelin, HVM4)

Victor Taelin's Interaction Calculus extends lambda calculus with two dual primitives:

- **Duplication** `! x &= v; t` — one value available in two locations (DUP)
- **Superposition** `&{a, b}` — two values in one location (SUP)

**Core grammar:**
```
Term ::= λx. body          -- Lambda (abstraction)
       | (f x)              -- Application
       | x                  -- Variable
       | &L{a, b}           -- Superposition (labeled L)
       | ! &L{x, y} = t; b  -- Duplication (labeled L)
       | *                  -- Erasure
```

**Four core interactions:**

| Rule | When | Effect |
|------|------|--------|
| APP-LAM | Application meets lambda | Beta-reduction (substitution) |
| DUP-SUP (same label) | Duplication meets superposition | Annihilation (pair extraction) |
| APP-SUP | Application meets superposition | Distribution (clone arg, apply to both) |
| DUP-LAM | Duplication meets lambda | Clone lambda, entangle body |

Plus erasure rules (ERA propagates through any node) and commutation (DUP-SUP with different labels).

### 3.4 Optimal Reduction (Lamping 1988)

- John Lamping (Xerox PARC) developed a data structure for family-complete reductions
- Enables **sharing of computations inside lambdas** — something GHC cannot do
- HVM achieves near-optimal reduction: computes normal form with minimal beta-reductions
- For higher-order computations, can be **exponentially faster** than standard evaluators

**Bookkeeping challenge**: Asperti & Mairson (1992) showed bookkeeping steps can dwarf useful work. HVM's solution: a **bookkeeping-free** variant that stores constant level on every initial fan node — covers 99% of lambda-calculus cases without full oracle machinery. Trades some optimality for massive practical efficiency.

---

## 4. Architecture Deep Dive

### 4.1 Term Representation (64-bit word)

```
SUB (1) | TAG (7) | EXT (18) | VAL (40)
[63]    [62-56]   [55-38]    [37-0]
```

- **SUB bit**: marks substituted variables (in-place write, no GC needed)
- **TAG**: node type (APP, LAM, VAR, SUP, DUP, DP0, DP1, ERA, NUM, OP2, MAT, REF, CTR, etc.)
- **EXT**: label for DUP/SUP, op code for OP2, constructor ID for CTR
- **VAL**: heap location (pointer into bump-allocated arena)

HVM4 has 46 tags total. Hot tags at indices 0-7: APP, VAR, LAM, DP0, DP1, SUP, DUP, ALO.

**TinyHVM inherits this layout** with minor modification: `[SUB:1 | TAG:7 | EXT:18 | VAL:38]` — 2 fewer VAL bits (38 vs 40), 256M heap slots. Same encoding strategy, same SUB-bit substitution.

### 4.2 Reduction Engine: Enter/Apply Trampoline

HVM4's `wnf()` is a stack-based two-phase loop with **no C recursion** on the hot path:

```
ENTER:  Walk head position, pushing eliminators (APP, DP0, DP1, OP2, MAT) as stack frames
APPLY:  Pop frames, dispatch interaction based on WHNF result in hand
```

All state lives on an explicit `WNF_STACK`. When APP meets LAM, beta-reduce inline. When DP0 meets SUP-same-label, annihilate. When OP2-NUM meets NUM, compute.

**TinyHVM's `thvm_reduce` is a direct port.** Enter phase descends into TAG_TOP arg0. Apply phase dispatches tensor ops when both args are realized. Difference: HVM4 reduces to WNF (weak head normal form), TinyHVM reduces TAG_TOP nodes to TAG_TEN (realized tensors).

### 4.3 Memory Model

- **Bump-allocated heap**: `heap_pos++`, no free, no GC
- **Per-thread heap banks**: lock-free parallel allocation (HVM2/3 multi-threaded)
- **SUB-bit substitution**: write new value in-place at variable location, mark bit 63
- **ALO nodes**: bridge static BOOK definitions to dynamic heap (lazy unfolding)
- **REF nodes**: provide laziness in strict setup, enable tail recursion in constant space

**What TinyHVM inherits:**
- Bump allocation (`heap_pos++`), no free
- SUB-bit substitution for variable binding
- `reduce_memo[]` indexed by heap loc (TinyHVM addition — HVM4 doesn't memoize WNF, we do because tensor ops are expensive)

### 4.4 Node Types

| Node | Ports | Role |
|------|-------|------|
| LAM | 1 principal + 1 aux (body) | Lambda abstraction |
| APP | 1 principal + 1 aux (arg) | Function application |
| SUP | 1 principal + 2 aux | Superposition (two values, one location) |
| DUP/DP0/DP1 | 1 principal + 2 aux | Duplication (one value, two locations) |
| ERA | 1 principal | Eraser (garbage collection) |
| NUM | 0 aux | Unboxed numeric (u24/i24/f24 in HVM2-3, u32 in HVM4) |
| OP2 | 1 principal + 1 aux | Binary operation on numerics |
| MAT/SWI | 1 principal + cases | Pattern matching / switch |
| REF | 0 aux | Reference to top-level definition (BOOK entry) |
| CTR (C00-C16) | variable | Algebraic data type constructors |

### 4.5 Lazy Cloning

HVM's signature innovation. When an object needs multiple references:
- A DUP node is placed on the wire
- Cloning happens **layer-by-layer, on demand** (lazy)
- Works for data AND lambdas (GHC can share thunks but not computations inside lambdas)
- Cost: O(size of actually-used portion), not O(total size)

This enables exponential speedups for higher-order programs where a lambda's body is large but only a small part is needed on each use.

### 4.6 Superposition-Based Search

Victor Taelin's key insight: SUP nodes enable a novel approach to program search.

```
-- Try two functions simultaneously
result = &{f(x), g(x)}   -- superposition: both computed in parallel
```

This is the foundation of:
- **SupGen**: program synthesizer (1000x faster than prior tools)
- **Architecture search**: `&{relu, gelu}` — try both activations, collapse later
- **Optimal looping**: 20x speedup over GHC on certain problems via SUP-based enumeration

The idea: represent the entire search space as a superposition, let interaction rules explore it in parallel. When a solution is found, collapse the SUP to extract it.

> For deep analysis of SupGen/NeoGen and how superposition-based search could
> replace BEAM search for kernel optimization, see **[supgen_kernel_search.md](supgen_kernel_search.md)**.

---

## 5. GPU Architecture

### 5.1 CUDA Backend (HVM2)

- Requires CUDA 12.x, NVIDIA GPUs only
- Each GPU thread processes one active pair (interaction)
- Near-ideal parallel speedup demonstrated:
  - M3 Max (1 thread): 400 MIPS
  - M3 Max (16 threads): 5,200 MIPS
  - RTX 4090 (32,768 threads): 74,000 MIPS
- Global memory implementation: 45 billion interactions per second
- 100x speedup vs V8 JavaScript on test programs

### 5.2 GPU Challenges

- **Irregular graph structure**: interaction nets have unpredictable memory access patterns — the antithesis of GPU-friendly coalesced access
- **Warp divergence**: different threads execute different interaction rules → branch divergence
- **Pointer chasing**: graph nodes scattered in heap memory, no spatial locality
- **Memory overhead**: each node is 2x64 bits minimum, vs dense tensor layouts

### 5.3 Compilation (HVM3/4)

HVM3 introduced compilation to native code for performance-critical fragments:
- Identifies compilable code patterns (no SUP/DUP → straight functional code)
- Generates C or CUDA, compiles with system compiler
- 10-100x speedup over interpreted HVM
- HVM4 can compile **even functions with superpositions** to zero-overhead machine code

---

## 6. Performance Assessment

### 6.1 Benchmarks

**vs GHC (Haskell):**
- Single-threaded: HVM typically 3x slower (GHC has decades of micro-optimization)
- 8 threads: HVM sorts large lists 2.5x faster (Radix Sort)
- Prime factorization with algebraic datatypes: HVM 20x faster (optimal looping via SUP)
- Tree summing: HVM 8-core (6.4s) vs GHC (19.2s)

**Scaling:**
- Near-ideal parallel speedup: ~12x on 16 CPU cores, ~20,000x on 32K GPU cores
- Interaction count is work-efficient (optimal reduction)

**Practical limitations (2024-2025):**
- 11x11 matrix multiply: Bend multi-core 20 min vs Go single-core 22 sec
- Single-core performance not competitive with mature compilers
- Bend's code generation is immature ("embarrassingly bad" per Taelin)
- Irregular memory access reduces GPU efficiency for real workloads

### 6.2 When HVM Wins

HVM's advantage is asymptotic, not constant-factor:
- Higher-order programs with shared lambdas (exponential speedup)
- Inherently parallel tree/graph algorithms
- Programs where lazy cloning avoids exponential blowup
- Search/enumeration via superpositions

### 6.3 When HVM Loses

- Dense linear algebra (matrix multiply, convolution) — 1000x+ overhead vs GPU kernels
- Programs with mostly first-order code and simple data
- Workloads needing cache-friendly sequential access
- Anything requiring f32/f64 precision (HVM2-3 limited to f24)

---

## 7. Bend Programming Language

Released May 2024 after "almost 10 years of hard work" (Taelin).

### 7.1 Key Features
- **Two syntax modes**: "Imp" (Python-like, default) and "Fun" (ML/Haskell-like), freely mixable
- Automatic parallelization — no threads, locks, mutexes, atomics
- Fast object allocations, full closure support, unrestricted recursion
- Pattern matching, algebraic data types
- Three numeric types: u24, i24, f24
- Compiles to HVM2 (current) / HVM3 (target)

### 7.2 Execution Modes
- `bend run` — interpret via Rust (sequential)
- `bend run-c` — interpret via C (parallel)
- `bend run-cu` — interpret via CUDA (massively parallel)
- `bend gen-c` — compile to standalone C
- `bend gen-cu` — compile to standalone CUDA

### 7.3 Current State
- 19.2K GitHub stars, Apache-2.0 license
- Missing features: no print command, limited I/O, no standard library
- Linux and macOS only (Windows via WSL2)
- Performance: CUDA interpreter 11,803 MIPS (181x over sequential)
- Active development through 2025-2026

### 7.4 Bend2 (Upcoming)
- Pythonic syntax + Haskell-inspired type system
- First-class dependent types and theorem proving
- AI-driven generation, verification, synthesis (addresses "AI Doom Loop" of accumulating bugs)
- LLM integration planned
- 99% open-source, metered generative features via paid API
- Raising $4M at $60M valuation via Wefunder

---

## 8. Kind Proof Assistant

- Dependently-typed proof language built on HVM
- Native type system running directly on interaction nets
- Potentially orders of magnitude faster than Lean (claim)
- HVM4 enables Kind as fast parallel proof verifier
- 3.7K GitHub stars
- Planned for formal release during HOC Series A

---

## 9. Higher Order Company

### 9.1 Company Profile
- **Founded**: 2023 by Victor Taelin
- **HQ**: Delaware, USA (team in Rio de Janeiro, Brazil)
- **Employees**: ~11
- **Industry**: Software Development / Symbolic AI

### 9.2 Funding
- **2023 Seed**: $4M — hired team of ~10 engineers, ~3 year runway
- **2025-26 Series**: $4M target at $60M valuation via Wefunder
- Invested $300K in 256 Apple M4 chips for symbolic AI research
- ~$1M remaining from seed as of HOC thesis (late 2024)

### 9.3 Thesis & Direction

From Taelin's HOC thesis (saved in `gists/hoc_thesis.md`):

> "Our ambitious goal is to create a purely symbolic AI architecture that replaces
> computationally heavy operations — like matrix multiplications and gradient
> descent — with efficient symbolic alternatives."

Key products:
1. **Bend/Bend2**: High-level parallel language (primary commercial product)
2. **NeoGen**: Proprietary symbolic synthesizer, up to 10,000x faster than prior tools
3. **SupGen**: HVM-powered program synthesizer, "Symbolic Transformer" direction
4. **Kind**: Proof assistant (future product, Series A timeline)

Research direction: symbolic AI via optimal evaluation. Believes interaction nets
can provide the "Merge Sort" of logic programming — an asymptotic breakthrough for
automated reasoning.

### 9.4 Victor Taelin

- Brazilian software developer and entrepreneur
- Education: Federal University of Rio de Janeiro
- 10+ years in functional programming and runtime systems
- Active on X (@VictorTaelin) with technical updates on HVM development
- 2025: heavily using AI-assisted development (Claude Code, GPT-5-high) for HVM optimization
- "Progress has never been so intense" — reports accelerated development velocity with AI tools

---

## 10. Academic Foundation

### Key Papers

| Paper | Authors | Year | Relevance |
|-------|---------|------|-----------|
| Interaction Nets | Lafont | 1990 | Graphical computation model from linear logic |
| Interaction Combinators | Lafont | 1997 | Universal system: 3 symbols, 6 rules |
| Symmetric Interaction Combinators | Mazza | 2007 | HVM2's variant (SIC) |
| An Algorithm for Optimal Lambda Calculus Reduction | Lamping | 1988 | Shared reduction inside lambdas |
| HVM2: A Parallel Evaluator for Interaction Combinators | Taelin et al. | 2024 | ICFP FProPer 2024 |
| Differential Linear Logic | Ehrhard-Regnier | 2003 | Theoretic foundation for IC + autograd |
| Differential Interaction Nets | Ehrhard-Regnier | 2006 | Differentiation as graph operation |

### Key Links

- [HVM4 source (C)](https://github.com/HigherOrderCO/HVM) — latest, TinyHVM's reference
- [HVM2 source (Rust/CUDA)](https://github.com/HigherOrderCO/HVM2) — 11.2K stars
- [HVM3 source (Haskell)](https://github.com/HigherOrderCO/HVM3)
- [Bend source](https://github.com/HigherOrderCO/Bend) — 19.2K stars
- [Kind source](https://github.com/HigherOrderCO/Kind) — 3.7K stars
- [HVM2 paper](https://paper.higherorderco.com/)
- [Lafont 1997](https://www.semanticscholar.org/paper/Interaction-Combinators-Lafont/6cfe09aa6e5da6ce98077b7a048cb1badd78cc76)
- [Interaction Calculus spec (Taelin gist)](https://gist.github.com/VictorTaelin/903f20e03d5c0bb0b1c0ee27c7e45801)
- [HVM3 Optimal Polarized Atomic Linker (gist)](https://gist.github.com/VictorTaelin/2aba162f2b04478dc53e5615f482db7b)
- [HOC complete historical overview (gist)](https://gist.github.com/VictorTaelin/77fd5a2a8a4a07e1da6157ebca3c7cf1)
- [Higher Order Company website](https://higherorderco.com/)
- [Wefunder campaign](https://wefunder.com/higher.order.co)

---

## 11. HVM & Tensor Computation

### 11.1 The Fundamental Tension

HVM operates on **individual terms** (scalars, lambdas, trees). ML operates on
**dense tensors** (multi-dimensional arrays of floats). A 1024x1024 matmul through
HVM's graph would require ~2 billion interactions — versus one GPU kernel dispatch.

This is not a fixable inefficiency — it's a fundamental mismatch between MIMD graph
rewriting (many different operations on scattered data) and SIMT tensor computation
(one operation on contiguous data).

### 11.2 TinyHVM's Hybrid Solution

TinyHVM resolves this by using HVM at two different levels:

```
IC Layer:  Graph topology, scheduling, autograd, fusion decisions
           └── DUP/ERA/SUP for CSE, DCE, kernel search
GPU Layer: Dense tensor arithmetic via Metal/MPS kernels
           └── One kernel dispatch per op, not one interaction per element
```

What the IC net manages:
- Lazy computation graph (TAG_TOP nodes = unrealized ops)
- Optimal sharing (DUP-SUP annihilation = CSE)
- Dead code elimination (ERA propagation = DCE)
- Autograd (GRAD node + provenance tracking)
- Fusion decisions (TOP-TOP chain detection)

What GPU kernels handle:
- Matrix multiplication (MPS/Accelerate BLAS)
- Elementwise ops (Metal compute shaders)
- Reductions (parallel reduce kernels)
- Fused kernels (JIT-compiled Metal)

This is the architecture validated in `report.md` and implemented in TinyHVM's
current ~5.9K LoC codebase.

### 11.3 What TinyHVM Takes from HVM4

**Adopted:**
- 64-bit term layout (SUB:1 | TAG:7 | EXT:18 | VAL:38)
- Enter/apply trampoline reducer (`thvm_reduce` ← HVM4's `wnf`)
- SUB-bit substitution for variable binding
- APP-LAM beta reduction, DUP-SUP annihilation/commutation
- ERA propagation for dead code elimination
- Bump-allocated heap, no GC
- TAG_REF + definition table (HVM4's BOOK)

**Adapted:**
- TAG_TOP (tensor ops) — not in HVM4, our extension for GPU dispatch
- TAG_TEN (tensor reference) — wraps GPU buffer ID, dtype, View
- reduce_memo[] — memoize expensive tensor reductions (HVM4 doesn't memoize)
- f32 numerics via bitcast (HVM4 has u32 only)
- View struct with strides/offset (from tinygrad's ShapeTracker, not HVM)

**Not adopted:**
- u24/f24 numerics (ML needs f32/f16 minimum)
- Scalar-per-interaction arithmetic (we dispatch GPU kernels)
- Multi-threaded work-stealing (not yet — single-threaded reducer currently)
- ALO nodes (we use TAG_REF directly)

---

## 12. Critiques & Limitations

### Performance
- Single-threaded 3x slower than GHC (decades less micro-optimization)
- Real-world Bend programs dramatically slower than equivalent Go/C for numerical work
- GPU efficiency limited by irregular memory access patterns and warp divergence
- Compilation (HVM3/4) narrows the gap but isn't yet widely deployed

### Maturity
- Bend just released — many missing features, no print command, limited I/O
- Windows support only via WSL2
- No established library ecosystem
- Code generation described as "embarrassingly bad" (will improve)

### Theoretical
- Bookkeeping-free reduction is not truly optimal in all cases (99% coverage)
- Quadratic complexity possible in certain pathological cases
- Lack of formal verification of the runtime itself (planned)

### ML/Tensor
- No built-in tensor operations
- Memory access patterns unsuitable for dense linear algebra
- No ML library ecosystem
- f24 precision insufficient for training
- Hybrid architecture (like TinyHVM) required for any ML application

---

## 13. Timeline & Milestones

| Date | Event |
|------|-------|
| 2022 | HVM1 released (Rust) — first practical IC evaluator |
| 2023 | Higher Order Company founded, $4M seed raised |
| May 2024 | Bend released publicly, HVM2 stable |
| May 2024 | HVM-CUDA: 45 billion IPS (global memory) |
| Aug 2024 | Symbolic AI research open-sourced |
| ICFP 2024 | HVM2 paper presented at FProPer workshop |
| Late 2024 | HOC thesis: pivot to Symbolic AI market |
| 2025 | HVM3 active development (compilation, lazy eval) |
| 2025 | HVM4 development: C rewrite, native types, SupGen built-in |
| 2025 | Bend2 announced, $4M funding round at $60M valuation |
| 2025 | AI-assisted development: Claude Code optimizes HVM3 to 328 MIPS (+51%) |
| 2026 (target) | Symbolic Transformer validation on ARC Prize challenges |

---

## 14. Cross-Reference to Existing Resources

| Resource | HVM Content |
|----------|-------------|
| `engineering_reference.md` §1 | HVM4 source study: term layout, reduction engine, interaction rules, memory model, FFI, what to adopt/avoid |
| `report.md` | Full feasibility report: tinygrad↔HVM alignment, hybrid architecture proposal, implementation roadmap |
| `ic_arch.md` | TinyHVM architecture derived from HVM4: term layout, reducer, autograd design |
| `ic_optimization.md` | IC rewriting as optimization: ERA=DCE, DUP-SUP=CSE, TOP-TOP=Fusion |
| `ic_autograd.md` | Theoretical foundation: Linear Logic → Differential Interaction Nets → IC-native autograd |
| `ic_pure_inet.md` | Pure inet training model: TAG_REF, UOP_ASSIGN, phased roadmap |
| `ic_fusion.md` | Fusion via intermediate nodes: FuseState, UOP_FUSING |
| `tinyhvm_spec.md` | Full spec: term grammar, interaction rules, tensor-specific extensions |
| `taelin_gists_index.md` | Index of 20 Taelin gists + HigherOrderCO repo inventory |
| `gists/ic_spec.md` | Full Interaction Calculus specification (grammar, WHNF rules) |
| `gists/hoc_thesis.md` | HOC business thesis: Symbolic AI market, SupGen, M4 cluster |
