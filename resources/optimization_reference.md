# Optimization Reference: Kernel Performance, Autotuning & Systematic Fast Code

Reference for TinyHVM development — aggregating Karpathy's autoresearch methodology,
autotuning theory, and performance engineering principles.

For tinygrad-specific internals (kernel pipeline, beam search, ShapeTracker, device
backends, architecture, releases), see **tinygrad_reference.md**.

---

## 1. Karpathy's Autoresearch Methodology

### 1.1 Core Concept

Give an AI agent a small but real training setup. Let it experiment autonomously.
It modifies code, trains for a fixed time, checks if the result improved, keeps
or discards, and repeats.

**Result**: 700 experiments over 2 days on already-optimized code → 20 genuine
improvements → 11% speedup (2.02h → 1.80h to GPT-2 quality).

### 1.2 Architecture

Three files, strict separation:

| File | Role | Who edits |
|------|------|-----------|
| `prepare.py` | Data prep, tokenizer, evaluation metric | Nobody (immutable) |
| `train.py` | Model, optimizer, training loop (~630 lines) | Agent only |
| `program.md` | Instructions, constraints, stopping criteria | Human only |

**Why ~630 lines**: entire codebase fits in LLM context window. The agent can
maintain holistic understanding. Minimizes code generation errors.

### 1.3 The Loop

```
while not done:
    1. Modify train.py (form hypothesis, edit code)
    2. Commit changes (git)
    3. Train for exactly 5 minutes (fixed wall-clock budget)
    4. Extract val_bpb from run.log
    5. Log to results.tsv: commit, val_bpb, memory_gb, status, description
    6. If improved: keep. If not: revert.
```

**~12 experiments/hour. ~100 experiments overnight.**

### 1.4 Scoring: val_bpb

Validation bits-per-byte. Lower is better. Vocabulary-size-independent so
architectural changes (different tokenizer configs, model sizes) are fairly
comparable. The metric is defined in the immutable `prepare.py`.

### 1.5 Constraints & Simplicity Criterion

- Only `train.py` is editable
- No new package dependencies
- VRAM is a soft constraint (some increase OK for meaningful gains)
- **"All else being equal, simpler is better"**
- A small improvement that adds ugly complexity → not worth it
- Removing something and getting equal or better results → great outcome
- Agent operates autonomously — no pausing for confirmation

### 1.6 What Makes This Generalizable

The pattern underneath has nothing to do with GPUs or neural networks. It works
on **anything you can score**:

1. **Fixed evaluation budget** — prevents gaming optimization time
2. **Single scalar metric** — unambiguous comparison
3. **Immutable evaluation** — agent can't cheat the metric
4. **Single modifiable file** — bounded blast radius
5. **Git-based rollback** — every experiment is reversible
6. **Simplicity pressure** — prevents complexity creep

### 1.7 Application to TinyHVM

Direct analogue for kernel optimization:
- `train.py` → kernel source or codegen template
- `val_bpb` → kernel wall-clock time or GFLOPS
- `prepare.py` → benchmark harness (immutable)
- `program.md` → optimization constraints (memory budget, correctness checks)
- Fixed budget → fixed number of iterations or fixed problem size

---

## 2. Karpathy's Recipe for Training Neural Networks

Six-stage systematic methodology. The core insight: **"neural net training fails
silently"** — wrong configs produce working but suboptimal models, not errors.

### Stage 1: Become One with the Data
Examine thousands of examples before writing code. Understand patterns,
duplicates, corrupted data, class imbalances, noise levels. This guides
every subsequent decision.

### Stage 2: End-to-End Pipeline with Dumb Baselines
Build complete train/eval skeleton with intentionally simple models:
- Fix random seeds for reproducibility
- Disable augmentation and fancy features
- Verify loss at initialization matches theory
- Test input-independent baselines (always predict mean)
- **Overfit a single batch** to verify the training loop works
- Visualize data immediately before the network sees it

### Stage 3: Overfit
Build a model large enough to fit training data. Proves architecture can learn.
**"Don't be a hero"** — copy successful designs from papers, don't invent exotic
architectures.

### Stage 4: Regularize
Improve validation by adding constraints in order of effectiveness:
1. More real data (most effective)
2. Data augmentation
3. Pretrained networks
4. Reduce model size / input dimensionality
5. Dropout, weight decay, early stopping

### Stage 5: Tune
Random search over hyperparameters, not grid search. Neural networks have
uneven sensitivity — random search covers the important dimensions better.

### Stage 6: Squeeze Out Final Performance
Ensembles. Let networks train longer than you think necessary.

### Meta-Principle
**Patience and attention to detail** correlate with success. Build complexity
incrementally. Verify each change produces expected improvements through
explicit hypotheses. **Never change two things at once.**

---

## 3. Performance Engineering Fundamentals

### 3.1 The Roofline Model

Every computation is bounded by one of two limits:
1. **Compute ceiling**: peak FLOPS of the hardware
2. **Memory bandwidth ceiling**: bytes/second from memory hierarchy

The crossover point is **arithmetic intensity** = FLOPS / bytes moved.

```
                    compute-bound
                   /
Performance       /
(FLOPS)     ____/
           /
          / bandwidth-bound
         /
        ──────────────────────
        Arithmetic Intensity (FLOP/byte)
```

**Apple Silicon crossover**: ~19-22 FLOP/byte on base chips. Most ML inference
is bandwidth-bound (below the ridge). Training with large batch GEMM can be
compute-bound (above the ridge).

**How to use it:**
1. Profile kernel: measure runtime, total FLOPs, total bytes moved
2. Calculate arithmetic intensity = FLOPs / bytes
3. Plot on roofline → immediately see which ceiling you're hitting
4. If bandwidth-bound: reduce data movement (fusion, caching, compression)
5. If compute-bound: use tensor cores, increase occupancy, reduce wasted ops
6. Measure again → verify optimization moved the point in the right direction

### 3.2 Measure, Don't Guess

The single most important principle in performance engineering.

**What to measure:**
- Wall-clock time (the only metric users care about)
- Hardware counters: cache misses, branch mispredictions, occupancy
- Bandwidth utilization: actual vs theoretical peak
- FLOPS utilization: actual vs theoretical peak

**How to measure on Metal:**
- Xcode Metal Debugger: per-kernel GPU counters
- Command buffer timestamps: nanosecond GPU timing
- Instruments: system-wide CPU/GPU trace
- `DEBUG=2` in tinygrad: per-kernel timing with GFLOPS/GB/s

**Common measurement mistakes:**
- Measuring cold code (include warmup runs)
- Not accounting for driver/compilation overhead
- Measuring too-small problems (overhead dominates)
- Not isolating the thing being measured

### 3.3 Bentley Rules (MIT 6.172)

Practical optimization checklist from Jon Bentley, taught in MIT's Performance
Engineering course:

**Data structure optimizations:**
- Packing: reduce memory footprint (TinyHVM's 64-bit term encoding)
- Augmenting: store derived data to avoid recomputation
- Precomputation: compute results at build time, not runtime
- Caching: remember recently computed results (reduce_memo)
- Lazy evaluation: delay computation until result is needed (tinygrad's entire model)
- Sparsity: skip zero elements

**Loop optimizations:**
- Hoisting: move invariant computation out of loops
- Sentinels: eliminate bounds checks in search loops
- Loop unrolling: reduce branch overhead
- Loop fusion: combine adjacent loops over same data
- Eliminating wasted iterations: early exit, tighter bounds

**Logic optimizations:**
- Constant folding and propagation
- Common subexpression elimination
- Short-circuit evaluation
- Algebraic identity reduction (x+0, x*1, etc.)

### 3.4 Memory Hierarchy Exploitation

The dominant factor in modern performance:

| Level | Apple M1 | Latency | Bandwidth |
|-------|----------|---------|-----------|
| Registers | ~128 per ALU | 0 cycles | ∞ |
| L1 cache | 128 KB (data) | ~3 cycles | ~4 TB/s |
| L2 cache | 12 MB (shared) | ~12 cycles | ~1 TB/s |
| DRAM | 8-128 GB | ~100 cycles | 68-800 GB/s |

**Rules:**
- Sequential access >> random access (prefetcher works)
- Smaller working set >> larger working set (fits in cache)
- Structure-of-arrays >> array-of-structures (vectorization)
- Tiling: process data in cache-sized blocks
- Prefetching: overlap computation with memory latency

---

## 4. Autotuning Theory & Practice

### 4.1 The No Free Lunch Theorem

No single optimization strategy universally works best across all targets and
workloads. The search space is combinatorial: structural transformations are
tightly coupled with hardware resource constraints.

**Implication**: you must search (or be hardware-specific). General-purpose
optimizers will always leave performance on the table.

### 4.2 Historical Systems

| System | Year | Approach | Domain |
|--------|------|----------|--------|
| ATLAS | 1998 | Empirical search over tile sizes | Dense linear algebra |
| FFTW | 1999 | Self-tuning via "codelets" + planner | FFT |
| Halide | 2012 | Separate algorithm from schedule | Image processing |
| AutoTVM | 2018 | ML-guided template search | Tensor programs |
| Ansor | 2020 | Sketch-guided hierarchical search | Tensor programs |
| Triton | 2019+ | Block-level tiled programming model | NN kernels |
| tinygrad | 2020+ | Beam search over discrete actions | General ML |

### 4.3 TVM / Ansor Approach

**AutoTVM**: define a search template with knobs (tile sizes, unroll factors,
vectorization widths). Use ML cost model (XGBoost) to predict performance.
Transfer learning across hardware targets.

**Ansor**: generate sketches (high-level structures) automatically, then
fine-tune with evolutionary search + learned cost model. Found programs
with up to 3.8× speedup over hand-tuned libraries.

Key insight: **separate the structure from the parameters**. Enumerate a
small number of structural choices, then search within each.

### 4.4 Triton's Approach

Block-level programming: operations on parametric tiles (power-of-2 sized).
Abstracts away:
- Memory coalescing
- Shared memory synchronization
- Tensor core scheduling
- Thread divergence

The programmer thinks in blocks, the compiler handles thread mapping.
Result: 2-10× easier to write vs CUDA, within 80-95% of expert CUDA perf.

Recent advances (2025): ML-Triton adds multi-level tiling hints (workgroup →
warp → intrinsic), enabling finer hardware matching.

### 4.5 Equality Saturation

Alternative to phase-ordered optimization. Instead of applying rewrites
sequentially (where order matters), explore **all rewrites simultaneously**:

1. Build an e-graph representing all equivalent programs
2. Apply rewrite rules to grow the e-graph
3. Extract the optimal program via cost function

**Results**: up to 16% speedup on tensor graphs, 48× less search time than
sequential approaches.

**Tools**: egg/egglog (Rust/Datalog), DialEgg (MLIR integration).

**Relevance to TinyHVM**: pattern-matching rewrite system is already close to
this — could potentially adopt e-graph representation for optimization phase.

### 4.6 Profile-Guided Optimization (PGO)

Feed runtime profiling data back into the compiler:
- **Instrumentation PGO**: 5-10% speedup (integer), up to 30% in hot loops
- **Hardware sampling PGO**: 3-7% speedup, ~80-93% of instrumentation gains
- **Key optimizations**: hot-path inlining, block reordering, branch prediction hints

Requires representative profiling workloads. Unrepresentative profiles can
make things worse.

---

## 5. Loop & Kernel Transformations Cookbook

### 5.1 Tiling (Blocking)

Split loops into blocks that fit in cache:

```c
// Before: stride through entire matrix
for (int i = 0; i < N; i++)
  for (int j = 0; j < N; j++)
    C[i][j] += A[i][k] * B[k][j];

// After: process TILE×TILE blocks
for (int ii = 0; ii < N; ii += TILE)
  for (int jj = 0; jj < N; jj += TILE)
    for (int kk = 0; kk < N; kk += TILE)
      for (int i = ii; i < ii+TILE; i++)
        for (int j = jj; j < jj+TILE; j++)
          for (int k = kk; k < kk+TILE; k++)
            C[i][j] += A[i][k] * B[k][j];
```

Optimal tile size depends on cache size. A tile of 2048 enables ~34% of
computations to execute within one fused tile.

### 5.2 Kernel Fusion

Combine multiple passes into one kernel to eliminate intermediate memory traffic:

```
// Before: 3 kernels, 2 intermediate buffers
kernel1: Y = relu(X)          # read X, write Y
kernel2: Z = Y * scale        # read Y, write Z
kernel3: out = Z + bias       # read Z, write out

// After: 1 kernel, 0 intermediate buffers
fused:   out = relu(X) * scale + bias   # read X, write out
```

**When to fuse**: memory-bound producer-consumer chains, small kernels,
measurable launch overhead. Practical speedups: 2-6× common, FlashAttention
achieves 20-50% over unfused.

**When NOT to fuse**: compute-bound kernels, fusion increases register pressure
beyond occupancy threshold, kernels have incompatible parallelism.

### 5.3 Vectorization

Use SIMD/simdgroup operations instead of scalar loops:

```metal
// Scalar: 1 op per cycle
float sum = 0;
for (int i = 0; i < 4; i++) sum += a[i] * b[i];

// Vectorized: 4 ops per cycle
float4 va = *(device float4*)a;
float4 vb = *(device float4*)b;
float sum = dot(va, vb);
```

Metal vector widths: float4 (preferred), half8. On Apple Silicon, SIMD shuffle
is 256 B/cycle — prefer `simd_shuffle` over threadgroup memory.

### 5.4 Loop Unrolling

Trade code size for fewer branches and better ILP:

```c
// Before: branch every iteration
for (int i = 0; i < N; i++) acc += data[i];

// After: 4× fewer branches, 4-wide ILP
for (int i = 0; i < N; i += 4) {
  acc0 += data[i];   acc1 += data[i+1];
  acc2 += data[i+2]; acc3 += data[i+3];
}
acc = acc0 + acc1 + acc2 + acc3;
```

Tinygrad's beam search explores unroll factors of `[0, 4, 7]` (0 = full unroll).

### 5.5 Data Layout Transformation

Row-major vs column-major matters enormously for access patterns:

```
// If iterating columns of A:
// Row-major A[i][j]:    stride = N (cache-hostile)
// Column-major A[j][i]: stride = 1 (cache-friendly)
```

For GEMM: transpose one operand so both inner loops have stride-1 access.
tinygrad's SWAP optimization permutes axes for this reason.

### 5.6 Specialization

Generate different kernel variants for different cases:

```metal
// Generic: runtime branching
if (stride == 1) { /* contiguous path */ }
else { /* strided path */ }

// Specialized: compile-time constants (Metal function constants)
constant bool is_contiguous [[function_constant(0)]];
// Compiler eliminates dead branch entirely
```

TinyHVM already does this: JIT-specialized kernels per leaf shape/stride
configuration. tinygrad's `BEAM` search effectively finds per-kernel
specialization parameters.

---

## 6. What Consistently Works

Distilled from all sources — principles that reliably produce speedups:

### 6.1 Universally True

1. **Measure before optimizing** — profile first, always
2. **Reduce data movement** — memory bandwidth is the bottleneck 90% of the time
3. **Fuse operations** — eliminate intermediate buffers
4. **Use hardware intrinsics** — tensor cores, SIMD, simdgroup_matrix
5. **Specialize hot paths** — compile-time constants eliminate branches
6. **Cache results** — avoid recomputation (reduce_memo, compiler caches)
7. **Minimize allocations** — reuse buffers, pool memory

### 6.2 Almost Always True

8. **Tile for cache** — block loops to fit working set in L1/L2
9. **Sequential access patterns** — stride-1 access enables prefetcher
10. **Batch work** — amortize dispatch overhead across many operations
11. **Prefer simpler algorithms** — simpler code often runs faster (branch prediction,
    compiler optimization, fewer cache misses)
12. **Automate the search** — humans are bad at predicting what's fast

### 6.3 Context-Dependent

13. **Increase occupancy** — helps if latency-bound, hurts if register-pressure-bound
14. **Use shared memory** — helps NVIDIA (explicit L1), may not help Apple Silicon
    (hardware-managed cache, SIMD shuffle faster)
15. **Unroll aggressively** — helps small loops, code bloat hurts large ones
16. **Quantize precision** — free speedup on Apple (fp16 = fp32 throughput),
    meaningful on NVIDIA (2× fp16 vs fp32)

### 6.4 Common Pitfalls

- **Optimizing the wrong thing** — the slow kernel isn't always where you think
- **Premature optimization** — get correct first, then profile, then optimize
- **Over-engineering** — a 2% gain requiring 500 lines of complexity isn't worth it (Karpathy's llm.c philosophy)
- **Benchmark gaming** — optimizing for benchmarks that don't reflect real workloads
- **Ignoring Amdahl's Law** — 10× speedup on 1% of runtime = 0.09% total improvement
- **Not re-measuring after changes** — optimizations can interact unexpectedly

---

## 7. Applied Strategy for TinyHVM

### 7.1 Current Optimization Stack

TinyHVM's JIT already does:
- Per-leaf stride/shape specialization (inline index expressions)
- Fast contiguous vs slow strided kernel dispatch
- GPU buffer reuse via pool
- Kernel fusion for elementwise chains

### 7.2 Next-Level Opportunities

**From tinygrad** (see tinygrad_reference.md §6, §5 for details)**:**
1. **Beam search over kernel configs** — axis types, tile sizes, unroll factors
2. **ShapeTracker-like lazy views** — avoid materializing reshape/permute/expand
3. **Three-level caching** — method cache + disk cache + beam cache
4. **Pattern-matching rewrites** — declarative optimization rules
5. **Tensor core mapping** — simdgroup_matrix for GEMM/conv

**From autoresearch — adopt:**
1. **Automated A/B testing** — kernel variant A vs B, measure, keep winner
2. **Fixed evaluation budget** — always benchmark same problem size/iterations
3. **Immutable benchmark harness** — can't accidentally "optimize" the test
4. **Simplicity pressure** — reject complexity that doesn't pay for itself
5. **Results logging** — TSV of every experiment for analysis

**From performance engineering — adopt:**
1. **Roofline analysis** — classify every kernel as compute/memory bound
2. **Amdahl-aware prioritization** — optimize kernels by % of total runtime
3. **Hardware counter profiling** — cache misses, occupancy, bandwidth util
4. **Regression testing** — detect performance regressions in CI

### 7.3 Concrete Next Steps

**Immediate (use what we have):**
- Add `DEBUG=2`-style per-kernel timing to TinyHVM
- Compute arithmetic intensity for each kernel → roofline classification
- Identify top-3 kernels by wall-clock time → focus optimization there

**Short-term (weeks):**
- Implement beam search over tile sizes for GEMM/conv kernels
- Add Metal simdgroup_matrix for matmul (8×8×8 on Apple Silicon)
- Implement kernel fusion for elementwise → reduce chains

**Medium-term (months):**
- Pattern-matching rewrite system for UOp optimization
- ShapeTracker-style lazy views to eliminate reshape materialization
- Disk-cached compilation with content-addressed hashing
- Automated benchmark harness with regression detection

---

## 8. Key References

### Foundational
- MIT 6.172 Performance Engineering (Leiserson) — ocw.mit.edu
- Agner Fog optimization guides — agner.org/optimize/
- Roofline Model (Williams, Waterman, Patterson, 2009)
- Bentley, "Writing Efficient Programs" (1982)

### Autotuning & Compilers
- Ansor: Generating High-Performance Tensor Programs (Zheng et al., 2020)
- Learning to Optimize Tensor Programs — AutoTVM (Chen et al., 2018)
- Triton: Tiled Neural Network Computations (Tillet, Kung, Cox, 2019)
- egg: Fast and Extensible Equality Saturation (Willsey et al., 2021)
- EQUALITY SATURATION FOR TENSOR GRAPH SUPEROPTIMIZATION (Yang et al., MLSys 2021)
- DialEgg: Dialect-Agnostic MLIR Optimizer using Equality Saturation (CGO 2025)

### Practice
- Karpathy, "A Recipe for Training Neural Networks" (2019) — karpathy.github.io
- Karpathy, autoresearch — github.com/karpathy/autoresearch
- Karpathy, llm.c — github.com/karpathy/llm.c
- tinygrad source — github.com/tinygrad/tinygrad
- "From 11% to 88% Peak Bandwidth: Custom Triton Kernels" (Mitra, 2025)
