# SupGen & Superposition-Based Kernel Optimization

> Can optimal-sharing program search replace BEAM search for kernel tuning?

---

## 1. SupGen: How It Works

SupGen (and its proprietary evolution **NeoGen**) is Victor Taelin's program
synthesizer built on HVM's superposition nodes. The core idea:

**Instead of evaluating N candidate programs one at a time, construct a single
superposed term containing ALL candidates and evaluate them simultaneously.
Optimal sharing ensures common subcomputations across candidates are computed
once.**

### 1.1 The Mechanism

The Interaction Calculus extends lambda calculus with two primitives:

```
SUP: &L{a, b}      -- superposition: "both a and b, labeled L"
DUP: !&L{x,y}=t; k -- duplication: "split t into x and y, continue with k"
```

When a function is applied to a superposition, the **APP-SUP** rule fires:
```
(f &L{a, b})  -->  !&L{f0, f1} = f;  &L{(f0 a), (f1 b)}
```
The function `f` is duplicated lazily (via DUP), and both branches evaluate
in parallel. If f is identical in both branches, the DUP is free.

When DUP meets SUP with the **same label** (annihilation):
```
!&L{x,y} = &L{a,b}  -->  [x<-a, y<-b]   -- zero cost, just substitution
```

When DUP meets SUP with **different labels** (commutation):
```
!&L{x,y} = &M{a,b}  -->  nested superpositions  -- creates sharing structure
```

When a branch fails, ERA (erasure) propagates without computing the erased
branch. This is automatic pruning.

### 1.2 Why It's Fast: Optimal Sharing

Two candidate programs that share a common prefix evaluate that prefix **once**.
This is Levy-optimal beta reduction -- no redundant beta step is ever performed.

The speedup is not parallelism (it works on a single core). It is **algorithmic
sharing** of computation across the search tree. The same mechanism that makes
HVM's lazy cloning exponentially faster for higher-order programs makes SupGen
exponentially faster for program search.

### 1.3 Benchmarks

| Problem | Search Space | Brute Force | SupGen | Speedup |
|---------|-------------|-------------|--------|---------|
| ADD-CARRY (16 unknown bits) | 2^16 = 65K | 262M interactions | 36K interactions | **7,277x** |
| XOR-XNOR discovery | — | 2.8s (Haskell) | 0.0085s (HVM) | **330x** |
| Lambda equation solving | — | 0.992s (Haskell) | 0.0011s (HVM) | **862x** |
| SAT 16-var | 2^16 | minutes (Rust) | ~1s (HVM) | — |
| Peano Sort (position 5.7M) | huge | 5m17s enumerate | 2s (NeoGen) | **159x** |

**Cost per candidate**: sub-1 interaction in the ADD-CARRY benchmark. The brute
force cost is ~4000 interactions per candidate; with SUPs, 65K candidates cost
36K total interactions.

Source: [ADD-CARRY gist](https://gist.github.com/VictorTaelin/d5c318348aaee7033eb3d18b0b0ace34),
[Accelerating DPS gist](https://gist.github.com/VictorTaelin/7fe49a99ebca42e5721aa1a3bb32e278),
[Lambda enumerator gist](https://gist.github.com/VictorTaelin/7c4c69a1f07b5c668be613f1032e7d4e)

### 1.4 NeoGen: Production SupGen

NeoGen is SupGen evolved. Key advances (2025-2026):

- **Type enumeration**: Can now discover programs from input/output examples alone
  (not just type signatures). Enumerates types as first-class objects.
- **Denoising interpretation**: NeoGen = "denoising proofs with holes" -- take a
  complete proof, remove random sub-expressions, NeoGen recovers them.
- **HVM4 compilation**: SUPs now compile to zero-overhead machine code (no interpreter).
- **Integration in Bend2**: Put a "hole" anywhere, compiler fills it via NeoGen.

NeoGen found **every primitive recursive function tested** "instantly":
Equality (0.0008s), DrawLine (0.001s), Insert (0.006s).

Source: [NeoGen benchmarks](https://x.com/VictorTaelin/status/1904727018899439853),
[HVM4 compiled SUPs](https://x.com/VictorTaelin/status/1985320306001477783)

### 1.5 KolmoLang: Optimized Search Space

KolmoLang is a stack-based DSL designed to minimize program description length
(approximating Kolmogorov complexity) while maximizing expressivity for SupGen:

- 10 instructions: lambda, match, put, emit, new, erase, duplicate, return, call
- `mul2`: ~21 bits (vs ~50 bits in binary lambda calculus)
- **Argument persistence** constraint: functions cannot reorder args between recursive
  calls, reducing enumeration space while maintaining expressivity via composition.

Source: [KolmoLang gist](https://gist.github.com/VictorTaelin/41d3c07d62cbb0978924fa6d504c4bee)

### 1.6 Unordered Superpositions (Critical Refinement)

Standard ordered SUPs cause quadratic overhead for chains. Unordered SUPs
(`{x y} == {y x}`) fix this:

- Single-port storage instead of two
- Runtime reorders based on access patterns ("opportunistic swaps")
- Reduces interaction count from quadratic to **linear** for structured data

Solving `X + 12345 = 1000000` (X = superposition of all naturals):
24M interactions, linear in nat length (1M). Without this fix: quadratic.

Source: [Truly Optimal Evaluation gist](https://gist.github.com/VictorTaelin/93c327e5b4e752b744d7798687977f8a)

---

## 2. Tinygrad's BEAM Search (What SupGen Would Replace)

### 2.1 How BEAM Works

BEAM is greedy iterative search over kernel schedule actions:

```
unoptimized kernel
  -> generate ~50 valid actions (from ~193 static actions)
  -> compile + time each on real GPU (parallelized across CPU cores)
  -> keep top N (BEAM width)
  -> repeat until no improvement > 0.01us
  -> cache result to disk
```

**Actions** (OptOps): TC, UPCAST, UNROLL, LOCAL, THREAD, GROUP, GROUPTOP,
NOLOCALS, PADTO, SWAP. Each is `(op, axis, amount)`.

**Metric**: raw wall-clock kernel time, measured 3x on hardware, min taken.

**Cost**: BEAM=2 on a matmul kernel takes ~40-60 seconds. Full model (MLPerf BERT)
with BEAM=5-8 takes 10-30 minutes. Results cached to disk.

Source: `tinygrad/codegen/opt/search.py` (~400 LOC)

### 2.2 BEAM Limitations

1. **Greedy pruning**: Discards candidates that look bad locally but may be globally
   optimal. A bad tiling choice at level 1 may unlock the best vectorization at level 3.
2. **No computation sharing**: Each candidate kernel is compiled and timed independently.
3. **No backtracking**: Once pruned, a branch is gone forever.
4. **Cost model = hardware**: Requires actual GPU execution per candidate. Slow.
5. **Timing instability**: ~30% variance reported on AMD hardware, causing unreliable results.
6. **Search space coverage**: BEAM=2 evaluates maybe 100-600 candidates total from a
   space of potentially millions of valid schedules.

### 2.3 Where Tinygrad Is Heading (Hotz, Dec 2025 - Mar 2026)

From tinycorp meeting transcripts:

> "I really think by the end of next year, instead of BEAM, it's going to be
> like LLM equals 1... we could fine tune a Qwen free coder on sitting there
> with GPUs, trying stuff, and then up-weighting all the traces where it
> eventually gets high performance kernel." — Hotz, Dec 22, 2025

> "BEAM almost doesn't make sense, and you almost do have to hand code...
> we'll never be able to build something that's generically good. There's so
> many little bits of subtlety and nuance." — Hotz, Mar 24, 2026

The proposed replacement: **LLM-generated decision trees** -- per-hardware heuristics
generated by a fine-tuned model, committed as code. "A couple thousand lines, all
LLM generated, all simple, with a very narrow API."

---

## 3. SupGen vs BEAM: Can Superpositions Replace Beam Search?

### 3.1 Where SupGen Wins (Unambiguously)

**Computation sharing across candidates.** When searching over tile sizes
{16, 32, 64} x vectorization {on, off} x parallelism {block, thread}, many
candidates share identical subcomputations. BEAM evaluates each independently.
SUPs share automatically. For a schedule space of 10^6 candidates, this could
mean 10^6 interactions vs 4x10^9 independent evaluations.

**No premature pruning.** BEAM with width k discards all but k candidates at
each level. SUPs explore the full combinatorial space. A schedule that looks bad
at one tiling level may become optimal when combined with a specific vectorization.

**Exponential cost reduction.** In the ADD-CARRY benchmark: sub-1 interaction per
candidate (7277x speedup). The bigger the search space, the bigger the win.

**Natural encoding.** Kernel schedules are highly structured -- tiling choices are
often independent across dimensions. This structure is exactly what SUPs exploit.

### 3.2 The Fundamental Obstacle

**The cost function is not a pure function.**

Kernel performance depends on hardware behavior: cache effects, bank conflicts,
warp divergence, memory coalescing. The "evaluation" of a schedule requires either:
- **Hardware execution** (what BEAM does) — cannot be expressed in lambda calculus
- **Learned cost model** (what Ansor does) — a neural/tree model, also impure

SUP-based search works on **pure functional computation**. It cannot simulate a GPU.
If the cost of a schedule can only be determined by running it on hardware, then
the sharing advantage evaporates — you still need N hardware runs for N candidates.

### 3.3 The Hybrid Path (Where SUPs Actually Help)

#### Approach A: Symbolic Cost Model + SUP Search

Build a **simplified symbolic cost model** as a pure function:

```
cost(schedule) = memory_accesses(schedule) * bandwidth_cost
               + arithmetic_ops(schedule) * compute_cost
               + sync_points(schedule) * barrier_cost
```

This is a pure function over schedule parameters. Encode it in the Interaction
Calculus. Then:

```
tile    = &0{16, &1{32, 64}}    -- 3 tile sizes
unroll  = &2{1, &3{2, 4}}       -- 3 unroll factors
local   = &4{True, False}       -- 2 local options

schedule = make_schedule(tile, unroll, local)
cost     = symbolic_cost(schedule, kernel_shape)
-- 18 candidates evaluated with optimal sharing
-- collapse(min(cost)) extracts the best
```

**Limitation**: Symbolic cost models are inaccurate. Real performance depends on
hardware details no symbolic model captures. But the model can **narrow the search
space** before expensive hardware evaluation.

#### Approach B: Structure Search + Parameter Tuning

Use SUPs to search over **discrete structural choices**:
- Which axes to tile vs leave untiled
- Where to place fusion boundaries
- Which axes to parallelize (block vs thread vs warp)
- Whether to use tensor cores

Then use conventional autotuning (or BEAM with narrow width) for **continuous
parameters** (specific tile sizes, unroll factors).

The structural space is small and highly combinatorial — perfect for SUP sharing.
The parameter space is where hardware measurement matters.

#### Approach C: Constraint Filtering via SUPs

Use SUP evaluation to **quickly eliminate invalid schedules** before hardware
evaluation. Many schedule combinations violate constraints:
- Shared memory exceeds limit
- Register pressure too high
- Warp occupancy below threshold
- Data layout incompatible with tensor core

Encode these constraints as pure functions. Evaluate all candidates against all
constraints simultaneously. Only surviving candidates get hardware timing.

This is analogous to HVM's SAT solver approach — implicit constraint propagation
via SUP evaluation.

#### Approach D: Rewrite Rule Discovery

Use SupGen to **discover new optimization rewrite rules** rather than searching
over schedules directly. Given pairs of (slow kernel, fast kernel) from BEAM
results, synthesize a transformation function that maps slow → fast:

```
-- Specification: find transform T such that
-- for all kernel shapes s: time(T(kernel(s))) < time(kernel(s))
-- with T expressible as a sequence of schedule actions

T = SupGen.search(spec, action_space)
```

If successful, T becomes a permanent heuristic — no search needed at runtime.
This is the most ambitious approach and the closest to what Hotz wants
("LLM-generated decision trees").

---

## 4. Comparison to Other Search/Optimization Approaches

### 4.1 Equality Saturation (egg / TENSAT)

E-graphs compactly represent many equivalent programs. Rewrite rules are applied
saturatively. Extraction picks the optimal.

| Property | Equality Saturation | SUP Search |
|----------|-------------------|------------|
| Representation | E-graph (equivalence classes) | Superposed terms |
| Search direction | Bottom-up (add equivalences) | Top-down (evaluate + prune) |
| Handles control flow | No (DAG-based) | Yes (full lambda calculus) |
| Domain | Term rewriting | General functional programs |
| For kernel opt | TENSAT: 16% over TASO | Unproven but higher ceiling |

TENSAT applies equality saturation to tensor **graph** optimization (operator
fusion, layout) — not kernel **schedule** optimization. It found 16% speedup
over TASO with 48x less optimization time. Complementary to, not replacement for,
schedule search.

Source: [TENSAT paper](https://arxiv.org/abs/2101.01332),
[egg library](https://egraphs-good.github.io/)

### 4.2 Superoptimization (STOKE)

STOKE uses MCMC random walk through x86-64 instruction sequences. Found
algorithmically distinct code matching gcc -O3. Limited to straight-line code.

SUPs could theoretically encode the STOKE search — all candidate instruction
sequences as a superposition, evaluated against a spec. The sharing would be
massive (most sequences share most instructions). Unexplored territory.

### 4.3 TVM Ansor

Sketch-based search: 6+ derivation rules produce <10 sketches per subgraph,
each with billions of annotation choices. XGBoost cost model + evolutionary search.

The **sketch structure** is exactly what SUPs could search: a SUP over sketch
variants (which loops to tile, where to fuse) with shared cost model evaluation.
Ansor's evolutionary search over annotations (tile sizes) would remain.

### 4.4 Halide Auto-Scheduler

Beam search over schedule space with a learned neural cost model. 2.29x over
previous auto-scheduler. Same greedy-pruning limitations as tinygrad's BEAM.

The cost model is learned (neural) — could potentially be expressed as a pure
function and evaluated with SUP sharing, but the neural forward pass is dense
matrix computation, not the kind of symbolic computation SUPs accelerate.

### 4.5 Sketch-Based Synthesis (CEGIS)

Closest conceptual parallel. A sketch is a partial program with holes. CEGIS
uses SAT/SMT to fill holes. SUPs encode the same holes but evaluate via
interaction nets instead of SAT solving.

For kernel schedules: each `Opt(op, axis, amount)` slot is a "hole". A SUP
over valid fillings of all holes simultaneously is a superposition of all valid
schedules. With 5 holes of 10 options each: 100K candidates, evaluated with
sharing in potentially thousands of interactions instead of 100K compilations.

---

## 5. Concrete Architecture for TinyHVM

### 5.1 What TinyHVM Already Has

TinyHVM is an interaction-net runtime. It already has:
- SUP/DUP nodes (TAG_SUP, TAG_DP0, TAG_DP1) with label-based dispatch
- The enter/apply trampoline reducer that handles SUP interactions
- `term_clone()` for deep cloning of IC terms
- TAG_REF for named definitions (recursive functions)

**SUP-based search is native to the runtime.** No new infrastructure needed for
the search mechanism itself. The question is how to connect it to kernel optimization.

### 5.2 Proposed Architecture

```
                         +---------------------------+
                         |    Schedule Search via     |
                         |    Superposed IC Terms     |
                         +---------------------------+
                                    |
                      IC reduction (optimal sharing)
                                    |
                         +---------------------------+
                         |   Symbolic Cost Model      |
                         | (pure function in IC)      |
                         +---------------------------+
                                    |
                           collapse: top K
                                    |
                         +---------------------------+
                         |   Hardware Verification    |
                         | (compile + time top K)     |
                         +---------------------------+
                                    |
                              best schedule
```

**Phase 1**: Symbolic pre-filter. Encode schedule validity and cost estimation
as IC terms. Search all schedule combinations via SUPs. Collapse to top K.

**Phase 2**: Hardware verification. Compile and time only the top K candidates
on real GPU. K << total search space (maybe 5-10 vs 100K+).

**Phase 3**: Rule extraction. Over many kernels, use SupGen to synthesize
general optimization rules from the (kernel, best_schedule) pairs.

### 5.3 Schedule Actions as IC Terms

```c
// Represent a schedule action as an IC term
// Opt(TC, 0, (0,2,1))  ->  term: APP(APP(APP(REF_TC, NUM_0), NUM_2), NUM_1)
// Opt(LOCAL, 1, 16)     ->  term: APP(APP(REF_LOCAL, NUM_1), NUM_16)

// A superposition of two actions:
// &0{ APP(REF_LOCAL, NUM_1, NUM_16), APP(REF_LOCAL, NUM_1, NUM_32) }
// = "local axis 1, tile size 16 OR 32"

// Full schedule = sequence of actions:
// APP(action1, APP(action2, APP(action3, REF_DONE)))
// Superposed schedule = superposed at each action slot
```

### 5.4 Symbolic Cost Model in IC

```c
// Cost model as a pure IC function:
// cost(kernel_shape, schedule) -> NUM(estimated_cycles)
//
// kernel_shape = { ndim, dims[], strides[], dtype }
// schedule = list of actions
//
// Simplified model:
// cost = global_loads * 4 + shared_loads * 1 + flops * 0.5 + barriers * 100
//
// This is a pure function — SUPs can search over it with optimal sharing.
// The model is inaccurate but cheap. Hardware timing corrects the top K.
```

### 5.5 Files to Create

| File | Purpose | Size |
|------|---------|------|
| `src/search/cost_model.c` | Symbolic cost model as IC terms | ~200 LOC |
| `src/search/schedule_sup.c` | Construct superposed schedule space | ~150 LOC |
| `src/search/collapse.c` | Extract top-K from superposed result | ~100 LOC |
| `src/search/verify.c` | Hardware timing of top-K candidates | ~100 LOC |

---

## 6. Key Questions & Risks

### Will the symbolic cost model be accurate enough?

Probably not for final selection — but it doesn't need to be. It only needs to
be accurate enough to **rank-order** candidates so the top-K contains the true
optimum. Even 50% correlation with real performance would reduce the search
space from 100K to 10 candidates.

Ansor's XGBoost model uses 164 features and achieves high correlation. A simpler
model (memory access count, arithmetic intensity, occupancy) may suffice for
filtering.

### How large is the useful search space?

tinygrad's BEAM has ~193 static actions, but typically 10-50 valid per step,
over 3-8 steps. Effective space: 10^3 to 10^8. SUPs handle this comfortably
(the ADD-CARRY benchmark covers 2^16 = 65K, and cost per candidate is sub-1).

### Can IC terms represent schedule transformations?

Yes — schedule actions are discrete, compositional, and their validity constraints
are pure functions. This is exactly the kind of structured, redundant search space
where SUPs provide maximal sharing.

### What's the latency budget?

BEAM takes 40-60 seconds per kernel. If SUP-based search takes 1 second for the
symbolic pre-filter + 5 seconds for hardware timing of top-K, that's an order of
magnitude improvement. Results are cached either way.

### Will this actually beat BEAM?

**For large search spaces**: Almost certainly yes. BEAM=2 covers ~100-600 candidates.
SUP search covers the entire space. The probability of finding the global optimum
is much higher.

**For small search spaces**: Probably comparable. If only 20 valid schedules exist,
BEAM=8 already evaluates most of them. The overhead of constructing IC terms may
not be worth it.

**The sweet spot**: Medium-large kernels (matmuls, convolutions, attention) with
many valid tiling/parallelism combinations. These are also the performance-critical
kernels.

---

## 7. Relationship to Other TinyHVM Plans

- **Lazy Graph Compiler** (`lazy_graph_compiler.md`): Produces the kernel DAGs that
  need schedule optimization. SUP search optimizes individual kernels within the DAG.
- **Work-Stealing** (`work_stealing.md`): Parallel IC reduction accelerates SUP
  evaluation. Independent SUP branches reduce on separate threads.
- **JIT/Ranges/Patterns** (planned): JIT replay skips schedule search entirely for
  cached kernels. SUP search is for cold-start kernel optimization.
- **Pattern Matcher**: Could use SupGen-discovered rewrite rules instead of hand-coded
  patterns. SupGen finds the rules; the pattern matcher applies them at runtime.

---

## 8. References

### SupGen / NeoGen
- [ADD-CARRY: Fast DPS with HVM Superpositions](https://gist.github.com/VictorTaelin/d5c318348aaee7033eb3d18b0b0ace34)
- [Accelerating DPS with SUP Nodes](https://gist.github.com/VictorTaelin/7fe49a99ebca42e5721aa1a3bb32e278)
- [Superposed Lambda Calculus Enumerator](https://gist.github.com/VictorTaelin/7c4c69a1f07b5c668be613f1032e7d4e)
- [Truly Optimal Evaluation with Unordered Superpositions](https://gist.github.com/VictorTaelin/93c327e5b4e752b744d7798687977f8a)
- [Simple SAT Solver via Superpositions](https://gist.github.com/VictorTaelin/9061306220929f04e7e6980f23ade615)
- [KolmoLang](https://gist.github.com/VictorTaelin/41d3c07d62cbb0978924fa6d504c4bee)
- [SupTT Spec](https://gist.github.com/VictorTaelin/903f20e0a75263edaad0a4f7e2b9fa05)
- [Scaling HVM for Optimal Theorem Proving](https://gist.github.com/VictorTaelin/a060db7bada170e50d61871a752daf6e)
- [NeoGen benchmarks (Taelin)](https://x.com/VictorTaelin/status/1904727018899439853)
- [NeoGen: Peano Sort](https://x.com/VictorTaelin/status/1957909540894089545)
- [HVM4: compiled superpositions](https://x.com/VictorTaelin/status/1985320306001477783)
- [HOC thesis (local)](resources/gists/hoc_thesis.md)
- [Show HN: SupGen](https://news.ycombinator.com/item?id=42771885)

### Tinygrad BEAM
- [tinygrad/codegen/opt/search.py](https://github.com/tinygrad/tinygrad) (~400 LOC)
- [tinygrad speed docs](https://docs.tinygrad.org/developer/speed/)
- [Hotz on BEAM future (Dec 2025)](https://github.com/geohotstan/tinycorp-meetings/blob/master/last-week-in-tinycorp/2025-12-22/meeting-transcript.md)
- [Hotz on BEAM limits (Mar 2026)](https://github.com/geohotstan/tinycorp-meetings/blob/master/last-week-in-tinycorp/2026-03-24/meeting-transcript.md)

### Alternative Approaches
- [TENSAT: equality saturation for tensors](https://arxiv.org/abs/2101.01332)
- [egg: equality saturation library](https://egraphs-good.github.io/)
- [Ansor: generating high-performance tensor programs](https://arxiv.org/abs/2006.06762)
- [Halide auto-scheduler (Adams et al. 2019)](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf)
- [STOKE: stochastic superoptimizer](https://theory.stanford.edu/~aiken/publications/papers/asplos13.pdf)
- [SKETCH: program synthesis](https://people.csail.mit.edu/asolar/papers/Solar-Lezama09.pdf)
- [About efficient reduction of lambda terms (Asperti)](https://arxiv.org/abs/1701.04240)
- [Lamping: optimal lambda calculus reduction](https://dl.acm.org/doi/10.1145/96709.96711)
- [Lafont: interaction combinators](https://www.sciencedirect.com/science/article/pii/S0890540197926432)
