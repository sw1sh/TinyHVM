# IC Rewriting as Optimization

Can we use interaction net reduction rules for kernel optimization? What do DUP/SUP/ERA give us that tinygrad's pattern matcher doesn't?

## What Tinygrad Does

Tinygrad optimizes kernels with a **pattern matcher** — a list of `(pattern, replacement)` rules applied to the UOp graph until fixpoint:

```python
# Tinygrad examples (from decompositions.py and symbolic.py):
add(x, 0) → x           # identity
mul(x, 1) → x           # identity
mul(x, 0) → 0           # annihilation
add(mul(a,b), mul(a,c)) → mul(a, add(b,c))  # distribute
reshape(reshape(x, s1), s2) → reshape(x, s2)  # fuse reshapes
```

This works well but has limitations:
- Rules are hand-written and hardware-specific
- No formal guarantee that multiple rules compose correctly
- Fusion decisions are heuristic (BEAM search over OptOps)
- Common subexpression elimination requires a separate pass

## What IC Gives Us

IC has three structural operations that map to compiler optimizations:

### 1. ERA = Dead Code Elimination (free)

If a tensor result is never used, the ERA node propagates through the TOP nodes and erases the entire subgraph:

```
         [MM]──→ [ADD]──→ [ERA]
          ↑        ↑
          x        b
```

ERA-TOP interaction rule: if the output of a TOP is erased, erase the TOP and propagate ERA to its inputs (if they have no other consumers). No separate DCE pass needed — it falls out of normal reduction.

### 2. DUP-SUP Annihilation = CSE (free)

When the same value is used twice, DUP creates two copies. If those copies are immediately consumed by a SUP (superposition), DUP and SUP annihilate:

```
Before:  x ──→ [DUP]──→ a ──→ [f]
                    └──→ b ──→ [f]   (same f applied twice)

After:  x ──→ [f]   (shared, computed once)
```

In a tensor graph, this means: if two ops consume the same intermediate, the system naturally shares the computation. In tinygrad, this requires explicit buffer management and copy avoidance. In IC, it's a consequence of the reduction rules.

**Concrete example:** Computing `loss1 = sum(mm(x,w1))` and `loss2 = sum(mm(x,w2))` — both share `x`. In IC, `x` gets DUP'd, and if `w1 == w2`, DUP-SUP annihilates and `mm(x,w)` is computed once. We don't need to check for this — it happens during reduction.

### 3. DUP-TOP = Kernel Fusion (the big one)

When a DUP propagates through a chain of TOPs, the interaction rules can *fuse* operations:

```
Before:  x ─→ [MM] ─→ [ADD] ─→ [RELU] ─→ result
                         ↑
                         b

After optimization:  x ─→ [FUSED_MM_ADD_RELU] ─→ result
                                    ↑
                                    b
```

How? When the reducer walks a chain of connected TOP nodes, instead of dispatching each to GPU separately, it can:
1. Recognize the pattern: `TOP(relu, TOP(add, TOP(mm, x, w), b))`
2. Emit a single fused kernel that does `relu(mm(x,w) + b)` in one GPU dispatch

This is what tinygrad's scheduler/lowerer does with explicit graph analysis. IC can do it during reduction: TOP-TOP interaction rules rewrite connected ops into fused ops.

### 4. SUP = Speculative Kernel Variants (novel)

SUP nodes represent superposition — "this is simultaneously A and B." We could use this to represent **multiple implementations** of an operation:

```
mm(A,B) = SUP(mm_naive(A,B), mm_tiled(A,B), mm_mps(A,B))
```

During reduction, the system picks the best variant (by profiling) and collapses the SUP. This is similar to BEAM search but expressed within the calculus itself — the optimization search space *is* an interaction net.

---

## What's Actually Worth Building

Not all of these are equally practical. Here's an honest assessment:

| IC Optimization | Tinygrad Equivalent | IC Advantage | Worth It? |
|---|---|---|---|
| ERA → DCE | Dead code elim pass | Falls out of reduction | **Yes** (it's free) |
| DUP-SUP → CSE | Buffer dedup / caching | Automatic sharing | **Yes** (it's free) |
| TOP-TOP → Fusion | Scheduler + Lowerer | Cleaner, confluent | **Maybe** (needs codegen) |
| SUP → Kernel search | BEAM search | Elegant but slow | **Later** (research) |

### Phase 2-3: Take the Free Wins

ERA-DCE and DUP-SUP-CSE emerge naturally from implementing the core IC reduction rules. We get them the moment we implement `DP0/DP1-TOP` and `ERA-TOP` interactions. Zero extra work.

### Phase 4+: Fusion via IC

Kernel fusion through IC rewriting requires a code generation backend (not pre-built kernels). Once we have Metal codegen, we can:
1. Walk the TOP chain before dispatching
2. Use interaction rules to merge connected TOPs into a single fused TOP
3. Emit one kernel for the fused op

This is where IC actually beats pattern matching: the rewriting is confluent (provably correct regardless of rule application order), and new fusion patterns are just new interaction rules — no need to re-engineer the scheduler.

### SUP-Based Optimization: Research Territory

Using SUP for kernel variant selection is theoretically clean but practically slow — you'd need to reduce all branches to profile them. BEAM search is more practical because it explores the space incrementally with early pruning.

**However**, SUP could work for *compile-time* decisions where we already know the hardware: `SUP(metal_mm, blas_mm)` collapses to `metal_mm` on Apple Silicon. This is just conditional compilation expressed as IC.

---

## The Honest Answer

DUP/SUP/ERA buy us three concrete things:

1. **Free DCE** — ERA propagation eliminates dead ops during normal reduction
2. **Free CSE** — DUP-SUP annihilation shares computation without explicit management
3. **A clean theory for fusion** — TOP-TOP interaction rules can express kernel fusion as confluent rewriting

The first two are real and free. The third requires codegen infrastructure to be useful. We're not there yet, but the architecture is set up for it.

What IC does NOT give us: SIMD vectorization, workgroup sizing, memory coalescing, register tiling. Those need hardware-specific optimization (BEAM search, tensor cores, etc.) that operates below the IC level. IC optimizes the *computation graph*. Hardware optimization happens in the *kernel generator*.
