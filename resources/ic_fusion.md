# IC Op Fusion via Intermediate Nodes

How TinyHVM fuses `SUM(MUL(a, b))` and the interaction with autograd.

## The Fusion

Instead of materializing `MUL(a, b)` into an intermediate buffer and then reducing it,
`thvm_reduce` detects the pattern `SUM(MUL(a, b))` and dispatches a single fused kernel:

```
SUM(MUL(a, b))  →  one kernel, no intermediate MUL buffer
```

For conv-like workloads (pool+expand+mul+sum), this eliminates hundreds of MB of intermediates
that would otherwise be allocated per training step (see memory impact section below).

### Implementation

In `thvm_reduce`, when processing `UOP_SUM`, look one level deeper:

```c
if (uop == UOP_SUM) {
    Term child = heap_read(ctx, loc);
    if (term_tag(child) == TAG_TOP && term_ext(child) == UOP_MUL) {
        // Pattern matched — but ONLY fuse if neither input requires grad
        if (!ma->requires_grad && !mb->requires_grad) {
            // dispatch fused mul+reduce kernel
        }
    }
}
```

The fused CPU path reads both inputs with strided indexing (handles broadcast views), then
accumulates `sum(a[i] * b[i])` directly without an N-element MUL buffer.

The Metal path calls `metal_mul_reduce_sum(...)` — a fused GPU kernel.

## The Autograd Gate (Critical)

**The fused path must be skipped when any MUL input requires a gradient.**

### Why

The GRAD handler for SUM recurses into its source via `src_ids[0]`. If `SUM(MUL(a,b))` is
fused, the SUM tensor's provenance records `src_ids[0]` as `a` (MUL's input), not as the MUL
output. The GRAD handler then tries to differentiate through `SUM → a`, skipping the MUL
backward entirely. For `loss = sum(diff * diff)` this loses the `2*diff` factor — gradients
are wrong.

When the MUL is NOT fused, the chain is:
```
SUM.src_ids[0] = sq_id         (sq = MUL(diff, diff))
sq.src_ids[0] = diff_id        (diff = pred - target)
```
GRAD fires: SUM bwd → MUL bwd (`da = gy * diff`, `db = gy * diff`) → SUB bwd → ... correct.

### The Correct Gate

```c
if (!ma->requires_grad && !mb->requires_grad) { /* fuse */ }
```

**Not** `!ctx->recording`. The `recording` flag is off during backward, but tensors still
carry `requires_grad = 1`. Two tensors on a gradient path can be encountered while
`recording = 0` (during backward reduction itself).

## General Fusion via FUSING Nodes

The current `SUM(MUL)` fast path is ad-hoc. The general mechanism is a `UOP_FUSING` node that:

1. **Wraps any matched subgraph** without destroying it — the original TAG_TOP subnet stays in the heap
2. **Dispatches one fused kernel** using the realized inputs
3. **Delegates GRAD transparently** to the original unfused subnet

```
Forward:
  FUSING(orig_subgraph) → realized tensor (one kernel, no intermediates)

Backward:
  GRAD(FUSING(orig), gy, x)  →  GRAD(orig, gy, x)
```

The GRAD handler for `UOP_FUSING` is one line: reconstruct the original term from `src_ids[0]` (the heap loc of the original TAG_TOP root) and fire GRAD on it. The original subnet is fully intact — no special per-op backward cases, no provenance hacks, no `!requires_grad` guards.

```c
case UOP_FUSING: {
    // src_ids[0] = heap loc of original unfused TAG_TOP root
    // (e.g. SUM_loc, which still points to MUL(a,b) in the heap)
    Term orig = term_new(TAG_TOP, ctx->tensors[y_id].orig_uop,
                         (u64)ctx->tensors[y_id].src_ids[0]);
    MEMO_RETURN(GRAD3(orig, gy, x));
}
```

### Interaction rules for fusion

`try_fuse()` walks the lazy graph pairwise. Each `⊳` is a single IC interaction:

```
FUSING ⊳ ELEMENTWISE(args...)  → absorb, new children = args
FUSING ⊳ MOVEMENT(arg)        → absorb as index transform
FUSING ⊳ REDUCE(arg)          → if first reduce: absorb; else STOP (two reduces = two kernels)
FUSING ⊳ LOAD/realized        → STOP, emit fused kernel
```

No `requires_grad` gate at all — since backward just walks the original graph.


## Memory Impact

With fusion (`pool+expand+mul+sum → one fused conv kernel`):

| | Without fusion | With fusion |
|---|---|---|
| conv1 intermediates | ~43 MB | ~2.3 MB |
| conv2 intermediates | ~471 MB | ~0.5 MB |
| Total | ~515 MB | ~2.8 MB |

~98% reduction in peak intermediate memory for conv layers.

## Relationship to Other Docs

- `ic_autograd.md`: gradient rules, seed shape, strided reduce, SUM backward invariant
- `ic_optimization.md`: ERA=DCE, DUP=CSE, TOP-TOP=fusion — theory
- `engineering_reference.md`: ggml/tinygrad patterns for reference
