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

## Future: General Fusion via FUSING Nodes

The current fused path is a special case for `SUM(MUL)`. The full vision is an IC-native
FUSING accumulator node that absorbs fuseable ops one at a time:

```
Step 0: SUM(unreduced_child) → create FUSING{reduce=SUM}
Step 1: FUSING{SUM} ⊳ MUL(a, b) → absorb → FUSING{SUM,MUL}(a, b)
Step 2: FUSING{SUM,MUL} ⊳ RELU(x) → absorb → FUSING{SUM,MUL,RELU}(x)
Step 3: FUSING{SUM,MUL,RELU} ⊳ LOAD(buf) → boundary → emit one kernel
```

Interaction rules:
```
FUSING ⊳ ELEMENTWISE(args...)  → absorb, new children = args
FUSING ⊳ MOVEMENT(arg)        → absorb as index transform
FUSING ⊳ REDUCE(arg)          → STOP (two reduces = two kernels)
FUSING ⊳ LOAD/materialized    → STOP, emit fused kernel
FUSING ⊳ requires_grad input  → STOP (autograd boundary)
```

The last rule is the same gate as today: fusion terminates at any input that needs a gradient.

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
