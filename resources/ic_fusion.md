# IC Op Fusion via Intermediate Nodes

How to replicate tinygrad's PatternMatcher-based kernel fusion using IC's
pairwise interaction rules, with intermediate "accumulator" nodes.

## Problem

TinyHVM currently materializes every UOp eagerly:

```
reduce(SUM(MUL(RELU(LOAD(x))))) =
  LOAD(x)           → buf_x      (kernel 1)
  RELU(buf_x)       → buf_relu   (kernel 2, allocates)
  MUL(buf_relu, w)  → buf_mul    (kernel 3, allocates)
  SUM(buf_mul)       → result    (kernel 4)
```

4 kernels, 3 intermediate buffers. Tinygrad fuses this into 1 kernel, 0
intermediate buffers.

## Tinygrad's Current Approach (from `tinygrad/schedule/kernelize.py`)

Tinygrad's fusion is **not** the old `linearize()` — it's now a series of
`graph_rewrite` passes using `PatternMatcher`:

1. **`kernelize_sym`** — simplify views, constant fold
2. **`pm_fuse` / `do_fuse`** — propagate FUSE markers through the graph:
   - Elementwise fuses into elementwise
   - Elementwise fuses INTO a reduce
   - Reduce does NOT fuse past another reduce
   - Movement ops (reshape, expand, permute) are free — index math only
3. **`group_realizes`** — decide materialization points
4. **`create_kernels`** — wrap fused subgraphs into `Ops.KERNEL` nodes
5. **`pm_lowerer`** — ShapeTracker → index math (load/store with computed indices)
6. **`block_create` + `block_merge`** — schedule instructions within a kernel
7. **`renderer/cstyle.py`** — emit Metal/CUDA source from the instruction list

The fusion rules (`pm_fuse`, line 217-238 of `kernelize.py`) are:

```python
# FUSE elementwise: push FUSE marker through ALU ops
(UPat(Ops.VIEW, src=(UPat({*GroupOp.ALU, Ops.CAST}),), name="view").fuse(),
 lambda alu, view: alu.replace(src=tuple(apply_swizzle(x).fuse() for x in alu.src)))

# FUSE reduce: mark reduce's input for fusion
(UPat(Ops.REDUCE_AXIS, name="r").fuse(),
 lambda r: r.replace(src=(r.src[0].fuse(),), arg=r.arg+(True,)))
```

Key insight: fusion is decided by **propagating markers** through the graph,
not by multi-node pattern matching.

## IC Limitation: Pairwise Interactions Only

IC fundamentally does pairwise interactions — two nodes meet at their
principal ports and rewrite. You cannot "see" a 3-node pattern like
`SUM(MUL(RELU(...)))` in one step.

## Solution: FUSING Accumulator Nodes

Use a single intermediate node type — `FUSING` — that absorbs fuseable ops
one at a time through pairwise interactions:

```
Step 0: thvm_reduce encounters SUM(unreduced_child)
        → create FUSING{reduce=SUM} node
        → FUSING now interacts with the unreduced child

Step 1: FUSING{SUM} ⊳ MUL(a, b)
        → MUL is elementwise, absorb it
        → FUSING{SUM, MUL}(a, b)
        → now interact with a and b

Step 2: FUSING{SUM, MUL} ⊳ RELU(x)
        → RELU is elementwise, absorb it
        → FUSING{SUM, MUL, RELU}(x)

Step 3: FUSING{SUM, MUL, RELU} ⊳ LOAD(buf)
        → LOAD is a materialized input — boundary!
        → emit ONE kernel that does: load → relu → mul → sum
```

Each `⊳` is a single pairwise IC interaction. The FUSING node carries:
- `ops[]`: accumulated operation chain
- `shapes[]`: shapes at each level (for index math)
- `n_inputs`: how many external inputs feed the chain
- `has_reduce`: whether a reduce op has been absorbed

### Interaction Rules

```
FUSING ⊳ ELEMENTWISE(args...)    → absorb, FUSING with args as new children
FUSING ⊳ MOVEMENT(arg)          → absorb as index transform, keep walking
FUSING ⊳ REDUCE(arg)            → STOP (two reduces = two kernels)
FUSING ⊳ LOAD/materialized      → STOP, emit fused kernel
FUSING ⊳ DUP                    → STOP (shared input = materialization point)
```

### Fusion State Machine

```
          ┌─ elem ──┐    ┌─ elem ──┐
          ▼          │    ▼          │
    ┌──────────┐     │  ┌─────────────────┐
──▶ │ BUILDING │─────┘  │ BUILDING+REDUCE │──── LOAD ──▶ emit fused kernel
    │ (elem)   │        │ (elem+reduce)   │
    └──────────┘        └─────────────────┘
          │                    ▲
          │── REDUCE ──────────┘
          │
          │── LOAD ──▶ emit fused elementwise kernel
```

Two states: "absorbing elementwise only" and "absorbed a reduce."
A second reduce triggers kernel emission.

## Implementation in TinyHVM

### Minimal Prototype: `reduce_fused()`

In `thvm_reduce`, when we encounter a TAG_TOP node, instead of eagerly
reducing children first, check if the op is fuseable:

```c
case TAG_TOP: {
    u32 uop = term_ext(t);

    // Check if this starts a fuseable chain
    if (is_reduce_op(uop) || is_elementwise_op(uop)) {
        // Try to build a fused op chain
        FuseState fs = {0};
        Term fused = try_fuse(ctx, t, &fs);
        if (fs.n_ops > 1) {
            // Emit a single fused kernel
            return dispatch_fused(ctx, &fs);
        }
    }

    // Fall through to normal eager reduce
    ...
}
```

`try_fuse()` walks the unreduced graph pairwise, accumulating:

```c
typedef struct {
    u32 ops[MAX_FUSE_DEPTH];  // op chain (outer to inner)
    u32 n_ops;
    Shape shapes[MAX_FUSE_DEPTH];
    Term inputs[MAX_FUSE_INPUTS]; // leaf inputs (already materialized)
    u32 n_inputs;
    bool has_reduce;
} FuseState;

Term try_fuse(TinyHVM *ctx, Term t, FuseState *fs) {
    u32 uop = term_ext(t);

    if (is_reduce_op(uop)) {
        if (fs->has_reduce) return t; // boundary: second reduce
        fs->has_reduce = true;
    } else if (!is_elementwise_op(uop) && !is_movement_op(uop)) {
        return t; // boundary: not fuseable
    }

    fs->ops[fs->n_ops++] = uop;

    // Walk children without reducing them
    Term child_a = heap_peek(ctx, ...); // peek, don't reduce
    if (term_tag(child_a) == TAG_TOP) {
        return try_fuse(ctx, child_a, fs); // absorb next level
    } else {
        // Hit a materialized input or LOAD — done
        fs->inputs[fs->n_inputs++] = thvm_reduce(ctx, child_a);
        return child_a;
    }
}
```

### Fused Kernel Dispatch

`dispatch_fused()` generates a single Metal kernel that chains the ops:

```metal
// Generated fused kernel for SUM(MUL(RELU(x), w)):
kernel void fused_reduce_elem(
    device float *out,
    device float *x,
    device float *w,
    uint gid [[thread_position_in_grid]])
{
    uint outer = gid;
    float acc = 0.0;
    for (uint r = 0; r < reduce_dim; r++) {
        uint idx = outer * reduce_dim + r;
        float v = x[idx];       // LOAD
        v = max(v, 0.0);        // RELU (fused)
        v = v * w[idx];         // MUL  (fused)
        acc += v;               // SUM  (fused reduce)
    }
    out[outer] = acc;
}
```

No intermediate buffers — the entire chain runs in registers.

### Phase 1: No Codegen (Reuse Existing Kernels)

Before we have Metal codegen, we can still fuse at the C level:

```c
void dispatch_fused_cpu(FuseState *fs, ...) {
    for (u32 i = 0; i < outer_numel; i++) {
        f32 acc = 0;
        for (u32 j = 0; j < reduce_dim; j++) {
            f32 v = input[i * reduce_dim + j];
            // Apply fused ops in order
            for (u32 k = fs->n_ops - 1; k > 0; k--) {
                switch (fs->ops[k]) {
                    case UOP_RELU: v = v > 0 ? v : 0; break;
                    case UOP_MUL:  v *= input_b[...]; break;
                    // ...
                }
            }
            if (fs->ops[0] == UOP_SUM) acc += v;
        }
        output[i] = acc;
    }
}
```

This already eliminates intermediate buffers on CPU.

## Impact on MNIST Memory

Current memory usage per training step (estimated):
```
conv1 pool:   BS×1×24×24×5×5   = 3.7M elements × 4B = 14.4 MB
conv1 expand: same              = 14.4 MB
conv1 mul:    same              = 14.4 MB
conv2 pool:   BS×32×8×8×5×5    = 40.9M elements     = 157 MB
conv2 expand: same              = 157 MB
conv2 mul:    same              = 157 MB
Total intermediates: ~515 MB (from conv alone)
```

With fusion (pool+expand+mul+sum → one fused conv kernel):
```
conv1 output: BS×32×24×24      = 0.59M elements     = 2.3 MB
conv2 output: BS×64×8×8        = 0.13M elements     = 0.5 MB
Total intermediates: ~2.8 MB (98% reduction)
```

## Relationship to Existing Docs

- `ic_optimization.md`: covers the theory (ERA=DCE, DUP=CSE, TOP-TOP=fusion)
- `engineering_reference.md`: covers ggml/tinygrad patterns (graph_compute, ShapeTracker)
- `flexible_gradients.md`: IC-native autograd (thvm_grad)
- **This doc**: concrete fusion mechanism via FUSING accumulator nodes
