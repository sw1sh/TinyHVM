# Fusion: FUSE to Structural KERNEL

`FUSE` propagates through the compute graph like `GRAD` propagates through the
backward graph: by local IC interactions, not by an imperative fusion pass.

The important split is now:

- `FUSE` marks fuseable structure and keeps propagating.
- `KERNEL` is the explicit fused kernel boundary on the heap.
- dispatch/cache is a later runtime concern keyed from that `KERNEL`, not the IR
  itself.

## Core Forms

```text
FUSE(payload)
KERNEL(left, right_or_meta, root_uop)
```

`FUSE2` is no longer part of the intended IR contract. It only remains as a
deprecated compatibility shim for older heaps/tests and is immediately lowered
to `KERNEL`.

## Entry FUSE Interactions

| Pattern | Result | Notes |
|---------|--------|-------|
| `FUSE(TEN)` | `TEN` | Leaf, nothing to fuse |
| `FUSE(NUM)` | `NUM` | Leaf |
| `FUSE(ERA)` | `ERA` | Leaf |
| `FUSE(binary(a,b))` | `KERNEL(FUSE(a), FUSE(b), binary)` | Structural fused node |
| `FUSE(unary(a))` | `KERNEL(FUSE(a), ERA, unary)` | Unary sentinel uses inert `ERA` |
| `FUSE(view_or_reduce(a, meta))` | `KERNEL(FUSE(a), meta, op)` | Keep metadata visible |
| `FUSE(SEQ(a,b))` | `KERNEL(FUSE(a), FUSE(b), SEQ)` | Ordering stays explicit |
| `FUSE(CTR(...))` | `CTR(FUSE(...), ...)` | Distribute |
| `FUSE(ASSIGN(dst, src))` | `ASSIGN(dst, FUSE(src))` | Fuse only the source |
| `FUSE(KERNEL(...))` | `KERNEL(...)` | Already structural |

## KERNEL Semantics

`KERNEL` is lazy structural IR, not an already-dispatched result.

- Heap layout: `TAG_TOP(UOP_KERNEL, loc)`, heap `[left, right_or_meta, NUM(root_uop)]`
- The node stays visible in the net until a strict context reaches it.
- `SEQ`-shaped kernels degrade back to `SEQ(left, right)` once both children are
  ready.
- Non-`SEQ` kernels dispatch only when reduction actually demands a `TEN`
  result, such as under `ASSIGN`, `LOG_PRINT`, or as the final eval result.

This means step graphs can show:

1. `FUSE` propagation
2. `KERNEL` materialization as a real heap node
3. `KERNEL -> TEN` dispatch
4. `ASSIGN` / cleanup

## Runtime Lowering

When a demanded `KERNEL` is ready:

1. The structural `KERNEL` subtree is converted back into a raw compute term.
2. `fuse_build_kernel()` lowers that compute term into a `KernelEntry`.
3. The resulting runtime entry is cached behind the structural kernel location.
4. Buffer-epoch checks decide whether a cached `TEN` result is still valid.

So the runtime still uses `KernelEntry`, `kid_results[]`, and epoch tracking, but
those are implementation details behind the structural heap node.

## Training Loop

```text
train(counter)(w) = IFZ(counter, w, λm. SEQ(ASSIGN(w, w*2), train(m)(w)))
```

1. Phase 1 exposes `SEQ(ASSIGN(w, MUL(w,2)), train(m)(w))`.
2. `FUSE` rewrites `MUL(w,2)` into a visible `KERNEL`.
3. `ASSIGN` forces that `KERNEL` to dispatch to `TEN`.
4. `ASSIGN` writes the updated buffer and bumps the buffer epoch.
5. Later loop iterations revisit the same structural kernel and re-dispatch when
   epoch checks detect stale cached results.

## Files

- `src/interact/tensor_ops.c` — `FUSE`, compatibility `FUSE2`, and `KERNEL` dispatch
- `src/interact/_.c` — shared kernel helpers and arity metadata
- `src/reduce/_.c` — reducer readiness for structural kernels
- `src/interact/combinators.c` — `DUP`/`ERA` behavior for 3-slot kernels
- `src/debug/dump.c` and `src/debug/graph.c` — visible kernel tracing
