# Fusion: FUSE as Propagating IC Agent

FUSE propagates through the compute graph like GRAD propagates through the
backward graph — via pure IC interactions, no imperative walk.

## FUSE Forms

```
FUSE(payload)           Entry point: propagate into compute structure
FUSE(op, child)         Absorbed one unary op, waiting for child to resolve
FUSE2(op, left, right)  Absorbed one binary op, waiting for children to resolve
```

FUSE **absorbs** compute ops. The op becomes part of FUSE's metadata, not a
separate TAG_TOP node. When children resolve to TEN or KERNEL, FUSE creates
or merges a KERNEL.

## Entry FUSE Interactions

| Pattern | Result | Notes |
|---------|--------|-------|
| `FUSE(TEN)` | `TEN` | Leaf, nothing to fuse |
| `FUSE(NUM)` | `NUM` | Leaf |
| `FUSE(ERA)` | `ERA` | Leaf |
| `FUSE(MUL(a,b))` | `FUSE2(MUL, FUSE(a), FUSE(b))` | Absorb binary compute |
| `FUSE(NEG(a))` | `FUSE(NEG, FUSE(a))` | Absorb unary compute |
| `FUSE(SUM(a,axes))` | `FUSE(SUM, FUSE(a))` | Absorb reduce |
| `FUSE(RESHAPE(a,sh))` | `FUSE(RESHAPE, FUSE(a))` | Absorb movement |
| `FUSE(SEQ(a,b))` | `FUSE2(SEQ, FUSE(a), FUSE(b))` | Compose via SEQ |
| `FUSE(CTR(a,b,...))` | `CTR(FUSE(a), FUSE(b), ...)` | Distribute |
| `FUSE(ASSIGN(d,s))` | `ASSIGN(d, FUSE(s))` | Fuse the source |
| `FUSE(KERNEL)` | `KERNEL` | Already fused |
| `FUSE(DP0/DP1)` | `DP0/DP1` | Let reducer resolve DUP |

## Unary FUSE(op, child) Interactions

| Pattern | Result |
|---------|--------|
| `FUSE(op, FUSE_child)` | WNF (waiting) |
| `FUSE(op, TEN)` | `KERNEL` — single-op kernel: op(TEN) |
| `FUSE(op, KERNEL)` | `KERNEL` — absorb op on top of existing kernel |

## Binary FUSE2(op, left, right) Interactions

| Pattern | Result |
|---------|--------|
| `FUSE2(op, FUSE, FUSE)` | WNF (waiting) |
| `FUSE2(op, TEN, TEN)` | `KERNEL` — single-op kernel: op(TEN, TEN) |
| `FUSE2(op, KERNEL, TEN)` | `KERNEL` — absorb op on top |
| `FUSE2(op, TEN, KERNEL)` | `KERNEL` — absorb op on top |
| `FUSE2(op, KERNEL, KERNEL)` | `KERNEL` — merge: op composes two kernels |
| `FUSE2(SEQ, non-KERNEL, b)` | `SEQ(a, b)` — degrade to ordering |

## KERNEL is Lazy

FUSE/FUSE2 produces KERNEL — a lazy kernel spec, NOT a dispatched result.
KERNEL stays WNF on the heap. It only dispatches when **demanded**:

```
FUSE(ADD(MUL(a,b), c)):
  → FUSE2(ADD, FUSE2(MUL, FUSE(a), FUSE(b)), FUSE(c))
  → FUSE2(ADD, KERNEL(MUL,a,b), TEN(c))    // inner MUL → KERNEL, NOT dispatched
  → KERNEL(ADD(MUL(a,b),c))                  // merged into one fused kernel
  → dispatches only when ASSIGN or root demands TAG_TEN
```

This prevents premature dispatch: a parent FUSE2 can absorb a child KERNEL
into a larger fused kernel before anything dispatches.

## Training Loop

```
train(counter)(w) = IFZ(counter, w, λm. SEQ(ASSIGN(w, w*2), train(m)(w)))
```

1. Phase 1: IFZ fires once → `SEQ(ASSIGN(w, MUL(w,2)), train(m)(w))`. SEQ blocks.
2. FUSE propagates: `MUL(w,2)` → `FUSE2(MUL, TEN, TEN)` → KERNEL.
3. ASSIGN demands TEN → KERNEL dispatches → ASSIGN writes buffer.
4. SEQ continues → recursive reference to same body (Y-combinator, no clone).
5. DUP on KERNEL → fresh dispatch instance. Epoch check → re-dispatch.
6. One kernel spec, reused across iterations.

## Encoding

- Entry FUSE: `TAG_TOP(UOP_FUSE, loc)`, heap `[payload]`
- Unary FUSE: `TAG_TOP(UOP_FUSE, loc)`, heap `[NUM(op), child]`
  - Distinguished by `term_tag(heap[loc]) == TAG_NUM`
- Binary FUSE2: `TAG_TOP(UOP_FUSE2, loc)`, heap `[NUM(op), left, right]`

## Files

- `src/interact/tensor_ops.c` — FUSE/FUSE2/KERNEL interaction handlers
- `src/reduce/_.c` — arity, direct_uop, apply-phase dispatch
- `src/interact/combinators.c` — DUP ⊳ FUSE2 commutation
- `src/tinyhvm.h` — UOP_FUSE2 definition
