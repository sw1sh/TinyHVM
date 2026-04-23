# Chapter 1: Terms

Everything in TinyHVM is a **term** — a 64-bit word that either *is* a value (a leaf) or *points* into the heap at a compound node. This chapter shows how terms are encoded, how they sit in the heap, and how to inspect both from Wolfram.

---

## Setup

```wolfram
Needs["TinyHVM`"]
```

```wolfram
TInit[]
```

The helpers used throughout this chapter — `TBitField`, `TShapeLegend`, `THeapStrip` — are part of the TinyHVM paclet, so the `Needs[]` above is all you need.

---

## The 64-Bit Encoding

Every term packs four fields into a single machine word:

```wolfram
TBitField[{
    {"SUB", 1, "flag"},
    {"TAG", 7, "node kind"},
    {"EXT", 18, "tag-specific"},
    {"VAL", 38, "value / ptr"}}]
```

| Field | Bits | Purpose |
|-------|------|---------|
| **SUB** | 1  | Substitution flag — marks that a VAR has been written to |
| **TAG** | 7  | Node type — one of 27 tags defined in [tinyhvm.h](../../src/tinyhvm.h) |
| **EXT** | 18 | Extension data — meaning depends on tag (label for SUP/DUP, UOp code for TOP, dtype for TEN) |
| **VAL** | 38 | Value or heap pointer (256 GB address space) |

No pointers, no heap headers, no vtables — the whole node fits in a register.

---

## Tag Table

Tags defined by the runtime (codes match `TAG_*` in [tinyhvm.h](../../src/tinyhvm.h)):

| Code | Name | Description |
|------|------|-------------|
| 0 | APP | Application — `(f x)` |
| 1 | LAM | Lambda — `λx. body` |
| 2 | VAR | Variable — bound reference |
| 3 | SUP | Superposition — `{a, b}` |
| 4 | DP0 | DUP projection 0 |
| 5 | DP1 | DUP projection 1 |
| 6 | ERA | Eraser — discards values |
| 7 | NUM | Number — u32/f32 inline in VAL |
| 8 | REF | Named definition reference |
| 9 | OP2 | Binary op — `Op(x, y)` |
| 10 | TEN | Materialized tensor (VAL = tensor id) |
| 11 | TOP | Lazy tensor op (EXT = UOp code) |
| 12 | CTR | Multi-arity constructor |
| 13 | BRI | Bridge — `θx. body` (dual of LAM) |
| 14 | ANN | Annotation — `{term : type}` |

The remainder (DSU/DDU, USP/UDP, MAT, ANY, SEQ, ALO, INC, EQL/AND/OR) appear in later chapters.

### Shape legend

`TINetGraph` uses a small alphabet of shapes — triangles for binders, hexagons for supers, ovals for references, rectangles for everything else. Memorize these so the interaction-net pictures later in the chapter are legible at a glance:

```wolfram
TShapeLegend[]
```

Treat the *shape family* (triangle vs hexagon vs rectangle) and *fill color* as the primary cues; each `TINetGraph` node also carries its heap slot in an `@n` tail label.

---

## Heap-owning nodes vs references

A term handle — what `TNum[42]` returns — is a single 64-bit word in the `g_terms[]` handle table. Terms split into two groups by how they relate to the heap:

**Heap-owning** (compound) nodes allocate a run of consecutive heap slots and put the heap base in `VAL`. A `LAM` with `VAL=k` owns `heap[k], heap[k+1]`.

**Non-owning** words are either self-contained values or references that point *into* the heap without allocating there:

- `NUM`, `ERA` — self-contained leaves (their whole state is in the word).
- `VAR`, `REF` — references: `VAL` points at an existing binding slot or definition; no allocation of its own.

| Tag | Owns heap? | Layout |
|-----|-----------|--------|
| NUM | no | value in `VAL` |
| ERA | no | no data |
| VAR | no | `VAL` = binding slot this reference points to |
| REF | no | `VAL` = definition slot |
| LAM | **yes, 2 slots** | `heap[val]` = binding site, `heap[val+1]` = body |
| APP | **yes, 2 slots** | `heap[val]` = fun, `heap[val+1]` = arg |
| SUP | **yes, 2 slots** | `heap[val]` = left, `heap[val+1]` = right |
| OP2 | **yes, 2 slots** | `heap[val]` = arg0, `heap[val+1]` = arg1 |

A heap slot is just a 64-bit word — the *same* 64-bit encoding a term handle uses. So an `OP2`'s slots can directly hold a `NUM` word (leaf payload), a `VAR` word (reference back to some binder), or a heap-owning word like another `OP2` whose own `VAL` points deeper into the heap. Leaves never allocate; they ride inline in whatever slot references them.

A subtle case: the first slot owned by a `LAM` is its **binding site**. It's *initialized* with a `VAR` word that refers back to itself, as a "this binder has no value yet" placeholder. Beta reduction overwrites that slot with the argument. So when `THeapRead[lam_base]` reports `Tag -> Var`, that's the binder — owned by the LAM, not an independent VAR allocation.

---

## The Heap

The heap is a flat array of 64-bit words. Slot `0` holds an ERA sentinel so that `VAL=0` means "no target." Compound nodes allocate consecutive slots:

```
heap:  [ ERA ][ word₁ ][ word₂ ][ word₃ ] ...
         ↑       ↑
        loc=0   loc=1
```

`heap_pos` tracks the bump pointer. `THeapSnapshot[]["HeapSize"]` returns `heap_pos - 1` so a fresh context reports `0`.

---

## Hands-on: the fresh heap

```wolfram
TInit[];
THeapSnapshot[]
```

After `TInit[]`, the snapshot reports `HeapSize = 0` — the sentinel at slot 0 is not counted.

---

## Hands-on: NUM is a leaf

`TNum[42]` builds a handle whose word has `TAG=NUM, VAL=42`. No heap slot is used:

```wolfram
TInit[];
n = TNum[42];
<|"Tag" -> TTermTag[n], "Ext" -> TTermExt[n], "Val" -> TTermVal[n]|>
```

Decomposed into the four fields, the word looks like this:

```wolfram
TBitField[{
    {"SUB", 1, 0},
    {"TAG", 7, TTermTag[n] <> " (7)"},
    {"EXT", 18, TTermExt[n]},
    {"VAL", 38, TTermVal[n]}}]
```

The heap didn't grow:

```wolfram
THeapSnapshot[]["HeapSize"]
```

Creating a second number does not grow the heap either — every `TNum` is a leaf:

```wolfram
m = TNum[7];
THeapSnapshot[]["HeapSize"]
```

---

## Hands-on: LAM allocates 2 slots

`TLam[x |-> body]` allocates two consecutive heap slots and returns a single `TTerm` handle tagged `Lam` with `VAL` = the heap base.

```wolfram
TInit[];
lam = TLam[x |-> x];
<|"Tag" -> TTermTag[lam], "Val" -> TTermVal[lam],
  "HeapSize" -> THeapSnapshot[]["HeapSize"]|>
```

Compare the two words side by side — the leaf versus the heap-owner. `NUM`'s `VAL` *is* the data; `LAM`'s `VAL` is a **pointer** (heap slot index):

```wolfram
Column[{
    Style["NUM 42  (leaf — VAL carries the value)", Bold],
    TBitField[{{"SUB", 1, 0}, {"TAG", 7, "NUM"},
               {"EXT", 18, 0}, {"VAL", 38, 42}}, ImageSize -> 560],
    Spacer[{0, 10}],
    Style["LAM  (owner — VAL is a heap pointer)", Bold],
    TBitField[{{"SUB", 1, 0}, {"TAG", 7, "LAM"},
               {"EXT", 18, 0},
               {"VAL", 38, "\[RightArrow] heap[" <> ToString[TTermVal[lam]] <> "]"}},
              ImageSize -> 560]
}, Alignment -> Left]
```

The LAM owns two slots. A compact strip view of the heap:

```wolfram
THeapStrip[]
```

Slot 1 is the **binder** — owned by the LAM, initialized as a self-referring `Var` placeholder ("no value yet"). Slot 2 is the **body** — for `λx. x` the body is a `Var` reference pointing back to slot 1, which is why `val=1`.

A richer body grows the heap further. `x + 1` needs an OP2 cell plus slots for its two arguments:

```wolfram
TInit[];
TLam[x |-> TOp2["Add", x, TNum[1]]];
THeapStrip[]
```

Reading left to right:

- **slot 1** — LAM's binder (holds a `Var` placeholder, overwritten on beta)
- **slot 2** — LAM's body: an `OP2`, `val=3` means its args start at slot 3
- **slot 3** — OP2's arg0: a `Var` reference back to slot 1 (the `x` occurrence)
- **slot 4** — OP2's arg1: a `Num 1` (the constant, sitting inline)

No slot here was allocated *for* a VAR or a NUM — the LAM allocated 2, the OP2 allocated 2, and their contents include VAR/NUM words inline. The same thing in the interaction-net view:

```wolfram
TInit[];
lamBody = TLam[x |-> TOp2["Add", x, TNum[1]]];
TINetGraph[lamBody, ImageSize -> 500]
```

The triangle is the LAM; its `body` edge lands on the rectangular OP2; the OP2's two ports (`a`, `b`) point at a VAR oval (back to the binder) and the NUM rectangle (the constant). The `@n` tail on each compound matches the `slot` labels in the heap strip one-for-one.

---

## Heap Snapshot Object

`THeapSnapshot[]` returns a `THeap` association with summary-box formatting. Five counters come from the C side: heap position, interaction count, tensor count, next SUP label, and the definition count.

```wolfram
TInit[];
TLam[x |-> TOp2["Add", x, TNum[1]]];
h = THeapSnapshot[];
h
```

Properties are indexed like an association:

```wolfram
{h["HeapSize"], h["InteractionCount"], h["TensorCount"]}
```

`TensorCount` is `0` because `TAG_TEN` — a materialized GPU buffer — hasn't entered the picture yet. Chapter 8 picks that up.

---

## Visualizing Terms

`TINetGraph[t]` renders the subgraph rooted at `t` — per-tag shapes, port labels, `@slot` heap-location tails, and the next active pair highlighted in red when one exists. It reads the same live heap that `thvm_dump_dot` walks from C, so the picture matches what the C debugger produces.

```wolfram
TInit[];
TINetGraph[TNum[42], ImageSize -> 260]
```

A standalone `NUM` is one leaf node with no edges — it owns no heap slots. Applying the lambda to an argument wires the pieces together:

```wolfram
TInit[];
app = TApp[TLam[x |-> TOp2["Add", x, TNum[1]]], TNum[41]];
TINetGraph[app, ImageSize -> 560]
```

APP sits at the root (`out` pins to the free port above it). Its two slots (`fun`, `arg`) feed the LAM and the constant `41`. The LAM's `body` edge goes to the OP2, whose `a`/`b` slots hold the VAR reference and the constant `1`. Each compound carries its heap location in the `@n` tail label, so the graph lines up 1-to-1 with the `THeapStrip[]` view above.

---

## Summary

- A term is a 64-bit word — `[SUB][TAG][EXT][VAL]` — `TBitField` renders any field decomposition for teaching.
- **Non-owning** words (NUM, ERA, VAR, REF) don't allocate: NUM/ERA are self-contained; VAR/REF are references whose `VAL` points to an existing slot.
- **Heap-owning** compound nodes (LAM, APP, SUP, OP2) allocate 2 consecutive slots and stash the base in `VAL`.
- The contents of a compound's slots are themselves 64-bit words — they may be leaves inline or references into the heap.
- `THeapSnapshot[]` returns a `THeap` with heap size, interaction count, tensor count, next label, and definition count.
- `THeapRead[loc]` reads one slot; `THeapReadRange[lo, count]` bulk-reads a run; `THeapStrip[]` / `THeapStrip[lo, hi]` draws the slots as a horizontal strip.
- `TINetGraph[t]` — the canonical interaction-net view; `TShapeLegend[]` is the companion that names its shape alphabet.
