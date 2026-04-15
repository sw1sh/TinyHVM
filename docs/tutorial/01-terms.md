# Chapter 1: Terms

Everything in TinyHVM is a **term** — a single 64-bit word that encodes a node in the interaction net.

---

## The 64-Bit Encoding

Every term packs four fields into 64 bits:

```
 63      62..56      55..38         37..0
[SUB:1]  [TAG:7]     [EXT:18]       [VAL:38]
```

| Field | Bits | Purpose |
|-------|------|---------|
| **SUB** | 1 | Substitution flag — marks that a VAR has been written to |
| **TAG** | 7 | Node type (APP, LAM, SUP, TEN, etc.) — 128 possible tags |
| **EXT** | 18 | Extension data — meaning depends on tag (label for SUP/DUP, UOp code for TOP, dtype for TEN) |
| **VAL** | 38 | Value/pointer — typically a heap location or tensor ID (256 GB address space) |

This encoding means every node fits in a single machine word. No pointers, no heap headers, no vtables — just bit packing.

---

## Tag Table

The TAG field identifies what kind of node this is:

| Code | Name | Description |
|------|------|-------------|
| 0 | APP | Application — `(f x)` |
| 1 | LAM | Lambda — `λx. body` |
| 2 | VAR | Variable — bound reference |
| 3 | SUP | Superposition — `{a, b}` |
| 4 | DP0 | DUP projection 0 |
| 5 | DP1 | DUP projection 1 |
| 6 | ERA | Eraser — discards values |
| 7 | NUM | Number — 32-bit integer |
| 8 | REF | Reference — named definition |
| 9 | OP2 | Binary operation — `Op(x, y)` |
| 10 | TEN | Tensor — materialized GPU data |
| 11 | TOP | Tensor operation — lazy compute node |
| 12 | CTR | Constructor — multi-arity node |
| 13 | BRI | Bridge — `θx. body` (dual of lambda) |
| 14 | ANN | Annotation — `{term : type}` |

---

## Leaves vs Compound Nodes

Some terms are **leaves** — their entire state is packed into the 64-bit word itself:
- **NUM**: the integer value lives in VAL
- **ERA**: no data at all
- **VAR**: points to a binding location
- **REF**: points to a definition slot

**Compound** nodes allocate words on the heap for their children:
- **LAM**: 2 heap words (var + body)
- **APP**: 2 heap words (fun + arg)
- **SUP**: 2 heap words (left + right)
- **OP2**: 2 heap words (arg0 + arg1)

---

## The Heap

The heap is a flat array of 64-bit words. Compound nodes allocate consecutive words:

```
heap:  [ word₀ ][ word₁ ][ word₂ ][ word₃ ] ...
         ↑                  ↑
         loc=0              loc=2
```

A lambda `λx. body` allocates 2 words: `heap[loc] = VAR`, `heap[loc+1] = body`. The lambda term itself is `TAG=LAM, VAL=loc`.

---

## Hands On

```wolfram
TInit[]
```

Create a leaf term — a number. NUM doesn't allocate heap words; the value lives inside the term:

```wolfram
n = TNum[42]
```

```wolfram
TTermTag[n]
```

The heap is empty because NUM is a leaf:

```wolfram
h = THeapSnapshot[];
h["HeapSize"]
```

Now create a compound term — a lambda allocates 2 heap words:

```wolfram
TInit[];
{lam, var} = TLam[x |-> x];
h = THeapSnapshot[];
h["HeapSize"]
```

Inspect what's on the heap:

```wolfram
THeapRead[0]
```

```wolfram
THeapRead[1]
```

The first word is the VAR (the binding site), the second is the body.

---

## Heap Snapshot

`THeapSnapshot[]` captures the full state as a `THeap` object with properties:

```wolfram
TInit[];
TLam[x |-> TOp2["Add", x, TNum[1]]];
h = THeapSnapshot[];
h
```

```wolfram
h["HeapSize"]
```

```wolfram
h["InteractionCount"]
```

```wolfram
h["TensorCount"]
```

---

## Visualizing Terms

```wolfram
TInit[];
TINetGraph[TNum[42]]
```

A NUM is a leaf — a single node, no edges.

```wolfram
TInit[];
app = TApp[TLam[x |-> x], TNum[42]];
TINetGraph[app]
```

A compound expression — APP connected to LAM connected to NUM.

---

## Summary

- A **term** is a 64-bit word: `[SUB][TAG][EXT][VAL]`
- The **TAG** identifies the node type (18 tags defined)
- **Leaves** (NUM, ERA, VAR, REF) carry their data in the term word itself
- **Compound nodes** (LAM, APP, SUP, OP2) allocate children on the heap
- `THeapSnapshot[]` captures the state as a `THeap` object
- `THeapRead[loc]` reads any heap location
- `TINetGraph[t]` draws the graph rooted at a term
