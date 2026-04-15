# TinyHVM Tutorial: Interaction Nets from First Principles

This tutorial builds your understanding of TinyHVM from the ground up — starting with how a single 64-bit word encodes a node, through the interaction rules that drive all computation, up to GPU tensor operations and automatic differentiation.

Each chapter has a companion Wolfram Notebook in `Notebooks/tutorial/` for interactive exploration.

## Chapters

1. **[Terms](01-terms.md)** — What is a term? The 64-bit encoding, tags, and the heap
2. **[Combinators](02-combinators.md)** — LAM, APP, ERA: the three atoms of computation
3. **[Interactions](03-interactions.md)** — Step-by-step execution, annihilation and commutation
4. **[Duplication](04-duplication.md)** — SUP/DUP, labels, and the sharing/copying duality
5. **[Numbers](05-numbers.md)** — NUM, OP2, and arithmetic in the interaction net
6. **[Recursion](06-recursion.md)** — REF, TDefine, and recursive definitions
7. **[Superposition](07-superposition.md)** — Parallel evaluation and program synthesis
8. **[Tensors](08-tensors.md)** — TEN, TOP, and GPU tensor computation in the interaction net

## Setup

Every chapter assumes you have the TinyHVM paclet loaded:

```wolfram
PacletDirectoryLoad[FileNameJoin[{NotebookDirectory[], "..", "..", "wl"}]];
Get["TinyHVM`"];
TInit[]
```

## Key Symbols

| Symbol | Purpose |
|--------|---------|
| `TTerm[id]` | An interaction net node |
| `TTensor[id]` | A tensor (materialized or lazy) |
| `THeap[<\|...\|>]` | Snapshot of the heap state |
| `TInteraction[<\|...\|>]` | A single interaction event |
| `TINetGraph[t]` | Visualize the interaction net as a graph |
| `TInteractionGraph[t]` | Graph with the next active pair highlighted |
| `TStepReduce[t]` | Execute one interaction, return `{result, interaction, heap}` |
| `THeapRead[loc]` | Read a raw term at a heap location |
| `THeapSnapshot[]` | Capture the current heap state |
