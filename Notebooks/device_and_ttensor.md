# TTensor, Devices & Interactive Reduction

This notebook demonstrates the TinyHVM v2 API: per-tensor device tracking, the `TTensor` head with rich property access, interaction net graph visualization from the C heap, and single-step reduction for interactive debugging.

## Setup

```wolfram
PacletDirectoryLoad[FileNameJoin[{NotebookDirectory[]}]];
Get["TinyHVM`"]
```

Initialize with Metal GPU backend:

```wolfram
TInit["metal"]
```

## TTensor Basics

`TCreate` now returns `TTensor` (not `TTerm`). Tensors carry shape, device, and view metadata:

```wolfram
t = TCreate[{{1., 2., 3.}, {4., 5., 6.}}, {2, 3}]
```

Property access via double-bracket indexing:

```wolfram
t[["Shape"]]
```

```wolfram
t[["Device"]]
```

```wolfram
t[["View"]]
```

The view shows strides, offset, contiguity, and mask info:

```wolfram
t[["Contiguous"]]
```

## Lazy vs Materialized

Before reduction, operations produce lazy `TTensor` nodes:

```wolfram
u = TCreate[{{10., 20., 30.}, {40., 50., 60.}}, {2, 3}];
lazy = TOp["Add"][t, u]
```

```wolfram
lazy[["Lazy"]]
```

```wolfram
lazy[["Op"]]
```

After reduction, the tensor is materialized:

```wolfram
mat = TReduce[lazy]
```

```wolfram
mat[["Materialized"]]
```

```wolfram
Normal[TGet[mat]]
```

## Natural Arithmetic

TTensor supports natural Wolfram Language syntax via UpValues:

```wolfram
a = TCreate[{1., 2., 3., 4.}, {2, 2}];
b = TCreate[{10., 20., 30., 40.}, {2, 2}];
Normal[TGet[TReduce[a + b]]]
```

```wolfram
Normal[TGet[TReduce[a - b]]]
```

Matrix multiplication via `Dot`:

```wolfram
Normal[TGet[TReduce[a . b]]]
```

Unary operations:

```wolfram
Normal[TGet[TReduce[Sqrt[a]]]]
```

## Device Queries

Every tensor knows which device it lives on:

```wolfram
gpu = TCreate[{1., 2., 3.}, {3}];
TDevice[gpu]
```

## Interaction Net Graph

`TINetGraph` walks the C heap and renders the actual interaction net structure. Build a lazy chain and visualize:

```wolfram
x = TCreate[{1., 2.}, {2}];
y = TCreate[{3., 4.}, {2}];
chain = TOp["Mul"][TOp["Add"][x, y], x];
TINetGraph[chain]
```

After reduction, the graph collapses to a single tensor node:

```wolfram
result = TReduce[chain];
TINetGraph[result]
```

## Single-Step Reduction

Enable tracing and step through interactions one at a time:

```wolfram
p = TCreate[{1., 2.}, {2}];
q = TCreate[{3., 4.}, {2}];
expr = TOp["Add"][p, q]
```

```wolfram
TTraceEnable[];
TTraceClear[];
{stepped, nSteps} = TReduceSteps[expr, 1]
```

View the interaction trace:

```wolfram
TTrace[]
```

Continue reducing:

```wolfram
{final, moreSteps} = TReduceSteps[stepped, 100];
Normal[TGet[final]]
```

```wolfram
TTraceDisable[]
```

## TTerm vs TTensor

`TTerm` is for interaction net primitives (lambda, application, superposition). `TTensor` is for tensor operations. They interoperate via `ToTTerm`/`ToTTensor`:

```wolfram
tensor = TCreate[{42.}, {1}];
term = ToTTerm[tensor]
```

```wolfram
back = ToTTensor[term];
Normal[TGet[back]]
```

Lambda calculus with tensors -- `(lam x. x)` applied to a tensor:

```wolfram
{lam, var} = TLam[ToTTerm[tensor]];
app = TApp[lam, ToTTerm[TCreate[{99.}, {1}]]];
Normal[TGet[TReduce[ToTTensor[app]]]]
```

## Cleanup

```wolfram
TFree[]
```
