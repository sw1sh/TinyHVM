# TTensor, Devices & IC-Native Reduction

This notebook demonstrates the TinyHVM v2 API: per-tensor device tracking, the `TTensor` head, interaction net graph visualization from the C heap — including DUP/SUP sharing nodes — and single-step reduction for interactive debugging.

## Setup

```wolfram
PacletDirectoryLoad[FileNameJoin[{NotebookDirectory[], "..", "wl"}]];
Get["TinyHVM`"]
```

Initialize with Metal GPU backend:

```wolfram
TInit["metal"]
```

## TTensor Basics

`TCreate` returns `TTensor` (not `TTerm`). Tensors carry shape, device, and view metadata:

```wolfram
t = TCreate[{{1., 2., 3.}, {4., 5., 6.}}]
```

Property access via SubValues:

```wolfram
t["Shape"]
t["Device"]
t["View"]
```

## Lazy vs Materialized

Operations produce lazy `TTensor` nodes until reduced:

```wolfram
u = TCreate[{{10., 20., 30.}, {40., 50., 60.}}];
lazy = TOp["Add"][t, u]
```

```wolfram
{lazy["Lazy"], lazy["Op"]}
```

After reduction, the tensor is materialized:

```wolfram
mat = TReduce[lazy]
Normal[TGet[mat]]
```

## Natural Arithmetic

TTensor supports natural Wolfram Language syntax via UpValues:

```wolfram
a = TCreate[{{1., 2.}, {3., 4.}}];
b = TCreate[{{10., 20.}, {30., 40.}}];
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

## Interaction Net Graph — DUP Sharing

`TINetGraph` walks the C heap and renders the actual interaction net. When a tensor is used in multiple positions, `linear_use` creates a **DUP node** (1-slot sharing) with two projections: DP0 and DP1 (orange in the graph).

### Simple chain (no sharing)

Two distinct inputs — no DUP nodes appear:

```wolfram
x = TCreate[{1., 2.}, {2}];
y = TCreate[{3., 4.}, {2}];
simple = TOp["Add"][x, y];
TINetGraph[simple]
```

### Shared input — DUP appears

Using `x` in **both** positions of a binary op triggers `linear_use`. The graph renders DUP as a single orange node with **dp0**/**dp1** edge labels showing which port each parent connects through:

```wolfram
x = TCreate[{2., 3.}, {2}];
xSquared = TOp["Mul"][x, x];
TINetGraph[xSquared, ImageSize -> Large]
```

The graph shows:
- **Mul** at the root with two edges to a single **Dup** node — labeled dp0 and dp1
- The Dup node points to the shared **TEN** leaf (the `x` tensor)

### Deeper sharing — DUP through a chain

When a shared tensor feeds through intermediate ops, the DUP node appears at the fan-out point:

```wolfram
x = TCreate[{1., 2., 3., 4.}, {2, 2}];
(* x used in Add and also directly in Mul *)
sum = TOp["Add"][x, TCreate[{10., 10., 10., 10.}, {2, 2}]];
product = TOp["Mul"][sum, x];
TINetGraph[product, ImageSize -> Large]
```

### After reduction — graph collapses

Reducing resolves all DUP nodes. The graph collapses to a single materialized tensor:

```wolfram
result = TReduce[xSquared];
TINetGraph[result]
```

## Interaction Tracing

Enable tracing to see which interaction rules fire during reduction. This reveals the DUP ⊳ TEN annihilation when shared tensors resolve:

```wolfram
p = TCreate[{5., 6.}, {2}];
expr = TOp["Mul"][p, p]
```

Visualize the lazy graph — note the single DUP node with dp0/dp1 edge labels:

```wolfram
TINetGraph[expr, ImageSize -> Large]
```

Reduce with tracing enabled:

```wolfram
TTraceEnable[];
TTraceClear[];
result = TReduce[expr];
TTrace[]
```

The trace shows the DUP ⊳ TEN interaction firing — both DP0 and DP1 resolve to the same tensor leaf via the 1-slot cache:

```wolfram
Normal[TGet[result]]
```

```wolfram
TTraceDisable[];
```

### Interpreting the trace

Each trace record has:
- **BeforeTag/BeforeExt**: the node that entered the interaction (e.g., "Dp0", "Top")
- **AfterTag/AfterExt**: what it reduced to (e.g., "Ten", "Num")
- **RuleId**: internal rule identifier

Key DUP interactions:
- **DUP ⊳ TEN**: Both projections get the same tensor. The 1-slot cache means the value is computed once.
- **DUP ⊳ SUP (same label)**: Annihilation — the SUP dissolves, each projection takes its branch.
- **DUP ⊳ TOP**: Commutation — the DUP pushes through the operation, creating DUP'd copies of the op's children.

## Interaction Net Primitives

`TTerm` is for interaction net primitives (lambda, application, superposition, duplication). `TTensor` is for tensor operations. They interoperate via `ToTTerm`/`ToTTensor`.

### Manual DUP/SUP construction

You can build DUP-SUP pairs directly with `TDup` and `TSup`:

```wolfram
t = TCreate[{42., 99.}, {2}];
{dp0, dp1} = TDup[ToTTerm[t]]
```

Both projections reduce to the same tensor:

```wolfram
{Normal[TGet[TReduce[ToTTensor[dp0]]]], Normal[TGet[TReduce[ToTTensor[dp1]]]]}
```

### Lambda calculus with tensors

`TLam` accepts a pure function — the variable is created internally and passed to your function for use in the body:

Identity function `(λx. x)` applied to a tensor — β-reduction returns the argument:

```wolfram
id = TLam[Function[x, x]];
app = TApp[id, ToTTerm[TCreate[{42., 99.}, {2}]]];
Normal[TGet[TReduce[ToTTensor[app]]]]
```

A function that uses its argument in a computation — `(λx. x + 10)`:

```wolfram
addTen = TLam[Function[x, ToTTerm[TOp["Add"][ToTTensor[x], TCreate[{10., 10.}, {2}]]]]];
app2 = TApp[addTen, ToTTerm[TCreate[{1., 2.}, {2}]]];
Normal[TGet[TReduce[ToTTensor[app2]]]]
```

### Visualize inet primitives

```wolfram
f = TLam[Function[x, x]];
appViz = TApp[f, ToTTerm[TCreate[{2.}, {1}]]];
TINetGraph[ToTTensor[appViz]]
```

## Gradient Backward with DUP

When computing gradients, shared tensors create independent GRAD paths. Each path deposits its contribution via accumulative ASSIGN — no counters, no special GRAD3_FWD macro. The sharing structure emerges from the DUP/SUP pairs created during forward construction.

```wolfram
(* x used twice: loss = x * x, so dloss/dx = 2x *)
x = TCreate[{3., 4.}, {2}];
TSetRequiresGrad[x];
loss = TReduce[TOp["Mul"][x, x]];
grad = TReduce[TGrad[loss, x]];
Normal[TGet[grad]]
(* Expected: {6., 8.} = 2 * {3., 4.} *)
```

## Cleanup

```wolfram
TFree[]
```
