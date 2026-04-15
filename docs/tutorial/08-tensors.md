# Chapter 8: Tensors — TEN, TOP, and GPU Computation

TinyHVM bridges interaction nets and GPU tensor computation. Tensors live inside the interaction net as nodes — created lazily, fused automatically, and dispatched to Metal/CPU backends.

---

## TEN: Materialized Tensors

`TCreate[data, shape]` creates a GPU tensor and returns a TAG_TEN node:

```wolfram
TInit[];
t = TCreate[{1, 2, 3, 4}, {2, 2}];
TTermTag[t]            (* "Ten" *)
TDimensions[t]         (* {2, 2} *)
TDevice[t]             (* "metal" or "cpu" *)
TGet[t]                (* NumericArray: {{1., 2.}, {3., 4.}} *)
```

TAG_TEN stores: `EXT = dtype` (e.g., float32), `VAL = tensor_id` (index into the tensor metadata array).

---

## TOP: Lazy Operations

Tensor operations don't execute immediately — they create TAG_TOP nodes:

```wolfram
TInit[];
a = TCreate[{1, 2, 3, 4}, {2, 2}];
b = TCreate[{10, 20, 30, 40}, {2, 2}];
c = TOp["Add"][a, b];
TTermTag[c]            (* "Top" — not computed yet *)
TTermExt[c]            (* 9 — UOP_ADD code *)
```

The computation graph is built lazily. Nothing hits the GPU until you reduce:

```wolfram
result = TReduce[c];
TTermTag[result]       (* "Ten" — now materialized *)
TGet[result]           (* {{11., 22.}, {33., 44.}} *)
```

---

## UOp Codes

The EXT field of a TOP node encodes which operation it represents:

| Code | Name | Type |
|------|------|------|
| 3 | Neg | Unary |
| 4 | Exp | Unary |
| 5 | Log | Unary |
| 6 | Relu | Unary |
| 8 | Sqrt | Unary |
| 9 | Add | Binary |
| 10 | Mul | Binary |
| 11 | Div | Binary |
| 14 | Sub | Binary |
| 15 | Sum | Reduce |
| 17 | MatMul | Special |
| 18 | Reshape | Movement |
| 19 | Permute | Movement |
| 20 | Expand | Movement |

---

## Natural Syntax

TTensor has UpValues for natural WL syntax:

```wolfram
TInit[];
a = TCreate[{1, 2, 3, 4}, {2, 2}];
b = TCreate[{10, 20, 30, 40}, {2, 2}];

c = a + b;         (* TOp["Add"] *)
d = a * b;         (* TOp["Mul"] *)
e = a . b;         (* TOp["MatMul"] *)
f = Sqrt[a];       (* TOp["Sqrt"] *)
```

---

## Computation Graphs in the Interaction Net

The lazy TOP nodes form a computation graph. Visualize it:

```wolfram
TInit[];
a = TCreate[{1, 2, 3}, {3}];
b = TCreate[{4, 5, 6}, {3}];
c = a + b;
d = c * a;
e = TOp["Sum"][d, {0}];

TINetGraph[e]
```

You'll see TOP nodes (orange triangles) connected to TEN nodes (blue squares), forming the computation DAG.

---

## Movement Operations

Operations like Reshape, Permute, and Expand don't copy data — they change the **view**:

```wolfram
TInit[];
t = TCreate[Range[12] // N, {3, 4}];
r = TOp["Reshape"][t, {4, 3}];
p = TOp["Permute"][t, {1, 0}];
e = TOp["Expand"][TCreate[{1, 2, 3} // N, {3, 1}], {3, 4}];

TView[TReduce[r]]     (* shape {4,3}, same underlying buffer *)
```

---

## Autograd: Gradients via Interaction Rules

TinyHVM's autograd works through the interaction net itself. Mark a tensor for gradient tracking, then use `TGrad`:

```wolfram
TInit[];
x = TCreate[{1, 2, 3, 4} // N, {2, 2}];
TSetRequiresGrad[x];

y = TOp["Sum"][x * x, {0, 1}];    (* sum of squares *)
dy = TGrad[y, x];
TGet[TReduce[dy]]                 (* 2x = {2, 4, 6, 8} *)
```

`TGrad[y, x]` inserts GRAD nodes into the interaction net. When reduced, they propagate the chain rule backward through the computation graph — all using the same annihilation/commutation rules.

Visualize the gradient graph:

```wolfram
TINetGraph[dy]    (* GRAD nodes wired through the forward graph *)
```

---

## Tensor + Lambda Interaction

Tensors live in the same interaction net as lambdas. You can mix them:

```wolfram
TInit[];
(* A lambda that takes a tensor and computes a loss *)
lossFn = TLam[x |-> TOp["Sum"][x * x, {0}]];

x = TCreate[{1, 2, 3} // N, {3}];
loss = TReduce[TApp[lossFn, x]];
TGet[loss]    (* {14.} — sum of squares *)
```

When APP meets TEN (function is a tensor, not a lambda), the tensor is released and the argument passes through — this is the **sequencing** interaction.

---

## Forward + Backward Pass

```wolfram
TInit[];
(* Simple linear model: y = w * x + b *)
w = TCreate[{0.5, -0.3} // N, {2}]; TSetRequiresGrad[w];
b = TCreate[{0.1} // N, {1}]; TSetRequiresGrad[b];
x = TCreate[{2.0, 3.0} // N, {2}];

(* Forward *)
pred = TOp["Sum"][w * x, {0}] + b;
target = TCreate[{1.0} // N, {1}];
loss = TOp["Sum"][(pred - target) * (pred - target), {0}];

(* Backward *)
dw = TGrad[loss, w];
db = TGrad[loss, b];

TGet[TReduce[loss]]    (* loss value *)
TGet[TReduce[dw]]      (* gradient w.r.t. w *)
TGet[TReduce[db]]      (* gradient w.r.t. b *)
```

---

## Summary

- **TAG_TEN** (blue): materialized GPU tensor — `VAL = tensor_id`, `EXT = dtype`
- **TAG_TOP** (orange): lazy operation — `EXT = UOp code`, `VAL = heap loc` with args
- Operations build a lazy computation graph; `TReduce` materializes it
- Movement ops (Reshape, Permute, Expand) are zero-copy view changes
- `TGrad[y, x]` inserts GRAD interaction rules for automatic differentiation
- Tensors and lambdas coexist in the same interaction net — same rules, same heap
- The interaction net IS the computation graph — there's no separate representation
