# TinyHVM -- Interaction Net Tensor Engine for Wolfram Language

## Overview

TinyHVM is a lazy interaction-net-based tensor computation engine with Metal GPU acceleration. Unlike traditional tensor libraries, it is built on HVM4-style interaction combinators -- lambda calculus primitives compose with tensor operations through declarative interaction rules.

Everything is lazy: operations build a computation graph on the heap. Nothing executes until `TReduce` triggers the interaction net reducer.

Key features:

- Lazy computation graphs with automatic fusion
- Metal GPU backend (Apple Silicon)
- Autograd via interaction rules
- Lambda calculus primitives (unique vs PyTorch/MLX)
- `TOp["Name"][args...]` dispatch pattern
- Neural network layer API with `TNet` sequential composition

## Setup

```wolfram
PacletDirectoryLoad[FileNameJoin[{NotebookDirectory[]}]];
Get["TinyHVM`"]
```

Initialize the Metal GPU backend:

```wolfram
TInit["metal"]
```

## Quick Start

Create GPU tensors from Wolfram Language lists:

```wolfram
a = TCreate[{{1., 2.}, {3., 4.}}];
TDimensions[a]
```

Read data back from GPU:

```wolfram
Normal[TGet[a]]
```

Elementwise operations are lazy -- they return `TTerm` handles. Force with `TReduce`:

```wolfram
b = TCreate[{{10., 20.}, {30., 40.}}];
Normal[TGet[TReduce[TOp["Add"][a, b]]]]
```

Natural WL syntax via UpValues -- `+`, `-`, `.` (Dot), `Sqrt`, `Exp`, `Log`:

```wolfram
Normal[TGet[TReduce[a + b]]]
```

```wolfram
eye = TCreate[IdentityMatrix[2]];
Normal[TGet[TReduce[a . eye]]]
```

## Neural Network Layers

TinyHVM provides composable layer constructors that automatically create weights and handle forward passes.

### Layer constructors

```wolfram
SeedRandom[42];
lin = TLinearLayer[4, 2]
```

```wolfram
conv = TConvLayer[1, 8, 3]
```

```wolfram
TParams[conv]
```

### TNet -- sequential composition

```wolfram
TFree[];
TInit["metal"];
SeedRandom[42];
net = TNet[{
    TConvLayer[1, 8, 3],
    TActivation["Relu"],
    TFlattenLayer[],
    TLinearLayer[8*26*26, 10]
}]
```

```wolfram
Length[TParams[net]]
```

Test the forward pass with a dummy batch:

```wolfram
xDummy = TCreate[ConstantArray[0.5, 2*1*28*28], {2, 1, 28, 28}];
TSetRequiresGrad[xDummy];
logits = TForward[net, xDummy];
TDimensions[logits]
```

```wolfram
Normal[TGet[TReduce[logits]]]
```

### WL Layer Interop

The same network can be built using standard Wolfram Language layer constructors. Input shape `{1, 28, 28}` drives automatic shape inference -- input channels, flatten size, and linear input features are all computed:

```wolfram
net2 = TNet[{
    ConvolutionLayer[8, {3, 3}],
    ElementwiseLayer[Ramp],
    FlattenLayer[],
    LinearLayer[10]
}, {1, 28, 28}]
```

```wolfram
Length[TParams[net2]]
```

## Training on MNIST

### Load MNIST

```wolfram
mnist = ResourceData["MNIST"];
trainData = mnist[Select[#Split == "Train" &]];
trainImages = Normal[trainData[All, "Image"]];
trainLabels = Normal[trainData[All, "Label"]];
Length[trainImages]
```

### Prepare data

```wolfram
imageToArray[img_] := Flatten[ImageData[ImageResize[img, {28, 28}], "Real32"]];
labelToInt[lbl_] := lbl /. Thread[Range[0, 9] -> Range[0, 9]]
```

### Build model

```wolfram
TFree[];
TInit["metal"];
SeedRandom[42];
net = TNet[{
    TConvLayer[1, 8, 3],
    TActivation["Relu"],
    TFlattenLayer[],
    TLinearLayer[8*26*26, 10]
}];
params = TParams[net];
nWeights = TTensorCount[]
```

### Training loop

`TTrainStep` handles gradient computation (via `TGradMulti`), SGD updates (via `TAssign`), and reduction in a single call:

```wolfram
bs = 32;
nSteps = 20;

losses = Table[
    Module[{bi, batchImgs, batchLbls, x, oh, logits, loss, lossVal},
        bi = Mod[step - 1, Floor[Length[trainImages]/bs]];
        batchImgs = trainImages[[bi*bs + 1 ;; (bi + 1)*bs]];
        batchLbls = trainLabels[[bi*bs + 1 ;; (bi + 1)*bs]];

        x = TCreate[Flatten[imageToArray /@ batchImgs], {bs, 1, 28, 28}];
        TSetRequiresGrad[x];

        oh = TCreate[
            Flatten[Table[
                ReplacePart[ConstantArray[0., 10], batchLbls[[i]] + 1 -> 1.],
                {i, bs}
            ]], {bs, 10}];

        logits = TForward[net, x];
        loss = TCrossEntropyLoss[logits, oh];
        lossVal = TTrainStep[params, loss, 0.01];
        TReset[nWeights];

        If[Mod[step, 5] == 1, Print["  step ", step, ": loss=", NumberForm[lossVal, 4]]];
        lossVal
    ],
    {step, nSteps}
];
ListLinePlot[losses, PlotLabel -> "Training Loss", AxesLabel -> {"Step", "Loss"},
    PlotRange -> {0, All}, ImageSize -> 400]
```

### Evaluate on test set

```wolfram
testData = mnist[Select[#Split == "Test" &]];
testImgs = Normal[testData[All, "Image"]][[1 ;; 320]];
testLbls = Normal[testData[All, "Label"]][[1 ;; 320]];
tbs = 32;
correct = 0;
Do[
    Module[{bImgs, bLbls, x, logits, preds, predList, evalKeep},
        evalKeep = TTensorCount[];
        bImgs = testImgs[[(b-1)*tbs + 1 ;; b*tbs]];
        bLbls = testLbls[[(b-1)*tbs + 1 ;; b*tbs]];
        x = TCreate[Flatten[imageToArray /@ bImgs], {tbs, 1, 28, 28}];
        logits = TForward[net, x];
        preds = Normal[TGet[TReduce[logits]]];
        predList = First[Ordering[#, -1]] - 1 & /@ Partition[preds, 10];
        correct += Count[Thread[predList == bLbls], True];
        TReset[evalKeep];
    ],
    {b, Length[testImgs]/tbs}
];
Print["Test accuracy: ", NumberForm[100. correct / Length[testImgs], 4], "% (", correct, "/", Length[testImgs], ")"]
```

## Computation Graph Visualization

Enable graph tracing to record the computation DAG:

```wolfram
TFree[];
TInit["metal"];
$TGraphTrace = True;
TGraphReset[]
```

Build a computation and visualize its graph:

```wolfram
a = TCreate[{{1., 2.}, {3., 4.}}];
b = TCreate[{{5., 6.}, {7., 8.}}];
c = TOp["Relu"][a . b + a];
TComputationGraph[c]
```

A more complex example -- softmax:

```wolfram
TGraphReset[];
x = TCreate[{{1., 2., 3.}, {4., 5., 6.}}];
xmax = TOp["Expand"][TOp["RMax"][x], {2, 3}];
shifted = TOp["Sub"][x, xmax];
e = TOp["Exp"][shifted];
esum = TOp["Expand"][TOp["Sum"][e, {1}], {2, 3}];
softmax = TOp["Div"][e, esum];
TComputationGraph[softmax, ImageSize -> Large]
```

```wolfram
$TGraphTrace = False
```

## Interaction Net Primitives

These are what make TinyHVM fundamentally different from PyTorch or MLX. The computation graph is an interaction net with lambda calculus nodes:

- `TLam` -- lambda abstraction
- `TApp` -- function application
- `TSup` -- superposition (optimal sharing)
- `TDup` -- linear duplication
- `TDefine` / `TRef` -- named definitions (recursive)
- `TIfz` -- if-zero branch (recursion control)

```wolfram
{lam, var} = TLam[a];
{lam, var}
```

```wolfram
app = TApp[lam, b];
app
```

## Single-Inet Recursive Training

The killer feature: the entire training loop can be expressed as a single interaction net program. One `TReduce` call runs N steps of forward + backward + weight update.

This uses `TDefine`/`TRef` for recursion and `TIfz` for the loop counter:

```wolfram
TFree[];
TInit["metal"];
SeedRandom[42]
```

```wolfram
(* 2-layer MLP: XOR problem, 4 samples, 4x4 -> 4x1 *)
x = TCreate[{{1., 0., 1., 0.}, {0., 1., 0., 1.}, {1., 1., 0., 0.}, {0., 0., 1., 1.}}];
TSetRequiresGrad[x];
y = TCreate[{{1.}, {0.}, {1.}, {0.}}];
lr = TCreate[{0.1}, {1}];

w1 = TCreate[RandomReal[{-0.5, 0.5}, 16], {4, 4}]; TSetRequiresGrad[w1];
b1 = TCreate[ConstantArray[0., 4], {1, 4}]; TSetRequiresGrad[b1];
w2 = TCreate[RandomReal[{-0.5, 0.5}, 4], {4, 1}]; TSetRequiresGrad[w2];
b2 = TCreate[ConstantArray[0., 1], {1, 1}]; TSetRequiresGrad[b2]
```

Build the forward pass, loss, gradients, and update chain as a single lazy graph:

```wolfram
(* Forward: relu(X @ W1 + B1) @ W2 + B2 *)
h = TOp["Relu"][x . w1 + TOp["Expand"][b1, {4, 4}]];
out = h . w2 + TOp["Expand"][b2, {4, 1}];

(* MSE loss: mean((out - y)^2) *)
diff = out - y;
loss = TOp["Mul"][
    TOp["Sum"][TOp["Sum"][TOp["Mul"][diff, diff], {1}], {0}],
    TCreate[{0.25}, {1}]
];
loss = TOp["Reshape"][loss, {1}];

(* Gradients *)
gW1 = TGrad[loss, w1]; gB1 = TGrad[loss, b1];
gW2 = TGrad[loss, w2]; gB2 = TGrad[loss, b2];

(* Log loss (prints during reduce) + SGD assigns *)
logLoss = TLogPrint[loss];
aW1 = TAssign[w1, w1 - lr * gW1];
aB1 = TAssign[b1, b1 - lr * gB1];
aW2 = TAssign[w2, w2 - lr * gW2];
aB2 = TAssign[b2, b2 - lr * gB2]
```

Wire into a recursive inet program:

```wolfram
(* Pre-allocate def slot for self-reference *)
trainId = TAllocDef[];

(* Inner lambda: body = chain of assigns + recursive call *)
{succLam, mVar} = TLamOpen[];
rec = TApp[TRef[trainId], mVar];
chain = TApp[logLoss, TApp[aW1, TApp[aB1, TApp[aW2, TApp[aB2, rec]]]]];
TLamSetBody[succLam, chain];

(* Outer lambda: IFZ(counter, done, succLam) *)
{defLam, nVar} = TLamOpen[];
TLamSetBody[defLam, TIfz[nVar, TCreate[{0.}, {1}], succLam]];
TSetDef[trainId, defLam]
```

Run 20 training steps with a **single** `TReduce`:

```wolfram
counter = TCreate[{20.}, {1}];
TReduce[TApp[TRef[trainId], counter]]
```

Verify the model learned -- forward pass with updated weights:

```wolfram
outPost = TReduce[x . w1 + TOp["Expand"][b1, {4, 4}]];
Normal[TGet[outPost]]
```

## Profiling

Enable profiling and inspect UOp-level performance:

```wolfram
TFree[];
TInit["metal"];
TProfileEnable[];
TProfileReset[]
```

```wolfram
a = TCreate[RandomReal[1., 1000*1000], {1000, 1000}];
b = TCreate[RandomReal[1., 1000*1000], {1000, 1000}];
TSetRequiresGrad[a]; TSetRequiresGrad[b];
c = TOp["Relu"][a . b + a];
g = TGrad[c, a];
TReduce[g];
TProfileSummary[]
```

Collect per-step profile snapshots during training for timeline visualization:

```wolfram
SeedRandom[42];
w = TCreate[RandomReal[{-0.1, 0.1}, 100*10], {100, 10}]; TSetRequiresGrad[w];
nW = TTensorCount[];
snapshots = Table[
    Module[{xx, yy, logits, loss},
        TProfileReset[];
        xx = TCreate[RandomReal[1., 32*100], {32, 100}]; TSetRequiresGrad[xx];
        yy = TCreate[ConstantArray[0., 32*10], {32, 10}];
        logits = xx . w;
        loss = TOp["Mul"][TOp["Sum"][TOp["Sum"][TOp["Mul"][logits, logits], {1}], {0}], TCreate[{1./32}, {1}]];
        TTrainStep[{w}, loss, 0.01];
        TReset[nW];
        TProfileData[]
    ],
    {step, 10}
];
TProfileTimeline[snapshots]
```

## Key Concepts

### Laziness

Every `TOp`, `TConv2d`, `TGrad`, etc. returns a `TTerm` immediately -- a node on the interaction net heap. No GPU work happens. Only `TReduce` (or `TGet`, which calls reduce internally) triggers the interaction net reducer, which walks the graph and dispatches Metal kernels.

### TReset Lifecycle

In training loops, weight tensors occupy IDs `[0, keep)`. After each step, `TReset[keep]` frees all activation/gradient tensors above that index and resets the heap. Weight tensors survive.

### Single-Inet Training

Unlike PyTorch's imperative loop, TinyHVM can encode the entire training loop as a recursive interaction net: `TDefine` registers a body, `TRef` clones it on each unfold, `TIfz` tests the counter. One `TReduce` drives everything -- forward, backward, and weight updates for N steps.

### The ERA Sentinel

Unary ops like `TOp["Neg"]` internally pass `ERA` (the eraser term) as the second argument. This is transparent at the WL level.

## Cleanup

```wolfram
TFree[]
```
