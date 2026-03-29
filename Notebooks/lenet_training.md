# LeNet on MNIST — TinyHVM vs Built-in

Train LeNet from scratch using TinyHVM's interaction net engine, then compare against Wolfram's pre-trained model. The entire training loop is embedded as a single recursive inet term — one TReduce call drives all steps.

## Setup

```wolfram
PacletDirectoryLoad[FileNameJoin[{NotebookDirectory[], "..", "wl"}]];
Get["TinyHVM`"]
TInit["metal"]
```

## Load MNIST Data

```wolfram
mnistTrain = ResourceData["MNIST", "TrainingData"];
mnistTest = ResourceData["MNIST", "TestData"];

trainImages = ImageData /@ mnistTrain[[All, 1]];
trainLabels = mnistTrain[[All, 2]];
testImages = ImageData /@ mnistTest[[All, 1]];
testLabels = mnistTest[[All, 2]];

Print["Train: ", Length[trainImages], "  Test: ", Length[testImages]]
```

## Compile LeNet Architecture

Convert the standard LeNet architecture to TinyHVM's TNet. This creates GPU tensors with random Xavier initialization — no training yet.

```wolfram
lenet = TCompileNet[
    NetChain[{
        ConvolutionLayer[6, {5, 5}],
        ElementwiseLayer[Ramp],
        PoolingLayer[{2, 2}, {2, 2}],
        ConvolutionLayer[16, {5, 5}],
        ElementwiseLayer[Ramp],
        PoolingLayer[{2, 2}, {2, 2}],
        FlattenLayer[],
        LinearLayer[120],
        ElementwiseLayer[Ramp],
        LinearLayer[84],
        ElementwiseLayer[Ramp],
        LinearLayer[10]
    }],
    {1, 28, 28}
]
```

```wolfram
(* Inspect parameters *)
Print["Parameters: ", Length[TParams[lenet]]]
Print["Total weights: ", Total[Times @@@ (TDimensions /@ TParams[lenet])]]
```

## Build Single-Inet Training Term

The entire training loop — forward pass, cross-entropy loss, backpropagation, and SGD weight updates for every step — is compiled into one recursive interaction net term. A single TReduce call executes all steps.

```wolfram
(* Prepare a training batch *)
bs = 64;
xBatch = TCreate[Flatten[N[trainImages[[1 ;; bs]]]], {bs, 1, 28, 28}];
yOneHot = TCreate[
    Flatten[N[IdentityMatrix[10][[trainLabels[[1 ;; bs]] + 1]]]],
    {bs, 10}]
```

```wolfram
(* Build the recursive training term: 50 steps on this batch *)
nSteps = 50;
trainTerm = TBuildTrainLoop[lenet, xBatch, yOneHot, nSteps, 0.01];
Print["Training term built \[LongDash] ", nSteps, " steps in one inet"]
```

## Train with Single TReduce

One call. The interaction net reducer unfolds the recursive definition, clones the forward graph each step, computes gradients, and updates weights — all driven by inet interaction rules.

```wolfram
(* Single reduce drives the entire training loop *)
AbsoluteTiming[TReduce[trainTerm]]
```

## Evaluate TinyHVM LeNet

```wolfram
(* Full MNIST training: per-step loop with batch cycling *)
nWeights = TTensorCount[];
nTrainSteps = 500;
lr = 0.01;
losses = {};

Do[
    bi = Mod[step - 1, Floor[Length[trainImages] / bs]];
    xb = TCreate[Flatten[N[trainImages[[bi*bs + 1 ;; (bi + 1)*bs]]]], {bs, 1, 28, 28}];
    yb = TCreate[Flatten[N[IdentityMatrix[10][[trainLabels[[bi*bs + 1 ;; (bi + 1)*bs]] + 1]]]], {bs, 10}];
    loss = TCrossEntropyLoss[TForward[lenet, xb], yb];
    lossVal = TTrainStep[TParams[lenet], loss, lr];
    AppendTo[losses, lossVal];
    TReset[nWeights];
    If[Mod[step, 100] == 0, Print["Step ", step, ": loss = ", NumberForm[lossVal, 4]]],
    {step, nTrainSteps}
]
```

```wolfram
ListLinePlot[losses, PlotLabel -> "Training Loss",
    FrameLabel -> {"Step", "Cross-Entropy Loss"}, PlotRange -> All,
    PlotStyle -> RGBColor[0.2, 0.4, 0.7]]
```

```wolfram
(* Evaluate on test set *)
nTest = Length[testImages];
nBatches = Floor[nTest / bs];
correct = 0;

Do[
    xb = TCreate[Flatten[N[testImages[[(b-1)*bs + 1 ;; b*bs]]]], {bs, 1, 28, 28}];
    logits = TForward[lenet, xb];
    probsRaw = Normal[TGet[TReduce[logits]]];
    preds = (Position[#, Max[#]][[1, 1]] - 1) & /@ Partition[probsRaw, 10];
    actual = testLabels[[(b-1)*bs + 1 ;; b*bs]];
    correct += Count[Thread[preds == actual], True];
    TReset[nWeights],
    {b, nBatches}
];
thvmAcc = N[100 correct / (nBatches * bs)];
Print["TinyHVM LeNet accuracy: ", NumberForm[thvmAcc, 4], "%"]
```

## Compare Against Pre-Trained LeNet

Load Wolfram's pre-trained LeNet and evaluate on the same test set for comparison.

```wolfram
trainedNet = NetModel["LeNet Trained on MNIST Data"]
```

```wolfram
(* Evaluate built-in trained model *)
testInputs = Image /@ testImages;
wlPreds = trainedNet[testInputs, "Decision"];
wlAcc = N[100 Count[Thread[wlPreds == testLabels], True] / Length[testLabels]];
Print["WL Pre-trained LeNet accuracy: ", NumberForm[wlAcc, 4], "%"]
```

```wolfram
(* Side-by-side comparison *)
Grid[{
    {"", "Accuracy"},
    {"TinyHVM LeNet (trained from scratch)", Row[{NumberForm[thvmAcc, 4], "%"}]},
    {"WL Pre-trained LeNet", Row[{NumberForm[wlAcc, 4], "%"}]}
}, Frame -> All, Alignment -> Left, Spacings -> {2, 1}]
```

## Import Pre-Trained Weights

We can also import the pre-trained weights directly into TinyHVM tensors, bypassing training entirely.

```wolfram
(* Import pre-trained weights into TinyHVM *)
lenetImported = TCompileNet[trainedNet, {1, 28, 28}, "ImportWeights" -> True]
```

```wolfram
(* Verify imported model matches *)
correct2 = 0;
nWeights2 = TTensorCount[];
Do[
    xb = TCreate[Flatten[N[testImages[[(b-1)*bs + 1 ;; b*bs]]]], {bs, 1, 28, 28}];
    logits = TForward[lenetImported, xb];
    probsRaw = Normal[TGet[TReduce[logits]]];
    preds = (Position[#, Max[#]][[1, 1]] - 1) & /@ Partition[probsRaw, 10];
    actual = testLabels[[(b-1)*bs + 1 ;; b*bs]];
    correct2 += Count[Thread[preds == actual], True];
    TReset[nWeights2],
    {b, nBatches}
];
importAcc = N[100 correct2 / (nBatches * bs)];
Print["Imported LeNet accuracy: ", NumberForm[importAcc, 4], "% (should match WL)"]
```

## Cleanup

```wolfram
TFree[]
```
