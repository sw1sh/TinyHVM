(* NetCompile.wl — Convert Wolfram Language neural networks to TinyHVM *)
(* Get'd from TinyHVM.wl inside Begin["`Private`"]. All public symbols declared there. *)

(* ── Extract layer list from WL net ──────────────────────────────── *)

extractLayerSpecs[net_NetChain] := Normal[net];
extractLayerSpecs[net_NetGraph] := Normal[net];
extractLayerSpecs[specs_List] := specs;

(* ── TCompileNet: WL net → TNet (fresh random weights) ──────────── *)

TCompileNet[net_, inputShape_List] := Block[{specs},
    specs = extractLayerSpecs[net];
    specs = Select[specs, !MatchQ[#, _SoftmaxLayer] &];
    TNet[specs, inputShape]
];

(* ── TCompileNet with weight import from trained net ─────────────── *)

TCompileNet[net_, inputShape_List, "ImportWeights" -> True] := Block[{
    layerList, layers = {}, shape = inputShape, result, layer, i
},
    layerList = Normal[net];
    Do[
        layer = layerList[[i]];
        If[MatchQ[layer, _SoftmaxLayer], Continue[]];
        result = compileLayerSpecImport[layer, shape, net, i];
        AppendTo[layers, result[[1]]];
        shape = result[[2]],
        {i, Length[layerList]}
    ];
    TNet[layers]
];

(* ── Per-layer weight import ─────────────────────────────────────── *)

compileLayerSpecImport[layer_ConvolutionLayer, {inC_, h_, w_}, net_, key_] :=
Block[{outC, k, stride, pad, wData, bData, tW, tB, pp, newH, newW},
    outC = layer["OutputChannels"];
    k = layer["KernelSize"];
    stride = layer["Stride"];
    pad = layer["PaddingSize"];

    wData = Normal[NetExtract[net, {key, "Weights"}]];
    bData = Normal[NetExtract[net, {key, "Biases"}]];

    tW = TCreate[Flatten[N[wData]], {outC, inC, k[[1]], k[[2]]}];
    tB = TCreate[Flatten[N[bData]], {outC}];
    TSetRequiresGrad[tW]; TSetRequiresGrad[tB];

    pp = If[MatrixQ[pad],
        {pad[[1,1]], pad[[1,2]], pad[[2,1]], pad[[2,2]]},
        {0, 0, 0, 0}];
    newH = Floor[(h + pp[[1]] + pp[[2]] - k[[1]]) / stride[[1]]] + 1;
    newW = Floor[(w + pp[[3]] + pp[[4]] - k[[2]]) / stride[[2]]] + 1;

    With[{ww = tW, bb = tB, ss = stride, ppp = pp,
          ic = inC, oc = outC, kk = k[[1]]},
        {TLayer[<|"Type" -> "Conv2D", "InChannels" -> ic, "OutChannels" -> oc,
            "KernelSize" -> kk, "Params" -> {ww, bb},
            "Forward" -> Function[{x}, TConv2d[x, ww, bb, 1, ss, ppp]]|>],
         {outC, newH, newW}}
    ]
];

compileLayerSpecImport[layer_LinearLayer, {inF_}, net_, key_] :=
Block[{outF, wData, bData, tW, tB},
    outF = layer["OutputSize"];
    If[ListQ[outF], outF = First[outF]];

    wData = Normal[NetExtract[net, {key, "Weights"}]];
    bData = Normal[NetExtract[net, {key, "Biases"}]];

    (* WL linear: {outF, inF} → TinyHVM: {inF, outF} *)
    tW = TCreate[Flatten[N[Transpose[wData]]], {inF, outF}];
    tB = TCreate[Flatten[N[bData]], {outF}];
    TSetRequiresGrad[tW]; TSetRequiresGrad[tB];

    With[{ww = tW, bb = tB, of = outF},
        {TLayer[<|"Type" -> "Linear", "InFeatures" -> inF, "OutFeatures" -> of,
            "Params" -> {ww, bb},
            "Forward" -> Function[{x}, Block[{bs = First[TDimensions[x]]},
                TOp["Add"][TOp["MatMul"][x, ww],
                    TOp["Expand"][TOp["Reshape"][bb, {1, of}], {bs, of}]]
            ]]|>],
         {of}}
    ]
];

(* Non-parametric layers: delegate to existing compileLayerSpec *)
compileLayerSpecImport[layer_ElementwiseLayer, shape_, _, _] := compileLayerSpec[layer, shape];
compileLayerSpecImport[layer_FlattenLayer, shape_, _, _] := compileLayerSpec[layer, shape];
compileLayerSpecImport[layer_PoolingLayer, shape_, _, _] := compileLayerSpec[layer, shape];
compileLayerSpecImport[_SoftmaxLayer, shape_, _, _] := Sequence[];
compileLayerSpecImport[layer_, shape_, _, _] := compileLayerSpec[layer, shape];

(* ── Single-inet training term ───────────────────────────────────── *)
(*
 * Builds a recursive inet term: N steps of forward → loss → grad → SGD
 * on a fixed batch. One TReduce drives the entire training loop.
 *
 * Pattern: train = λcounter. IFZ(counter, done, λm.
 *   logLoss → assign1 → ... → assignK → APP(REF(train), m))
 *
 * Each REF unfold clones the lazy graph with fresh nodes.
 * Weight tensors are shared (ASSIGN updates in-place).
 *)

TBuildTrainLoop[net_TNet, xBatch_TTensor, yOneHot_TTensor,
                nSteps_Integer, lr_?NumericQ] :=
Block[{params, nP, lrTen, defSlot,
        outerLam, outerVar, innerLam, innerVar,
        logits, loss, grads, logLoss, assigns, rec, chain},
    params = TParams[net];
    nP = Length[params];
    lrTen = TCreate[{N[lr]}, {1}];

    (* Pre-allocate recursive definition slot *)
    defSlot = TAllocDef[];

    (* Outer: λcounter. IFZ(counter, done, innerLam) *)
    {outerLam, outerVar} = TLamOpen[];

    (* Inner: λm. <one training step, then recurse with m> *)
    {innerLam, innerVar} = TLamOpen[];

    (* Forward pass — lazy graph referencing net's weight tensors *)
    logits = TForward[net, xBatch];
    loss = TCrossEntropyLoss[logits, yOneHot];

    (* Per-parameter gradients (not GradMulti — inet cloning needs individual walks) *)
    grads = Table[TGrad[loss, params[[i]]], {i, nP}];

    (* Print loss each step via LogPrint side-effect *)
    logLoss = TLogPrint[loss];

    (* SGD: param[i] -= lr * grad[i] *)
    assigns = Table[
        TAssign[params[[i]],
            TOp["Sub"][params[[i]], TOp["Mul"][lrTen, grads[[i]]]]],
        {i, nP}
    ];

    (* Chain: logLoss → assign_1 → ... → assign_K → recurse(m) *)
    rec = TApp[TRef[defSlot], innerVar];
    chain = Fold[TApp[#2, #1] &, rec, Reverse[assigns]];
    chain = TApp[logLoss, chain];

    TLamSetBody[innerLam, chain];
    TLamSetBody[outerLam, TIfz[outerVar, TCreate[{0.}, {1}], innerLam]];
    TSetDef[defSlot, outerLam];

    (* Entry: APP(REF(train), N) — starts the countdown *)
    TApp[TRef[defSlot], TCreate[{N[nSteps]}, {1}]]
];

(* ── Evaluate accuracy ───────────────────────────────────────────── *)

TEvalAccuracy[net_TNet, testImages_List, testLabels_List, batchSize_Integer: 64] :=
Block[{nTest, nBatches, correct = 0, nWeights, xb, logits, probs, preds, actual},
    nTest = Length[testImages];
    nBatches = Floor[nTest / batchSize];
    nWeights = TTensorCount[];

    Do[
        xb = TCreate[
            Flatten[N[testImages[[(b-1)*batchSize + 1 ;; b*batchSize]]]],
            {batchSize, 1, 28, 28}];
        logits = TForward[net, xb];
        probs = Normal[TGet[TReduce[logits]]];
        preds = Flatten[Partition[probs, 10]];
        preds = (Position[#, Max[#]][[1, 1]] - 1) & /@
                Partition[Normal[TGet[TReduce[logits]]], 10];
        actual = testLabels[[(b-1)*batchSize + 1 ;; b*batchSize]];
        correct += Count[Thread[preds == actual], True];
        TReset[nWeights],
        {b, nBatches}
    ];
    N[100 correct / (nBatches * batchSize)]
];
