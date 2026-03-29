(* Layers.wl — Neural network layer abstractions for TinyHVM *)
(* Get'd from TinyHVM.wl inside Begin["`Private`"]. All public symbols declared there. *)

(* ── Layer constructors — all return TLayer[<|...|>] ──────────────────── *)

TLinearLayer[inF_Integer, outF_Integer] := Module[{w, b, bound},
    bound = 1./Sqrt[N[inF]];
    w = TCreate[RandomReal[{-bound, bound}, inF * outF], {inF, outF}];
    b = TCreate[ConstantArray[0., outF], {outF}];
    TSetRequiresGrad[w]; TSetRequiresGrad[b];
    With[{ww = w, bb = b, inf = inF, of = outF},
        TLayer[<|"Type" -> "Linear", "InFeatures" -> inf, "OutFeatures" -> of,
          "Params" -> {ww, bb},
          "Forward" -> Function[{x},
            Module[{bs = First[TDimensions[x]]},
                TOp["Add"][TOp["MatMul"][x, ww],
                    TOp["Expand"][TOp["Reshape"][bb, {1, of}], {bs, of}]]
            ]]|>]
    ]
];

TConvLayer[inC_Integer, outC_Integer, k_Integer, opts___Rule] :=
Module[{w, b, stride, padding, groups, hasBias, bound},
    stride = "Stride" /. {opts} /. {"Stride" -> {1, 1}};
    padding = "Padding" /. {opts} /. {"Padding" -> {0, 0, 0, 0}};
    groups = "Groups" /. {opts} /. {"Groups" -> 1};
    hasBias = "Bias" /. {opts} /. {"Bias" -> True};
    bound = Sqrt[6./(inC * k * k)];
    w = TCreate[RandomReal[{-bound, bound}, outC * (inC/groups) * k * k],
                {outC, inC/groups, k, k}];
    TSetRequiresGrad[w];
    If[hasBias,
        b = TCreate[ConstantArray[0., outC], {outC}];
        TSetRequiresGrad[b];
        With[{ww = w, bb = b, gg = groups, ss = stride, pp = padding,
              ic = inC, oc = outC, kk = k},
            TLayer[<|"Type" -> "Conv2D", "InChannels" -> ic, "OutChannels" -> oc,
              "KernelSize" -> kk, "Params" -> {ww, bb},
              "Forward" -> Function[{x}, TConv2d[x, ww, bb, gg, ss, pp]]|>]
        ],
        With[{ww = w, gg = groups, ss = stride, pp = padding,
              ic = inC, oc = outC, kk = k},
            TLayer[<|"Type" -> "Conv2D", "InChannels" -> ic, "OutChannels" -> oc,
              "KernelSize" -> kk, "Params" -> {ww},
              "Forward" -> Function[{x}, TConv2d[x, ww, None, gg, ss, pp]]|>]
        ]
    ]
];

TActivation[name_String] := With[{n = name},
    TLayer[<|"Type" -> "Activation", "Name" -> n, "Params" -> {},
      "Forward" -> Function[{x}, TOp[n][x]]|>]
];

TFlattenLayer[] := TLayer[<|"Type" -> "Flatten", "Params" -> {},
    "Forward" -> Function[{x},
        Module[{dims = TDimensions[x]},
            TOp["Reshape"][x, {First[dims], Times @@ Rest[dims]}]
        ]]|>];

TMaxPoolLayer[k_Integer:2, opts___Rule] := Module[{stride},
    stride = "Stride" /. {opts} /. {"Stride" -> {k, k}};
    With[{kk = k, ss = stride},
        TLayer[<|"Type" -> "MaxPool", "PoolSize" -> kk, "Params" -> {},
          "Forward" -> Function[{x}, TMaxPool2d[x, {kk, kk}, ss]]|>]
    ]
];

(* ── WL spec compilation with shape flow ──────────────────────────────── *)

(* Output shape for a TLayer *)
iLayerOutputShape[TLayer[assoc_], shape_] := Switch[assoc["Type"],
    "Linear", {assoc["OutFeatures"]},
    "Conv2D", {assoc["OutChannels"], shape[[2]] - assoc["KernelSize"] + 1,
               shape[[3]] - assoc["KernelSize"] + 1},
    "Activation", shape,
    "Flatten", {Times @@ shape},
    "MaxPool", With[{k = assoc["PoolSize"]},
        {shape[[1]], Floor[shape[[2]]/k], Floor[shape[[3]]/k]}],
    _, shape
];

(* --- Unevaluated WL layer specs (NN framework not loaded) --- *)

iCompileSpec[HoldPattern[ConvolutionLayer[outC_Integer, {k_Integer, k2_Integer}]], {inC_, h_, w_}] :=
    {TConvLayer[inC, outC, k], {outC, h - k + 1, w - k2 + 1}};
iCompileSpec[HoldPattern[ConvolutionLayer[outC_Integer, k_Integer]], shape_] :=
    iCompileSpec[ConvolutionLayer[outC, {k, k}], shape];

iCompileSpec[HoldPattern[LinearLayer[outF_Integer]], {inF_}] :=
    {TLinearLayer[inF, outF], {outF}};

iCompileSpec[HoldPattern[ElementwiseLayer[Ramp]], shape_] := {TActivation["Relu"], shape};
iCompileSpec[HoldPattern[ElementwiseLayer["ReLU" | "Relu"]], shape_] := {TActivation["Relu"], shape};

iCompileSpec[HoldPattern[FlattenLayer[]], shape_] := {TFlattenLayer[], {Times @@ shape}};

iCompileSpec[HoldPattern[PoolingLayer[{k_Integer, k2_Integer}, ___]], {c_, h_, w_}] :=
    {TMaxPoolLayer[k], {c, Floor[h/k], Floor[w/k2]}};
iCompileSpec[HoldPattern[PoolingLayer[k_Integer, rest___]], shape_] :=
    iCompileSpec[PoolingLayer[{k, k}, rest], shape];

(* --- Evaluated WL layer objects (atomic — extract via Information) --- *)

iCompileSpec[spec_ConvolutionLayer, {inC_, h_, w_}] := Module[{outC, k},
    {outC, k} = Replace[Quiet[Information[spec, "InputForm"]],
        HoldForm[ConvolutionLayer[oc_, ks_, ___]] :> {oc, ks}];
    If[IntegerQ[k], k = {k, k}];
    {TConvLayer[inC, outC, k[[1]]], {outC, h - k[[1]] + 1, w - k[[2]] + 1}}
];

iCompileSpec[spec_LinearLayer, {inF_}] := Module[{outF},
    outF = Replace[Quiet[Information[spec, "InputForm"]],
        HoldForm[LinearLayer[out_, ___]] :> out];
    If[ListQ[outF], outF = First[outF]];
    {TLinearLayer[inF, outF], {outF}}
];

iCompileSpec[_ElementwiseLayer, shape_] := {TActivation["Relu"], shape};

iCompileSpec[_FlattenLayer, shape_] := {TFlattenLayer[], {Times @@ shape}};

iCompileSpec[spec_PoolingLayer, {c_, h_, w_}] := Module[{k},
    k = Replace[Quiet[Information[spec, "InputForm"]],
        HoldForm[PoolingLayer[ks_, ___]] :> ks];
    If[IntegerQ[k], k = {k, k}];
    {TMaxPoolLayer[k[[1]]], {c, Floor[h/k[[1]]], Floor[w/k[[2]]]}}
];

(* --- TLayer passthrough --- *)

iCompileSpec[layer_TLayer, shape_] := {layer, iLayerOutputShape[layer, shape]};

(* ── Sequential composition ─────────────────────────────────────────────── *)

(* From TLayer list *)
TNet[layers_List] /; AllTrue[layers, MatchQ[#, _TLayer] &] := With[{
    allParams = Join @@ (First[#]["Params"] & /@ layers),
    ls = layers},
    TNet[<|"Layers" -> ls, "Params" -> allParams,
      "Forward" -> Function[{x0}, Fold[First[#2]["Forward"][#1] &, x0, ls]]|>]
];

(* From WL specs + input shape — compile with shape flow *)
TNet[specs_List, inputShape_List] := Module[{layers = {}, shape = inputShape, result},
    Do[
        result = iCompileSpec[spec, shape];
        AppendTo[layers, result[[1]]];
        shape = result[[2]],
        {spec, specs}
    ];
    TNet[layers]
];

(* Accessors *)
TForward[TNet[assoc_Association], x_TTensor] := assoc["Forward"][x];
TForward[layer_TLayer, x_TTensor] := First[layer]["Forward"][x];
TParams[TNet[assoc_Association]] := assoc["Params"];
TParams[layer_TLayer] := First[layer]["Params"];

(* ── Loss functions ─────────────────────────────────────────────────────── *)

TCrossEntropyLoss[logits_TTensor, oneHot_TTensor] :=
Module[{dims, bs, nc, xmax, shifted, e, esum, probs, eps, clamped, lp, masked},
    dims = TDimensions[logits];
    bs = dims[[1]]; nc = dims[[2]];
    xmax = TOp["Expand"][TOp["RMax"][logits], {bs, nc}];
    shifted = TOp["Sub"][logits, xmax];
    e = TOp["Exp"][shifted];
    esum = TOp["Expand"][TOp["Sum"][e, {1}], {bs, nc}];
    probs = TOp["Div"][e, esum];
    eps = TOp["Expand"][TCreate[{1.*^-7}, {1, 1}], {bs, nc}];
    clamped = TOp["Max"][probs, eps];
    lp = TOp["Log"][clamped];
    masked = TOp["Mul"][oneHot, lp];
    TOp["Mul"][
        TOp["Neg"][TOp["Sum"][TOp["Sum"][masked, {1}], {0}]],
        TCreate[{1./bs}, {1, 1}]
    ]
];

(* ── SGD + training step ─────────────────────────────────────────────── *)

TSGD[params_List, gradSlots_List, lrTen_TTensor] :=
    Fold[TApp[TAssign[params[[#2]],
        TOp["Sub"][params[[#2]], TOp["Mul"][lrTen, gradSlots[[#2]]]]], #1] &,
        TCreate[{0.}, {1}],
        Reverse[Range[Length[params]]]
    ];

TTrainStep[params_List, loss_TTensor, lr_?NumericQ] :=
Module[{n = Length[params], gradSlots, gradTerm, lrTen, sgdChain, lossVal},
    gradSlots = Table[
        TCreate[ConstantArray[0., Times @@ TDimensions[params[[i]]]],
                TDimensions[params[[i]]]],
        {i, n}
    ];
    gradTerm = TGradMulti[loss, params, gradSlots];
    lrTen = TCreate[{N[lr]}, {1}];
    sgdChain = TSGD[params, gradSlots, lrTen];
    TReduce[TApp[gradTerm, sgdChain]];
    lossVal = First[Normal[TGet[TReduce[loss]]]];
    Do[TRealize[params[[i]]], {i, n}];
    lossVal
];

(* ── Formatting ─────────────────────────────────────────────────────────── *)

TLayer /: MakeBoxes[layer:TLayer[assoc_Association], StandardForm] := Module[
    {type, visibleItems, hiddenItems, icon, nParams},
    type = assoc["Type"];
    nParams = With[{dims = Quiet[TDimensions /@ assoc["Params"]]},
        If[AllTrue[dims, ListQ], Total[Times @@@ dims], Length[assoc["Params"]]]
    ];

    visibleItems = {tMakeItem["Type", type]};
    Switch[type,
        "Linear",
            AppendTo[visibleItems, tMakeItem["Size",
                ToString[assoc["InFeatures"]] <> "\[RightArrow]" <> ToString[assoc["OutFeatures"]]]],
        "Conv2D",
            AppendTo[visibleItems, tMakeItem["Channels",
                ToString[assoc["InChannels"]] <> "\[RightArrow]" <> ToString[assoc["OutChannels"]]]];
            AppendTo[visibleItems, tMakeItem["Kernel",
                ToString[assoc["KernelSize"]] <> "\[Times]" <> ToString[assoc["KernelSize"]]]],
        "Activation",
            AppendTo[visibleItems, tMakeItem["Function", assoc["Name"]]],
        "MaxPool",
            AppendTo[visibleItems, tMakeItem["Size", ToString[assoc["PoolSize"]]]],
        _, Null
    ];

    hiddenItems = {tMakeItem["Parameters", nParams]};

    icon = Switch[type,
        "Linear", Graphics[{RGBColor[0.63, 0.28, 0.64], Rectangle[]}, ImageSize -> {28, 28}],
        "Conv2D", Graphics[{RGBColor[0.20, 0.44, 0.69], Rectangle[]}, ImageSize -> {28, 28}],
        "Activation", Graphics[{RGBColor[0.44, 0.74, 0.27], Polygon[{{0,0},{0.5,1},{1,0}}]}, ImageSize -> {28, 28}],
        "Flatten", Graphics[{GrayLevel[0.5], Rectangle[{0, 0.3}, {1, 0.7}]}, ImageSize -> {28, 28}],
        "MaxPool", Graphics[{RGBColor[0.2, 0.6, 0.9], Rectangle[]}, ImageSize -> {28, 28}],
        _, None
    ];

    InterpretationBox @@ {
        BoxForm`ArrangeSummaryBox[TLayer, layer, icon, visibleItems, hiddenItems, StandardForm],
        layer, Selectable -> False, Editable -> False, SelectWithContents -> True
    }
];

TNet /: MakeBoxes[net:TNet[assoc_Association], StandardForm] := Module[
    {nLayers, nParams, visibleItems, hiddenItems, icon, layerTypes},
    nLayers = Length[assoc["Layers"]];
    nParams = With[{dims = Quiet[TDimensions /@ assoc["Params"]]},
        If[AllTrue[dims, ListQ], Total[Times @@@ dims], Length[assoc["Params"]]]
    ];
    layerTypes = (First[#]["Type"] & /@ assoc["Layers"]);

    visibleItems = {
        tMakeItem["Layers", nLayers],
        tMakeItem["Parameters", nParams]
    };
    hiddenItems = {
        tMakeItem["Architecture", StringRiffle[layerTypes, " \[RightArrow] "]]
    };
    icon = Graphics[{RGBColor[0.2, 0.5, 0.8],
        Rectangle[{0, 0}, {1, 0.25}], Rectangle[{0, 0.35}, {1, 0.60}],
        Rectangle[{0, 0.70}, {1, 0.95}]}, ImageSize -> {28, 28}];

    InterpretationBox @@ {
        BoxForm`ArrangeSummaryBox[TNet, net, icon, visibleItems, hiddenItems, StandardForm],
        net, Selectable -> False, Editable -> False, SelectWithContents -> True
    }
];
