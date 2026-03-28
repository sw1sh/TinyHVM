(* ::Package:: *)
(* TinyHVM — Wolfram Language interface to TinyHVM interaction net tensor engine *)

BeginPackage["TinyHVM`"];

(* ── Core heads ─────────────────────────────────────────────────────────── *)

TTerm::usage = "TTerm[id] represents a TinyHVM term (tensor, lambda, application, or any inet node).";
TOp::usage = "TOp[\"Name\"][args...] applies a TinyHVM tensor operation lazily.";

(* ── Context lifecycle ──────────────────────────────────────────────────── *)

TInit::usage = "TInit[\"metal\"|\"cpu\"] initializes the TinyHVM runtime. Returns True on success.";
TFree::usage = "TFree[] destroys the TinyHVM context and frees all GPU memory.";
TReset::usage = "TReset[keep] frees all tensors above index 'keep' and resets the heap. Weight tensors in [0,keep) survive.";

(* ── Data transfer ──────────────────────────────────────────────────────── *)

TCreate::usage = "TCreate[data, shape] creates a GPU tensor from host data. data can be a flat List or NumericArray.";
TGet::usage = "TGet[term] reads tensor data from GPU to a NumericArray. Reduces lazily if needed.";
TReduce::usage = "TReduce[term] triggers interaction net reduction, returning the reduced term.";
TRealize::usage = "TRealize[term] forces reduction in-place (updates the stored term).";

(* ── Queries ────────────────────────────────────────────────────────────── *)

TDimensions::usage = "TDimensions[term] returns the shape of a tensor or lazy op term.";
TTermTag::usage = "TTermTag[term] returns the tag code of a term (0=APP,1=LAM,...,10=TEN,11=TOP).";
TTermExt::usage = "TTermExt[term] returns the EXT field (UOp code for lazy ops, dtype for tensors).";
TTensorCount::usage = "TTensorCount[] returns the number of tensors in the context.";
THeapPos::usage = "THeapPos[] returns the current heap position.";
TInteractionCount::usage = "TInteractionCount[] returns the total interaction count.";

(* ── Interaction net primitives ─────────────────────────────────────────── *)

TLam::usage = "TLam[body] creates a lambda term. Returns {TTerm[lamId], TTerm[varId]}.";
TApp::usage = "TApp[fun, arg] creates an application term.";
TSup::usage = "TSup[a, b] creates a superposition term.";
TDup::usage = "TDup[z] duplicates a term. Returns {TTerm[dp0], TTerm[dp1]}.";
TDefine::usage = "TDefine[body] registers a named definition. Returns the name (Integer).";
TRef::usage = "TRef[name] creates a reference to a named definition.";

(* ── Autograd ───────────────────────────────────────────────────────────── *)

TSetRequiresGrad::usage = "TSetRequiresGrad[term] marks a tensor for gradient tracking.";
TGrad::usage = "TGrad[y, x] returns a lazy gradient term dy/dx.";
TGradMulti::usage = "TGradMulti[loss, params, slots] returns a lazy term that deposits all gradients via ASSIGN.";
TBackward::usage = "TBackward[loss, params] computes gradients for all params. Returns list of grad TTerms.";

(* ── Advanced ops ───────────────────────────────────────────────────────── *)

TWhere::usage = "TWhere[cond, then, else] creates a lazy elementwise select.";
TAssign::usage = "TAssign[dst, src] creates a lazy in-place update term.";
TIfz::usage = "TIfz[counter, zeroCase, succLam] creates an if-zero branch.";
TLogPrint::usage = "TLogPrint[term] wraps a term with debug printing.";

(* ── CNN layers ─────────────────────────────────────────────────────────── *)

TConv2d::usage = "TConv2d[x, w, b, groups, stride, padding] creates a lazy 2D convolution.";
TMaxPool2d::usage = "TMaxPool2d[x, kernel, stride] creates a lazy 2D max pooling.";
TPool::usage = "TPool[x, kernel, stride, nSpatial] creates a lazy sliding-window pool.";

(* ── Debug ──────────────────────────────────────────────────────────────── *)

TPrintTerm::usage = "TPrintTerm[term] prints the internal term representation.";
TProfileReport::usage = "TProfileReport[] prints profiling info.";
TProfileReset::usage = "TProfileReset[] resets profiling counters.";

(* ── Neural network layers ─────────────────────────────────────────────── *)

TLinearLayer::usage = "TLinearLayer[inF, outF] creates a dense linear layer with trainable weights.";
TConvLayer::usage = "TConvLayer[inC, outC, k] creates a 2D convolutional layer.";
TActivation::usage = "TActivation[\"Relu\"] creates an activation layer (no parameters).";
TFlattenLayer::usage = "TFlattenLayer[] flattens spatial dims, keeping batch dim.";
TMaxPoolLayer::usage = "TMaxPoolLayer[k] creates a max pooling layer.";
TLayer::usage = "TLayer[<|...|>] represents a neural network layer with weights and forward function.";
TNet::usage = "TNet[{layer1, layer2, ...}] composes layers into a sequential network. TNet[specs, inputShape] compiles WL layer specs with shape inference.";
TForward::usage = "TForward[net, x] runs forward pass through a network.";
TParams::usage = "TParams[net] returns flat list of all trainable parameters.";
TCrossEntropyLoss::usage = "TCrossEntropyLoss[logits, oneHot] computes cross-entropy loss.";
TSGD::usage = "TSGD[params, gradSlots, lrTen] builds an SGD update chain.";
TTrainStep::usage = "TTrainStep[params, loss, lr] performs one gradient step, returns loss value.";

(* ── Graph visualization ──────────────────────────────────────────────── *)

TComputationGraph::usage = "TComputationGraph[term] returns a Graph of the computation DAG.";
$TGraphTrace::usage = "Set to True to record computation graph edges.";
TGraphReset::usage = "TGraphReset[] clears the recorded computation graph.";

(* ── Profiling ────────────────────────────────────────────────────────── *)

TProfileEnable::usage = "TProfileEnable[] enables UOp-level profiling.";
TProfileData::usage = "TProfileData[] returns profile data as a structured Association.";
TProfileSummary::usage = "TProfileSummary[] returns a formatted profiling summary.";
TProfileTimeline::usage = "TProfileTimeline[snapshots] plots timing across training steps.";

(* ── Inet helpers (recursive defs) ────────────────────────────────────── *)

TAllocDef::usage = "TAllocDef[] pre-allocates a definition slot for recursive references.";
TSetDef::usage = "TSetDef[name, body] sets the body of a pre-allocated definition.";
TLamOpen::usage = "TLamOpen[] creates a lambda with placeholder body. Returns {lam, var}.";
TLamSetBody::usage = "TLamSetBody[lam, body] sets the body of a lambda created with TLamOpen.";

(* ── Library path ───────────────────────────────────────────────────────── *)

$TinyHVMLibrary::usage = "Path to the TinyHVM dynamic library.";

Begin["`Private`"];

(* ── Library path resolution ────────────────────────────────────────────── *)

$TinyHVMLibrary = With[{found = FindLibrary["TinyHVM"]},
    If[StringQ[found],
        found,
        FileNameJoin[{
            DirectoryName[$InputFileName, 2],  (* up from Kernel/ to TinyHVM/ *)
            "LibraryResources", "MacOSX-ARM64",
            "TinyHVM.dylib"
        }]
    ]
];

(* Metallib path: same directory as dylib *)
$metallibPath = FileNameJoin[{DirectoryName[$TinyHVMLibrary], "shaders.metallib"}];

(* ── ID counter ─────────────────────────────────────────────────────────── *)

$nextId = 1;
allocId[] := $nextId++;

(* ── UOp codes (must match tinyhvm.h) ───────────────────────────────────── *)

$uopCode = <|
    "Load" -> 0, "Store" -> 1, "Copy" -> 2,
    "Neg" -> 3, "Exp" -> 4, "Log" -> 5, "Relu" -> 6, "Cast" -> 7, "Sqrt" -> 8,
    "Add" -> 9, "Mul" -> 10, "Div" -> 11, "Max" -> 12, "Cmp" -> 13, "Sub" -> 14,
    "Sum" -> 15, "RMax" -> 16, "MatMul" -> 17,
    "Reshape" -> 18, "Permute" -> 19, "Expand" -> 20, "Shrink" -> 21, "Pad" -> 22,
    "Fusing" -> 23, "Assign" -> 24, "Where" -> 25, "Ifz" -> 26,
    "LogPrint" -> 27, "Grad" -> 28
|>;

(* Tag names for display *)
$tagName = <|
    0 -> "App", 1 -> "Lam", 2 -> "Var", 3 -> "Sup",
    4 -> "Dp0", 5 -> "Dp1", 6 -> "Era",
    7 -> "Num", 8 -> "Ref", 9 -> "Op2",
    10 -> "Ten", 11 -> "Top", 12 -> "Ctr"
|>;

(* Unary ops (second arg = ERA) *)
$unaryOps = {"Neg", "Exp", "Log", "Relu", "Sqrt", "Cast"};

(* ── Library function handles ───────────────────────────────────────────── *)

$libraryLoaded = False;

(* All function handles initialized to None *)
thvmSetMetallibPathFn = None;
thvmInitFn = None;
thvmFreeFn = None;
thvmResetFn = None;
thvmTensorFn = None;
thvmToHostFn = None;
thvmDimensionsFn = None;
thvmTermTagFn = None;
thvmTermExtFn = None;
thvmTensorCountFn = None;
thvmOpFn = None;
thvmReduceFn = None;
thvmRealizeFn = None;
thvmReshapeFn = None;
thvmExpandFn = None;
thvmPermuteFn = None;
thvmPadFn = None;
thvmShrinkFn = None;
thvmSumAxesFn = None;
thvmLamFn = None;
thvmAppFn = None;
thvmSupFn = None;
thvmDupFn = None;
thvmDefineFn = None;
thvmRefFn = None;
thvmWhereFn = None;
thvmAssignFn = None;
thvmIfzFn = None;
thvmLogPrintFn = None;
thvmSetRequiresGradFn = None;
thvmGradFn = None;
thvmGradMultiFn = None;
thvmBackwardFn = None;
thvmConv2dFn = None;
thvmMaxPool2dFn = None;
thvmPoolFn = None;
thvmPrintTermFn = None;
thvmProfileReportFn = None;
thvmProfileResetFn = None;
thvmHeapPosFn = None;
thvmInteractionCountFn = None;
thvmAllocDefFn = None;
thvmSetDefFn = None;
thvmLamOpenFn = None;
thvmLamSetBodyFn = None;
thvmProfileEnableFn = None;
thvmProfileDataFn = None;

loadLibrary[] := If[!$libraryLoaded && FileExistsQ[$TinyHVMLibrary],
    $libraryLoaded = True;

    (* Context lifecycle *)
    thvmSetMetallibPathFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmSetMetallibPath",
        {"UTF8String"}, "Void"];
    thvmInitFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmInit",
        {"UTF8String"}, Integer];
    thvmFreeFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmFree",
        {}, "Void"];
    thvmResetFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmReset",
        {Integer}, "Void"];

    (* Tensor creation & data *)
    thvmTensorFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmTensor",
        {Integer, LibraryDataType[NumericArray], {Integer, 1}}, "Void"];
    thvmToHostFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmToHost",
        {Integer}, LibraryDataType[NumericArray]];
    thvmDimensionsFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmDimensions",
        {Integer}, {Integer, 1}];
    thvmTermTagFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmTermTag",
        {Integer}, Integer];
    thvmTermExtFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmTermExt",
        {Integer}, Integer];
    thvmTensorCountFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmTensorCount",
        {}, Integer];

    (* Core op *)
    thvmOpFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmOp",
        {Integer, Integer, Integer, Integer}, "Void"];

    (* Reduction *)
    thvmReduceFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmReduce",
        {Integer, Integer}, "Void"];
    thvmRealizeFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmRealize",
        {Integer}, "Void"];

    (* Movement ops *)
    thvmReshapeFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmReshape",
        {Integer, Integer, {Integer, 1}}, "Void"];
    thvmExpandFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmExpand",
        {Integer, Integer, {Integer, 1}}, "Void"];
    thvmPermuteFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmPermute",
        {Integer, Integer, {Integer, 1}}, "Void"];
    thvmPadFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmPad",
        {Integer, Integer, {Integer, 1}}, "Void"];
    thvmShrinkFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmShrink",
        {Integer, Integer, {Integer, 1}}, "Void"];
    thvmSumAxesFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmSumAxes",
        {Integer, Integer, {Integer, 1}}, "Void"];

    (* Inet primitives *)
    thvmLamFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmLam",
        {Integer, Integer, Integer}, "Void"];
    thvmAppFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmApp",
        {Integer, Integer, Integer}, "Void"];
    thvmSupFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmSup",
        {Integer, Integer, Integer}, "Void"];
    thvmDupFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmDup",
        {Integer, Integer, Integer}, "Void"];
    thvmDefineFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmDefine",
        {Integer}, Integer];
    thvmRefFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmRef",
        {Integer, Integer}, "Void"];

    (* Advanced ops *)
    thvmWhereFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmWhere",
        {Integer, Integer, Integer, Integer}, "Void"];
    thvmAssignFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmAssign",
        {Integer, Integer, Integer}, "Void"];
    thvmIfzFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmIfz",
        {Integer, Integer, Integer, Integer}, "Void"];
    thvmLogPrintFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmLogPrint",
        {Integer, Integer}, "Void"];

    (* Autograd *)
    thvmSetRequiresGradFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmSetRequiresGrad",
        {Integer}, "Void"];
    thvmGradFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmGrad",
        {Integer, Integer, Integer}, "Void"];
    thvmGradMultiFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmGradMulti",
        {Integer, Integer, {Integer, 1}, {Integer, 1}}, "Void"];
    thvmBackwardFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmBackward",
        {Integer, {Integer, 1}, {Integer, 1}}, "Void"];

    (* CNN layers *)
    thvmConv2dFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmConv2d",
        {Integer, Integer, Integer, Integer, Integer, {Integer, 1}, {Integer, 1}}, "Void"];
    thvmMaxPool2dFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmMaxPool2d",
        {Integer, Integer, {Integer, 1}, {Integer, 1}}, "Void"];
    thvmPoolFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmPool",
        {Integer, Integer, {Integer, 1}, {Integer, 1}, Integer}, "Void"];

    (* Debug *)
    thvmPrintTermFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmPrintTerm",
        {Integer}, "Void"];
    thvmProfileReportFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmProfileReport",
        {}, "Void"];
    thvmProfileResetFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmProfileReset",
        {}, "Void"];
    thvmHeapPosFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmHeapPos",
        {}, Integer];
    thvmInteractionCountFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmInteractionCount",
        {}, Integer];

    (* Inet helpers *)
    thvmAllocDefFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmAllocDef",
        {}, Integer];
    thvmSetDefFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmSetDef",
        {Integer, Integer}, "Void"];
    thvmLamOpenFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmLamOpen",
        {Integer, Integer}, "Void"];
    thvmLamSetBodyFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmLamSetBody",
        {Integer, Integer}, "Void"];

    (* Profiling *)
    thvmProfileEnableFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmProfileEnable",
        {}, "Void"];
    thvmProfileDataFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmProfileData",
        {}, LibraryDataType[NumericArray]];
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Context lifecycle                                                       *)
(* ════════════════════════════════════════════════════════════════════════ *)

TInit[device_String:"metal"] := Module[{ok},
    loadLibrary[];
    (* Set metallib path so Metal init can find shaders *)
    If[FileExistsQ[$metallibPath],
        thvmSetMetallibPathFn[$metallibPath]
    ];
    ok = thvmInitFn[device];
    $nextId = 1;
    ok === 1
];

TFree[] := (loadLibrary[]; thvmFreeFn[]; $nextId = 1;);

TReset[keep_Integer] := (loadLibrary[]; thvmResetFn[keep];);

(* ════════════════════════════════════════════════════════════════════════ *)
(* Data transfer                                                           *)
(* ════════════════════════════════════════════════════════════════════════ *)

TCreate[data_List, shape_List] := Module[{na, out = allocId[]},
    loadLibrary[];
    na = NumericArray[N[Flatten[data]], "Real32"];
    thvmTensorFn[out, na, shape];
    recordNode[out, "Tensor", {}];
    TTerm[out]
];

TCreate[na_NumericArray, shape_List] := Module[{out = allocId[]},
    loadLibrary[];
    thvmTensorFn[out, na, shape];
    recordNode[out, "Tensor", {}];
    TTerm[out]
];

(* Convenience: infer shape from list structure *)
TCreate[data_List] := TCreate[data, Dimensions[data]];

TGet[TTerm[id_Integer]] := (
    loadLibrary[];
    thvmToHostFn[id]
);

TReduce[TTerm[id_Integer]] := Module[{out = allocId[]},
    loadLibrary[];
    thvmReduceFn[id, out];
    TTerm[out]
];

TRealize[TTerm[id_Integer]] := (
    loadLibrary[];
    thvmRealizeFn[id];
);

(* ════════════════════════════════════════════════════════════════════════ *)
(* Queries                                                                 *)
(* ════════════════════════════════════════════════════════════════════════ *)

TDimensions[TTerm[id_Integer]] := (
    loadLibrary[];
    Normal[thvmDimensionsFn[id]]
);

TTermTag[TTerm[id_Integer]] := (
    loadLibrary[];
    thvmTermTagFn[id]
);

TTermExt[TTerm[id_Integer]] := (
    loadLibrary[];
    thvmTermExtFn[id]
);

TTensorCount[] := (loadLibrary[]; thvmTensorCountFn[]);
THeapPos[] := (loadLibrary[]; thvmHeapPosFn[]);
TInteractionCount[] := (loadLibrary[]; thvmInteractionCountFn[]);

(* ════════════════════════════════════════════════════════════════════════ *)
(* TOp dispatch — binary ops                                               *)
(* ════════════════════════════════════════════════════════════════════════ *)

TOp[op_String][a_TTerm, b_TTerm] /;
    KeyExistsQ[$uopCode, op] && !MemberQ[$unaryOps, op] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmOpFn[out, $uopCode[op], a[[1]], b[[1]]];
    recordNode[out, op, {a[[1]], b[[1]]}];
    TTerm[out]
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* TOp dispatch — unary ops (ERA as second arg)                            *)
(* ════════════════════════════════════════════════════════════════════════ *)

TOp[op_String][a_TTerm] /; MemberQ[$unaryOps, op] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmOpFn[out, $uopCode[op], a[[1]], 0];  (* 0 = ERA *)
    recordNode[out, op, {a[[1]]}];
    TTerm[out]
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* TOp dispatch — movement ops                                             *)
(* ════════════════════════════════════════════════════════════════════════ *)

TOp["Reshape"][t_TTerm, shape_List] := Module[{out = allocId[]},
    loadLibrary[];
    thvmReshapeFn[out, t[[1]], shape];
    recordNode[out, "Reshape", {t[[1]]}];
    TTerm[out]
];

TOp["Expand"][t_TTerm, shape_List] := Module[{out = allocId[]},
    loadLibrary[];
    thvmExpandFn[out, t[[1]], shape];
    recordNode[out, "Expand", {t[[1]]}];
    TTerm[out]
];

TOp["Permute"][t_TTerm, axes_List] := Module[{out = allocId[]},
    loadLibrary[];
    thvmPermuteFn[out, t[[1]], axes];
    recordNode[out, "Permute", {t[[1]]}];
    TTerm[out]
];

TOp["Pad"][t_TTerm, pairs_List] := Module[{out = allocId[]},
    loadLibrary[];
    thvmPadFn[out, t[[1]], Flatten[pairs]];
    TTerm[out]
];

TOp["Shrink"][t_TTerm, pairs_List] := Module[{out = allocId[]},
    loadLibrary[];
    thvmShrinkFn[out, t[[1]], Flatten[pairs]];
    TTerm[out]
];

TOp["Sum"][t_TTerm, axes_List] := Module[{out = allocId[]},
    loadLibrary[];
    thvmSumAxesFn[out, t[[1]], axes];
    recordNode[out, "Sum", {t[[1]]}];
    TTerm[out]
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Interaction net primitives                                              *)
(* ════════════════════════════════════════════════════════════════════════ *)

TLam[body_TTerm] := Module[{lamId = allocId[], varId = allocId[]},
    loadLibrary[];
    thvmLamFn[lamId, varId, body[[1]]];
    {TTerm[lamId], TTerm[varId]}
];

TApp[fun_TTerm, arg_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmAppFn[out, fun[[1]], arg[[1]]];
    TTerm[out]
];

TSup[a_TTerm, b_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmSupFn[out, a[[1]], b[[1]]];
    TTerm[out]
];

TDup[z_TTerm] := Module[{dp0 = allocId[], dp1 = allocId[]},
    loadLibrary[];
    thvmDupFn[z[[1]], dp0, dp1];
    {TTerm[dp0], TTerm[dp1]}
];

TDefine[body_TTerm] := (
    loadLibrary[];
    thvmDefineFn[body[[1]]]
);

TRef[name_Integer] := Module[{out = allocId[]},
    loadLibrary[];
    thvmRefFn[out, name];
    TTerm[out]
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Advanced ops                                                            *)
(* ════════════════════════════════════════════════════════════════════════ *)

TWhere[cond_TTerm, then_TTerm, else_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmWhereFn[out, cond[[1]], then[[1]], else[[1]]];
    TTerm[out]
];

TAssign[dst_TTerm, src_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmAssignFn[out, dst[[1]], src[[1]]];
    TTerm[out]
];

TIfz[counter_TTerm, zeroCase_TTerm, succLam_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmIfzFn[out, counter[[1]], zeroCase[[1]], succLam[[1]]];
    TTerm[out]
];

TLogPrint[t_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmLogPrintFn[out, t[[1]]];
    TTerm[out]
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Autograd                                                                *)
(* ════════════════════════════════════════════════════════════════════════ *)

TSetRequiresGrad[t_TTerm] := (
    loadLibrary[];
    thvmSetRequiresGradFn[t[[1]]];
);

TGrad[y_TTerm, x_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmGradFn[out, y[[1]], x[[1]]];
    TTerm[out]
];

TGradMulti[loss_TTerm, params:{__TTerm}, slots:{__TTerm}] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmGradMultiFn[out, loss[[1]], params[[All, 1]], slots[[All, 1]]];
    TTerm[out]
];

TBackward[loss_TTerm, params:{__TTerm}] :=
Module[{gradIds, n = Length[params]},
    loadLibrary[];
    gradIds = Table[allocId[], n];
    thvmBackwardFn[loss[[1]], params[[All, 1]], gradIds];
    TTerm /@ gradIds
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* CNN layers                                                              *)
(* ════════════════════════════════════════════════════════════════════════ *)

TConv2d[x_TTerm, w_TTerm, b_TTerm, groups_Integer:1,
        stride:{_Integer, _Integer}:{1, 1},
        padding:{_Integer, _Integer, _Integer, _Integer}:{0, 0, 0, 0}] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmConv2dFn[out, x[[1]], w[[1]], b[[1]], groups, stride, padding];
    recordNode[out, "Conv2D", {x[[1]], w[[1]], b[[1]]}];
    TTerm[out]
];

(* No-bias variant *)
TConv2d[x_TTerm, w_TTerm, None, groups_Integer:1,
        stride:{_Integer, _Integer}:{1, 1},
        padding:{_Integer, _Integer, _Integer, _Integer}:{0, 0, 0, 0}] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmConv2dFn[out, x[[1]], w[[1]], 0, groups, stride, padding];
    TTerm[out]
];

TMaxPool2d[x_TTerm, kernel:{_Integer, _Integer}:{2, 2},
           stride:{_Integer, _Integer}:{2, 2}] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmMaxPool2dFn[out, x[[1]], kernel, stride];
    TTerm[out]
];

TPool[x_TTerm, kernel_List, stride_List, nSpatial_Integer:2] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmPoolFn[out, x[[1]], kernel, stride, nSpatial];
    TTerm[out]
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Debug / profiling                                                       *)
(* ════════════════════════════════════════════════════════════════════════ *)

TPrintTerm[t_TTerm] := (loadLibrary[]; thvmPrintTermFn[t[[1]]];);
TProfileReport[] := (loadLibrary[]; thvmProfileReportFn[];);
TProfileReset[] := (loadLibrary[]; thvmProfileResetFn[];);

(* ════════════════════════════════════════════════════════════════════════ *)
(* Inet helpers — recursive definitions                                    *)
(* ════════════════════════════════════════════════════════════════════════ *)

TAllocDef[] := (loadLibrary[]; thvmAllocDefFn[]);

TSetDef[name_Integer, body_TTerm] := (loadLibrary[]; thvmSetDefFn[name, body[[1]]];);

TLamOpen[] := Module[{lamId = allocId[], varId = allocId[]},
    loadLibrary[];
    thvmLamOpenFn[lamId, varId];
    {TTerm[lamId], TTerm[varId]}
];

TLamSetBody[lam_TTerm, body_TTerm] := (loadLibrary[]; thvmLamSetBodyFn[lam[[1]], body[[1]]];);

(* ════════════════════════════════════════════════════════════════════════ *)
(* UpValues — natural WL syntax for tensor arithmetic                      *)
(* ════════════════════════════════════════════════════════════════════════ *)

TTerm /: Plus[a_TTerm, b_TTerm] := TOp["Add"][a, b];
TTerm /: Times[a_TTerm, b_TTerm] := TOp["Mul"][a, b];
TTerm /: Subtract[a_TTerm, b_TTerm] := TOp["Sub"][a, b];
TTerm /: Dot[a_TTerm, b_TTerm] := TOp["MatMul"][a, b];
TTerm /: Sqrt[t_TTerm] := TOp["Sqrt"][t];
TTerm /: Exp[t_TTerm] := TOp["Exp"][t];
TTerm /: Log[t_TTerm] := TOp["Log"][t];
TTerm /: Minus[t_TTerm] := TOp["Neg"][t];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Formatting — summary boxes                                              *)
(* ════════════════════════════════════════════════════════════════════════ *)

(* UOp name table for TAG_TOP display *)
$uopName = <|
    0 -> "Load", 1 -> "Store", 2 -> "Copy",
    3 -> "Neg", 4 -> "Exp", 5 -> "Log", 6 -> "Relu", 7 -> "Cast", 8 -> "Sqrt",
    9 -> "Add", 10 -> "Mul", 11 -> "Div", 12 -> "Max", 13 -> "Cmp", 14 -> "Sub",
    15 -> "Sum", 16 -> "RMax", 17 -> "MatMul",
    18 -> "Reshape", 19 -> "Permute", 20 -> "Expand", 21 -> "Shrink", 22 -> "Pad",
    23 -> "Fusing", 24 -> "Assign", 25 -> "Where", 26 -> "Ifz",
    27 -> "LogPrint", 28 -> "Grad"
|>;

(* Tag → icon character for visual distinction *)
$tagIcon = <|
    "Ten" -> "\[FilledSmallSquare]",
    "Top" -> "\[RightTriangle]",
    "Lam" -> "\[Lambda]",
    "App" -> "\[Application]",
    "Sup" -> "\[CirclePlus]",
    "Dp0" -> "\[LeftAngleBracket]",
    "Dp1" -> "\[RightAngleBracket]",
    "Era" -> "\[EmptySet]",
    "Ref" -> "\[RightArrow]",
    "Ctr" -> "\[FilledDiamond]"
|>;

tMakeItem[name_, value_] := BoxForm`MakeSummaryItem[{name <> ": ", value}, StandardForm];

TTerm /: MakeBoxes[t:TTerm[id_Integer], StandardForm] := Module[
    {tag, tagStr, dims, shapeStr, visibleItems, hiddenItems, icon},

    tag = Quiet[TTermTag[t]];
    tagStr = Lookup[$tagName, tag, "?"];

    (* Build visible items based on term type *)
    visibleItems = {tMakeItem["Tag", tagStr]};

    (* For tensors and lazy ops: show dimensions *)
    If[tag === 10 || tag === 11,
        dims = Quiet[TDimensions[t]];
        If[ListQ[dims],
            shapeStr = StringRiffle[ToString /@ dims, "\[ThinSpace]\[Times]\[ThinSpace]"];
            AppendTo[visibleItems, tMakeItem["Shape", shapeStr]]
        ]
    ];

    (* For lazy ops (TAG_TOP): show the UOp name *)
    If[tag === 11,
        With[{ext = Quiet[TTermExt[t]]},
            If[IntegerQ[ext],
                AppendTo[visibleItems,
                    tMakeItem["Op", Lookup[$uopName, ext, "UOp" <> ToString[ext]]]]
            ]
        ]
    ];

    (* For ERA: mark as completed reduction *)
    If[tag === 6,
        visibleItems = {tMakeItem["Status", "Erased"]}
    ];

    hiddenItems = {tMakeItem["ID", id]};

    (* For tensors: show element count *)
    If[tag === 10 && ListQ[dims],
        AppendTo[hiddenItems, tMakeItem["Elements", Times @@ dims]]
    ];

    (* Icon: colored shape depending on term type *)
    icon = Switch[tagStr,
        "Ten", Graphics[{RGBColor[0.2, 0.6, 0.9], Rectangle[]}, ImageSize -> {28, 28}],
        "Top", Graphics[{RGBColor[0.9, 0.5, 0.2], Polygon[{{0,0},{1,0.5},{0,1}}]}, ImageSize -> {28, 28}],
        "Lam", Graphics[{RGBColor[0.5, 0.8, 0.3], Disk[]}, ImageSize -> {28, 28}],
        "App", Graphics[{RGBColor[0.7, 0.3, 0.7], Disk[]}, ImageSize -> {28, 28}],
        "Era", Graphics[{GrayLevel[0.5], Thickness[0.08], Circle[{0.5, 0.5}, 0.35]}, ImageSize -> {28, 28}],
        "Ref", Graphics[{RGBColor[0.8, 0.4, 0.1], Polygon[{{0.2,0},{0.8,0},{1,0.5},{0.8,1},{0.2,1},{0,0.5}}]}, ImageSize -> {28, 28}],
        _, None
    ];

    InterpretationBox @@ {
        BoxForm`ArrangeSummaryBox[
            TTerm,
            t,
            icon,
            visibleItems,
            hiddenItems,
            StandardForm
        ],
        t,
        Selectable -> False, Editable -> False, SelectWithContents -> True
    }
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Load subpackages                                                        *)
(* ════════════════════════════════════════════════════════════════════════ *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "Visualization.wl"}]];
Get[FileNameJoin[{DirectoryName[$InputFileName], "Layers.wl"}]];

End[];
EndPackage[];
