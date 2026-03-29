(* ::Package:: *)
(* TinyHVM — Wolfram Language interface to TinyHVM interaction net tensor engine *)

BeginPackage["TinyHVM`"];

(* ── Core heads ─────────────────────────────────────────────────────────── *)

TTerm::usage = "TTerm[id] represents a TinyHVM interaction net node (lambda, application, superposition, etc.).";
TTensor::usage = "TTensor[id] represents a TinyHVM tensor (materialized TAG_TEN or lazy TAG_TOP).";
TOp::usage = "TOp[\"Name\"][args...] applies a TinyHVM tensor operation lazily.";

(* ── Conversion ─────────────────────────────────────────────────────────── *)

ToTTensor::usage = "ToTTensor[TTerm[id]] converts a TTerm to a TTensor.";
ToTTerm::usage = "ToTTerm[TTensor[id]] converts a TTensor to a TTerm.";

(* ── Context lifecycle ──────────────────────────────────────────────────── *)

TInit::usage = "TInit[\"metal\"|\"cpu\"] initializes the TinyHVM runtime. Returns True on success.";
TFree::usage = "TFree[] destroys the TinyHVM context and frees all GPU memory.";
TReset::usage = "TReset[keep] frees all tensors above index 'keep' and resets the heap. Weight tensors in [0,keep) survive.";

(* ── Data transfer ──────────────────────────────────────────────────────── *)

TCreate::usage = "TCreate[data, shape] creates a GPU tensor from host data. Returns TTensor.";
TGet::usage = "TGet[tensor] reads tensor data from GPU to a NumericArray. Reduces lazily if needed.";
TReduce::usage = "TReduce[tensor] triggers interaction net reduction, returning the reduced tensor or term.";
TRealize::usage = "TRealize[tensor] forces reduction in-place (updates the stored term).";

(* ── Device ──────────────────────────────────────────────────────────────── *)

TDevice::usage = "TDevice[tensor] returns the device string (\"cpu\" or \"metal\").";
TToDevice::usage = "TToDevice[tensor, device] lazily transfers a tensor to another device.";
TView::usage = "TView[tensor] returns an Association with view metadata.";

(* ── Queries ────────────────────────────────────────────────────────────── *)

TDimensions::usage = "TDimensions[tensor] returns the shape of a tensor or lazy op.";
TTermTag::usage = "TTermTag[term] returns the tag name as a string (\"Ten\", \"Top\", \"Lam\", etc.).";
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
TNum::usage = "TNum[n] creates an integer term (TAG_NUM). n must be a non-negative integer.";
TOp2::usage = "TOp2[op, x, y] creates a binary integer operation (TAG_OP2). op: \"Add\", \"Sub\", \"Mul\", \"Div\", \"Eq\", \"Mod\".";
TNumValue::usage = "TNumValue[term] extracts the integer value from a reduced TAG_NUM term.";
TSupValues::usage = "TSupValues[term] recursively extracts all branches from a nested SUP tree into a flat list.";
TSupNumValues::usage = "TSupNumValues[term] extracts all SUP branches as integer values.";

(* ── Autograd ───────────────────────────────────────────────────────────── *)

TSetRequiresGrad::usage = "TSetRequiresGrad[tensor] marks a tensor for gradient tracking.";
TGrad::usage = "TGrad[y, x] returns a lazy gradient tensor dy/dx.";
TGradMulti::usage = "TGradMulti[loss, params, slots] returns a lazy term that deposits all gradients via ASSIGN.";
TBackward::usage = "TBackward[loss, params] computes gradients for all params. Returns list of TTensors.";

(* ── Advanced ops ───────────────────────────────────────────────────────── *)

TWhere::usage = "TWhere[cond, then, else] creates a lazy elementwise select.";
TAssign::usage = "TAssign[dst, src] creates a lazy in-place update term.";
TIfz::usage = "TIfz[counter, zeroCase, succLam] creates an if-zero branch.";
TLogPrint::usage = "TLogPrint[tensor] wraps a tensor with debug printing.";

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
TNet::usage = "TNet[{layer1, layer2, ...}] composes layers into a sequential network.";
TForward::usage = "TForward[net, x] runs forward pass through a network.";
TParams::usage = "TParams[net] returns flat list of all trainable parameters.";
TCrossEntropyLoss::usage = "TCrossEntropyLoss[logits, oneHot] computes cross-entropy loss.";
TSGD::usage = "TSGD[params, gradSlots, lrTen] builds an SGD update chain.";
TTrainStep::usage = "TTrainStep[params, loss, lr] performs one gradient step, returns loss value.";

(* ── Net compilation ──────────────────────────────────────────────────── *)

TCompileNet::usage = "TCompileNet[net, inputShape] converts a WL NetChain/NetGraph to a TNet with fresh weights. Use \"ImportWeights\"->True to transfer trained weights.";
TBuildTrainLoop::usage = "TBuildTrainLoop[net, xBatch, yOneHot, nSteps, lr] builds a single inet term that trains for nSteps on a fixed batch. One TReduce drives the entire loop.";
TEvalAccuracy::usage = "TEvalAccuracy[net, testImages, testLabels] evaluates classification accuracy on test data.";

(* ── Graph visualization ──────────────────────────────────────────────── *)

TINetGraph::usage = "TINetGraph[tensor|term] returns a Graph of the interaction net by walking the C heap.";

(* ── Single-step reduction & tracing ──────────────────────────────────── *)

TTraceEnable::usage = "TTraceEnable[] enables interaction tracing.";
TTraceDisable::usage = "TTraceDisable[] disables interaction tracing.";
TTraceClear::usage = "TTraceClear[] clears the trace buffer.";
TTrace::usage = "TTrace[] returns a list of interaction trace records.";
TReduceSteps::usage = "TReduceSteps[tensor, n] reduces at most n interactions, returns {result, stepsTaken}.";

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
            DirectoryName[$InputFileName, 2],  (* up from Kernel/ to wl/ *)
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
    "LogPrint" -> 27, "Grad" -> 28, "ToDevice" -> 29
|>;

(* Tag code → string name *)
$tagName = <|
    0 -> "App", 1 -> "Lam", 2 -> "Var", 3 -> "Sup",
    4 -> "Dp0", 5 -> "Dp1", 6 -> "Era",
    7 -> "Num", 8 -> "Ref", 9 -> "Op2",
    10 -> "Ten", 11 -> "Top", 12 -> "Ctr"
|>;

(* Device index → string name *)
$deviceName = <|0 -> "cpu", 1 -> "metal"|>;

(* UOp code → string name *)
$uopName = <|
    0 -> "Load", 1 -> "Store", 2 -> "Copy",
    3 -> "Neg", 4 -> "Exp", 5 -> "Log", 6 -> "Relu", 7 -> "Cast", 8 -> "Sqrt",
    9 -> "Add", 10 -> "Mul", 11 -> "Div", 12 -> "Max", 13 -> "Cmp", 14 -> "Sub",
    15 -> "Sum", 16 -> "RMax", 17 -> "MatMul",
    18 -> "Reshape", 19 -> "Permute", 20 -> "Expand", 21 -> "Shrink", 22 -> "Pad",
    23 -> "Fusing", 24 -> "Assign", 25 -> "Where", 26 -> "Ifz",
    27 -> "LogPrint", 28 -> "Grad", 29 -> "ToDevice"
|>;

(* Unary ops (second arg = ERA) *)
$unaryOps = {"Neg", "Exp", "Log", "Relu", "Sqrt", "Cast", "RMax", "LogPrint"};

(* ── Conversion between TTerm and TTensor ──────────────────────────────── *)

ToTTensor[TTerm[id_]] := TTensor[id];
ToTTerm[TTensor[id_]] := TTerm[id];

(* Extract ID from either head *)
termId[TTerm[id_]] := id;
termId[TTensor[id_]] := id;

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
thvmNumFn = None;
thvmOp2Fn = None;
thvmNumValueFn = None;
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
thvmViewInfoFn = None;
thvmTensorDeviceFn = None;
thvmToDeviceFn = None;
thvmHeapGraphFn = None;
thvmTraceEnableFn = None;
thvmTraceClearFn = None;
thvmTraceDataFn = None;
thvmReduceStepsFn = None;

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
    thvmNumFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmNum",
        {Integer, Integer}, "Void"];
    thvmOp2Fn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmOp2",
        {Integer, Integer, Integer, Integer}, "Void"];
    thvmNumValueFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmNumValue",
        {Integer}, Integer];
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

    (* Per-tensor device & view *)
    thvmViewInfoFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmViewInfo",
        {Integer}, LibraryDataType[NumericArray]];
    thvmTensorDeviceFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmTensorDevice",
        {Integer}, Integer];
    thvmToDeviceFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmToDevice",
        {Integer, Integer, Integer}, "Void"];

    (* Heap graph visualization *)
    thvmHeapGraphFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmHeapGraph",
        {Integer}, LibraryDataType[NumericArray]];

    (* Interaction tracing & step reduction *)
    thvmTraceEnableFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmTraceEnable",
        {Integer}, "Void"];
    thvmTraceClearFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmTraceClear",
        {}, "Void"];
    thvmTraceDataFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmTraceData",
        {}, LibraryDataType[NumericArray]];
    thvmReduceStepsFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmReduceSteps",
        {Integer, Integer, Integer}, Integer];
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Context lifecycle                                                       *)
(* ════════════════════════════════════════════════════════════════════════ *)

TInit[device_String:"metal"] := Module[{ok},
    loadLibrary[];
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
(* Data transfer — all tensor functions return TTensor                     *)
(* ════════════════════════════════════════════════════════════════════════ *)

TCreate[data_List, shape_List] := Module[{na, out = allocId[]},
    loadLibrary[];
    na = NumericArray[N[Flatten[data]], "Real32"];
    thvmTensorFn[out, na, shape];
    TTensor[out]
];

TCreate[na_NumericArray, shape_List] := Module[{out = allocId[]},
    loadLibrary[];
    thvmTensorFn[out, na, shape];
    TTensor[out]
];

TCreate[data_List] := TCreate[data, Dimensions[data]];

TGet[TTensor[id_Integer]] := (loadLibrary[]; thvmToHostFn[id]);
TGet[TTerm[id_Integer]] := (loadLibrary[]; thvmToHostFn[id]);

TReduce[TTensor[id_Integer]] := Module[{out = allocId[]},
    loadLibrary[];
    thvmReduceFn[id, out];
    TTensor[out]
];

TReduce[TTerm[id_Integer]] := Module[{out = allocId[], tag},
    loadLibrary[];
    thvmReduceFn[id, out];
    tag = thvmTermTagFn[out];
    If[tag === 10 || tag === 11, TTensor[out], TTerm[out]]
];

TRealize[TTensor[id_Integer]] := (loadLibrary[]; thvmRealizeFn[id];);
TRealize[TTerm[id_Integer]] := (loadLibrary[]; thvmRealizeFn[id];);

(* ════════════════════════════════════════════════════════════════════════ *)
(* Queries                                                                 *)
(* ════════════════════════════════════════════════════════════════════════ *)

TDimensions[TTensor[id_Integer]] := (loadLibrary[]; Normal[thvmDimensionsFn[id]]);
TDimensions[TTerm[id_Integer]] := (loadLibrary[]; Normal[thvmDimensionsFn[id]]);

(* TTermTag returns a string, not an integer *)
TTermTag[TTensor[id_Integer]] := Module[{tag},
    loadLibrary[];
    tag = thvmTermTagFn[id];
    Lookup[$tagName, tag, "Unknown"]
];

TTermTag[TTerm[id_Integer]] := Module[{tag},
    loadLibrary[];
    tag = thvmTermTagFn[id];
    Lookup[$tagName, tag, "Unknown"]
];

TTermExt[TTensor[id_Integer]] := (loadLibrary[]; thvmTermExtFn[id]);
TTermExt[TTerm[id_Integer]] := (loadLibrary[]; thvmTermExtFn[id]);

TTensorCount[] := (loadLibrary[]; thvmTensorCountFn[]);
THeapPos[] := (loadLibrary[]; thvmHeapPosFn[]);
TInteractionCount[] := (loadLibrary[]; thvmInteractionCountFn[]);

(* ════════════════════════════════════════════════════════════════════════ *)
(* Device & View queries                                                   *)
(* ════════════════════════════════════════════════════════════════════════ *)

TDevice[TTensor[id_Integer]] := Module[{devIdx},
    loadLibrary[];
    devIdx = thvmTensorDeviceFn[id];
    Lookup[$deviceName, devIdx, "unknown"]
];

TView[TTensor[id_Integer]] := Module[{raw, r, dims, strides, offset, numel, contig, hasMask, result},
    loadLibrary[];
    raw = Normal[thvmViewInfoFn[id]];
    r = Round[raw[[1]]];
    dims = Round[raw[[2 ;; r + 1]]];
    strides = Round[raw[[r + 2 ;; 2 r + 1]]];
    offset = Round[raw[[2 r + 2]]];
    numel = Round[raw[[2 r + 3]]];
    contig = raw[[2 r + 4]] > 0.5;
    hasMask = raw[[2 r + 5]] > 0.5;
    result = <|
        "Shape" -> dims, "Strides" -> strides,
        "Offset" -> offset, "NumEl" -> numel,
        "Contiguous" -> contig, "HasMask" -> hasMask
    |>;
    If[hasMask,
        AssociateTo[result, {
            "MaskBegin" -> Round[raw[[2 r + 6 ;; 3 r + 5]]],
            "MaskEnd" -> Round[raw[[3 r + 6 ;; 4 r + 5]]]
        }]
    ];
    result
];

TToDevice[t_TTensor, device_String] := Module[{out = allocId[], devIdx},
    loadLibrary[];
    devIdx = Switch[device, "cpu", 0, "metal", 1, _, Return[$Failed]];
    thvmToDeviceFn[out, t[[1]], devIdx];
    TTensor[out]
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* TTensor property access via SubValues                                   *)
(* ════════════════════════════════════════════════════════════════════════ *)

TTensor[id_Integer]["Shape"]        := TDimensions[TTensor[id]];
TTensor[id_Integer]["Device"]       := TDevice[TTensor[id]];
TTensor[id_Integer]["DType"]        := TDType[TTensor[id]];
TTensor[id_Integer]["View"]         := TView[TTensor[id]];
TTensor[id_Integer]["Contiguous"]   := TView[TTensor[id]]["Contiguous"];
TTensor[id_Integer]["Materialized"] := TTermTag[TTensor[id]] === "Ten";
TTensor[id_Integer]["Lazy"]         := TTermTag[TTensor[id]] === "Top";
TTensor[id_Integer]["Op"]           := With[{tag = TTermTag[TTensor[id]]},
    If[tag === "Top", Lookup[$uopName, TTermExt[TTensor[id]], "Unknown"], None]
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* TOp dispatch — binary ops                                               *)
(* ════════════════════════════════════════════════════════════════════════ *)

TOp[op_String][a_TTensor, b_TTensor] /;
    KeyExistsQ[$uopCode, op] && !MemberQ[$unaryOps, op] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmOpFn[out, $uopCode[op], a[[1]], b[[1]]];
    TTensor[out]
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* TOp dispatch — unary ops (ERA as second arg)                            *)
(* ════════════════════════════════════════════════════════════════════════ *)

TOp[op_String][a_TTensor] /; MemberQ[$unaryOps, op] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmOpFn[out, $uopCode[op], a[[1]], 0];
    TTensor[out]
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* TOp dispatch — movement ops                                             *)
(* ════════════════════════════════════════════════════════════════════════ *)

TOp["Reshape"][t_TTensor, shape_List] := Module[{out = allocId[]},
    loadLibrary[];
    thvmReshapeFn[out, t[[1]], shape];
    TTensor[out]
];

TOp["Expand"][t_TTensor, shape_List] := Module[{out = allocId[]},
    loadLibrary[];
    thvmExpandFn[out, t[[1]], shape];
    TTensor[out]
];

TOp["Permute"][t_TTensor, axes_List] := Module[{out = allocId[]},
    loadLibrary[];
    thvmPermuteFn[out, t[[1]], axes];
    TTensor[out]
];

TOp["Pad"][t_TTensor, pairs_List] := Module[{out = allocId[]},
    loadLibrary[];
    thvmPadFn[out, t[[1]], Flatten[pairs]];
    TTensor[out]
];

TOp["Shrink"][t_TTensor, pairs_List] := Module[{out = allocId[]},
    loadLibrary[];
    thvmShrinkFn[out, t[[1]], Flatten[pairs]];
    TTensor[out]
];

TOp["Sum"][t_TTensor, axes_List] := Module[{out = allocId[]},
    loadLibrary[];
    thvmSumAxesFn[out, t[[1]], axes];
    TTensor[out]
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Interaction net primitives (TTerm only — not tensor ops)                *)
(* ════════════════════════════════════════════════════════════════════════ *)

TLam[body_TTerm] := Module[{lamId = allocId[], varId = allocId[]},
    loadLibrary[];
    thvmLamFn[lamId, varId, body[[1]]];
    {TTerm[lamId], TTerm[varId]}
];

(* Pure function form: TLam[var |-> body_using_var] *)
TLam[f_Function] := Module[{lam, var, body},
    {lam, var} = TLamOpen[];
    body = f[var];
    If[!MatchQ[body, _TTerm],
        Message[TLam::badret, body]; Return[$Failed]];
    TLamSetBody[lam, body];
    lam
];
TLam::badret = "Lambda body must return TTerm, got `1`.";


TApp[fun_TTerm, arg_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmAppFn[out, fun[[1]], arg[[1]]];
    TTerm[out]
];
(* Auto-convert TTensor to TTerm at inet boundary *)
TApp[fun_TTensor, arg_TTensor] := TApp[ToTTerm[fun], ToTTerm[arg]];
TApp[fun_TTensor, arg_TTerm]   := TApp[ToTTerm[fun], arg];
TApp[fun_TTerm, arg_TTensor]   := TApp[fun, ToTTerm[arg]];

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

TNum[n_Integer] := Module[{out = allocId[]},
    loadLibrary[];
    thvmNumFn[out, n];
    TTerm[out]
];

$op2Code = <|"Add" -> 0, "Sub" -> 1, "Mul" -> 2, "Div" -> 3, "Eq" -> 4, "Mod" -> 5|>;

TOp2[op_String, x_TTerm, y_TTerm] /; KeyExistsQ[$op2Code, op] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmOp2Fn[out, $op2Code[op], x[[1]], y[[1]]];
    TTerm[out]
];

TNumValue[TTerm[id_Integer]] := (loadLibrary[]; thvmNumValueFn[id]);

(* Recursively extract all branches from a nested SUP tree *)
TSupValues[t_TTerm] := Module[{reduced, tag},
    reduced = TReduce[t];
    tag = TTermTag[reduced];
    If[tag === "Sup",
        Module[{dp0, dp1},
            {dp0, dp1} = TDup[reduced];
            Join[TSupValues[dp0], TSupValues[dp1]]
        ],
        {reduced}
    ]
];

(* Extract SUP branches as integer values *)
TSupNumValues[t_TTerm] := TNumValue /@ TSupValues[t];

TDefine[body_TTerm] := (loadLibrary[]; thvmDefineFn[body[[1]]]);

TRef[name_Integer] := Module[{out = allocId[]},
    loadLibrary[];
    thvmRefFn[out, name];
    TTerm[out]
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Advanced ops — accept TTensor                                           *)
(* ════════════════════════════════════════════════════════════════════════ *)

TWhere[cond_TTensor, then_TTensor, else_TTensor] := Module[{out = allocId[]},
    loadLibrary[];
    thvmWhereFn[out, cond[[1]], then[[1]], else[[1]]];
    TTensor[out]
];

TAssign[dst_TTensor, src_TTensor] := Module[{out = allocId[]},
    loadLibrary[];
    thvmAssignFn[out, dst[[1]], src[[1]]];
    TTensor[out]
];

TIfz[counter_TTensor, zeroCase_TTensor, succLam_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmIfzFn[out, counter[[1]], zeroCase[[1]], succLam[[1]]];
    TTensor[out]
];
(* TTerm overloads — lambda variables are TTerm, not TTensor *)
TIfz[counter_TTerm, zeroCase_TTensor, succLam_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmIfzFn[out, counter[[1]], zeroCase[[1]], succLam[[1]]];
    TTensor[out]
];

TLogPrint[t_TTensor] := Module[{out = allocId[]},
    loadLibrary[];
    thvmLogPrintFn[out, t[[1]]];
    TTensor[out]
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Autograd — accept/return TTensor                                        *)
(* ════════════════════════════════════════════════════════════════════════ *)

TSetRequiresGrad[t_TTensor] := (loadLibrary[]; thvmSetRequiresGradFn[t[[1]]];);

TGrad[y_TTensor, x_TTensor] := Module[{out = allocId[]},
    loadLibrary[];
    thvmGradFn[out, y[[1]], x[[1]]];
    TTensor[out]
];

TGradMulti[loss_TTensor, params:{__TTensor}, slots:{__TTensor}] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmGradMultiFn[out, loss[[1]], params[[All, 1]], slots[[All, 1]]];
    TTensor[out]
];

TBackward[loss_TTensor, params:{__TTensor}] :=
Module[{gradIds, n = Length[params]},
    loadLibrary[];
    gradIds = Table[allocId[], n];
    thvmBackwardFn[loss[[1]], params[[All, 1]], gradIds];
    TTensor /@ gradIds
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* CNN layers — accept/return TTensor                                      *)
(* ════════════════════════════════════════════════════════════════════════ *)

TConv2d[x_TTensor, w_TTensor, b_TTensor, groups_Integer:1,
        stride:{_Integer, _Integer}:{1, 1},
        padding:{_Integer, _Integer, _Integer, _Integer}:{0, 0, 0, 0}] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmConv2dFn[out, x[[1]], w[[1]], b[[1]], groups, stride, padding];
    TTensor[out]
];

TConv2d[x_TTensor, w_TTensor, None, groups_Integer:1,
        stride:{_Integer, _Integer}:{1, 1},
        padding:{_Integer, _Integer, _Integer, _Integer}:{0, 0, 0, 0}] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmConv2dFn[out, x[[1]], w[[1]], 0, groups, stride, padding];
    TTensor[out]
];

TMaxPool2d[x_TTensor, kernel:{_Integer, _Integer}:{2, 2},
           stride:{_Integer, _Integer}:{2, 2}] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmMaxPool2dFn[out, x[[1]], kernel, stride];
    TTensor[out]
];

TPool[x_TTensor, kernel_List, stride_List, nSpatial_Integer:2] :=
Module[{out = allocId[]},
    loadLibrary[];
    thvmPoolFn[out, x[[1]], kernel, stride, nSpatial];
    TTensor[out]
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Debug / profiling                                                       *)
(* ════════════════════════════════════════════════════════════════════════ *)

TPrintTerm[t_TTerm] := (loadLibrary[]; thvmPrintTermFn[t[[1]]];);
TPrintTerm[t_TTensor] := (loadLibrary[]; thvmPrintTermFn[t[[1]]];);
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
TLamSetBody[lam_TTerm, body_TTensor] := TLamSetBody[lam, ToTTerm[body]];

(* ════════════════════════════════════════════════════════════════════════ *)
(* UpValues — natural WL syntax on TTensor                                 *)
(* ════════════════════════════════════════════════════════════════════════ *)

TTensor /: Plus[a_TTensor, b_TTensor]  := TOp["Add"][a, b];
TTensor /: Times[a_TTensor, b_TTensor] := TOp["Mul"][a, b];
TTensor /: Times[-1, t_TTensor]        := TOp["Neg"][t];
TTensor /: Times[-1., t_TTensor]       := TOp["Neg"][t];
TTensor /: Dot[a_TTensor, b_TTensor]   := TOp["MatMul"][a, b];
TTensor /: Power[t_TTensor, Rational[1, 2]] := TOp["Sqrt"][t];
TTensor /: Sqrt[t_TTensor]  := TOp["Sqrt"][t];
TTensor /: Exp[t_TTensor]   := TOp["Exp"][t];
TTensor /: Log[t_TTensor]   := TOp["Log"][t];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Formatting — TTensor summary boxes                                      *)
(* ════════════════════════════════════════════════════════════════════════ *)

(* Tag → icon character *)
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

TTensor /: MakeBoxes[t:TTensor[id_Integer], StandardForm] := Module[
    {tag, dims, shapeStr, device, opName,
     visibleItems = {}, hiddenItems = {}, icon},

    tag = Quiet[TTermTag[t]];

    (* Visible: materialized vs lazy *)
    If[tag === "Ten",
        AppendTo[visibleItems, tMakeItem["Status", "Materialized"]],
        If[tag === "Top",
            With[{ext = Quiet[TTermExt[t]]},
                opName = Lookup[$uopName, ext, "UOp" <> ToString[ext]];
                AppendTo[visibleItems, tMakeItem["Op", opName]]
            ],
            AppendTo[visibleItems, tMakeItem["Tag", tag]]
        ]
    ];

    (* Visible: shape *)
    dims = Quiet[TDimensions[t]];
    If[ListQ[dims],
        shapeStr = StringRiffle[ToString /@ dims, "\[ThinSpace]\[Times]\[ThinSpace]"];
        AppendTo[visibleItems, tMakeItem["Shape", shapeStr]]
    ];

    (* Visible: device *)
    If[tag === "Ten",
        device = Quiet[TDevice[t]];
        If[StringQ[device], AppendTo[visibleItems, tMakeItem["Device", device]]]
    ];

    (* Hidden: ID, element count *)
    hiddenItems = {tMakeItem["ID", id]};
    If[tag === "Ten" && ListQ[dims],
        AppendTo[hiddenItems, tMakeItem["Elements", Times @@ dims]]
    ];

    (* Icon: blue for materialized, orange for lazy *)
    icon = Switch[tag,
        "Ten", Graphics[{RGBColor[0.2, 0.6, 0.9], Rectangle[]}, ImageSize -> {28, 28}],
        "Top", Graphics[{RGBColor[0.9, 0.5, 0.2], Polygon[{{0,0},{1,0.5},{0,1}}]}, ImageSize -> {28, 28}],
        _, Graphics[{GrayLevel[0.7], Rectangle[]}, ImageSize -> {28, 28}]
    ];

    InterpretationBox @@ {
        BoxForm`ArrangeSummaryBox[
            TTensor,
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

(* TTerm MakeBoxes — for inet primitives only *)
TTerm /: MakeBoxes[t:TTerm[id_Integer], StandardForm] := Module[
    {tag, tagStr, visibleItems, hiddenItems, icon},

    tag = Quiet[TTermTag[t]];
    tagStr = If[StringQ[tag], tag, "?"];

    visibleItems = {tMakeItem["Tag", tagStr]};

    If[tag === "Era",
        visibleItems = {tMakeItem["Status", "Erased"]}
    ];

    hiddenItems = {tMakeItem["ID", id]};

    icon = Switch[tagStr,
        "Lam", Graphics[{RGBColor[0.5, 0.8, 0.3], Disk[]}, ImageSize -> {28, 28}],
        "App", Graphics[{RGBColor[0.7, 0.3, 0.7], Disk[]}, ImageSize -> {28, 28}],
        "Era", Graphics[{GrayLevel[0.5], Thickness[0.08], Circle[{0.5, 0.5}, 0.35]}, ImageSize -> {28, 28}],
        "Ref", Graphics[{RGBColor[0.8, 0.4, 0.1], Polygon[{{0.2,0},{0.8,0},{1,0.5},{0.8,1},{0.2,1},{0,0.5}}]}, ImageSize -> {28, 28}],
        "Sup", Graphics[{RGBColor[0.3, 0.7, 0.9], Disk[]}, ImageSize -> {28, 28}],
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
(* Single-step reduction & tracing                                         *)
(* ════════════════════════════════════════════════════════════════════════ *)

TTraceEnable[]  := (loadLibrary[]; thvmTraceEnableFn[1]);
TTraceDisable[] := (loadLibrary[]; thvmTraceEnableFn[0]);
TTraceClear[]   := (loadLibrary[]; thvmTraceClearFn[]);

TTrace[] := Module[{raw, n},
    loadLibrary[];
    raw = Round[Normal[thvmTraceDataFn[]]];
    n = Length[raw] / 5;
    Table[<|
        "BeforeTag" -> Lookup[$tagName, raw[[5 i - 4]], "?"],
        "BeforeExt" -> raw[[5 i - 3]],
        "AfterTag"  -> Lookup[$tagName, raw[[5 i - 2]], "?"],
        "AfterExt"  -> raw[[5 i - 1]],
        "RuleId"    -> raw[[5 i]]
    |>, {i, n}]
];

TReduceSteps[t_TTensor, n_Integer] := Module[{out = allocId[], steps, tag},
    loadLibrary[];
    steps = thvmReduceStepsFn[out, t[[1]], n];
    tag = thvmTermTagFn[out];
    {If[tag === 10 || tag === 11, TTensor[out], TTerm[out]], steps}
];
TReduceSteps[t_TTerm, n_Integer] := TReduceSteps[ToTTensor[t], n];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Load subpackages                                                        *)
(* ════════════════════════════════════════════════════════════════════════ *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "Visualization.wl"}]];
Get[FileNameJoin[{DirectoryName[$InputFileName], "Layers.wl"}]];
Get[FileNameJoin[{DirectoryName[$InputFileName], "NetCompile.wl"}]];

End[];
EndPackage[];
