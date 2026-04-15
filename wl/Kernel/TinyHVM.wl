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
TTermVal::usage = "TTermVal[term] returns the VAL field (heap location for compound terms, tensor id for TEN, etc.).";
TTensorCount::usage = "TTensorCount[] returns the number of tensors in the context.";
THeapPos::usage = "THeapPos[] returns the current heap position.";
TInteractionCount::usage = "TInteractionCount[] returns the total interaction count.";

(* ── Interaction net primitives ─────────────────────────────────────────── *)

TLam::usage = "TLam[body] creates a lambda term. Returns {TTerm[lamId], TTerm[varId]}.";
TApp::usage = "TApp[fun, arg] creates an application term.";
TSup::usage = "TSup[a, b] creates a superposition with a fresh label. TSup[label, a, b] uses an explicit label.";
TDup::usage = "TDup[z] duplicates a term with a fresh label. TDup[label, z] uses an explicit label. Returns {TTerm[dp0], TTerm[dp1]}.";
TFreshLabel::usage = "TFreshLabel[] allocates a fresh SUP/DUP label (monotonic counter).";
TDefine::usage = "TDefine[body] registers a named definition. Returns the name (Integer).";
TRef::usage = "TRef[name] creates a reference to a named definition.";
TNum::usage = "TNum[n] creates an integer term (TAG_NUM). n must be a non-negative integer.";
TOp2::usage = "TOp2[op, x, y] creates a binary integer operation (TAG_OP2). op: \"Add\", \"Sub\", \"Mul\", \"Div\", \"Eq\", \"Mod\".";
TNumValue::usage = "TNumValue[term] extracts the integer value from a reduced TAG_NUM term.";
TSupValues::usage = "TSupValues[term] recursively extracts all branches from a nested SUP tree into a flat list.";
TSupNumValues::usage = "TSupNumValues[term] extracts all SUP branches as integer values.";

TBri::usage = "TBri[body] creates a bridge term (θx.body, dual of lambda). Returns {TTerm[briId], TTerm[varId]}.";
TAnn::usage = "TAnn[term, type] creates an annotation term {term : type}. Transparent during reduction.";
TDsu::usage = "TDsu[labelExpr, a, b] creates a dynamic SUP. Label is an expression reduced at interaction time.";
TDdu::usage = "TDdu[labelExpr, val, bod] creates a dynamic DUP. Label is an expression, bod receives both copies.";
TInc::usage = "TInc[term] creates a priority wrapper. Transparent to reduce, lower priority in collapse.";
TCollapse::usage = "TCollapse[term] extracts all branches from a superposed term into a flat list (priority-queue, respects INC).";
TCollapsePar::usage = "TCollapsePar[term, nThreads] parallel work-stealing collapse. Falls back to TCollapse for nThreads<=1.";
TCollapseGrouped::usage = "TCollapseGrouped[term] collapses and returns <|\"values\" -> {TTerm...}, \"bf\" -> {<|label->branch,...|>...}|>.  Each bf (branch function) is an Association mapping SUP labels to branch choices (0=left, 1=right).";
TGroupBy::usage = "TGroupBy[gr, {labels}] partitions grouped collapse results by branch choices on the given label subset.  Returns <|{b1,b2,...} -> {values...}, ...|>.";
TBranchFunction::usage = "TBranchFunction[gr, i] returns the branch function (label->branch Association) for the i-th result.";
TGroupings::usage = "TGroupings[{a,b,c,...}, f] builds a single SUP term encoding all Groupings (binary tree parenthesizations) of the elements under f.  With TTerm elements, atoms are shared (OK for TNum etc).  With Function elements (factories), each leaf calls its factory for a fresh TTerm — correct for lambdas/SUPs.";

(* ── Synthesis ─────────────────────────────────────────────────────────── *)

FindBooleanAlternative::usage = "FindBooleanAlternative[expr, {op1, op2, ...}] finds Boolean expressions equivalent to expr using only the given operators.  Method->\"TinyHVM\" uses superposition synthesis (all candidates reduced in parallel via DUP-SUP annihilation).";

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
TDotGraph::usage = "TDotGraph[tensor|term] returns a dump.c-style Graph (boxed TOP/TEN nodes, port labels, heap-loc tail labels).";
TInteractionGraph::usage = "TInteractionGraph[term] returns an interaction net graph with the next active pair highlighted in red.";

(* ── Heap introspection ──────────────────────────────────────────────── *)

THeap::usage = "THeap[<|...|>] represents a snapshot of the TinyHVM heap state. Access properties via THeap[h][\"Key\"].";
TInteraction::usage = "TInteraction[<|...|>] represents a single interaction event with rule name and before/after tags.";
THeapSnapshot::usage = "THeapSnapshot[] captures the current heap state as a THeap object.";
THeapRead::usage = "THeapRead[loc] reads a term at a heap location. Returns <|\"Tag\", \"TagCode\", \"Ext\", \"Val\", \"Loc\"|>.";
THeapReadRange::usage = "THeapReadRange[lo, count] reads `count` consecutive heap slots starting at `lo`. Returns a list of THeapRead-style associations.";
TStepReduce::usage = "TStepReduce[term] reduces exactly one interaction. Returns {result, TInteraction, THeap}.";

(* ── Single-step reduction & tracing ──────────────────────────────────── *)

TTraceEnable::usage = "TTraceEnable[] enables interaction tracing.";
TTraceDisable::usage = "TTraceDisable[] disables interaction tracing.";
TTraceClear::usage = "TTraceClear[] clears the trace buffer.";
TTrace::usage = "TTrace[] returns a list of TInteraction objects from the trace buffer.";
TReduceSteps::usage = "TReduceSteps[tensor, n] reduces at most n interactions, returns {result, stepsTaken}.";
TStep::usage = "TStep[tensor] reduces until the walker-visible graph changes (skipping admin reductions). Returns {nextState, interactionsFired}.";
TStepTrace::usage = "TStepTrace[tensor, maxSteps] repeatedly applies TStep, returning the list of visibly distinct states from the initial term to fixed point or maxSteps visible changes.";

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
    "Kernel" -> 23, "Assign" -> 24, "Where" -> 25, "Ifz" -> 26,
    "LogPrint" -> 27, "Grad" -> 28, "ToDevice" -> 29,
    "Fuse" -> 31, "Exec" -> 33
|>;

(* Tag code → string name. Mirrors TAG_* in src/tinyhvm.h. *)
$tagName = <|
    0 -> "App", 1 -> "Lam", 2 -> "Var", 3 -> "Sup",
    4 -> "Dp0", 5 -> "Dp1", 6 -> "Era",
    7 -> "Num", 8 -> "Ref", 9 -> "Op2",
    10 -> "Ten", 11 -> "Top", 12 -> "Ctr",
    13 -> "Bri", 14 -> "Ann",
    15 -> "Dsu", 16 -> "Ddu", 17 -> "Inc",
    18 -> "Eql", 19 -> "And", 20 -> "Or",
    21 -> "Mat", 22 -> "Any",
    23 -> "Usp", 24 -> "Udp",
    25 -> "Seq", 26 -> "Alo"
|>;

(* Device index → string name *)
$deviceName = <|0 -> "cpu", 1 -> "metal"|>;

(* UOp code → string name. Mirrors UOP_* in src/tinyhvm.h. *)
$uopName = <|
    0 -> "Load", 1 -> "Store", 2 -> "Copy",
    3 -> "Neg", 4 -> "Exp", 5 -> "Log", 6 -> "Relu", 7 -> "Cast", 8 -> "Sqrt",
    9 -> "Add", 10 -> "Mul", 11 -> "Div", 12 -> "Max", 13 -> "Cmp", 14 -> "Sub",
    15 -> "Sum", 16 -> "RMax", 17 -> "MatMul",
    18 -> "Reshape", 19 -> "Permute", 20 -> "Expand", 21 -> "Shrink", 22 -> "Pad",
    23 -> "Kernel", 24 -> "Assign", 25 -> "Where", 26 -> "Ifz",
    27 -> "LogPrint", 28 -> "Grad", 29 -> "ToDevice",
    30 -> "Detach", 31 -> "Fuse", 32 -> "Legacy32", 33 -> "Exec"
|>;

(* UOp categories (mirrors src/fuse/_.c is_view_op / is_elementwise / is_binary) *)
$viewOps        = {"Reshape", "Permute", "Expand", "Shrink", "Pad"};
$binaryOps      = {"Add", "Sub", "Mul", "Div", "Max", "Cmp"};
$elementwiseOps = Join[$binaryOps, {"Neg", "Relu", "Exp", "Log", "Sqrt", "Cast"}];
$reduceOps      = {"Sum", "RMax"};

(* Unary ops (second arg = ERA) *)
$unaryOps = {"Neg", "Exp", "Log", "Relu", "Sqrt", "Cast", "RMax",
             "LogPrint", "Fuse"};

(* Heap-tag arity: number of consecutive heap slots starting at term_val(t).
   Mirrors dot_visible_heap_loc_tag + per-tag layout in dump.c.
   Atoms (TEN, NUM, REF, VAR, ERA, ANY) → 0. *)
$heapTagArity = <|
    "App" -> 2, "Lam" -> 2, "Bri" -> 2,
    "Sup" -> 2, "Usp" -> 2,
    "Dp0" -> 2, "Dp1" -> 2,  (* share dup_loc; a/b at loc, loc+1 *)
    "Op2" -> 2,
    "Eql" -> 2, "And" -> 2, "Or" -> 2,
    "Mat" -> 2,
    "Ann" -> 2,
    "Dsu" -> 3, "Ddu" -> 3,
    "Udp" -> 1, "Inc" -> 1,
    "Seq" -> 2,
    "Ctr" -> 2,
    "Alo" -> 2,
    "Top" -> Automatic   (* arity depends on uop; see uopArity *)
|>;

(* UOp arity (number of heap slots at term_val of a TOP node).
   Unary ops still allocate 2 slots; the second is ERA. *)
uopArity[uop_String] := Which[
    MemberQ[$viewOps, uop],            2,                   (* in, shape *)
    MemberQ[$binaryOps, uop],          2,
    uop === "MatMul",                   2,
    uop === "Sum" || uop === "RMax",    2,                   (* in, axes *)
    uop === "Grad",                     2,                   (* y, gy *)
    uop === "Where",                    3,
    uop === "Ifz",                      3,
    uop === "Assign",                   2,
    uop === "Kernel",                   3,                   (* left, right/meta, NUM(root_uop) *)
    uop === "Fuse",                     1,
    uop === "Exec",                     3,                   (* NUM(kid), deps, NUM(flags) *)
    uop === "Detach" || uop === "LogPrint" || uop === "ToDevice", 1,
    True,                                1
];

(* Port name for slot index i of a TOP/<uop>. Mirrors dot_uop_port_name. *)
uopPortName[uop_String, i_Integer] := Which[
    uop === "Sum" || uop === "RMax",   If[i == 0, "in", "axes"],
    uop === "Grad",                     If[i == 0, "y", "gy"],
    MemberQ[$viewOps, uop],             If[i == 0, "in", "shape"],
    MemberQ[$binaryOps, uop] || uop === "MatMul", If[i == 0, "a", "b"],
    uop === "Kernel",                   If[i == 0, "a", "b"],
    True, "in"
];

(* Port name for slot index i of a heap tag. Mirrors dot_heap_port_name. *)
heapPortName[tag_String, i_Integer] := Which[
    tag === "Lam" || tag === "Bri",     If[i == 0, "var", "body"],
    tag === "Sup" || tag === "Usp",     If[i == 0, "a", "b"],
    tag === "Mat",                      If[i == 0, "ok", "fb"],
    tag === "Ann",                      If[i == 0, "term", "type"],
    tag === "Dsu" || tag === "Ddu",     {"label", "a", "b"}[[i + 1]],
    tag === "Udp" || tag === "Inc",     "in",
    tag === "Seq",                      If[i == 0, "first", "then"],
    tag === "Alo",                      If[i == 0, "book", "env"],
    True,                               If[i == 0, "a", "b"]
];

(* dump.c node colors, keyed by tag name. *)
$heapTagFill = <|
    "App" -> "#f3f3f3",
    "Sup" -> "#e4d6fc", "Usp" -> "#e4d6fc",
    "Mat" -> "#eaf2ff",
    "Lam" -> "#f2e8ff", "Bri" -> "#f2e8ff",
    "Ref" -> "#eeeeee", "Var" -> "#eeeeee",
    "Alo" -> "#e8f7ff",
    "Top" -> "#cce5ff",     (* dump.c uses #cce5ff for TOP boxes (e.g. ADD/MUL) *)
    "Ten" -> "#e0e0e0",     (* tensor leaves *)
    "Era" -> "#ffffff"
|>;

(* dump.c node shapes. *)
$heapTagShape = <|
    "Lam" -> "Triangle", "App" -> "InvTriangle",
    "Sup" -> "Hexagon", "Usp" -> "Hexagon", "Ctr" -> "Hexagon",
    "Ref" -> "Oval", "Var" -> "Oval",
    "Alo" -> "Box3d"
|>;

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
thvmTermValFn = None;
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
thvmFreshLabelFn = None;
thvmBriFn = None;
thvmAnnFn = None;
thvmDsuFn = None;
thvmDduFn = None;
thvmIncFn = None;
thvmCollapseFn = None;
thvmCollapseParFn = None;
thvmCollapseGroupedFn = None;
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
thvmStepToNextVisibleFn = None;
thvmHeapReadFn = None;
thvmHeapReadRangeFn = None;
thvmHeapSnapshotFn = None;
thvmNextInteractionFn = None;

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
    thvmTermValFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmTermVal",
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
        {Integer, Integer, Integer, Integer}, "Void"];
    thvmDupFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmDup",
        {Integer, Integer, Integer, Integer}, "Void"];
    thvmFreshLabelFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmFreshLabel",
        {}, Integer];
    thvmBriFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmBri",
        {Integer, Integer}, Integer];
    thvmAnnFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmAnn",
        {Integer, Integer, Integer}, "Void"];
    thvmDsuFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmDsu",
        {Integer, Integer, Integer, Integer}, "Void"];
    thvmDduFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmDdu",
        {Integer, Integer, Integer, Integer}, "Void"];
    thvmIncFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmInc",
        {Integer, Integer}, "Void"];
    thvmCollapseFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmCollapse",
        {Integer, Integer}, Integer];
    thvmCollapseParFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmCollapsePar",
        {Integer, Integer, Integer}, Integer];
    thvmCollapseGroupedFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmCollapseGrouped",
        {Integer, Integer}, {Integer, 1}];
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
    thvmStepToNextVisibleFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmStepToNextVisible",
        {Integer, Integer, Integer}, Integer];

    (* Heap introspection *)
    thvmHeapReadFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmHeapRead",
        {Integer}, {Integer, 1}];
    thvmHeapReadRangeFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmHeapReadRange",
        {Integer, Integer}, {Integer, 2}];
    thvmHeapSnapshotFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmHeapSnapshot",
        {}, {Integer, 1}];
    thvmNextInteractionFn = LibraryFunctionLoad[$TinyHVMLibrary, "thvmNextInteraction",
        {Integer}, {Integer, 1}];
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Context lifecycle                                                       *)
(* ════════════════════════════════════════════════════════════════════════ *)

TInit[device_String:"cpu"] := Module[{ok},
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

TTermVal[TTensor[id_Integer]] := (loadLibrary[]; thvmTermValFn[id]);
TTermVal[TTerm[id_Integer]] := (loadLibrary[]; thvmTermValFn[id]);

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

(* TSup[a, b] — fresh label; TSup[label, a, b] — explicit label *)
TSup[a_TTerm, b_TTerm] := TSup[TFreshLabel[], a, b];
TSup[label_Integer, a_TTerm, b_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmSupFn[out, label, a[[1]], b[[1]]];
    TTerm[out]
];

TFreshLabel[] := (loadLibrary[]; thvmFreshLabelFn[]);

(* TDup[z] — fresh label; TDup[label, z] — explicit label *)
TDup[z_TTerm] := TDup[TFreshLabel[], z];
TDup[label_Integer, z_TTerm] := Module[{dp0 = allocId[], dp1 = allocId[]},
    loadLibrary[];
    thvmDupFn[label, z[[1]], dp0, dp1];
    {TTerm[dp0], TTerm[dp1]}
];

(* TBri[body] — create bridge (θx.body), returns {TTerm[bri], TTerm[var]} *)
TBri[body_TTerm] := Module[{out = allocId[], varId},
    loadLibrary[];
    varId = thvmBriFn[out, body[[1]]];
    {TTerm[out], TTerm[varId]}
];

(* Pure function form: TBri[var |-> body_using_var] — same pattern as TLam *)
TBri[f_Function] := Module[{bri, var, body},
    {bri, var} = TBri[TNum[0]];  (* placeholder body *)
    body = f[var];
    If[!MatchQ[body, _TTerm],
        Message[TBri::badret, body]; Return[$Failed]];
    TLamSetBody[bri, body];  (* BRI has same heap layout as LAM *)
    bri
];
TBri::badret = "Bridge body must return TTerm, got `1`.";

(* TAnn[term, type] — create annotation {term : type} *)
TAnn[term_TTerm, type_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmAnnFn[out, term[[1]], type[[1]]];
    TTerm[out]
];

(* TDsu[labelExpr, a, b] — dynamic SUP (label is an expression) *)
TDsu[label_TTerm, a_TTerm, b_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmDsuFn[out, label[[1]], a[[1]], b[[1]]];
    TTerm[out]
];

(* TDdu[labelExpr, val, bod] — dynamic DUP (label is an expression) *)
TDdu[label_TTerm, val_TTerm, bod_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmDduFn[out, label[[1]], val[[1]], bod[[1]]];
    TTerm[out]
];

(* TInc[term] — priority wrapper (transparent to reduce, lower priority in collapse) *)
TInc[term_TTerm] := Module[{out = allocId[]},
    loadLibrary[];
    thvmIncFn[out, term[[1]]];
    TTerm[out]
];

(* TCollapse[term] — PQ-based collapse, respects INC priority *)
TCollapse[t_TTerm] := Module[{base, count},
    loadLibrary[];
    base = allocId[];
    $nextId += 255;
    count = thvmCollapseFn[t[[1]], base];
    Table[TTerm[base + i], {i, 0, count - 1}]
];

(* TCollapsePar[term, nThreads] — parallel work-stealing collapse *)
TCollapsePar[t_TTerm, nThreads_Integer:4] := Module[{base, count},
    loadLibrary[];
    base = allocId[];
    $nextId += 255;
    count = thvmCollapseParFn[t[[1]], base, nThreads];
    Table[TTerm[base + i], {i, 0, count - 1}]
];

(* TCollapseGrouped[term] — collapse with label-path tracking.
   Returns <|"values" -> {TTerm...}, "bf" -> {<|label->branch,...|>...}|>
   Each bf (branch function) maps SUP labels to branch choices (0=left, 1=right).
   Recover bf for result i:  gr["bf"][[i]]
   Look up branch for label L: gr["bf"][[i]][L] *)
TCollapseGrouped[t_TTerm] := Module[{base, packed, count, pos, values, bfs},
    loadLibrary[];
    base = allocId[];
    $nextId += 255;
    packed = thvmCollapseGroupedFn[t[[1]], base];
    count = packed[[1]];
    values = Table[TTerm[base + i], {i, 0, count - 1}];
    pos = 2;
    bfs = Table[Module[{plen, bf},
        plen = packed[[pos]]; pos++;
        bf = Association @@ Table[
            packed[[pos + 2 j]] -> packed[[pos + 2 j + 1]],
            {j, 0, plen - 1}];
        pos += 2 plen + 1;
        bf
    ], {i, count}];
    <|"values" -> values, "bf" -> bfs|>
];

(* TGroupBy[gr, {labels}] — partition results by branch choices on label subset.
   Returns <|{b1,b2,...} -> {values...}, ...|> *)
TGroupBy[gr_Association, labels_List] :=
    GroupBy[
        Thread[{gr["bf"], gr["values"]}],
        Lookup[First[#], labels] &,
        Map[Last]
    ];

(* TBranchFunction[gr, i] — the branch function for the i-th collapsed result *)
TBranchFunction[gr_Association, i_Integer] := gr["bf"][[i]];

(* TGroupings[elems, f] — Groupings as a lazy SUP term.
   Builds one superposed TTerm encoding all C(n-1) parenthesizations (Catalan number).
   f must be (TTerm, TTerm) -> TTerm, e.g. TOp2["Add", ##]&.
   Each structural choice (split point) gets a fresh label — independent dimensions.
   Collapse to enumerate; TCollapseGrouped to recover which splits were chosen.

   TTerm form: elements are atoms (TNum etc), shared freely across alternatives.
   Function form: elements are factories called at each leaf — correct for non-atomic
   elements (lambdas, SUPs) that cannot be shared across collapse branches. *)
TGroupings[{x_TTerm}, _] := x;
TGroupings[{x_TTerm, y_TTerm}, f_] := f[x, y];
TGroupings[elems:{__TTerm}, f_] := With[{
    alts = Table[
        f[TGroupings[elems[[;; k]], f], TGroupings[elems[[k + 1 ;;]], f]],
        {k, 1, Length[elems] - 1}]},
    iSupTree[alts]
];
(* Factory form: each element is a zero-arg Function creating a fresh TTerm *)
TGroupings[facs:{__Function}, f_] := iGroupFacs[facs, f];
iGroupFacs[{fac_Function}, _] := fac[];
iGroupFacs[{f1_Function, f2_Function}, f_] := f[f1[], f2[]];
iGroupFacs[facs:{__Function}, f_] := With[{
    alts = Table[
        f[iGroupFacs[facs[[;; k]], f], iGroupFacs[facs[[k + 1 ;;]], f]],
        {k, 1, Length[facs] - 1}]},
    iSupTree[alts]
];
(* Encode a list of alternatives as a balanced binary SUP tree *)
iSupTree[{x_}] := x;
iSupTree[alts_List] := With[{mid = Ceiling[Length[alts] / 2]},
    TSup[TFreshLabel[], iSupTree[alts[[;; mid]]], iSupTree[alts[[mid + 1 ;;]]]]
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

(* Recursively extract all branches from a nested SUP tree.
   Delegates to TCollapse (C-side) which uses correct labels for DUP-SUP annihilation. *)
TSupValues[t_TTerm] := TCollapse[t];

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
        "Era", Graphics[{LightDarkSwitched[GrayLevel[0.5], GrayLevel[0.6]], Thickness[0.08], Circle[{0.5, 0.5}, 0.35]}, ImageSize -> {28, 28}],
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
    n = Length[raw] / 7;
    Table[
        iMakeInteraction[
            raw[[7 i - 6]], raw[[7 i - 5]],
            raw[[7 i - 4]], raw[[7 i - 3]],
            raw[[7 i - 2]], raw[[7 i - 1]], raw[[7 i]]
        ],
    {i, n}]
];

TReduceSteps[t_TTensor, n_Integer] := Module[{out = allocId[], steps, tag},
    loadLibrary[];
    steps = thvmReduceStepsFn[out, t[[1]], n];
    tag = thvmTermTagFn[out];
    {If[tag === 10 || tag === 11, TTensor[out], TTerm[out]], steps}
];
TReduceSteps[t_TTerm, n_Integer] := TReduceSteps[ToTTensor[t], n];

(* TStep: reduce until the walker-visible graph changes (skipping admin
   reductions that don't affect the root-visible tree). Default budget 100
   admin interactions. Returns {nextState, fired}. *)
TStep[t_TTensor, maxAttempts_Integer: 100] := Module[{out = allocId[], fired, tag},
    loadLibrary[];
    fired = thvmStepToNextVisibleFn[out, t[[1]], maxAttempts];
    tag = thvmTermTagFn[out];
    {If[tag === 10 || tag === 11, TTensor[out], TTerm[out]], fired}
];
TStep[t_TTerm, args___] := TStep[ToTTensor[t], args];

(* Signature from actual walker output — what TDotGraph renders. Two states
   with identical node/edge sets hash to the same sig. *)
walkerGraphSig[t_] := Module[{w = heapWalk[rootTermOf[t]]},
    Hash[{
        Sort[KeyValueMap[{#1, #2["Tag"], #2["Ext"]} &, w["Nodes"]]],
        Sort[Map[{#["From"], #["To"], #["Port"]} &, w["Edges"]]]
    }]];

(* TStepTrace: repeatedly apply TStep and snapshot the walker output at
   each step. Returns a list of <|"Term", "Walk"|> records — the walk is
   captured at the moment of the step, so later reductions mutating heap
   don't invalidate earlier snapshots.

   Stops at fixed point, on dispatch to a pure tensor (TAG_TEN), or when
   maxSteps reached. Pure public-graph reduction only — no materialization. *)
TStepTrace[t_, maxSteps_Integer: 50] := Module[
    {cur = t, acc, sigs, nxt, fired, tag, sig, walk, i = 0},
    walk = heapWalk[rootTermOf[t]];
    sig = Hash[{
        Sort[KeyValueMap[{#1, #2["Tag"], #2["Ext"]} &, walk["Nodes"]]],
        Sort[Map[{#["From"], #["To"], #["Port"]} &, walk["Edges"]]]}];
    acc = {<|"Term" -> t, "Walk" -> walk|>};
    sigs = {sig};
    While[i++ < maxSteps,
        {nxt, fired} = TStep[cur, 200];
        If[fired == 0, Break[]];
        tag = thvmTermTagFn[nxt[[1]]];
        If[tag === 10, Break[]];
        walk = heapWalk[rootTermOf[nxt]];
        sig = Hash[{
            Sort[KeyValueMap[{#1, #2["Tag"], #2["Ext"]} &, walk["Nodes"]]],
            Sort[Map[{#["From"], #["To"], #["Port"]} &, walk["Edges"]]]}];
        cur = nxt;
        If[MemberQ[sigs, sig], Continue[]];
        AppendTo[acc, <|"Term" -> nxt, "Walk" -> walk|>];
        AppendTo[sigs, sig]];
    acc
];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Heap introspection — THeap, TInteraction, TStepReduce, THeapRead        *)
(* ════════════════════════════════════════════════════════════════════════ *)

(* ── Interaction rule classification ─────────────────────────────────── *)

$interactionRuleName = <|
    {"App", "Lam"} -> "Beta",
    {"App", "Sup"} -> "APP-SUP Commutation",
    {"App", "Era"} -> "APP-ERA Erasure",
    {"App", "Ten"} -> "APP-TEN Sequencing",
    {"App", "Num"} -> "APP-NUM Sequencing",
    {"App", "Bri"} -> "APP-BRI Commutation",
    {"Dp0", "Sup"} -> "DUP-SUP Annihilation",
    {"Dp1", "Sup"} -> "DUP-SUP Annihilation",
    {"Dp0", "Lam"} -> "DUP-LAM Commutation",
    {"Dp1", "Lam"} -> "DUP-LAM Commutation",
    {"Dp0", "Era"} -> "DUP-ERA Erasure",
    {"Dp1", "Era"} -> "DUP-ERA Erasure",
    {"Dp0", "Num"} -> "DUP-NUM Copy",
    {"Dp1", "Num"} -> "DUP-NUM Copy",
    {"Dp0", "Ten"} -> "DUP-TEN Copy",
    {"Dp1", "Ten"} -> "DUP-TEN Copy",
    {"Dp0", "Dp0"} -> "DUP-DUP Commutation",
    {"Dp0", "Dp1"} -> "DUP-DUP Commutation",
    {"Dp1", "Dp0"} -> "DUP-DUP Commutation",
    {"Dp1", "Dp1"} -> "DUP-DUP Commutation",
    {"Op2", "Num"} -> "OP2-NUM Compute",
    {"Op2", "Sup"} -> "OP2-SUP Commutation",
    {"Ann", "Bri"} -> "ANN-BRI Annihilation",
    {"Ann", "Lam"} -> "ANN-LAM Commutation",
    {"Ref", _}     -> "REF Expansion"
|>;

iRuleName[beforeTag_String, afterTag_String] :=
    Lookup[$interactionRuleName, Key[{beforeTag, afterTag}],
        Lookup[$interactionRuleName, Key[{beforeTag, _}],
            Which[
                beforeTag === afterTag, "Annihilation",
                beforeTag === "Era" || afterTag === "Era", "Erasure",
                True, "Commutation"
            ]
        ]
    ];

iRuleColor[ruleName_String] := Which[
    StringContainsQ[ruleName, "Annihilation" | "Beta" | "Compute" | "Sequencing"], RGBColor[0.3, 0.7, 0.3],
    StringContainsQ[ruleName, "Commutation" | "Expansion" | "Copy"], RGBColor[0.9, 0.6, 0.2],
    StringContainsQ[ruleName, "Erasure"], GrayLevel[0.5],
    True, GrayLevel[0.6]
];

(* Build a TInteraction from raw trace fields *)
iMakeInteraction[beforeTagCode_, beforeExt_, afterTagCode_, afterExt_,
                 ruleId_, beforeLoc_, afterLoc_] :=
Module[{bt, at, rn},
    bt = Lookup[$tagName, beforeTagCode, "?"];
    at = Lookup[$tagName, afterTagCode, "?"];
    rn = iRuleName[bt, at];
    TInteraction[<|
        "RuleName"   -> rn,
        "BeforeTag"  -> bt,
        "AfterTag"   -> at,
        "BeforeExt"  -> beforeExt,
        "AfterExt"   -> afterExt,
        "RuleId"     -> ruleId,
        "BeforeLoc"  -> beforeLoc,
        "AfterLoc"   -> afterLoc
    |>]
];

(* ── TInteraction SubValues and MakeBoxes ────────────────────────────── *)

TInteraction[data_Association][key_String] := data[key];

TInteraction /: MakeBoxes[t:TInteraction[data_Association], StandardForm] := Module[
    {ruleName, beforeTag, afterTag, visibleItems, hiddenItems, icon, col},

    ruleName = Lookup[data, "RuleName", "?"];
    beforeTag = Lookup[data, "BeforeTag", "?"];
    afterTag = Lookup[data, "AfterTag", "?"];
    col = iRuleColor[ruleName];

    visibleItems = {
        tMakeItem["Rule", ruleName],
        tMakeItem["Transition", beforeTag <> " \[RightArrow] " <> afterTag]
    };

    hiddenItems = {
        tMakeItem["RuleId", Lookup[data, "RuleId", 0]],
        tMakeItem["BeforeExt", Lookup[data, "BeforeExt", 0]],
        tMakeItem["AfterExt", Lookup[data, "AfterExt", 0]],
        tMakeItem["BeforeLoc", Lookup[data, "BeforeLoc", 0]],
        tMakeItem["AfterLoc", Lookup[data, "AfterLoc", 0]]
    };

    icon = Graphics[{col, EdgeForm[{Thick, col}],
        Polygon[{{0, 0}, {1, 0.5}, {0, 1}}]}, ImageSize -> {28, 28}];

    InterpretationBox @@ {
        BoxForm`ArrangeSummaryBox[
            TInteraction, t, icon,
            visibleItems, hiddenItems, StandardForm
        ],
        t,
        Selectable -> False, Editable -> False, SelectWithContents -> True
    }
];

(* ── THeap SubValues and MakeBoxes ───────────────────────────────────── *)

THeap[data_Association][key_String] := data[key];

THeapSnapshot[] := Module[{raw},
    loadLibrary[];
    raw = thvmHeapSnapshotFn[];
    (* Subtract sentinels: heap_pos starts at 1, tensor_count starts at 1 *)
    THeap[<|
        "HeapSize"         -> raw[[1]] - 1,
        "InteractionCount" -> raw[[2]],
        "TensorCount"      -> raw[[3]] - 1,
        "NextLabel"        -> raw[[4]],
        "DefCount"         -> raw[[5]],
        "Timestamp"        -> AbsoluteTime[]
    |>]
];

THeap /: MakeBoxes[t:THeap[data_Association], StandardForm] := Module[
    {visibleItems, hiddenItems, icon},

    visibleItems = {
        tMakeItem["Heap", ToString[Lookup[data, "HeapSize", 0]] <> " words"],
        tMakeItem["Interactions", Lookup[data, "InteractionCount", 0]]
    };

    hiddenItems = {
        tMakeItem["Tensors", Lookup[data, "TensorCount", 0]],
        tMakeItem["NextLabel", Lookup[data, "NextLabel", 0]],
        tMakeItem["Definitions", Lookup[data, "DefCount", 0]]
    };

    icon = Graphics[{RGBColor[0.2, 0.7, 0.7], EdgeForm[{LightDarkSwitched[GrayLevel[0.3], GrayLevel[0.6]]}],
        Rectangle[{0, 0}, {1, 0.3}], Rectangle[{0, 0.35}, {1, 0.65}],
        Rectangle[{0, 0.7}, {1, 1}]}, ImageSize -> {28, 28}];

    InterpretationBox @@ {
        BoxForm`ArrangeSummaryBox[
            THeap, t, icon,
            visibleItems, hiddenItems, StandardForm
        ],
        t,
        Selectable -> False, Editable -> False, SelectWithContents -> True
    }
];

(* ── THeapRead ───────────────────────────────────────────────────────── *)

THeapRead[loc_Integer] := Module[{raw},
    loadLibrary[];
    raw = thvmHeapReadFn[loc];
    <|"Tag" -> Lookup[$tagName, raw[[1]], "?"],
      "TagCode" -> raw[[1]],
      "Ext" -> raw[[2]],
      "Val" -> raw[[3]],
      "Loc" -> loc|>
];

(* Bulk read [lo, lo+count). Returns a list of associations like THeapRead. *)
THeapReadRange[lo_Integer, count_Integer] := Module[{raw},
    loadLibrary[];
    raw = Normal[thvmHeapReadRangeFn[lo, count]];
    Table[
        <|"Tag" -> Lookup[$tagName, raw[[i, 1]], "?"],
          "TagCode" -> raw[[i, 1]],
          "Ext" -> raw[[i, 2]],
          "Val" -> raw[[i, 3]],
          "Loc" -> lo + i - 1|>,
        {i, count}]
];

(* ── TStepReduce ─────────────────────────────────────────────────────── *)

TStepReduce[t_TTensor] := Module[{result, steps, traces, interaction, heap},
    loadLibrary[];
    TTraceEnable[]; TTraceClear[];
    {result, steps} = TReduceSteps[t, 1];
    traces = TTrace[];
    TTraceDisable[];
    interaction = If[Length[traces] > 0, First[traces], None];
    heap = THeapSnapshot[];
    {result, interaction, heap}
];
TStepReduce[t_TTerm] := TStepReduce[ToTTensor[t]];

(* ════════════════════════════════════════════════════════════════════════ *)
(* Load subpackages                                                        *)
(* ════════════════════════════════════════════════════════════════════════ *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "Heap.wl"}]];
Get[FileNameJoin[{DirectoryName[$InputFileName], "Visualization.wl"}]];
Get[FileNameJoin[{DirectoryName[$InputFileName], "Layers.wl"}]];
Get[FileNameJoin[{DirectoryName[$InputFileName], "NetCompile.wl"}]];
Get[FileNameJoin[{DirectoryName[$InputFileName], "Synthesis.wl"}]];

End[];
EndPackage[];
