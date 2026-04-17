(* Visualization.wl — Graph visualization and profiling for TinyHVM *)
(* Get'd from TinyHVM.wl inside Begin["`Private`"]. All public symbols declared there. *)

(* $tagName is defined in TinyHVM.wl — do not redefine here *)

(* ── TINetGraph — dump.c-style renderer (boxed nodes, port labels, heap locs) ── *)

(* NUM carries ext=0 for u32, ext=1 for f32. Render the raw value accordingly. *)
iNumText[ext_Integer, val_Integer] := If[ext == 1,
    Quiet@Check[
        ToString[First[ImportByteArray[
            ByteArray[IntegerDigits[BitAnd[val, 16^^FFFFFFFF], 256, 4]],
            "Real32"]]],
        ToString[val]],
    ToString[val]];

(* Pull chain from cached KernelEntry's FusedOp[] via thvmKernelOpChainFn. *)
iDotKernelOpChain[val_Integer] := Module[{s, topTagCode},
    topTagCode = Lookup[$tagCode, "Top", Missing["NotFound"]];
    If[!IntegerQ[topTagCode], Return[""]];
    s = Quiet@Check[thvmKernelOpChainFn[topTagCode, $uopCode["Kernel"], val], ""];
    If[StringQ[s], ToUpperCase[s], ""]];

(* Infer shape of a compound term at heap base `val` by walking to first TEN leaf. *)
iDotInferShape[val_Integer] := Module[{seen = <||>, go, dims},
    go[loc_] := If[loc <= 0 || KeyExistsQ[seen, loc], None,
        seen[loc] = True;
        Module[{n = Quiet@Check[THeapRead[loc], None], arity, i, d},
            If[!AssociationQ[n], Return[None, Module]];
            Which[
                n["Tag"] === "Ten",
                    Quiet[TDimensions[TTensor[n["Val"]]]],
                n["Tag"] === "Top" || KeyExistsQ[$heapTagArity, n["Tag"]],
                    arity = termChildCount[n];
                    Do[d = go[n["Val"] + i];
                       If[ListQ[d], Return[d, Module]],
                       {i, 0, arity - 1}]; None,
                True, None]]];
    dims = go[val];
    If[ListQ[dims], dims, None]];

(* Multi-line label for a graph node, matching dump.c's \n-separated fields.
   Optional `nodeRec` gives the captured-at-walk derived fields
   (KernelOpChain, Shape) so trace replays don't read mutated heap. *)
iDotLabelLines[tag_String, uop_String, ext_, val_, hloc_, nodeRec_:<||>] := Switch[tag,
    "Top",
        Module[{uopName = uop,
                opLine = Nothing, dimLine = Nothing, dims, chain},
            chain = Lookup[nodeRec, "KernelOpChain", None];
            If[chain === None && uopName === "Kernel" && IntegerQ[val] && val > 0,
                chain = iDotKernelOpChain[val]];
            If[StringQ[chain] && chain =!= "", opLine = chain];
            If[uopName =!= "Fuse",
                dims = Lookup[nodeRec, "Shape", None];
                If[dims === None, dims = iDotInferShape[val]];
                If[ListQ[dims] && Length[dims] > 0,
                    dimLine = "[" <> StringRiffle[ToString /@ dims, "\[Cross]"] <> "]"]];
            DeleteCases[{ToUpperCase[uopName], opLine, dimLine, "@" <> ToString[hloc]},
                        Nothing]],
    "Ten",
        Module[{tens = TTensor[val], dims, vv, dev, parts},
            dims = Quiet[TDimensions[tens]];
            dev  = Quiet[tens["Device"]];
            vv = Quiet@Check[
                Module[{v = Normal[TGet[tens]]},
                    If[Length[Flatten[v]] <= 6,
                        "vals=[" <> StringRiffle[
                            ToString /@ Round[Flatten[v], 0.01], ","] <> "]",
                        Nothing]],
                Nothing];
            parts = {"t" <> ToString[val]};
            If[ListQ[dims],
                AppendTo[parts, "[" <> StringRiffle[ToString /@ dims, "\[Cross]"] <> "]"]];
            If[StringQ[dev], AppendTo[parts, "f32 " <> dev]];
            If[StringQ[vv], AppendTo[parts, vv]];
            parts],
    "Num", {"NUM", iNumText[ext, val]},
    "Era", {"\[FilledSmallCircle]"},
    "App", {"APP", "#" <> ToString[ext] <> "@" <> ToString[hloc]},
    "Lam", {"LAM", "#" <> ToString[ext] <> "@" <> ToString[hloc]},
    "Bri", {"BRI", "#" <> ToString[ext] <> "@" <> ToString[hloc]},
    "Ref", {"REF", "#" <> ToString[ext] <> "@" <> ToString[hloc]},
    "Mat", {"MAT", "#" <> ToString[ext] <> "@" <> ToString[hloc]},
    "Sup", {ToUpperCase[tag] <> " #" <> ToString[ext] <> "@" <> ToString[hloc]},
    "Usp", {ToUpperCase[tag] <> " #" <> ToString[ext] <> "@" <> ToString[hloc]},
    _,  DeleteCases[{ToUpperCase[tag],
         If[hloc > 0, "@" <> ToString[hloc], Nothing]}, Nothing]
];

(* Fill color for a node; dump.c palette with dark-mode variants. *)
iDotFill[tag_String, uop_String] := Which[
    tag === "Top",
        Module[{},
            Which[
                uop === "Kernel",           RGBColor["#ccffcc"],
                uop === "Assign",           RGBColor["#ffd700"],
                uop === "Fuse",             RGBColor["#f0f0f0"],
                MemberQ[$viewOps, uop],     RGBColor["#fff3cd"],
                MemberQ[$reduceOps, uop],   RGBColor["#d9edf7"],
                True,                       RGBColor["#cce5ff"]
            ]],
    True, RGBColor[Lookup[$heapTagFill, tag, "#f3f3f3"]]
];

(* Port-label for the `argi`-th slot of a child (parent of an edge). *)
iDotPortLabel[childTag_String, childUOp_String, argi_] := If[childTag === "Top",
    uopPortName[childUOp, argi],
    heapPortName[childTag, argi]
];

TINetGraph[t_TTensor, opts___?OptionQ] := TINetGraph[ToTTerm[t], opts];
TINetGraph[t_TTerm,   opts___?OptionQ] := (loadLibrary[];
    TINetGraph[heapWalk[rootTermOf[t]], opts, "TermId" -> termId[t]]);
TINetGraph[snapshot_?AssociationQ /; KeyExistsQ[snapshot, "Walk"], opts___?OptionQ] :=
    TINetGraph[snapshot["Walk"], opts,
        "TermId" -> If[MissingQ[snapshot["Term"]], None, termId[snapshot["Term"]]]];
TINetGraph[walkIn_?AssociationQ /; KeyExistsQ[walkIn, "Nodes"], opts___?OptionQ] := Module[
    {walk = walkIn, nodes, edges, keys, textColor, bg, lineFor, fillFor, vsf, gEdges, eLabels,
     nextInfo, activeSlot = -1, activeKey = None, hlColor, edgePairs, baseEdgeStyles, eStyleRules,
     passedOpts = Flatten[{opts}],
     termIdForHl = None},
    termIdForHl = "TermId" /. passedOpts /. "TermId" -> None;
    nodes  = walk["Nodes"];
    edges  = walk["Edges"];

    (* dump.c "out" free-slot above the root term. *)
    Module[{rootKey = walk["Root"], freeKey = "free_out"},
        If[StringQ[rootKey] && KeyExistsQ[nodes, rootKey],
            nodes[freeKey] = <|"Tag" -> "Free", "TagCode" -> -1,
                               "Ext" -> 0, "Val" -> 0, "Loc" -> 0,
                               "Key" -> freeKey, "DisplayLoc" -> 0|>;
            AppendTo[edges, <|"From" -> rootKey, "To" -> freeKey,
                              "Port" -> "out", "FromSlot" -> 0|>]]];

    keys = Keys[nodes];

    (* Find next active pair. nextInfo = {found, sourceSlot, tag, ext}.
       sourceSlot==0 means the root term itself is the active source; the
       highlighted edge is then the one feeding the root's first arg. *)
    nextInfo = If[termIdForHl === None, {0, -1, -1, -1},
        Quiet@Check[thvmNextInteractionFn[termIdForHl], {0, -1, -1, -1}]];
    If[ListQ[nextInfo] && Length[nextInfo] >= 4 && nextInfo[[1]] == 1,
        activeSlot = nextInfo[[2]];
        Module[{slot = nextInfo[[2]],
                activeTag = Lookup[$tagName, nextInfo[[3]], "?"],
                activeExt = nextInfo[[4]],
                activeUOp = Lookup[$uopName, nextInfo[[4]], "?"]},
            Do[With[{n = nodes[k]},
                If[(slot > 0 && (n["Loc"] == slot || Lookup[n, "DisplayLoc", -2] == slot)) ||
                   (slot <= 0 && n["Tag"] === activeTag &&
                    If[activeTag === "Top",
                        Lookup[n, "UOp", Lookup[$uopName, n["Ext"], "?"]] === activeUOp,
                        n["Ext"] == activeExt]),
                    activeKey = k; Break[]]],
                {k, keys}]];
        If[activeSlot <= 0 && activeKey =!= None,
            activeSlot = nodes[activeKey]["Val"]]];
    hlColor = RGBColor[0.85, 0.10, 0.10];

    (* dump.c style: white bg, black text/edges. *)
    textColor = GrayLevel[0.05];
    bg        = LightDarkSwitched[White, Black];

    lineFor[k_] := With[{n = nodes[k]},
        iDotLabelLines[n["Tag"], Lookup[n, "UOp", ""], n["Ext"], n["Val"],
            Lookup[n, "DisplayLoc", n["Loc"]], n]];
    fillFor[k_] := With[{n = nodes[k]}, iDotFill[n["Tag"], Lookup[n, "UOp", ""]]];

    (* Per-tag geometric shape, mirroring dump.c's shape= attribute. *)
    shapeFor[tag_String] := Replace[Lookup[$heapTagShape, tag, "Box"], "Box3d" -> "Box"];

    shapePrim[shape_, pos_, {w_, h_}] := Switch[shape,
        "Triangle",    Polygon[Map[pos + # &, {{0, h}, {-w, -h}, {w, -h}}]],
        "InvTriangle", Polygon[Map[pos + # &, {{0, -h}, {-w, h}, {w, h}}]],
        "Hexagon",     Polygon[Map[pos + # &,
                            Table[{w Cos[Pi/6 + k Pi/3], h Sin[Pi/6 + k Pi/3]}, {k, 0, 5}]]],
        "Oval",        Disk[pos, {w, h}],
        _,             Rectangle[pos - {w, h}, pos + {w, h}]
    ];

    nodeHalfSize[tag_String, lines_List] := Module[
        {n = Max[1, Length[lines]],
         maxLen = Max[Join[{1}, StringLength /@ (ToString /@ lines)]]},
        Switch[tag,
            "Lam" | "Bri",
                {1.35 + 0.03 maxLen, 0.50 + 0.17 n},
            "App",
                {1.30 + 0.03 maxLen, 0.50 + 0.17 n},
            "Ref" | "Var",
                {0.95 + 0.025 maxLen, 0.42 + 0.15 n},
            "Sup" | "Usp" | "Ctr",
                {1.10 + 0.03 maxLen, 0.44 + 0.15 n},
            _,
                {1.05 + 0.028 maxLen, 0.40 + 0.15 n}
        ]
    ];

    nodeGraphic[tag_String, lines_List, fill_] := Module[
        {shape = shapeFor[tag], half = nodeHalfSize[tag, lines], pad = 0.16},
        Graphics[
            {
                EdgeForm[Directive[GrayLevel[0.15], AbsoluteThickness[1.0]]],
                FaceForm[fill],
                shapePrim[shape, {0, 0}, half],
                Inset[
                    Column[
                        Style[#, 10, FontFamily -> "Helvetica", FontColor -> textColor] & /@ lines,
                        Alignment -> Center, Spacings -> 0.03
                    ],
                    {0, 0}
                ]
            },
            PlotRange -> {{-half[[1]] - pad, half[[1]] + pad},
                          {-half[[2]] - pad, half[[2]] + pad}},
            ImagePadding -> 0,
            Background -> None
        ]
    ];

    vsf[k_] := If[nodes[k]["Tag"] === "Free",
        Function[{pos, name, sz},
            Inset[
                Graphics[{GrayLevel[0.55], AbsoluteThickness[1.0], Circle[{0, 0}, 1]},
                         ImageSize -> 14, PlotRange -> {{-1.2, 1.2}, {-1.2, 1.2}}, ImagePadding -> 0, PlotRangePadding -> 0],
                pos, Center
            ]
        ],
        With[{tag = nodes[k]["Tag"], ls = lineFor[k], fill = fillFor[k]},
            Function[{pos, name, sz},
                Inset[nodeGraphic[tag, ls, fill], pos, Center]
            ]
        ]
    ];

    edgePairs = Map[
        Function[edge, {DirectedEdge[edge["From"], edge["To"],
                                     edge["Port"] -> Lookup[edge, "FromSlot", 0]],
                        edge}],
        edges];
    gEdges = edgePairs[[All, 1]];
    eLabels = Map[
        Function[pair, Module[{e = pair[[1]], edge = pair[[2]], from, port, slot, srcTag, labelColor},
            from = e[[1]];
            port = First[Last[e]];
            slot = Last[Last[e]];
            srcTag = Lookup[Lookup[nodes, from, <||>], "Tag", None];
            labelColor = Switch[Lookup[edge, "Style", ""],
                "KernelSemantic", RGBColor["#006600"],
                "RefDef", GrayLevel[0.45],
                _, StandardGray
            ];
            e -> Placed[
                Column[DeleteCases[{
                    Style[port, 10, FontFamily -> "Helvetica", FontColor -> labelColor],
                    (* Only show @slot tail label for tensor sources (matches dump.c). *)
                    If[slot > 0 && srcTag === "Ten",
                        Style["@" <> ToString[slot], 7, FontFamily -> "Helvetica",
                              FontSlant -> Italic, FontColor -> GrayLevel[0.45]],
                        Nothing]
                }, Nothing], Alignment -> Center, Spacings -> 0.02],
                {0.52, {0, 0}}]]],
        edgePairs];
    baseEdgeStyles = Map[
        Function[pair,
            pair[[1]] -> Switch[Lookup[pair[[2]], "Style", ""],
                "KernelSemantic",
                    Directive[RGBColor["#009900"], AbsoluteThickness[1.0],
                              Dashing[{0.03, 0.03}], Arrowheads[0.03]],
                "RefDef",
                    Directive[GrayLevel[0.6], AbsoluteThickness[1.0],
                              Dashing[{0.03, 0.03}], Arrowheads[0.03]],
                _,
                    Directive[LightDarkSwitched[GrayLevel[0.05], GrayLevel[0.95]], AbsoluteThickness[1.0], Arrowheads[0.03]]
            ]],
        edgePairs];
    (* Highlight only the single edge at the active source slot — matches dump.c. *)
    eStyleRules = If[activeSlot < 0, {},
        Cases[gEdges,
            e:DirectedEdge[_, _, _ -> activeSlot] :>
                (e -> Directive[hlColor, AbsoluteThickness[2.4], Arrowheads[0.035]])]];

    Graph[keys, gEdges,
        VertexShapeFunction -> (# -> vsf[#] & /@ keys),
        VertexSize   -> {0.9, 0.45},
        GraphLayout  -> {"LayeredDigraphEmbedding", "Orientation" -> Bottom},
        EdgeLabels   -> eLabels,
        EdgeStyle    -> Join[baseEdgeStyles, eStyleRules],
        PerformanceGoal -> "Quality",
        Background   -> bg,
        Sequence @@ FilterRules[passedOpts, Except["TermId"]]
    ]
];

(* ── Profile data ───────────────────────────────────────────────────────── *)

TProfileEnable[] := (loadLibrary[]; thvmProfileEnableFn[]);

TProfileData[] := Module[{raw, n = 29},
    loadLibrary[];
    raw = Normal[thvmProfileDataFn[]];
    If[Length[raw] < 3 n + 18, Return[$Failed]];
    <|
        "UOpTime" -> AssociationThread[
            Take[Values[$uopName], n], Take[raw, n] / 1.*^6],
        "UOpCount" -> AssociationThread[
            Take[Values[$uopName], n], Round /@ raw[[n + 1 ;; 2 n]]],
        "PhaseMs" -> <|
            "Forward" -> raw[[3 n + 1]] / 1.*^6,
            "Backward" -> raw[[3 n + 2]] / 1.*^6,
            "Adam" -> raw[[3 n + 3]] / 1.*^6,
            "Reset" -> raw[[3 n + 4]] / 1.*^6,
            "Other" -> raw[[3 n + 5]] / 1.*^6|>,
        "Memory" -> <|
            "AllocMB" -> raw[[3 n + 6]] / 1.*^6,
            "PeakMB" -> raw[[3 n + 8]] / 1.*^6,
            "CurrentMB" -> raw[[3 n + 9]] / 1.*^6|>,
        "Tensors" -> <|
            "Peak" -> Round[raw[[3 n + 10]]],
            "Created" -> Round[raw[[3 n + 11]]],
            "Freed" -> Round[raw[[3 n + 12]]]|>,
        "Dispatches" -> Total[Round /@ raw[[n + 1 ;; 2 n]]]
    |>
];

TProfileSummary[] := Module[{pd, phaseRows, uopRows, active},
    pd = TProfileData[];
    If[pd === $Failed, Return["Profiling not enabled. Call TProfileEnable[] first."]];

    phaseRows = Select[List @@@ Normal[pd["PhaseMs"]], #[[2]] > 0.001 &];
    active = Select[
        MapThread[{#1, #2, #3} &,
            {Keys[pd["UOpTime"]], Values[pd["UOpTime"]], Values[pd["UOpCount"]]}],
        #[[3]] > 0 &
    ];
    uopRows = SortBy[active, -#[[2]] &];

    Column[{
        Style["Phase Timing", Bold, 14],
        Grid[Prepend[phaseRows, {"Phase", "ms"}],
            Frame -> All, Background -> {None, {GrayLevel[0.9], None}},
            Alignment -> {{Left, Right}}, Spacings -> {2, 0.5}],
        "",
        Style["Top UOps by Time", Bold, 14],
        Grid[Prepend[Take[uopRows, UpTo[10]], {"UOp", "ms", "Count"}],
            Frame -> All, Background -> {None, {GrayLevel[0.9], None}},
            Alignment -> {{Left, Right, Right}}, Spacings -> {2, 0.5}],
        "",
        Style["Resources", Bold, 14],
        Grid[{
            {"Peak Memory", ToString[pd["Memory", "PeakMB"]] <> " MB"},
            {"Tensors Created", pd["Tensors", "Created"]},
            {"Peak Tensors", pd["Tensors", "Peak"]},
            {"Total Dispatches", pd["Dispatches"]}
        }, Frame -> All, Alignment -> {{Left, Right}}, Spacings -> {2, 0.5}]
    }, Spacings -> 1]
];

TProfileTimeline[snapshots_List] := Module[{phases, labels, dispatches, mem},
    phases = Values[#["PhaseMs"]] & /@ snapshots;
    labels = Keys[First[snapshots]["PhaseMs"]];
    dispatches = #["Dispatches"] & /@ snapshots;
    mem = #["Memory", "PeakMB"] & /@ snapshots;
    GraphicsGrid[{
        {ListLinePlot[Transpose[phases], PlotLegends -> labels,
            PlotLabel -> "Phase Timing per Step (ms)",
            AxesLabel -> {"Step", "ms"}, ImageSize -> 350],
         ListLinePlot[dispatches, PlotLabel -> "GPU Dispatches per Step",
            AxesLabel -> {"Step", "Dispatches"}, ImageSize -> 350]},
        {ListLinePlot[mem, PlotLabel -> "Peak Memory (MB)",
            AxesLabel -> {"Step", "MB"}, ImageSize -> 350, PlotRange -> {0, All}],
         SpanFromLeft}
    }, ImageSize -> 750]
];
