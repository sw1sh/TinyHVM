(* Visualization.wl — Graph visualization and profiling for TinyHVM *)
(* Get'd from TinyHVM.wl inside Begin["`Private`"]. All public symbols declared there. *)

(* ── Inet graph from C heap ───────────────────────────────────────────── *)

(* $tagName is defined in TinyHVM.wl — do not redefine here *)

(* Tag colors for vertices *)
$tagColor = <|
    0  -> RGBColor[0.63, 0.28, 0.64],   (* APP — purple *)
    1  -> RGBColor[0.63, 0.28, 0.64],   (* LAM — purple *)
    2  -> LightDarkSwitched[GrayLevel[0.7], GrayLevel[0.5]],   (* VAR *)
    3  -> RGBColor[0.85, 0.33, 0.10],   (* SUP — orange *)
    4  -> RGBColor[0.85, 0.33, 0.10],   (* DP0 — orange *)
    5  -> RGBColor[0.85, 0.33, 0.10],   (* DP1 — orange *)
    6  -> LightDarkSwitched[GrayLevel[0.5], GrayLevel[0.6]],   (* ERA *)
    7  -> LightDarkSwitched[GrayLevel[0.6], GrayLevel[0.5]],   (* NUM *)
    8  -> RGBColor[0.44, 0.74, 0.27],   (* REF — green *)
    9  -> RGBColor[0.93, 0.49, 0.19],   (* OP2 — orange *)
    10 -> RGBColor[0.2, 0.6, 0.9],      (* TEN — blue *)
    11 -> RGBColor[0.93, 0.49, 0.19],   (* TOP — orange *)
    12 -> LightDarkSwitched[GrayLevel[0.6], GrayLevel[0.5]]    (* CTR *)
|>;

(* Build label for a graph node *)
iNodeLabel[tag_, ext_, val_] := Switch[tag,
    10, (* TEN *)
        Module[{dims = Quiet[TDimensions[TTensor[val]]]},
            If[ListQ[dims],
                "T" <> ToString[val] <> "\n" <> ToString[dims],
                "T" <> ToString[val]
            ]
        ],
    11, (* TOP *)
        If[KeyExistsQ[$uopName, ext], $uopName[ext], "UOp" <> ToString[ext]],
    7,  (* NUM *) "NUM",
    6,  (* ERA *) "\[FilledSmallCircle]",
    8,  (* REF *) "REF",
    _, Lookup[$tagName, tag, "?"]
];

(* ── Shared graph data extraction ────────────────────────────────────── *)

iNetGraphData[t_] := Module[
    {raw, nNodes, nEdges, nodesRaw, edgesRaw,
     tags, exts, vals, hlocs, i, j, tag, ext, val,
     dupGroups, rep, nodeMap, keptNodes,
     edges, labels, colors, elabels, heapLocs,
     src, dst, mSrc, mDst, portLabel, edge, seen},
    loadLibrary[];
    raw = Round[Normal[thvmHeapGraphFn[termId[t]]]];
    nNodes = raw[[1]];
    nEdges = raw[[2]];
    nodesRaw = raw[[3 ;; 2 + nNodes * 4]];
    edgesRaw = raw[[3 + nNodes * 4 ;; 2 + nNodes * 4 + nEdges * 2]];

    (* Extract per-node data: 4 fields per node *)
    tags  = Table[nodesRaw[[4 i - 3]], {i, nNodes}];
    exts  = Table[nodesRaw[[4 i - 2]], {i, nNodes}];
    vals  = Table[nodesRaw[[4 i - 1]], {i, nNodes}];
    hlocs = Table[nodesRaw[[4 i]],     {i, nNodes}];

    (* Group DP0/DP1 nodes by val (dup_loc) — merge into single DUP vertex *)
    dupGroups = <||>;
    Do[If[tags[[i]] == 4 || tags[[i]] == 5,
        val = vals[[i]];
        If[KeyExistsQ[dupGroups, val],
            AppendTo[dupGroups[val], i - 1],
            dupGroups[val] = {i - 1}]],
        {i, nNodes}];

    (* Build node remapping: DP nodes -> first representative *)
    nodeMap = Association[Table[i -> i, {i, 0, nNodes - 1}]];
    Do[rep = First[group]; Do[nodeMap[n] = rep, {n, group}],
        {group, Values[dupGroups]}];
    keptNodes = DeleteDuplicates[Values[nodeMap]];

    (* Build labels, colors, and heap-location map for kept nodes *)
    labels = <||>; colors = <||>; heapLocs = <||>;
    Do[tag = tags[[idx + 1]]; ext = exts[[idx + 1]]; val = vals[[idx + 1]];
        heapLocs[idx] = hlocs[[idx + 1]];
        If[tag == 4 || tag == 5,
            labels[idx] = "Dup"; colors[idx] = RGBColor[0.85, 0.33, 0.10],
            labels[idx] = iNodeLabel[tag, ext, val];
            colors[idx] = Lookup[$tagColor, tag, GrayLevel[0.7]]],
        {idx, keptNodes}];

    (* Build edges with port labels, remapping and deduplicating *)
    edges = {}; elabels = <||>; seen = <||>;
    Do[src = edgesRaw[[2 j - 1]]; dst = edgesRaw[[2 j]];
        mSrc = nodeMap[src]; mDst = nodeMap[dst];
        If[mSrc =!= mDst,
            If[tags[[src + 1]] == 4 || tags[[src + 1]] == 5,
                (* Edge from DP node: labeled multi-edge from merged DUP *)
                portLabel = If[tags[[src + 1]] == 4, "dp0", "dp1"];
                edge = DirectedEdge[mSrc, mDst, portLabel];
                If[!KeyExistsQ[seen, edge],
                    seen[edge] = True; AppendTo[edges, edge];
                    elabels[edge] = Placed[portLabel, {0.5, {0, -1.5}}]],
                (* Regular edge: dedup by endpoints *)
                If[!KeyExistsQ[seen, {mSrc, mDst}],
                    seen[{mSrc, mDst}] = True;
                    AppendTo[edges, DirectedEdge[mSrc, mDst]]]]],
        {j, nEdges}];

    <|"KeptNodes" -> keptNodes, "Edges" -> edges,
      "Labels" -> labels, "Colors" -> colors, "EdgeLabels" -> elabels,
      "HeapLocs" -> heapLocs, "Tags" -> tags, "NodeMap" -> nodeMap|>
];

(* ── TINetGraph — standard graph (unchanged behavior) ────────────────── *)

TINetGraph[t_TTensor, opts___?OptionQ] := TINetGraph[ToTTerm[t], opts];
TINetGraph[t_TTerm, opts___?OptionQ] := Module[{gd},
    gd = iNetGraphData[t];
    Graph[gd["KeptNodes"], gd["Edges"],
        VertexLabels -> Normal[gd["Labels"]],
        VertexStyle -> Normal[gd["Colors"]],
        VertexSize -> 0.6,
        VertexLabelStyle -> Directive[9, Bold],
        GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Left},
        EdgeStyle -> LightDarkSwitched[GrayLevel[0.5], GrayLevel[0.6]],
        EdgeLabels -> Normal[gd["EdgeLabels"]],
        EdgeLabelStyle -> Directive[8, Italic, LightDarkSwitched[GrayLevel[0.3], GrayLevel[0.7]]],
        ImageSize -> Medium,
        opts
    ]
];

(* ── TInteractionGraph — graph with highlighted active pair ──────────── *)

TInteractionGraph[t_TTensor, opts___?OptionQ] := TInteractionGraph[ToTTerm[t], opts];
TInteractionGraph[t_TTerm, opts___?OptionQ] := Module[
    {gd, nextInfo, found, sourceSlot, hlNode = None,
     vstyle, estyleRules},
    loadLibrary[];
    gd = iNetGraphData[t];

    (* Find the next active pair *)
    nextInfo = thvmNextInteractionFn[termId[t]];
    found = nextInfo[[1]];
    sourceSlot = nextInfo[[2]];

    (* Find graph node whose heap location matches the source slot *)
    If[found == 1,
        Do[If[gd["HeapLocs"][idx] == sourceSlot, hlNode = idx; Break[]],
            {idx, gd["KeptNodes"]}]
    ];

    (* Build vertex styles: highlight the active node with red border *)
    vstyle = Normal[gd["Colors"]];
    If[hlNode =!= None,
        vstyle = Join[vstyle,
            {hlNode -> Directive[
                EdgeForm[{Thick, RGBColor[0.9, 0.2, 0.2]}],
                Lookup[gd["Colors"], hlNode, GrayLevel[0.7]]]}]
    ];

    (* Build edge styles: highlight edges from the active node *)
    estyleRules = {};
    If[hlNode =!= None,
        Do[If[MatchQ[e, DirectedEdge[hlNode, _, ___] | DirectedEdge[_, hlNode, ___]],
            AppendTo[estyleRules, e -> Directive[Thick, RGBColor[0.9, 0.2, 0.2]]]],
            {e, gd["Edges"]}]
    ];

    Graph[gd["KeptNodes"], gd["Edges"],
        VertexLabels -> Normal[gd["Labels"]],
        VertexStyle -> vstyle,
        VertexSize -> 0.6,
        VertexLabelStyle -> Directive[9, Bold],
        GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Left},
        EdgeStyle -> Join[{LightDarkSwitched[GrayLevel[0.5], GrayLevel[0.6]]}, estyleRules],
        EdgeLabels -> Normal[gd["EdgeLabels"]],
        EdgeLabelStyle -> Directive[8, Italic, LightDarkSwitched[GrayLevel[0.3], GrayLevel[0.7]]],
        ImageSize -> Medium,
        opts
    ]
];

(* ── TDotGraph — dump.c-style renderer (boxed nodes, port labels, heap locs) ── *)

(* Walk inner KERNEL→KERNEL chain to build "INNER+OUTER" op label. *)
iDotKernelOpChain[val_Integer] := Module[{seen = <||>, parts = {}, walk, lim = 0},
    walk[loc_] := Module[{n, op, child},
        If[loc <= 0 || lim > 8 || KeyExistsQ[seen, loc], Return[Null, Module]];
        seen[loc] = True; lim++;
        n = Quiet@Check[THeapRead[loc + 2], <||>];
        If[!AssociationQ[n] || n["Tag"] =!= "Num", Return[Null, Module]];
        op = ToUpperCase[Lookup[$uopName, n["Val"], "?"]];
        child = Quiet@Check[THeapRead[loc], <||>];
        If[AssociationQ[child] && child["Tag"] === "Top" &&
           Lookup[$uopName, child["Ext"], ""] === "Kernel",
            walk[child["Val"]]];
        AppendTo[parts, op]];
    walk[val];
    If[parts === {}, "", StringRiffle[parts, "+"]]];

(* Infer shape of a compound term at heap base `val` by walking to first TEN leaf. *)
iDotInferShape[val_Integer] := Module[{seen = <||>, go, dims},
    go[loc_] := If[loc <= 0 || KeyExistsQ[seen, loc], None,
        seen[loc] = True;
        Module[{n = Quiet@Check[THeapRead[loc], None], arity, i, d},
            If[!AssociationQ[n], Return[None, Module]];
            Which[
                n["Tag"] === "Ten",
                    Quiet[TDimensions[TTensor[n["Val"]]]],
                n["Tag"] === "Top",
                    arity = uopArity[Lookup[$uopName, n["Ext"], "?"]];
                    Do[d = go[n["Val"] + i];
                       If[ListQ[d], Return[d, Module]],
                       {i, 0, arity - 1}]; None,
                KeyExistsQ[$heapTagArity, n["Tag"]] &&
                    IntegerQ[$heapTagArity[n["Tag"]]],
                    arity = $heapTagArity[n["Tag"]];
                    Do[d = go[n["Val"] + i];
                       If[ListQ[d], Return[d, Module]],
                       {i, 0, arity - 1}]; None,
                True, None]]];
    dims = go[val];
    If[ListQ[dims], dims, None]];

(* Multi-line label for a graph node, matching dump.c's \n-separated fields. *)
iDotLabelLines[tag_, ext_, val_, hloc_] := Switch[tag,
    11,  (* TOP *)
        Module[{uop = Lookup[$uopName, ext, "UOP" <> ToString[ext]],
                opLine = Nothing, dimLine = Nothing, dims, chain},
            If[uop === "Kernel" && IntegerQ[val] && val > 0,
                chain = iDotKernelOpChain[val];
                If[StringQ[chain] && chain =!= "", opLine = chain]];
            If[uop =!= "Fuse",
                dims = iDotInferShape[val];
                If[ListQ[dims] && Length[dims] > 0,
                    dimLine = "[" <> StringRiffle[ToString /@ dims, "\[Cross]"] <> "]"]];
            DeleteCases[{ToUpperCase[uop], opLine, dimLine, "@" <> ToString[hloc]},
                        Nothing]],
    10,  (* TEN *)
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
    7,  (* NUM *) {"NUM", ToString[ext]},
    6,  (* ERA *) {"\[FilledSmallCircle]"},
    _,  {ToUpperCase[Lookup[$tagName, tag, "?"]],
         If[hloc > 0, "@" <> ToString[hloc], Nothing]}
];

(* Fill color for a node; dump.c palette with dark-mode variants. *)
iDotFill[tag_, ext_] := Which[
    tag == 10, RGBColor["#e0e0e0"],
    tag ==  6, White,
    tag == 11,
        Module[{uop = Lookup[$uopName, ext, ""]},
            Which[
                uop === "Kernel",           RGBColor["#ccffcc"],
                uop === "Fuse",             RGBColor["#f0f0f0"],
                MemberQ[$viewOps, uop],     RGBColor["#fff3cd"],
                MemberQ[$reduceOps, uop],   RGBColor["#d9edf7"],
                True,                       RGBColor["#cce5ff"]
            ]],
    True, RGBColor["#f3f3f3"]
];

(* Port-label for the `argi`-th slot of a child (parent of an edge). *)
iDotPortLabel[childTag_, childExt_, argi_] := If[childTag == 11,
    uopPortName[Lookup[$uopName, childExt, ""], argi],
    heapPortName[Lookup[$tagName, childTag, "?"], argi]
];

TDotGraph[t_TTensor, opts___?OptionQ] := TDotGraph[ToTTerm[t], opts];
TDotGraph[t_TTerm,   opts___?OptionQ] := (loadLibrary[];
    TDotGraph[heapWalk[rootTermOf[t]], opts, "TermId" -> termId[t]]);
TDotGraph[snapshot_?AssociationQ /; KeyExistsQ[snapshot, "Walk"], opts___?OptionQ] :=
    TDotGraph[snapshot["Walk"], opts,
        "TermId" -> If[MissingQ[snapshot["Term"]], None, termId[snapshot["Term"]]]];
TDotGraph[walkIn_?AssociationQ /; KeyExistsQ[walkIn, "Nodes"], opts___?OptionQ] := Module[
    {walk = walkIn, nodes, edges, keys, textColor, bg, lineFor, fillFor, vsf, gEdges, eLabels,
     nextInfo, activeSlot = -1, activeKey = None, hlColor, eStyleRules,
     vertexCoords = Automatic, passedOpts = Flatten[{opts}],
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
        Module[{slot = nextInfo[[2]], tagC = nextInfo[[3]], extC = nextInfo[[4]]},
            Do[With[{n = nodes[k]},
                If[(slot > 0 && (n["Loc"] == slot || Lookup[n, "DisplayLoc", -2] == slot)) ||
                   (slot <= 0 && n["TagCode"] == tagC && n["Ext"] == extC),
                    activeKey = k; Break[]]],
                {k, keys}]];
        If[activeSlot <= 0 && activeKey =!= None,
            activeSlot = nodes[activeKey]["Val"]]];
    hlColor = RGBColor[0.85, 0.10, 0.10];

    (* dump.c style: white bg, black text/edges. *)
    textColor = GrayLevel[0.05];
    bg        = White;

    lineFor[k_] := With[{n = nodes[k]},
        iDotLabelLines[n["TagCode"], n["Ext"], n["Val"],
            Lookup[n, "DisplayLoc", n["Loc"]]]];
    fillFor[k_] := With[{n = nodes[k]}, iDotFill[n["TagCode"], n["Ext"]]];

    (* Per-tag geometric shape, mirroring dump.c's shape= attribute. *)
    shapeFor[tag_String] := Switch[tag,
        "Lam",  "Triangle",
        "App",  "InvTriangle",
        "Sup" | "Usp" | "Ctr", "Hexagon",
        "Ref" | "Var", "Oval",
        "Alo",  "Pentagon",
        _,      "Box"
    ];

    shapePrim[shape_, pos_, sz_] := With[{w = sz[[1]], h = sz[[2]]},
        Switch[shape,
            "Triangle",    Polygon[{pos+{0,h}, pos+{-w,-h}, pos+{w,-h}}],
            "InvTriangle", Polygon[{pos+{0,-h}, pos+{-w,h}, pos+{w,h}}],
            "Hexagon",     Polygon[pos + # & /@ ({Cos[#], Sin[#]} & /@
                               (Range[0, 5] 2 Pi/6 + Pi/6)) Transpose[{{w, w, w, w, w, w}, {h, h, h, h, h, h}}]],
            "Oval",        Disk[pos, sz],
            "Pentagon",    Polygon[pos + # & /@ ({Cos[#], Sin[#]} & /@
                               (Range[0, 4] 2 Pi/5 + Pi/2)) Transpose[{{w, w, w, w, w}, {h, h, h, h, h}}]],
            _,             Rectangle[pos - sz, pos + sz]
        ]
    ];

    vsf[k_] := If[nodes[k]["Tag"] === "Free",
        Function[{pos, name, sz},
            Inset[Graphics[{GrayLevel[0.55], AbsoluteThickness[1.0],
                            Circle[{0, 0}, 1]},
                           ImageSize -> 14], pos, Center]],
      With[{ls = lineFor[k], fill = fillFor[k]},
        Function[{pos, name, sz},
            Inset[
                Framed[
                    Column[
                        Style[#, 11, FontFamily -> "Helvetica",
                            FontColor -> GrayLevel[0.05]] & /@ ls,
                        Alignment -> Center, Spacings -> 0.05],
                    Background -> fill,
                    FrameStyle -> Directive[GrayLevel[0.2], AbsoluteThickness[1.0]],
                    FrameMargins -> 5,
                    RoundingRadius -> 0],
                pos, Center]
        ]]];

    gEdges = DirectedEdge[#["From"], #["To"], #["Port"] -> #["FromSlot"]] & /@ edges;
    eLabels = Function[e, Module[{from = e[[1]], port = First[Last[e]],
                                   slot = Last[Last[e]], srcTag},
            srcTag = Lookup[nodes, from, <||>][["Tag"]];
            e -> Placed[
                Column[DeleteCases[{
                    Style[port, 10, Bold, GrayLevel[0.05]],
                    (* Only show @slot tail label for tensor sources (matches dump.c). *)
                    If[slot > 0 && srcTag === "Ten",
                        Style["@" <> ToString[slot], 7, Italic, GrayLevel[0.45]],
                        Nothing]
                }, Nothing], Alignment -> Center, Spacings -> 0.05],
                {0.7, {0, 0}}]]] /@ gEdges;
    (* Highlight only the single edge at the active source slot — matches dump.c. *)
    eStyleRules = If[activeSlot < 0, {},
        Cases[gEdges,
            e:DirectedEdge[_, _, _ -> activeSlot] :>
                (e -> Directive[hlColor, AbsoluteThickness[2.4], Arrowheads[0.035]])]];

    (* Manual coords: y = longest path from leaf (root at top), x = slot per layer. *)
    Module[{depth = <||>, layers, coords, sourcesOf, parentsOf, sources, queue, k0, d, maxD},
        sourcesOf[v_] := Cases[edges, e_ /; e["To"] === v :> e["From"]];
        parentsOf[v_] := Cases[edges, e_ /; e["From"] === v :> e["To"]];
        Do[depth[k] = If[sourcesOf[k] === {}, 0, -1], {k, keys}];
        sources = Select[keys, depth[#] == 0 &];
        queue = sources;
        While[queue =!= {},
            k0 = First[queue]; queue = Rest[queue];
            Do[
                d = depth[k0] + 1;
                If[d > depth[p], depth[p] = d; AppendTo[queue, p]],
                {p, parentsOf[k0]}]];
        maxD = Max[Values[depth]];
        layers = GroupBy[keys, depth];
        Module[{ySpacing = If[maxD <= 0, 1.0, Min[1.0, 4.0/maxD]]},
            coords = Association@@Flatten@KeyValueMap[
                Function[{lvl, ks},
                    MapIndexed[#1 -> {(#2[[1]] - (Length[ks]+1)/2.) * 1.4,
                                       lvl * ySpacing} &, ks]],
                layers]];
        vertexCoords = Normal[coords]];

    Module[{xs = #[[2, 1]] & /@ vertexCoords,
            ys = #[[2, 2]] & /@ vertexCoords,
            pad = 1.0},
        Graph[keys, gEdges,
            VertexShapeFunction -> (# -> vsf[#] & /@ keys),
            VertexSize   -> {0.55, 0.25},
            VertexCoordinates -> vertexCoords,
            EdgeLabels   -> eLabels,
            EdgeStyle    -> Join[eStyleRules,
                                 {Directive[Black, AbsoluteThickness[1.0], Arrowheads[0.025]]}],
            PerformanceGoal -> "Quality",
            PlotRange    -> {
                {Min[xs] - pad, Max[xs] + pad},
                {Min[ys] - 0.5, Max[ys] + 0.5}},
            ImageSize    -> 460,
            Background   -> White,
            Sequence @@ FilterRules[passedOpts, Except["TermId"]]
        ]
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
