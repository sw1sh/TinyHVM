(* Visualization.wl — Graph visualization and profiling for TinyHVM *)
(* Get'd from TinyHVM.wl inside Begin["`Private`"]. All public symbols declared there. *)

(* ── Inet graph from C heap ───────────────────────────────────────────── *)

(* $tagName is defined in TinyHVM.wl — do not redefine here *)

(* Tag colors for vertices *)
$tagColor = <|
    0  -> RGBColor[0.63, 0.28, 0.64],   (* APP — purple *)
    1  -> RGBColor[0.63, 0.28, 0.64],   (* LAM — purple *)
    2  -> GrayLevel[0.7],               (* VAR — gray *)
    3  -> RGBColor[0.85, 0.33, 0.10],   (* SUP — orange *)
    4  -> RGBColor[0.85, 0.33, 0.10],   (* DP0 — orange *)
    5  -> RGBColor[0.85, 0.33, 0.10],   (* DP1 — orange *)
    6  -> GrayLevel[0.5],               (* ERA — dark gray *)
    7  -> GrayLevel[0.6],               (* NUM — gray *)
    8  -> RGBColor[0.44, 0.74, 0.27],   (* REF — green *)
    9  -> RGBColor[0.93, 0.49, 0.19],   (* OP2 — orange *)
    10 -> RGBColor[0.2, 0.6, 0.9],      (* TEN — blue *)
    11 -> RGBColor[0.93, 0.49, 0.19],   (* TOP — orange *)
    12 -> GrayLevel[0.6]                (* CTR — gray *)
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

TINetGraph[t_TTensor, opts___?OptionQ] := TINetGraph[ToTTerm[t], opts];
TINetGraph[t_TTerm, opts___?OptionQ] := Module[
    {raw, nNodes, nEdges, nodesRaw, edgesRaw,
     tags, exts, vals, i, j, tag, ext, val,
     dupGroups, rep, nodeMap, keptNodes,
     verts, edges, labels, colors, elabels,
     src, dst, mSrc, mDst, portLabel, edge, seen},
    loadLibrary[];
    raw = Round[Normal[thvmHeapGraphFn[t[[1]]]]];
    nNodes = raw[[1]];
    nEdges = raw[[2]];
    nodesRaw = raw[[3 ;; 2 + nNodes * 3]];
    edgesRaw = raw[[3 + nNodes * 3 ;; 2 + nNodes * 3 + nEdges * 2]];

    (* Extract per-node data *)
    tags = Table[nodesRaw[[3 i - 2]], {i, nNodes}];
    exts = Table[nodesRaw[[3 i - 1]], {i, nNodes}];
    vals = Table[nodesRaw[[3 i]], {i, nNodes}];

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

    (* Build labels and colors for kept nodes *)
    labels = <||>; colors = <||>;
    Do[tag = tags[[idx + 1]]; ext = exts[[idx + 1]]; val = vals[[idx + 1]];
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

    Graph[keptNodes, edges,
        VertexLabels -> Normal[labels],
        VertexStyle -> Normal[colors],
        VertexSize -> 0.6,
        VertexLabelStyle -> Directive[9, Bold],
        GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Left},
        EdgeStyle -> GrayLevel[0.5],
        EdgeLabels -> Normal[elabels],
        EdgeLabelStyle -> Directive[8, Italic, GrayLevel[0.3]],
        ImageSize -> Medium,
        opts
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
