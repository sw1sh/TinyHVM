(* Visualization.wl — Graph visualization and profiling for TinyHVM *)
(* Get'd from TinyHVM.wl inside Begin["`Private`"]. All public symbols declared there. *)

(* ── Inet graph from C heap ───────────────────────────────────────────── *)

(* Tag names for display *)
$tagName = <|0 -> "APP", 1 -> "LAM", 2 -> "VAR", 3 -> "SUP",
    4 -> "DP0", 5 -> "DP1", 6 -> "ERA", 7 -> "NUM", 8 -> "REF",
    9 -> "OP2", 10 -> "TEN", 11 -> "TOP", 12 -> "CTR"|>;

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
    {ds, nodesRaw, edgesRaw, nNodes, nEdges, verts, edges, labels, colors, i,
     tag, ext, val},
    loadLibrary[];
    ds = thvmHeapGraphFn[t[[1]]];
    nodesRaw = Normal[ds["Nodes"]];
    edgesRaw = Normal[ds["Edges"]];
    nNodes = Length[nodesRaw] / 3;
    nEdges = Length[edgesRaw] / 2;

    (* Build vertices with labels and colors *)
    verts = Range[nNodes] - 1; (* 0-indexed *)
    labels = <||>;
    colors = <||>;
    Do[
        tag = nodesRaw[[3 i - 2]];
        ext = nodesRaw[[3 i - 1]];
        val = nodesRaw[[3 i]];
        labels[i - 1] = iNodeLabel[tag, ext, val];
        colors[i - 1] = Lookup[$tagColor, tag, GrayLevel[0.7]],
        {i, nNodes}
    ];

    (* Build edges *)
    edges = Table[
        edgesRaw[[2 j - 1]] -> edgesRaw[[2 j]],
        {j, nEdges}
    ];

    Graph[verts, edges,
        VertexLabels -> Normal[labels],
        VertexStyle -> Normal[colors],
        VertexSize -> 0.6,
        VertexLabelStyle -> Directive[9, Bold],
        GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Left},
        EdgeStyle -> GrayLevel[0.5],
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
