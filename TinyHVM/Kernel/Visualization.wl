(* Visualization.wl — Graph visualization and profiling for TinyHVM *)
(* Get'd from TinyHVM.wl inside Begin["`Private`"]. All public symbols declared there. *)

(* ── Computation graph ──────────────────────────────────────────────────── *)

$TGraphTrace = False;
$TGraph = <||>;

recordNode[id_, op_, inputs_List] := If[$TGraphTrace,
    $TGraph[id] = <|"Op" -> op, "Inputs" -> inputs|>
];

TGraphReset[] := ($TGraph = <||>);

opColor[op_String] := Switch[op,
    "Tensor", RGBColor[0.2, 0.6, 0.9],
    "Add" | "Mul" | "Sub" | "Div" | "Max", RGBColor[0.93, 0.49, 0.19],
    "Neg" | "Exp" | "Log" | "Relu" | "Sqrt", RGBColor[0.44, 0.74, 0.27],
    "Sum" | "RMax", RGBColor[0.84, 0.24, 0.24],
    "MatMul", RGBColor[0.63, 0.28, 0.64],
    "Conv2D" | "MaxPool2D", RGBColor[0.20, 0.44, 0.69],
    "Reshape" | "Expand" | "Permute" | "Pad" | "Shrink", GrayLevel[0.6],
    "Grad" | "GradMulti", RGBColor[0.85, 0.33, 0.10],
    _, GrayLevel[0.7]
];

TComputationGraph[t_TTensor, opts___?OptionQ] :=
Module[{id = t[[1]], visited = <||>, queue, edges = {}, labels = <||>,
        colors = <||>, cur, info, inp},
    queue = {id};
    While[queue =!= {},
        cur = First[queue]; queue = Rest[queue];
        If[!KeyExistsQ[visited, cur],
            visited[cur] = True;
            If[KeyExistsQ[$TGraph, cur],
                info = $TGraph[cur];
                labels[cur] = info["Op"];
                colors[cur] = opColor[info["Op"]];
                Do[AppendTo[edges, inp -> cur];
                   If[!KeyExistsQ[visited, inp], AppendTo[queue, inp]],
                   {inp, info["Inputs"]}],
                labels[cur] = "T" <> ToString[cur];
                colors[cur] = opColor["Tensor"]
            ]
        ]
    ];
    Graph[Keys[visited], edges,
        VertexLabels -> Normal[labels],
        VertexStyle -> Normal[colors],
        VertexSize -> 0.6,
        VertexLabelStyle -> Directive[10, Bold],
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
