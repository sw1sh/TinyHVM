(* Visualization.wl — Graph visualization and profiling for TinyHVM *)
(* Get'd from TinyHVM.wl inside Begin["`Private`"]. All public symbols declared there. *)

ClearAll[
    $visualizationSourceDirectory,
    ensureDOTImporterLoaded,
    numText,
    dotKernelOpChain,
    dotInferShape,
    dotLabelLines,
    dotFill,
    activeHighlightSlot,
    shapeForTag,
    shapePrim,
    nodeHalfSize,
    nodeGraphic,
    colorHexString,
    dotEscapeString,
    dotQuote,
    dotNodeId,
    dotShapeName,
    dotAttrsString,
    edgeLabelLines,
    edgeLabelColor,
    displayWalkData,
    dumpNumText,
    rootTensorDOTString,
    rootNumDOTString,
    exactRootDOTString,
    dotNodeAttrs,
    dotEdgeAttrs,
    walkDOTString,
    withDOTImport,
    THeapStripColor
];

$visualizationSourceDirectory = DirectoryName[$InputFileName];

ensureDOTImporterLoaded[] := Block[{},
    If[
        ! NameQ["TinyHVM`ImportCallGraphDOT`ImportDOTString"],
        Quiet @ Check[
            Get @ FileNameJoin[{$visualizationSourceDirectory, "ImportCallGraphDOT.wl"}],
            $Failed
        ]
    ]
]

ensureDOTImporterLoaded[];

(* NUM carries ext=0 for u32, ext=1 for f32. Render the raw value accordingly. *)
numText[ext_Integer, val_Integer] := If[
    ext == 1,
    Quiet @ Check[
        ToString[
            First @ ImportByteArray[
                ByteArray[IntegerDigits[BitAnd[val, 16^^FFFFFFFF], 256, 4]],
                "Real32"
            ]
        ],
        ToString[val]
    ],
    ToString[val]
]

dumpNumText[val_Integer] := Block[{x, ax},
    x = Quiet @ Check[
        First @ ImportByteArray[
            ByteArray[Reverse[IntegerDigits[BitAnd[val, 16^^FFFFFFFF], 256, 4]]],
            "Real32"
        ],
        val
    ];
    If[! NumericQ[x], Return[ToString[val]]];
    ax = Abs[N[x]];
    If[
        x == 0,
        "0",
        If[
            ax >= 10^4 || ax < 10^-4,
            ToString[
                ScientificForm[
                    x,
                    4,
                    NumberFormat -> (Row[{#1, "e", #3}] &)
                ],
                OutputForm
            ],
            ToString[
                NumberForm[x, 4, ExponentFunction -> (Null &)],
                OutputForm
            ]
        ]
    ]
]

rootTensorDOTString[node_Association] := Block[{tid, dims, dev, dimText},
    tid = node["Val"];
    dims = Quiet[TDimensions[TTensor[tid]]];
    dev = Quiet[TTensor[tid]["Device"]];
    dimText = If[ListQ[dims], StringRiffle[ToString /@ dims, "\[Cross]"], ""];
    StringRiffle[
        {
            "digraph G {",
            "  rankdir=BT;",
            "  node [fontname=\"Helvetica\", fontsize=10, style=filled, shape=box, margin=\"0.1,0.05\"];",
            "  edge [fontsize=8, fontname=\"Helvetica\"];",
            "",
            "  t" <> ToString[tid] <> " [label=\"t" <> ToString[tid] <> "\\n[" <> dimText <> "]\\nf32 " <> ToString[dev] <> "\",shape=box,fillcolor=\"#e0e0e0\",color=\"#1f78ff\",penwidth=2.2];",
            "  rootout_t" <> ToString[tid] <> " [label=\"\",shape=circle,width=0.14,height=0.14,fixedsize=true,fillcolor=\"#ffffff\",color=\"#888888\",fontsize=1];",
            "  t" <> ToString[tid] <> " -> rootout_t" <> ToString[tid] <> " [label=\"out\"];",
            "}"
        },
        "\n"
    ] <> "\n"
]

rootNumDOTString[node_Association] := StringRiffle[
    {
        "digraph G {",
        "  rankdir=BT;",
        "  node [fontname=\"Helvetica\", fontsize=10, style=filled, shape=box, margin=\"0.1,0.05\"];",
        "  edge [fontsize=8, fontname=\"Helvetica\"];",
        "",
        "  num_root [label=\"" <> dumpNumText[node["Val"]] <> "\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8,color=\"#1f78ff\",penwidth=2.2];",
        "}"
    },
    "\n"
] <> "\n"

exactRootDOTString[walk_Association, termId_: None] := Block[{data, root},
    data = displayWalkData[walk, termId];
    root = Lookup[data["Nodes"], data["Root"], Missing["NotFound"]];
    If[! AssociationQ[root], Return[None]];
    If[Length[data["Nodes"]] =!= 2 || Length[data["Edges"]] =!= 1, Return[None]];
    Switch[
        root["Tag"],
        "Ten", rootTensorDOTString[root],
        "Num", rootNumDOTString[root],
        _, None
    ]
]

(* Pull chain from cached KernelEntry's FusedOp[] via thvmKernelOpChainFn. *)
dotKernelOpChain[val_Integer] := Block[{
    s,
    topTagCode = Lookup[$tagCode, "Top", Missing["NotFound"]]
},
    If[! IntegerQ[topTagCode], Return[""]];
    s = Quiet @ Check[thvmKernelOpChainFn[topTagCode, $uopCode["Kernel"], val], ""];
    If[StringQ[s], ToUpperCase[s], ""]
]

(* Infer shape of a compound term at heap base `val` by walking to first TEN leaf. *)
dotInferShape[val_Integer] := Block[{
    seen = <||>,
    go,
    dims
},
    go[loc_] := If[
        loc <= 0 || KeyExistsQ[seen, loc],
        None,
        seen[loc] = True;
        Block[{
            n = Quiet @ Check[THeapRead[loc], None],
            arity,
            i,
            d
        },
            If[! AssociationQ[n], Return[None, Block]];
            Which[
                n["Tag"] === "Ten",
                    Quiet[TDimensions[TTensor[n["Val"]]]],
                n["Tag"] === "Top" || KeyExistsQ[$heapTagArity, n["Tag"]],
                    arity = termChildCount[n];
                    Do[
                        d = go[n["Val"] + i];
                        If[ListQ[d], Return[d, Block]],
                        {i, 0, arity - 1}
                    ];
                    None,
                True,
                    None
            ]
        ]
    ];
    dims = go[val];
    If[ListQ[dims], dims, None]
]

(* Multi-line label for a graph node, using the same newline-separated fields
   across the WL graph and DOT exporters.
   Optional `nodeRec` gives the captured-at-walk derived fields
   (KernelOpChain, Shape) so trace replays don't read mutated heap. *)
dotLabelLines[
    tag_String,
    uop_String,
    ext_,
    val_,
    hloc_,
    nodeRec_: <||>
] := Switch[
    tag,
    "Top",
        Block[{
            uopName = uop,
            opLine = Nothing,
            dimLine = Nothing,
            dims,
            chain
        },
            chain = Lookup[nodeRec, "KernelOpChain", None];
            If[chain === None && uopName === "Kernel" && IntegerQ[val] && val > 0,
                chain = dotKernelOpChain[val]
            ];
            If[StringQ[chain] && chain =!= "", opLine = chain];
            If[uopName =!= "Fuse",
                dims = Lookup[nodeRec, "Shape", None];
                If[dims === None, dims = dotInferShape[val]];
                If[ListQ[dims] && Length[dims] > 0,
                    dimLine = "[" <> StringRiffle[ToString /@ dims, "\[Cross]"] <> "]"
                ]
            ];
            DeleteCases[
                {ToUpperCase[uopName], opLine, dimLine, "@" <> ToString[hloc]},
                Nothing
            ]
        ],
    "Ten",
        Block[{
            tens = TTensor[val],
            dims,
            valsText,
            dev,
            parts
        },
            dims = Quiet[TDimensions[tens]];
            dev = Quiet[tens["Device"]];
            valsText = Quiet @ Check[
                Block[{v = Normal[TGet[tens]]},
                    If[
                        Length[Flatten[v]] <= 6,
                        "vals=[" <> StringRiffle[ToString /@ Round[Flatten[v], 0.01], ","] <> "]",
                        Nothing
                    ]
                ],
                Nothing
            ];
            parts = {"t" <> ToString[val]};
            If[ListQ[dims],
                AppendTo[parts, "[" <> StringRiffle[ToString /@ dims, "\[Cross]"] <> "]"]
            ];
            If[StringQ[dev], AppendTo[parts, "f32 " <> dev]];
            If[StringQ[valsText], AppendTo[parts, valsText]];
            parts
        ],
    "Num",
        {"NUM", numText[ext, val]},
    "Era",
        {"\[FilledSmallCircle]"},
    "App",
        {"APP", "#" <> ToString[ext] <> "@" <> ToString[hloc]},
    "Lam",
        {"LAM", "#" <> ToString[ext] <> "@" <> ToString[hloc]},
    "Bri",
        {"BRI", "#" <> ToString[ext] <> "@" <> ToString[hloc]},
    "Ref",
        {"REF", "#" <> ToString[ext] <> "@" <> ToString[hloc]},
    "Mat",
        {"MAT", "#" <> ToString[ext] <> "@" <> ToString[hloc]},
    "Sup",
        {ToUpperCase[tag] <> " #" <> ToString[ext] <> "@" <> ToString[hloc]},
    "Usp",
        {ToUpperCase[tag] <> " #" <> ToString[ext] <> "@" <> ToString[hloc]},
    _,
        DeleteCases[
            {
                ToUpperCase[tag],
                If[hloc > 0, "@" <> ToString[hloc], Nothing]
            },
            Nothing
        ]
]

(* Fill color for a node in the shared WL/DOT display path. *)
dotFill[tag_String, uop_String] := Which[
    tag === "Top",
        Which[
            uop === "Kernel", RGBColor["#ccffcc"],
            uop === "Assign", RGBColor["#ffd700"],
            uop === "Fuse", RGBColor["#f0f0f0"],
            MemberQ[$viewOps, uop], RGBColor["#fff3cd"],
            MemberQ[$reduceOps, uop], RGBColor["#d9edf7"],
            True, RGBColor["#cce5ff"]
        ],
    True,
        RGBColor[Lookup[$heapTagFill, tag, "#f3f3f3"]]
]

activeHighlightSlot[walk_Association, termId_] := Block[{
    nodes = walk["Nodes"],
    keys = Keys[walk["Nodes"]],
    nextInfo,
    activeSlot = -1,
    activeKey = None
},
    If[termId === None, Return[-1]];
    nextInfo = Quiet @ Check[thvmNextInteractionFn[termId], {0, -1, -1, -1}];
    If[ListQ[nextInfo] && Length[nextInfo] >= 4 && nextInfo[[1]] == 1,
        activeSlot = nextInfo[[2]];
        Block[{
            slot = nextInfo[[2]],
            activeTag = Lookup[$tagName, nextInfo[[3]], "?"],
            activeExt = nextInfo[[4]],
            activeUOp = Lookup[$uopName, nextInfo[[4]], "?"]
        },
            Do[
                With[{n = nodes[k]},
                    If[
                        (slot > 0 && (n["Loc"] == slot || Lookup[n, "DisplayLoc", -2] == slot)) ||
                        (
                            slot <= 0 &&
                            n["Tag"] === activeTag &&
                            If[
                                activeTag === "Top",
                                Lookup[n, "UOp", Lookup[$uopName, n["Ext"], "?"]] === activeUOp,
                                n["Ext"] == activeExt
                            ]
                        ),
                        activeKey = k;
                        Break[]
                    ]
                ],
                {k, keys}
            ]
        ];
        If[activeSlot <= 0 && activeKey =!= None,
            activeSlot = nodes[activeKey]["Val"]
        ]
    ];
    activeSlot
]

shapeForTag[tag_String] := Replace[Lookup[$heapTagShape, tag, "Box"], "Box3d" -> "Box"]

shapePrim[shape_, pos_, {w_, h_}] := Switch[
    shape,
    "Triangle",
        Polygon[Map[pos + # &, {{0, h}, {-w, -h}, {w, -h}}]],
    "InvTriangle",
        Polygon[Map[pos + # &, {{0, -h}, {-w, h}, {w, h}}]],
    "Hexagon",
        Polygon[
            Map[pos + # &, Table[{w Cos[Pi / 6 + k Pi / 3], h Sin[Pi / 6 + k Pi / 3]}, {k, 0, 5}]]
        ],
    "Oval",
        Disk[pos, {w, h}],
    _,
        Rectangle[pos - {w, h}, pos + {w, h}]
]

nodeHalfSize[tag_String, lines_List] := Block[{
    n = Max[1, Length[lines]],
    maxLen = Max[Join[{1}, StringLength /@ (ToString /@ lines)]]
},
    Switch[
        tag,
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
]

nodeGraphic[tag_String, lines_List, fill_, textColor_] := Block[{
    shape = shapeForTag[tag],
    half = nodeHalfSize[tag, lines],
    pad = 0.16,
    n, fs, lineH, ys
},
    (* First line (primary label) is slightly larger & bold, the rest are smaller.
       Text primitives are positioned directly so Rasterize renders without
       stray Inset bounding boxes. *)
    n = Length[lines];
    fs = If[n == 0, {}, Prepend[ConstantArray[9, n - 1], 11]];
    lineH = 2 half[[2]] / (n + 1);
    ys = Table[half[[2]] - i lineH, {i, 1, n}];
    Graphics[
        {
            EdgeForm[Directive[GrayLevel[0.15], AbsoluteThickness[1.0]]],
            FaceForm[fill],
            shapePrim[shape, {0, 0}, half],
            MapThread[
                Function[{line, y, size, bold},
                    Text[
                        Style[line,
                            FontFamily -> "Helvetica",
                            FontColor -> textColor,
                            FontSize -> size,
                            FontWeight -> If[bold, Bold, Plain]],
                        {0, y}]],
                {lines, ys, fs, Prepend[ConstantArray[False, Max[n - 1, 0]], True]}]
        },
        PlotRange -> {
            {-half[[1]] - pad, half[[1]] + pad},
            {-half[[2]] - pad, half[[2]] + pad}
        },
        ImagePadding -> 0,
        Background -> None
    ]
]

colorHexString[color_] := Block[{rgb},
    rgb = List @@ ColorConvert[color, "RGB"];
    "#" <> StringJoin[IntegerString[Clip[Round[255 rgb], {0, 255}], 16, 2]]
]

dotEscapeString[s_String] := StringReplace[
    s,
    {
        "\\" -> "\\\\",
        "\"" -> "\\\"",
        "\n" -> "\\n"
    }
]

dotQuote[s_String] := "\"" <> dotEscapeString[s] <> "\""

dotNodeId[key_] := dotQuote[ToString[key]]

dotShapeName[tag_String] := Switch[
    tag,
    "Free", "circle",
    "Era", "point",
    "Lam" | "Bri", "triangle",
    "App", "invtriangle",
    "Sup" | "Usp" | "Ctr", "hexagon",
    "Ref" | "Var", "oval",
    _, ToLowerCase[shapeForTag[tag]]
]

dotAttrsString[attrs_Association] := StringRiffle[
    KeyValueMap[#1 <> "=" <> #2 &, attrs],
    ", "
]

edgeLabelLines[edge_Association, nodes_Association] := Block[{
    slot = Lookup[edge, "FromSlot", 0],
    srcTag = Lookup[Lookup[nodes, edge["From"], <||>], "Tag", None]
},
    DeleteCases[
        {
            edge["Port"],
            If[slot > 0 && srcTag === "Ten", "@" <> ToString[slot], Nothing]
        },
        Nothing
    ]
]

edgeLabelColor[edge_Association] := Switch[
    Lookup[edge, "Style", ""],
    "KernelSemantic", "#006600",
    "RefDef", "#777777",
    _, "#222222"
]

displayWalkData[walk_Association, termId_: None] := Block[{
    nodes = Association[walk["Nodes"]],
    edges = walk["Edges"],
    rootKey = walk["Root"],
    freeKey = "free_out"
},
    If[StringQ[rootKey] && KeyExistsQ[nodes, rootKey],
        nodes[freeKey] = <|
            "Tag" -> "Free",
            "TagCode" -> -1,
            "Ext" -> 0,
            "Val" -> 0,
            "Loc" -> 0,
            "Key" -> freeKey,
            "DisplayLoc" -> 0
        |>;
        AppendTo[
            edges,
            <|"From" -> rootKey, "To" -> freeKey, "Port" -> "out", "FromSlot" -> 0|>
        ]
    ];
    <|
        "Nodes" -> nodes,
        "Edges" -> edges,
        "Root" -> rootKey,
        "Keys" -> Keys[nodes],
        "ActiveSlot" -> activeHighlightSlot[
            <|"Nodes" -> nodes, "Edges" -> edges, "Root" -> rootKey|>,
            termId
        ]
    |>
]

dotNodeAttrs[node_Association] := Block[{
    tag = node["Tag"],
    uop = Lookup[node, "UOp", ""],
    hloc = Lookup[node, "DisplayLoc", node["Loc"]],
    label
},
    If[tag === "Free",
        Return[
            <|
                "label" -> dotQuote[""],
                "shape" -> dotQuote["circle"],
                "width" -> "0.18",
                "height" -> "0.18",
                "fixedsize" -> "true",
                "style" -> dotQuote["solid"],
                "color" -> dotQuote["#808080"]
            |>
        ]
    ];

    If[tag === "Era",
        Return[
            <|
                "label" -> dotQuote[""],
                "shape" -> dotQuote["point"],
                "width" -> "0.12",
                "height" -> "0.12",
                "color" -> dotQuote["#666666"],
                "fillcolor" -> dotQuote["#ffffff"],
                "style" -> dotQuote["filled"]
            |>
        ]
    ];

    label = StringRiffle[
        dotLabelLines[tag, uop, node["Ext"], node["Val"], hloc, node],
        "\n"
    ];

    <|
        "label" -> dotQuote[label],
        "shape" -> dotQuote[dotShapeName[tag]],
        "style" -> dotQuote["filled"],
        "fillcolor" -> dotQuote[colorHexString[dotFill[tag, uop]]],
        "color" -> dotQuote["#222222"],
        "fontname" -> dotQuote["Helvetica"],
        "fontsize" -> "10",
        "penwidth" -> "1.2"
    |>
]

dotEdgeAttrs[edge_Association, nodes_Association, activeSlot_Integer] := Block[{
    slot = Lookup[edge, "FromSlot", 0],
    style = Lookup[edge, "Style", ""],
    labelLines = edgeLabelLines[edge, nodes],
    labelColor = edgeLabelColor[edge],
    attrs
},
    attrs = Switch[
        style,
        "KernelSemantic",
            <|
                "color" -> dotQuote["#009900"],
                "style" -> dotQuote["dashed"],
                "penwidth" -> "1.0"
            |>,
        "RefDef",
            <|
                "color" -> dotQuote["#999999"],
                "style" -> dotQuote["dashed"],
                "penwidth" -> "1.0"
            |>,
        _,
            <|
                "color" -> dotQuote["#111111"],
                "penwidth" -> "1.0"
            |>
    ];
    If[slot == activeSlot,
        attrs = Join[
            attrs,
            <|
                "color" -> dotQuote["#d91c1c"],
                "penwidth" -> "2.4"
            |>
        ]
    ];
    Join[
        attrs,
        <|
            "label" -> dotQuote[StringRiffle[labelLines, "\n"]],
            "fontname" -> dotQuote["Helvetica"],
            "fontsize" -> If[slot > 0 && srcTag === "Ten", "8", "9"],
            "fontcolor" -> dotQuote[labelColor]
        |>
    ]
]

walkDOTString[walk_Association, termId_: None] := Block[{
    exact = exactRootDOTString[walk, termId],
    data = displayWalkData[walk, termId],
    nodes,
    edges,
    nodeLines,
    edgeLines
},
    If[StringQ[exact], Return[exact]];
    nodes = data["Nodes"];
    edges = data["Edges"];

    nodeLines = KeyValueMap[
        "    " <> dotNodeId[#1] <> " [" <> dotAttrsString[dotNodeAttrs[#2]] <> "];" &,
        KeySort[nodes]
    ];
    edgeLines = Map[
        Function[edge,
            "    " <> dotNodeId[edge["From"]] <> " -> " <> dotNodeId[edge["To"]] <>
                " [" <> dotAttrsString[dotEdgeAttrs[edge, nodes, data["ActiveSlot"]]] <> "];"
        ],
        edges
    ];

    StringRiffle[
        Join[
            {
                "digraph TinyHVM {",
                "    graph [rankdir=BT, bgcolor=\"#ffffff\", pad=\"0.2\", nodesep=\"0.25\", ranksep=\"0.35\"];",
                "    node [fontname=\"Helvetica\", margin=\"0.08,0.04\"];",
                "    edge [fontname=\"Helvetica\", arrowsize=\"0.7\"];"
            },
            nodeLines,
            edgeLines,
            {"}"}
        ],
        "\n"
    ]
]

withDOTImport[dot_String, opts___?OptionQ] := Block[{
    passedOpts = Flatten[{opts}],
    method,
    importOpts,
    result = $Failed
},
    ensureDOTImporterLoaded[];
    method = Method /. passedOpts /. Method -> "GraphvizGraphics";
    importOpts = Append[
        DeleteCases[
            passedOpts,
            HoldPattern[Method -> _] | HoldPattern["TermId" -> _]
        ],
        Method -> method
    ];
    result = TinyHVM`ImportCallGraphDOT`ImportDOTString[dot, Sequence @@ importOpts];
    result
]

dumpCDOTString[t_TTensor] := dumpCDOTString[ToTTerm[t]]

dumpCDOTString[t_TTerm] := Block[{root},
    loadLibrary[];
    root = rootTermOf[t];
    thvmHeapDOTRawFn[
        Lookup[$tagCode, root["Tag"], 0],
        root["Ext"],
        root["Val"]
    ]
]

TINetGraphDOT[t_TTensor, opts___?OptionQ] := TINetGraphDOT[ToTTerm[t], opts]

TINetGraphDOT[t_TTerm, opts___?OptionQ] := Block[{},
    loadLibrary[];
    TINetGraphDOT[heapWalk[rootTermOf[t]], opts, "TermId" -> termId[t]]
]

TINetGraphDOT[
    snapshot_ ? AssociationQ /; KeyExistsQ[snapshot, "Walk"],
    opts___?OptionQ
] := TINetGraphDOT[
    snapshot["Walk"],
    opts,
    "TermId" -> If[MissingQ[snapshot["Term"]], None, termId[snapshot["Term"]]]
]

TINetGraphDOT[
    walk_ ? AssociationQ /; KeyExistsQ[walk, "Nodes"],
    opts___?OptionQ
] := Block[{
    passedOpts = Flatten[{opts}],
    termIdForHighlight
},
    termIdForHighlight = "TermId" /. passedOpts /. "TermId" -> None;
    walkDOTString[walk, termIdForHighlight]
]

Options[TINetGraphImport] = Options[TinyHVM`ImportCallGraphDOT`ImportDOTString];

TINetGraphImport[t_TTensor, opts___?OptionQ] := TINetGraphImport[ToTTerm[t], opts]

TINetGraphImport[t_TTerm, opts___?OptionQ] := Block[{},
    loadLibrary[];
    TINetGraphImport[heapWalk[rootTermOf[t]], opts, "TermId" -> termId[t]]
]

TINetGraphImport[
    snapshot_ ? AssociationQ /; KeyExistsQ[snapshot, "Walk"],
    opts___?OptionQ
] := TINetGraphImport[
    snapshot["Walk"],
    opts,
    "TermId" -> If[MissingQ[snapshot["Term"]], None, termId[snapshot["Term"]]]
]

TINetGraphImport[
    walk_ ? AssociationQ /; KeyExistsQ[walk, "Nodes"],
    opts___?OptionQ
] := withDOTImport[TINetGraphDOT[walk, opts], opts]

TINetGraph[t_TTensor, opts___?OptionQ] := TINetGraph[ToTTerm[t], opts]

TINetGraph[t_TTerm, opts___?OptionQ] := Block[{},
    loadLibrary[];
    TINetGraph[heapWalk[rootTermOf[t]], opts, "TermId" -> termId[t]]
]

TINetGraph[
    snapshot_ ? AssociationQ /; KeyExistsQ[snapshot, "Walk"],
    opts___?OptionQ
] := TINetGraph[
    snapshot["Walk"],
    opts,
    "TermId" -> If[MissingQ[snapshot["Term"]], None, termId[snapshot["Term"]]]
]

TINetGraph[
    walkIn_ ? AssociationQ /; KeyExistsQ[walkIn, "Nodes"],
    opts___?OptionQ
] := Block[{
    data,
    nodes,
    edges,
    keys,
    textColor,
    bg,
    lineFor,
    fillFor,
    vsf,
    gEdges,
    eLabels,
    hlColor,
    edgePairs,
    baseEdgeStyles,
    eStyleRules,
    passedOpts = Flatten[{opts}],
    termIdForHl = None
},
    termIdForHl = "TermId" /. passedOpts /. "TermId" -> None;
    data = displayWalkData[walkIn, termIdForHl];
    nodes = data["Nodes"];
    edges = data["Edges"];
    keys = data["Keys"];
    hlColor = RGBColor[0.85, 0.10, 0.10];

    textColor = GrayLevel[0.05];
    bg = LightDarkSwitched[White, Black];

    lineFor[k_] := With[
        {n = nodes[k]},
        dotLabelLines[n["Tag"], Lookup[n, "UOp", ""], n["Ext"], n["Val"], Lookup[n, "DisplayLoc", n["Loc"]], n]
    ];
    fillFor[k_] := With[{n = nodes[k]}, dotFill[n["Tag"], Lookup[n, "UOp", ""]]];

    (* Eagerly evaluate the node Graphics *before* capturing it in the VSF closure.
       Leaving `nodeGraphic[...]` unevaluated inside `Function[{pos,...}, Inset[...]]`
       causes the FE to emit stray pink bounding boxes during Rasterize, because
       the inner Graphics re-resolves every draw. Baking the result into `ng`
       keeps the closure a pure Inset of a fully-formed Graphics. *)
    vsf[k_] := If[
        nodes[k]["Tag"] === "Free",
        With[{ng = Graphics[
                {GrayLevel[0.55], AbsoluteThickness[1.0], Circle[{0, 0}, 1]},
                ImageSize -> 14,
                PlotRange -> {{-1.2, 1.2}, {-1.2, 1.2}},
                ImagePadding -> 0,
                PlotRangePadding -> 0]},
            Function[{pos, name, sz}, Inset[ng, pos, Center]]
        ],
        With[{ng = nodeGraphic[nodes[k]["Tag"], lineFor[k], fillFor[k], textColor]},
            Function[{pos, name, sz}, Inset[ng, pos, Center]]
        ]
    ];

    edgePairs = Map[
        Function[edge,
            {
                DirectedEdge[edge["From"], edge["To"], edge["Port"] -> Lookup[edge, "FromSlot", 0]],
                edge
            }
        ],
        edges
    ];
    gEdges = edgePairs[[All, 1]];
    eLabels = Map[
        Function[pair,
            Block[{
                e = pair[[1]],
                edge = pair[[2]],
                labelColor
            },
                labelColor = Switch[
                    edgeLabelColor[edge],
                    "#006600", RGBColor["#006600"],
                    "#777777", GrayLevel[0.45],
                    _, StandardGray
                ];
                e -> Placed[
                    Column[
                        DeleteCases[
                            MapIndexed[
                                If[
                                    #2[[1]] == 1,
                                    Style[#, 10, FontFamily -> "Helvetica", FontColor -> labelColor],
                                    Style[
                                        #,
                                        7,
                                        FontFamily -> "Helvetica",
                                        FontSlant -> Italic,
                                        FontColor -> GrayLevel[0.45]
                                    ]
                                ] &,
                                edgeLabelLines[edge, nodes]
                            ],
                            Nothing
                        ],
                        Alignment -> Center,
                        Spacings -> 0.02
                    ],
                    {0.52, {0, 0}}
                ]
            ]
        ],
        edgePairs
    ];
    baseEdgeStyles = Map[
        Function[pair,
            pair[[1]] -> Switch[
                Lookup[pair[[2]], "Style", ""],
                "KernelSemantic",
                    Directive[
                        RGBColor["#009900"],
                        AbsoluteThickness[1.0],
                        Dashing[{0.03, 0.03}],
                        Arrowheads[0.03]
                    ],
                "RefDef",
                    Directive[
                        GrayLevel[0.6],
                        AbsoluteThickness[1.0],
                        Dashing[{0.03, 0.03}],
                        Arrowheads[0.03]
                    ],
                _,
                    Directive[
                        LightDarkSwitched[GrayLevel[0.05], GrayLevel[0.95]],
                        AbsoluteThickness[1.0],
                        Arrowheads[0.03]
                    ]
            ]
        ],
        edgePairs
    ];
    eStyleRules = If[
        data["ActiveSlot"] < 0,
        {},
        Cases[
            gEdges,
            e : DirectedEdge[_, _, _ -> data["ActiveSlot"]] :>
                (e -> Directive[hlColor, AbsoluteThickness[2.4], Arrowheads[0.035]])
        ]
    ];

    Graph[
        keys,
        gEdges,
        VertexShapeFunction -> (# -> vsf[#] & /@ keys),
        VertexSize -> {0.9, 0.45},
        GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Bottom},
        EdgeLabels -> eLabels,
        EdgeStyle -> Join[baseEdgeStyles, eStyleRules],
        PerformanceGoal -> "Quality",
        Background -> bg,
        Sequence @@ FilterRules[passedOpts, Except["TermId"]]
    ]
]

(* ── Pedagogical helpers (TBitField / TShapeLegend / THeapStrip) ────────── *)

Options[TBitField] = {ImageSize -> 640};

TBitField[fields : {{_, _Integer, _} ..}, opts : OptionsPattern[]] := Block[{
    bits = fields[[All, 2]],
    labels = fields[[All, 1]],
    values = fields[[All, 3]],
    total, w, xs, colors, minW = 0.08
},
    total = Total[bits];
    (* Proportional widths with a minimum so narrow fields still hold their label. *)
    w = Max[#, minW] & /@ (bits / total);
    w = w / Total[w];
    xs = Accumulate[Prepend[Most[w], 0]];
    colors = {
        RGBColor["#fde2e2"], RGBColor["#d1e7dd"],
        RGBColor["#fff3cd"], RGBColor["#cfe2ff"],
        RGBColor["#e2d4f7"], RGBColor["#f8d7da"]
    }[[1 ;; Length[fields]]];
    Graphics[
        MapThread[
            Function[{name, nbits, val, x, width, col},
                {FaceForm[col],
                 EdgeForm[Directive[GrayLevel[0.2], AbsoluteThickness[1.1]]],
                 Rectangle[{x, 0}, {x + width, 1}],
                 Text[Style[ToString[name], Bold, 13, FontColor -> Black],
                      {x + width/2, 0.78}],
                 Text[Style[ToString[nbits] <> " bits", 9, FontColor -> GrayLevel[0.35]],
                      {x + width/2, 0.52}],
                 Text[Style[ToString[val], 11, FontColor -> RGBColor[0.15, 0.15, 0.55]],
                      {x + width/2, 0.22}]}],
            {labels, bits, values, xs, w, colors}],
        PlotRange -> {{-0.01, 1.01}, {-0.1, 1.1}},
        ImageSize -> OptionValue[ImageSize],
        AspectRatio -> 1/6,
        Background -> White]
]

(* Shapes/colors mirror the gallery you see in TINetGraph, hand-drawn so they
   render predictably at small sizes. *)
TShapeLegend[] := Block[{items, cell},
    items = {
        {"LAM", "binder",      "Triangle",    RGBColor[Lookup[$heapTagFill, "Lam", "#f2e8ff"]]},
        {"APP", "application", "InvTriangle", RGBColor[Lookup[$heapTagFill, "App", "#f3f3f3"]]},
        {"SUP", "super",       "Hexagon",     RGBColor[Lookup[$heapTagFill, "Sup", "#e4d6fc"]]},
        {"VAR", "reference",   "Oval",        RGBColor[Lookup[$heapTagFill, "Var", "#eeeeee"]]},
        {"NUM", "leaf (u32)",  "Rectangle",   RGBColor["#fde2e2"]},
        {"OP2", "primop",      "Rectangle",   RGBColor["#cfe2ff"]},
        {"ERA", "eraser",      "Dot",         RGBColor["#ffffff"]}
    };
    cell[{name_, role_, shape_, col_}] := Column[{
        Graphics[{
            FaceForm[col],
            EdgeForm[Directive[GrayLevel[0.15], AbsoluteThickness[1.2]]],
            Switch[shape,
                "Triangle",    Polygon[{{0, 0.9}, {-1, -0.7}, {1, -0.7}}],
                "InvTriangle", Polygon[{{0, -0.9}, {-1, 0.7}, {1, 0.7}}],
                "Hexagon",     Polygon[Table[{Cos[Pi/6 + k Pi/3], Sin[Pi/6 + k Pi/3]}, {k, 0, 5}]],
                "Oval",        Disk[{0, 0}, {1.1, 0.65}],
                "Rectangle",   Rectangle[{-1, -0.6}, {1, 0.6}],
                "Dot",         Disk[{0, 0}, 0.25]],
            Text[Style[name, Bold, 14, FontColor -> Black], {0, 0.05}]},
            ImageSize -> 90,
            PlotRange -> {{-1.2, 1.2}, {-1.1, 1.1}},
            Background -> White],
        Style[role, 10, FontColor -> GrayLevel[0.4]]},
        Alignment -> Center, Spacings -> 0.2];
    Grid[{cell /@ items},
        Spacings -> {1, 1}, Alignment -> Center,
        Frame -> All, FrameStyle -> GrayLevel[0.85]]
]

(* Colors picked up from $heapTagFill so THeapStrip and TINetGraph agree. *)
THeapStripColor[tag_String] := Switch[tag,
    "Lam" | "Bri",  RGBColor[Lookup[$heapTagFill, "Lam", "#f2e8ff"]],
    "App",          RGBColor[Lookup[$heapTagFill, "App", "#f3f3f3"]],
    "Sup" | "Usp",  RGBColor[Lookup[$heapTagFill, "Sup", "#e4d6fc"]],
    "Var" | "Ref",  RGBColor[Lookup[$heapTagFill, "Var", "#eeeeee"]],
    "Num",          RGBColor["#fde2e2"],
    "Op2",          RGBColor["#cfe2ff"],
    "Era",          RGBColor["#ffffff"],
    "Ten",          RGBColor[Lookup[$heapTagFill, "Ten", "#e0e0e0"]],
    "Top",          RGBColor[Lookup[$heapTagFill, "Top", "#cce5ff"]],
    _,              RGBColor["#f3f3f3"]]

THeapStrip[] := THeapStrip[1, THeapSnapshot[]["HeapSize"]]

THeapStrip[lo_Integer, hi_Integer] /; hi >= lo := Block[{rows, size = hi - lo + 1},
    rows = Table[
        With[{h = THeapRead[i]},
            {i, h["Tag"], h["Val"], THeapStripColor[h["Tag"]]}],
        {i, lo, hi}];
    Graphics[
        MapIndexed[
            Function[{row, idx},
                With[{
                    x = idx[[1]] - 1,
                    loc = row[[1]], tag = row[[2]], val = row[[3]], col = row[[4]]
                },
                    {FaceForm[col],
                     EdgeForm[Directive[GrayLevel[0.15], AbsoluteThickness[1.1]]],
                     Rectangle[{x, 0}, {x + 1, 1.2}],
                     Text[Style["slot " <> ToString[loc], 9, FontColor -> GrayLevel[0.4]],
                          {x + 0.5, 1.05}],
                     Text[Style[ToUpperCase[ToString[tag]], Bold, 12, FontColor -> Black],
                          {x + 0.5, 0.72}],
                     Text[Style["val=" <> ToString[val], 10,
                                FontColor -> RGBColor[0.15, 0.15, 0.55]],
                          {x + 0.5, 0.35}]}]],
            rows],
        PlotRange -> {{-0.05, size + 0.05}, {-0.1, 1.3}},
        ImageSize -> 120 * size,
        AspectRatio -> 1.3 / size,
        Background -> White]
]

THeapStrip[lo_Integer, hi_Integer] /; hi < lo := Graphics[{},
    ImageSize -> 40, Background -> White]

TProfileEnable[] := (loadLibrary[]; thvmProfileEnableFn[])

TProfileData[] := Block[{
    raw,
    n = 29
},
    loadLibrary[];
    raw = Normal[thvmProfileDataFn[]];
    If[Length[raw] < 3 n + 18, Return[$Failed]];
    <|
        "UOpTime" -> AssociationThread[Take[Values[$uopName], n], Take[raw, n] / 1.*^6],
        "UOpCount" -> AssociationThread[Take[Values[$uopName], n], Round /@ raw[[n + 1 ;; 2 n]]],
        "PhaseMs" -> <|
            "Forward" -> raw[[3 n + 1]] / 1.*^6,
            "Backward" -> raw[[3 n + 2]] / 1.*^6,
            "Adam" -> raw[[3 n + 3]] / 1.*^6,
            "Reset" -> raw[[3 n + 4]] / 1.*^6,
            "Other" -> raw[[3 n + 5]] / 1.*^6
        |>,
        "Memory" -> <|
            "AllocMB" -> raw[[3 n + 6]] / 1.*^6,
            "PeakMB" -> raw[[3 n + 8]] / 1.*^6,
            "CurrentMB" -> raw[[3 n + 9]] / 1.*^6
        |>,
        "Tensors" -> <|
            "Peak" -> Round[raw[[3 n + 10]]],
            "Created" -> Round[raw[[3 n + 11]]],
            "Freed" -> Round[raw[[3 n + 12]]]
        |>,
        "Dispatches" -> Total[Round /@ raw[[n + 1 ;; 2 n]]]
    |>
]

TProfileSummary[] := Block[{
    pd,
    phaseRows,
    uopRows,
    active
},
    pd = TProfileData[];
    If[pd === $Failed, Return["Profiling not enabled. Call TProfileEnable[] first."]];

    phaseRows = Select[List @@@ Normal[pd["PhaseMs"]], #[[2]] > 0.001 &];
    active = Select[
        MapThread[
            {#1, #2, #3} &,
            {Keys[pd["UOpTime"]], Values[pd["UOpTime"]], Values[pd["UOpCount"]]}
        ],
        #[[3]] > 0 &
    ];
    uopRows = SortBy[active, -#[[2]] &];

    Column[
        {
            Style["Phase Timing", Bold, 14],
            Grid[
                Prepend[phaseRows, {"Phase", "ms"}],
                Frame -> All,
                Background -> {None, {GrayLevel[0.9], None}},
                Alignment -> {{Left, Right}},
                Spacings -> {2, 0.5}
            ],
            "",
            Style["Top UOps by Time", Bold, 14],
            Grid[
                Prepend[Take[uopRows, UpTo[10]], {"UOp", "ms", "Count"}],
                Frame -> All,
                Background -> {None, {GrayLevel[0.9], None}},
                Alignment -> {{Left, Right, Right}},
                Spacings -> {2, 0.5}
            ],
            "",
            Style["Resources", Bold, 14],
            Grid[
                {
                    {"Peak Memory", ToString[pd["Memory", "PeakMB"]] <> " MB"},
                    {"Tensors Created", pd["Tensors", "Created"]},
                    {"Peak Tensors", pd["Tensors", "Peak"]},
                    {"Total Dispatches", pd["Dispatches"]}
                },
                Frame -> All,
                Alignment -> {{Left, Right}},
                Spacings -> {2, 0.5}
            ]
        },
        Spacings -> 1
    ]
]

TProfileTimeline[snapshots_List] := Block[{
    phases,
    labels,
    dispatches,
    mem
},
    phases = Values[#["PhaseMs"]] & /@ snapshots;
    labels = Keys[First[snapshots]["PhaseMs"]];
    dispatches = #["Dispatches"] & /@ snapshots;
    mem = #["Memory", "PeakMB"] & /@ snapshots;
    GraphicsGrid[
        {
            {
                ListLinePlot[
                    Transpose[phases],
                    PlotLegends -> labels,
                    PlotLabel -> "Phase Timing per Step (ms)",
                    AxesLabel -> {"Step", "ms"},
                    ImageSize -> 350
                ],
                ListLinePlot[
                    dispatches,
                    PlotLabel -> "GPU Dispatches per Step",
                    AxesLabel -> {"Step", "Dispatches"},
                    ImageSize -> 350
                ]
            },
            {
                ListLinePlot[
                    mem,
                    PlotLabel -> "Peak Memory (MB)",
                    AxesLabel -> {"Step", "MB"},
                    ImageSize -> 350,
                    PlotRange -> {0, All}
                ],
                SpanFromLeft
            }
        },
        ImageSize -> 750
    ]
]
