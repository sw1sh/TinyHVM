(* ImportDOTString imports a Graphviz DOT digraph string as a Wolfram Language Graph or Graphics. *)

BeginPackage["TinyHVM`ImportCallGraphDOT`"];

ImportDOTString::usage =
    "ImportDOTString[dot, opts] imports a Graphviz DOT digraph string as a Wolfram Language Graph or Graphics.";

Begin["`Private`"];

(* Compared with `dot -Tpng`:
   - `"UseGraphvizLayout" -> True` (default) uses `dot -Tplain` to recover node centers and
     node sizes, then rebuilds the graph in WL coordinates.
   - Extreme aspect ratios are clamped so raster exports stay readable.
   - `Method -> "Graphviz"` / `"GraphvizRaster"` returns a framed PNG preview rather than a
     `Graph`.
   - `Method -> "GraphvizGraphics"` uses `dot -Tjson` drawing primitives.
   - `Method -> "Parser"` / `"Wolfram"` keeps everything in WL, including the native layered
     layout fallback. *)

ClearAll[
    ImportDOTString,
    importDOTFile,
    findDotAuto,
    rankDirOrientation,
    parseAttrs,
    parseGraphDefaults,
    attrBlock,
    stripComments,
    nodeIdPat,
    nodeLinePat,
    unquote,
    hexToRGB,
    colorToRGB,
    resolvePlotLabel,
    dropImportDOTMethodOpts,
    importGraphvizPNG,
    importGraphvizGraphics,
    jsonBezierSegs,
    bezierPoint,
    bezierSplit,
    pointInConvexPolygonQ,
    shapeLocalPolygon,
    shapeInsideQ,
    bezierClipToInside,
    bezierClipToShape,
    clipBezierChainToNodeShapes,
    bezierTangent,
    rectFromCenterWH,
    rectsOverlapQ,
    textLabelWH,
    chooseFreeLabelPoint,
    importWolframGraph,
    importGraphBuiltIn,
    resolveDotExecutable,
    graphvizPlainNodeXY,
    parsePlainNodeLine,
    labelMinPlotWH,
    parseDotNodeLine,
    parseDotEdgeLine,
    parseDotGraphData,
    nodeKey,
    nodeFillColor,
    nodeStrokeColor,
    nodeLabel,
    nodeDotFontSize,
    nodeShape,
    fallbackLayoutData,
    scaleLayoutData,
    resolveLayoutData,
    layoutDisplayData,
    vertexSizingData,
    graphvizJsonTextPrims,
    graphvizJsonNodePrims,
    graphvizJsonArrowPoly,
    buildGraphvizJsonData,
    edgeRenderData
];

Quiet @ Check[
    Get @ FileNameJoin[{DirectoryName[$InputFileName], "LayeredDigraphSugiyama.wl"}],
    $Failed
];

(* Geometry helpers *)

(* Minimum {width, height} in plot units so VertexLabels fit. Empty labels add no padding. *)
labelMinPlotWH[lbl_String, fontSize_ ? NumericQ] := Block[{
    lines, mw, lh,
    cw, lhPt, t, padX, padY
},
    t = StringTrim[lbl];
    If[t === "", Return[{0., 0.}]];

    lines = StringSplit[lbl, "\n"];
    mw = If[lines === {}, 0, Min[72, Max[StringLength /@ lines]]];
    lh = Max[1, Length[lines]];
    cw = 0.51 fontSize;
    lhPt = Max[8.5, 1.12 fontSize];
    padX = 12. + 0.08 fontSize;
    padY = 8. + 0.06 fontSize;

    {
        Max[fontSize * 3.2 + 4., mw * cw + padX],
        Max[fontSize * 2.6 + 2., lh * lhPt + padY]
    }
]

(* First pure-WL port chunk from Graphviz `splines.c` / `shapes.c`:
   cubic Bezier subdivision + binary-search clipping to node boundaries. *)

bezierPoint[curve : {{_, _} ..}, t_ ? NumericQ] := Block[{
    p0, p1, p2, p3,
    a, b, c, d, e
},
    {p0, p1, p2, p3} = N /@ curve[[1 ;; 4]];
    a = (1. - t) p0 + t p1;
    b = (1. - t) p1 + t p2;
    c = (1. - t) p2 + t p3;
    d = (1. - t) a + t b;
    e = (1. - t) b + t c;
    (1. - t) d + t e
]

bezierSplit[curve : {{_, _} ..}, t_ ? NumericQ] := Block[{
    p0, p1, p2, p3,
    a, b, c, d, e, f
},
    {p0, p1, p2, p3} = N /@ curve[[1 ;; 4]];
    a = (1. - t) p0 + t p1;
    b = (1. - t) p1 + t p2;
    c = (1. - t) p2 + t p3;
    d = (1. - t) a + t b;
    e = (1. - t) b + t c;
    f = (1. - t) d + t e;
    {{p0, a, d, f}, {f, e, c, p3}}
]

pointInConvexPolygonQ[poly : {{_, _} ..}, p : {_, _}] := Block[{
    edges, sgn
},
    edges = Partition[Append[poly, First[poly]], 2, 1];
    sgn =
        DeleteCases[
            Sign[
                Chop[
                    (#[[2, 1]] - #[[1, 1]]) (p[[2]] - #[[1, 2]]) -
                    (#[[2, 2]] - #[[1, 2]]) (p[[1]] - #[[1, 1]])
                ]
            ] & /@ edges,
            0
        ];
    sgn === {} || SameQ @@ sgn
]

shapeLocalPolygon[shape_String, {rx_ ? NumericQ, ry_ ? NumericQ}] := Block[{
    sh = ToLowerCase[shape]
},
    Which[
        sh === "triangle",
            {{0., ry}, {-rx, -ry}, {rx, -ry}},
        sh === "invtriangle",
            {{0., -ry}, {-rx, ry}, {rx, ry}},
        StringContainsQ[sh, "diamond"],
            {{0., ry}, {rx, 0.}, {0., -ry}, {-rx, 0.}},
        StringContainsQ[sh, "hexagon"],
            Table[{rx Cos[Pi / 6 + k Pi / 3], ry Sin[Pi / 6 + k Pi / 3]}, {k, 0, 5}],
        StringContainsQ[sh, "pentagon"],
            Table[{rx Cos[Pi / 2 + 2 k Pi / 5], ry Sin[Pi / 2 + 2 k Pi / 5]}, {k, 0, 4}],
        MemberQ[{"box", "rectangle", "rect", "square", "msquare"}, sh] || StringContainsQ[sh, "record"],
            {{-rx, -ry}, {rx, -ry}, {rx, ry}, {-rx, ry}},
        True,
            {}
    ]
]

shapeInsideQ[shape_String, size : {w_ ? NumericQ, h_ ? NumericQ}, p : {_, _}] := Block[{
    rx = Max[10.^-9, w / 2.], ry = Max[10.^-9, h / 2.],
    sh, poly, r0
},
    sh = ToLowerCase[shape];
    Which[
        MemberQ[{"box", "rectangle", "rect", "square", "msquare"}, sh] || StringContainsQ[sh, "record"],
            Abs[p[[1]]] <= rx && Abs[p[[2]]] <= ry,
        MemberQ[{"circle", "doublecircle", "point", "mcircle"}, sh],
            r0 = Min[rx, ry];
            p[[1]]^2 + p[[2]]^2 <= r0^2,
        MemberQ[{"ellipse", "oval"}, sh],
            (p[[1]] / rx)^2 + (p[[2]] / ry)^2 <= 1.,
        StringContainsQ[sh, "diamond"],
            Abs[p[[1]]] / rx + Abs[p[[2]]] / ry <= 1.,
        MemberQ[{"triangle", "invtriangle", "hexagon", "pentagon"}, sh],
            poly = shapeLocalPolygon[sh, {rx, ry}];
            pointInConvexPolygonQ[poly, p],
        True,
            (p[[1]] / rx)^2 + (p[[2]] / ry)^2 <= 1.
    ]
]

bezierClipToInside[curve : {{_, _} ..}, insideFn_, leftInside_ : True] := Block[{
    low = 0., high = 1.,
    pt, opt, best = curve[[1 ;; 4]], found = False,
    t, left, right, seg = curve[[1 ;; 4]]
},
    pt = If[leftInside, curve[[1]], curve[[4]]];
    While[
        True,
        opt = pt;
        t = (high + low) / 2.;
        {left, right} = bezierSplit[curve[[1 ;; 4]], t];
        seg = If[leftInside, right, left];
        pt = If[leftInside, right[[1]], left[[4]]];
        If[
            TrueQ[insideFn[pt]],
            If[leftInside, low = t, high = t];
            best = seg;
            found = True,
            If[leftInside, high = t, low = t]
        ];
        If[Max[Abs[opt - pt]] <= 0.5, Break[]]
    ];
    If[found, best, seg]
]

bezierClipToShape[
    curve : {{_, _} ..},
    center : {_, _},
    size : {_, _},
    shape_String,
    leftInside_ : Automatic
] := Block[{
    local, inside, li, clipped
},
    local = N[(# - center) & /@ curve[[1 ;; 4]]];
    inside = shapeInsideQ[shape, size, #] &;
    li = If[leftInside === Automatic, TrueQ[inside[local[[1]]]], leftInside];
    clipped = bezierClipToInside[local, inside, li];
    (# + center) & /@ clipped
]

clipBezierChainToNodeShapes[
    pts : {{_, _} ..},
    tailCenter : {_, _},
    tailSize : {_, _},
    tailShape_String,
    headCenter : {_, _},
    headSize : {_, _},
    headShape_String
] := Block[{
    ps = N[pts], pn, start = 1, end,
    seg, tailInside, headInside
},
    pn = Length[ps];
    If[pn < 4, Return[ps]];

    end = pn - 3;
    tailInside = shapeInsideQ[tailShape, tailSize, # - tailCenter] &;
    While[start <= pn - 3 && TrueQ[tailInside[ps[[start + 3]]]], start += 3];
    If[
        start <= pn - 3,
        seg = bezierClipToShape[ps[[start ;; start + 3]], tailCenter, tailSize, tailShape, True];
        ps[[start ;; start + 3]] = seg
    ];

    headInside = shapeInsideQ[headShape, headSize, # - headCenter] &;
    While[end >= 1 && TrueQ[headInside[ps[[end]]]], end -= 3];
    If[
        end >= 1,
        seg = bezierClipToShape[ps[[end ;; end + 3]], headCenter, headSize, headShape, False];
        ps[[end ;; end + 3]] = seg
    ];

    While[start < pn - 3 && Norm[ps[[start]] - ps[[start + 3]]] <= 10.^-3, start += 3];
    While[end > 1 && Norm[ps[[end]] - ps[[end + 3]]] <= 10.^-3, end -= 3];
    ps[[start ;; end + 3]]
]

bezierTangent[curve : {{_, _} ..}, t_ ? NumericQ] := Block[{
    p0, p1, p2, p3
},
    {p0, p1, p2, p3} = N /@ curve[[1 ;; 4]];
    3. ((1. - t)^2 (p1 - p0) + 2. (1. - t) t (p2 - p1) + t^2 (p3 - p2))
]

rectFromCenterWH[c : {_, _}, wh : {_, _}, pad_ : 0.] :=
    {c - wh / 2 - {pad, pad}, c + wh / 2 + {pad, pad}};

rectsOverlapQ[{ll1_, ur1_}, {ll2_, ur2_}] :=
    !(ur1[[1]] < ll2[[1]] || ur2[[1]] < ll1[[1]] || ur1[[2]] < ll2[[2]] || ur2[[2]] < ll1[[2]]);

textLabelWH[lbl_String, fontSize_ ? NumericQ] := Block[{
    t = StringTrim[lbl]
},
    If[t === "", {0., 0.}, 0.66 labelMinPlotWH[t, fontSize]]
]

chooseFreeLabelPoint[cands_List, wh : {_, _}, forbidden_List] := Block[{
    rect
},
    FirstCase[
        cands,
        p_ /; (rect = rectFromCenterWH[p, wh, 2.]; ! AnyTrue[forbidden, rectsOverlapQ[rect, #] &]) :> p,
        First[cands]
    ]
]

(* Public options *)

Options[ImportDOTString] = Join[
    {
        Method -> Automatic,
        DotExecutable -> Automatic,
        "UseGraphvizLayout" -> True,
        "LayoutScale" -> Automatic,
        "DPI" -> 150,
        "RasterSize" -> Automatic
    },
    Options[Graph]
];

ImportDOTString::dotfail =
    "Graphviz rasterization failed on `1`: `2`.";
ImportDOTString::impfail =
    "Built-in Import failed; using the bundled DOT parser.";
ImportDOTString::badmeth = "Unknown Method `1`.";
ImportDOTString::gvplain =
    "Could not use dot -Tplain for layout; using WL layered layout.";

dropImportDOTMethodOpts[opts_List] := Block[{
    o = opts
},
    o = DeleteCases[o, Verbatim[Method] -> _, {1}];
    o = DeleteCases[o, Verbatim[DotExecutable] -> _, {1}];
    o = DeleteCases[o, Verbatim["DPI"] -> _, {1}];
    o = DeleteCases[o, Verbatim["RasterSize"] -> _, {1}];
    o = DeleteCases[o, Verbatim["UseGraphvizLayout"] -> _, {1}];
    o = DeleteCases[o, Verbatim["LayoutScale"] -> _, {1}];
    o
]

findDotAuto[] := Block[{
    proc, p2, c
},
    (* Prefer absolute paths: notebook kernels often have a minimal PATH. *)
    Do[
        c = FileNameJoin[{d, "dot"}];
        If[FileExistsQ[c] && Quiet @ RunProcess[{c, "-V"}]["ExitCode"] === 0, Return[c]],
        {d, {"/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"}}
    ];

    proc = Quiet @ RunProcess[{"which", "dot"}];
    If[
        AssociationQ[proc] && proc["ExitCode"] === 0 && StringTrim[proc["StandardOutput"]] =!= "",
        Return[StringTrim[proc["StandardOutput"]]]
    ];

    p2 = Quiet @ RunProcess[{"dot", "-V"}];
    If[AssociationQ[p2] && p2["ExitCode"] === 0, "dot", $Failed]
]

resolveDotExecutable[opts : OptionsPattern[ImportDOTString]] := Block[{
    dot = OptionValue[DotExecutable]
},
    Which[
        StringQ[dot] && dot =!= "Automatic" && FileExistsQ[ExpandFileName[dot]],
            ExpandFileName[dot],
        dot === Automatic || dot === "Automatic",
            findDotAuto[],
        True,
            $Failed
    ]
]

parsePlainNodeLine[line_String] := Block[{
    trim, tok, name,
    x, y, w, h, sh9
},
    trim = StringTrim[line];
    If[! StringStartsQ[trim, "node "], Return[$Failed]];

    tok = StringSplit[trim, RegularExpression["\\s+"]];
    If[Length[tok] < 6 || tok[[1]] =!= "node", Return[$Failed]];

    (* TinyHVM / typical graphs: unquoted id `t0`, `rootout_t3` — token 2 *)
    name = unquote[tok[[2]]];
    x = Quiet @ Check[ToExpression[tok[[3]]], $Failed];
    y = Quiet @ Check[ToExpression[tok[[4]]], $Failed];
    If[! MatchQ[x, _Real | _Integer] || ! MatchQ[y, _Real | _Integer], Return[$Failed]];

    (* Plain format: node name x y width height label ... — w/h in inches. *)
    If[
        Length[tok] >= 7,
        {w, h} = Quiet @ Check[{ToExpression[tok[[5]]], ToExpression[tok[[6]]]}, {$Failed, $Failed}];
        If[
            MatchQ[w, _Real | _Integer] && MatchQ[h, _Real | _Integer] && w > 0 && h > 0,
            If[
                Length[tok] >= 10,
                sh9 = ToLowerCase @ StringTrim @ StringReplace[tok[[9]], "\"" -> ""];
                Return[{name, x, y, w, h, sh9}],
                Return[{name, x, y, w, h}]
            ]
        ]
    ];

    {name, x, y}
]

(* Returns <| "graphSize" -> {gw,gh}, "coords" -> <| name -> {xw,yw}, ... |>,
   "sizes" -> <| name -> {wp, hp} in points (matches coords) |> |>,
   WL-style coords (y up, inches * 72). Graphviz plain is y-up — no y flip. *)
graphvizPlainNodeXY[path_String, opts : OptionsPattern[ImportDOTString]] := Block[{
    dot, proc, text, lines,
    gw, gh, gr, scale,
    posAssoc, sizeAssoc, shapeAssoc
},
    dot = resolveDotExecutable[opts];
    If[dot === $Failed, Return[$Failed]];
    proc = Quiet @ RunProcess[{dot, "-Tplain", ExpandFileName[path]}];
    If[! AssociationQ[proc] || proc["ExitCode"] =!= 0, Return[$Failed]];

    text = proc["StandardOutput"];
    lines = Select[StringSplit[text, "\n"], StringTrim[#] =!= "" &];
    gr =
        With[{gl = Select[lines, StringStartsQ[StringTrim[#], "graph "] &]},
            If[
                gl === {},
                $Failed,
                With[{tok = StringSplit[StringTrim[First[gl]]]},
                    If[Length[tok] >= 4 && tok[[1]] === "graph", Quiet @ Check[ToExpression /@ tok[[2 ;; 4]], $Failed], $Failed]
                ]
            ]
        ];
    If[! ListQ[gr] || Length[gr] =!= 3 || MemberQ[gr, $Failed | $Aborted], Return[$Failed]];

    gw = gr[[2]];
    gh = gr[[3]];
    If[! (NumericQ[gw] && NumericQ[gh] && gw > 0 && gh > 0), Return[$Failed]];

    scale = 72.0;
    posAssoc = <||>;
    sizeAssoc = <||>;
    shapeAssoc = <||>;
    Do[
        If[
            StringStartsQ[StringTrim[ln], "node "],
            With[{p = parsePlainNodeLine[ln]},
                Which[
                    ListQ[p] && Length[p] === 6,
                        AssociateTo[posAssoc, p[[1]] -> {scale p[[2]], scale p[[3]]}];
                        AssociateTo[sizeAssoc, p[[1]] -> {scale p[[4]], scale p[[5]]}];
                        AssociateTo[shapeAssoc, p[[1]] -> p[[6]]],
                    ListQ[p] && Length[p] === 5,
                        AssociateTo[posAssoc, p[[1]] -> {scale p[[2]], scale p[[3]]}];
                        AssociateTo[sizeAssoc, p[[1]] -> {scale p[[4]], scale p[[5]]}],
                    ListQ[p] && Length[p] === 3,
                        AssociateTo[posAssoc, p[[1]] -> {scale p[[2]], scale p[[3]]}]
                ]
            ]
        ],
        {ln, lines}
    ];
    If[Length[posAssoc] === 0, Return[$Failed]];

    <|"graphSize" -> {gw, gh}, "coords" -> posAssoc, "sizes" -> sizeAssoc, "shapes" -> shapeAssoc|>
    ]

rankDirOrientation[raw_String] := Block[{
    rd
},
    rd = ToUpperCase @ First[
        Flatten @ {
            StringCases[
                raw,
                "rankdir" ~~ Whitespace... ~~ "=" ~~ Whitespace... ~~ "\"" ~~ rr : Shortest[__] ~~ "\"" :> rr
            ],
            StringCases[
                raw,
                "rankdir" ~~ Whitespace... ~~ "=" ~~ Whitespace... ~~ a : LetterCharacter ~~ b : LetterCharacter :> StringJoin[a, b]
            ]
        },
        "TB"
    ];
    Lookup[<|"TB" -> Top, "BT" -> Bottom, "LR" -> Left, "RL" -> Right|>, rd, Top]
]

parseAttrs[attrStr_String] :=
    Association[
        Rule @@@ StringCases[
            attrStr,
            RegularExpression["(\\w+)\\s*=\\s*(?:\"([^\"]*)\"|([^,\\]\\s]+))"] :>
                {"$1", If["$2" =!= "", "$2", "$3"]}
        ]
    ];

(* DOT labels can contain `[` / `]` (e.g. "[2,4]"), so do not stop at the
   first closing bracket. Take everything between the first `[` and the last
   `]` on the statement line. *)
attrBlock[s_String] := Block[{
    l,
    r
},
    l = StringPosition[s, "["];
    r = StringPosition[s, "]"];
    If[l === {} || r === {} || r[[-1, 1]] <= l[[1, 1]], "",
        StringTake[s, {l[[1, 1]] + 1, r[[-1, 1]] - 1}]
    ]
]

(* Top-level `node [ ... ]` / `edge [ ... ]` defaults (Graphviz merges into each stmt). *)
parseGraphDefaults[raw_String, head_String] := Block[{
    lines,
    cand,
    attrStr
},
    lines = Select[StringSplit[raw, "\n"], StringTrim[#] =!= "" &];
    cand =
        Quiet @ Cases[
            lines,
            s_ /; StringMatchQ[StringTrim[s], head ~~ Whitespace... ~~ "[" ~~ __] :> s
        ];
    If[cand === {}, Return[<||>]];
    attrStr = attrBlock[StringTrim[First[cand]]];
    parseAttrs[attrStr]
]

stripComments[s_String] :=
    StringReplace[s, {Shortest["//" ~~ __] ~~ EndOfLine -> "", "\r" -> ""}];

(* Prefix up to "[" — used in StringCases for the node id. *)
nodeIdPat = RegularExpression["^\\s*([A-Za-z_][A-Za-z0-9_]*|\"[^\"]+\")\\s*\\["];
(* Whole-line test: StringMatchQ matches the full string, so allow ".*" after "[". *)
nodeLinePat = RegularExpression["^\\s*([A-Za-z_][A-Za-z0-9_]*|\"[^\"]+)\\s*\\[.*"];

unquote[s_String] := If[StringMatchQ[s, "\"" ~~ __ ~~ "\""], StringTrim[s, "\""], s];

hexToRGB[hex_String] :=
    RGBColor @@ (IntegerDigits[FromDigits[StringDrop[hex, 1], 16], 256, 3] / 255.);

colorToRGB[hexToRGB_, s_String] := Which[
    StringMatchQ[s, "#" ~~ __], hexToRGB[s],
    MemberQ[{"white", "White"}, s], RGBColor[1, 1, 1],
    MemberQ[{"black", "Black"}, s], RGBColor[0, 0, 0],
    True, Quiet @ Check[ColorConvert[s, RGBColor], GrayLevel[0.55]]
]

resolvePlotLabel[path_String, label_] :=
    Which[
        label === None, None,
        label === Automatic, FileBaseName[path],
        True, label
    ]

importGraphvizPNG[path_String, opts : OptionsPattern[ImportDOTString]] := Block[{
    dot, dpi, proc, img,
    w0, cap, rs, title
},
    dot = resolveDotExecutable[opts];
    If[dot === $Failed, Return[$Failed]];

    dpi = Round[OptionValue["DPI"]];
    proc = Quiet @ RunProcess[{dot, "-Tpng", "-Gdpi=" <> ToString[dpi], ExpandFileName[path]}];
    If[! AssociationQ[proc] || proc["ExitCode"] =!= 0, Return[$Failed]];

    img = ImportString[proc["StandardOutput"], "PNG"];
    If[Head[img] =!= Image, Return[$Failed]];

    w0 = First[ImageDimensions[img]];
    cap = 900;
    rs = OptionValue["RasterSize"];
    img = Which[
        rs === Automatic,
            If[w0 > cap, ImageResize[img, UpTo[cap]], img],
        IntegerQ[rs],
            ImageResize[img, UpTo[rs]],
        MatchQ[rs, {_Integer | UpTo[_], _Integer | UpTo[_]} | UpTo[_]],
            ImageResize[img, rs],
        True,
            img
    ];

    title = resolvePlotLabel[path, OptionValue[PlotLabel]];

    If[
        title === None,
        Framed[
            img,
            Background -> White,
            FrameMargins -> 8,
            FrameStyle -> GrayLevel[0.88],
            RoundingRadius -> 3
        ],
        Framed[
            Column[
                {Style[title, Bold, 13, FontFamily -> "Helvetica"], img},
                Spacings -> 1.1,
                Alignment -> Center
            ],
            Background -> White,
            FrameMargins -> 10,
            FrameStyle -> GrayLevel[0.85],
            RoundingRadius -> 4
        ]
    ]
]

jsonBezierSegs[pts_List] := Block[{
    n = Length[pts]
},
    Which[
        n < 4,
            {},
        n == 4,
            {pts},
        Mod[n - 1, 3] == 0,
            Table[pts[[1 + 3 (k - 1) ;; 4 + 3 (k - 1)]], {k, 1, (n - 1) / 3}],
        True,
            {pts[[1 ;; 4]]}
    ]
]

importGraphvizGraphics[path_String, opts : OptionsPattern[ImportDOTString]] := Block[{
    dot, proc, data, bb,
    x0, y0, x1, y1,
    toColor, styleDash, drawOpsPrims,
    edgePrims, nodePrims, title
},
    dot = resolveDotExecutable[opts];
    If[dot === $Failed, Return[$Failed]];

    proc = Quiet @ RunProcess[{dot, "-Tjson", ExpandFileName[path]}];
    If[! AssociationQ[proc] || proc["ExitCode"] =!= 0, Return[$Failed]];

    data = Quiet @ Check[ImportString[proc["StandardOutput"], "RawJSON"], $Failed];
    If[! AssociationQ[data], Return[$Failed]];

    bb = ToExpression /@ StringSplit[Lookup[data, "bb", "0,0,100,100"], ","];
    If[Length[bb] =!= 4, bb = {0, 0, 100, 100}];
    {x0, y0, x1, y1} = bb;

    toColor[s_] := colorToRGB[hexToRGB, s];

    styleDash[st_] := Which[
        st === "dashed", Dashing[{6, 4}],
        st === "dotted", Dashing[{1, 4}],
        True, Sequence @@ {}
    ];

    drawOpsPrims[ops_List] := Block[{
        stroke = GrayLevel[0.], fill = None, fs = 10., face = "Helvetica",
        style = "solid", prims = {}, pts, rect, txt, pt, align
    },
        Do[
            Switch[
                Lookup[op, "op", ""],
                "c",
                    stroke = toColor[Lookup[op, "color", "black"]],
                "C",
                    fill = toColor[Lookup[op, "color", "white"]],
                "F",
                    fs = 0.52 N @ Lookup[op, "size", 10.];
                    face = Lookup[op, "face", "Helvetica"],
                "S",
                    style = Lookup[op, "style", "solid"],
                "P",
                    pts = Lookup[op, "points", {}];
                    AppendTo[
                        prims,
                        {
                            FaceForm[fill],
                            EdgeForm[{stroke, AbsoluteThickness[0.9], styleDash[style]}],
                            Polygon[pts]
                        }
                    ],
                "E",
                    rect = Lookup[op, "rect", {0., 0., 1., 1.}];
                    AppendTo[
                        prims,
                        {
                            FaceForm[fill],
                            EdgeForm[{stroke, AbsoluteThickness[0.9], styleDash[style]}],
                            Disk[rect[[1 ;; 2]], rect[[3 ;; 4]]]
                        }
                    ],
                "b",
                    pts = Lookup[op, "points", {}];
                    AppendTo[
                        prims,
                        Join[{Directive[stroke, AbsoluteThickness[1.1], styleDash[style]]}, BezierCurve /@ jsonBezierSegs[pts]]
                    ],
                "T",
                    txt = Lookup[op, "text", ""];
                    pt = Lookup[op, "pt", {0., 0.}];
                    align = Lookup[op, "align", "c"];
                    AppendTo[
                        prims,
                        Text[
                            Style[
                                txt,
                                FontSize -> fs,
                                FontFamily -> face,
                                FontWeight -> Plain,
                                FontColor -> stroke,
                                TextAlignment -> Which[align === "l", Left, align === "r", Right, True, Center]
                            ],
                            pt,
                            {Which[align === "l", -1, align === "r", 1, True, 0], -1}
                        ]
                    ],
                _,
                    Null
            ],
            {op, ops}
        ];
        Flatten[prims, 1]
    ];

    edgePrims =
        Flatten[
            Table[
                Join[
                    drawOpsPrims[Lookup[e, "_draw_", {}]],
                    drawOpsPrims[Lookup[e, "_hdraw_", {}]],
                    drawOpsPrims[Lookup[e, "_ldraw_", {}]],
                    drawOpsPrims[Lookup[e, "_tldraw_", {}]]
                ],
                {e, Lookup[data, "edges", {}]}
            ],
            1
        ];

    nodePrims =
        Flatten[
            Table[
                Join[
                    drawOpsPrims[Lookup[obj, "_draw_", {}]],
                    drawOpsPrims[Lookup[obj, "_ldraw_", {}]]
                ],
                {obj, Lookup[data, "objects", {}]}
            ],
            1
        ];

    title = resolvePlotLabel[path, OptionValue[PlotLabel]];

    Show[
        Graphics[
            Join[edgePrims, nodePrims],
            PlotRange -> {{x0, x1}, {y0, y1}},
            PlotRangePadding -> Scaled[0.02],
            Background -> White,
            ImagePadding -> 12,
            AspectRatio -> If[x1 > x0, (y1 - y0) / (x1 - x0), Automatic]
        ],
        PlotLabel ->
            If[
                title === None,
                None,
                Style[
                    title,
                    FontFamily -> "Helvetica",
                    FontSize -> 13,
                    FontWeight -> "SemiBold",
                    FontColor -> GrayLevel[0.2]
                ]
            ],
        PerformanceGoal -> "Quality"
    ]
]


(* Wolfram graph import phases *)

parseDotNodeLine[line_String] := Block[{
    m, id, attrStr
},
    If[
        StringMatchQ[StringTrim[line], RegularExpression["(?i)^(node|edge)\\s*\\["]],
        Return[Nothing]
    ];

    m = StringCases[line, nodeIdPat :> "$1"];
    If[m === {}, Return[Nothing]];

    id = ToString[unquote[First[m]]];
    attrStr = attrBlock[line];

    <|"name" -> id, "attrs" -> parseAttrs[attrStr]|>
]


parseDotEdgeLine[line_String] := Block[{
    trim, left, tail, right, attrStr
},
    trim = StringTrim[line];
    If[! StringContainsQ[trim, "->"], Return[Nothing]];

    {left, tail} = StringSplit[trim, "->", 2];
    left = StringTrim[left];
    tail = StringTrim @ StringReplace[tail, Shortest["//" ~~ __] -> ""];
    right = StringTrim[StringTrim[StringSplit[tail, "[", 2][[1]]], ";"];
    attrStr = attrBlock[tail];

    <|
        "src" -> ToString[unquote[left]],
        "tgt" -> ToString[unquote[right]],
        "attrs" -> parseAttrs[attrStr]
    |>
]


parseDotGraphData[raw_String] := Block[{
    lines, line,
    nodeDefaults, edgeDefaults,
    nodeLines, edgeLines, nodes, edges,
    allNames, edgeNodes, nodeAttrs
},
    lines = Select[StringSplit[raw, "\n"], StringTrim[#] =!= "" &];
    nodeDefaults = parseGraphDefaults[raw, "node"];
    edgeDefaults = parseGraphDefaults[raw, "edge"];

    nodeLines =
        Flatten @ Reap[
            Do[
                line = StringTrim[l];
                If[StringContainsQ[line, "->"] || StringStartsQ[line, "//"], Continue[]];
                If[
                    StringMatchQ[
                        line,
                        Whitespace... ~~ ("graph" | "node" | "edge" | "digraph" | "subgraph" | "{") ~~ __
                    ],
                    Continue[]
                ];
                If[StringMatchQ[line, "}" ~~ ___], Continue[]];
                If[StringContainsQ[line, "["] && StringMatchQ[line, nodeLinePat], Sow[line]],
                {l, lines}
            ]
        ][[2]];

    edgeLines = Select[lines, StringContainsQ[#, "->"] &];

    nodes = DeleteCases[parseDotNodeLine /@ nodeLines, Nothing | Null];
    edges = DeleteCases[parseDotEdgeLine /@ edgeLines, Nothing | Null];

    If[
        AssociationQ[nodeDefaults] && Length[nodeDefaults] > 0,
        nodes =
            Table[
                <|"name" -> n["name"], "attrs" -> Join[nodeDefaults, n["attrs"]]|>,
                {n, nodes}
            ]
    ];

    If[
        AssociationQ[edgeDefaults] && Length[edgeDefaults] > 0,
        edges =
            Table[
                <|"src" -> e["src"], "tgt" -> e["tgt"], "attrs" -> Join[edgeDefaults, e["attrs"]]|>,
                {e, edges}
            ]
    ];

    allNames = #["name"] & /@ nodes;
    edgeNodes = Union[#["src"] & /@ edges, #["tgt"] & /@ edges];
    nodes =
        Join[
            nodes,
            <|
                "name" -> #,
                "attrs" -> Join[
                    If[AssociationQ[nodeDefaults], nodeDefaults, <||>],
                    <|"fillcolor" -> "#e0e0e0", "label" -> #|>
                ]
            |> & /@ Complement[edgeNodes, allNames]
        ];

    nodeAttrs = Association[#["name"] -> #["attrs"] & /@ nodes];

    <|"nodes" -> nodes, "edges" -> edges, "nodeAttrs" -> nodeAttrs|>
]


nodeKey[v_, vNames_List] := Which[
    StringQ[v], v,
    IntegerQ[v] && 1 <= v <= Length[vNames], vNames[[v]],
    Head[v] === Symbol, SymbolName[v],
    True, ToString[v]
]


nodeFillColor[nodeAttrs_Association, toColor_, vNames_List, n_] := Block[{
    k = nodeKey[n, vNames], a, fc, c0, st
},
    a = Lookup[nodeAttrs, k, <||>];
    fc = Lookup[a, "fillcolor", ""];
    c0 = Lookup[a, "color", ""];
    st = ToLowerCase[ToString[Lookup[a, "style", ""]]];

    Which[
        StringQ[fc] && fc =!= "", toColor[fc],
        StringContainsQ[st, "filled"] && StringQ[c0] && c0 =!= "", toColor[c0],
        StringQ[c0] && c0 =!= "", toColor[c0],
        True, GrayLevel[0.88]
    ]
]


nodeStrokeColor[nodeAttrs_Association, toColor_, vNames_List, n_] := Block[{
    k = nodeKey[n, vNames], a, c0
},
    a = Lookup[nodeAttrs, k, <||>];
    c0 = Lookup[a, "color", ""];
    If[StringQ[c0] && c0 =!= "", toColor[c0], GrayLevel[0.]]
]


nodeLabel[nodeAttrs_Association, vNames_List, n_] := Block[{
    k = nodeKey[n, vNames], lbl
},
    lbl = Lookup[Lookup[nodeAttrs, k, <||>], "label", k];
    If[StringQ[lbl], StringReplace[lbl, "\\n" -> "\n"], k]
]


nodeDotFontSize[nodeAttrs_Association, vNames_List, n_, fallback_] := Block[{
    k = nodeKey[n, vNames], a, fs
},
    a = Lookup[nodeAttrs, k, <||>];
    fs = Quiet @ Check[ToExpression[ToString[Lookup[a, "fontsize", ""]]], $Failed];
    If[NumericQ[fs] && fs > 0, Clip[fs, {1., 14.}], fallback]
]


nodeShape[nodeAttrs_Association, plainShapes_, vNames_List, n_] := Block[{
    k = nodeKey[n, vNames], dotShape, plainShape
},
    dotShape = Lookup[Lookup[nodeAttrs, k, <||>], "shape", ""];
    dotShape =
        If[
            StringQ[dotShape],
            ToLowerCase[StringTrim @ StringReplace[ToString[dotShape, InputForm], "\"" -> ""]],
            ""
        ];

    plainShape = If[AssociationQ[plainShapes], Lookup[plainShapes, k, ""], ""];
    plainShape = If[StringQ[plainShape], ToLowerCase[StringTrim[plainShape]], ""];

    Which[
        dotShape =!= "" && dotShape =!= "automatic", dotShape,
        AssociationQ[plainShapes] && KeyExistsQ[plainShapes, k] && plainShape =!= "" && plainShape =!= "automatic", plainShape,
        True, "ellipse"
    ]
]


fallbackLayoutData[
    nodeAttrs_Association,
    vNames_List,
    edges_List,
    orient_,
    getLabel_,
    dotFs_
] := Block[{
    wh0, medNw0, vtxFS0,
    e2, sug, sh0,
    wAttr, hAttr
},
    wh0 =
        Association @ Table[
            v -> Block[{a = Lookup[nodeAttrs, v, <||>], lbl = getLabel[v]},
                wAttr = Quiet @ Check[ToExpression[ToString[Lookup[a, "width", ""]]], $Failed];
                hAttr = Quiet @ Check[ToExpression[ToString[Lookup[a, "height", ""]]], $Failed];
                If[
                    NumericQ[wAttr] && NumericQ[hAttr] && wAttr > 0 && hAttr > 0,
                    72. {wAttr, hAttr},
                    labelMinPlotWH[lbl, dotFs[v, 9]]
                ]
            ],
            {v, vNames}
        ];

    medNw0 = If[Length[wh0] > 0, Median[First /@ Values[wh0]], 48.];
    vtxFS0 = Max[7, Min[11, Round[medNw0 / 6.5]]];

    wh0 =
        Association @ Table[
            v -> Block[{a = Lookup[nodeAttrs, v, <||>], lbl = getLabel[v]},
                wAttr = Quiet @ Check[ToExpression[ToString[Lookup[a, "width", ""]]], $Failed];
                hAttr = Quiet @ Check[ToExpression[ToString[Lookup[a, "height", ""]]], $Failed];
                If[
                    NumericQ[wAttr] && NumericQ[hAttr] && wAttr > 0 && hAttr > 0,
                    72. {wAttr, hAttr},
                    labelMinPlotWH[lbl, dotFs[v, vtxFS0]]
                ]
            ],
            {v, vNames}
        ];

    e2 = {#[["src"]], #[["tgt"]]} & /@ edges;
    sug = sugiyamaDotLayout[vNames, e2, wh0, orient];
    sh0 = Association[# -> nodeShape[nodeAttrs, <||>, vNames, #] & /@ vNames];

    Join[sug, <|"shapes" -> sh0|>]
]


scaleLayoutData[gv_, layoutScale_] := Block[{
    scaled = gv
},
    If[
        layoutScale === 1. || ! AssociationQ[scaled] || ! KeyExistsQ[scaled, "coords"],
        Return[scaled]
    ];

    Join[
        scaled,
        Join[
            <|
                "coords" -> AssociationMap[Function[v, If[VectorQ[v], layoutScale v, v]], scaled["coords"]]
            |>,
            If[
                KeyExistsQ[scaled, "sizes"],
                <|
                    "sizes" -> AssociationMap[Function[v, If[VectorQ[v], layoutScale v, v]], scaled["sizes"]]
                |>,
                <||>
            ],
            If[
                KeyExistsQ[scaled, "graphSize"],
                <|"graphSize" -> layoutScale scaled["graphSize"]|>,
                <||>
            ]
        ]
    ]
]


resolveLayoutData[
    path_String,
    nodeAttrs_Association,
    vNames_List,
    edges_List,
    orient_,
    getLabel_,
    dotFs_,
    opts : OptionsPattern[ImportDOTString]
] := Block[{
    gv, layoutScale
},
    gv =
        If[
            TrueQ[OptionValue["UseGraphvizLayout"]],
            graphvizPlainNodeXY[path, opts],
            fallbackLayoutData[nodeAttrs, vNames, edges, orient, getLabel, dotFs]
        ];

    layoutScale =
        If[
            OptionValue["LayoutScale"] === Automatic,
            With[{n = Length[vNames]}, Min[1.45, Max[1., 1. + Max[0., n - 14] * 0.018]]],
            OptionValue["LayoutScale"]
        ];

    If[! NumericQ[layoutScale] || layoutScale <= 0, layoutScale = 1.];
    scaleLayoutData[gv, layoutScale]
]


layoutDisplayData[vNames_List, edges_List, gv_] := Block[{
    pos, vcRules, hasPlainCoords,
    plainSizes, plainShapes,
    graphAspect, graphAspectUse, imgSize,
    vCnt, eCnt, medNw, vtxFS, edgeFS, ahSz,
    gw, gh, wPix
},
    pos = If[AssociationQ[gv] && KeyExistsQ[gv, "coords"], gv["coords"], <||>];
    vcRules = Lookup[pos, #, Missing["m"]] & /@ vNames;
    hasPlainCoords = FreeQ[vcRules, _Missing];
    plainSizes = If[AssociationQ[gv] && KeyExistsQ[gv, "sizes"], gv["sizes"], <||>];
    plainShapes = If[AssociationQ[gv] && KeyExistsQ[gv, "shapes"], gv["shapes"], <||>];

    graphAspect =
        If[
            AssociationQ[gv] && KeyExistsQ[gv, "graphSize"],
            Quiet @ Check[Divide @@ Reverse[gv["graphSize"]], Automatic],
            Automatic
        ];

    graphAspectUse =
        If[NumericQ[graphAspect] && 0.28 <= graphAspect <= 3.6, graphAspect, Automatic];

    imgSize =
        If[
            ! TrueQ[hasPlainCoords] || ! AssociationQ[gv] || ! KeyExistsQ[gv, "graphSize"],
            Which[
                ! NumericQ[graphAspect], {Max[520, 42 * Length[vNames]], Automatic},
                graphAspect < 0.32, {Min[1320, Max[560, 95 * Length[vNames]]], 420},
                graphAspect > 3., {520, Min[920, Max[280, 130 + 72 * Length[vNames]]]},
                True, {460, Max[220, Round[460 * graphAspect]]}
            ],
            Block[{gw, gh, wPix},
                {gw, gh} = gv["graphSize"];
                wPix = Min[2000, Max[520, Round[72. gw / 0.42]]];
                {wPix, Automatic}
            ]
        ];

    vCnt = Length[vNames];
    eCnt = Length[edges];
    medNw = If[plainSizes =!= <||> && Length[plainSizes] > 0, Median[First /@ Values[plainSizes]], 48.];
    vtxFS = Clip[Round[medNw / 5.9], {7.5, 13.}];
    edgeFS = Clip[Round[medNw / 7.2] - Min[1, Floor[eCnt / 45]], {7.5, 12.}];
    ahSz = Max[0.01, Min[0.02, 0.18 / Sqrt[N[vCnt]]]];

    <|
        "pos" -> pos,
        "vcRules" -> vcRules,
        "hasPlainCoords" -> hasPlainCoords,
        "plainSizes" -> plainSizes,
        "plainShapes" -> plainShapes,
        "graphAspectUse" -> graphAspectUse,
        "imgSize" -> imgSize,
        "vCnt" -> vCnt,
        "eCnt" -> eCnt,
        "vtxFS" -> vtxFS,
        "edgeFS" -> edgeFS,
        "ahSz" -> ahSz
    |>
]


vertexSizingData[
    vNames_List,
    gv_,
    plainSizes_,
    pos_Association,
    getShape_,
    getLabel_,
    getVertexFontSize_
] := Block[{
    vertexSizeOpt, vertexSizeAssoc, nodeRectAssoc,
    sz, oval
},
    vertexSizeOpt =
        Block[{sz = plainSizes},
            If[sz === <||> && AssociationQ[gv] && KeyExistsQ[gv, "sizes"], sz = gv["sizes"]];
            If[
                sz =!= <||>,
                With[
                    {
                        oval =
                            Function[{wh, nm},
                                Block[{sh = getShape[nm], w = wh[[1]], h = wh[[2]], r, wh1, whS, lb},
                                    r = If[h == 0. || ! NumericQ[h], 1., w / h];
                                    wh1 =
                                        Which[
                                            MemberQ[{"box", "rectangle", "rect", "square", "msquare"}, sh], wh,
                                            StringContainsQ[sh, "record"], wh,
                                            MemberQ[{"invtriangle", "triangle", "diamond", "invdiamond", "hexagon", "parallelogram", "pentagon", "cds", "cylinder", "Msquare", "Mcircle", "Mdiamond"}, sh] ||
                                                StringContainsQ[sh, "trap"] || StringContainsQ[sh, "inv"], wh,
                                            r < 1.18, {Max[w, 1.32 h], h},
                                            True, wh
                                        ];
                                    whS =
                                        Which[
                                            MemberQ[{"circle", "point", "doublecircle"}, sh] && Max[wh1] < 16., 0.78 wh1,
                                            MemberQ[{"triangle", "invtriangle"}, sh], 0.68 wh1,
                                            True, wh1
                                        ];
                                    lb = labelMinPlotWH[getLabel[nm], getVertexFontSize[nm]];
                                    {Max[whS[[1]], lb[[1]]], Max[whS[[2]], lb[[2]]]}
                                ]
                            ]
                    },
                    (#[[1]] -> oval[#[[2]], #[[1]]]) & /@ Normal[sz]
                ],
                1.35
            ]
        ];

    vertexSizeAssoc =
        If[
            ListQ[vertexSizeOpt],
            Association[vertexSizeOpt],
            Association @ Table[v -> Lookup[plainSizes, v, {42., 24.}], {v, vNames}]
        ];

    nodeRectAssoc =
        Association @ Table[
            v -> rectFromCenterWH[Lookup[pos, v, {0., 0.}], Lookup[vertexSizeAssoc, v, {42., 24.}], 4.],
            {v, vNames}
        ];

    <|
        "sizeOpt" -> vertexSizeOpt,
        "sizeAssoc" -> vertexSizeAssoc,
        "nodeRects" -> nodeRectAssoc
    |>
]


graphvizJsonTextPrims[ops_List, toColor_] := Block[{
    fs = 10., face = "Helvetica", col = GrayLevel[0.],
    prims = {}, pt, txt, align
},
    Do[
        Switch[
            Lookup[op, "op", ""],
            "F",
                fs = N @ Lookup[op, "size", 10.];
                face = Lookup[op, "face", "Helvetica"],
            "c",
                col = toColor[Lookup[op, "color", "black"]],
            "T",
                pt = Lookup[op, "pt", {0., 0.}];
                txt = Lookup[op, "text", ""];
                align = Lookup[op, "align", "c"];
                AppendTo[
                    prims,
                    Text[
                        Style[
                            txt,
                            FontSize -> fs,
                            FontFamily -> face,
                            FontWeight -> Plain,
                            FontColor -> col,
                            TextAlignment -> Which[align === "l", Left, align === "r", Right, True, Center]
                        ],
                        pt
                    ]
                ],
            _,
                Null
        ],
        {op, ops}
    ];
    prims
]


graphvizJsonNodePrims[obj_Association, toColor_] := Block[{
    stroke = GrayLevel[0.], fill = None,
    prims = {}, rect, pts
},
    Do[
        Switch[
            Lookup[op, "op", ""],
            "c",
                stroke = toColor[Lookup[op, "color", "black"]],
            "C",
                fill = toColor[Lookup[op, "color", "white"]],
            "P",
                pts = Lookup[op, "points", {}];
                AppendTo[prims, {FaceForm[fill], EdgeForm[{stroke, AbsoluteThickness[0.8]}], Polygon[pts]}],
            "E",
                rect = Lookup[op, "rect", {0., 0., 1., 1.}];
                AppendTo[prims, {FaceForm[fill], EdgeForm[{stroke, AbsoluteThickness[0.8]}], Disk[rect[[1 ;; 2]], rect[[3 ;; 4]]]}],
            _,
                Null
        ],
        {op, Lookup[obj, "_draw_", {}]}
    ];
    Join[prims, graphvizJsonTextPrims[Lookup[obj, "_ldraw_", {}], toColor]]
]


graphvizJsonArrowPoly[ops_List, toColor_] := Block[{
    stroke = GrayLevel[0.], fill = GrayLevel[0.],
    prims = {}, pts
},
    Do[
        Switch[
            Lookup[op, "op", ""],
            "c",
                stroke = toColor[Lookup[op, "color", "black"]],
            "C",
                fill = toColor[Lookup[op, "color", "black"]],
            "P",
                pts = Lookup[op, "points", {}];
                AppendTo[prims, {FaceForm[fill], EdgeForm[{stroke, AbsoluteThickness[0.8]}], Polygon[pts]}],
            _,
                Null
        ],
        {op, ops}
    ];
    prims
]


buildGraphvizJsonData[
    path_String,
    eStyleAssoc_Association,
    toColor_,
    opts : OptionsPattern[ImportDOTString]
] := Block[{
    dot, proc, data, objs, idToName,
    nodeTexts, edgePrims, tail, head, pts
},
    If[! TrueQ[OptionValue["UseGraphvizLayout"]], Return[<|"nodeTexts" -> <||>, "edges" -> <||>|>]];

    dot = resolveDotExecutable[opts];
    If[dot === $Failed, Return[<|"nodeTexts" -> <||>, "edges" -> <||>|>]];

    proc = Quiet @ RunProcess[{dot, "-Tjson", ExpandFileName[path]}];
    If[! AssociationQ[proc] || proc["ExitCode"] =!= 0, Return[<|"nodeTexts" -> <||>, "edges" -> <||>|>]];

    data = Quiet @ Check[ImportString[proc["StandardOutput"], "RawJSON"], $Failed];
    If[! AssociationQ[data], Return[<|"nodeTexts" -> <||>, "edges" -> <||>|>]];

    objs = Lookup[data, "objects", {}];
    idToName =
        Association @ Table[
            Lookup[obj, "_gvid", Missing["g"]] -> Lookup[obj, "name", ""],
            {obj, objs}
        ];

    nodeTexts =
        Association @ Table[
            With[{nm = Lookup[obj, "name", ""]},
                nm -> graphvizJsonTextPrims[Lookup[obj, "_ldraw_", {}], toColor]
            ],
            {obj, objs}
        ];

    edgePrims =
        Association @ Table[
            With[
                {
                    tail = Lookup[idToName, Lookup[e, "tail", Missing["t"]], ""],
                    head = Lookup[idToName, Lookup[e, "head", Missing["h"]], ""],
                    pts = Lookup[
                        FirstCase[Lookup[e, "_draw_", {}], op_ /; Lookup[op, "op", ""] == "b" :> op, <||>],
                        "points",
                        {}
                    ]
                },
                DirectedEdge[tail, head] ->
                    Join[
                        {
                            Lookup[
                                eStyleAssoc,
                                DirectedEdge[tail, head],
                                Directive[GrayLevel[0.22], CapForm["Round"], JoinForm["Round"], AbsoluteThickness[1.1]]
                            ]
                        },
                        (BezierCurve[#] & /@ jsonBezierSegs[pts]),
                        graphvizJsonArrowPoly[Lookup[e, "_hdraw_", {}], toColor],
                        graphvizJsonTextPrims[Lookup[e, "_ldraw_", {}], toColor],
                        graphvizJsonTextPrims[Lookup[e, "_tldraw_", {}], toColor]
                    ]
            ],
            {e, Lookup[data, "edges", {}]}
        ];

    <|"nodeTexts" -> nodeTexts, "edges" -> edgePrims|>
]


edgeRenderData[
    edges_List,
    pos_Association,
    vertexSizeAssoc_Association,
    nodeRectAssoc_Association,
    getShape_,
    toColor_,
    ahSz_,
    edgeFS_,
    hasPlainCoords_,
    useJsonDraw_,
    gvJsonEdges_Association
] := Block[{
    eStyles, eStyleAssoc, edgeMetaAssoc,
    plainEdgeGeomAssoc, plainEdgeSF, edgeLbls
},
    eStyles =
        Table[
            Block[{style, color, pen, pw, rgb},
                style = ToLowerCase[StringTrim @ ToString[Lookup[e["attrs"], "style", "solid"], InputForm]];
                color = Lookup[e["attrs"], "color", "black"];
                pen = Lookup[e["attrs"], "penwidth", "1"];
                pw = Quiet @ Check[Interpreter["Number"][ToString[pen]], 1.];
                If[! NumericQ[pw], pw = 1.];
                rgb = toColor[color];

                DirectedEdge[e["src"], e["tgt"]] ->
                    Directive[
                        rgb,
                        CapForm["Round"],
                        JoinForm["Round"],
                        AbsoluteThickness[Max[1.15, 0.95 pw]],
                        Arrowheads[{{ahSz, 1.}}],
                        If[style === "dashed", Dashing[{6, 4}], If[style === "dotted", Dashing[{1, 5}], Sequence @@ {}]]
                    ]
            ],
            {e, edges}
        ];

    eStyleAssoc = Association[eStyles];

    edgeMetaAssoc =
        Association @ Table[
            With[{e = edges[[k]], a = edges[[k]]["attrs"]},
                DirectedEdge[e["src"], e["tgt"]] ->
                    Block[{main, tail, fs, lfs, efcol, mfc, tfc, cc},
                        main = Lookup[a, "label", ""];
                        tail = Lookup[a, "taillabel", ""];

                        fs = Quiet @ Check[ToExpression[ToString[Lookup[a, "fontsize", ""]]], $Failed];
                        If[! NumericQ[fs] || fs <= 0, fs = edgeFS];
                        fs = Clip[1.18 fs, {8.5, 13.5}];

                        lfs = Quiet @ Check[ToExpression[ToString[Lookup[a, "labelfontsize", ""]]], $Failed];
                        If[! NumericQ[lfs] || lfs <= 0, lfs = 7.];
                        lfs = Clip[1.15 lfs, {8., 11.5}];

                        efcol = Lookup[a, "fontcolor", ""];
                        mfc =
                            If[
                                ! StringQ[efcol] || efcol === "",
                                GrayLevel[0.],
                                cc = toColor[efcol];
                                If[Head[cc] === RGBColor, Darker[cc, 0.18], GrayLevel[0.08]]
                            ];
                        tfc =
                            If[
                                ! StringQ[efcol] || efcol === "",
                                GrayLevel[0.22],
                                cc = toColor[efcol];
                                If[Head[cc] === RGBColor, Darker[cc, 0.1], GrayLevel[0.22]]
                            ];

                        <|"main" -> main, "tail" -> tail, "fs" -> fs, "lfs" -> lfs, "mfc" -> mfc, "tfc" -> tfc|>
                    ]
            ],
            {k, Length[edges]}
        ];

    plainEdgeGeomAssoc =
        Block[{
            occupiedRects = Values[nodeRectAssoc], assoc = <||>,
            ed, p1, p2, d, len, nrm, curveMag,
            tailKey, headKey, tailSize, headSize, tailShape, headShape,
            c1, c2, curvePts, meta0, m, t, fs, lfs, mwh, twh, mpos, tpos,
            curvePt, curveTan, mkCandidates
        },
            Do[
                ed = DirectedEdge[e["src"], e["tgt"]];
                p1 = Lookup[pos, e["src"], {0., 0.}];
                p2 = Lookup[pos, e["tgt"], {0., 0.}];
                d = p2 - p1;
                len = Norm[d];

                If[
                    len <= 10.^-9,
                    assoc[ed] = <|"curve" -> {p1, p1, p2, p2}, "mainPos" -> None, "tailPos" -> None|>;
                    Continue[]
                ];

                nrm = {-d[[2]], d[[1]]} / len;
                curveMag = If[Abs[d[[1]]] > 6., Min[24., 0.16 len], 0.];

                tailKey = e["src"];
                headKey = e["tgt"];
                tailSize = Lookup[vertexSizeAssoc, tailKey, {42., 24.}];
                headSize = Lookup[vertexSizeAssoc, headKey, {42., 24.}];
                tailShape = getShape[tailKey];
                headShape = getShape[headKey];

                c1 = If[curveMag > 0., p1 + 0.34 d - 0.82 Sign[d[[1]]] curveMag nrm, p1 + d / 3.];
                c2 = If[curveMag > 0., p2 - 0.34 d - 0.82 Sign[d[[1]]] curveMag nrm, p1 + 2. d / 3.];

                curvePts = clipBezierChainToNodeShapes[{p1, c1, c2, p2}, p1, tailSize, tailShape, p2, headSize, headShape];
                If[Length[curvePts] < 4, curvePts = {p1, c1, c2, p2}];

                mkCandidates =
                    Function[
                        {t0, wh},
                        Block[{base, delta, offs, tanShift},
                            base = Max[4.5, 0.28 wh[[2]] + 1.5];
                            delta = Min[4.5, 0.06 len];
                            tanShift = Min[6., 0.09 len];
                            offs = {base, base + delta, base + 2. delta};
                            DeleteDuplicates @ Flatten[
                                Table[
                                    curvePt = bezierPoint[curvePts, tt];
                                    curveTan = bezierTangent[curvePts, tt];
                                    If[Norm[curveTan] <= 10.^-9, curveTan = d];
                                    curveTan = curveTan / Max[10.^-9, Norm[curveTan]];
                                    curvePt + s off {-curveTan[[2]], curveTan[[1]]} + u tanShift curveTan,
                                    {tt, DeleteDuplicates @ Clip[{t0, t0 - 0.05, t0 + 0.05, t0 - 0.1, t0 + 0.1}, {0.1, 0.9}]},
                                    {u, {0, -1, 1}},
                                    {s, {1, -1}},
                                    {off, offs}
                                ],
                                2
                            ]
                        ]
                    ];

                meta0 = Lookup[edgeMetaAssoc, ed, <||>];
                m = Lookup[meta0, "main", ""];
                fs = Lookup[meta0, "fs", 8.];
                mwh = textLabelWH[m, fs];
                mpos = None;
                If[
                    m =!= "" && Max[mwh] > 0.,
                    mpos = chooseFreeLabelPoint[mkCandidates[0.56, mwh], mwh, occupiedRects];
                    AppendTo[occupiedRects, rectFromCenterWH[mpos, mwh, 2.]]
                ];

                t = Lookup[meta0, "tail", ""];
                lfs = Lookup[meta0, "lfs", 7.];
                twh = textLabelWH[t, lfs];
                tpos = None;
                If[
                    t =!= "" && Max[twh] > 0.,
                    tpos = chooseFreeLabelPoint[mkCandidates[0.24, twh], twh, occupiedRects];
                    AppendTo[occupiedRects, rectFromCenterWH[tpos, twh, 2.]]
                ];

                assoc[ed] = <|"curve" -> curvePts, "mainPos" -> mpos, "tailPos" -> tpos|>,
                {e, edges}
            ];
            assoc
        ];

    plainEdgeSF =
        With[
            {meta = edgeMetaAssoc, est = eStyleAssoc, ah = ahSz, geo = plainEdgeGeomAssoc},
            Function[
                {pts, ed},
                Block[{m, t, fs, lfs, st, mfc, tfc, geom, curvePts, mpos, tpos},
                    If[useJsonDraw && KeyExistsQ[gvJsonEdges, ed], Return[gvJsonEdges[ed]]];
                    If[Length[pts] < 2, Return[{}]];

                    geom = Lookup[geo, ed, <||>];
                    curvePts = Lookup[geom, "curve", {pts[[1]], pts[[1]], pts[[-1]], pts[[-1]]}];
                    mpos = Lookup[geom, "mainPos", None];
                    tpos = Lookup[geom, "tailPos", None];

                    m = Lookup[Lookup[meta, ed, <||>], "main", ""];
                    t = Lookup[Lookup[meta, ed, <||>], "tail", ""];
                    fs = Lookup[Lookup[meta, ed, <||>], "fs", 8.];
                    lfs = Lookup[Lookup[meta, ed, <||>], "lfs", 7.];
                    mfc = Lookup[Lookup[meta, ed, <||>], "mfc", GrayLevel[0.]];
                    tfc = Lookup[Lookup[meta, ed, <||>], "tfc", GrayLevel[0.4]];
                    st =
                        Lookup[
                            est,
                            ed,
                            Directive[
                                GrayLevel[0.22],
                                CapForm["Round"],
                                JoinForm["Round"],
                                AbsoluteThickness[1.1],
                                Arrowheads[{{ah, 1.}}]
                            ]
                        ];

                    Join[
                        {st, Arrow[BezierCurve[curvePts]]},
                        If[
                            ListQ[mpos],
                            {
                                Text[
                                    Style[m, FontSize -> fs, FontFamily -> "Helvetica", FontWeight -> Plain, FontColor -> mfc, Background -> White, TextAlignment -> Center],
                                    mpos
                                ]
                            },
                            {}
                        ],
                        If[
                            ListQ[tpos],
                            {
                                Text[
                                    Style[t, FontSize -> lfs, FontFamily -> "Helvetica", FontWeight -> Plain, FontColor -> tfc, Background -> White, TextAlignment -> Center],
                                    tpos
                                ]
                            },
                            {}
                        ]
                    ]
                ]
            ]
        ];

    edgeLbls =
        If[
            ! hasPlainCoords,
            DeleteCases[
                Table[
                    Block[{lbl, color, st, txt, efs, fs},
                        lbl = Lookup[e["attrs"], "label", ""];
                        fs = Quiet @ Check[ToExpression[ToString[Lookup[e["attrs"], "fontsize", ""]]], $Failed];
                        efs = If[NumericQ[fs] && fs > 0, Clip[fs, {6., 12.}], edgeFS];
                        color = Lookup[e["attrs"], "fontcolor", Lookup[e["attrs"], "color", "black"]];
                        st = If[StringMatchQ[color, "#" ~~ __], toColor[color], GrayLevel[0.22]];
                        txt =
                            Style[
                                lbl,
                                FontSize -> efs,
                                FontFamily -> "Helvetica",
                                FontWeight -> Plain,
                                FontColor -> st,
                                Background -> White,
                                TextAlignment -> Center
                            ];
                        If[StringQ[lbl] && lbl =!= "", DirectedEdge[e["src"], e["tgt"]] -> Placed[txt, 0.38], Nothing]
                    ],
                    {e, edges}
                ],
                Nothing | Null
            ],
            {}
        ];

    <|
        "styles" -> eStyles,
        "labels" -> edgeLbls,
        "shapeFunction" -> plainEdgeSF,
        "styleAssoc" -> eStyleAssoc
    |>
]

importWolframGraph[path_String, opts : OptionsPattern[ImportDOTString]] := Block[{
    raw, parsed, nodes, edges, nodeAttrs,
    vNames, edgeObjs,
    orient, edgeShapeChoice, layoutEdge,
    pass, plbl, useJsonDraw,
    toColor, getFillColor, getStrokeColor, getLabel, dotFs, getVertexFontSize, getShape,
    gv, displayData, vertexData, jsonData, edgeData
},
    raw = stripComments @ Import[path, "Text"];
    parsed = parseDotGraphData[raw];
    nodes = parsed["nodes"];
    edges = parsed["edges"];
    nodeAttrs = parsed["nodeAttrs"];

    vNames = #["name"] & /@ nodes;
    edgeObjs = Union[DirectedEdge[#["src"], #["tgt"]] & /@ edges];
    orient = rankDirOrientation[raw];
    pass = FilterRules[dropImportDOTMethodOpts @ Flatten[{opts}], Options[Graph]];
    plbl = resolvePlotLabel[path, OptionValue[PlotLabel]];

    toColor = colorToRGB[hexToRGB, #] &;
    getFillColor = nodeFillColor[nodeAttrs, toColor, vNames, #] &;
    getStrokeColor = nodeStrokeColor[nodeAttrs, toColor, vNames, #] &;
    getLabel = nodeLabel[nodeAttrs, vNames, #] &;
    dotFs = nodeDotFontSize[nodeAttrs, vNames, #1, #2] &;

    gv = resolveLayoutData[path, nodeAttrs, vNames, edges, orient, getLabel, dotFs, opts];
    displayData = layoutDisplayData[vNames, edges, gv];

    If[
        TrueQ[OptionValue["UseGraphvizLayout"]] && ! displayData["hasPlainCoords"],
        Message[ImportDOTString::gvplain]
    ];

    getVertexFontSize = dotFs[#, displayData["vtxFS"]] &;
    getShape = nodeShape[nodeAttrs, displayData["plainShapes"], vNames, #] &;

    vertexData =
        vertexSizingData[
            vNames,
            gv,
            displayData["plainSizes"],
            displayData["pos"],
            getShape,
            getLabel,
            getVertexFontSize
        ];

    jsonData = buildGraphvizJsonData[path, <||>, toColor, opts];
    useJsonDraw = False;

    edgeData =
        edgeRenderData[
            edges,
            displayData["pos"],
            vertexData["sizeAssoc"],
            vertexData["nodeRects"],
            getShape,
            toColor,
            displayData["ahSz"],
            displayData["edgeFS"],
            displayData["hasPlainCoords"],
            useJsonDraw,
            jsonData["edges"]
        ];

    edgeShapeChoice =
        If[displayData["eCnt"] > 32 || displayData["vCnt"] > 26, Automatic, "CurvedArc"];

    layoutEdge =
        If[
            displayData["hasPlainCoords"],
            Flatten @ {
                {
                    VertexCoordinates -> displayData["vcRules"],
                    EdgeShapeFunction -> edgeData["shapeFunction"]
                },
                If[NumericQ[displayData["graphAspectUse"]], {AspectRatio -> displayData["graphAspectUse"]}, {}]
            },
            {
                GraphLayout -> {
                    "VertexLayout" -> {"LayeredDigraphEmbedding", "Orientation" -> orient},
                    "EdgeLayout" -> "DividedEdgeBundling"
                },
                EdgeShapeFunction -> edgeShapeChoice
            }
        ];


    Graph[
        vNames,
        edgeObjs,
        Sequence @@ pass,
        Sequence @@ layoutEdge,
        VertexLabels ->
            If[
                useJsonDraw,
                None,
                Table[
                    With[{lab = getLabel[v]},
                        If[
                            StringTrim[lab] === "",
                            v -> None,
                            v -> Placed[
                                Style[
                                    lab,
                                    FontFamily -> "Helvetica",
                                    FontSize -> Max[9., Min[17., 1.48 getVertexFontSize[v]]],
                                    FontWeight -> Plain,
                                    FontColor -> GrayLevel[0.],
                                    TextAlignment -> Center
                                ],
                                Center
                            ]
                        ]
                    ],
                    {v, vNames}
                ]
            ],
        VertexShapeFunction ->
            Function[
                {coord, v, size},
                Block[{sh, bg, sc, rx, ry, tw},
                    sh = getShape[v];
                    bg = getFillColor[v];
                    sc = getStrokeColor[v];
                    rx = size[[1]] / 2;
                    ry = size[[2]] / 2;
                    tw = Clip[0.45 + 0.08 Min[rx, ry] / 20., {0.38, 0.85}];

                    Which[
                        MemberQ[{"box", "rectangle", "rect", "square", "msquare"}, sh],
                            {FaceForm[bg], EdgeForm[{sc, AbsoluteThickness[tw]}], Polygon[{coord + {-rx, -ry}, coord + {rx, -ry}, coord + {rx, ry}, coord + {-rx, ry}}]},
                        StringContainsQ[sh, "record"],
                            {FaceForm[bg], EdgeForm[{sc, AbsoluteThickness[tw]}], Polygon[{coord + {-rx, -ry}, coord + {rx, -ry}, coord + {rx, ry}, coord + {-rx, ry}}]},
                        sh === "invtriangle",
                            {FaceForm[bg], EdgeForm[{sc, AbsoluteThickness[tw]}], Polygon[{coord + {0, -0.84 ry}, coord + {-0.82 rx, 0.72 ry}, coord + {0.82 rx, 0.72 ry}}]},
                        sh === "triangle",
                            {FaceForm[bg], EdgeForm[{sc, AbsoluteThickness[tw]}], Polygon[{coord + {0, 0.84 ry}, coord + {-0.82 rx, -0.72 ry}, coord + {0.82 rx, -0.72 ry}}]},
                        StringContainsQ[sh, "diamond"],
                            {FaceForm[bg], EdgeForm[{sc, AbsoluteThickness[tw]}], Polygon[{coord + {0, ry}, coord + {rx, 0}, coord + {0, -ry}, coord + {-rx, 0}}]},
                        StringContainsQ[sh, "hexagon"],
                            {FaceForm[bg], EdgeForm[{sc, AbsoluteThickness[tw]}], Polygon[Table[coord + {rx Cos[Pi / 6 + k Pi / 3], ry Sin[Pi / 6 + k Pi / 3]}, {k, 0, 5}]]},
                        StringContainsQ[sh, "pentagon"],
                            {FaceForm[bg], EdgeForm[{sc, AbsoluteThickness[tw]}], Polygon[Table[coord + {rx Cos[Pi / 2 + 2 k Pi / 5], ry Sin[Pi / 2 + 2 k Pi / 5]}, {k, 0, 4}]]},
                        MemberQ[{"circle", "doublecircle", "point", "ellipse", "oval", "mcircle"}, sh],
                            {FaceForm[bg], EdgeForm[{sc, AbsoluteThickness[tw]}], If[MemberQ[{"circle", "doublecircle", "point", "mcircle"}, sh], Disk[coord, Min[rx, ry]], Disk[coord, {rx, ry}]]},
                        True,
                            {FaceForm[bg], EdgeForm[{sc, AbsoluteThickness[tw]}], Disk[coord, {rx, ry}]}
                    ]
                ]
            ],
        EdgeStyle -> edgeData["styles"],
        EdgeLabels -> If[displayData["hasPlainCoords"] || useJsonDraw, None, edgeData["labels"]],
        VertexSize -> vertexData["sizeOpt"],
        Background -> White,
        Epilog -> If[useJsonDraw, Flatten[Values[jsonData["nodeTexts"]]], {}],
        PlotRangePadding -> Scaled[0.04],
        ImagePadding -> With[{p = Min[130, 32 + 2 Length[vNames] + Floor[displayData["eCnt"] / 4]]}, {{p, p + 8}, {p - 4, p - 4}}],
        ImageSize -> displayData["imgSize"],
        PlotLabel -> If[plbl === None, None, Style[plbl, FontFamily -> "Helvetica", FontSize -> 13, FontWeight -> "SemiBold", FontColor -> GrayLevel[0.2]]],
        PerformanceGoal -> "Quality"
    ]
    ]

importGraphBuiltIn[path_String, opts : OptionsPattern[ImportDOTString]] := Block[{
    pass
},
    pass = FilterRules[dropImportDOTMethodOpts @ Flatten[{opts}], Options[Graph]];
    Quiet @ Check[Import[path, "DOT", Sequence @@ pass], $Failed]
]

importDOTFile[file_String, opts : OptionsPattern[ImportDOTString]] := Block[{
    path,
    method,
    png,
    g0
},
    path = Quiet @ Check[AbsoluteFileName[file], file];
    If[! FileExistsQ[path], Return[$Failed]];

    method = OptionValue[ImportDOTString, {opts}, Method];

    If[
        method === "Graphviz" || method === "GraphvizRaster" || method === "DotPNG",
        png = importGraphvizPNG[path, opts];
        If[MatchQ[png, _Framed], Return[png]];
        Message[ImportDOTString::dotfail, path, "dot or PNG import"];
        Return[$Failed]
    ];

    Which[
        method === "GraphvizGraphics",
            importGraphvizGraphics[path, opts],
        method === "Parser" || method === "Wolfram",
            importWolframGraph[path, opts],
        method === Automatic || method === "Import",
            g0 = importWolframGraph[path, opts];
            If[GraphQ[g0], Return[g0]];
            g0 = importGraphBuiltIn[path, opts];
            If[
                GraphQ[g0],
                g0,
                Message[ImportDOTString::impfail];
                importWolframGraph[path, opts]
            ],
        True,
            Message[ImportDOTString::badmeth, method];
            $Failed
    ]
]

ImportDOTString[dot_String, opts : OptionsPattern[]] := Block[{
    path = FileNameJoin[
        {
            $TemporaryDirectory,
            "thvm-import-dot-" <> IntegerString[Hash[{AbsoluteTime[], RandomInteger[10^9]}], 16] <> ".dot"
        }
    ],
    result = $Failed
},
    Internal`WithLocalSettings[
        Export[path, dot, "String"],
        result = importDOTFile[path, opts],
        Quiet @ Check[DeleteFile[path], Null]
    ];
    result
]

End[];

EndPackage[];
