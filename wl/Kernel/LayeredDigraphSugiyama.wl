(* Layered digraph layout — Sugiyama-style pipeline (Graphviz `dot` core ideas:
   Gansner–Koutsofios–North–Vo, IEEE TSE 1993). Longest-path ranks, median
   crossing sweeps, greedy x with separation. Coordinates in points (y-up).

   Not a full `dot` clone: no same-rank, ports, or spline bundles. *)

ClearAll[
    sugiyamaDotLayout,
    breakCyclesDAG,
    longestPathRanks,
    buildLayers,
    posInLayer,
    medianSweepLayers,
    placeLayersXY,
    applyRankDir,
    graphBBoxInches
];

breakCyclesDAG[v_List, e_List] := Block[{
    g, fe, g2, c
},
    g = Graph[v, DirectedEdge @@@ e];
    If[AcyclicGraphQ[g], Return[{g, e}]];

    fe = Quiet @ Check[FindMinimumFeedbackArcSet[g], {}];
    If[
        ListQ[fe] && fe =!= {} && Length[fe] < Length[EdgeList[g]],
        Return[
            {
                EdgeDelete[g, fe],
                Cases[e, {a_, b_} /; ! MemberQ[fe, DirectedEdge[a, b]]]
            }
        ]
    ];

    g2 = g;
    While[
        ! AcyclicGraphQ[g2],
        c = Quiet @ Check[FindCycle[g2, {2, Infinity}], {}];
        If[c === {} || c[[1]] === {}, Break[]];
        g2 = EdgeDelete[g2, c[[1, 1]]]
    ];

    {g2, EdgeList[g2] /. DirectedEdge[a_, b_] :> {a, b}}
]

longestPathRanks[g_Graph] := Block[{
    top, r, preds, vert, i
},
    top = TopologicalSort[g];
    r = <||>;
    Do[r[v] = 0, {v, top}];
    Do[
        vert = top[[i]];
        preds = Cases[EdgeList[g], DirectedEdge[p_, vert] :> p];
        r[vert] = If[preds === {}, 0, Max[Lookup[r, preds, 0] + 1]],
        {i, Length[top]}
    ];
    r
]

buildLayers[r_Association] := Block[{
    mx, buckets
},
    mx = Max[0, Append[Values[r], 0]];
    buckets = Table[{}, {mx + 1}];
    KeyValueMap[(buckets[[#2 + 1]] = Append[buckets[[#2 + 1]], #1]) &, r];
    buckets
]

posInLayer[v_, layer_List] := Block[{
    p
},
    p = FirstPosition[layer, w_ /; w === v, {0}];
    If[p[[1]] === 0, 0., N[p[[1]]]]
]

medianSweepLayers[layers_List, g_Graph, iters_Integer] := Block[{
    layout = layers, nLayers = Length[layers],
    pred, succ, sortedLayer, k, edgeList
},
    edgeList = EdgeList[g];
    Do[
        pred = <||>;
        Scan[
            Function[e, pred[e[[2]]] = Append[Lookup[pred, e[[2]], {}], e[[1]]]],
            edgeList
        ];
        For[
            k = 2,
            k <= nLayers,
            k++,
            sortedLayer = SortBy[
                layout[[k]],
                Function[v,
                    Block[{ps},
                        ps = Select[Lookup[pred, v, {}], MemberQ[layout[[k - 1]], #] &];
                        If[ps === {}, 0., Median[posInLayer[#, layout[[k - 1]]] & /@ ps]]
                    ]
                ]
            ];
            layout[[k]] = sortedLayer
        ];

        succ = <||>;
        Scan[
            Function[e, succ[e[[1]]] = Append[Lookup[succ, e[[1]], {}], e[[2]]]],
            edgeList
        ];
        For[
            k = nLayers - 1,
            k >= 1,
            k--,
            sortedLayer = SortBy[
                layout[[k]],
                Function[v,
                    Block[{ss},
                        ss = Select[Lookup[succ, v, {}], MemberQ[layout[[k + 1]], #] &];
                        If[ss === {}, 0., Median[posInLayer[#, layout[[k + 1]]] & /@ ss]]
                    ]
                ]
            ];
            layout[[k]] = sortedLayer
        ],
        {iters}
    ];
    layout
]

placeLayersXY[
    layers_List,
    wh_Association,
    nodeSep_ ? NumericQ,
    layerSep_ ? NumericQ
] := Block[{
    nLayers = Length[layers], out = <||>,
    xCur, v, wid, hid,
    maxRank, layerY, r, row, mx
},
    maxRank = Max[0, nLayers - 1];
    layerY[r0_Integer] := (maxRank - (r0 - 1)) * layerSep;

    Do[
        row = layers[[r]];
        xCur = 0.;
        Do[
            v = row[[i]];
            {wid, hid} = Lookup[wh, v, {40., 24.}];
            out[v] = {xCur + wid / 2, layerY[r]};
            xCur += wid + nodeSep,
            {i, Length[row]}
        ],
        {r, nLayers}
    ];

    If[
        out =!= <||>,
        mx = Mean[Values[out][[All, 1]]];
        out = Association[KeyValueMap[#1 -> (#2 - {mx, 0}) &, out]]
    ];

    out
]

applyRankDir[coord_Association, orient_] :=
    Association[
        KeyValueMap[
            Function[
                {k, c},
                k -> Switch[
                    orient,
                    Top, c,
                    Bottom, {c[[1]], -c[[2]]},
                    Left, {-c[[2]], c[[1]]},
                    Right, {c[[2]], -c[[1]]},
                    _, c
                ]
            ],
            coord
        ]
    ];

graphBBoxInches[coord_Association, wh_Association] := Block[{
    pts, gw, gh
},
    If[coord === <||>, Return[{6, 4}]];

    pts =
        Flatten[
            KeyValueMap[
                Block[{c = #2, wid, hid},
                    {wid, hid} = Lookup[wh, #1, {40., 24.}];
                    {
                        {c[[1]] - wid / 2, c[[2]] - hid / 2},
                        {c[[1]] + wid / 2, c[[2]] + hid / 2}
                    }
                ] &,
                coord
            ],
            1
        ];

    gw = (Max[pts[[All, 1]]] - Min[pts[[All, 1]]]) / 72.;
    gh = (Max[pts[[All, 2]]] - Min[pts[[All, 2]]]) / 72.;
    {Max[gw, 0.5], Max[gh, 0.5]}
]

sugiyamaDotLayout[
    vNames_List,
    edges_List,
    wh_Association,
    orient_
] := Block[{
    e2, g, r, layers, layers2,
    coord, coord2, gbox,
    nodeSep = 40., layerSep = 118., iters = 18
},
    e2 = Select[edges, MemberQ[vNames, #[[1]]] && MemberQ[vNames, #[[2]]] &];

    If[
        e2 === {},
        Block[{x = 0., out = <||>, wid, hid, v, i},
            Do[
                v = vNames[[i]];
                {wid, hid} = Lookup[wh, v, {40., 24.}];
                out[v] = {x + wid / 2, 0.};
                x += wid + nodeSep,
                {i, Length[vNames]}
            ];
            coord2 = applyRankDir[out, orient];
            Return[<|"coords" -> coord2, "sizes" -> wh, "graphSize" -> graphBBoxInches[coord2, wh]|>]
        ]
    ];

    {g, e2} = breakCyclesDAG[vNames, e2];
    r = longestPathRanks[g];
    layers = buildLayers[r];
    layers2 = medianSweepLayers[layers, g, iters];
    coord = placeLayersXY[layers2, wh, nodeSep, layerSep];
    coord2 = applyRankDir[coord, orient];
    gbox = graphBBoxInches[coord2, wh];
    <|"coords" -> coord2, "sizes" -> wh, "graphSize" -> gbox|>
]
