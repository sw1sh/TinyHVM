testFile = Replace[$InputFileName, "" -> "/Users/swish/src/TinyHVM/wl/Tests/heap_dot_match.wlt"];
repoWLRoot = DirectoryName[DirectoryName[testFile]];
Get[FileNameJoin[{repoWLRoot, "Kernel", "TinyHVM.wl"}]];

syntheticWalk = <|
    "Nodes" -> <|
        "h_root" -> <|"Tag" -> "App", "Ext" -> 0, "Val" -> 10, "Loc" -> 10, "DisplayLoc" -> 10|>,
        "h_fun" -> <|"Tag" -> "Lam", "Ext" -> 0, "Val" -> 20, "Loc" -> 20, "DisplayLoc" -> 20|>,
        "var21" -> <|"Tag" -> "Var", "Ext" -> 0, "Val" -> 21, "Loc" -> 21, "DisplayLoc" -> 21|>,
        "num_arg" -> <|"Tag" -> "Num", "Ext" -> 0, "Val" -> 7, "Loc" -> 0, "DisplayLoc" -> 0|>
    |>,
    "Edges" -> {
        <|"From" -> "h_fun", "To" -> "h_root", "Port" -> "fun", "FromSlot" -> 10|>,
        <|"From" -> "num_arg", "To" -> "h_root", "Port" -> "arg", "FromSlot" -> 11|>,
        <|"From" -> "h_fun", "To" -> "var21", "Port" -> "var", "FromSlot" -> 20|>
    },
    "Root" -> "h_root"
|>;

VerificationTest[
    Block[{data, dot},
        data = TinyHVM`Private`displayWalkData[syntheticWalk];
        dot = TINetGraphDOT[syntheticWalk];
        StringStartsQ[dot, "digraph TinyHVM {"] &&
            AllTrue[
                Keys[data["Nodes"]],
                StringContainsQ[dot, TinyHVM`Private`dotNodeId[#] <> " ["] &
            ] &&
            AllTrue[
                data["Edges"],
                StringContainsQ[
                    dot,
                    TinyHVM`Private`dotNodeId[#["From"]] <> " -> " <> TinyHVM`Private`dotNodeId[#["To"]] <> " ["
                ] &
            ]
    ],
    True,
    TestID -> "synthetic-walk-dot-covers-display-walk-data"
]

VerificationTest[
    Block[{data, graph, imported},
        data = TinyHVM`Private`displayWalkData[syntheticWalk];
        graph = TINetGraph[syntheticWalk];
        imported = TINetGraphImport[syntheticWalk, Method -> "Parser"];
        {
            Sort[VertexList[graph]],
            EdgeCount[graph],
            Sort[VertexList[imported]],
            EdgeCount[imported]
        }
    ],
    Block[{data = TinyHVM`Private`displayWalkData[syntheticWalk]},
        {
            Sort[data["Keys"]],
            Length[data["Edges"]],
            Sort[data["Keys"]],
            Length[data["Edges"]]
        }
    ],
    TestID -> "parser-import-matches-pure-wl-graph-structure"
]

VerificationTest[
    Block[{ok, term, walk},
        ok = TInit["cpu"];
        If[! TrueQ[ok], Return[False]];
        term = TNum[7];
        walk = TinyHVM`Private`heapWalk[TinyHVM`Private`rootTermOf[term]];
        TINetGraphDOT[term] === TINetGraphDOT[walk]
    ],
    True,
    SameTest -> SameQ,
    TestID -> "term-entrypoint-matches-pure-wl-walk-entrypoint"
]

VerificationTest[
    Block[{ok, term, walk},
        ok = TInit["cpu"];
        If[! TrueQ[ok], Return[False]];
        term = TNum[7];
        walk = TinyHVM`Private`heapWalk[TinyHVM`Private`rootTermOf[term]];
        TinyHVM`Private`dumpCDOTString[term] == TINetGraphDOT[walk]
    ],
    True,
    TestID -> "dump-c-dot-matches-pure-wl-dot-for-num-root"
]

VerificationTest[
    Block[{ok, tensor, walk},
        ok = TInit["cpu"];
        If[! TrueQ[ok], Return[False]];
        tensor = TCreate[{1.0}, {1}];
        walk = TinyHVM`Private`heapWalk[TinyHVM`Private`rootTermOf[tensor]];
        TinyHVM`Private`dumpCDOTString[tensor] == TINetGraphDOT[walk]
    ],
    True,
    TestID -> "dump-c-dot-matches-pure-wl-dot-for-tensor-root"
]

Quiet[TFree[]];
