(* Heap.wl — WL-side heap walker built on THeapRead / THeapReadRange / TTermVal.
   Get'd from TinyHVM.wl inside Begin["`Private`"].
   Rendering lives in Visualization.wl; this file only produces node/edge lists. *)

(* Synthetic atom key — TEN nodes don't own a heap slot, so we key by tensor id. *)
atomKey[n_Association] := Switch[n["Tag"],
    "Ten",  "t" <> ToString[n["Val"]],
    "Num",  If[Lookup[n, "Loc", 0] > 0,
               "num" <> ToString[n["Loc"]],
               "num" <> ToString[n["Val"]] <> "_" <> ToString[n["Ext"]]],
    "Era",  If[Lookup[n, "Loc", 0] > 0, "era" <> ToString[n["Loc"]], "era"],
    "Ref",  If[Lookup[n, "Loc", 0] > 0,
               "ref" <> ToString[n["Loc"]],
               "ref" <> ToString[n["Ext"]]],
    "Var",  "var" <> ToString[n["Val"]],
    "Any",  If[Lookup[n, "Loc", 0] > 0, "any" <> ToString[n["Loc"]], "any"],
    _,      "h" <> ToString[n["Loc"]]
];

(* Does this term own arity-sized heap slots starting at val?
   For UOPs with metadata slots (e.g. Kernel's NUM(root_uop)), only descend
   into data inputs — metadata is surfaced via the node label instead. *)
uopDataArity[uop_String] := Which[
    uop === "Kernel", 2,   (* left, right — skip NUM(root_uop) in slot 2 *)
    uop === "Exec",   1,   (* NUM(kid), deps, NUM(flags) — only deps is data *)
    True,             uopArity[uop]
];
termChildCount[n_Association] := Module[{tag = n["Tag"]},
    Which[
        tag === "Top",
            Module[{uop = Lookup[n, "UOp", Lookup[$uopName, n["Ext"], "?"]]},
                If[uop === "Kernel" && IntegerQ[n["Val"]] && n["Val"] > 0 &&
                   kernelMonolithicAt[n["Val"]],
                    0,
                    uopDataArity[uop]]],
        KeyExistsQ[$heapTagArity, tag],
            With[{a = $heapTagArity[tag]}, If[IntegerQ[a], a, 0]],
        True, 0
    ]
];

(* Port name for child slot i of term n. *)
termPortName[n_Association, i_Integer] := Which[
    n["Tag"] === "Top", uopPortName[Lookup[n, "UOp", Lookup[$uopName, n["Ext"], "?"]], i],
    True,               heapPortName[n["Tag"], i]
];

kernelMonolithicAt[val_Integer] := Module[{slot1},
    slot1 = Quiet@Check[THeapRead[val + 1], None];
    AssociationQ[slot1] && slot1["Tag"] === "Any"
];

computeUOpQ[uop_String] := MemberQ[$elementwiseOps, uop] ||
    MemberQ[$reduceOps, uop] || MemberQ[$viewOps, uop];

walkNodeDisplayLoc[n_Association] := If[
    n["Tag"] === "Var", n["Val"],
    If[n["Tag"] === "Top" || KeyExistsQ[$heapTagArity, n["Tag"]],
        n["Val"], n["Loc"]]
];

(* Walk the heap from `rootTerm` (a THeapRead-shaped association).
   Returns <|"Nodes" -> <|key -> record|>,
            "Edges" -> {<|From, To, Port, FromSlot|>...},
            "Root"  -> rootKey|>. *)
(* For a compound term (TOP or heap-arity tag), its identity is its args base (val),
   NOT the slot it happened to be read from. Atoms have no heap identity. *)
termKey[n_Association] := If[
    n["Tag"] === "Top" || KeyExistsQ[$heapTagArity, n["Tag"]],
    "h" <> ToString[n["Val"]],
    atomKey[n]
];

walkNodeRecord[n_Association] := Module[{derived = <||>, chain, dims},
    If[n["Tag"] === "Top" && IntegerQ[n["Val"]] && n["Val"] > 0,
        derived["UOp"] = Lookup[n, "UOp", Lookup[$uopName, n["Ext"], "?"]];
        If[derived["UOp"] === "Kernel",
            chain = kernelOpChainAt[n["TagCode"], n["Ext"], n["Val"]];
            If[StringQ[chain] && chain =!= "", derived["KernelOpChain"] = chain]];
        dims = inferShapeAt[n["Val"]];
        If[ListQ[dims] && Length[dims] > 0, derived["Shape"] = dims]];
    <|n,
        "Key" -> termKey[n],
        "DisplayLoc" -> walkNodeDisplayLoc[n],
        derived|>
];

refDefRoot[n_Association] := Module[{d},
    If[n["Tag"] =!= "Ref", Return[None]];
    d = Quiet@Check[TDefRead[n["Ext"]], None];
    If[AssociationQ[d] && d["Tag"] =!= "Era", d, None]
];

(* Walk the heap chain of nested KERNELs at args base `val`, returning the
   chained op label (e.g. "MUL+ADD"). Captured at walk time so renderers
   that consume a snapshot don't have to re-read mutated heap. *)
(* Chain comes from the cached KernelEntry's FusedOp[] (built lazily by
   fuse_build_kernel for monolithic on-heap kernels). Same path dump.c uses,
   exposed via thvm_kernel_op_chain. *)
kernelOpChainAt[tagCode_Integer, ext_Integer, val_Integer] := Module[{s},
    s = Quiet@Check[thvmKernelOpChainFn[tagCode, ext, val], ""];
    If[StringQ[s], ToUpperCase[s], ""]];

(* Infer shape of a compound term at heap base `val` by walking to first
   TEN leaf — also captured at walk time. *)
inferShapeAt[val_Integer] := Module[{seen = <||>, go, dims},
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

kernelSemanticLeavesAt[val_Integer] := Module[{payload, leaves = <||>, seen = <||>, visit},
    payload = Quiet@Check[THeapRead[val], None];
    If[!AssociationQ[payload], Return[{}]];
    visit[n_Association] := Module[{uop, loc, arity, child, childUOp, key, port},
        If[n["Tag"] =!= "Top" || !IntegerQ[n["Val"]] || n["Val"] <= 0, Return[]];
        loc = n["Val"];
        If[KeyExistsQ[seen, loc], Return[]];
        seen[loc] = True;
        uop = Lookup[$uopName, n["Ext"], "?"];
        If[!computeUOpQ[uop], Return[]];
        arity = termChildCount[n];
        Do[
            child = Quiet@Check[THeapRead[loc + i], None];
            If[!AssociationQ[child] || MemberQ[{"Any", "Era"}, child["Tag"]], Continue[]];
            childUOp = If[child["Tag"] === "Top", Lookup[$uopName, child["Ext"], "?"], ""];
            If[child["Tag"] === "Top" && computeUOpQ[childUOp], visit[child]],
            {i, 0, arity - 1}
        ];
        Do[
            child = Quiet@Check[THeapRead[loc + i], None];
            If[!AssociationQ[child] || MemberQ[{"Any", "Era"}, child["Tag"]], Continue[]];
            childUOp = If[child["Tag"] === "Top", Lookup[$uopName, child["Ext"], "?"], ""];
            If[child["Tag"] === "Top" && computeUOpQ[childUOp], Continue[]];
            key = termKey[child];
            port = ToUpperCase[uop] <> ":" <> uopPortName[uop, i];
            If[!KeyExistsQ[leaves, key],
                leaves[key] = <|"Node" -> child, "Port" -> port|>],
            {i, 0, arity - 1}
        ];
    ];
    visit[payload];
    Values[leaves]
];

heapWalk[rootTerm_Association] := Module[
    {nodes = <||>, edges = {}, rootKey, visit},
    visit[nAssoc_Association] := Module[
        {n = nAssoc, k, dk, defRoot, nChildren, childLoc, child, pname, ck, semanticLeaves},
        k = termKey[n];
        If[KeyExistsQ[nodes, k], Return[k]];
        nodes[k] = walkNodeRecord[n];
        n = nodes[k];
        If[n["Tag"] === "Ref",
            defRoot = refDefRoot[n];
            If[AssociationQ[defRoot],
                dk = visit[defRoot];
                AppendTo[edges, <|
                    "From" -> k, "To" -> dk,
                    "Port" -> "def", "FromSlot" -> 0,
                    "Style" -> "RefDef"|>]
            ]
        ];
        If[n["Tag"] === "Top" && Lookup[n, "UOp", ""] === "Kernel" &&
           IntegerQ[n["Val"]] && n["Val"] > 0 && kernelMonolithicAt[n["Val"]],
            semanticLeaves = kernelSemanticLeavesAt[n["Val"]];
            Do[
                With[{leafNode = leaf["Node"], ck = termKey[leaf["Node"]]},
                    If[!KeyExistsQ[nodes, ck], nodes[ck] = walkNodeRecord[leafNode]];
                    AppendTo[edges, <|
                        "From" -> ck, "To" -> k,
                        "Port" -> leaf["Port"], "FromSlot" -> 0,
                        "Style" -> "KernelSemantic"|>]],
                {leaf, semanticLeaves}];
            Return[k]];
        nChildren = termChildCount[n];
        If[nChildren > 0 && n["Val"] > 0,
            Do[
                childLoc = n["Val"] + i;
                child = THeapRead[childLoc];
                (* Skip ANY placeholders — unreduced slots have no source to render. *)
                If[child["Tag"] =!= "Any",
                    pname = termPortName[n, i];
                    ck = visit[child];
                    If[MemberQ[{"Lam", "Bri"}, n["Tag"]] && i == 0 && child["Tag"] === "Var",
                        AppendTo[edges, <|
                            "From" -> k, "To" -> ck,
                            "Port" -> pname, "FromSlot" -> childLoc|>],
                        AppendTo[edges, <|
                            "From" -> ck, "To" -> k,
                            "Port" -> pname, "FromSlot" -> childLoc|>]
                    ]],
                {i, 0, nChildren - 1}]
        ];
        k
    ];
    rootKey = visit[rootTerm];
    <|"Nodes" -> nodes, "Edges" -> edges, "Root" -> rootKey|>
];

(* Get the root term of a TTensor/TTerm handle as a THeapRead-shaped assoc. *)
rootTermOf[t_] := Module[{id, tagCode, ext, val, tag},
    id = termId[t];
    tagCode = thvmTermTagFn[id];
    ext     = thvmTermExtFn[id];
    val     = thvmTermValFn[id];
    tag     = Lookup[$tagName, tagCode, "?"];
    iHeapTermAssoc[tagCode, ext, val,
      If[tag === "Top" || KeyExistsQ[$heapTagArity, tag], val, 0]]
];
