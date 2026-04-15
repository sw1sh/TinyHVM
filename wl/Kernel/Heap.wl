(* Heap.wl — WL-side heap walker built on THeapRead / THeapReadRange / TTermVal.
   Get'd from TinyHVM.wl inside Begin["`Private`"].
   Rendering lives in Visualization.wl; this file only produces node/edge lists. *)

(* Synthetic atom key — TEN nodes don't own a heap slot, so we key by tensor id. *)
atomKey[n_Association] := Switch[n["Tag"],
    "Ten",  "t" <> ToString[n["Val"]],
    "Num",  "num" <> ToString[n["Val"]] <> "_" <> ToString[n["Ext"]],
    "Era",  "era",
    "Ref",  "ref" <> ToString[n["Val"]],
    "Var",  "var" <> ToString[n["Val"]],
    "Any",  "any",
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
        tag === "Top",  uopDataArity[Lookup[$uopName, n["Ext"], "?"]],
        KeyExistsQ[$heapTagArity, tag],
            With[{a = $heapTagArity[tag]}, If[IntegerQ[a], a, 0]],
        True, 0
    ]
];

(* Port name for child slot i of term n. *)
termPortName[n_Association, i_Integer] := Which[
    n["Tag"] === "Top", uopPortName[Lookup[$uopName, n["Ext"], "?"], i],
    True,               heapPortName[n["Tag"], i]
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

heapWalk[rootTerm_Association] := Module[
    {nodes = <||>, edges = {}, rootKey, visit},
    visit[nAssoc_Association] := Module[
        {n = nAssoc, k, nChildren, childLoc, child, pname,
         derived, chain, dims},
        k = termKey[n];
        If[KeyExistsQ[nodes, k], Return[k]];
        derived = <||>;
        If[n["Tag"] === "Top" && IntegerQ[n["Val"]] && n["Val"] > 0,
            If[Lookup[$uopName, n["Ext"], ""] === "Kernel",
                chain = kernelOpChainAt[n["TagCode"], n["Ext"], n["Val"]];
                If[StringQ[chain] && chain =!= "", derived["KernelOpChain"] = chain]];
            dims = inferShapeAt[n["Val"]];
            If[ListQ[dims] && Length[dims] > 0, derived["Shape"] = dims]];
        (* Store display loc (val for compound, Loc for atoms). *)
        nodes[k] = <|n,
            "Key" -> k,
            "DisplayLoc" -> If[n["Tag"] === "Top" ||
                               KeyExistsQ[$heapTagArity, n["Tag"]],
                               n["Val"], n["Loc"]],
            derived|>;
        nChildren = termChildCount[n];
        If[nChildren > 0 && n["Val"] > 0,
            Do[
                childLoc = n["Val"] + i;
                child = THeapRead[childLoc];
                (* Skip ANY placeholders — unreduced slots have no source to render. *)
                If[child["Tag"] =!= "Any",
                    pname = termPortName[n, i];
                    With[{ck = visit[child]},
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
    <|"Tag" -> tag, "TagCode" -> tagCode, "Ext" -> ext, "Val" -> val,
      "Loc" -> If[tag === "Top" || KeyExistsQ[$heapTagArity, tag], val, 0]|>
];
