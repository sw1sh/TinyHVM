(* Synthesis.wl - FindBooleanAlternative via brute-force or TinyHVM superposition *)

$boolArity = Prepend[Not -> 1] @
    AssociationThread[{And, Or, Nand, Nor, Xor, Implies, Equivalent} -> 2];

Options[FindBooleanAlternative] = {"MaxSize" -> 8, Method -> Automatic};

FindBooleanAlternative::wrongOps =
    "List can contain only: Not, And, Or, Nand, Nor, Xor, Implies, Equivalent";

FindBooleanAlternative[expr_, ops_List,
    n : _Integer ? Positive | All | Automatic : Automatic,
    OptionsPattern[]] /;
    ContainsOnly[ops, Keys @ $boolArity] ||
    Message[FindBooleanAlternative::wrongOps] :=
Enclose @ With[{maxSize = ConfirmBy[OptionValue["MaxSize"], IntegerQ]},
    If[ OptionValue[Method] === "TinyHVM",
        fbaSup[expr, ops, n, maxSize],
        fbaGroupings[expr, ops, n, maxSize]
    ]
];

(* ======================================================================= *)
(* Brute-force via Groupings + BooleanTable                               *)
(* ======================================================================= *)

fbaGroupings[expr_, ops_, n_, maxSize_] := Block[{
    vars = Replace[BooleanVariables[expr], k_Integer :> Array[\[FormalX], k]],
    opsArities = Lookup[$boolArity, ops],
    tab = BooleanTable[expr],
    limit = Replace[n, {Automatic -> 1, All -> Infinity}]
},
    If[n === Automatic, Replace[{x_, ___} :> x], Identity] @ Take[
        SortBy[LeafCount] @ FoldWhile[
            {r, s} |-> Join[r,
                Select[
                    DeleteDuplicates @ Groupings[Tuples[vars, s], opsArities],
                    BooleanTable[#, vars] === tab &
                ]
            ],
            {},
            Range[maxSize],
            Length[#] < limit &
        ],
        UpTo[limit]
    ]
];

(* ======================================================================= *)
(* TinyHVM superposition synthesis                                         *)
(*                                                                         *)
(* Superpose all candidate circuits as lambdas in a single interaction     *)
(* net.  DUP-SUP annihilation correlates each candidate across all truth   *)
(* table evaluations.  TCollapseGrouped recovers which SUP branches were   *)
(* chosen; the decoder tree maps branch functions back to WL expressions.  *)
(*                                                                         *)
(* NOTE: calls TInit["metal"] internally - resets any existing context.    *)
(* ======================================================================= *)

(* -- Boolean ops as integer arithmetic on {0,1} ----------------------- *)

boolOpTerm[And][a_, b_]        := TOp2["Mul", a, b];
boolOpTerm[Or][a_, b_]         := TOp2["Sub", TNum[1], TOp2["Mul", TOp2["Sub", TNum[1], a], TOp2["Sub", TNum[1], b]]];
boolOpTerm[Nand][a_, b_]       := TOp2["Sub", TNum[1], TOp2["Mul", a, b]];
boolOpTerm[Nor][a_, b_]        := TOp2["Mul", TOp2["Sub", TNum[1], a], TOp2["Sub", TNum[1], b]];
boolOpTerm[Xor][a_, b_]        := TOp2["Sub", TNum[1], TOp2["Eq", a, b]];
boolOpTerm[Implies][a_, b_]    := TOp2["Sub", TNum[1], TOp2["Mul", a, TOp2["Sub", TNum[1], b]]];
boolOpTerm[Equivalent][a_, b_] := TOp2["Eq", a, b];

(* -- Expand gate ops: add Not∘op variants, dedup by truth table -------- *)

expandGateOps[binOps_, False] := {#, False} & /@ binOps;
expandGateOps[binOps_, True] := Block[{result = {}, seen = {}},
    Do[Block[{tt = BooleanTable[op[p, q], {p, q}], ntt},
        If[!MemberQ[seen, tt],  AppendTo[result, {op, False}]; AppendTo[seen, tt]];
        ntt = Not /@ tt;
        If[!MemberQ[seen, ntt], AppendTo[result, {op, True}];  AppendTo[seen, ntt]]
    ], {op, binOps}];
    result
];

(* -- Balanced n-way DUP ------------------------------------------------ *)

dupN[t_, 1] := {t};
dupN[t_, n_Integer] := Block[{a, b},
    {a, b} = TDup[t];
    Join[dupN[a, Ceiling[n / 2]], dupN[b, n - Ceiling[n / 2]]]
];

(* -- Balanced SUP tree over {TTerm, decoderValue} pairs ---------------- *)

dualSup[{{t_, d_}}] := {t, d};
dualSup[pairs_List] := Block[{mid, l, left, right},
    mid = Ceiling[Length[pairs] / 2];
    l = TFreshLabel[];
    left = dualSup[pairs[[;; mid]]];
    right = dualSup[pairs[[mid + 1 ;;]]];
    {TSup[l, left[[1]], right[[1]]],
     {"choice", l, left[[2]], right[[2]]}}
];

(* -- Navigate a binary choice tree with a branch function -------------- *)

navigateChoice[bf_, {"choice", l_, left_, right_}] :=
    If[Lookup[bf, l, 0] === 0, navigateChoice[bf, left], navigateChoice[bf, right]];
navigateChoice[_, leaf_] := leaf;

(* -- Leaf lambda: λv1…λvn. vk  or  λv1…λvn. 1−vk --------------------- *)

makeLeafLam[nVars_, k_, neg_] := leafLam0[nVars, k, neg, {}];
leafLam0[nVars_, k_, neg_, vars_] /; Length[vars] === nVars :=
    If[neg, TOp2["Sub", TNum[1], vars[[k]]], vars[[k]]];
leafLam0[nVars_, k_, neg_, vars_] :=
    TLam[v |-> leafLam0[nVars, k, neg, Append[vars, v]]];

(* -- Wire dual factory: {TTerm, {"wire", choiceTree}} ------------------ *)
(* leafSpecs: {{varIdx, negated, symbolicExpr}, ...} *)

makeWireDual[nVars_, leafSpecs_] := Block[{pairs, sup, tree},
    pairs = Table[
        {makeLeafLam[nVars, leafSpecs[[i, 1]], leafSpecs[[i, 2]]],
         leafSpecs[[i, 3]]},
        {i, Length[leafSpecs]}
    ];
    {sup, tree} = dualSup[pairs];
    {sup, {"wire", tree}}
];

(* -- Op lambda: λv1…λvn. op(f(v1a,…), g(v1b,…)) ---------------------- *)

makeOpLam[nVars_, opSpec_, fi_, gi_] := opLam0[nVars, opSpec, fi, gi, {}, {}];
opLam0[nVars_, {op_, neg_}, fi_, gi_, fArgs_, gArgs_] /;
    Length[fArgs] === nVars :=
Block[{result = boolOpTerm[op][Fold[TApp, fi, fArgs], Fold[TApp, gi, gArgs]]},
    If[neg, TOp2["Sub", TNum[1], result], result]
];
opLam0[nVars_, opSpec_, fi_, gi_, fArgs_, gArgs_] :=
    TLam[v |-> Block[{v0, v1},
        {v0, v1} = TDup[v];
        opLam0[nVars, opSpec, fi, gi,
            Append[fArgs, v0], Append[gArgs, v1]]
    ]];

(* -- Gate dual factory: {TTerm, {"gate", opTree, leftDec, rightDec}} --- *)

makeGateDual[nVars_, gateOps_][{ft_, fd_}, {gt_, gd_}] :=
Block[{nOps = Length[gateOps], fCopies, gCopies, opLams, pairs, sup, opTree},
    fCopies = dupN[ft, nOps];
    gCopies = dupN[gt, nOps];
    opLams = MapThread[makeOpLam[nVars, #1, #2, #3] &,
        {gateOps, fCopies, gCopies}];
    pairs = MapThread[List, {opLams, gateOps}];
    {sup, opTree} = dualSup[pairs];
    {sup, {"gate", opTree, fd, gd}}
];

(* -- Structural (Catalan) grouping - dual construction ----------------- *)

buildDual[{fac_}, _] := fac[];
buildDual[{f1_, f2_}, g_] := g[f1[], f2[]];
buildDual[facs_List, g_] := structSup @ Table[
    g[buildDual[facs[[;; k]], g], buildDual[facs[[k + 1 ;;]], g]],
    {k, 1, Length[facs] - 1}
];

structSup[{x_}] := x;
structSup[pairs_List] := Block[{mid, l, left, right},
    mid = Ceiling[Length[pairs] / 2];
    l = TFreshLabel[];
    left = structSup[pairs[[;; mid]]];
    right = structSup[pairs[[mid + 1 ;;]]];
    {TSup[l, left[[1]], right[[1]]],
     {"struct", l, left[[2]], right[[2]]}}
];

(* -- Decode branch function → Boolean expression ----------------------- *)

decodeBranchFunction[bf_, {"struct", l_, left_, right_}] :=
    If[Lookup[bf, l, 0] === 0, decodeBranchFunction[bf, left], decodeBranchFunction[bf, right]];
decodeBranchFunction[bf_, {"gate", opTree_, leftSub_, rightSub_}] :=
    opSpecToExpr[navigateChoice[bf, opTree],
        decodeBranchFunction[bf, leftSub], decodeBranchFunction[bf, rightSub]];
decodeBranchFunction[bf_, {"wire", wireTree_}] := navigateChoice[bf, wireTree];

opSpecToExpr[{op_, False}, a_, b_] := op[a, b];
opSpecToExpr[{op_, True}, a_, b_] := Not[op[a, b]];

(* -- Main driver ------------------------------------------------------- *)

fbaSup[expr_, ops_, n_, maxSize_] := Block[{
    vars = Replace[BooleanVariables[expr], k_Integer :> Array[\[FormalX], k]],
    tab, nVars, binOps, hasNot, gateOps, leafSpecs,
    inputs, targets, limit, results = {}
},
    tab = BooleanTable[expr];
    nVars = Length[vars];
    binOps = Select[ops, $boolArity[#] === 2 &];
    hasNot = MemberQ[ops, Not];

    (* Need at least one binary op for tree synthesis *)
    If[binOps === {}, Return[fbaGroupings[expr, ops, n, maxSize]]];

    gateOps = expandGateOps[binOps, hasNot];
    leafSpecs = Join[
        Table[{k, False, vars[[k]]}, {k, nVars}],
        If[hasNot, Table[{k, True, Not[vars[[k]]]}, {k, nVars}], {}]
    ];
    inputs = Tuples[{1, 0}, nVars];  (* match BooleanTable order: TT, TF, FT, FF *)
    targets = Boole[tab];
    limit = Replace[n, {Automatic -> 1, All -> Infinity}];

    Do[
        If[Length[results] >= limit, Break[]];
        TInit["metal"];
        Block[{circuit, decoder, copies, checks, allMatch, gr, vals, bfs, newExprs},
            {circuit, decoder} = buildDual[
                ConstantArray[makeWireDual[nVars, leafSpecs] &, nLeaves],
                makeGateDual[nVars, gateOps]];
            copies = dupN[circuit, Length[inputs]];
            checks = MapThread[
                TOp2["Eq", Fold[TApp, #1, TNum /@ #2], TNum[#3]] &,
                {copies, inputs, targets}];
            allMatch = Fold[TOp2["Mul", #1, #2] &, checks];
            gr = TCollapseGrouped[allMatch];
            vals = TNumValue /@ gr["values"];
            bfs = Pick[gr["bf"], vals, 1];
            newExprs = DeleteDuplicates[decodeBranchFunction[#, decoder] & /@ bfs];
            results = DeleteDuplicates[Join[results, newExprs]]
        ],
        {nLeaves, 1, maxSize}
    ];

    results = SortBy[results, LeafCount];
    If[n === Automatic, Replace[results, {x_, ___} :> x], Take[results, UpTo[limit]]]
];
