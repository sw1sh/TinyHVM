(* Synthesis.wl - FindBooleanAlternative via brute-force or TinyHVM superposition *)

$iBoolArity = Prepend[Not -> 1] @
    AssociationThread[{And, Or, Nand, Nor, Xor, Implies, Equivalent} -> 2];

Options[FindBooleanAlternative] = {"MaxSize" -> 8, Method -> Automatic};

FindBooleanAlternative::wrongOps =
    "List can contain only: Not, And, Or, Nand, Nor, Xor, Implies, Equivalent";

FindBooleanAlternative[expr_, ops_List,
    n : _Integer ? Positive | All | Automatic : Automatic,
    OptionsPattern[]] /;
    ContainsOnly[ops, Keys @ $iBoolArity] ||
    Message[FindBooleanAlternative::wrongOps] :=
Enclose @ With[{maxSize = ConfirmBy[OptionValue["MaxSize"], IntegerQ]},
    If[ OptionValue[Method] === "TinyHVM",
        iFBASup[expr, ops, n, maxSize],
        iFBAGroupings[expr, ops, n, maxSize]
    ]
];

(* ======================================================================= *)
(* Brute-force via Groupings + BooleanTable                               *)
(* ======================================================================= *)

iFBAGroupings[expr_, ops_, n_, maxSize_] := Module[{
    vars = Replace[BooleanVariables[expr], k_Integer :> Array[\[FormalX], k]],
    opsArities = Lookup[$iBoolArity, ops],
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

iBoolA[And][a_, b_]        := TOp2["Mul", a, b];
iBoolA[Or][a_, b_]         := TOp2["Sub", TNum[1], TOp2["Mul", TOp2["Sub", TNum[1], a], TOp2["Sub", TNum[1], b]]];
iBoolA[Nand][a_, b_]       := TOp2["Sub", TNum[1], TOp2["Mul", a, b]];
iBoolA[Nor][a_, b_]        := TOp2["Mul", TOp2["Sub", TNum[1], a], TOp2["Sub", TNum[1], b]];
iBoolA[Xor][a_, b_]        := TOp2["Sub", TNum[1], TOp2["Eq", a, b]];
iBoolA[Implies][a_, b_]    := TOp2["Sub", TNum[1], TOp2["Mul", a, TOp2["Sub", TNum[1], b]]];
iBoolA[Equivalent][a_, b_] := TOp2["Eq", a, b];

(* -- Expand gate ops: add Not∘op variants, dedup by truth table -------- *)

iExpandGateOps[binOps_, False] := {#, False} & /@ binOps;
iExpandGateOps[binOps_, True] := Module[{result = {}, seen = {}},
    Do[Module[{tt = BooleanTable[op[p, q], {p, q}], ntt},
        If[!MemberQ[seen, tt],  AppendTo[result, {op, False}]; AppendTo[seen, tt]];
        ntt = Not /@ tt;
        If[!MemberQ[seen, ntt], AppendTo[result, {op, True}];  AppendTo[seen, ntt]]
    ], {op, binOps}];
    result
];

(* -- Balanced n-way DUP ------------------------------------------------ *)

iDupN[t_, 1] := {t};
iDupN[t_, n_Integer] := Module[{a, b},
    {a, b} = TDup[t];
    Join[iDupN[a, Ceiling[n / 2]], iDupN[b, n - Ceiling[n / 2]]]
];

(* -- Balanced SUP tree over {TTerm, decoderValue} pairs ---------------- *)

iDualSup[{{t_, d_}}] := {t, d};
iDualSup[pairs_List] := Module[{mid, l, left, right},
    mid = Ceiling[Length[pairs] / 2];
    l = TFreshLabel[];
    left = iDualSup[pairs[[;; mid]]];
    right = iDualSup[pairs[[mid + 1 ;;]]];
    {TSup[l, left[[1]], right[[1]]],
     {"choice", l, left[[2]], right[[2]]}}
];

(* -- Navigate a binary choice tree with a branch function -------------- *)

iNavChoice[bf_, {"choice", l_, left_, right_}] :=
    If[Lookup[bf, l, 0] === 0, iNavChoice[bf, left], iNavChoice[bf, right]];
iNavChoice[_, leaf_] := leaf;

(* -- Leaf lambda: λv1…λvn. vk  or  λv1…λvn. 1−vk --------------------- *)

iMkLeafLam[nVars_, k_, neg_] := iLeafLam0[nVars, k, neg, {}];
iLeafLam0[nVars_, k_, neg_, vars_] /; Length[vars] === nVars :=
    If[neg, TOp2["Sub", TNum[1], vars[[k]]], vars[[k]]];
iLeafLam0[nVars_, k_, neg_, vars_] :=
    TLam[v |-> iLeafLam0[nVars, k, neg, Append[vars, v]]];

(* -- Wire dual factory: {TTerm, {"wire", choiceTree}} ------------------ *)
(* leafSpecs: {{varIdx, negated, symbolicExpr}, ...} *)

iMkWireDual[nVars_, leafSpecs_] := Module[{pairs, sup, tree},
    pairs = Table[
        {iMkLeafLam[nVars, leafSpecs[[i, 1]], leafSpecs[[i, 2]]],
         leafSpecs[[i, 3]]},
        {i, Length[leafSpecs]}
    ];
    {sup, tree} = iDualSup[pairs];
    {sup, {"wire", tree}}
];

(* -- Op lambda: λv1…λvn. op(f(v1a,…), g(v1b,…)) ---------------------- *)

iMkOpLam[nVars_, opSpec_, fi_, gi_] := iOpLam0[nVars, opSpec, fi, gi, {}, {}];
iOpLam0[nVars_, {op_, neg_}, fi_, gi_, fArgs_, gArgs_] /;
    Length[fArgs] === nVars :=
Module[{result = iBoolA[op][Fold[TApp, fi, fArgs], Fold[TApp, gi, gArgs]]},
    If[neg, TOp2["Sub", TNum[1], result], result]
];
iOpLam0[nVars_, opSpec_, fi_, gi_, fArgs_, gArgs_] :=
    TLam[v |-> Module[{v0, v1},
        {v0, v1} = TDup[v];
        iOpLam0[nVars, opSpec, fi, gi,
            Append[fArgs, v0], Append[gArgs, v1]]
    ]];

(* -- Gate dual factory: {TTerm, {"gate", opTree, leftDec, rightDec}} --- *)

iMkGateDual[nVars_, gateOps_][{ft_, fd_}, {gt_, gd_}] :=
Module[{nOps = Length[gateOps], fCopies, gCopies, opLams, pairs, sup, opTree},
    fCopies = iDupN[ft, nOps];
    gCopies = iDupN[gt, nOps];
    opLams = MapThread[iMkOpLam[nVars, #1, #2, #3] &,
        {gateOps, fCopies, gCopies}];
    pairs = MapThread[List, {opLams, gateOps}];
    {sup, opTree} = iDualSup[pairs];
    {sup, {"gate", opTree, fd, gd}}
];

(* -- Structural (Catalan) grouping - dual construction ----------------- *)

iBuildDual[{fac_}, _] := fac[];
iBuildDual[{f1_, f2_}, g_] := g[f1[], f2[]];
iBuildDual[facs_List, g_] := iStructSup @ Table[
    g[iBuildDual[facs[[;; k]], g], iBuildDual[facs[[k + 1 ;;]], g]],
    {k, 1, Length[facs] - 1}
];

iStructSup[{x_}] := x;
iStructSup[pairs_List] := Module[{mid, l, left, right},
    mid = Ceiling[Length[pairs] / 2];
    l = TFreshLabel[];
    left = iStructSup[pairs[[;; mid]]];
    right = iStructSup[pairs[[mid + 1 ;;]]];
    {TSup[l, left[[1]], right[[1]]],
     {"struct", l, left[[2]], right[[2]]}}
];

(* -- Decode branch function → Boolean expression ----------------------- *)

iDecodeBF[bf_, {"struct", l_, left_, right_}] :=
    If[Lookup[bf, l, 0] === 0, iDecodeBF[bf, left], iDecodeBF[bf, right]];
iDecodeBF[bf_, {"gate", opTree_, leftSub_, rightSub_}] :=
    iOpToExpr[iNavChoice[bf, opTree],
        iDecodeBF[bf, leftSub], iDecodeBF[bf, rightSub]];
iDecodeBF[bf_, {"wire", wireTree_}] := iNavChoice[bf, wireTree];

iOpToExpr[{op_, False}, a_, b_] := op[a, b];
iOpToExpr[{op_, True}, a_, b_] := Not[op[a, b]];

(* -- Main driver ------------------------------------------------------- *)

iFBASup[expr_, ops_, n_, maxSize_] := Module[{
    vars = Replace[BooleanVariables[expr], k_Integer :> Array[\[FormalX], k]],
    tab, nVars, binOps, hasNot, gateOps, leafSpecs,
    inputs, targets, limit, results = {}
},
    tab = BooleanTable[expr];
    nVars = Length[vars];
    binOps = Select[ops, $iBoolArity[#] === 2 &];
    hasNot = MemberQ[ops, Not];

    (* Need at least one binary op for tree synthesis *)
    If[binOps === {}, Return[iFBAGroupings[expr, ops, n, maxSize]]];

    gateOps = iExpandGateOps[binOps, hasNot];
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
        Module[{circuit, decoder, copies, checks, allMatch, gr, vals, bfs, newExprs},
            {circuit, decoder} = iBuildDual[
                ConstantArray[iMkWireDual[nVars, leafSpecs] &, nLeaves],
                iMkGateDual[nVars, gateOps]];
            copies = iDupN[circuit, Length[inputs]];
            checks = MapThread[
                TOp2["Eq", Fold[TApp, #1, TNum /@ #2], TNum[#3]] &,
                {copies, inputs, targets}];
            allMatch = Fold[TOp2["Mul", #1, #2] &, checks];
            gr = TCollapseGrouped[allMatch];
            vals = TNumValue /@ gr["values"];
            bfs = Pick[gr["bf"], vals, 1];
            newExprs = DeleteDuplicates[iDecodeBF[#, decoder] & /@ bfs];
            results = DeleteDuplicates[Join[results, newExprs]]
        ],
        {nLeaves, 1, maxSize}
    ];

    results = SortBy[results, LeafCount];
    If[n === Automatic, Replace[results, {x_, ___} :> x], Take[results, UpTo[limit]]]
];
