#!/usr/bin/env wolframscript
(* XOR synthesis with full circuit recovery: bf → Boolean expression *)
PacletDirectoryLoad[FileNameJoin[{DirectoryName[$InputFileName], "..", "wl"}]];
Needs["TinyHVM`"];

TInit["metal"];

bNot[x_] := TOp2["Sub", TNum[1], x];
bAnd[a_, b_] := TOp2["Mul", a, b];

(* Wire factory: returns {TTerm, decoderNode} *)
mkWireDual[] := Module[{l1, l2, l3},
  l1 = TFreshLabel[]; l2 = TFreshLabel[]; l3 = TFreshLabel[];
  {TSup[l1, TLam[x |-> TLam[y |-> x]],
      TSup[l2, TLam[x |-> TLam[y |-> y]],
          TSup[l3, TLam[x |-> TLam[y |-> bNot[x]]],
              TLam[x |-> TLam[y |-> bNot[y]]]]]],
   {"wire", l1, l2, l3}}
];

(* Gate: takes two {TTerm, decoder} pairs, returns {TTerm, decoder} *)
mkGateDual[{ft_, fd_}, {gt_, gd_}] := Module[{l, f0, f1, g0, g1},
  l = TFreshLabel[];
  {f0, f1} = TDup[ft]; {g0, g1} = TDup[gt];
  {TSup[l,
      TLam[x |-> TLam[y |-> Module[{x0, x1, y0, y1},
          {x0, x1} = TDup[x]; {y0, y1} = TDup[y];
          bAnd[TApp[TApp[f0, x0], y0], TApp[TApp[g0, x1], y1]]]]],
      TLam[x |-> TLam[y |-> Module[{x0, x1, y0, y1},
          {x0, x1} = TDup[x]; {y0, y1} = TDup[y];
          bNot[bAnd[TApp[TApp[f1, x0], y0], TApp[TApp[g1, x1], y1]]]]]]],
   {"gate", l, fd, gd}}
];

(* Balanced SUP tree over {TTerm, decoder} pairs *)
dualSupTree[{x_}] := x;
dualSupTree[pairs_List] := Module[{mid, l, left, right},
  mid = Ceiling[Length[pairs]/2];
  l = TFreshLabel[];
  left = dualSupTree[pairs[[;;mid]]];
  right = dualSupTree[pairs[[mid+1;;]]];
  {TSup[l, left[[1]], right[[1]]],
   {"struct", l, left[[2]], right[[2]]}}
];

(* TGroupings replacement returning {TTerm, decoderTree} *)
buildGrouped[{fac_}, _] := fac[];
buildGrouped[{f1_, f2_}, g_] := g[f1[], f2[]];
buildGrouped[facs_List, g_] := Module[{alts},
  alts = Table[
    g[buildGrouped[facs[[;;k]], g], buildGrouped[facs[[k+1;;]], g]],
    {k, 1, Length[facs]-1}];
  dualSupTree[alts]
];

(* === Decoder: bf Association → Boolean expression === *)
decode[bf_, {"struct", l_, left_, right_}] :=
  If[Lookup[bf, l, 0] === 0, decode[bf, left], decode[bf, right]];
decode[bf_, {"gate", l_, left_, right_}] :=
  If[Lookup[bf, l, 0] === 0, And, Nand][decode[bf, left], decode[bf, right]];
decode[bf_, {"wire", l1_, l2_, l3_}] :=
  If[Lookup[bf, l1, 0] === 0, a,
  If[Lookup[bf, l2, 0] === 0, b,
  If[Lookup[bf, l3, 0] === 0, Not[a], Not[b]]]];

(* === Build and solve === *)
Print["=== XOR Circuit Synthesis ===\n"];

{circuitTerm, decoder} =
  buildGrouped[ConstantArray[mkWireDual[] &, 4], mkGateDual];

(* DUP circuit for 4 truth-table evaluations *)
{c0, ct} = TDup[circuitTerm];
{c1, ct2} = TDup[ct];
{c2, c3} = TDup[ct2];
checks = MapThread[
  TOp2["Eq", TApp[TApp[#1, TNum[#2]], TNum[#3]], TNum[#4]] &,
  {{c0, c1, c2, c3}, {0, 0, 1, 1}, {0, 1, 0, 1}, {0, 1, 1, 0}}];
allMatch = Fold[bAnd, checks];

(* Grouped collapse *)
{time, gr} = AbsoluteTiming[TCollapseGrouped[allMatch]];
vals = TNumValue /@ gr["values"];
xorBFs = Pick[gr["bf"], vals, 1];

Print["Candidates: ", Length[vals]];
Print["XOR hits:   ", Length[xorBFs]];
Print["Time:       ", time, "s"];
Print["Interactions: ", TInteractionCount[]];

(* Decode each XOR bf into a Boolean expression *)
xorExprs = decode[#, decoder] & /@ xorBFs;
uniqueXOR = DeleteDuplicates[xorExprs];

Print["\nUnique circuits: ", Length[uniqueXOR], "\n"];
Do[
  expr = uniqueXOR[[i]];
  tt = BooleanTable[expr, {a, b}];
  Print[i, ". ", InputForm[expr]];
  Print["   truth table: ", tt,
    If[tt === {False, True, True, False}, "  \[Checkmark]", "  \[Cross]"]];
, {i, Length[uniqueXOR]}];

(* Verify ALL are XOR *)
Print["\nAll correct: ",
  AllTrue[uniqueXOR, BooleanTable[#, {a, b}] === {False, True, True, False} &]];
