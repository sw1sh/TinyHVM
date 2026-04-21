#!/usr/bin/env wolframscript
(* Test: lambda-based XOR synthesis with 4 wires via TGroupings *)
PacletDirectoryLoad[FileNameJoin[{DirectoryName[$InputFileName], "..", "wl"}]];
Needs["TinyHVM`"];

TInit["metal"];

bNot[x_] := TOp2["Sub", TNum[1], x];
bAnd[a_, b_] := TOp2["Mul", a, b];

(* Wire: lambda (a,b) -> one of {a, b, NOT(a), NOT(b)} *)
mkWire[] := TSup[
  TLam[a |-> TLam[b |-> a]],
  TSup[TLam[a |-> TLam[b |-> b]],
    TSup[TLam[a |-> TLam[b |-> bNot[a]]],
      TLam[a |-> TLam[b |-> bNot[b]]]]]];

(* Gate: combine two sub-circuit lambdas with AND or NAND *)
mkGate[f_TTerm, g_TTerm] := Module[{l = TFreshLabel[], f0, f1, g0, g1},
  {f0, f1} = TDup[f]; {g0, g1} = TDup[g];
  TSup[l,
    TLam[a |-> TLam[b |-> Module[{a0, a1, b0, b1},
      {a0, a1} = TDup[a]; {b0, b1} = TDup[b];
      bAnd[TApp[TApp[f0, a0], b0], TApp[TApp[g0, a1], b1]]]]],
    TLam[a |-> TLam[b |-> Module[{a0, a1, b0, b1},
      {a0, a1} = TDup[a]; {b0, b1} = TDup[b];
      bNot[bAnd[TApp[TApp[f1, a0], b0], TApp[TApp[g1, a1], b1]]]]]]]
];

buildSynth[] := Module[{circuit, c0, ct, c1, ct2, c2, c3, checks},
  circuit = TGroupings[ConstantArray[mkWire[] &, 4], mkGate];
  {c0, ct} = TDup[circuit]; {c1, ct2} = TDup[ct]; {c2, c3} = TDup[ct2];
  checks = MapThread[
    TOp2["Eq", TApp[TApp[#1, TNum[#2]], TNum[#3]], TNum[#4]] &,
    {{c0, c1, c2, c3}, {0, 0, 1, 1}, {0, 1, 0, 1}, {0, 1, 1, 0}}];
  Fold[bAnd, checks]];

Print["=== Lambda 4-wire XOR synthesis ==="];
{t, results} = AbsoluteTiming[TNumValue /@ TCollapse[buildSynth[]]];
xorCount = Total[results];
Print["  candidates: ", Length[results]];
Print["  XOR: ", xorCount];
Print["  time: ", t, "s"];
Print["  interactions: ", TInteractionCount[]];
If[xorCount > 0, Print["  PASS"], Print["  FAIL"]];

(* Grouped collapse *)
Print["\n=== Grouped collapse ==="];
TInit["metal"];
gr = TCollapseGrouped[buildSynth[]];
vals = TNumValue /@ gr["values"];
xorBFs = Pick[gr["bf"], vals, 1];
Print["  candidates: ", Length[vals]];
Print["  XOR: ", Length[xorBFs]];
If[Length[xorBFs] > 0,
    Print["  Example bf: ", xorBFs[[1]]];
    Print["  bf labels: ", Length[xorBFs[[1]]], " entries"];
    Print["  PASS"]];
