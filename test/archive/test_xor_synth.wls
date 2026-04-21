#!/usr/bin/env wolframscript
(* test_xor_synth.wls — XOR synthesis via TGroupings + thread scaling
   3-wire direct approach: shared labels across input pairs, no lambdas. *)

PacletDirectoryLoad[FileNameJoin[{DirectoryName[$InputFileName], "..", "wl"}]];
Needs["TinyHVM`"];

bNot[x_] := TOp2["Sub", TNum[1], x];
bAnd[a_, b_] := TOp2["Mul", a, b];

(* Wire for one input pair, using pre-allocated labels for cross-pair correlation *)
mkWireFor[ai_, bi_, wlabs_List] :=
    TSup[wlabs[[1]], TNum[ai],
        TSup[wlabs[[2]], TNum[bi],
            TSup[wlabs[[3]], bNot[TNum[ai]], bNot[TNum[bi]]]]];

(* Gate with a specific label: AND(0) or NAND(1) *)
mkGateFor[x_, y_, l_Integer] := TSup[l, bAnd[x, y], bNot[bAnd[x, y]]];

(* Build ALL circuits for one input pair.
   wireLabels: {{l1,l2,l3}, ...} per wire position
   gateLabels: flat list, consumed in order by mkGate calls *)
buildCircuitForInput[ai_, bi_, wireLabels_, gateLabels_] :=
Module[{gIdx = 0, wireFacs, gate},
    wireFacs = Table[
        With[{a = ai, b = bi, wl = wireLabels[[i]]},
            mkWireFor[a, b, wl] &],
        {i, Length[wireLabels]}];
    gate[x_TTerm, y_TTerm] := Module[{l},
        gIdx++; l = gateLabels[[gIdx]];
        mkGateFor[x, y, l]];
    TGroupings[wireFacs, gate]
];

(* Full synthesis: build same circuit for 4 input pairs with shared labels,
   AND all truth-table checks together *)
buildSynth[nWires_Integer: 3] := Module[{
    wireLabels = Table[Table[TFreshLabel[], 3], nWires],
    gateLabels = Table[TFreshLabel[], 50],
    inputs = {{0,0}, {0,1}, {1,0}, {1,1}},
    xorTT = {0, 1, 1, 0},
    terms, checks},

    terms = Table[
        buildCircuitForInput[inputs[[i,1]], inputs[[i,2]],
            wireLabels, gateLabels],
        {i, 4}];
    checks = MapThread[TOp2["Eq", #1, TNum[#2]] &, {terms, xorTT}];
    Fold[bAnd, checks]
];

(* === 1. Basic collapse === *)
Print["=== XOR Synthesis via TGroupings (direct, shared labels) ==="];
TInit["metal"];
{time1, results1} = AbsoluteTiming[TNumValue /@ TCollapse[buildSynth[]]];
xorCount = Total[results1];
Print["  candidates: ", Length[results1]];
Print["  XOR circuits: ", xorCount];
Print["  time: ", time1, "s"];
Print["  interactions: ", TInteractionCount[]];

If[xorCount > 0,
    Print["  PASS: found XOR circuits"],
    Print["  FAIL: no XOR circuits found"]];

(* === 2. Grouped collapse === *)
Print["\n=== Grouped Collapse ==="];
TInit["metal"];
gr = TCollapseGrouped[buildSynth[]];
vals = TNumValue /@ gr["values"];
hits = Pick[gr["bf"], vals, 1];
Print["  candidates: ", Length[vals]];
Print["  XOR circuits: ", Length[hits]];
If[Length[hits] > 0,
    Print["  Example bf: ", hits[[1]]];
    Print["  PASS: bf recovered"],
    Print["  FAIL: no hits"]];

(* === 3. Thread scaling === *)
Print["\n=== Thread Scaling ==="];
Print[StringPadRight["Threads", 10], StringPadRight["Time(s)", 12],
      StringPadRight["Results", 10], StringPadRight["XOR", 8], "Speedup"];

baseTime = Null;
Do[
    TInit["metal"];
    {t, r} = AbsoluteTiming[
        If[n <= 1, TNumValue /@ TCollapse[buildSynth[]],
                   TNumValue /@ TCollapsePar[buildSynth[], n]]];
    xc = Total[r];
    If[baseTime === Null, baseTime = t];
    speedup = If[t > 0, baseTime / t, 0.0];
    Print[StringPadRight[ToString[n], 10],
          StringPadRight[ToString[NumberForm[t, {5, 3}]], 12],
          StringPadRight[ToString[Length[r]], 10],
          StringPadRight[ToString[xc], 8],
          ToString[NumberForm[speedup, {4, 2}]]],
    {n, {1, 2, 4, 8}}
];
