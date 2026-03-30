#!/usr/bin/env wolframscript
(* test_grouping.wls — Groupings as SUP lazy term + grouped collapse + TGroupBy
   Compares TGroupings[elems, f] against built-in Groupings[elems, f]. *)

PacletDirectoryLoad[FileNameJoin[{DirectoryName[$InputFileName], "..", "wl"}]];
Needs["TinyHVM`"];
TInit["metal"];

passed = 0; failed = 0;
check[name_, cond_] := If[TrueQ[cond], passed++; Print["  PASS: ", name],
    failed++; Print["  FAIL: ", name]];

add = TOp2["Add", ##] &;
mul = TOp2["Mul", ##] &;

(* ═══════════════════════════════════════════════════════════════════════════
   1. TGroupings[{a,b,c}, f] vs Groupings[{a,b,c}, f]
   ═══════════════════════════════════════════════════════════════════════════
   Groupings[{a,b,c}, f] = {f[f[a,b],c], f[a,f[b,c]]}  (Catalan(2)=2)
   With f=Add, values {1,2,3}: both groupings give 6.
   With f=Sub: f[f[1,2],3] = (1-2)-3 = -4;  f[1,f[2,3]] = 1-(2-3) = 2 *)

Print["\n=== 1. Groupings[3 elems, Sub] — 2 parenthesizations ==="];
t3 = TGroupings[{TNum[1], TNum[2], TNum[3]}, TOp2["Sub", ##] &];
vals3 = Sort[TNumValue /@ TCollapse[t3]];
(* Interpret as signed: -4 wraps in u32 to 2^32-4 = 4294967292 *)
Print["  collapsed: ", vals3];
(* WL reference: *)
wlRef3 = Sort[Groupings[{1, 2, 3}, Subtract]];
Print["  WL Groupings: ", wlRef3, " = {-4, 2}"];
check["3 elems: count matches Catalan(2)=2", Length[vals3] === 2];

(* ═══════════════════════════════════════════════════════════════════════════
   2. TGroupings[{a,b,c,d}, Add] — Catalan(3)=5 parenthesizations
   ═══════════════════════════════════════════════════════════════════════════
   All groupings of 1+2+3+4 under Add give 10 (addition is associative).
   5 identical results. *)

Print["\n=== 2. Groupings[4 elems, Add] — Catalan(3)=5 ==="];
t4 = TGroupings[TNum /@ {1, 2, 3, 4}, add];
vals4 = TNumValue /@ TCollapse[t4];
Print["  collapsed: ", vals4];
check["4 elems: 5 results (Catalan(3))", Length[vals4] === 5];
check["4 elems Add: all = 10", AllTrue[vals4, # === 10 &]];

(* ═══════════════════════════════════════════════════════════════════════════
   3. TGroupings[{a,b,c,d}, Mul] — 5 distinct structures, same value
   ═══════════════════════════════════════════════════════════════════════════
   1*2*3*4 = 24 regardless of parenthesization (multiplication is assoc). *)

Print["\n=== 3. Groupings[4 elems, Mul] — Catalan(3)=5 ==="];
t4m = TGroupings[TNum /@ {1, 2, 3, 4}, mul];
vals4m = TNumValue /@ TCollapse[t4m];
Print["  collapsed: ", vals4m];
check["4 elems Mul: 5 results", Length[vals4m] === 5];
check["4 elems Mul: all = 24", AllTrue[vals4m, # === 24 &]];

(* ═══════════════════════════════════════════════════════════════════════════
   4. Grouped collapse — recover branch functions from TGroupings
   ═══════════════════════════════════════════════════════════════════════════
   TGroupings[{a,b,c}, f] has 1 label (1 structural choice: split at 1 or 2).
   bf for each result maps that label to 0 (left=split-at-1) or 1 (right=split-at-2). *)

Print["\n=== 4. Grouped collapse on TGroupings[3 elems] ==="];
t3g = TGroupings[TNum /@ {10, 20, 30}, add];
gr3 = TCollapseGrouped[t3g];
Print["  values: ", TNumValue /@ gr3["values"]];
Print["  branch functions: ", gr3["bf"]];
check["grouped 3 elems: 2 results", Length[gr3["values"]] === 2];
check["bf is Association", AllTrue[gr3["bf"], AssociationQ]];
(* Each bf has exactly 1 key (one structural decision) *)
check["bf has 1 label each", AllTrue[gr3["bf"], Length[#] === 1 &]];
(* The label is the same for both results (same structural choice) *)
labels3 = Keys[gr3["bf"][[1]]];
check["both bfs share same label", Keys[gr3["bf"][[2]]] === labels3];
(* branch 0 and branch 1 *)
branches3 = Values /@ gr3["bf"];
check["branches are {0} and {1}", Sort[branches3] === {{0}, {1}}];

(* ═══════════════════════════════════════════════════════════════════════════
   5. TGroupBy on TGroupings[{a,b,c,d}, Add]
   ═══════════════════════════════════════════════════════════════════════════
   4 elements → 3 split points → 2 labels (balanced binary SUP tree over splits).
   Label L1: split-at-1 (left) vs split-at-{2,3} (right).
   Label L2: split-at-2 (left) vs split-at-3 (right) within the right subtree.
   Total: 5 results (2 from left of L1, 3 from right). *)

Print["\n=== 5. TGroupBy on TGroupings[4 elems, Add] ==="];
t5 = TGroupings[TNum /@ {1, 2, 3, 4}, add];
gr5 = TCollapseGrouped[t5];
vals5 = TNumValue /@ gr5["values"];
bfs5 = gr5["bf"];
Print["  values: ", vals5];
Print["  branch functions:"];
Do[Print["    ", vals5[[i]], " -> ", bfs5[[i]]], {i, Length[vals5]}];

(* Find all labels and the top-level one (appears in ALL bfs) *)
allLabels = Union @@ (Keys /@ bfs5);
Print["  all labels: ", allLabels];
check["4 elems: 4 labels (top split tree + recursive)", Length[allLabels] === 4];

topLabel = SelectFirst[allLabels, Function[lab, AllTrue[bfs5, KeyExistsQ[#, lab] &]]];
Print["  top-level label (in all bfs): ", topLabel];

(* Group by top label: split-at-{1,2} vs split-at-3 *)
g1 = TGroupBy[gr5, {topLabel}];
Print["  GroupBy top label: ", Map[TNumValue /@ # &, g1]];
check["GroupBy top: 2 groups (3+2 results)", Sort[Length /@ Values[g1]] === {2, 3}];

(* ═══════════════════════════════════════════════════════════════════════════
   6. Non-associative op: TGroupings[{2,3,4}, Sub] shows distinct values
   ═══════════════════════════════════════════════════════════════════════════
   f[f[2,3],4] = (2-3)-4 = -5 (mod 2^32)
   f[2,f[3,4]] = 2-(3-4) = 3
   Groupings MATTER when the op is non-associative. *)

Print["\n=== 6. Non-assoc op: Groupings[{2,3,4}, Sub] ==="];
sub = TOp2["Sub", ##] &;
t6 = TGroupings[TNum /@ {2, 3, 4}, sub];
gr6 = TCollapseGrouped[t6];
vals6 = TNumValue /@ gr6["values"];
bfs6 = gr6["bf"];
Print["  values: ", vals6];
Print["  branch functions: ", bfs6];
check["non-assoc: 2 distinct values", Length[Union[vals6]] === 2];
(* One is 3, the other is 2^32-5 = 4294967291 *)
check["non-assoc: includes 3", MemberQ[vals6, 3]];
check["non-assoc: includes 2^32-5", MemberQ[vals6, 2^32 - 5]];
(* Each value has a unique bf — we can recover which parenthesization produced it *)
Print["  val=3 was parenthesization: bf=", SelectFirst[Thread[{vals6, bfs6}], First[#] === 3 &][[2]]];
Print["  val=", 2^32-5, " was parenthesization: bf=", SelectFirst[Thread[{vals6, bfs6}], First[#] === 2^32-5 &][[2]]];

(* ═══════════════════════════════════════════════════════════════════════════
   7. Same-label correlation: shared structure
   ═══════════════════════════════════════════════════════════════════════════
   If two parts of a computation share the same label, DUP-SUP annihilation
   forces them to take the same branch — pairwise, not Cartesian. *)

Print["\n=== 7. Same-label pairwise vs different-label Cartesian ==="];
l = TFreshLabel[];
x = TSup[l, TNum[1], TNum[2]];
y = TSup[l, TNum[10], TNum[20]];
tSame = TOp2["Add", x, y];
resSame = Sort[TNumValue /@ TCollapse[tSame]];
Print["  same label: ", resSame, " (pairwise: 1+10=11, 2+20=22)"];
check["same label: 2 results (pairwise)", resSame === {11, 22}];

la = TFreshLabel[]; lb = TFreshLabel[];
x2 = TSup[la, TNum[1], TNum[2]];
y2 = TSup[lb, TNum[10], TNum[20]];
tDiff = TOp2["Add", x2, y2];
resDiff = Sort[TNumValue /@ TCollapse[tDiff]];
Print["  diff labels: ", resDiff, " (Cartesian: all 4 combos)"];
check["diff labels: 4 results (Cartesian)", resDiff === {11, 12, 21, 22}];

(* Grouped collapse shows the correlation *)
grDiff = TCollapseGrouped[tDiff];
Print["  Cartesian bf:"];
Do[Print["    ", TNumValue[grDiff["values"][[i]]], " -> ", grDiff["bf"][[i]]],
    {i, Length[grDiff["values"]]}];
gA = TGroupBy[grDiff, {la}];
Print["  GroupBy la: ", Map[Sort[TNumValue /@ #] &, gA]];
check["GroupBy la: {0}->{11,21}, {1}->{12,22}",
    Sort[TNumValue /@ gA[{0}]] === {11, 21} && Sort[TNumValue /@ gA[{1}]] === {12, 22}];

(* ═══════════════════════════════════════════════════════════════════════════ *)
Print["\n", passed, " passed, ", failed, " failed"];
If[failed > 0, Exit[1]];
