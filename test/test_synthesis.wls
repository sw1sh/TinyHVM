#!/usr/bin/env wolframscript
(* test_synthesis.wls — FindBooleanAlternative with both methods *)
PacletDirectoryLoad[FileNameJoin[{DirectoryName[$InputFileName], "..", "wl"}]];
Needs["TinyHVM`"];

(* === 1. Brute-force method === *)
Print["=== Brute-force method ===\n"];

(* AND from OR/NOT — small, both methods find it quickly *)
{t1, r1} = AbsoluteTiming[
    FindBooleanAlternative[And[a, b], {Or, Not}]];
Print["AND from {Or,Not}: ", InputForm[r1]];
Print["  Correct: ",
    BooleanTable[r1, {a, b}] === BooleanTable[And[a, b], {a, b}]];
Print["  Time: ", t1, "s"];

(* === 2. TinyHVM method === *)
Print["\n=== TinyHVM method ===\n"];

(* XOR from AND/NOT — the showcase: brute-force can't find this easily *)
{t2, r2} = AbsoluteTiming[
    FindBooleanAlternative[Xor[a, b], {And, Not}, All,
        "MaxSize" -> 4, Method -> "TinyHVM"]];
Print["XOR from {And,Not}: ", Length[r2], " circuits in ", t2, "s"];
Do[Print["  ", InputForm[c]], {c, Take[r2, UpTo[4]]}];
If[Length[r2] > 4, Print["  ..."]];
Print["  All correct: ",
    AllTrue[r2, BooleanTable[#, {a, b}] === BooleanTable[Xor[a, b], {a, b}] &]];

(* AND from OR/NOT — compare with brute-force *)
r3 = FindBooleanAlternative[And[a, b], {Or, Not}, Automatic,
    Method -> "TinyHVM"];
Print["\nAND from {Or,Not}: ", InputForm[r3]];
Print["  Correct: ",
    BooleanTable[r3, {a, b}] === BooleanTable[And[a, b], {a, b}]];

(* OR from AND/NOT *)
r4 = FindBooleanAlternative[Or[a, b], {And, Not}, All,
    "MaxSize" -> 3, Method -> "TinyHVM"];
Print["\nOR from {And,Not}: ", Length[r4], " circuits"];
Do[Print["  ", InputForm[c]], {c, Take[r4, UpTo[4]]}];
If[Length[r4] > 4, Print["  ..."]];
Print["  All correct: ",
    AllTrue[r4, BooleanTable[#, {a, b}] === BooleanTable[Or[a, b], {a, b}] &]];

Print["\nDone."];
