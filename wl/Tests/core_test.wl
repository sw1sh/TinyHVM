(* core_test.wl — Basic verification tests for TinyHVM *)
(* Run: wolframscript -f core_test.wl from the TinyHVM repo root *)

PacletDirectoryLoad[DirectoryName[DirectoryName[$InputFileName]]];
Needs["TinyHVM`"];

$testsPassed = 0;
$testsFailed = 0;

check[name_String, expr_, expected_] := Module[{result = expr},
    If[result === expected || (NumericQ[result] && NumericQ[expected] && Abs[result - expected] < 0.001),
        $testsPassed++;
        Print["  PASS: ", name],
        $testsFailed++;
        Print["  FAIL: ", name, " — got ", result, ", expected ", expected]
    ]
];

checkTrue[name_String, expr_] := check[name, expr, True];

Print["=== TinyHVM Core Tests ===\n"];

(* ── Context ─────────────────────────────────────────────── *)
Print["--- Context ---"];
checkTrue["TInit metal", TInit["metal"]];
check["TTensorCount starts at 0", TTensorCount[], 0];

(* ── Tensor creation ─────────────────────────────────────── *)
Print["\n--- Tensor creation ---"];
t1 = TCreate[{1., 2., 3., 4.}, {2, 2}];
checkTrue["TCreate returns TTensor", MatchQ[t1, _TTensor]];
check["TDimensions", TDimensions[t1], {2, 2}];
check["TTensorCount after create", TTensorCount[], 1];

(* ── Data roundtrip ──────────────────────────────────────── *)
Print["\n--- Data roundtrip ---"];
data = Flatten[Normal[TGet[t1]]];
check["TGet roundtrip [1]", data[[1]], 1.];
check["TGet roundtrip [4]", data[[4]], 4.];

(* ── Elementwise ops ─────────────────────────────────────── *)
Print["\n--- Elementwise ops ---"];
t2 = TCreate[{10., 20., 30., 40.}, {2, 2}];
tAdd = TOp["Add"][t1, t2];
checkTrue["TOp Add returns TTensor", MatchQ[tAdd, _TTensor]];
addResult = Flatten[Normal[TGet[TReduce[tAdd]]]];
check["Add result [1]", addResult[[1]], 11.];
check["Add result [4]", addResult[[4]], 44.];

tMul = TOp["Mul"][t1, t2];
mulResult = Flatten[Normal[TGet[TReduce[tMul]]]];
check["Mul result [1]", mulResult[[1]], 10.];
check["Mul result [4]", mulResult[[4]], 160.];

(* ── Unary ops ───────────────────────────────────────────── *)
Print["\n--- Unary ops ---"];
tNeg = TOp["Neg"][t1];
negResult = Flatten[Normal[TGet[TReduce[tNeg]]]];
check["Neg result [1]", negResult[[1]], -1.];

tSqrt = TOp["Sqrt"][t1];
sqrtResult = Flatten[Normal[TGet[TReduce[tSqrt]]]];
check["Sqrt result [1]", sqrtResult[[1]], 1.];
check["Sqrt result [4]", sqrtResult[[4]], 2.];

(* ── UpValues ────────────────────────────────────────────── *)
Print["\n--- UpValues ---"];
tPlus = t1 + t2;
plusResult = Flatten[Normal[TGet[TReduce[tPlus]]]];
check["Plus UpValue [1]", plusResult[[1]], 11.];

tDot = TCreate[{1., 0., 0., 1.}, {2, 2}];
tMatMul = t1 . tDot;
mmResult = Flatten[Normal[TGet[TReduce[tMatMul]]]];
check["Dot UpValue [1]", mmResult[[1]], 1.];

(* ── Movement ops ────────────────────────────────────────── *)
Print["\n--- Movement ops ---"];
tFlat = TOp["Reshape"][t1, {4}];
check["Reshape dims", TDimensions[tFlat], {4}];

tPerm = TOp["Permute"][t1, {1, 0}];
permResult = Flatten[Normal[TGet[TReduce[tPerm]]]];
check["Permute (transpose) [2]", permResult[[2]], 3.];

(* ── Interaction net primitives ──────────────────────────── *)
Print["\n--- Inet primitives ---"];
(* Build: (λx. x) applied to t1 — should reduce to t1 *)
{lam, var} = TLam[ToTTerm[t1]];
checkTrue["TLam returns pair", Length[{lam, var}] == 2];

appTerm = TApp[lam, t2];
checkTrue["TApp returns TTerm", MatchQ[appTerm, _TTerm]];

(* ── Reset ───────────────────────────────────────────────── *)
Print["\n--- Reset ---"];
nBefore = TTensorCount[];
TReset[1];  (* keep only tensor 0 *)
nAfter = TTensorCount[];
checkTrue["TReset reduces tensor count", nAfter < nBefore];

(* ── Cleanup ─────────────────────────────────────────────── *)
TFree[];

(* ── Summary ─────────────────────────────────────────────── *)
Print["\n=== Results: ", $testsPassed, " passed, ", $testsFailed, " failed ==="];
If[$testsFailed > 0, Exit[1]];
