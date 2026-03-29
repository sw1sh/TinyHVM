# SupGen: Superposition-Based Program Synthesis in TinyHVM

TinyHVM's interaction net runtime natively supports **superpositions** — a mechanism that lets you evaluate multiple candidate programs simultaneously with optimal sharing of common subcomputations. This notebook demonstrates a DSL for program synthesis using SUP nodes.

---

## 1. Setup

```wolfram
PacletDirectoryLoad[FileNameJoin[{NotebookDirectory[], "..", "wl"}]];
Get["TinyHVM`"]
TInit["metal"]
```

---

## 2. Superposition Basics

A **superposition** `TSup[a, b]` represents "both `a` and `b` simultaneously." It's not a pair — it's a quantum-like choice that propagates through computation.

### Creating and projecting superpositions

```wolfram
(* Create integer terms *)
a = TNum[42];
b = TNum[99];

(* Superpose them: "both 42 and 99" *)
s = TSup[a, b];

(* DUP projects each branch *)
{dp0, dp1} = TDup[s];
{TNumValue[TReduce[dp0]], TNumValue[TReduce[dp1]]}
```

### Nested superpositions encode search spaces

```wolfram
(* 4-element search space: {10, 20, 30, 40} *)
space = TSup[TNum[10], TSup[TNum[20], TSup[TNum[30], TNum[40]]]];

(* Extract all branches *)
TSupNumValues[space]
```

---

## 3. The Key Rule: APP-SUP Distribution

When you apply a function to a superposition, the **APP-SUP rule** distributes the application across all branches:

```
APP(f, SUP(a, b))  →  SUP(APP(f, a), APP(f, b))
```

The argument `f` is shared — if it's the same function applied to all branches, the common structure is computed once. This is the core of SupGen.

### Lambda applied to a superposition

```wolfram
(* A function: double its input *)
double = TLam[x |-> TOp2["Add", x, x]];

(* Apply to a superposition of {3, 7} *)
input = TSup[TNum[3], TNum[7]];
result = TApp[double, input];

(* Both branches evaluate simultaneously *)
TSupNumValues[result]
(* Expected: {6, 14} *)
```

### Function superposition: multiple candidate functions

```wolfram
(* SUP of two functions: f(x) = x+1 vs f(x) = x*2 *)
f1 = TLam[x |-> TOp2["Add", x, TNum[1]]];
f2 = TLam[x |-> TOp2["Mul", x, TNum[2]]];
candidates = TSup[f1, f2];

(* Apply both functions to 5 *)
result = TApp[candidates, TNum[5]];
TSupNumValues[result]
(* Expected: {6, 10} — f1(5)=6, f2(5)=10 *)
```

---

## 4. Integer Arithmetic with OP2

`TOp2[op, x, y]` performs integer arithmetic on `TNum` values. Operations: `"Add"`, `"Sub"`, `"Mul"`, `"Div"`, `"Eq"`, `"Mod"`.

OP2 also distributes across SUPs — `OP2(+, SUP(a,b), y)` becomes `SUP(OP2(+,a,y), OP2(+,b,y))`.

```wolfram
(* Arithmetic on superposed values *)
x = TSup[TNum[3], TNum[7]];

(* Add 10 to both branches *)
sum = TOp2["Add", x, TNum[10]];
TSupNumValues[sum]
(* Expected: {13, 17} *)

(* Equality check: which branch equals 17? *)
check = TOp2["Eq", sum, TNum[17]];
TSupNumValues[check]
(* Expected: {0, 1} — second branch matches *)
```

---

## 5. Program Synthesis: Finding f(x) = x + c

**Problem**: Given f(2) = 7, find the constant `c` such that f(x) = x + c.

**Search space**: c ∈ {0, 1, 2, 3, 4, 5, 6, 7} — 8 candidates.

**Method**: Superpose all 8 candidates, evaluate simultaneously, check which matches.

```wolfram
(* Build search space: c ∈ {0..7} as a nested SUP tree *)
cSpace = Fold[TSup[#2, #1]&, TNum[7], Reverse @ Table[TNum[i], {i, 0, 6}]];

(* Verify: all 8 values present *)
TSupNumValues[cSpace]

(* Build candidate function: f(x) = x + c *)
f = TLam[x |-> TOp2["Add", x, cSpace]];

(* Apply to test input x = 2 *)
outputs = TApp[f, TNum[2]];

(* All 8 outputs computed simultaneously *)
allOutputs = TSupNumValues[outputs]
(* Expected: {2, 3, 4, 5, 6, 7, 8, 9} *)
```

```wolfram
(* Which candidate gives f(2) = 7? *)
checks = TOp2["Eq", outputs, TNum[7]];
matches = TSupNumValues[checks]
(* Expected: {0, 0, 0, 0, 0, 1, 0, 0} — c=5 wins! *)

Position[matches, 1] - 1
(* Expected: {{5}} — confirming c = 5 *)
```

---

## 6. Multi-Constraint Search: f(x) = ax + b

**Problem**: Find (a, b) such that f(1) = 5 and f(3) = 11.
(Solution: a = 3, b = 2)

**Search space**: a ∈ {0..3}, b ∈ {0..3} — 16 candidates.

```wolfram
(* Build 2D search space *)
aSpace = Fold[TSup[#2, #1]&, TNum[3], Reverse @ Table[TNum[i], {i, 0, 2}]];
bSpace = Fold[TSup[#2, #1]&, TNum[3], Reverse @ Table[TNum[i], {i, 0, 2}]];

(* Candidate function: f(x) = a*x + b *)
f = TLam[x |-> TOp2["Add", TOp2["Mul", aSpace, x], bSpace]];

(* Test constraint 1: f(1) = 5 *)
out1 = TApp[f, TNum[1]];
check1 = TOp2["Eq", out1, TNum[5]];

(* Extract results — 16 candidates (4×4 grid) *)
c1 = TSupNumValues[check1]
```

```wolfram
(* Build fresh search space for second constraint *)
(* (Each TApp consumes the lambda — rebuild for constraint 2) *)
aSpace2 = Fold[TSup[#2, #1]&, TNum[3], Reverse @ Table[TNum[i], {i, 0, 2}]];
bSpace2 = Fold[TSup[#2, #1]&, TNum[3], Reverse @ Table[TNum[i], {i, 0, 2}]];

g = TLam[x |-> TOp2["Add", TOp2["Mul", aSpace2, x], bSpace2]];

(* Test constraint 2: f(3) = 11 *)
out2 = TApp[g, TNum[3]];
check2 = TOp2["Eq", out2, TNum[11]];

c2 = TSupNumValues[check2]
```

```wolfram
(* AND both constraints — both must be 1 *)
both = c1 * c2;
solutions = Position[both, 1]

(* Decode: position → (a, b) pair *)
(* The search tree is: for each a value, nest all b values *)
(* So position p → a = Quotient[p-1, 4], b = Mod[p-1, 4] *)
{aVal, bVal} = QuotientRemainder[solutions[[1, 1]] - 1, 4]
(* Expected: a = 3, b = 2 *)
```

---

## 7. Interaction Counting: The Sharing Advantage

The power of superpositions is **computation sharing**. When multiple candidates share common subexpressions, they're computed once. Let's measure this.

```wolfram
(* Method 1: Superposed search — all 8 candidates at once *)
TInit["metal"];
i0 = TInteractionCount[];

c = Fold[TSup[#2, #1]&, TNum[7], Reverse @ Table[TNum[i], {i, 0, 6}]];
f = TLam[x |-> TOp2["Add", x, c]];
result = TApp[f, TNum[2]];
reduced = TReduce[result];

supInteractions = TInteractionCount[] - i0;
Print["Superposed: ", supInteractions, " interactions for 8 candidates"]
```

```wolfram
(* Method 2: Sequential — evaluate each candidate independently *)
TInit["metal"];
i0 = TInteractionCount[];

seqResults = Table[
    Module[{fi, ri},
        fi = TLam[x |-> TOp2["Add", x, TNum[c]]];
        ri = TApp[fi, TNum[2]];
        TNumValue[TReduce[ri]]
    ],
    {c, 0, 7}
];

seqInteractions = TInteractionCount[] - i0;
Print["Sequential: ", seqInteractions, " interactions for 8 candidates"]
Print["Speedup: ", N[seqInteractions / supInteractions], "x"]
```

---

## 8. Tracing Superposition Evaluation

Use interaction tracing to see APP-SUP and OP2-SUP rules firing in real time.

```wolfram
TInit["metal"];
TTraceEnable[];
TTraceClear[];

(* Simple case: add 10 to SUP(3, 7) *)
s = TSup[TNum[3], TNum[7]];
result = TOp2["Add", s, TNum[10]];
reduced = TReduce[result];

(* View the trace *)
trace = TTrace[];
Dataset[trace]
```

```wolfram
(* The trace shows:
   1. OP2 enters, reduces left arg → finds SUP(3,7)
   2. OP2-SUP fires: distributes to SUP(OP2(+,3,10), OP2(+,7,10))
   3. Each inner OP2 fires: 3+10=13, 7+10=17
   4. Result: SUP(13, 17)
*)
TSupNumValues[reduced]
TTraceDisable[];
```

---

## 9. Building a Search DSL

Let's wrap the search pattern into reusable combinators.

```wolfram
(* SearchSpace: list → nested SUP *)
SearchSpace[values_List] := Fold[TSup[#2, #1]&, TNum[Last[values]],
    Reverse @ Most[TNum /@ values]];

(* SearchEval: apply superposed function to input, get all results *)
SearchEval[f_TTerm, input_Integer] := TSupNumValues[TApp[f, TNum[input]]];

(* SearchFind: find which candidates satisfy a predicate *)
SearchFind[f_TTerm, input_Integer, target_Integer] := Module[
    {result, checks},
    result = TApp[f, TNum[input]];
    checks = TOp2["Eq", result, TNum[target]];
    TSupNumValues[checks]
];
```

```wolfram
(* Example: search for c in f(x) = x + c where f(10) = 42 *)
TInit["metal"];

c = SearchSpace[Range[0, 63]];  (* 64 candidates! *)
f = TLam[x |-> TOp2["Add", x, c]];

(* Evaluate all 64 candidates at x=10 *)
allResults = SearchEval[f, 10];
Print["All outputs: ", allResults]

(* Find which c gives f(10) = 42 *)
matches = SearchFind[f, 10, 42];
solution = FirstPosition[matches, 1][[1]] - 1
Print["Solution: c = ", solution]
(* Expected: c = 32 *)
```

---

## 10. Toward Kernel Optimization

The same SUP mechanism can search over **kernel schedule spaces**:

- **Tile sizes**: `TSup[16, TSup[32, 64]]` — 3 candidates
- **Vectorization**: `TSup[1, TSup[2, 4]]` — 3 unroll factors
- **Parallelism**: `TSup["block", "thread"]` — 2 strategies

With a symbolic cost model as a pure function in the interaction calculus, all combinations evaluate simultaneously with optimal sharing. The top-K candidates from symbolic evaluation then get hardware-timed.

This is the path from lambda-calculus SupGen to **GPU kernel autotuning** — the subject of `resources/supgen_kernel_search.md`.

```wolfram
TFree[]
```
