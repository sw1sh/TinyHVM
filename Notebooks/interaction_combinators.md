# Interaction Combinators: The Theory Behind TinyHVM

This notebook explores the mathematical foundation of TinyHVM's reduction engine: **interaction combinators** (Lafont 1997). We use the `DiagrammaticComputation` package to visualize the interaction rules as string diagrams, then demonstrate them live in TinyHVM.

---

## 1. Setup

```wolfram
PacletDirectoryLoad[FileNameJoin[{NotebookDirectory[], "..", "wl"}]];
PacletDirectoryLoad["/Users/swish/src/wolfram/DiagrammaticComputation"];
Get["TinyHVM`"];
<< Wolfram`DiagrammaticComputation`
AppendTo[$ContextPath, "Wolfram`DiagrammaticComputation`Diagram`Rewriting`Rules`"];
```

```wolfram
TInit[]
```

---

## 2. Three Symbols, Two Rules

Interaction combinators have exactly **three symbols** — each drawn as a node with one principal port (tip) and two auxiliary ports (base):

```wolfram
Row[{
  Diagram[Interpretation["i", "CON"], {x1, SuperStar[x2]}, {p},
    "Shape" -> "RoundedUpsideDownTriangle", "FloatingPorts" -> {True, False},
    "Width" -> 1, "Height" -> 1, ImageSize -> {80, 120}],
  Spacer[30],
  Diagram[Interpretation["j", "DUP"], {x1, SuperStar[x2]}, {p},
    "Shape" -> "RoundedUpsideDownTriangle", "FloatingPorts" -> {True, False},
    "Width" -> 1, "Height" -> 1, "Style" -> Hue[0.709, 0.445, 1], ImageSize -> {80, 120}],
  Spacer[30],
  EraserDiagram[x, ImageSize -> {60, 60}]
}, BaseStyle -> {FontSize -> 16}]
```

The **grey** node is $\delta_i$ (CON when $i = 0$). The **purple** node is $\delta_j$ (DUP when $j \ne 0$). The **circle** is $\varepsilon$ (ERA — eraser).

Interactions happen only when two principal ports meet. There are exactly two cases:

### Annihilation: same symbol ($\delta_i$ meets $\delta_i$)

The nodes cancel, and auxiliary ports cross-connect:

```wolfram
AnnihilationRule[
  Interpretation[i, _], Interpretation[i, _],
  {SuperStar[x1], x2}, {SuperStar[y1], y2}
]
```

### Commutation: different symbols ($\delta_i$ meets $\delta_j$, $i \ne j$)

The nodes pass through each other, duplicating with swapped labels:

```wolfram
CommutationRule[
  Interpretation[i, _], Interpretation[j, _],
  {x1, SuperStar[x2]}, {y1, y2},
  "ShowLabel" -> True
] /; i =!= j
```

These two rules are **computationally universal** — any Turing machine can be encoded using only $\delta$ and $\varepsilon$ nodes.

---

## 3. Lambda Calculus Encoding

The lambda calculus maps to interaction combinators by assigning CON ($\delta_0$) to lambda/application:

```wolfram
With[{
  conDown = Diagram[Interpretation["", "CON"], {a, SuperStar[b]}, {p},
    "Shape" -> "RoundedUpsideDownTriangle", "FloatingPorts" -> {False, False},
    "Width" -> 0.6, "Height" -> 0.6, "ShowLabel" -> False, ImageSize -> {30, 40}],
  conUp = Diagram[Interpretation["", "CON"], {}, {SuperStar[a], b},
    "Shape" -> "RoundedTriangle", "FloatingPorts" -> {False, False},
    "Width" -> 0.6, "Height" -> 0.6, "ShowLabel" -> False, ImageSize -> {30, 40}],
  dupDown = Diagram[Interpretation["", "DUP"], {a, SuperStar[b]}, {p},
    "Shape" -> "RoundedUpsideDownTriangle", "FloatingPorts" -> {False, False},
    "Width" -> 0.6, "Height" -> 0.6, "Style" -> Hue[0.709, 0.445, 1], "ShowLabel" -> False, ImageSize -> {30, 40}],
  dupUp = Diagram[Interpretation["", "DUP"], {}, {SuperStar[a], b},
    "Shape" -> "RoundedTriangle", "FloatingPorts" -> {False, False},
    "Width" -> 0.6, "Height" -> 0.6, "Style" -> Hue[0.709, 0.445, 1], "ShowLabel" -> False, ImageSize -> {30, 40}],
  era = EraserDiagram[x, ImageSize -> {25, 25}]
},
Grid[{
  {Style["Face", Bold], Style["Symbol", Bold], Style["Role", Bold], Style["Shape", Bold]},
  {"LAM", Row[{Subscript["\[Delta]", "0"], " constructor"}], "\[Lambda]x. body", conDown},
  {"APP", Row[{Subscript["\[Delta]", "0"], " destructor"}], "(f a)", conUp},
  {"SUP", Row[{Subscript["\[Delta]", "k"], " constructor"}], "superposition", dupDown},
  {"DP0/DP1", Row[{Subscript["\[Delta]", "k"], " destructor"}], "duplication", dupUp},
  {"ERA", "\[Epsilon]", "discard", era}
}, Frame -> All, Alignment -> {Left, Center}, Spacings -> {2, 1},
  Background -> {None, {LightDarkSwitched[GrayLevel[0.9], GrayLevel[0.2]], None}}]
]
```

The "constructor" and "destructor" faces of the same symbol differ only in **main port polarity**: which direction the principal port points.

### All 8 Lambda Calculus Interaction Rules

```wolfram
Column[
  KeyValueMap[
    Function[{name, rule},
      Labeled[
        Row[List @@ rule, Style[" \[RightArrow] ", 24, Gray]],
        Style[name, Bold, 14], Top
      ]
    ],
    $LambdaInteractionRules
  ],
  Spacer[20]
]
```

---

## 4. Port Polarity

Every node has three ports. The **principal port** is where interactions happen. The two **auxiliary ports** carry data.

The key insight: **constructor** and **destructor** are the same symbol with opposite principal port polarity.

```wolfram
constructor = Diagram[
  Interpretation["i", "CON"], {x, SuperStar[y]}, {},
  "Shape" -> "RoundedUpsideDownTriangle", "FloatingPorts" -> {True, False},
  "Width" -> 1, "Height" -> 1, ImageSize -> {120, 140}
];
destructor = Diagram[
  Interpretation["i", "CON"], {}, {SuperStar[a], b},
  "Shape" -> "RoundedTriangle", "FloatingPorts" -> {False, True},
  "Width" -> 1, "Height" -> 1, ImageSize -> {120, 140}
];
Labeled[
  Row[{constructor, Spacer[60], destructor}],
  Style[Row[{"Same symbol ", Subscript["\[Delta]", "i"], " \[LongDash] opposite principal port polarity"}], Italic, 14],
  Bottom
]
```

This is why TinyHVM uses separate tags (LAM vs APP, SUP vs DP0/DP1) despite them being the same symbol theoretically: the tag encodes both symbol identity AND polarity in a single value, enabling fast switch dispatch.

### Why separate tags beat polarity bits

```wolfram
Grid[{
  {Style["Encoding", Bold], Style["WHNF check", Bold], Style["Dispatch", Bold]},
  {"Separate tags\n(LAM=3, APP=4, ...)", "switch(tag) direct jump", "O(1) jump table"},
  {"Symbol + polarity bit\n(CON + pol=0/1)", "(tag & 0b10) == 0\n+ extra mask", "Nested switch or\nmasked dispatch"}
}, Frame -> All, Alignment -> Left, Spacings -> {2, 1},
  Background -> {None, {LightDarkSwitched[GrayLevel[0.9], GrayLevel[0.2]], None, None}}]
```

---

## 5. Annihilation = Beta Reduction

When APP meets LAM (same symbol CON, principal-to-principal), they **annihilate**. In lambda calculus, this is beta reduction:

$(\lambda x. \text{body})\ \text{arg} \quad\to\quad \text{body}[x := \text{arg}]$

```wolfram
TInit[];
id = TLam[x |-> x];
result = TApp[id, TNum[42]];
{TNumValue[TReduce[result]], "interactions" -> TInteractionCount[]}
```

```wolfram
TInit[];
double = TLam[x |-> TOp2["Add", x, x]];
result = TApp[double, TNum[21]];
{TNumValue[TReduce[result]], "interactions" -> TInteractionCount[]}
```

---

## 6. Commutation = Duplication Through Lambda

When DUP meets LAM (different symbols: $\delta_k$ meets $\delta_0$), they **commute**. The lambda is cloned — one copy for each DUP output:

```wolfram
$LambdaInteractionRules["Dup"]
```

In TinyHVM:

```wolfram
TInit[];
f = TLam[x |-> TOp2["Add", x, TNum[1]]];
{dp0, dp1} = TDup[f];
r0 = TNumValue[TReduce[TApp[dp0, TNum[10]]]];
r1 = TNumValue[TReduce[TApp[dp1, TNum[20]]]];
{r0, r1}
```

---

## 7. DUP-SUP: Same Symbol Annihilation

When DUP meets SUP (same symbol $\delta_k$, same label), they **annihilate**: each DUP projection selects one branch of the superposition.

```wolfram
$LambdaInteractionRules["DupReduce"]
```

```wolfram
TInit[];
s = TSup[0, TNum[10], TNum[20]];
{dp0, dp1} = TDup[0, s];
{TNumValue[TReduce[dp0]], TNumValue[TReduce[dp1]]}
```

---

## 8. APP-SUP: Function Applied to Superposition

When APP meets SUP (different symbols: CON meets DUP), they **commute**: the function is cloned and applied to each branch independently.

```wolfram
TInit[];
f = TLam[x |-> TOp2["Add", x, TNum[1]]];
result = TApp[f, TSup[TNum[3], TNum[7]]];
TSupNumValues[result]
```

Scaling up — apply a function to a superposition of 8 values:

```wolfram
TInit[];
space = Fold[TSup[#2, #1] &, TNum[7], Reverse @ Table[TNum[i], {i, 0, 6}]];
f = TLam[x |-> TOp2["Mul", x, TOp2["Add", x, TNum[1]]]];
result = TApp[f, space];
TNumValue /@ TCollapse[result]
```

---

## 9. Erasure

When ERA meets any node, it propagates through, erasing all connected structure:

```wolfram
$LambdaInteractionRules["Erase"]
```

```wolfram
$LambdaInteractionRules["EraseReduce"]
```

```wolfram
$LambdaInteractionRules["EraseDup"]
```

In TinyHVM, ERA appears when a lambda variable is unused (the slot receives ERA during allocation), or when a DUP projection is never consumed.

---

## 10. ICC Bridges: The Third Symbol

The **Interaction Calculus of Constructions** (ICC) adds a third combinator symbol for types. The **bridge** $\theta x. \text{body}$ is the contra-variant dual of lambda:

- $\lambda x. \text{body}$ — variable $x$ is input provided by APP (caller $\to$ callee)
- $\theta f. \text{body}$ — variable $f$ is the thing being typed, provided by ANN (value $\to$ type)

### The four ICC interaction rules

```wolfram
Grid[{
  {Style["Interaction", Bold], Style["Rule", Bold], Style["Type", Bold]},
  {"APP-LAM", "(\[Lambda]x.f  a) \[RightArrow] f[x := a]", "Annihilation (CON-CON)"},
  {"ANN-BRI", "{v : \[Theta]f.T} \[RightArrow] T[f := v]", "Annihilation (ANN-ANN)"},
  {"APP-BRI", "(\[Theta]f.T  a) \[RightArrow] \[Theta]x.(T[f:=\[Lambda]$k.x] {$k:a})", "Commutation (CON-ANN)"},
  {"ANN-LAM", "{v : \[Lambda]x.T} \[RightArrow] \[Lambda]x.{(v $k) : T[x:=\[Theta]$k.x]}", "Commutation (ANN-CON)"}
}, Frame -> All, Alignment -> Left, Spacings -> {2, 1},
  Background -> {None, {LightDarkSwitched[GrayLevel[0.9], GrayLevel[0.2]], None, None, None, None}}]
```

### Bridge in TinyHVM

```wolfram
TInit[];
bri = TBri[x |-> TOp2["Add", x, TNum[100]]];
ann = TAnn[TNum[5], bri];
TNumValue[TReduce[ann]]
```

```wolfram
TInit[];
ann = TAnn[TNum[42], TNum[0]];
TNumValue[TReduce[ann]]
```

---

## 11. Type Encodings from Bridges

All type constructors are **derived** from bridges, not primitive:

```wolfram
Grid[{
  {Style["Type", Bold], Style["Encoding", Bold], Style["Description", Bold]},
  {"Fun A B", "\[Lambda]A \[Lambda]B \[Theta]f \[Lambda]x {(f {x:A}) : B}", Row[{"Simple function A ", "\[RightArrow]", " B"}]},
  {"All A B", "\[Lambda]A \[Lambda]B \[Theta]f \[Lambda]x {(f {x:A}) : (B x)}", Row[{"\[CapitalPi]-type: \[CapitalPi](x:A). B(x)"}]},
  {"Ind A B", "\[Lambda]A \[Lambda]B \[Theta]f \[Lambda]x {(f {x:A}) : (B f x)}", "Self/inductive type"},
  {"Any", "\[Theta]x x", "Top type (wildcard)"}
}, Frame -> All, Alignment -> Left, Spacings -> {2, 1},
  Background -> {None, {LightDarkSwitched[GrayLevel[0.9], GrayLevel[0.2]], None, None, None, None}}]
```

The bridge $\theta f$ binds the **thing being typed**, so the return type can depend on the value itself. This is what makes self-types and inductive types possible — impossible with just $\lambda$ and application.

---

## 12. Dynamic Labels and Collapse

TinyHVM extends standard interaction combinators with **dynamic labels** and **collapse**:

### Dynamic SUP (DSU)

A superposition whose label is computed at interaction time:

```wolfram
TInit[];
dsu = TDsu[TNum[42], TNum[10], TNum[20]];
result = TReduce[dsu];
{TTermTag[result], TNumValue /@ TCollapse[result]}
```

### Collapse

Collapse reduces a term to WHNF, then recursively flattens all SUP nodes into a flat list of leaves:

```wolfram
TInit[];
nested = TSup[TNum[1], TSup[TNum[10], TNum[20]]];
TNumValue /@ TCollapse[nested]
```

```wolfram
TInit[];
f = TLam[x |-> TOp2["Add", x, TNum[1]]];
space = TSup[TNum[3], TSup[TNum[5], TNum[7]]];
result = TApp[f, space];
TNumValue /@ TCollapse[result]
```

---

## 13. Church Encodings: Data as Interaction Nets

In the interaction calculus, **all data is lambda terms**. There are no built-in booleans, numbers, or lists — only nodes and wires. This is how programs and data become the same thing.

### Booleans

Church booleans are selectors: **true** picks the first argument, **false** picks the second.

$\text{true} = \lambda t.\lambda f.\, t \qquad \text{false} = \lambda t.\lambda f.\, f$

To decode, apply a Church boolean to 1 and 0:

```wolfram
TInit[];
space = TSup[
  TLam[t |-> TLam[f |-> t]],
  TLam[t |-> TLam[f |-> f]]
];
decoded = TApp[TApp[space, TNum[1]], TNum[0]];
TNumValue /@ TCollapse[decoded]
```

### NOT as a Higher-Order Function

NOT flips the arguments: $\text{not} = \lambda b.\lambda t.\lambda f.\, (b\; f\; t)$

Applied to a superposition of both booleans, APP-SUP distributes NOT across both branches simultaneously:

```wolfram
TInit[];
boolNot = TLam[b |-> TLam[t |-> TLam[f |-> TApp[TApp[b, f], t]]]];
space = TSup[
  TLam[t |-> TLam[f |-> t]],
  TLam[t |-> TLam[f |-> f]]
];
result = TApp[boolNot, space];
decoded = TApp[TApp[result, TNum[1]], TNum[0]];
TNumValue /@ TCollapse[decoded]
```

### Natural Numbers

Church naturals apply a successor function $n$ times: $\text{zero} = \lambda s.\lambda z.\, z$, $\text{succ}(n) = \lambda s.\lambda z.\, s\,(n\;s\;z)$.

Since `succ` uses `s` twice, we must explicitly DUP it — interaction nets are linear:

```wolfram
TInit[];
mkZero[] := TLam[s |-> TLam[z |-> z]];
mkSucc[n_TTerm] := TLam[s |-> Module[{s0, s1},
  {s0, s1} = TDup[s];
  TLam[z |-> TApp[s0, TApp[TApp[n, s1], z]]]
]];
three = mkSucc[mkSucc[mkSucc[mkZero[]]]];
decoded = TApp[TApp[three, TLam[x |-> TOp2["Add", x, TNum[1]]]], TNum[0]];
TNumValue[TReduce[decoded]]
```

---

## 14. Program Synthesis via Superposition

Instead of evaluating $N$ candidate programs one at a time, **superpose all candidates into a single term** and evaluate simultaneously. The APP-SUP rule distributes evaluation across all branches with optimal sharing.

### Boolean function synthesis: find XOR

**Problem**: Which of the 16 two-input boolean functions has truth table $\{0, 1, 1, 0\}$?

Superpose all $2^4 = 16$ candidate truth tables. Each output bit is an independent $\text{SUP}(0,1)$. The OP2-SUP rule distributes the equality checks across all 16 combinations simultaneously:

```wolfram
TInit[];
mkBit[] := TSup[TNum[0], TNum[1]];
b00 = mkBit[]; b01 = mkBit[]; b10 = mkBit[]; b11 = mkBit[];
target = {0, 1, 1, 0};
allPass = TOp2["Mul",
  TOp2["Mul", TOp2["Eq", b00, TNum[target[[1]]]], TOp2["Eq", b01, TNum[target[[2]]]]],
  TOp2["Mul", TOp2["Eq", b10, TNum[target[[3]]]], TOp2["Eq", b11, TNum[target[[4]]]]]];
results = TNumValue /@ TCollapse[allPass];
{"candidates" -> Length[results], "matches" -> Total[results]}
```

### 3-variable synthesis: find Majority

Scale up: which of the $2^8 = 256$ three-input boolean functions computes **majority vote** (output 1 iff at least 2 of 3 inputs are 1)?

```wolfram
TInit[];
mkBit[] := TSup[TNum[0], TNum[1]];
bits = Table[mkBit[], 8];
majorityTable = {0, 0, 0, 1, 0, 1, 1, 1};
checks = MapThread[TOp2["Eq", #1, TNum[#2]] &, {bits, majorityTable}];
allPass = Fold[TOp2["Mul", #1, #2] &, First[checks], Rest[checks]];
results = TNumValue /@ TCollapse[allPass];
{"candidates" -> Length[results], "matches" -> Total[results]}
```

256 candidates tested in one reduction. Exactly one match.

### Operation synthesis: search over program structure

Not just constants — search over the **structure** of the program. Which operation transforms $4 \to 12$?

```wolfram
TInit[];
candidates = TSup[
  TLam[x |-> TOp2["Add", x, TNum[3]]],
  TSup[TLam[x |-> TOp2["Mul", x, TNum[3]]],
    TSup[TLam[x |-> TOp2["Sub", x, TNum[3]]],
      TLam[x |-> Module[{x0, x1}, {x0, x1} = TDup[x]; TOp2["Mul", x0, x1]]]]]];
checks = TOp2["Eq", TApp[candidates, TNum[4]], TNum[12]];
names = {"x+3", "x*3", "x-3", Row[{"x", Superscript["", "2"]}]};
results = TNumValue /@ TCollapse[checks];
Pick[names, results, 1]
```

### Composition synthesis: find $g \circ h$

**Problem**: Find primitives $g$ and $h$ from $\{+1, \times 2, \times 3, +5\}$ such that $g(h(3)) = 13$ **and** $g(h(1)) = 7$.

$4 \times 4 = 16$ candidate compositions, all evaluated simultaneously via nested APP-SUP:

```wolfram
TInit[];
mkOps[] := TSup[
  TLam[x |-> TOp2["Add", x, TNum[1]]],
  TSup[TLam[x |-> TOp2["Mul", x, TNum[2]]],
    TSup[TLam[x |-> TOp2["Mul", x, TNum[3]]],
      TLam[x |-> TOp2["Add", x, TNum[5]]]]]];
c1 = TNumValue /@ TCollapse[TOp2["Eq", TApp[mkOps[], TApp[mkOps[], TNum[3]]], TNum[13]]];
c2 = TNumValue /@ TCollapse[TOp2["Eq", TApp[mkOps[], TApp[mkOps[], TNum[1]]], TNum[7]]];
solution = c1 * c2;
opNames = {"+1", "\[Times]2", "\[Times]3", "+5"};
pos = FirstPosition[solution, 1][[1]] - 1;
Row[{"g=", opNames[[Quotient[pos, 4] + 1]], ", h=", opNames[[Mod[pos, 4] + 1]]}]
```

### Sorting by exhaustive permutation search

**Problem**: Sort $[5, 2, 8]$. Superpose all 6 permutations, check which one is non-decreasing.

We use the trick that for values $< 1000$, unsigned subtraction $b - a$ stays small when $b \geq a$ and wraps to $> 10^9$ when $b < a$, so $\lfloor (b-a)/1000 \rfloor = 0$ iff $b \geq a$:

```wolfram
TInit[];
geq[a_, b_] := TOp2["Eq", TOp2["Div", TOp2["Sub", b, a], TNum[1000]], TNum[0]];
pack3[x_, y_, z_] := TOp2["Add", TOp2["Add", TOp2["Mul", x, TNum[10000]], TOp2["Mul", y, TNum[100]]], z];
perms = TSup[
  pack3[TNum[5], TNum[2], TNum[8]],
  TSup[pack3[TNum[5], TNum[8], TNum[2]],
    TSup[pack3[TNum[2], TNum[5], TNum[8]],
      TSup[pack3[TNum[2], TNum[8], TNum[5]],
        TSup[pack3[TNum[8], TNum[2], TNum[5]],
          pack3[TNum[8], TNum[5], TNum[2]]]]]]];
{p0, p1} = TDup[perms]; {p2, p3} = TDup[p1];
x = TOp2["Div", p0, TNum[10000]];
y = TOp2["Mod", TOp2["Div", p2, TNum[100]], TNum[100]];
z = TOp2["Mod", p3, TNum[100]];
sorted = TOp2["Mul", geq[x, y], geq[y, z]];
results = TNumValue /@ TCollapse[sorted];
permNames = {"[5,2,8]", "[5,8,2]", "[2,5,8]", "[2,8,5]", "[8,2,5]", "[8,5,2]"};
Pick[permNames, results, 1]
```

Taelin's NeoGen scales this to real algorithms: **Peano Sort synthesized in 2 seconds** (159x over brute-force enumeration of 5.7M candidates). ADD-CARRY with 16 unknown bits: 65K candidates in 36K total interactions — **sub-1 interaction per candidate**.

---

## 15. Bridges as Types: Specification-Driven Search

Bridges turn types into **executable specifications**. Annotate a candidate with $\{f : \theta g.\, \text{spec}(g)\}$ and ANN-BRI reduction tests it — the spec IS the type.

### Bridge as function spec

The bridge $\theta f.\, (f(3) == 7)$ is a type that accepts only functions mapping $3 \to 7$:

```wolfram
TInit[];
spec = TBri[f |-> TOp2["Eq", TApp[f, TNum[3]], TNum[7]]];
good = TLam[x |-> TOp2["Add", x, TNum[4]]];
bad = TLam[x |-> TOp2["Mul", x, TNum[2]]];
{dp0, dp1} = TDup[spec];
{TNumValue[TReduce[TAnn[good, dp0]]], TNumValue[TReduce[TAnn[bad, dp1]]]}
```

### Typed composition search

Same composition problem as above, but using bridges as the test mechanism. The bridge IS the type — it checks the function against its specification:

```wolfram
TInit[];
mkSpec[input_, target_] := TBri[f |-> TOp2["Eq", TApp[f, TNum[input]], TNum[target]]];
mkCompose[] := Module[{g, h},
  g = TSup[TLam[x |-> TOp2["Add", x, TNum[1]]], TSup[TLam[x |-> TOp2["Mul", x, TNum[2]]],
      TSup[TLam[x |-> TOp2["Mul", x, TNum[3]]], TLam[x |-> TOp2["Add", x, TNum[5]]]]]];
  h = TSup[TLam[x |-> TOp2["Add", x, TNum[1]]], TSup[TLam[x |-> TOp2["Mul", x, TNum[2]]],
      TSup[TLam[x |-> TOp2["Mul", x, TNum[3]]], TLam[x |-> TOp2["Add", x, TNum[5]]]]]];
  TLam[x |-> TApp[g, TApp[h, x]]]];
c1 = TNumValue /@ TCollapse[TAnn[mkCompose[], mkSpec[3, 13]]];
c2 = TNumValue /@ TCollapse[TAnn[mkCompose[], mkSpec[1, 7]]];
solution = c1 * c2;
{"candidates" -> Length[solution], "matches" -> Total[solution]}
```

### From predicates to dependent types

The examples above use bridges as flat predicates: $\theta f.\, \text{check}(f)$. Full ICC goes further — the bridge body can contain **nested annotations**, distributing type checks inward via the ANN-LAM commutation rule:

$\{v : \lambda x.T\} \;\to\; \lambda x.\,\{(v\;\$k) : T[x := \theta\$k.x]\}$

This turns the type $\lambda A.\lambda B.\,\theta f.\,\lambda x.\,\{(f\;\{x:A\}) : B\}$ into a **bidirectional type checker** that verifies $f$ maps $A$-typed inputs to $B$-typed outputs — with no separate type-checking pass. The return type can depend on the input (**Pi type**: $B(x)$) or on the function itself (**self type**: $B(f,x)$), enabling polymorphism and induction as derived concepts rather than axioms.

---

## 16. The Interaction Table

Every pair of tags in TinyHVM corresponds to one of the two universal rules:

```wolfram
symbols = {Row[{"CON (", "\[Lambda]", "/APP)"}], "DUP (SUP/DP)", Row[{"ANN (", "\[Theta]", "/ANN)"}], Row[{"ERA (", "\[Epsilon]", ")"}]};
rules = {
  {"Annihilate\n(beta)", "Commute\n(clone \[Lambda])", "Commute\n(ICC)", "Propagate\n(erase body)"},
  {"Commute\n(clone \[Lambda])", "Annihilate\n(select branch)", "Commute\n(dup type)", "Propagate\n(erase copy)"},
  {"Commute\n(ICC)", "Commute\n(dup type)", "Annihilate\n(type check)", "Propagate\n(erase type)"},
  {"Propagate", "Propagate", "Propagate", "Annihilate\n(nothing)"}
};
Grid[
  Prepend[
    MapThread[Prepend, {rules, Style[#, Bold] & /@ symbols}],
    Prepend[Style[#, Bold] & /@ symbols, ""]
  ],
  Frame -> All, Alignment -> Center, Spacings -> {1.5, 1},
  Background -> {
    {LightDarkSwitched[GrayLevel[0.92], GrayLevel[0.15]], None, None, None, None},
    {LightDarkSwitched[GrayLevel[0.92], GrayLevel[0.15]], None, None, None, None}
  },
  ItemStyle -> Directive[FontSize -> 11]
]
```

The diagonal is always **annihilation** (same symbol). Off-diagonal is always **commutation** (different symbols). ERA is special — it propagates (a degenerate commutation that erases).

---

## 17. Summary

```wolfram
Grid[{
  {Style["Concept", Bold], Style["Theory", Bold], Style["TinyHVM Tags", Bold]},
  {Row[{"Symbol ", Subscript["\[Delta]", "0"], " (CON)"}], "Lambda/Application", "TAG_LAM (3), TAG_APP (4)"},
  {Row[{"Symbol ", Subscript["\[Delta]", "k"], " (DUP)"}], "Superposition/Duplication", "TAG_SUP (5), TAG_DP0 (6), TAG_DP1 (7)"},
  {Row[{"Symbol ", Subscript["\[Delta]", "*"], " (ANN)"}], "Annotation/Bridge", "TAG_ANN (14), TAG_BRI (13)"},
  {Row[{"Symbol ", "\[Epsilon]", " (ERA)"}], "Erasure", "TAG_ERA (2)"},
  {"Same symbol \[RightArrow]", "Annihilate", "Beta, select branch, type check"},
  {"Diff symbol \[RightArrow]", "Commute", Row[{"Clone ", "\[Lambda]", ", distribute APP, dup type"}]}
}, Frame -> All, Alignment -> Left, Spacings -> {2, 1},
  Background -> {None, {LightDarkSwitched[GrayLevel[0.9], GrayLevel[0.2]], None, None, None, None, None, None}}]
```

```wolfram
TFree[]
```
