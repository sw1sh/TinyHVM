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

```wolfram
TInit[];
zero = TLam[s |-> TLam[z |-> z]];
succ = TLam[n |-> TLam[s |-> TLam[z |->
  TApp[s, TApp[TApp[n, s], z]]
]]];
two = TApp[succ, TApp[succ, zero]];
three = TApp[succ, two];
decoded = TApp[TApp[three, TLam[x |-> TOp2["Add", x, TNum[1]]]], TNum[0]];
TNumValue[TReduce[decoded]]
```

---

## 14. Program Synthesis via Superposition

This is the core application of interaction combinators to search. Instead of evaluating $N$ candidate programs one at a time, **superpose all candidates into a single term** and evaluate them simultaneously. The APP-SUP rule distributes the evaluation across all branches.

### Finding a constant

**Problem**: Find $c$ such that $c + 3 = 7$.

```wolfram
TInit[];
cSpace = Fold[TSup[#2, #1] &, TNum[7], Reverse @ Table[TNum[i], {i, 0, 6}]];
results = TOp2["Add", cSpace, TNum[3]];
checks = TOp2["Eq", results, TNum[7]];
matches = TNumValue /@ TCollapse[checks];
{"matches" -> matches, "c" -> (FirstPosition[matches, 1][[1]] - 1)}
```

### Finding a Boolean function

**Problem**: Find $f: \{0,1\} \to \{0,1\}$ such that $f(1) = 0$ and $f(0) = 1$.

There are exactly 4 such functions. We superpose all of them and test both constraints:

```wolfram
TInit[];
mkCandidates[] := TSup[
  TLam[x |-> TNum[1]],
  TSup[TLam[x |-> TNum[0]],
    TSup[TLam[x |-> x],
      TLam[x |-> TOp2["Sub", TNum[1], x]]]]];
c1 = TNumValue /@ TCollapse[TOp2["Eq", TApp[mkCandidates[], TNum[1]], TNum[0]]];
c2 = TNumValue /@ TCollapse[TOp2["Eq", TApp[mkCandidates[], TNum[0]], TNum[1]]];
names = {"const1", "const0", "id", "not"};
solution = c1 * c2;
{"c1 (f(1)=0)" -> c1, "c2 (f(0)=1)" -> c2, "winner" -> Pick[names, solution, 1]}
```

### Finding a linear function

**Problem**: Find $(a, b)$ such that $f(x) = ax + b$ with $f(1) = 5$ and $f(3) = 11$.

16 candidates ($a, b \in \{0,1,2,3\}$), evaluated simultaneously:

```wolfram
TInit[];
mkSearch[] := Module[{a, b},
  a = Fold[TSup[#2, #1] &, TNum[3], Reverse @ Table[TNum[i], {i, 0, 2}]];
  b = Fold[TSup[#2, #1] &, TNum[3], Reverse @ Table[TNum[i], {i, 0, 2}]];
  TLam[x |-> TOp2["Add", TOp2["Mul", a, x], b]]
];
c1 = TNumValue /@ TCollapse[TOp2["Eq", TApp[mkSearch[], TNum[1]], TNum[5]]];
c2 = TNumValue /@ TCollapse[TOp2["Eq", TApp[mkSearch[], TNum[3]], TNum[11]]];
solution = c1 * c2;
pos = FirstPosition[solution, 1][[1]] - 1;
{"a" -> Quotient[pos, 4], "b" -> Mod[pos, 4], "checks" -> solution}
```

### SupGen Benchmarks (Taelin, 2024–2026)

These search patterns scale dramatically via **optimal sharing** — common subcomputations across candidates are evaluated once:

```wolfram
Grid[{
  {Style["Problem", Bold], Style["Space", Bold], Style["Brute Force", Bold], Style["SupGen", Bold], Style["Speedup", Bold]},
  {"ADD-CARRY (16 bits)", Superscript["2", "16"], "262M interactions", "36K interactions", Row[{"7,277", "\[Times]"}]},
  {"XOR-XNOR discovery", "\[Dash]", "2.8s (Haskell)", "0.0085s (HVM)", Row[{"330", "\[Times]"}]},
  {Row[{"\[Lambda]", "-equation solving"}], "\[Dash]", "0.992s (Haskell)", "0.0011s (HVM)", Row[{"862", "\[Times]"}]},
  {"Peano Sort (pos 5.7M)", "huge", "5m17s enumerate", "2s (NeoGen)", Row[{"159", "\[Times]"}]}
}, Frame -> All, Alignment -> Left, Spacings -> {2, 0.8},
  Background -> {None, {LightDarkSwitched[GrayLevel[0.9], GrayLevel[0.2]], None}},
  ItemStyle -> Directive[FontSize -> 11]]
```

The cost per candidate in the ADD-CARRY benchmark is **sub-1 interaction** — 65,536 candidates cost only 36K total interactions. This is not parallelism; it is algorithmic sharing.

---

## 15. Dependent Types: The ICC Vision

The program synthesis above is **untyped** — every candidate is evaluated, even nonsensical ones. The **Interaction Calculus of Constructions** (ICC) adds type-directed pruning: ill-typed candidates are erased *before evaluation*, via the same interaction rules.

### Types from bridges

All type constructors are **derived** from the bridge primitive $\theta$:

$\text{Fun}\;A\;B = \theta f.\,\lambda x.\,\{(f\;\{x : A\}) : B\}$

$\text{All}\;A\;B = \theta f.\,\lambda x.\,\{(f\;\{x : A\}) : (B\;x)\}$

$\text{Ind}\;A\;B = \theta f.\,\lambda x.\,\{(f\;\{x : A\}) : (B\;f\;x)\}$

The key difference: in **All** (Pi type), the return type $B$ depends on the *argument* $x$. In **Ind** (self type), the return type depends on both the *function itself* $f$ and the argument. This is impossible without bridges — the function being typed has no name in scope with just $\lambda$.

### Type checking as reduction

When you annotate a value with a type $\{v : T\}$, two things can happen:

**ANN-BRI (annihilation)**: If $T = \theta f.\, \text{body}$, the bridge binds the value: $\{v : \theta f.T\} \to T[f := v]$. This is the type check *succeeding* — the value flows into the type body for further inspection.

```wolfram
TInit[];
anyType = TBri[x |-> x];
ann = TAnn[TNum[42], anyType];
TNumValue[TReduce[ann]]
```

```wolfram
TInit[];
checkPositive = TBri[v |-> TOp2["Mul", v, v]];
ann = TAnn[TNum[7], checkPositive];
TNumValue[TReduce[ann]]
```

**ANN-LAM (commutation)**: If $T = \lambda x.\, \text{body}$, the annotation distributes inward — checking each part of the function recursively:

$\{v : \lambda x.T\} \;\to\; \lambda x.\,\{(v\;\$k) : T[x := \theta\$k.x]\}$

This is **bidirectional type inference** emerging from interaction rules. No separate type checker needed.

### Typed search: pruning candidates

In a typed SupGen search, each candidate is annotated with its required type. When reduction encounters a type mismatch (an annotation that can't simplify), the branch gets stuck and is erased:

```
candidates = SUP(f1, f2, f3, f4)
typed = {candidates : (Nat → Nat)}       ← annotation flows inward via ANN-LAM
                                            ill-typed branches → stuck → ERA
survivors = collapse(typed)               ← only well-typed functions remain
```

The type system acts as a **compile-time filter** on the search space. For program ASTs, most random trees are garbage — they divide by zero, produce wrong shapes, or violate invariants. Types prune them for free.

### Self types: what makes ICC unique

Standard type theory (Calculus of Constructions) cannot express inductive types as a primitive — it needs axioms. ICC's **Ind** type encodes induction directly:

$\text{Ind}\;A\;B = \theta f.\,\lambda x.\,\{(f\;\{x : A\}) : (B\;f\;x)\}$

The $f$ in $B\;f\;x$ is the function being typed — it appears in its own return type. This self-reference is what makes inductive proofs possible: a recursive function's type can express "I work correctly on all recursive calls."

Church-encoded natural numbers, booleans, lists, and equality proofs are all instances of this pattern. SupGen can search over programs of these types, synthesizing functions that are *provably correct by construction*.

```wolfram
Grid[{
  {Style["Type", Bold], Style["ICC Encoding", Bold], Style["What it enables", Bold]},
  {"Any", Row[{"\[Theta]", "x. x"}], "Top type \[Dash] accepts anything"},
  {Row[{"A ", "\[RightArrow]", " B"}], Row[{"\[Theta]", "f. \[Lambda]x. {(f {x:A}) : B}"}], "Simple function types"},
  {Row[{"\[CapitalPi](x:A). B(x)"}], Row[{"\[Theta]", "f. \[Lambda]x. {(f {x:A}) : (B x)}"}], "Dependent functions (polymorphism)"},
  {Row[{"\[CapitalPi]f(x:A). B(f,x)"}], Row[{"\[Theta]", "f. \[Lambda]x. {(f {x:A}) : (B f x)}"}], "Self types (induction, recursion)"},
  {Row[{"\[CapitalSigma](x:A). B(x)"}], Row[{"\[Theta]", "p. (p \[Lambda]fst.\[Lambda]snd.\[Lambda]p(p {fst:A} {snd:(B fst)}))"}], "Dependent pairs (existentials)"}
}, Frame -> All, Alignment -> Left, Spacings -> {2, 0.8},
  Background -> {None, {LightDarkSwitched[GrayLevel[0.9], GrayLevel[0.2]], None}},
  ItemStyle -> Directive[FontSize -> 11]]
```

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
