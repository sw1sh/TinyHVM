# Chapter 7: Superposition — Parallel Evaluation and Synthesis

SUP isn't just for duplication bookkeeping — it's a computational primitive. By superposing multiple candidates into a single term, you evaluate **all of them simultaneously** with optimal sharing.

---

## SUP as "All at Once"

A superposition `TSup[a, b]` represents both `a` and `b` existing at the same time:

```wolfram
TInit[];
s = TSup[TNum[3], TNum[7]];
TNumValue /@ TCollapse[s]    (* {3, 7} *)
```

`TCollapse` flattens all branches into a list. But the power comes from applying functions to superpositions.

---

## APP-SUP Commutation

When you apply a function to a superposition, the function is cloned and applied to each branch independently:

```wolfram
TInit[];
f = TLam[x |-> TOp2["Add", x, TNum[1]]];
result = TApp[f, TSup[TNum[3], TNum[7]]];
TNumValue /@ TCollapse[result]    (* {4, 8} *)
```

The APP-SUP rule distributed `f` across both branches. One function application, two results.

---

## Scaling Up

Apply to 8 values simultaneously:

```wolfram
TInit[];
space = Fold[TSup[#2, #1] &, TNum[7], Reverse @ Table[TNum[i], {i, 0, 6}]];
f = TLam[x |-> TOp2["Mul", x, TOp2["Add", x, TNum[1]]]];    (* x*(x+1) *)
result = TApp[f, space];
TNumValue /@ TCollapse[result]
(* {0, 2, 6, 12, 20, 30, 42, 56} *)
```

8 candidates, evaluated in one reduction. The shared parts of the computation (the function structure) are not duplicated — only the parts that differ.

---

## Program Synthesis: Find XOR

**Problem**: Which of the 16 two-input boolean functions has truth table `{0, 1, 1, 0}`?

Superpose all 16 candidates. Each output bit is a `SUP(0, 1)`:

```wolfram
TInit[];
mkBit[] := TSup[TNum[0], TNum[1]];
b00 = mkBit[]; b01 = mkBit[]; b10 = mkBit[]; b11 = mkBit[];
target = {0, 1, 1, 0};
allPass = TOp2["Mul",
    TOp2["Mul", TOp2["Eq", b00, TNum[target[[1]]]], TOp2["Eq", b01, TNum[target[[2]]]]],
    TOp2["Mul", TOp2["Eq", b10, TNum[target[[3]]]], TOp2["Eq", b11, TNum[target[[4]]]]]
];
results = TNumValue /@ TCollapse[allPass];
{"candidates" -> Length[results], "matches" -> Total[results]}
```

16 candidates tested in one reduction. Exactly one match (XOR).

---

## Searching Over Program Structure

Not just constants — search over the **structure** of programs:

```wolfram
TInit[];
candidates = TSup[
    TLam[x |-> TOp2["Add", x, TNum[3]]],
    TSup[TLam[x |-> TOp2["Mul", x, TNum[3]]],
        TSup[TLam[x |-> TOp2["Sub", x, TNum[3]]],
            TLam[x |-> Module[{x0, x1}, {x0, x1} = TDup[x]; TOp2["Mul", x0, x1]]]
        ]
    ]
];
checks = TOp2["Eq", TApp[candidates, TNum[4]], TNum[12]];
names = {"x+3", "x*3", "x-3", "x^2"};
results = TNumValue /@ TCollapse[checks];
Pick[names, results, 1]    (* {"x*3"} *)
```

---

## Composition Synthesis

Find `g` and `h` from `{+1, *2, *3, +5}` such that `g(h(3)) = 13` and `g(h(1)) = 7`:

```wolfram
TInit[];
mkOps[] := TSup[
    TLam[x |-> TOp2["Add", x, TNum[1]]],
    TSup[TLam[x |-> TOp2["Mul", x, TNum[2]]],
        TSup[TLam[x |-> TOp2["Mul", x, TNum[3]]],
            TLam[x |-> TOp2["Add", x, TNum[5]]]
        ]
    ]
];
c1 = TNumValue /@ TCollapse[TOp2["Eq", TApp[mkOps[], TApp[mkOps[], TNum[3]]], TNum[13]]];
c2 = TNumValue /@ TCollapse[TOp2["Eq", TApp[mkOps[], TApp[mkOps[], TNum[1]]], TNum[7]]];
solution = c1 * c2;
opNames = {"+1", "*2", "*3", "+5"};
pos = FirstPosition[solution, 1][[1]] - 1;
Row[{"g=", opNames[[Quotient[pos, 4] + 1]], ", h=", opNames[[Mod[pos, 4] + 1]]}]
```

16 candidate compositions, all evaluated simultaneously.

---

## TCollapse and TCollapseGrouped

`TCollapse[term]` reduces and flattens all SUP nodes into a list:

```wolfram
TInit[];
nested = TSup[TNum[1], TSup[TNum[2], TNum[3]]];
TNumValue /@ TCollapse[nested]    (* {1, 2, 3} *)
```

`TCollapseGrouped[term]` also returns **branch functions** — which SUP label choices led to each result:

```wolfram
TInit[];
s = TSup[TNum[10], TNum[20]];
gr = TCollapseGrouped[s];
gr["values"]         (* the collapsed values *)
TBranchFunction[gr, 1]    (* label -> branch choices for result 1 *)
```

---

## Why This Works

The APP-SUP commutation rule is the engine:

```
(f  {a, b})  →  !{a', b'} = DUP(f);  {(a' a), (b' b)}
```

The function `f` is duplicated (via DUP-SUP machinery), and applied to each branch. But crucially, **shared sub-expressions are not re-evaluated** — the DUP only copies what's needed, and identical subterms cancel via DUP-SUP annihilation.

This is Lévy-optimal reduction: no redundant work.

---

## Summary

- `TSup[a, b]` superposes two values
- APP-SUP distributes computation across branches
- `TCollapse[t]` extracts all branches as a flat list
- Superposition enables **parallel evaluation** of exponentially many candidates
- Search over values (constants), structures (programs), and compositions
- Optimal sharing means sublinear work per candidate
