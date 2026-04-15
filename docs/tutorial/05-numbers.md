# Chapter 5: Numbers — NUM and OP2

So far we've seen structural nodes (LAM, APP, SUP, DUP, ERA). Now we add **data** — numbers and arithmetic.

---

## NUM: Native Numbers

`TNum[n]` creates a TAG_NUM term. The number is stored directly in the VAL field:

```wolfram
TInit[];
n = TNum[42];
TTermTag[n]          (* "Num" *)
TNumValue[n]         (* 42 *)
TINetGraph[n]        (* single NUM node *)
```

NUM is a **leaf** — it has no children on the heap. The value lives inside the 64-bit term itself.

---

## OP2: Binary Operations

`TOp2[op, x, y]` creates a binary operation node. The operation name goes in the EXT field:

```wolfram
TInit[];
expr = TOp2["Add", TNum[3], TNum[4]];
TINetGraph[expr]     (* OP2 node with two NUM children *)
```

Available operations: `"Add"`, `"Sub"`, `"Mul"`, `"Div"`, `"Eq"`, `"Mod"`.

---

## OP2-NUM Interaction

When OP2 has both arguments reduced to NUM, it fires:

```wolfram
TInit[];
expr = TOp2["Add", TNum[3], TNum[4]];

{result, interaction, heap} = TStepReduce[expr];
interaction["RuleName"]      (* "OP2-NUM Compute" *)
TNumValue[result]            (* 7 *)
```

The OP2 and both NUM nodes disappear, replaced by a single NUM with the result.

---

## Expression Trees

Build compound expressions by nesting:

```wolfram
TInit[];
(* (3 + 4) * (10 - 2) *)
expr = TOp2["Mul",
    TOp2["Add", TNum[3], TNum[4]],
    TOp2["Sub", TNum[10], TNum[2]]
];
TINetGraph[expr]
```

Step through the reduction:

```wolfram
{t1, i1, _} = TStepReduce[expr];
i1["RuleName"]     (* one OP2-NUM fires *)
TINetGraph[t1]

{t2, i2, _} = TStepReduce[t1];
i2["RuleName"]     (* another OP2-NUM fires *)
TINetGraph[t2]

{t3, i3, _} = TStepReduce[t2];
i3["RuleName"]     (* final multiplication *)
TNumValue[t3]      (* 56 *)
```

---

## OP2-SUP Commutation

When OP2 meets a SUP (different symbol), it commutes — the operation distributes:

```wolfram
TInit[];
(* Add 1 to both branches of a superposition *)
expr = TOp2["Add", TSup[TNum[10], TNum[20]], TNum[1]];

TTraceEnable[]; TTraceClear[];
result = TReduce[expr];
traces = TTrace[];
TTraceDisable[];

(* Find the commutation *)
Select[traces, StringContainsQ[#["RuleName"], "Commutation"] &]

(* Result: SUP of two sums *)
TNumValue /@ TCollapse[result]    (* {11, 21} *)
```

---

## Arithmetic with Lambda

Combine lambdas and numbers for reusable functions:

```wolfram
TInit[];
square = TLam[x |-> Module[{x0, x1}, {x0, x1} = TDup[x]; TOp2["Mul", x0, x1]]];
result = TReduce[TApp[square, TNum[7]]];
TNumValue[result]    (* 49 *)
```

Note: since `x` is used twice in `x * x`, we must explicitly DUP it — interaction nets are linear.

---

## Church Numerals vs Native Numbers

In pure lambda calculus, numbers are encoded as functions (Church numerals):

```wolfram
TInit[];
(* Church numeral 3: applies f three times *)
mkZero[] := TLam[s |-> TLam[z |-> z]];
mkSucc[n_TTerm] := TLam[s |-> Module[{s0, s1},
    {s0, s1} = TDup[s];
    TLam[z |-> TApp[s0, TApp[TApp[n, s1], z]]]
]];
three = mkSucc[mkSucc[mkSucc[mkZero[]]]];

(* Decode: apply successor 3 times to 0 *)
decoded = TReduce[TApp[TApp[three, TLam[x |-> TOp2["Add", x, TNum[1]]]], TNum[0]]];
TNumValue[decoded]   (* 3 *)
```

Church numerals work but require many interactions. Native NUM/OP2 is a shortcut — the same result in one interaction instead of many.

---

## Summary

- `TNum[n]` — native integer (leaf node, value in VAL field)
- `TOp2[op, x, y]` — binary operation (fires when both args are NUM)
- OP2-NUM interaction: compute and return result as NUM
- OP2-SUP commutation: distribute operation across branches
- Native numbers are O(1) per operation; Church numerals are O(n)
