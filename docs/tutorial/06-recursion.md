# Chapter 6: Recursion — REF and TDefine

Interaction nets are inherently acyclic — there are no cycles in the graph. To express recursion, we use **named definitions** that unfold on demand.

---

## TDefine and TRef

`TDefine[body]` registers a term as a named definition and returns its name (an integer). `TRef[name]` creates a reference that will unfold to the definition's body when reduced.

```wolfram
TInit[];
(* Define a constant *)
name = TDefine[TLam[x |-> TNum[42]]];

(* Reference it *)
ref = TRef[name];
TTermTag[ref]     (* "Ref" *)

(* Reduce: REF unfolds to the definition body *)
result = TReduce[TApp[ref, TNum[0]]];
TNumValue[result]    (* 42 *)
```

---

## REF Expansion

When the reducer encounters a REF node, it **copies** the definition body from a separate "book" heap into the main heap. This is how recursion works — each REF expansion creates a fresh copy:

```wolfram
TInit[];
name = TDefine[TLam[x |-> x]];

TTraceEnable[]; TTraceClear[];
result = TReduce[TApp[TRef[name], TNum[7]]];
traces = TTrace[];
TTraceDisable[];

Select[traces, StringContainsQ[#["RuleName"], "REF"] &]
```

---

## Recursive Definitions with TAllocDef / TSetDef

For recursive definitions, the body needs to reference itself. Use `TAllocDef` to pre-allocate a slot, then `TSetDef` to fill it:

```wolfram
TInit[];

(* Pre-allocate *)
name = TAllocDef[];

(* Build body that references itself *)
body = TLam[n |-> Module[{n0, n1},
    {n0, n1} = TDup[n];
    TIfz[n0,
        TNum[0],                                         (* base case: 0 → 0 *)
        TLam[pred |-> TOp2["Add", pred, TApp[TRef[name], TOp2["Sub", n1, TNum[1]]]]]
    ]
]];

(* Register *)
TSetDef[name, body];

(* Test: sum from 0 to n *)
result = TReduce[TApp[TRef[name], TNum[5]]);
TNumValue[result]    (* 10 = 0+1+2+3+4+5... wait, let's check *)
```

---

## TIfz: Conditional on Zero

`TIfz[counter, zeroCase, succLam]` branches on whether `counter` reduces to 0:
- If counter = 0: returns `zeroCase`
- If counter > 0: applies `succLam` to `counter - 1`

```wolfram
TInit[];
(* If 0 then "yes" else "no" *)
r0 = TReduce[TIfz[TNum[0], TNum[1], TLam[_ |-> TNum[0]]]];
TNumValue[r0]    (* 1 — zero case *)

r1 = TReduce[TIfz[TNum[5], TNum[1], TLam[_ |-> TNum[0]]]];
TNumValue[r1]    (* 0 — nonzero case *)
```

---

## Countdown Example

```wolfram
TInit[];
name = TAllocDef[];

(* countdown(n) = if n == 0 then 0 else countdown(n-1) *)
body = TLam[n |-> Module[{n0, n1},
    {n0, n1} = TDup[n];
    TIfz[n0,
        TNum[0],
        TLam[_ |-> TApp[TRef[name], TOp2["Sub", n1, TNum[1]]]]
    ]
]];
TSetDef[name, body];

(* Run *)
result = TReduce[TApp[TRef[name], TNum[10]]];
TNumValue[result]    (* 0 *)

(* How many interactions? *)
TInteractionCount[]
```

Each recursive call unfolds the REF, creating a fresh copy of the body. The interaction count grows linearly with the depth.

---

## Visualizing Recursive Unfolding

```wolfram
TInit[];
name = TAllocDef[];
body = TLam[n |-> TIfz[n, TNum[0], TLam[_ |-> TApp[TRef[name], TNum[0]]]]];
TSetDef[name, body];

term = TApp[TRef[name], TNum[2]];

(* Step 1: REF expands *)
{t1, i1, _} = TStepReduce[term];
i1["RuleName"]
TINetGraph[t1]

(* Step 2: beta reduction *)
{t2, i2, _} = TStepReduce[t1];
i2["RuleName"]
TINetGraph[t2]
```

Watch the graph grow (REF expansion) and shrink (beta reduction) as the recursion unfolds.

---

## Summary

- `TDefine[body]` registers a definition, returns its name
- `TRef[name]` creates a reference that unfolds on reduction
- `TAllocDef[]` + `TSetDef[name, body]` for recursive (self-referencing) definitions
- REF expansion copies the body fresh each time — no cycles in the graph
- `TIfz[n, zero, succ]` provides conditional branching
- Interaction count = number of REF expansions + reductions = measure of work
