# Chapter 3: Interactions — Step-by-Step Execution

Computation in an interaction net happens through **interactions** — two nodes meet at their principal ports and rewrite according to a rule. In this chapter we slow things down and watch each interaction fire one at a time.

---

## The Two Universal Rules

Every interaction in the system is an instance of one of two rules:

1. **Annihilation** — same symbol meets same symbol. They cancel, and their auxiliary ports cross-connect.
2. **Commutation** — different symbols meet. They pass through each other, duplicating with swapped connections.

That's it. Every interaction in TinyHVM — beta reduction, duplication, erasure, tensor dispatch — is a specialization of one of these two patterns.

---

## Single-Step Reduction with TStepReduce

`TStepReduce[term]` fires exactly **one** interaction and returns three things:

```wolfram
TInit[];
term = TApp[TLam[x |-> x], TNum[42]];

{result, interaction, heap} = TStepReduce[term]
```

- `result` — the term after this single interaction
- `interaction` — a `TInteraction` object describing what happened
- `heap` — a `THeap` snapshot of the state after

Inspect the interaction:

```wolfram
interaction["RuleName"]     (* "Beta" *)
interaction["BeforeTag"]    (* "App" *)
interaction["AfterTag"]     (* "Lam" *)
```

The `TInteraction` object tells you: APP met LAM, triggering a Beta reduction.

---

## Watching the Active Pair

Before reducing, you can see **which pair will interact next** using `TInteractionGraph`:

```wolfram
TInit[];
term = TApp[TLam[x |-> x], TNum[42]];
TInteractionGraph[term]
```

The graph highlights the active pair — the APP and LAM nodes — with red edges. This is the pair that will interact on the next step.

---

## Multi-Step Walkthrough

Let's trace a more interesting reduction: `(λx. x + x) 21`

```wolfram
TInit[];
term = TApp[TLam[x |-> TOp2["Add", x, x]], TNum[21]];
```

This requires multiple interactions. Let's step through:

```wolfram
(* Step 1: APP-LAM beta reduction *)
{t1, i1, h1} = TStepReduce[term];
i1["RuleName"]
TInteractionGraph[t1]

(* Step 2: the result depends on what's next... *)
{t2, i2, h2} = TStepReduce[t1];
i2["RuleName"]

(* Continue until fully reduced *)
{t3, i3, h3} = TStepReduce[t2];
i3["RuleName"]
```

At each step, `TInteractionGraph` shows you the next active pair, and the `TInteraction` tells you exactly which rule fired.

---

## Full Trace

For a complete reduction log, use tracing:

```wolfram
TInit[];
term = TApp[TLam[x |-> TOp2["Add", x, x]], TNum[21]];
TTraceEnable[];
TTraceClear[];
result = TReduce[term];
traces = TTrace[];
TTraceDisable[];

(* Each trace is a TInteraction object *)
traces // Column
```

Each entry shows the rule that fired and the before/after tags. The total number of interactions:

```wolfram
Length[traces]
```

---

## Annihilation in Detail

When APP meets LAM (both are the CON symbol — same symbol, different polarity), they **annihilate**:

```
Before:   APP ←→ LAM
              ╲   ╱
          arg   var  body

After:    var := arg    (substitute)
          result = body
```

The argument is written into the variable's location, and the body becomes the result. Two nodes disappear, replaced by a direct connection.

```wolfram
TInit[];
(* Watch annihilation: APP meets LAM *)
term = TApp[TLam[x |-> TOp2["Mul", x, TNum[3]]], TNum[7]];
TInteractionGraph[term]     (* red edge between APP and LAM *)

{r, interaction, _} = TStepReduce[term];
interaction["RuleName"]     (* "Beta" — annihilation of APP-LAM *)

(* After beta: OP2 with x substituted by 7 *)
TINetGraph[r]

(* One more step: OP2-NUM compute *)
{r2, i2, _} = TStepReduce[r];
i2["RuleName"]              (* "OP2-NUM Compute" *)
```

---

## Interaction Count

Every interaction increments a global counter:

```wolfram
TInit[];
TInteractionCount[]        (* 0 — fresh context *)

TReduce[TApp[TLam[x |-> x], TNum[1]]];
TInteractionCount[]        (* 1 — one beta reduction *)

TReduce[TApp[TLam[x |-> TOp2["Add", x, x]], TNum[5]]];
TInteractionCount[]        (* multiple — beta + DUP + OP2 *)
```

The interaction count is the primary measure of computational work in an interaction net. It's analogous to "steps" in a Turing machine.

---

## Summary

- **Annihilation**: same symbol, principal ports meet → nodes cancel, wires cross-connect
- **Commutation**: different symbols → nodes pass through each other
- `TStepReduce[t]` fires one interaction → `{result, TInteraction, THeap}`
- `TInteractionGraph[t]` highlights the next active pair in red
- `TTrace[]` returns the full reduction log as `TInteraction` objects
- `TInteractionCount[]` counts total interactions
