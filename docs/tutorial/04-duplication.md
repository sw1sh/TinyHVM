# Chapter 4: Duplication — SUP, DUP, and Labels

Interaction nets are **linear** — every wire connects exactly two ports. If you want to use a value twice, you must explicitly **duplicate** it. This is where SUP (superposition) and DUP (duplication) come in.

---

## SUP: Superposition

A superposition `{a, b}` represents two values existing simultaneously:

```wolfram
TInit[];
s = TSup[TNum[10], TNum[20]];
TINetGraph[s]
```

The SUP node has two children. It doesn't pick one — it holds both.

---

## DUP: Duplication

To use a value twice, you DUP it. This creates two projections:

```wolfram
TInit[];
x = TNum[42];
{dp0, dp1} = TDup[x];
```

`dp0` and `dp1` are DUP projection nodes. When reduced, each yields a copy of the original value:

```wolfram
TReduce[dp0]    (* 42 *)
TReduce[dp1]    (* 42 *)
```

---

## Labels: The Key to Correctness

Every SUP/DUP pair has a **label** — an integer that identifies which duplication scope they belong to. This label determines whether they annihilate or commute:

```wolfram
TInit[];
(* Explicit label 0 *)
s = TSup[0, TNum[10], TNum[20]];
{dp0, dp1} = TDup[0, s];               (* same label: annihilate *)
{TNumValue[TReduce[dp0]], TNumValue[TReduce[dp1]]}
(* {10, 20} — dp0 got the left, dp1 got the right *)
```

When DUP and SUP have the **same label**, they annihilate — each projection selects one branch:

```wolfram
TInit[];
TTraceEnable[]; TTraceClear[];
s = TSup[0, TNum[10], TNum[20]];
{dp0, dp1} = TDup[0, s];
TReduce[dp0];
traces = TTrace[];
TTraceDisable[];
traces[[1]]["RuleName"]    (* "DUP-SUP Annihilation" *)
```

---

## DUP-LAM Commutation

When DUP meets LAM (different symbols), they **commute** — the lambda is cloned:

```wolfram
TInit[];
f = TLam[x |-> TOp2["Add", x, TNum[1]]];
{f0, f1} = TDup[f];

(* Each copy works independently *)
r0 = TReduce[TApp[f0, TNum[10]]];    (* 11 *)
r1 = TReduce[TApp[f1, TNum[20]]];    (* 21 *)
{TGet[r0], TGet[r1]}
```

Step through to see the commutation:

```wolfram
TInit[];
f = TLam[x |-> TOp2["Add", x, TNum[1]]];
{f0, f1} = TDup[f];

(* Before: DUP node connected to LAM *)
TInteractionGraph[f0]

(* Fire one step *)
{r, interaction, _} = TStepReduce[f0];
interaction["RuleName"]    (* "DUP-LAM Commutation" *)

(* After: two copies of the lambda body *)
TINetGraph[r]
```

---

## Fresh Labels

When you omit the label, `TDup` allocates a fresh one:

```wolfram
TInit[];
{dp0a, dp1a} = TDup[TNum[1]];     (* auto label *)
{dp0b, dp1b} = TDup[TNum[2]];     (* different auto label *)
```

Fresh labels ensure different DUP operations don't accidentally annihilate with each other's SUPs.

```wolfram
TFreshLabel[]    (* get the next label without creating a DUP *)
```

---

## DUP-DUP Commutation

When two DUP nodes with **different labels** meet, they commute — passing through each other:

```wolfram
TInit[];
(* Nested duplication: DUP label 0, then DUP label 1 *)
x = TNum[99];
{a, b} = TDup[0, x];
{a0, a1} = TDup[1, a];

TTraceEnable[]; TTraceClear[];
TReduce[a0];
traces = TTrace[];
TTraceDisable[];

(* Look for DUP-DUP commutation in the trace *)
Select[traces, StringContainsQ[#["RuleName"], "DUP-DUP"] &]
```

---

## Linearity in Practice

Why explicit duplication? Because interaction nets are **linear** — every wire has exactly one reader and one writer. This is what makes them:

1. **Garbage-free** — no tracing GC needed, nodes cancel immediately
2. **Parallel-safe** — no shared mutable state
3. **Optimal** — Lévy's notion of optimal sharing is built in

```wolfram
TInit[];
(* This uses x twice, so TLam internally DUPs it *)
double = TLam[x |-> TOp2["Add", x, x]];
TINetGraph[double]    (* see the DUP node connecting x to both OP2 inputs *)
```

---

## Summary

| Interaction | Same/Diff Symbol | Result |
|------------|-----------------|--------|
| DUP-SUP (same label) | Same | **Annihilate** — each projection selects one branch |
| DUP-SUP (diff label) | Different | **Commute** — SUP passes through, DUP clones |
| DUP-LAM | Different | **Commute** — lambda is cloned |
| DUP-NUM | Different | **Copy** — number is duplicated |
| DUP-ERA | — | **Erasure** — DUP is discarded |
| DUP-DUP (diff label) | Different | **Commute** — DUPs pass through each other |

- Labels are the mechanism that distinguishes annihilation from commutation
- `TDup[x]` with auto label; `TDup[label, x]` with explicit label
- `TSup[a, b]` with auto label; `TSup[label, a, b]` with explicit label
