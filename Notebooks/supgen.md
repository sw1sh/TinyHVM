# SupGen: Program Synthesis via Superposition

Advanced demonstrations of **superposition-based program synthesis**. Instead of evaluating $N$ candidates one at a time, superpose all $N$ into a single interaction net and reduce simultaneously. Optimal sharing means interaction count grows sub-linearly in $N$.

---

## 1. Setup

```wolfram
PacletDirectoryLoad[FileNameJoin[{NotebookDirectory[], "..", "wl"}]];
Get["TinyHVM`"];
```

```wolfram
TInit[]
```

---

## 2. Synthesizing XOR from AND/NOT Gates

**Problem**: Find **all** circuits that compute $\text{XOR}(a,b)$ using only AND and NOT gates. A `FindBooleanAlternative` for a restricted gate set.

Superpose all binary tree circuits with 4 leaves into a single interaction net. Each leaf independently picks a wire from $\{a,\; b,\; \lnot a,\; \lnot b\}$. Each internal node independently picks AND or NAND. TGroupings enumerates all $C_3 = 5$ tree shapes:

$$\underbrace{5\text{ structures}}_{\text{Catalan}(3)} \times \underbrace{2^3\text{ gate configs}}_{\text{AND/NAND per node}} \times \underbrace{4^4\text{ wirings}}_{\text{per leaf}} = 10{,}240\text{ candidates}$$

DUP the superposed circuit 4 times (one per truth-table input pair). DUP-SUP annihilation ensures the **same** circuit is evaluated on all inputs.

### Building blocks

```wolfram
TInit[];
bNot[x_] := TOp2["Sub", TNum[1], x];
bAnd[a_, b_] := TOp2["Mul", a, b];
```

### Dual construction: TTerm + decoder tree

Each factory returns `{TTerm, decoderNode}` so we can reconstruct the circuit from a branch function later.

```wolfram
(* Wire: superposition of 4 input lambdas, tracked by 3 labels *)
mkWire[] := Module[{l1, l2, l3},
  l1 = TFreshLabel[]; l2 = TFreshLabel[]; l3 = TFreshLabel[];
  {TSup[l1, TLam[x |-> TLam[y |-> x]],
      TSup[l2, TLam[x |-> TLam[y |-> y]],
          TSup[l3, TLam[x |-> TLam[y |-> bNot[x]]],
              TLam[x |-> TLam[y |-> bNot[y]]]]]],
   {"wire", l1, l2, l3}}];

(* Gate: AND or NAND, tracked by 1 label *)
mkGate[{ft_, fd_}, {gt_, gd_}] := Module[{l, f0, f1, g0, g1},
  l = TFreshLabel[];
  {f0, f1} = TDup[ft]; {g0, g1} = TDup[gt];
  {TSup[l,
      TLam[x |-> TLam[y |-> Module[{x0, x1, y0, y1},
          {x0, x1} = TDup[x]; {y0, y1} = TDup[y];
          bAnd[TApp[TApp[f0, x0], y0], TApp[TApp[g0, x1], y1]]]]],
      TLam[x |-> TLam[y |-> Module[{x0, x1, y0, y1},
          {x0, x1} = TDup[x]; {y0, y1} = TDup[y];
          bNot[bAnd[TApp[TApp[f1, x0], y0], TApp[TApp[g1, x1], y1]]]]]]],
   {"gate", l, fd, gd}}];

(* Balanced SUP tree over dual pairs *)
dualSupTree[{x_}] := x;
dualSupTree[ps_List] := Module[{m = Ceiling[Length[ps]/2], l, L, R},
  l = TFreshLabel[];
  L = dualSupTree[ps[[;;m]]]; R = dualSupTree[ps[[m+1;;]]];
  {TSup[l, L[[1]], R[[1]]], {"struct", l, L[[2]], R[[2]]}}];

(* TGroupings replacement that tracks structure *)
buildGrouped[{f_}, _] := f[];
buildGrouped[{f1_, f2_}, g_] := g[f1[], f2[]];
buildGrouped[fs_List, g_] := dualSupTree[Table[
  g[buildGrouped[fs[[;;k]], g], buildGrouped[fs[[k+1;;]], g]],
  {k, Length[fs]-1}]];
```

### Decoder: branch function to Boolean expression

```wolfram
decode[bf_, {"struct", l_, L_, R_}] :=
  decode[bf, If[Lookup[bf, l, 0] === 0, L, R]];
decode[bf_, {"gate", l_, L_, R_}] :=
  If[Lookup[bf, l, 0] === 0, And, Nand][decode[bf, L], decode[bf, R]];
decode[bf_, {"wire", l1_, l2_, l3_}] :=
  If[Lookup[bf, l1, 0] === 0, a,
  If[Lookup[bf, l2, 0] === 0, b,
  If[Lookup[bf, l3, 0] === 0, Not[a], Not[b]]]];
```

### Synthesis

```wolfram
{circuit, decoder} = buildGrouped[ConstantArray[mkWire[] &, 4], mkGate];

{c0, ct} = TDup[circuit]; {c1, ct2} = TDup[ct]; {c2, c3} = TDup[ct2];
checks = MapThread[
  TOp2["Eq", TApp[TApp[#1, TNum[#2]], TNum[#3]], TNum[#4]] &,
  {{c0, c1, c2, c3}, {0, 0, 1, 1}, {0, 1, 0, 1}, {0, 1, 1, 0}}];
allMatch = Fold[bAnd, checks];
```

### Collapse and decode

```wolfram
gr = TCollapseGrouped[allMatch];
vals = TNumValue /@ gr["values"];
xorBFs = Pick[gr["bf"], vals, 1];

xorCircuits = DeleteDuplicates[decode[#, decoder] & /@ xorBFs];
Column[xorCircuits, Spacings -> 0.5]
```

Two canonical forms, each with 8 commutative variants:

| Circuit | Form |
|---------|------|
| `And[Nand[a,b], Nand[Not[a],Not[b]]]` | $\lnot(a \land b) \;\land\; \lnot(\lnot a \land \lnot b)$ |
| `Nand[Nand[a,Not[b]], Nand[b,Not[a]]]` | $\lnot(\lnot(a \land \lnot b) \;\land\; \lnot(b \land \lnot a))$ |

### Verification

```wolfram
AllTrue[xorCircuits, BooleanTable[#, {a, b}] === {False, True, True, False} &]
```

All 16 circuits verified against the XOR truth table.

### High-level interface

All of the above is packaged as `FindBooleanAlternative` with `Method -> "TinyHVM"`:

```wolfram
FindBooleanAlternative[Xor[a, b], {And, Not}, All, "MaxSize" -> 4, Method -> "TinyHVM"]
```

### Why parallelism is hard here

`TCollapsePar` uses a Chase-Lev work-stealing deque (LIFO for cache locality, FIFO steal for load balance). However, `allMatch` synthesis has an inherent bottleneck: **DUP-SUP commutation creates shared heap state between branches**. `TReduce` must fire all interaction rules serially before the flat SUP-of-NUMs tree can be walked. The parallel tree walk itself is trivially fast ($\sim \mu s$), so thread scaling is $\approx 1\times$.

```wolfram
buildSynth[] := Module[{circuit, dec, c0, ct, c1, ct2, c2, c3, checks},
  {circuit, dec} = buildGrouped[ConstantArray[mkWire[] &, 4], mkGate];
  {c0, ct} = TDup[circuit]; {c1, ct2} = TDup[ct]; {c2, c3} = TDup[ct2];
  checks = MapThread[
    TOp2["Eq", TApp[TApp[#1, TNum[#2]], TNum[#3]], TNum[#4]] &,
    {{c0, c1, c2, c3}, {0, 0, 1, 1}, {0, 1, 0, 1}, {0, 1, 1, 0}}];
  Fold[bAnd, checks]];

timing = Table[Module[{t, r},
  TInit[];
  {t, r} = AbsoluteTiming[
    If[n <= 1, TNumValue /@ TCollapse[buildSynth[]],
               TNumValue /@ TCollapsePar[buildSynth[], n]]];
  <|"threads" -> n, "time" -> t, "results" -> Length[r],
    "XOR" -> Total[r], "speedup" -> Null|>
], {n, {1, 2, 4, 8}}];
timing[[1, "speedup"]] = 1.0;
Do[timing[[i, "speedup"]] = timing[[1, "time"]] / timing[[i, "time"]],
  {i, 2, 4}];
Grid[Prepend[Values /@ timing,
  {"Threads", "Time (s)", "Results", "XOR", "Speedup"}],
  Frame -> All, Alignment -> Left]
```

The bottleneck is architectural. Consider what happens when `TDup[circuit]` creates `{c0, c1}`:

$$\text{DUP}(\ell, x) \;\longrightarrow\; \text{DP}_0(\ell, \texttt{loc}), \; \text{DP}_1(\ell, \texttt{loc})$$

Both projections point to the **same heap cell** `loc`. When `c0` reduces and hits a SUP inside `x`, the DUP-SUP interaction writes the result back to that shared cell. Meanwhile `c1` also needs to read and transform the same cell. This is a **data race** if two threads reduce `c0` and `c1` concurrently — they'd both read-modify-write `heap[loc]`.

TinyHVM serializes this: one thread fires all DUP-SUP commutations, resolving every shared reference before any parallel work begins. The result is a flat SUP-of-NUMs tree that the work-stealing deque walks in parallel — but that walk is trivially fast ($\mu s$), so the speedup is $\approx 1\times$.

**HVM's alternative: copy-on-DUP.** Instead of sharing `loc`, DUP eagerly (or lazily) **clones** the subterm:

$$\text{DUP}(\ell, x) \;\longrightarrow\; x', \; x'' \quad \text{(independent copies)}$$

Now `c0 = x'` and `c1 = x''` share **no mutable state**. Thread A reduces $x'$, thread B reduces $x''$, with zero synchronization. The cost: $O(|x|)$ memory per DUP instead of $O(1)$. But the payoff is genuine parallelism — each SUP branch becomes an independent computation that can run on its own core.

This is the fundamental tradeoff:

| | **Shared-heap (TinyHVM)** | **Copy-on-DUP (HVM)** |
|---|---|---|
| DUP cost | $O(1)$ — pointer to shared cell | $O(\|x\|)$ — clone subterm |
| Parallel reduce | Serialized by shared cells | Fully parallel |
| Memory | Optimal sharing | Proportional to parallelism |
| Best for | Sequential eval with max sharing | Parallel eval across cores |

For TinyHVM, the path to parallel reduce would be a **hybrid**: share by default (keeping the $O(1)$ DUP), but when spawning a parallel task, perform a **deep clone** of the subterm so the worker gets an independent copy. This is essentially what `thvm_clone` (used by REF unfolding) already does — the infrastructure exists, but wiring it into the collapse-and-reduce pipeline requires careful lifetime management of the cloned heaps.

---

## 3. Sorting by Permutation Search

**Problem**: Sort $[5, 2, 8, 1]$. Instead of writing a sorting algorithm, **superpose all 24 permutations** and filter for the non-decreasing one.

Pack 4 elements as digits of an integer $a \cdot 10^6 + b \cdot 10^4 + c \cdot 10^2 + d$, extract each digit, check ordering via unsigned arithmetic.

We use a **balanced** SUP tree (depth $\lceil\log_2 n\rceil$) instead of a right-skewed chain (depth $n-1$) — deep chains hit the reduce stack limit when DUP commutations compound:

```wolfram
TInit[];
(* Balanced binary SUP tree — depth log2(n) instead of n *)
balancedSup[{x_}] := x;
balancedSup[xs_List] := With[{mid = Ceiling[Length[xs]/2]},
  TSup[balancedSup[xs[[;;mid]]], balancedSup[xs[[mid+1;;]]]]];

vals = {5, 2, 8, 1};
allPerms = Permutations[vals];

(* Superpose all 24 permutations as packed integers *)
packPerm[p_] := TNum[p[[1]] 1000000 + p[[2]] 10000 + p[[3]] 100 + p[[4]]];
perms = balancedSup[packPerm /@ allPerms];

(* DUP packed value for 4 digit extractions *)
{p0, pt} = TDup[perms]; {p1, pt2} = TDup[pt]; {p2, p3} = TDup[pt2];

(* Extract each element *)
x = TOp2["Div", p0, TNum[1000000]];
yRaw = TOp2["Mod", TOp2["Div", p1, TNum[10000]], TNum[100]];
zRaw = TOp2["Mod", TOp2["Div", p2, TNum[100]], TNum[100]];
w = TOp2["Mod", p3, TNum[100]];

(* DUP y and z — each used in two adjacent comparisons *)
{y0, y1} = TDup[yRaw]; {z0, z1} = TDup[zRaw];

(* a <= b via unsigned trick: (b - a) / 100 = 0 iff b >= a for values < 100 *)
leq[a_, b_] := TOp2["Eq", TOp2["Div", TOp2["Sub", b, a], TNum[100]], TNum[0]];
sorted = TOp2["Mul", TOp2["Mul", leq[x, y0], leq[y1, z0]], leq[z1, w]];

results = TNumValue /@ TCollapse[sorted];
permStrs = ("[" <> StringRiffle[ToString /@ #, ","] <> "]") & /@ allPerms;
Pick[permStrs, results, 1]
```

24 permutations tested in one reduction. Expected output: $[1, 2, 5, 8]$.

### Why this works

Each DUP of the superposed `perms` creates a copy with **identical SUP labels**. When digit-extraction operations commute with these SUPs, the labels ensure each branch corresponds to the same permutation across all four extractions. The final MUL-of-checks tests all 24 permutations in parallel.

Taelin's NeoGen scales this approach: **Peano Sort synthesized in 2 seconds** (159$\times$ over brute-force enumeration of 5.7M candidates). ADD-CARRY with 16 unknown bits: 65K candidates in 36K total interactions — sub-1 interaction per candidate.

---

## 4. Constraint Satisfaction

**Problem**: Find all $(x, y, z)$ with $x, y, z \in \{1, 2, 3, 4\}$ satisfying:

$$x + y = 5, \quad y + z = 6, \quad x \neq z$$

$4^3 = 64$ candidate assignments. Superpose all values for each variable and filter:

```wolfram
TInit[];
mkVal[] := TSup[TNum[1], TSup[TNum[2], TSup[TNum[3], TNum[4]]]];
x = mkVal[]; y = mkVal[]; z = mkVal[];

(* DUP: x appears in constraints 1 and 3, y in 1 and 2, z in 2 and 3 *)
{x0, x1} = TDup[x]; {y0, y1} = TDup[y]; {z0, z1} = TDup[z];

c1 = TOp2["Eq", TOp2["Add", x0, y0], TNum[5]];
c2 = TOp2["Eq", TOp2["Add", y1, z0], TNum[6]];
c3 = TOp2["Sub", TNum[1], TOp2["Eq", x1, z1]];
allSat = TOp2["Mul", TOp2["Mul", c1, c2], c3];

results = TNumValue /@ TCollapse[allSat];
{"candidates" -> Length[results], "solutions" -> Total[results],
 "interactions" -> TInteractionCount[]}
```

Three solutions out of 64: $(1,4,2)$, $(2,3,3)$, $(3,2,4)$.

The key insight: each variable's SUP has a unique label. DUP-SUP annihilation correlates the same variable across constraints, while different variables' SUPs commute freely — producing the full Cartesian product of assignments.

---

## 5. Composition Synthesis

**Problem**: Find $g, h \in \{+1, \times 2, \times 3, +5\}$ such that $g(h(2)) = 7$ **and** $g(h(4)) = 13$.

$4 \times 4 = 16$ candidate compositions. Each `mkOps[]` creates a fresh, independent superposition:

```wolfram
TInit[];
mkOps[] := TSup[
  TLam[x |-> TOp2["Add", x, TNum[1]]],
  TSup[TLam[x |-> TOp2["Mul", x, TNum[2]]],
    TSup[TLam[x |-> TOp2["Mul", x, TNum[3]]],
      TLam[x |-> TOp2["Add", x, TNum[5]]]]]];

(* Two constraints — fresh SUPs per constraint, intersect results *)
c1 = TNumValue /@ TCollapse[TOp2["Eq", TApp[mkOps[], TApp[mkOps[], TNum[2]]], TNum[7]]];
c2 = TNumValue /@ TCollapse[TOp2["Eq", TApp[mkOps[], TApp[mkOps[], TNum[4]]], TNum[13]]];
solution = c1 * c2;

opNames = {"+1", "\[Times]2", "\[Times]3", "+5"};
pos = FirstPosition[solution, 1][[1]] - 1;
Row[{"g = ", opNames[[Quotient[pos, 4] + 1]],
     ",  h = ", opNames[[Mod[pos, 4] + 1]]}]
```

Expected: $g = +1,\; h = \times 3$ — since $h(2) = 6,\; g(6) = 7$ and $h(4) = 12,\; g(12) = 13$.

---

## 6. Subset Sum at Scale

**Problem**: Given $N$ items with weights $w_1, \ldots, w_N$, find all subsets summing to target $T$. Each item is independently included or excluded: $2^N$ candidates.

For $N = 16$: **65,536 candidates** in a single superposition. Each item contributes a binary choice via SUP:

```wolfram
TInit[];
n = 16;
SeedRandom[42];
weights = RandomInteger[{1, 20}, n];
target = Total[weights[[{1, 4, 7, 11, 15}]]];

(* Each item: SUP(0, weight_i) — include or exclude *)
choices = Table[TSup[TNum[0], TNum[weights[[i]]]], {i, n}];

(* Sum all choices — DUP distributes across SUPs, producing 2^n branches *)
sum = choices[[1]];
Do[sum = TOp2["Add", sum, choices[[i]]], {i, 2, n}];

(* Check against target *)
match = TOp2["Eq", sum, TNum[target]];
{t, results} = AbsoluteTiming[TNumValue /@ TCollapse[match]];
Grid[{{"weights", weights}, {"target", target},
  {"candidates", Length[results]}, {"solutions", Total[results]},
  {"time", t}, {"interactions", TInteractionCount[]}},
  Alignment -> Left, Frame -> All]
```

65,536 candidates checked in one reduction pass. The ADD-SUP commutation distributes addition across all $2^{16}$ branches automatically — no explicit subset enumeration.

### Scaling to $N = 20$ ($> 1$ million candidates)

```wolfram
TInit[];
n = 20;
SeedRandom[42];
weights = RandomInteger[{1, 20}, n];
target = Total[weights[[{1, 4, 7, 11, 15, 18}]]];

choices = Table[TSup[TNum[0], TNum[weights[[i]]]], {i, n}];
sum = choices[[1]];
Do[sum = TOp2["Add", sum, choices[[i]]], {i, 2, n}];
match = TOp2["Eq", sum, TNum[target]];

{t, results} = AbsoluteTiming[TNumValue /@ TCollapse[match]];
Grid[{{"candidates", Length[results]}, {"solutions", Total[results]},
  {"time", t}, {"interactions", TInteractionCount[]}},
  Alignment -> Left, Frame -> All]
```

$2^{20} = 1{,}048{,}576$ candidates evaluated in a single interaction net. The interaction count grows sub-linearly: DUP-SUP annihilation shares the prefix sums across exponentially many branches.

---

## 7. The Principle

```wolfram
Grid[{
  {Style["Problem", Bold], Style["Search space", Bold], Style["Technique", Bold]},
  {"XOR from AND/NOT", "10,240 candidates", "Superpose lambdas, DUP for correlated eval"},
  {"Sort 4 elements", "24 permutations", "Pack/extract digits, unsigned comparison"},
  {"CSP (3 vars, 3 constraints)", "64 assignments", "DUP each variable for multi-constraint"},
  {"Composition g\[SmallCircle]h", "16 pairs", "Fresh SUPs per constraint, intersect results"},
  {"Subset Sum N=16", "65,536 candidates", "Binary SUP per item, ADD distributes across branches"},
  {"Subset Sum N=20", "1,048,576 candidates", "Same technique, 1M+ branches in one net"}
}, Frame -> All, Alignment -> Left, Spacings -> {2, 0.8},
  Background -> {None, {LightDarkSwitched[GrayLevel[0.9], GrayLevel[0.2]], None}}]
```

All four problems share the same mechanism: **superpose the search space, evaluate all candidates with shared sub-computation, filter survivors**. The interaction net's DUP-SUP annihilation rule handles correlation automatically — same labels annihilate (selecting a specific branch), different labels commute (preserving independence). No explicit constraint propagation, no SAT solver. Just reduction.

```wolfram
TFree[]
```
