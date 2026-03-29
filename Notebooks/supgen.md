# SupGen: Superposition-Based Program Synthesis in TinyHVM

TinyHVM's interaction net runtime natively supports **superpositions** — a mechanism that lets you evaluate multiple candidate programs simultaneously with optimal sharing of common subcomputations. This notebook demonstrates a DSL for program synthesis using SUP nodes, then explores the deeper implications: kernel autotuning, neural architecture search, cost model synthesis, and typed-calculus-guided discovery.

---

## 1. Setup

```wolfram
With[{dir = Quiet[NotebookDirectory[]]},
  If[StringQ[dir], PacletDirectoryLoad[FileNameJoin[{dir, "..", "wl"}]]]];
Get["TinyHVM`"];
TInit[]
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

The function `f` is cloned — one copy for each branch. Crucially, if the function is applied to a SUP *before* beta-reduction, non-linear variable use (like `x + x`) stays correct: each branch gets its own substitution.

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
```

```wolfram
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
```

```wolfram
(* Build candidate function: f(x) = x + c *)
f = TLam[x |-> TOp2["Add", x, cSpace]];

(* Apply to test input x = 2 — all 8 outputs computed simultaneously *)
outputs = TApp[f, TNum[2]];
TSupNumValues[outputs]
(* Expected: {2, 3, 4, 5, 6, 7, 8, 9} *)
```

```wolfram
(* Which candidate gives f(2) = 7? *)
checks = TOp2["Eq", outputs, TNum[7]];
matches = TSupNumValues[checks]
(* Expected: {0, 0, 0, 0, 0, 1, 0, 0} — c=5 wins! *)
```

```wolfram
(* Decode: position of match → c value *)
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
Position[both, 1]
```

```wolfram
(* Decode: position → (a, b) pair *)
(* The search tree is: for each a value, nest all b values *)
(* So position p → a = Quotient[p-1, 4], b = Mod[p-1, 4] *)
{aVal, bVal} = QuotientRemainder[Position[both, 1][[1, 1]] - 1, 4]
(* Expected: a = 3, b = 2 *)
```

---

## 7. Interaction Counting: The Sharing Advantage

The power of superpositions is **computation sharing**. When multiple candidates share common subexpressions, they're computed once. Let's measure this.

```wolfram
(* Method 1: Superposed search — all 8 candidates at once *)
TInit[];
i0 = TInteractionCount[];

c = Fold[TSup[#2, #1]&, TNum[7], Reverse @ Table[TNum[i], {i, 0, 6}]];
f = TLam[x |-> TOp2["Add", x, c]];
result = TApp[f, TNum[2]];
reduced = TReduce[result];

supInteractions = TInteractionCount[] - i0
```

```wolfram
(* Method 2: Sequential — evaluate each candidate independently *)
TInit[];
i0 = TInteractionCount[];

seqResults = Table[
    Module[{fi, ri},
        fi = TLam[x |-> TOp2["Add", x, TNum[ci]]];
        ri = TApp[fi, TNum[2]];
        TNumValue[TReduce[ri]]
    ],
    {ci, 0, 7}
];

seqInteractions = TInteractionCount[] - i0
```

```wolfram
(* Superposed evaluation uses far fewer interactions *)
{Row[{"Superposed: ", supInteractions, " interactions"}],
 Row[{"Sequential: ", seqInteractions, " interactions"}],
 Row[{"Speedup: ", N[seqInteractions / supInteractions], "x"}]} // Column
```

---

## 8. Tracing Superposition Evaluation

Use interaction tracing to see APP-SUP and OP2-SUP rules firing in real time.

```wolfram
TInit[];
TTraceEnable[];
TTraceClear[];

(* Simple case: add 10 to SUP(3, 7) *)
s = TSup[TNum[3], TNum[7]];
result = TOp2["Add", s, TNum[10]];
reduced = TReduce[result];

(* View the trace — shows OP2-SUP rule firing *)
Dataset[TTrace[]]
```

```wolfram
(* Verify result: SUP(13, 17) *)
TTraceDisable[];
TSupNumValues[reduced]
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
TInit[];

c = SearchSpace[Range[0, 63]];  (* 64 candidates! *)
f = TLam[x |-> TOp2["Add", x, c]];

(* Evaluate all 64 candidates at x=10 *)
SearchEval[f, 10]
(* Expected: {10, 11, 12, ..., 73} *)
```

```wolfram
(* Build a fresh lambda for the find (terms are linear — used once) *)
c2 = SearchSpace[Range[0, 63]];
f2 = TLam[x |-> TOp2["Add", x, c2]];

(* Find which c gives f(10) = 42 *)
matches = SearchFind[f2, 10, 42];
solution = FirstPosition[matches, 1][[1]] - 1
(* Expected: c = 32 *)
```

---

## 10. Applications: Where SupGen Meets TinyHVM

The sections above demonstrated SupGen on integer arithmetic — a toy setting. But the mechanism is completely general: **any pure function expressible as an interaction net can be superposed.** TinyHVM's tensor operations, kernel scheduling, and gradient computation are all interaction net terms. This makes SupGen applicable to problems that matter.

### 10.1. Kernel Schedule Optimization

GPU kernels have large schedule spaces: tile sizes, vectorization widths, loop orderings, memory layouts. Traditional autotuners (like tinygrad's or Halide's) enumerate candidates and benchmark each one sequentially. SupGen evaluates a symbolic cost model across all candidates simultaneously.

```
schedule_space = SUP(
  {tile=16, vec=1},
  SUP({tile=32, vec=2},
      SUP({tile=64, vec=4},
          {tile=128, vec=8})))

cost = λsched. symbolic_flops(sched) + symbolic_memory_traffic(sched)
results = APP(cost, schedule_space)  (* all 4 costs computed with sharing *)
```

The symbolic cost model shares subcomputation across candidates — the matrix dimensions, reduction axis lengths, and broadcast patterns are the same for all schedules. Only the tile/vector parameters differ. This turns O(n) independent evaluations into O(1) shared evaluation with O(n) final projections.

**What TinyHVM needs**: Encode schedule parameters as TAG_NUM terms within SUP trees. The cost model is a pure lambda that takes a schedule and computes estimated cycles. The existing OP2-SUP distribution handles the arithmetic automatically. Top-K candidates from symbolic evaluation then get hardware-timed.

### 10.2. Cost Model Synthesis

Before you can search over schedules, you need a cost model. The usual approach: hand-write a cost model per hardware target, debug it for months, and hope it generalizes. SupGen can **synthesize the cost model itself**.

**Problem**: Given (schedule, measured_time) pairs from profiling, find a cost function that predicts runtime.

The cost model is a program in a small DSL — say, arithmetic expressions over schedule parameters. The search space is the set of all such programs up to a given depth:

```
(* AST search space for cost models *)
(* Leaves: schedule parameters like tile_m, tile_n, n_warps *)
(* Nodes: +, *, /, max *)

cost_candidates = SUP(
  λs. s.flops / s.bandwidth,          (* roofline *)
  SUP(
    λs. s.flops / s.bandwidth + s.tiles * LAUNCH_OVERHEAD,  (* roofline + launch *)
    SUP(
      λs. s.memory_ops * LATENCY + s.flops / s.throughput,  (* memory-bound *)
      λs. max(s.flops / s.throughput, s.memory_ops * LATENCY)  (* max(compute,memory) *)
    )))
```

Apply each candidate cost model to every profiled schedule, compare against measured time, find the one that minimizes prediction error — all in one superposed evaluation. The cost model that survives is the one that explains the hardware behavior.

This is **program synthesis applied to performance engineering**: you're searching for the *right abstraction* of the hardware, not just the right schedule.

### 10.3. Neural Architecture Search (NAS)

A neural network is a function `f(x; θ)` where the architecture defines the function structure and `θ` are the learned weights. Traditional NAS trains hundreds of candidate architectures independently. SupGen can evaluate forward passes of all candidates simultaneously.

**Key insight**: Most candidate architectures share significant structure. A ResNet-18 and a ResNet-34 share the first N blocks. A search over `{3, 5, 7}` kernel sizes in a conv layer shares the input tensor, padding logic, and everything downstream.

```
(* Architecture search space for a single layer *)
layer_candidates = SUP(
  λx. conv(x, kernel=3),
  SUP(
    λx. conv(x, kernel=5),
    λx. conv(x, kernel=7)))

(* Architecture search space for depth *)
depth_candidates = SUP(
  λx. block(block(x)),           (* 2 blocks *)
  SUP(
    λx. block(block(block(x))),  (* 3 blocks *)
    λx. block(block(block(block(x))))))  (* 4 blocks *)
```

When `APP(layer_candidates, input)` fires, APP-SUP distributes the input to all three conv variants. The input tensor is shared (cloned at the pointer level, not copied). Each conv produces a different output tensor, but downstream operations that are identical across architectures (like the final classifier head) are computed once.

**In TinyHVM**: The forward pass is already an interaction net term (that's how the tensor graph works). A SUP over architectures is a SUP over forward-pass terms. Reduction drives the forward pass; APP-SUP distributes across architecture choices. The same mechanism that does `double(SUP(3,7)) → {6,14}` does `forward(SUP(resnet18, resnet34), batch) → {loss1, loss2}`.

For a realistic NAS, you'd combine this with a few-shot training loop: run K gradient steps on each candidate (sharing compute for shared prefix layers), then select the architecture with lowest validation loss. The training loop itself is an inet term, so the entire train-and-evaluate pipeline can be superposed.

### 10.4. Hyperparameter Search

The same mechanism handles learning rate schedules, optimizer choices, regularization strengths:

```
lr_space = SUP(0.001, SUP(0.003, SUP(0.01, 0.03)))
optimizer_space = SUP(sgd, SUP(adam, adamw))
```

A training loop parameterized by these choices becomes a superposition of training runs. Shared computation: the data loading, forward pass (for identical architectures), loss computation (same loss function applied to different predictions), and gradient topology (same architecture → same gradient structure, just different values).

### 10.5. Data Augmentation Policy Search

Which augmentation pipeline works best? Instead of training separate runs:

```
aug_space = SUP(
  λimg. random_crop(img),
  SUP(
    λimg. random_crop(flip(img)),
    λimg. cutout(random_crop(img))))
```

Superpose the augmentation policies. The model weights are shared across all branches (same initial weights, same optimizer). After N steps, project each branch and compare validation accuracy.

---

## 11. Typed Calculus as Search Constraint

The searches above are **untyped**: every candidate is evaluated, even nonsensical ones. In a search over program ASTs, most random trees are garbage — they might divide by zero, produce the wrong output shape, or violate invariants. Evaluating them wastes compute.

**Typed superpositions** add constraints that prune the search space *before* evaluation. The type system acts as a filter: only well-typed candidates survive to be evaluated. This is the connection between SupGen and the Curry-Howard correspondence.

### 11.1. Types as Constraints

In a simply-typed lambda calculus, a type `A → B` constrains a function to accept `A` and produce `B`. If we're searching for a function `f : Int → Int` that satisfies `f(2) = 7`, the type constraint eliminates candidates that would produce strings, booleans, or error values.

In TinyHVM's interaction net, types are encoded as **labels** on SUP/DUP pairs. A labeled SUP `SUP_L(a, b)` only annihilates with a `DUP_L` of the same label. Different labels commute — they pass through each other without annihilating. This means:

```
DUP_1(SUP_1(a, b)) → {a, b}          (* same label: annihilate *)
DUP_1(SUP_2(a, b)) → SUP_2(DUP_1(a), DUP_1(b))  (* different: commute *)
```

Labels encode which "dimension" of the search space a superposition belongs to. When you search over two independent parameters (say, learning rate and architecture), they get different labels so they don't interfere.

### 11.2. Shape Constraints for Tensor Programs

For neural architecture search, the most important "type" is the **tensor shape**. A conv layer with kernel size 3 on a 28×28 input produces a 26×26 output (no padding) or 28×28 output (same padding). Downstream layers must match.

In TinyHVM, shapes are carried on tensor metadata. A typed SupGen search would:

1. Build the architecture search space as a SUP tree
2. Propagate shapes through each candidate symbolically (no actual compute)
3. Candidates whose shapes don't compose are pruned to ERA (erasure)
4. Only shape-valid architectures get evaluated

```
(* Pseudocode: typed architecture search *)

(* Shape propagation is a pure function *)
shapeOf[conv[k_, pad_]] := λinShape. {inShape[[1]] - k + 1 + 2*pad, ...}

(* Build typed candidates — only those whose shapes compose survive *)
typed_candidates = typeCheck[
  SUP(conv3_nopad, SUP(conv5_nopad, conv3_samepad)),
  inputShape -> {28, 28},
  outputShape -> {28, 28}  (* classifier expects this *)
]
(* conv3_nopad and conv5_nopad produce wrong output shapes → pruned *)
(* Only conv3_samepad survives → no wasted compute *)
```

This is shape inference as a type system: the shape propagation function is the type checker, and candidates that fail are eliminated at "compile time" (before the forward pass runs).

### 11.3. Linear Types for Resource Safety

TinyHVM's interaction nets are inherently **linear**: each term is used exactly once. This is a feature, not a bug. Linear types prevent:

- **Double-free**: A tensor buffer used by two operations gets freed twice. Linear types ensure each buffer has exactly one consumer.
- **Aliasing bugs**: Two operations write to the same buffer concurrently. Linear types make aliasing explicit (via DUP).
- **Memory leaks**: A tensor is allocated but never consumed. Linear types ensure every allocation reaches a consumer.

For SupGen, linear types constrain the search space to *resource-safe* programs. If you're synthesizing a kernel schedule, linear types ensure every buffer is read exactly once (or explicitly duplicated), preventing the synthesizer from generating programs that leak memory or have race conditions.

### 11.4. Refinement Types for Correctness

Going beyond simple types, **refinement types** encode arbitrary predicates:

- `{x : Int | x > 0}` — positive integers
- `{t : Tensor | shape(t) = [B, C, H, W]}` — tensors of a specific shape
- `{f : Schedule | f.tile_m * f.tile_n ≤ SHARED_MEM}` — schedules that fit in shared memory

In SupGen, refinement types act as **early pruning**: before evaluating a candidate, check if it satisfies the refinement predicate. If not, replace it with ERA. This is implemented as:

```
(* Refinement check as a pure function *)
refine[pred_, candidate_] :=
  If[pred[candidate], candidate, ERA]

(* Apply refinement to a search space *)
refined = mapSup[refine[fitsInSharedMem, #]&, schedule_space]
```

The `mapSup` function distributes the refinement check across all branches of the superposition. Branches that fail are erased. The remaining branches are exactly the feasible region of the search space.

### 11.5. Toward Dependent Types: Programs That Prove Their Own Correctness

The most powerful typed SupGen would use **dependent types**, where the type of a value depends on the value itself. In this regime, a synthesized program carries a proof that it satisfies its specification.

For kernel optimization: a schedule of type `{s : Schedule | cost(s) ≤ cost(s') ∀s' ∈ space}` is *provably optimal* within the search space. The synthesizer doesn't just find a good schedule — it finds one with a machine-checkable proof of optimality.

This is ambitious but not fantasy. The building blocks exist in TinyHVM:

- **Interaction nets** can encode proof terms (Curry-Howard)
- **SUP/DUP** can search over proof strategies
- **Type labels** constrain the search to well-typed proofs
- **Reduction** normalizes proof terms, checking validity

The practical version is more modest: encode the spec as a type, search over programs of that type, and let the type checker prune invalid candidates. Even without full dependent types, this dramatically reduces the search space compared to untyped enumeration.

---

## 12. The Bigger Picture

SupGen in TinyHVM is not just a search algorithm — it's a **computational paradigm** where search and evaluation are unified. The same reduction engine that runs your forward pass also runs your architecture search. The same interaction rules that fuse your kernels also fuse your search space.

| Application | Search space | Constraint | Evaluator |
|---|---|---|---|
| Constant synthesis | integers | input/output examples | OP2 arithmetic |
| Kernel autotuning | tile/vec/parallel params | hardware limits (shared mem, registers) | symbolic cost model |
| Cost model synthesis | arithmetic ASTs | prediction accuracy on profiled data | error metric |
| Neural architecture search | layer types, depths, widths | shape compatibility | forward pass + loss |
| Hyperparameter search | lr, optimizer, augmentation | convergence | training loop |
| Typed program synthesis | lambda terms of type τ | type inhabitation | specification predicate |

The key property that makes this practical is **optimal sharing**: computation that is common across candidates is performed once. The more structure the candidates share, the greater the speedup over brute-force enumeration. For neural architecture search where candidates share 90%+ of their structure, the theoretical speedup is 10×+.

The next step is implementing the tensor-level SupGen: superpositions of TinyHVM tensor operations, where APP-SUP distributes across lazy tensor graphs and the fusion engine handles the rest. This is `resources/supgen_kernel_search.md`.

```wolfram
TFree[]
```
