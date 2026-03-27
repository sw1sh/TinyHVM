# IC-Native Autograd: Lazy Reverse-Mode AD in a Graph Reducer

How TinyHVM computes gradients, the theoretical background it draws from, and an honest
assessment of where the theory applies and where it doesn't.

---

## Part I: Theoretical Landscape

### 1. Linear Logic (Girard 1987)

Linear logic treats values as **resources** used exactly once. Two structural rules that classical
logic takes for granted become explicit:

- **Contraction** (copying) → explicit DUP, guarded by the `!` (bang) exponential modality
- **Weakening** (discarding) → explicit ERA, also guarded by `!`

In interaction net terms: DUP and ERA are the combinators for these. Every copy is visible in the
graph. This is the foundation Lafont's interaction combinators (1997) build on, and what HVM uses.

The exponential `!A` ("of course A") means "as many copies of A as needed." It comes with four
rules governing how copies are created and consumed:

| Rule | Symbol | What it does |
|---|---|---|
| Dereliction | `der` | Extract one linear use from `!A` → `A` |
| Weakening | `wk` | Discard `!A` (no uses) |
| Contraction | `ctr` | Split `!A` into two copies `!A ⊗ !A` |
| **Codereliction** | `d` | (DLL only) Inject one linear probe `A` → `!A` |

The first three are standard linear logic. Codereliction is the addition from differential LL.

### 2. Differential Linear Logic (Ehrhard & Regnier 2003)

DLL adds codereliction `d` as the **dual** of dereliction. Where dereliction extracts one use
from an unlimited supply, codereliction injects one linear perturbation into an unlimited context.

The sequent rules:

```
Dereliction:     Γ, A ⊢ Δ          Codereliction:    Γ, A ⊢ Δ
              ─────────────                         ─────────────
              Γ, !A ⊢ Δ                            Γ, ?A ⊢ Δ

Cocontraction:  Γ, !A, !A ⊢ Δ      Coweakening:     Γ ⊢ Δ
              ─────────────                         ─────────────
              Γ, !A ⊢ Δ                            Γ, !A ⊢ Δ
```

The **differential lambda calculus** reformulates this as term rewriting rules:

```
d(λx.t) · u = λx.(dt · u)          -- derivative under abstraction
d(t u) · v = (dt · v) u + t (dv)   -- Leibniz rule (product/chain rule)
d(x) · u = u    if x is the variable being differentiated
d(x) · u = 0    otherwise
```

This is a **syntactic** operation on terms. No numerical approximation, no symbolic algebra engine.
The rules mechanically transform a term into its derivative term. The `+` is formal sum
(superposition of terms), which is where the additive structure of linear logic matters.

**What codereliction means operationally**: given a function `f : !A → B` (which can use its
argument arbitrarily many times), codereliction produces `df : A ⊸ (!A → B)` — a function
that takes one linear perturbation and one evaluation point, producing the linear response.
This is the **best linear approximation** of `f` around a point. It is the derivative.

### 3. Differential Interaction Nets (Ehrhard & Regnier 2006)

The DLL rules are reformulated as interaction net rewrites. The key new interaction:

```
DUP-λ rule in DLL:
  DUP(λx.t)  →  Σ_i (λx₁.t₁ᵢ) ⊗ (λx₂.t₂ᵢ)
```

Where the sum ranges over all ways to "split" the body of the lambda between two copies.
In standard IC (Lafont), DUP-LAM produces two independent copies. In differential IC,
DUP-LAM produces copies **plus differential residuals** — the terms that account for how
perturbing one copy affects the other through shared variables.

The interaction rules implement the chain rule as graph topology. When a DUP node meets
a constructor (like a tensor op), it produces:
- Two copies of the constructor (one per DUP branch)
- Cross-terms capturing the derivative contribution

This is beautiful mathematically. It is also inherently **forward-mode**.

### 4. Why DLL Gives Forward-Mode, Not Reverse-Mode

The differential lambda calculus propagates a **tangent** (perturbation) forward:

```
Given f : R^n → R^m
  Forward-mode: df(x) · v = Jf(x) · v    (Jacobian-vector product, JVP)
  One input perturbation v → one output perturbation
  Cost: O(n) evaluations to get full Jacobian
```

For ML, we need **reverse-mode** (backpropagation):

```
  Reverse-mode: df(x)ᵀ · u = Jf(x)ᵀ · u  (vector-Jacobian product, VJP)
  One output cotangent u → all input gradients
  Cost: O(1) evaluations for scalar loss (m=1)
```

Forward-mode is efficient for `f : R → R^m` (few inputs, many outputs).
Reverse-mode is efficient for `f : R^n → R` (many inputs, scalar output — i.e., loss functions).

The codereliction `d` in DLL naturally gives JVP (forward-mode). To get VJP (reverse-mode),
you need to **transpose** the linear map, which requires additional structure:

- **Linear negation** `A⊥`: reverse the polarity of a type, turning inputs into outputs
- **Transposition of linear maps**: `(A ⊸ B)` becomes `(B⊥ ⊸ A⊥)` — flow reversal
- This is related to Girard's **geometry of interaction** (GoI): reversing the flow of tokens
  through a proof net

The key theoretical gap: **DLL gives you the Jacobian. Getting the transposed Jacobian
(the adjoint) requires separate machinery.**

### 5. Approaches That Bridge Forward → Reverse

#### 5a. Linear Negation (Brunel, Mazza, Pagani 2020)

**"Backpropagation in the Simply Typed Lambda-calculus with Linear Negation"** — the paper
at arxiv:1909.13768.

Core idea: define a compositional program transformation `B⟦−⟧` from the simply-typed
lambda calculus to itself augmented with a **linear negation** type constructor `A⊥`.

For first-order types (reals, products), `R⊥ = R` (reals are self-dual — a cotangent is
just another real). For function types:

```
(A → B)⊥  =  B⊥ → A⊥     (reverse the arrow — a cotangent transformer)
```

The transformation maps `f : A → B` to `B⟦f⟧ : A → (B × (B⊥ → A⊥))`:
- Forward: compute `f(x)` and simultaneously build a **continuation** that maps output
  cotangents back to input cotangents
- The continuation IS the backpropagator

They prove this has the same complexity as first-order backprop (no overhead from
higher-order features). The transformation is **purely syntactic** — no mutation, no tape.

**Connection to DLL**: this is NOT using codereliction. Instead, it uses the standard
linear logic duality `(−)⊥` applied to types. The "linear negation" turns the forward
type structure inside-out to get the reverse-mode flow. The forward-mode DLL derivative
and the Brunel-Mazza-Pagani reverse-mode transformation are related but distinct operations.

**Connection to interaction nets**: the paper does not use interaction nets. The transformation
is on lambda terms. However, the linear negation structure maps naturally onto IC topology:
a `B⊥ → A⊥` continuation is a wire that flows "backward" through the net. If you represented
their transformation as an interaction net, backward flow would correspond to tokens travelling
in the opposite direction through net wires — which IS geometry of interaction.

**Nobody has done this.** The paper is purely theoretical. No implementation, no benchmarks,
no system built on it.

#### 5b. Categorical / Reverse Derivative Categories (Cockett et al. 2020)

**"Reverse Derivative Categories"** axiomatizes what a "reverse derivative" operation must
satisfy, categorically.

Key result: **a reverse derivative = a forward derivative + a dagger structure on the
subcategory of linear maps.**

- A **dagger category** has an involution `†` on morphisms: `f : A → B` gives `f† : B → A`
- This involution is exactly **transposition** of linear maps
- Forward derivative gives you `Df : A → (A ⊸ B)` (the Jacobian as a linear map)
- Dagger gives you `(Df)† : A → (B ⊸ A)` (the transposed Jacobian)
- Reverse derivative = composition of these two

This cleanly separates what DLL provides (forward derivative) from what's additionally
needed for backprop (dagger/transpose). The dagger is strictly extra structure — categories
with forward derivatives don't automatically have reverse derivatives.

**Relevance to TinyHVM**: the adjoint functions in our GRAD handler are exactly this dagger.
Each `adjoint_mul`, `adjoint_mm`, etc. is the hand-written transpose of the corresponding
forward-mode derivative. The categorical framework tells us that this manual transposition
is inherent — there's no way to automatically derive reverse-mode from forward-mode without
the dagger structure.

#### 5c. Cartesian Differential / Difference Categories (Blute et al. 2009, Alvarez-Picallo & Lemay 2020)

Cartesian differential categories axiomatize the differential combinator `D` satisfying
chain rule, linearity, and Schwarz theorem as categorical identities. The differential
lambda calculus is the internal language of these categories.

Cartesian **difference** categories (Alvarez-Picallo & Lemay 2020) generalize this to
handle discrete derivatives (finite differences), not just smooth ones. Every cartesian
differential category is a cartesian difference category, but not vice versa. This
framework provides a tangent bundle monad and accounts for both smooth and discrete
approximations.

**For AD specifically**: these categories model forward-mode naturally. Reverse-mode
requires the additional reverse derivative category structure (the dagger).

#### 5d. Continuation-Based Reverse AD (Pearlmutter & Siskind 2008, Wang et al. 2019)

**"Lambda the Ultimate Backpropagator"** (Pearlmutter & Siskind 2008): reverse-mode AD
as a first-class operator in an augmented lambda calculus. Uses CPS transformation internally
— the continuation captures what still needs to be differentiated, and reversing the flow
of continuations gives reverse-mode.

**"Demystifying Differentiable Programming: Shift/Reset the Penultimate Backpropagator"**
(Wang et al. 2019): reveals the tight connection between reverse-mode AD and **delimited
continuations** (shift/reset). Backpropagation IS a control flow reversal:

```
Forward pass:  evaluate f(x), capture continuation k at each op
Backward pass: invoke continuations in reverse order
shift/reset:   k captures "the rest of the forward computation"
               invoking k backwards = backpropagation
```

This is arguably the most practical theoretical insight for implementation. It shows that
reverse-mode AD doesn't need tapes, graphs, or special data structures — just control
flow inversion via continuations.

**Connection to IC**: Continuations in interaction nets are represented as wires going
"upward" in the net (the continuation of a redex). The shift/reset story suggests that
IC-native reverse AD would involve capturing and inverting these continuation wires —
essentially the geometry of interaction picture, where tokens travel backward through the net.

#### 5e. Categorical AD via Homomorphisms (Elliott 2018)

**"The Simple Essence of Automatic Differentiation"**: derives both forward and reverse-mode
AD from a single principle — the AD transformation must be a **category homomorphism** (functor)
preserving cartesian structure. Different representations of the derivative type give different
AD algorithms:

- Functions `(a → b)` as derivatives → forward-mode AD
- Continuation-based `(b → a)` as derivatives → reverse-mode AD
- "Generalized matrices" → classical Jacobian representation

No tapes, no graphs, no mutation. AD falls out of requiring the program transformation to
respect categorical structure. The reverse-mode variant arises by choosing a dualized
(continuation-based) representation of linear maps.

#### 5f. Pure Higher-Order Reverse AD (Vakar 2021, Krawiec et al. 2022)

**"Reverse AD at Higher Types"** (Vakar 2021, ESOP best paper): defines forward and
reverse-mode source-code transformations on a standard higher-order language, arising from
a categorical universal property. In the most elegant formulation, the transformations
**generate code with linear types**. Shows how to implement without linear types using
abstract data types.

**"Provably correct, asymptotically efficient, higher-order reverse-mode AD"**
(Krawiec, Peyton Jones, Krishnaswami et al. 2022, POPL): a simple implementation of
reverse-mode AD that extends to higher-order functions with run time and memory consumption
linear in the original program. Correctness proven via logical relations.

These represent the state of the art in principled higher-order reverse AD, but are
implemented as source-to-source transformations in Haskell, not as interaction net rewrites.

#### 5g. String Diagrams (Alvarez-Picallo, Ghica, Sprunger, Zanasi 2023)

**"Functorial String Diagrams for Reverse-Mode Automatic Differentiation"** (CSL 2023):
enhances string diagram calculus for monoidal categories with hierarchical features to
capture cartesian closed structure. Formulates the Pearlmutter-Siskind AD algorithm
in this graphical language and proves its soundness for the first time.

They define **hypernets** — hierarchical hypergraphs that provide a sound and complete
representation of these extended string diagrams.

**Relevance to IC**: string diagrams for monoidal categories are the closest graphical
cousin of interaction nets. Hypernets are a concrete graph data structure for representing
reverse-mode AD transformations. If anyone were to implement "IC-native backprop," the
hypernet representation would be the most directly applicable reference.

### 6. The Gap: Nobody Has Built This

The state of affairs as of 2025:

| Approach | Mode | Higher-Order | Implemented | Uses IC/nets |
|---|---|---|---|---|
| Ehrhard-Regnier DLL (2003/2006) | Forward | Yes | No (theory only) | Yes (diff nets) |
| Brunel-Mazza-Pagani (2020) | Reverse | Yes | No (theory only) | No (lambda terms) |
| Cockett et al. RDC (2020) | Reverse | Yes (categorical) | No (theory only) | No |
| Elliott (2018) | Both | Limited | Haskell library | No |
| Wang et al. shift/reset (2019) | Reverse | Yes | Scala (LMS) | No |
| Vakar (2021) | Both | Yes | Haskell | No |
| Krawiec et al. (2022) | Reverse | Yes | Haskell | No |
| Alvarez-Picallo et al. (2023) | Reverse | Yes | Hypernets | Closest (string diagrams) |
| PyTorch/JAX | Reverse | No (first-order graphs) | Yes (production) | No |
| **TinyHVM** | Reverse | Untested | Yes (C/Metal) | **Partially** (IC heap, not IC rules) |

**The honest summary**: there's rich theory connecting linear logic to AD, but all practical
implementations either (a) use standard reverse-mode AD with provenance/tapes (PyTorch, JAX,
TinyHVM), or (b) are Haskell/Scala research prototypes not used for real ML training.

Nobody has built an interaction-net-based system where backpropagation emerges from
DUP/ERA/SUP reduction rules rather than hand-written adjoints.

---

## Part II: What TinyHVM Actually Does

TinyHVM's autograd is **standard reverse-mode AD with lazy term representation in an IC heap**.

### The Mechanism

Each TOP node stores **provenance** — which UOp produced it and which source tensor IDs fed into it
(`creator_op`, `src_ids[0]`, `src_ids[1]`). `thvm_grad(ctx, y, x)` creates a lazy `UOP_GRAD`
term. When reduced, the GRAD handler reads `y`'s provenance and applies the corresponding chain
rule — emitting new `UOP_GRAD` terms as lazy TAG_TOP nodes. Those nodes reduce through the same
GRAD handler, working backward until the base case (`y == x`) is reached and `gy` is returned.

No DUP nodes appear in the backward pass. `GRAD3(a, da, x)` creates `TAG_TOP(UOP_GRAD)` nodes,
not `TAG_DP0`/`TAG_DP1`. The gradient computation is a hand-written chain-rule switch on
`creator_op`, not an IC graph rewrite.

### What IS IC-native about it

- **Single reduction engine**: gradients are lazy heap terms that reduce through the same
  `thvm_reduce()` as forward ops — no separate backward engine
- **Demand-driven**: only compute gradients that are actually forced (lazy evaluation)
- **O(1) provenance per tensor**: no tape with O(ops) memory overhead
- **REACHES pruning**: DFS to check if gradient can flow from `y` to `x`, efficiently skipping
  irrelevant branches

These are real engineering wins from "lazy evaluation of reverse-mode AD in a graph reducer."

### What is NOT IC-native about it

- No DUP-TOP interaction rules (DUP meeting a tensor op to produce derivative residuals)
- No ERA-TOP dead-code elimination via IC reduction
- No SUP-TOP distribution (cloning ops across superposition branches)
- No net reversal for reverse-mode — gradients are explicit chain-rule dispatch, not graph polarity
  reversal
- The GRAD handler is an imperative switch statement, not IC interaction rules

### How it maps onto the theory

| Theory concept | TinyHVM equivalent | Faithful? |
|---|---|---|
| Codereliction `d` | Not present | — |
| Forward-mode JVP | Not present (planned, see backprop plan Phase 3) | — |
| Adjoint/transpose (dagger) | Hand-written `adjoint_table[]` per op | Manual, not derived |
| Linear negation `A⊥` | Cotangent tensors have same type as primals | Implicit (R⊥ = R) |
| Continuation reversal | Provenance walk reverses dataflow | Structural, not control-flow |
| ERA = weakening | ERA-as-identity in ADD/MUL rules | Partial (not IC-level) |
| DUP = contraction | DUP-SUP for lambda terms, not for tensors | Partial |

---

## Gradient Rules

| Op | Forward | Grad w.r.t. a | Grad w.r.t. b |
|---|---|---|---|
| ADD | `z = a + b` | `gy` | `gy` |
| SUB | `z = a - b` | `gy` | `-gy` |
| MUL | `z = a * b` | `gy * b` | `gy * a` |
| DIV | `z = a / b` | `gy / b` | `-gy * a / b²` |
| MM  | `z = mm(A,B)` | `mm(gy, Bᵀ)` | `mm(Aᵀ, gy)` |
| RELU | `z = relu(a)` | `gy * (a > 0)` | — |
| EXP | `z = exp(a)` | `gy * z` | — |
| LOG | `z = log(a)` | `gy / a` | — |
| SUM | `z = sum(a)` | `expand(gy, a.shape)` | — |
| EXPAND | `z = expand(a, shape)` | `sum_to_shape(gy, z.shape, a.shape)` | — |

**SUM backward invariant**: `gy` always has the **keepdims shape** — reduced axes are set to
1 rather than removed. So `expand(gy, input.shape)` is a direct broadcast, no reshape needed.
The old code incorrectly reshaped `gy` to all-ones before expanding, which failed when `gy`
was already a non-scalar keepdims shape from an outer SUM.

---

## Gradient Seed Shape

`thvm_grad(ctx, y, x)` seeds with `ones` shaped to match `y`. Previously the seed was always
a scalar `[1]` regardless of `y`'s shape. This caused `test_grad_mm` to abort: the MM backward
computes `mm(gy, Bᵀ)` which requires `gy` to be rank-2.

**Current behavior**: seed = `tensor_fill(ctx, y.shape, 1.0f)`.

---

## Fusion and Autograd Interaction

The fused `SUM(MUL(a, b))` kernel is a performance optimization that skips materializing the
intermediate MUL result. **This fusion is only safe when no MUL input requires a gradient.**

Gate: `if (!ma->requires_grad && !mb->requires_grad) { /* fuse */ }`

When either MUL input requires grad, the MUL node must materialize separately. The GRAD handler
for SUM needs to recurse through MUL to apply the chain rule; if MUL was fused away, the SUM's
`src_ids[0]` would point to the MUL's input (e.g. `diff`) instead of its output (`sq = diff²`),
causing the MUL backward to be skipped and losing the `2*diff` factor.

The `recording` flag still exists — it gates **provenance writes** at tensor creation
(`if (ctx->recording) { md->creator_op = ...; }`). What was removed is using `recording`
as the **fusion gate**. Previously the fused `SUM(MUL)` path had `goto skip_fused_mul_sum`
when `ctx->recording` — replaced with `!ma->requires_grad && !mb->requires_grad`. Two tensors
can be on a gradient path (`requires_grad = 1`) even when `recording = 0` (e.g. during
backward itself), so `requires_grad` is the correct fusion gate.

---

## Strided Reduce Correctness

`thvm_sum_axes` passes through the SUM reducer which reads the source buffer. If the source is
a non-contiguous **expand view** (stride=0 in broadcast dims), a flat `buf_read` reads the
wrong data — the backing buffer may be 1 element while the view claims N. Fix: both the
multi-axis and single-axis reduce paths now materialize non-contiguous inputs using view strides
before accumulating.

---

## Honest Comparison: Theory vs. Implementation

| Claim | Reality |
|---|---|
| "Backward = DUP propagating through forward graph" | Metaphor. Code is chain-rule dispatch on `creator_op` |
| "No tape, just graph reduction" | True. Gradients are lazy terms in the same heap. But the computation is standard reverse AD |
| "DUP = differentiation" | True in DLL forward-mode (codereliction). Not what happens in TinyHVM. Reverse-mode needs the adjoint/dagger, which is separate |
| "Higher-order gradients via GRAD-of-GRAD" | Would work in principle. Closer to the DLL spirit. Untested |
| IC reduction for backprop | Future aspiration, not current reality. See `resources/plans/ic_native_backprop.md` |

### Where the theory could realistically contribute

**1. Forward-mode AD (JVP)** — planned as Phase 3 of ic_native_backprop. This IS where DLL
applies directly. Tangent propagation forward through ops, with tangent splitting at fan-out
points being literal DUP in linear logic. The tangent rules are the same as codereliction
applied to each op.

**2. Higher-order differentiation** — if TinyHVM ever needs gradients of gradients (Hessian-vector
products, etc.), forward-over-reverse `jvp(x → vjp(f, x), x, v)` computes `H·v` without
materializing the Hessian. This requires both forward-mode (Phase 3) and reverse-mode (current).

**3. ERA-based dead code elimination** — the closest the theory gets to practical impact today.
ERA-TOP interaction rules can replace REACHES DFS for pruning dead gradient branches. This is
not DLL-specific; it's just standard IC reduction applied to tensor ops. Planned as Phase 0
of ic_native_backprop.

**4. Understanding the adjoint manually** — the reverse derivative categories framework
(Cockett et al.) clarifies that the hand-written adjoint table IS the dagger structure,
and that there's no shortcut. You can't derive the reverse-mode rule from the forward-mode
rule without explicitly providing the transpose. This is actually useful: it means the current
approach (manual adjoints) is not a hack to be replaced, but the correct implementation of
a categorical structure that must exist.

### What would NOT help

- Implementing full codereliction/cocontraction in the reducer — gives forward-mode, which
  we could implement more simply via tangent subnets
- Trying to derive reverse-mode from DUP interaction rules — DUP gives you copies, not
  transposed derivatives. The theory is clear: reverse ≠ forward
- Building differential interaction nets for tensor ops — massive implementation effort,
  inherently forward-mode, no practical precedent

---

## Why Not Just Use a Tape?

| | Tape (PyTorch-style) | TinyHVM (lazy reverse AD in IC heap) |
|---|---|---|
| Data structure | Separate ops list | Same graph |
| Forward/backward | Two phases | One reduction |
| Memory | O(ops) tape | O(1) provenance per tensor |
| Higher-order gradients | Tape of tapes (tricky) | GRAD-of-GRAD (untested but natural) |
| Implementation size | Separate backward engine | ~200 lines of GRAD handler |

---

## References

### Foundations

1. Girard (1987). "Linear logic." *TCS* 50(1), 1-101.
2. Lafont (1997). "Interaction combinators." *Information and Computation* 137(1), 69-101.
3. Ehrhard & Regnier (2003). "The differential lambda-calculus." *TCS* 309(1-3), 1-41.
4. Ehrhard & Regnier (2006). "Differential interaction nets." *TCS* 364(2), 166-195.
5. Ehrhard (2018). "An introduction to differential linear logic: proof-nets, models and antiderivatives." *MSCS* 28(7).

### Categorical Frameworks

6. Blute, Cockett & Seely (2009). "Cartesian differential categories." *TAC* 22(23), 622-672.
7. Cockett, Cruttwell, Gallagher, Lemay, MacAdam, Plotkin & Pronk (2020). "Reverse derivative categories." *CSL* 2020.
8. Alvarez-Picallo & Lemay (2020). "Cartesian difference categories." *FoSSaCS* 2020.
9. Blute, Cockett, Lemay & Seely (2020). "Differential categories revisited." *Applied Categorical Structures* 28.

### Reverse-Mode AD Theory

10. Pearlmutter & Siskind (2008). "Reverse-mode AD in a functional framework: lambda the ultimate backpropagator." *TOPLAS* 30(2).
11. Elliott (2018). "The simple essence of automatic differentiation." *ICFP* 2018.
12. Wang, Zheng, Decker, Wu, Essertel & Rompf (2019). "Demystifying differentiable programming: shift/reset the penultimate backpropagator." *ICFP* 2019.
13. Abadi & Plotkin (2020). "A simple differentiable programming language." *POPL* 2020.
14. Brunel, Mazza & Pagani (2020). "Backpropagation in the simply typed lambda-calculus with linear negation." *POPL* 2020.
15. Vakar (2021). "Reverse AD at higher types: pure, principled and denotationally correct." *ESOP* 2021.
16. Krawiec, Peyton Jones, Krishnaswami, Ellis, Eisenberg & Fitzgibbon (2022). "Provably correct, asymptotically efficient, higher-order reverse-mode automatic differentiation." *POPL* 2022.

### Graphical / Net-Based

17. Alvarez-Picallo, Ghica, Sprunger & Zanasi (2023). "Functorial string diagrams for reverse-mode automatic differentiation." *CSL* 2023.
