# ICC Bridges — Why a Separate Dual of Lambda

## The Misconception: "APP is the dual of LAM"

In standard interaction calculus, APP and LAM are **not** duals. They are the
**same combinator symbol** (CON) seen from different ports:

```
      LAM                    APP
       │                      │
   ┌───●───┐              ┌───●───┐
   │  CON  │              │  CON  │
   └─┬───┬─┘              └─┬───┬─┘
     │   │                   │   │
    var body               fun  arg
```

When two CON nodes meet principal-to-principal (APP-LAM), they **annihilate**
(beta reduction). This is the same-symbol annihilation rule of interaction
combinators.

APP is LAM's **eliminator**, not its dual. The actual duality table in
interaction combinators is:

```
Symbol 1:  CON  (lambda / application)    — annihilation = beta
Symbol 2:  DUP  (superposition / dup)     — annihilation = copy
```

When CON meets DUP, they **commute** (pass through each other).

## The Problem: Where Do Types Go?

If you want type checking via interaction net reduction, you need types to
**flow through the same net as computation**. But if you use LAM/APP for types:

- `{val : λx.T}` would trigger APP-LAM annihilation → beta reduction
- The type information is destroyed instead of being propagated

You need a **third combinator symbol** that can carry type information without
being consumed by the computation rules.

## The Bridge: θx.term

Taelin's insight (credited to **Franchu** on HOC Discord): add one new combinator
symbol with two faces:

```
Symbol 3:  ANN/BRI  (annotation / bridge)
   BRI (θx. body)  — the binder face (like LAM)
   ANN ({t : T})   — the eliminator face (like APP)
```

The bridge `θx. body` binds a variable just like lambda, but the variable flows
in the **opposite semantic direction**:

| Node | Variable represents | Direction |
|------|---------------------|-----------|
| λx. body | input that will be provided by APP | caller → callee |
| θf. body | the thing being typed (provided by ANN) | value → type |

## The Four Interaction Rules

The complete interaction calculus of constructions has 4 rules between 3 symbols:

```
APP-LAM:  (λx.f a)          → f[x := a]          ANNIHILATION (same symbol: CON)
ANN-BRI:  {v : θf.T}        → T[f := v]          ANNIHILATION (same symbol: ANN/BRI)

APP-BRI:  (θf.T a)          → θx.(T[f:=λ$k.x] {$k:a})    COMMUTATION (CON vs ANN/BRI)
ANN-LAM:  {v : λx.T}        → λx.{(v $k) : T[x:=θ$k.x]}  COMMUTATION (ANN/BRI vs CON)
```

(Plus the standard DUP interactions: DUP-CON commutation, DUP-DUP annihilation)

The pattern:
- **Same symbol** → annihilate (simplify)
- **Different symbol** → commute (pass through, create new structure)

The commutation rules are what make type checking work: they recursively
distribute annotations inward through the term structure, checking types
at each level.

## From the ICC Repo (VictorTaelin)

Source: https://github.com/VictorTaelin/interaction-calculus-of-constructions

> "In the Interaction Calculus interpretation of Interaction Combinators, a lambda
> can be seen as the opposite of an application, and a duplication can be seen as
> the opposite of a superposition. Likewise, the opposite of an annotation node
> is the bridge (θx. T), and it can be used to express a variety of types,
> including dependent functions (Π(x: A) B[x]), dependent pairs (Σ(x: A) B[x])
> and even self types."

> "None of these are primitives; the only added primitive is the annotator node,
> which is identical to a lambda in form, shape and computation; yet, by just
> reducing the annotator via the conventional interaction combinator rules, the
> result is a process that is effectively equivalent to dependent type checking."

## Type Encodings from Bridges

All type constructors are **derived**, not primitive:

```
Fun = λA λB θf λx {(f {x: A}): B}            -- simple function: A → B
All = λA λB θf λx {(f {x: A}): (B x)}        -- Pi type: Π(x:A). B(x)
Ind = λA λB θf λx {(f {x: A}): (B f x)}      -- Self/inductive type
Sig = λA λB θp (p λfst λsnd λp(p {fst: A} {snd: (B fst)}))  -- Sigma type
Any = θx x                                     -- top type (wildcard)
```

The key: `θf` binds the **thing being typed** so the return type can depend on
it (as in `Ind` where `B` receives both `f` and `x`). This is impossible with
just lambda/application because the function being typed has no name in scope.

## Why This Matters for SupGen

In HVM4's SupGen (superposition-based program synthesis):

1. **Type-directed pruning**: Programs are superposed (`SUP`). Each branch is
   annotated with its type (`ANN`). Ill-typed branches hit a stuck ANN-LAM
   commutation or produce an ill-formed term → prune via ERA.

2. **Bidirectional type inference**: The APP-BRI and ANN-LAM commutation rules
   implement bidirectional type checking as net reduction. No separate
   type-checking pass needed.

3. **Self types for induction**: The `Ind` encoding allows inductive proofs,
   which the standard Calculus of Constructions cannot express. This means
   SupGen can synthesize programs that require induction.

## TinyHVM Implementation

Current status (Phase 4 complete):

```c
TAG_BRI  = 13   // Bridge: θx.body — heap layout identical to LAM: [var_slot, body]
TAG_ANN  = 14   // Annotation: {term:type} — heap: [term, type]
```

Interaction rules implemented:
- **APP-BRI**: Same beta rule as APP-LAM (simplified — full ICC commutation deferred)
- **ANN**: Transparent strip (returns inner term — full type checking deferred)
- **DUP-BRI**: Same commutation as DUP-LAM (creates two bridges)
- **DUP-ANN**: DUP through both term and type slots

### What's Deferred

The full ICC commutation rules (APP-BRI creating new bridges+annotations, ANN-LAM
distributing annotations inward) are not yet implemented. Currently:

- APP-BRI does simple beta (same as APP-LAM)
- ANN strips the annotation (no type checking)

This is sufficient for the current use case (labeled SUP/DUP search spaces).
Full ICC type checking will be added when type-directed pruning is needed.

### Exact HVM1 Rules (for future implementation)

From ICC.hvm1:
```
(APP (Lam fun.bod) arg) = (fun.bod arg)                                    // APP-LAM: annihilation
(APP (Bri fun.bod) arg) = (Bri λx(APP (fun.bod (Lam λ$k(x))) (ANN $k arg))) // APP-BRI: commutation
(ANN val (Lam typ.bod)) = (Lam λx(ANN (APP val $k) (typ.bod (Bri λ$k(x))))) // ANN-LAM: commutation
(ANN val (Bri typ.bod)) = (typ.bod val)                                     // ANN-BRI: annihilation
```

Note the `$k` global variable wiring between newly created nodes.

## Sources

### Repos
- [VictorTaelin/interaction-calculus-of-constructions](https://github.com/VictorTaelin/interaction-calculus-of-constructions) — The ICC spec, HVM1/TS/HVML implementations, book examples
- [VictorTaelin/Interaction-Type-Theory](https://github.com/VictorTaelin/Interaction-Type-Theory) — Predecessor (ITT), net diagrams, decay/coherence formulation
- [VictorTaelin/Interaction-Calculus](https://github.com/VictorTaelin/Interaction-Calculus) — Base IC spec (no types)

### Gists (saved in resources/gists/)
- [icc_spec.md](gists/icc_spec.md) — `c360b392...` — ICC spec with bridge/annotation definitions
- [icc_type_checker.ts](gists/icc_type_checker.ts) — `dd291148...` — ITT-flavored CoC type checker
- [ic_spec.md](gists/ic_spec.md) — `903f20e0...` — Base IC spec (the SupTT spec)

### Key Quote (from HOC historical overview)
> "Franchu (on HOC's Discord) realized that a very elegant primitive, that we now
> call 'The Bridge', could be used to derive the core axioms of type theory (such
> as Pi types and Sigma types). In other words, Π and Σ can be further broken
> down, and I strongly believe that may have strong applications in the study of
> mathematical foundations."
> — VictorTaelin, gist `77fd5a2a...`

### Tweet
- [@VictorTaelin, ~Aug 2024](https://x.com/VictorTaelin/status/1831798521755812118) — IC README update referencing bridges and ICC
