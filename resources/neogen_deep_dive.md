# NeoGen Deep Dive: Superposition-Based Program Synthesis

> From SupGen's 7,277x speedups to NeoGen's "instant" function discovery —
> what it is, how it works, what HVM4 has, what TinyHVM has, and what to build next.

---

## 1. From SupGen to NeoGen: Evolution

### 1.1 SupGen (Open-Source, 2024-2025)

SupGen is Victor Taelin's program synthesizer built on HVM's superposition nodes.
The core idea: instead of evaluating N candidates one at a time, **construct a single
superposed term containing ALL candidates and reduce them simultaneously**. Optimal
sharing ensures common subcomputations across candidates are computed once.

```
-- 65,536 candidates, brute force: 262M interactions
-- 65,536 candidates, SupGen:      36K interactions
-- Cost per candidate: 0.55 interactions (vs 4000)
-- Speedup: 7,277x
```

The mechanism:

```
-- Function applied to a superposition:
(f &L{a, b})  -->  !&L{f0, f1} = f;  &L{(f0 a), (f1 b)}

-- Same-label DUP-SUP (annihilation) — zero cost:
!&L{x,y} = &L{a,b}  -->  [x<-a, y<-b]

-- Different-label DUP-SUP (commutation) — creates sharing:
!&L{x,y} = &M{a,b}  -->  nested superpositions

-- Failed branch — ERA propagates without computing:
(branch that fails) --> ERA  -- automatic pruning
```

This is not parallelism. It works on a single core. It is **algorithmic sharing** of
computation across the search tree — Levy-optimal beta reduction. No redundant beta
step is ever performed.

**Benchmarks:**

| Problem | Search Space | Brute Force | SupGen | Speedup |
|---------|-------------|-------------|--------|---------|
| ADD-CARRY (16 bits) | 2^16 = 65K | 262M interactions | 36K | **7,277x** |
| XOR-XNOR discovery | — | 2.8s (Haskell) | 0.0085s (HVM) | **330x** |
| Lambda equation solving | — | 0.992s (Haskell) | 0.0011s (HVM) | **862x** |
| SAT 16-var | 2^16 | minutes (Rust) | ~1s (HVM) | — |
| Peano Sort (pos 5.7M) | huge | 5m17s enumerate | 2s (NeoGen) | **159x** |

### 1.2 NeoGen (Proprietary, 2025-2026)

NeoGen is SupGen evolved. The key advances:

1. **Type enumeration**: Programs discovered from I/O examples alone, not just type
   signatures. Types are first-class superposed objects.

2. **Denoising interpretation**: NeoGen = "denoising proofs with holes." Take a complete
   proof, remove random sub-expressions, NeoGen recovers them.

3. **HVM4 AOT compilation**: SUP interactions compile to machine code. No interpreter.

4. **Bend2 integration**: Put a "hole" (`ANY` wildcard) anywhere in a program, the
   compiler fills it via NeoGen synthesis.

5. **KolmoLang**: Stack-based DSL minimizing description length (~21 bits for `mul2`
   vs ~50 bits in binary lambda calculus).

NeoGen found **every primitive recursive function tested** "instantly":
- Equality: 0.0008s
- DrawLine: 0.001s
- Insert: 0.006s
- Multiplication (from I/O pairs): collapse_supgen_mul.hvm in HVM4 tests

### 1.3 HOC's Strategic Direction

From the HOC thesis document:

> "Our ambitious goal is to create a purely symbolic AI architecture that replaces
> computationally heavy operations — like matrix multiplications and gradient
> descent — with efficient symbolic alternatives."

$1M budget through end of 2026. 256 Apple M4 processors (better for pattern-matching
and dynamic allocations than GPUs). SupGen demonstrated 1000x over existing solutions,
validating the symbolic computation thesis.

---

## 2. AOT Compilation: How HVM4 Compiles Superpositions

### 2.1 The Pipeline

```
.hvm source  →  parse + validate  →  AOT emit  →  clang -O2  →  native binary
                                       │
                                       └── emit.c (60KB): case-tree compiler
```

CLI modes:
- `--to-c`: Emit standalone C to stdout
- `--as-c`: Emit → compile → run once
- `-o <path>`: Emit → compile → save binary

Source: `/Users/swish/src/HVM4/clang/aot/emit.c`, `build.c`, `_.c`

### 2.2 What "Compiled Superpositions" Actually Means

HVM4's AOT is a **case-tree compiler**. Each top-level `@definition` becomes one C
function. The compiler walks the definition tree in two phases:

**Phase 1: Spine compilation** (`aot_emit_spine`)
- LAM → `if (!arg) fallback; else continue with extended substitution`
- MAT → `switch(tag) { case C_id: hit; default: miss; }`
- DUP → allocate DP0/DP1 pair, extend substitution
- APP → push arg to stack, continue with function
- REF → direct call to compiled function (no indirection)

**Phase 2: Tail expression compilation** (`aot_emit_expr`)
- When reaching a non-spine node, emit C code to construct the term value
- Handles constant folding, repeated constructors, and direct compiled calls

#### SUPs in Tail Position (Compiled)

When a SUP appears as a **tail expression** (return value), it compiles to direct
C code that constructs the SUP term word:

```c
// HVM source: @test = &{1, 2}
// Compiled to:
static Term FN_test(Term *stack, u32 *sp, u32 sb) {
    Term a_0 = term_new_num(1);
    Term b_0 = term_new_num(2);
    return term_new_sup(label_0, a_0, b_0);
}
```

Both branches are recursively emitted as C expressions. The label is embedded as a
compile-time constant. With `clang -O2`, this becomes 3-4 mov instructions + a pack.

#### SUPs on the Spine (Fallback to Interpreter)

When a SUP appears on the **evaluation spine** (being reduced), the compiled code
falls back to the WNF interpreter:

```c
case SUP:
case INC: {
    aot_emit_ret_fallback_app(f, loc, sub, arg, pad2);
}
```

This is because SUPs on the spine trigger interaction rules (DUP-SUP annihilation,
commutation) that involve **runtime decisions** — which label matches, whether to
annihilate or commute. These can't be statically resolved at compile time.

The fallback reconstructs an ALO (Abstract Lambda Object) from the current
substitution chain and delegates to the WNF interpreter.

### 2.3 Key AOT Optimizations

From recent HVM4 commits:

| Optimization | Commit | Effect |
|-------------|--------|--------|
| Direct calls | `2fe43be` | Compiled refs call other compiled refs directly (no function pointer) |
| Self-tail loops | `dc6a616` | Self-recursive REF spines become `goto` |
| Head arg passing | `29a1e46` | Lambda args passed as raw C locals, not heap cells |
| APP-LAM fusion | `dbe2fda` | Fuses APP+LAM into single path, avoids intermediate allocation |
| Strict MAT-DP | `82fcc1f` | Pre-evaluates DP dependencies before matching |
| Alloc elimination | `8900b71` | Telemetry tracks which allocations are eliminated |
| Expression folding | `cd1c0af` | OP2 with known operands folds at compile time |

### 2.4 Substitution Management

The compiled code maintains a stack-allocated substitution chain:

```c
typedef struct {
    u32 len;                   // Depth of substitution stack
    u32 self_id;               // For self-recursive calls
    u8  kind[AOT_SUB_CAP];    // LAM or DUP markers
    u32 lab[AOT_SUB_CAP];     // DUP labels
} AotSubst;
```

When a definition is too deep or hits unsupported nodes, the fallback reconstructs
an ALO from the substitution chain and returns to interpretation. The result is
**mostly-compiled definitions with well-defined interpretation fallback points**.

### 2.5 Performance

HVM4 AOT achieves 130-190 MIPS (190 with AI-assisted optimization — absolute record
for the standard benchmark). This is competitive with GHC single-core while providing
optimal sharing that GHC cannot.

---

## 3. Unordered Superpositions

### 3.1 The Problem: Quadratic DUP Chains

Standard ordered SUPs `&L{a, b}` are ordered pairs. When you have a chain of DUP
nodes accessing a value through multiple levels, the ordered structure forces
quadratic overhead:

```
-- Accessing element N requires traversing N DUP nodes
-- Each DUP-SUP commutation creates new SUP pairs
-- For N accesses: O(N²) interactions
```

Example: Solving `X + 12345 = 1000000` where X is a superposition of all naturals.
With ordered SUPs: quadratic in nat length (1M). The chain of DUPs accessing
each digit creates nested superpositions that grow quadratically.

### 3.2 The Fix: Unordered SUPs

Unordered SUPs `%L{x,y}` satisfy `%L{x,y} == %L{y,x}`. Key properties:

```
-- Single output port (both ports merged):
!%L{x} = v     -- unordered DUP has ONE output, not two

-- UDUP is never consumed — acts as infinite ref-counted producer

-- UDUP-USUP with matching labels (consume ONE branch):
!%L{x} = %L{a,b}  →  !%L{x}=b; a

-- UDUP-USUP with different labels:
!%L{x} = %M{a,b}  →  standard commutation

-- Opportunistic swaps:
Runtime reorders branches based on access patterns
```

Result: `X + 12345 = 1000000` solved in 24M interactions, **linear** in nat
length (1M). Without the fix: quadratic.

### 3.3 Implementation Status

**HVM4: NOT IMPLEMENTED YET.**

HVM4 has no unordered SUP nodes. The codebase has standard ordered SUPs (`TAG_SUP=5`)
with two-slot storage at `heap[loc+0]` and `heap[loc+1]`. No single-port variants,
no opportunistic swaps, no `USUP`/`UDUP` tags.

HVM4 **does** have DSU/DDU (dynamic SUP/DUP) which compute labels at runtime, but
these are a different feature — dynamic labels, not unordered branches.

**TinyHVM: NOT IMPLEMENTED YET.**

TinyHVM mirrors HVM4's ordered SUP structure. The `labeled_sup_types.md` plan
document notes:

> "Unordered superpositions (HVM3, not yet confirmed in HVM4)... HVM4's current
> public code doesn't show unordered nodes yet."

### 3.4 Relevance to TinyHVM

Unordered DUP would give O(1) access when a tensor is used N times in a computation
graph, instead of O(N) chain traversal through ordered DUP nodes. This matters for:

- **Tensor sharing**: A tensor used by 10 downstream ops currently creates a DUP
  chain 10 levels deep. Unordered DUP would make all accesses O(1).
- **SUP search**: When searching over N candidates, each candidate may share
  structure with others. Ordered DUP chains create O(N²) overhead; unordered
  would be O(N).

**Implementation path**: Wait for HVM4 to implement unordered SUPs, then port.
The interaction rules are specified in `labeled_sup_types.md` lines 167-179 but
the runtime mechanics (single-port storage, opportunistic swaps) need careful
design.

---

## 4. Type Enumeration via ICC

### 4.1 What It Is

NeoGen's breakthrough: instead of requiring a type signature to search, NeoGen
can **enumerate types themselves as superpositions**, discovering programs from
input/output examples alone.

The mechanism uses **ICC (Interaction Calculus of Constructions)** — a dependent
type system built directly on interaction nets, using **Bridge nodes** as the
dual of Lambda nodes.

### 4.2 ICC: The Theory

From `icc_spec.md`:

```
data Term
  = (Lam bod)     -- Lambda: λx. body
  | (App fun arg)  -- Application: (f x)
  | (Val bod)      -- Bridge (θ): θx. body (dual of Lambda)
  | (Ann val typ)  -- Annotation: {value : type}
  | (Var idx)      -- Variable
```

**The key insight**: Bridge (Val/θ) is the **type-level dual** of Lambda.
When APP meets Bridge, type information flows **backward** through the net:

```
(APP fun arg) = match fun {
  Lam: (fun.bod arg)                                    -- standard beta reduction
  Val: (Val λx(APP (fun.bod (Lam λ$k(x))) (ANN $k arg))) -- type backward flow
  var: (App var arg)
}

(ANN val typ) = match typ {
  Lam: (Lam λx(ANN (APP val $k) (typ.bod (Val λ$k(x)))))  -- type forward flow
  Val: (typ.bod val)                                        -- type erasure
  var: (Ann val var)
}
```

### 4.3 Type Encodings

Types are encoded as IC terms using Bridge and Lambda:

```
-- Function type: A -> B
(Arr A B) = (Val λf(Lam λx(ANN (APP f (ANN x A)) B)))
-- θf. λx. {(f {x: A}): B}

-- Pi type: ∀(x: A). B[x]
(All A B) = (Val λf(Lam λx(ANN (APP f (ANN x A)) (B x))))
-- θf. λx. {(f {x: A}): (B x)}

-- Inductive type: μ(f: F). ∀(x: A). B[f,x]
(Ind A B) = (Val λf(Lam λx(ANN (APP f (ANN x A)) (B f x))))

-- Any type: θv. v (wildcard)
Any = (Val λx(x))

-- Set (the type of types)
Set = (Var (- 0 1))
```

### 4.4 How Type Enumeration Works for Synthesis

1. **Specification**: Given I/O pairs `(input₁, output₁), ..., (inputₙ, outputₙ)`

2. **Candidate construction**: Build a superposed term where each branch is a
   candidate program, using SUPs with distinct labels for independent choices:
   ```
   candidate = &L0{choice_A, &L1{choice_B, choice_C}}
   ```

3. **Type checking as reduction**: Wrap each candidate in type annotations:
   ```
   annotated = ANN(candidate, expected_type)
   ```

4. **Automatic pruning**: Ill-typed candidates hit contradictions in the ICC
   reduction rules. The Bridge-Lambda interaction creates type constraints that
   propagate through the net. Failed constraints produce ERA (erasure), which
   propagates without computing the erased branch.

5. **I/O verification**: Surviving candidates are applied to inputs and checked
   against expected outputs using EQL (structural equality):
   ```
   valid = AND(candidate(input₁) === output₁,
               candidate(input₂) === output₂)
   ```

6. **Collapse**: The collapser enumerates all surviving branches. Priority
   ordering (INC nodes) ensures the "simplest" candidates are enumerated first
   (smaller description length → lower priority key → popped first).

### 4.5 HVM4's Concrete Example: Synthesizing Multiplication

From `test/collapse_supgen_mul.hvm` and `test/gen_mul4.hvm`:

```hvm
-- Build a superposed expression tree with K levels of choice
@expr = λ&L. λ{
  0n: λf. λx. x;          -- base: identity
  1n+: λ&K. λf. λx.
    ! f &(L) = f           -- duplicate f
    ! x &(L) = x           -- duplicate x
    &(L){f₀(x₀),           -- branch 1: apply f to x
         1n+@expr(L+1,K,f₁,x₁)}  -- branch 2: successor + recurse
}

-- Search for multiplication:
@main =
  ! &F = λF. @muln(1, 100000n, F)
  ! e0 = @Y(F)(3n) === 12n   -- 3*4 = 12
  ! e1 = @Y(F)(4n) === 16n   -- 4*4 = 16
  @when([e0,e1], [F])         -- filter: keep only programs passing both tests

-- Result: λa.λ{0n:0n; 1n+:λb.4n+a(b)}
-- (multiplication by 4, discovered from two I/O examples)
```

The `@when` function acts as a filter: it takes a list of boolean conditions
and a list of values. If all conditions are true (1), it returns the values.
If any condition is false (0), it returns ERA (erases the branch).

Because the conditions are EQL comparisons on superposed values, different
branches of the superposition get different EQL results. Branches that fail
any test are erased. The collapser enumerates only surviving programs.

### 4.6 Implementation Status

**HVM4**: Has all the pieces:
- EQL nodes (TAG=36) with SUP interactions
- AND/OR (TAG=37/38) for constraint composition
- MAT (TAG=12) for pattern matching with SUP commutation
- ANY (TAG=40) wildcard for type holes
- INC (TAG=41) for priority ordering in collapse
- Full collapse algorithm with work-stealing priority queue

**TinyHVM**: Missing key pieces for type-directed synthesis:
- No EQL nodes (structural equality)
- No MAT nodes (pattern matching)
- No AND/OR (short-circuit boolean logic)
- No ANY (wildcard)
- Has INC (TAG_INC=17) but collapse doesn't use it for priority ordering
- Has Bridge (TAG_BRI=13) and Annotation (TAG_ANN=14) — the ICC foundation

---

## 5. The Collapse Algorithm

### 5.1 Two Phases

**CNF (Collapsed Normal Form)** — `clang/cnf/_.c`:
- Reduce to WNF, lift first SUP to top, return immediately
- ERA propagates upward without printing
- DUPs convert to quoted vars (BJV/BJ0/BJ1), disappearing from output

**eval_collapse** — `clang/eval/collapse.c`:
- Breadth-first traversal with work-stealing priority queue
- Lower numeric keys popped first
- SUP increases key (explore later), INC decreases key (explore sooner)
- When a branch has no SUP: print `cnf(term)`
- Multi-threaded via `wspq` (work-stealing priority queue)

### 5.2 Label Behavior

**Different labels → cross product (full combinatorial search):**
```
[&A{1,2}, &B{3,4}]  →  [1,3], [1,4], [2,3], [2,4]
```

**Same labels → pairwise annihilation (correlated choices):**
```
[&A{1,2}, &A{3,4}]  →  [1,3], [2,4]
```

This is the key design lever: **independent parameters get different labels
(full cross product), correlated parameters get the same label (pairwise).**

For kernel search: LOCAL_SIZE and UNROLL are independent → different labels →
all combinations explored. Two parameters that must be chosen together → same
label → only valid pairs explored.

### 5.3 The Collapse Monad (Formal Specification)

```haskell
data Collapse a = Sup Int (Collapse a) (Collapse a) | Val a

bind :: Collapse a -> (a -> Collapse b) -> Collapse b
bind a f = fork a (repeat id) where
  fork (Val v)     paths = pass (f v) (map (\x -> x E) paths)
  fork (Sup k x y) paths = Sup k (fork x (mut k putO paths))
                                  (fork y (mut k putI paths))
  pass (Val v)     _     = Val v
  pass (Sup k x y) paths = case paths !! k of
    E   -> Sup k x y                -- unseen label: keep (cross product)
    O p -> pass x (mut k (\_->p) paths)  -- seen left: follow left (pairwise)
    I p -> pass y (mut k (\_->p) paths)  -- seen right: follow right (pairwise)
```

Source: `labeled_sup_types.md` lines 181-198

---

## 6. KolmoLang: Optimized Search Space

KolmoLang is a stack-based DSL designed to minimize program description length
(approximating Kolmogorov complexity) while maximizing SupGen expressivity.

### 6.1 Instruction Set (10 total)

```
LAM  λ<bod>          -- push external arg to stack, continue with body
MAT  ?<nil><con>     -- match top: []:nil, (x,xs):push xs,x, continue con
PUT  +<idx><nxt>     -- insert popped value at stack position idx
EMI  ^<idx><nxt>     -- emit popped value to result slot idx
NEW  $<nxt>          -- push empty list to stack
ERA  -<nxt>          -- pop and discard top of stack
DUP  &<nxt>          -- pop, push two copies
RET  .               -- return stack
CAL  @<fid><nxt>     -- call function fid, append result, continue
```

### 6.2 Example: Sorting in ~150 bits

```
-- mul2 : ℕ → ℕ     = λ?$.$^0^0@0.          (~21 bits)
-- half : ℕ → ℕ     = λ?$.-?$.^0@0.
-- decs : [ℕ] → ([ℕ],ℕ)  = λ?$$.?$^0@012.-^1@012.
-- xs2h : [ℕ] → [ℕ]      = λ?$.+0@012^0@111.
-- h2xs : ([ℕ],ℕ) → [ℕ]  = λ?λ-$.?λ$+0@021.-+0λ&^0@021.
-- sort : [ℕ] → [ℕ]      = λ@011$@121.
```

**Argument persistence**: Functions cannot reorder args between recursive calls.
This reduces the enumeration space while maintaining expressivity via composition.
For SupGen, smaller description length = fewer bits of choice = smaller search
space = faster synthesis.

Source: `/Users/swish/src/TinyHVM/resources/gists/kolmolang.hvml`

---

## 7. TinyHVM Feature Gap Analysis

### 7.1 What TinyHVM Has

| Feature | Tag | Status | Notes |
|---------|-----|--------|-------|
| SUP (ordered) | TAG_SUP=3 | Complete | Full DUP-SUP annihilation + commutation |
| DP0/DP1 | TAG_DP0=4, TAG_DP1=5 | Complete | Label-based dispatch |
| DSU (dynamic SUP) | TAG_DSU=15 | Complete | Reduces label_expr to NUM, creates SUP |
| DDU (dynamic DUP) | TAG_DDU=16 | Complete | Reduces label_expr to NUM, clones val |
| INC (priority) | TAG_INC=17 | Partial | Declared; collapse ignores priority |
| Bridge (ICC dual) | TAG_BRI=13 | Complete | θx.body — type-level dual of lambda |
| Annotation | TAG_ANN=14 | Complete | {term : type} — transparent to reduce |
| Collapse (basic) | thvm_collapse | Complete | DFS, collects all leaves |
| Collapse (grouped) | thvm_collapse_grouped | Complete | Tracks (label, branch) paths |
| Collapse (parallel) | thvm_collapse_par | Complete | Work-stealing, multi-threaded |
| Fresh labels | thvm_fresh_label | Complete | Monotonic ctx->next_sup_label++ |
| Term clone | term_clone | Complete | Deep copy for REF unfolding |

### 7.2 What TinyHVM Is Missing

| Feature | HVM4 Tag | Priority | Why It Matters |
|---------|----------|----------|----------------|
| **EQL** | TAG=36 | **High** | Structural equality — needed for I/O verification in synthesis |
| **MAT** | TAG=12 | **High** | Pattern matching — needed for type-directed search, constructor dispatch |
| **AND/OR** | TAG=37/38 | **Medium** | Short-circuit booleans — compose multiple constraints |
| **ANY** | TAG=40 | **Medium** | Wildcard — "hole" that matches anything, duplicates freely |
| **Constructors C00-C15** | TAG=13-29 | **High** | Generic constructors for pattern matching |
| **Priority collapse** | — | **High** | Collapse ordered by INC depth — enumerate "best" first |
| **SWI** | TAG=31 | Low | Numeric switch (syntactic variant of MAT) |
| **USE** | TAG=32 | Low | Strict unboxing |
| **UNS** | TAG=39 | Low | Unscoped lambda/var binding |
| **BJV/BJ0/BJ1** | TAG=42-44 | Low | De Bruijn quoted vars (for CNF readback) |
| **PRI** | TAG=45 | Low | Native/FFI function call |
| Unordered SUPs | — | Deferred | Not in HVM4 yet; wait for upstream |

### 7.3 What to Build (Ordered by Impact)

**Tier 1: Enable SUP-based filtering and constraint search**

1. **Priority-aware collapse** (~100 LOC)
   - Modify `thvm_collapse` to respect INC wrapper depth
   - Lower INC count = higher priority = explored first
   - Enables "enumerate best candidates first, stop after K"

2. **EQL nodes** (~200 LOC)
   - Add TAG_EQL with interactions: EQL-NUM, EQL-SUP-L, EQL-SUP-R, EQL-ERA
   - Strict on both sides (reduce both, then compare)
   - Returns NUM(1) or NUM(0)
   - Port directly from HVM4: `clang/wnf/eql_*.c`

3. **AND/OR nodes** (~100 LOC)
   - Short-circuit: AND(0, x) → 0 without reducing x
   - SUP interactions: AND-SUP duplicates with label
   - Port from HVM4: `clang/wnf/and_*.c`, `clang/wnf/or_*.c`

**Tier 2: Enable pattern matching and type-directed search**

4. **Constructor tags C00-C15** (~150 LOC)
   - Multi-field constructors: C_k(f0, f1, ..., fn)
   - Each C_k tag has arity stored in EXT field
   - Heap layout: fields at consecutive locations

5. **MAT nodes** (~200 LOC)
   - Pattern match with handler map + fallback
   - APP-MAT interaction: match on constructor tag, apply handler
   - MAT-SUP interaction: commute (duplicate MAT, apply to both branches)

**Tier 3: Full NeoGen-class synthesis**

6. **ANY wildcard** (~50 LOC)
   - Duplicates itself on DUP (always succeeds)
   - EQL(ANY, x) → 1 (matches anything)

7. **AOT compilation** — see Section 8

---

## 8. AOT Compilation for TinyHVM

### 8.1 What AOT Means for TinyHVM

TinyHVM's current reducer is an **interpreter** — the enter/apply trampoline
dispatches on TAG at runtime. AOT compilation would turn specific patterns
(especially SUP search configurations) into direct C code.

### 8.2 Architecture

```
IC term definition           Compile-time                Runtime
─────────────────           ────────────                ───────
@cost_model(sched)    →     aot_emit_cost_model()  →   static Term FN_cost_model(...)
                            │                           │
                            ├─ LAM spines → if/switch   ├─ direct C control flow
                            ├─ MAT → switch(tag)        ├─ no tag dispatch overhead
                            ├─ OP2 → inline arithmetic  ├─ native arithmetic
                            └─ SUP tail → construct     └─ fallback to interpreter
                                                            for DUP-SUP interactions
```

### 8.3 What to Compile

**Not everything needs AOT.** The 80/20 rule applies:

1. **Cost model functions** — Pure functions evaluated millions of times during
   SUP search. These are the inner loop. Compiling them to C eliminates tag
   dispatch overhead per interaction.

2. **Constraint validators** — Pure functions checking schedule validity (shared
   memory limits, register pressure, occupancy). Called once per candidate but
   the candidate space is huge.

3. **Search space constructors** — Functions that build the superposed schedule
   space. Run once per search, so AOT benefit is small. Leave interpreted.

4. **DUP-SUP interactions** — These are **runtime decisions** (which label
   matches?) and cannot be statically compiled. Always interpreted.

### 8.4 Implementation Path

**Phase 1: Inline C for hot pure functions** (~200 LOC)

Instead of encoding the cost model as IC terms and reducing them, write the
cost model as a C function and call it via TAG_PRI (primitive/FFI):

```c
// Register as a primitive function
thvm_register_prim(ctx, "cost_model", cost_model_fn);

// In the IC term:
// APP(REF("cost_model"), schedule_term)
// Reduces to: call cost_model_fn(schedule_term) directly
```

This is not true AOT but gives 90% of the benefit with 10% of the work.
HVM4's PRI (TAG=45) does exactly this.

**Phase 2: Case-tree compilation for REF definitions** (~500 LOC)

Port HVM4's `aot_emit_spine()` for the subset of IC terms that appear in
search-related definitions:

- LAM → C function parameter
- MAT → switch statement
- OP2 → inline arithmetic
- REF → direct C function call
- SUP → construct term (fallback to interpreter for interactions)

```c
// Before (interpreter): ~15ns per interaction
// TAG dispatch → heap read → rule match → heap write

// After (compiled): ~2ns per interaction
// Direct C call → inline arithmetic → return term
```

**Phase 3: Full AOT pipeline** (~2000 LOC, future)

- Emit complete standalone C from IC definitions
- Compile with system clang
- Link with TinyHVM runtime for DUP-SUP interaction fallback
- Match HVM4's emit.c architecture

### 8.5 Concrete Benefit Estimate

For a symbolic cost model evaluating 100K schedule candidates:
- **Interpreted**: 100K candidates × ~50 interactions × 15ns = ~75ms
- **Compiled cost model**: 100K candidates × ~50 interactions × 2ns = ~10ms
- **With SUP sharing**: ~1000 total interactions × 2ns = ~2μs

The dominant win is SUP sharing (7500x), not AOT (7.5x). **Build SUP search
first, add AOT later for the inner-loop functions.**

---

## 9. Putting It Together: What to Build Now

### 9.1 Immediate (Uses Only Existing Infrastructure)

**Constraint filtering via SUPs.** TinyHVM already has everything needed:

```c
// Superpose LOCAL_SIZE options
u32 l0 = thvm_fresh_label(ctx);
u32 l1 = thvm_fresh_label(ctx);
Term local = thvm_sup(ctx, l0,
    term_num(16),
    thvm_sup(ctx, l1, term_num(32), term_num(64)));

// Apply constraint function (pure IC):
// check_shared_mem(local, kernel_shape) → NUM(valid?)
Term result = thvm_app(ctx, thvm_app(ctx, check_fn, local), shape);

// Reduce — SUP sharing evaluates all 3 in one pass
thvm_reduce(ctx, result);

// Collapse — enumerate surviving branches
CollapseResult cr = thvm_collapse(ctx, result);
```

### 9.2 Short-Term (Add EQL + Priority Collapse)

With EQL and priority-aware collapse, you get I/O-verified search:

```c
// Build superposed schedule
Term schedule = build_superposed_schedules(ctx, kernel);

// Apply symbolic cost model
Term cost = thvm_app(ctx, cost_model_ref, schedule);

// Wrap in INC for priority (lower cost = higher priority)
Term prioritized = thvm_inc(ctx, cost);

// Collapse with priority ordering — cheapest first
CollapseResult cr = thvm_collapse_ordered(ctx, prioritized, top_k);
// cr.terms[0..top_k-1] = top K candidates by estimated cost
```

### 9.3 Medium-Term (Full Synthesis Pipeline)

With MAT + constructors + EQL + AND/OR:

```c
// Define multiplication as a superposed search:
// For each I/O pair, check candidate(input) === output
// Filter with AND across all pairs
// Collapse to enumerate all valid programs

// This is the gen_mul4.hvm pattern, running natively in TinyHVM
```

### 9.4 What NOT to Build

- **Unordered SUPs**: Not in HVM4 yet. Wait for upstream.
- **Full AOT pipeline**: Overkill until SUP search is validated. Use C primitives
  for hot functions instead.
- **KolmoLang compiler**: Only needed if doing general program synthesis. For
  kernel schedule search, the search space is already well-structured.
- **BJV/BJ0/BJ1 quoted vars**: Only needed for human-readable CNF output. Not
  needed for programmatic collapse.

---

## 10. References

### SupGen / NeoGen
- [ADD-CARRY: Fast DPS with HVM Superpositions](https://gist.github.com/VictorTaelin/d5c318348aaee7033eb3d18b0b0ace34)
- [Accelerating DPS with SUP Nodes](https://gist.github.com/VictorTaelin/7fe49a99ebca42e5721aa1a3bb32e278)
- [Superposed Lambda Calculus Enumerator](https://gist.github.com/VictorTaelin/7c4c69a1f07b5c668be613f1032e7d4e)
- [Truly Optimal Evaluation with Unordered Superpositions](https://gist.github.com/VictorTaelin/93c327e5b4e752b744d7798687977f8a)
- [Simple SAT Solver via Superpositions](https://gist.github.com/VictorTaelin/9061306220929f04e7e6980f23ade615)
- [KolmoLang](https://gist.github.com/VictorTaelin/41d3c07d62cbb0978924fa6d504c4bee)
- [NeoGen benchmarks (Taelin)](https://x.com/VictorTaelin/status/1904727018899439853)
- [NeoGen: Peano Sort](https://x.com/VictorTaelin/status/1957909540894089545)
- [HVM4: compiled superpositions](https://x.com/VictorTaelin/status/1985320306001477783)

### HOC / HVM4
- [HOC thesis](resources/gists/hoc_thesis.md)
- [ICC spec](resources/gists/icc_spec.md)
- [KolmoLang source](resources/gists/kolmolang.hvml)
- [HVM4 source](/Users/swish/src/HVM4/clang/)
- [HVM4 AOT emit](/Users/swish/src/HVM4/clang/aot/emit.c)
- [HVM4 collapser doc](/Users/swish/src/HVM4/docs/hvm/collapser.md)
- [HVM4 interaction rules](/Users/swish/src/HVM4/docs/hvm/interactions/)

### TinyHVM Internal
- [supgen_kernel_search.md](resources/supgen_kernel_search.md) — BEAM comparison and hybrid architecture
- [labeled_sup_types.md](resources/plans/labeled_sup_types.md) — SUP type system and collapse monad
- [hvm_reference.md](resources/hvm_reference.md) — HVM version history and architecture
