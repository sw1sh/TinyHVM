# IC-Native Backpropagation

## Design

Everything is lazy reduction. `thvm_grad_multi(ctx, loss, params, grad_slots, n)` creates
a single TAG_TOP(UOP_GRAD) term. `thvm_reduce` drives the entire backward pass through
interaction rules — no graph walks, no eager tape, no separate backward implementation.

Sharing is inet-native: when a term is used in multiple positions, a DUP node makes the
sharing explicit. Gradient accumulation at fan-outs emerges from additive ASSIGN at the
base case — no counters, no `dup_loc`, no `GRAD3_FWD`.

## Nodes, Ports, and Active Pairs

### Node Types and Ports

Every node has exactly **one principal (active) port** and zero or more auxiliary ports.
Interactions fire **only** between two nodes whose principal ports are connected (an
"active pair"). There are no "DP0 interactions" or "DP1 interactions" — DP0/DP1 are
auxiliary-port labels of the DUP node, not separate interaction participants.

```
Node     Principal Port    Aux Ports          Domain
────     ──────────────    ─────────          ──────
APP      fun position      arg                λ-calculus
LAM      binder            var, body          λ-calculus
DUP      value             proj₀, proj₁      sharing
SUP      consumer          branch₀, branch₁  sharing
TEN      (leaf — no aux)                      tensor
TOP      arg₀              arg₁, result       tensor ops
ERA      (leaf — no aux)                      erasure
NUM      (leaf — no aux)                      scalars
REF      (leaf — unfolds)                     definitions
CTR      (compound)        fields...          data
```

### Active-Pair Interaction Rules

Following HVM4's interaction combinator model (Lafont '97), every rule is a pair of
nodes linked through their principal ports. The notation `A ⊳ B` means "A's principal
port is connected to B's principal port."

#### Lambda calculus

```
APP ⊳ LAM  (β-reduction)
  (λx.body arg) → body[x ← arg]

APP ⊳ SUP  (commutation)
  (&L{f,g} a) → ! A &L = a; &L{(f A₀),(g A₁)}

APP ⊳ ERA  (erasure)
  (ERA a) → ERA
```

#### DUP interactions (all through DUP's principal port)

The DUP node's principal port connects to the value being duplicated. The result
wires copies to both auxiliary ports (proj₀ and proj₁).

```
DUP ⊳ SUP  (annihilation, same label)
  ! X &L = &L{a,b} → X₀ ← a, X₁ ← b

DUP ⊳ SUP  (commutation, different label)
  ! X &L = &R{a,b} → ! A &L = a; ! B &L = b; X₀ ← &R{A₀,B₀}; X₁ ← &R{A₁,B₁}

DUP ⊳ LAM  (commutation)
  ! X &L = λx.f → X₀ ← λ$x0.G₀; X₁ ← λ$x1.G₁; x ← &L{$x0,$x1}; ! G &L = f

DUP ⊳ NOD  (commutation — applies to TOP, CTR, any compound node)
  ! X &L = T{a,b,...} → ! A &L = a; ! B &L = b; ...; X₀ ← T{A₀,B₀,...}; X₁ ← T{A₁,B₁,...}

DUP ⊳ TEN  (copy atom — TEN is a leaf like NAM)
  ! X &L = TEN(tid) → X₀ ← TEN(tid), X₁ ← TEN(tid)

DUP ⊳ ERA  (erasure)
  ! X &L = ERA → X₀ ← ERA, X₁ ← ERA

DUP ⊳ NUM  (copy atom)
  ! X &L = NUM(v) → X₀ ← NUM(v), X₁ ← NUM(v)
```

#### Tensor operations (TOP interactions)

```
TOP ⊳ TEN  (fire op — both args resolved)
  TOP(uop, TEN(a), TEN(b)) → dispatch(uop, a, b) → TEN(result)

TOP(GRAD) ⊳ TEN  (gradient chain rule)
  GRAD(TEN(y), gy, x) → chain rule based on y.creator_op

TOP(ASSIGN) ⊳ TEN  (gradient deposit — accumulative)
  ASSIGN(TEN(slot), grad) → slot.buf = ADD(slot.buf, grad.buf); return TEN(slot)
```

### Why DP0/DP1 Are Not Interaction Nodes

In the current code, `case TAG_DP0` and `case TAG_DP1` in `thvm_interact` define
separate interaction handlers. This is structurally wrong:

- DP0/DP1 are **auxiliary port labels**, not nodes. A DUP node has one principal
  port (the value) and two aux ports labeled 0 and 1.
- In the reducer, when a DP0 or DP1 term enters, it represents "I am a consumer
  waiting for the DUP's principal port to resolve." The reducer should:
  1. Push the DUP frame
  2. Enter the value at the principal port
  3. When the value reaches WHNF, fire the DUP ⊳ X rule
  4. Return the appropriate projection (proj₀ or proj₁)

The difference between DP0 and DP1 is **which result to return**, not which
interaction rule to fire. One DUP ⊳ X rule, two possible results.

```c
// Current (WRONG — two separate interaction handlers):
case TAG_DP0: { ... if (TAG_SUP) take slot 0 ... }
case TAG_DP1: { ... if (TAG_SUP) take slot 1 ... }

// Correct (ONE interaction rule, branch on projection):
DUP ⊳ X:
  fire the appropriate rule (DUP-SUP, DUP-TEN, DUP-NOD, ...)
  return proj[dp_index]   // dp_index = 0 or 1
```

## DUP: Optimal Sharing for Tensors

### Construction

When `thvm_op(ctx, op, a, b)` encounters the same term used in multiple positions,
`linear_use` creates a DUP-SUP pair:

```c
// Second use of term t:
u64 sup_loc = heap_alloc(ctx, 2);
heap_set(ctx, sup_loc, t);       // branch 0
heap_set(ctx, sup_loc + 1, t);   // branch 1
Term sup = term_new(TAG_SUP, label, sup_loc);

// Patch first consumer to DP0, return DP1 for second consumer
heap_set(ctx, first_consumer_loc, term_new(TAG_DP0, label, sup_loc));
return term_new(TAG_DP1, label, sup_loc);
```

The label `L` is fresh. Both SUP branches hold the same term `t`.

### Reduction: DUP ⊳ SUP (same label) = Annihilation

When the DUP's principal port connects to a SUP with matching label:

```
! X &L = &L{t, t}
→ X₀ ← t (branch 0)
  X₁ ← t (branch 1)
```

Both projections get the same term `t`. If `t` is lazy (TAG_TOP), each consumer
reduces it independently. The key question: does this double-evaluate?

### Sharing vs Double-Evaluation

In a pure inet, terms are consumed by interactions — a lazy node can only fire
once. In TinyHVM, terms are 64-bit values that can appear in multiple heap slots.
If both SUP branches hold the same TAG_TOP term, reducing one overwrites the
TOP's heap slots, so the second reduction sees already-resolved TEN args and
re-fires the kernel — producing a duplicate tensor.

**Solution: single-slot sharing node.** For tensor sharing (same-label SUP where
both branches are identical), the SUP degenerates to a single slot:

```
heap[sup_loc] = t         // ONE shared slot
DP0(sup_loc) → reduces heap[sup_loc], caches TEN result
DP1(sup_loc) → reads cached TEN from heap[sup_loc]
```

This is the correct implementation of DUP ⊳ TEN: both projections get the same
TEN. The first projection forces reduction (TOP → TEN), caches at `heap[sup_loc]`.
The second projection finds TEN already there.

This works because the DUP ⊳ SUP(same label) annihilation conceptually eliminates
the SUP, and both projections access the same underlying value. The single-slot
representation makes the caching automatic.

### N-Way Sharing

For 3+ uses, build a binary tree of DUP-SUP pairs (matching HVM4's DUP-NOD
commutation pattern):

```
x used 3 times:
  SUP_L1{x, SUP_L2{x, x}}
  Consumer A ← DP0_L1
  Consumer B ← DP0_L2 (from DP1_L1 → SUP_L2)
  Consumer C ← DP1_L2 (from DP1_L1 → SUP_L2)
```

No counters. Binary tree of binary DUPs. Gradient combination mirrors the tree.

## GRAD as Active-Port Interaction

GRAD is a TOP node: `TOP(UOP_GRAD, y, gy, x)`. It fires through the standard
TOP interaction mechanism — principal port is arg₀ (y).

```
GRAD ⊳ TEN(y):
  if y == x:       base case → ASSIGN(slot, gy) (accumulative)
  if y == CTR:     multi-target → match params, ASSIGN each
  else:            chain rule → look up y.creator_op, produce GRAD terms
```

### Chain Rule via Interaction

Each backward rule creates lazy TOP nodes and new GRAD terms:
- **UN_GRAD(da)**: unary op → `GRAD(input, da, x)` via tail-call GRAD_STEP
- **BIN_GRAD(da, db)**: binary op → `ADD(GRAD(a, da, x), GRAD(b, db, x))`

### GRAD Through DUP: Additive ASSIGN

When a tensor is DUP'd (used in multiple forward ops), each consumer creates an
independent GRAD path. Both paths eventually reach the same target parameter and
deposit via ASSIGN. ASSIGN is **accumulative**:

```
ASSIGN ⊳ TEN(slot):
  existing = read(slot.buf)
  write(slot.buf, ADD(existing, grad.buf))
  return TEN(slot)
```

Gradient slots are zero-initialized by `thvm_backward`. First ASSIGN writes the
gradient. Second ASSIGN ADDs its contribution. Order doesn't matter — ADD is
commutative and associative.

```
Example: loss = MUL(ADD(x, y), x) — x is DUP'd

Forward:  DUP(x) → DP0 to ADD.arg0, DP1 to MUL.arg1
          DUP ⊳ SUP annihilates → both get TEN(x)

Backward: GRAD(loss, 1, target)
  MUL backward → BIN_GRAD:
    ADD(GRAD(ADD_out, gy*x, target), GRAD(x, gy*ADD_out, target))
  ADD backward → GRAD(x, da, target) → ASSIGN(slot, da)
  x direct    → GRAD(x, db, target) → ASSIGN(slot, db)

  ASSIGN #1: slot = 0 + da
  ASSIGN #2: slot = da + db  ✓
```

No `GRAD3_FWD`. No `dup_loc`. No counters. Each GRAD path walks provenance
independently. ASSIGN accumulates at the leaf.

## Key Properties

1. **No backward_local.** ONE gradient implementation via IC interaction rules.

2. **Lazy gy.** The trampoline fires GRAD without reducing arg₁ (gy). Chain rule
   formulas wrap gy in new lazy ops, creating fusable chains. Only base case and
   deposit explicitly reduce gy.

3. **Lazy ENSURE.** The GRAD handler does NOT ENSURE inputs at the top. Only ops
   that read data (MM, RMAX, LOG, DIV, MAX) call ENSURE. Others create lazy ops
   that reference deferred tensors — the materialize path handles them.

4. **Dead-branch skip.** BIN_GRAD checks `requires_grad` on both inputs. Single-live
   branch uses UN_GRAD (tail call) instead of ADD + full reduction.

5. **RELU backward uses y.** `CMP(y, 0)` instead of `CMP(at, 0)`. Since `relu(x) > 0
   iff x > 0`, the mask is identical. This allows RELU in fused reduce chains (virtual
   intermediates don't have input data, but y is the real output tensor).

## Integration with Fusion

### Fuser and DUP: Rewrite Rules See Through Sharing

The fuser operates as a rewrite rule (fires in `rewrite_apply` before depth-first
reduction). When it walks a TAG_TOP chain and encounters a DUP node at a leaf:

```
fuse_walk encounters DP0/DP1 at a leaf position:
  → follow DUP's principal port to shared value
  → if value is TEN: standard leaf (dedup by tid)
  → if value is TOP: extend fused chain through the shared op
```

When both inputs to a fused op come from the same DUP (same `sup_loc`), leaf
deduplication binds the same buffer to both kernel input slots. The compiled
kernel references one buffer twice — no data duplication.

### DUP'd Lazy Intermediates → CSE in Codegen

```
z = ADD(x, y)           ← lazy TOP
w = MUL(DP0(z), DP1(z)) ← both args share the same lazy ADD

Fuser walks MUL:
  arg0 = DP0 → principal port → ADD(x, y) → ops + leaves
  arg1 = DP1 → principal port → ADD(x, y) → same ops + leaves (dedup'd)

Codegen: float t0 = in0[i] + in1[i];  // ADD computed once (CSE)
         float t1 = t0 * t0;           // MUL uses register twice
```

Both DUP projections resolve to the same leaf tids, producing the same leaf
indices. Codegen sees one intermediate referenced twice → single register.

### Backward Fusion

Deferred elementwise dispatch creates backward tensors with `buf_id=0`. These form
chains that `tensor_materialize` fuses at boundaries:

```
GRAD(MUL): MUL(gy, bt) → deferred
GRAD(ADD): ADD(da, db)  → deferred
ENSURE at SUM → materialize_walk → fused 2-op kernel for MUL+ADD
```

Two GRAD paths from DUP'd forward intermediates create independent backward chains.
Each fuses independently. Shared input tensors bind to the same buffer. No special
handling beyond standard leaf dedup.

## Persistent Fused Graphs

### Cache Layer

Each training step creates the same computation structure (same ops, same shapes,
same DUP pattern). The fuser JIT-compiles Metal shaders each step. With a
persistence cache, compile once, replay with new buffer bindings:

```
Step 1: lazy graph → fuse → compile shader → cache (key = graph hash) → dispatch
Step N: same structure → hash hit → skip compile → rebind buffers → dispatch
```

### DUP in the Graph Hash

DUP nodes are deterministic structure — same model architecture produces the same
sharing pattern every step. The graph hash includes:

1. **Op sequence**: the fused ops in topological order
2. **Leaf binding slots**: which buffer position maps to which input
3. **DUP edges**: which leaf slots share a buffer (detected via tid dedup)

The compiled kernel is parameterized by buffer pointers. DUP means "same pointer
for slots i and j." Encoded in the dispatch table, not in kernel code.

### Inet Reduction + Cache = No Conflict

The persistent graph is a **cache**, not a replacement for inet reduction.
`thvm_reduce` still walks the lazy graph via enter/apply. Rewrite rules still fire.
But `fuse_or_reduce` checks the cache before compiling:

```
fuse_or_reduce(ctx, t):
  hash = graph_hash(t)        // hash the TAG_TOP subtree including DUP structure
  if cache[hash]:
    rebind_buffers(cache[hash], resolve_leaves(t))
    dispatch(cache[hash])
  else:
    compile, cache, dispatch
```

Fresh lazy graph each step. Same DUP pattern. Same hash. Cache hit. The inet
model (terms consumed by interactions) is preserved — each step's terms are fresh.

## SUM Provenance

Fused SUM(elementwise) records proper provenance for backward:
- Virtual intermediates (lower IDs) with pre-reduce shapes, buf_id from dst
- dst has creator_op=UOP_SUM, src_ids[0]=virtual ew_last
- Backward: SUM expand → elementwise chain → leaves

## Gradient Correctness

Verified at float32 precision:
- MLP + CE: 3.5e-8 max error (numpy parity)
- 2-conv + CE: 7.2e-7 max error (numpy parity)
- conv+relu+pool+CE: <0.05 relative error (finite differences)
- All fused reduce axes (non-trailing, multi-axis): exact (0 error)

## Implementation Changes (from current code)

1. **Unify DP0/DP1 interact**: Merge `case TAG_DP0` and `case TAG_DP1` into a
   single DUP interaction that fires the appropriate rule (DUP ⊳ SUP, DUP ⊳ TEN,
   DUP ⊳ NOD, DUP ⊳ ERA) and returns `proj[dp_index]`.

2. **`linear_use`**: Apply to ALL tags (remove `TAG_TOP` guard). Standard 2-slot
   SUP with matching label. No counter slots. No `dup_loc` writes.

3. **`GRAD3_FWD` macro**: Delete entirely. Replace with plain `GRAD3`.

4. **`BIN_GRAD` / `UN_GRAD`**: Use `GRAD3` directly instead of `GRAD3_FWD`.

5. **ASSIGN interaction**: Change from set to accumulate (`ADD(existing, new)`).
   Grad slots zero-initialized by `thvm_backward` (already done).

6. **`TensorMeta.dup_loc`**: Remove field. DUP is inet structure, not metadata.

7. **N-way sharing**: Build binary SUP tree for 3+ uses.
   Current `linear_use` increments counter — replace with nested SUP.
