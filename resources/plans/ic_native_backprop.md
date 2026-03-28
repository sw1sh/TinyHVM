# IC-Native Backpropagation

## Design

Everything is lazy reduction. `thvm_grad_multi(ctx, loss, params, grad_slots, n)` creates
a single TAG_TOP(UOP_GRAD) term. `thvm_reduce` drives the entire backward pass through
interaction rules — no graph walks, no eager tape, no separate backward implementation.

Sharing is inet-native: when a term is used in multiple positions, a DUP node (SUP/DP0/DP1)
makes the sharing explicit. Gradient accumulation at fan-outs emerges from additive ASSIGN
at the base case — no counters, no `dup_loc`, no `GRAD3_FWD`.

## DUP: Optimal Sharing

### Inet Principle

Every wire connects exactly two ports. To use a value twice, create a DUP node:

```
Producer ──→ SUP(x) ──→ DP0 ──→ Consumer A
                    └──→ DP1 ──→ Consumer B
```

The SUP holds a single shared value. DP0 and DP1 are projections. First projection to
reduce forces the value and caches the result. Second projection reads the cache.
This is **optimal sharing** — computation happens once, result is shared.

### Construction: `linear_use` in `thvm_op`

When `thvm_op(ctx, op, a, b)` is called, each argument passes through `linear_use`:

```c
static Term linear_use(TinyHVM *ctx, Term t, u64 dest_loc) {
    // Apply to ALL terms (not just TAG_TOP)
    if (term_tag(t) == TAG_ERA) return t;

    // Hash-probe the use table
    if (first_use(t))   → record dest_loc, return t unchanged
    if (second_use(t))  → create 1-slot DUP node, patch first site to DP0, return DP1
    if (third_use(t)+)  → error/extend (binary tree of SUPs for N-way)
}
```

Key differences from the old hack:
- **Applies to all tags**, not just TAG_TOP. Weights (TAG_TEN) get DUPs too.
- **1-slot DUP**, not 4-slot. No counters, no gradient accumulator slots.
  `heap[dl] = shared_term`. That's it.
- **No `dup_loc` in TensorMeta.** The DUP is inet structure, not metadata.

### DUP Interactions

```
DP0-TEN: return TEN (share buffer, bump refcount)
DP1-TEN: return TEN (same)
DP0-TOP: reduce TOP → TEN, cache at heap[dl], return TEN
DP1-TOP: read cached TEN from heap[dl], return TEN
DP0-SUP: standard inet rule — take branch 0
DP1-SUP: standard inet rule — take branch 1
```

For N-way sharing (3+ uses), build a balanced tree of binary SUPs:
```
x used 3 times → SUP(x, SUP(x, x))
  DP0 → x (first consumer)
  DP1 → SUP(x, x)
        DP0 → x (second consumer)
        DP1 → x (third consumer)
```

No counters needed. Binary tree of binary DUPs. Gradient combination mirrors
the tree structure.

## GRAD Interaction (src/interact/_.c)

```
GRAD(y, gy, x):
  if y == x:       base case → deposit gy via ASSIGN (additive)
  if y == CTR:     multi-target → match params, deposit via ASSIGN
  else:            chain rule → look up y.creator_op, produce new GRAD terms
```

Each backward rule creates lazy ops and new GRAD3 terms:
- `UN_GRAD(da)`: unary → GRAD3(input, da, x) via GRAD_STEP (tail call)
- `BIN_GRAD(da, db)`: binary → ADD(GRAD3(a, da, x), GRAD3(b, db, x))

### GRAD Through DUP

Old system: `GRAD3_FWD` reads `dup_loc`, maintains a counter, parks early arrivals,
fires ADD on last arrival. This is imperative — not inet.

New system: **additive ASSIGN at the base case.** Each GRAD path walks independently.
When two paths reach the same target parameter, each deposits via ASSIGN. ASSIGN
accumulates (ADDs to existing value):

```
ASSIGN(slot, grad):
  existing = read(slot)
  write(slot, ADD(existing, grad))
  return ERA
```

Gradient slots are zero-initialized by `thvm_backward`. First ASSIGN writes the
gradient. Second ASSIGN ADDs the second contribution. Order doesn't matter —
ADD is commutative. No counters, no synchronization, no rendezvous.

Example: `loss = f(g(x), h(x))` where x is DUP'd.

```
Forward:  SUP(x, x) → DP0 to g, DP1 to h
          g(DP0) reduces, h(DP1) reduces — both get same TEN(x)

Backward: GRAD(loss, 1, target)
  → f backward: ADD(GRAD(g_out, da, target), GRAD(h_out, db, target))
  → g backward: GRAD(x, grad_g, target) → ASSIGN(slot, grad_g)
  → h backward: GRAD(x, grad_h, target) → ASSIGN(slot, grad_h)

  ASSIGN #1: slot = 0 + grad_g = grad_g
  ASSIGN #2: slot = grad_g + grad_h  ✓
```

The ADD from BIN_GRAD evaluates both sides. Each side walks independently to x
and deposits. The order of reduction determines which ASSIGN fires first, but
the result is the same.

### Why Counters Were Wrong

The counter approach (`GRAD3_FWD`) was:
1. Non-inet: imperative state mutation (decrement counter, park, fire on zero)
2. Broken for TAG_TEN: excluded weights because "weights appear in both forward
   and SGD chains" — but the real fix is additive ASSIGN, not exclusion
3. Coupled to `linear_use`: gradient accumulation depended on the construction-time
   DUP structure, not on inet interactions

With additive ASSIGN, none of these problems exist. GRAD walks provenance.
ASSIGN accumulates. DUP is just optimal sharing. Each concern is independent.

## Key Properties

1. **No backward_local.** ONE gradient implementation via IC interaction rules.

2. **Lazy gy.** The trampoline fires GRAD without reducing arg1 (gy). Chain rule
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

### Fuser Sees Through DUP

`fuse_walk_inner` in `src/fuse/_.c` already handles DP0/DP1:

```c
if (term_tag(t) == TAG_DP0 || term_tag(t) == TAG_DP1) {
    Term shared = heap_read(ctx, term_val(t));
    return fuse_walk_inner(ctx, shared, ...);
}
```

The fuser looks through DUP to the shared value. Leaf deduplication ensures
the same tensor appears once in the kernel's buffer list. When both inputs
to an op come from the same DUP'd tensor, the fused kernel binds the same
buffer to both input slots — no data duplication.

### DUP'd Lazy Intermediates

When a DUP'd term is a lazy chain (TAG_TOP), the fuser can include the
chain in the fused kernel with CSE:

```
z = ADD(x, y)           ← lazy
w = MUL(DP0(z), DP1(z)) ← both args are the same lazy ADD

Fuser walks MUL:
  arg0 = DP0 → look through → ADD(x, y) → walk → ops + leaves
  arg1 = DP1 → look through → ADD(x, y) → same ops + leaves (dedup'd)

Codegen: float t0 = in0[i] + in1[i];  // ADD computed once
         float t1 = t0 * t0;           // MUL uses it twice
```

The leaf dedup mechanism (`leaf_ids[i] == tid` check) naturally handles this:
both DUP branches resolve to the same leaves, producing the same leaf index.
The codegen sees the same intermediate referenced twice → single register, used
twice. This is CSE for free.

### Backward Fusion

Deferred elementwise dispatch creates backward tensors with `buf_id=0`. These form
chains that `tensor_materialize` fuses at boundaries:

```
GRAD(MUL): MUL(gy, bt) → deferred
GRAD(ADD): ADD(da, db)  → deferred
ENSURE at SUM → materialize_walk → fused 2-op kernel for MUL+ADD
```

With DUP, two GRAD paths may create independent backward chains that share
input tensors. Each chain fuses independently. The shared inputs are bound
to the same buffer in both kernels. No special handling needed.

## Persistent Fused Graphs

### The Opportunity

Each training step creates the same computation graph (same ops, same shapes,
same sharing pattern). The fuser JIT-compiles Metal shaders each step. With
persistence, compile once, replay with new buffer bindings:

```
Step 1: lazy graph → fuse → compile shader → cache (key = graph hash) → dispatch
Step 2: same structure → hash matches → skip compile → rebind buffers → dispatch
```

### DUP in Persistent Graphs

DUP nodes are deterministic structure — same model architecture produces the
same sharing pattern every step. The persistent graph captures:

1. **Op sequence**: the fused ops in order
2. **Leaf binding slots**: which buffer position maps to which input
3. **DUP edges**: which slots share a buffer (from DUP'd inputs)

The compiled kernel is parameterized by buffer pointers. DUP means "same pointer
for slots i and j." This is encoded in the dispatch table, not the kernel code.

### What Changes per Step

| Component | Step 1 | Step N |
|-----------|--------|--------|
| Lazy graph | Created fresh | Created fresh (same structure) |
| DUP structure | Same | Same (deterministic) |
| Fusion pattern | Computed | Cached (hash hit) |
| Metal shader | Compiled | Reused |
| Buffer bindings | Resolved | Resolved (new buffers, same layout) |

### Interaction with `thvm_reduce`

The persistent graph is NOT a replacement for inet reduction — it's a cache.
`thvm_reduce` still walks the lazy graph via enter/apply. Rewrite rules still
fire. But when a rewrite rule invokes `fuse_or_reduce`, the fuser checks the
cache before compiling:

```
fuse_or_reduce(ctx, t):
  hash = graph_hash(t)   // hash the TAG_TOP subtree structure
  if cache[hash]:
    rebind_buffers(cache[hash], resolve_leaves(t))
    dispatch(cache[hash])
  else:
    compile, cache, dispatch
```

The DUP nodes are part of the hash — they affect leaf dedup and buffer binding.
Same DUP pattern → same hash → cache hit.

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

1. **`linear_use`**: Apply to ALL tags (remove `TAG_TOP` guard). Single-slot DUP
   (remove counter slots 1-3). No `dup_loc` writes.

2. **`GRAD3_FWD` macro**: Delete entirely. Replace with plain `GRAD3`.

3. **`BIN_GRAD` / `UN_GRAD`**: Use `GRAD3` directly instead of `GRAD3_FWD`.

4. **ASSIGN interaction**: Change from set to accumulate (`ADD(existing, new)`).
   Grad slots zero-initialized by `thvm_backward` (already done).

5. **`TensorMeta.dup_loc`**: Remove field. DUP is inet structure, not metadata.

6. **DP0/DP1 interact**: Remove the `dup_loc` write (`ctx->tensors[tid].dup_loc = loc`).
   DUP is transparent — just return the shared value.

7. **N-way sharing**: Build binary SUP tree for 3+ uses.
   Current `linear_use` increments counter — replace with nested SUP.
