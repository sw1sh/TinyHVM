# HVM4 FFI Pattern: Side-Effect Ops Inside Interaction Combinators

How HVM4 handles side-effectful primitives (`PRI`), and the minimal port
of that pattern to TinyHVM's `UOP_ASSIGN`.

## Problem

A tensor `ASSIGN(dst, src)` is a side effect — it blits `src` into `dst`'s
buffer and yields `dst`. Plain `DUP ⊳ TOP` commutation duplicates an
`ASSIGN` node when an upstream `DUP` demands two copies of a value that
happens to flow through an `ASSIGN`. That produces two structurally
separate `ASSIGN` nodes with `DP`-wrapped args:

```
DUP ⊳ ASSIGN(d, s)
  -->
  ASSIGN₀ = ASSIGN(DP0(d'), DP0(s'))
  ASSIGN₁ = ASSIGN(DP1(d'), DP1(s'))
```

For a pure compute op that would be fine — both copies produce the same
tensor. For a side-effect op it's wrong in spirit (the blit runs twice)
and wrong in practice in TinyHVM, because the two copies' `DP`
projections don't stay consistent under deeper `DUP ⊳ TOP` cascades and
at least one branch eventually erases itself, leaving a stuck
`TOP/ASSIGN` carrier in the final `CTR`.

## HVM4's answer

HVM4's primitive system (see
[clang/prim/fn/log.c](../HVM4/clang/prim/fn/log.c) and
[clang/prim/fn/log_go_0.c](../HVM4/clang/prim/fn/log_go_0.c)) handles
this by writing each PRI's interaction rules inside the primitive's
own function. `%log` takes a string list; `%log_go_0(acc, list)` walks
the list, and when `list` WNFs to a `SUP`, the primitive handles the
`SUP` case *itself*:

```c
case SUP: {
  // %log_go_0(acc, &L{x,y})
  // -----------------------  log-go-0-sup
  // &L{%log(acc0(x)), %log(acc1(y))}
  u32  lab = term_ext(list_wnf);
  u64  sup_loc = term_val(list_wnf);
  Term x = heap_read(sup_loc + 0);
  Term y = heap_read(sup_loc + 1);
  Copy A = term_clone(lab, acc);      // <-- the primitive clones acc
  Term app0 = term_new_app(A.k0, x);
  Term log0 = term_new_pri(table_find("log", 3), 1, &app0);
  Term app1 = term_new_app(A.k1, y);
  Term log1 = term_new_pri(table_find("log", 3), 1, &app1);
  return term_new_sup(lab, log0, log1);
}
```

The generic `DUP ⊳ NOD` rule does duplicate `PRI` nodes structurally in
HVM4 — `term_arity(PRI)` returns the primitive's arity, and `wnf_dup_nod`
clones args. But the primitive's OWN body handles the case where its
args *already carry a SUP* pushed down from an upstream DUP. So the
primitive effectively decides how to distribute itself over
duplication, instead of being blindly cloned at the `DUP ⊳ TOP` site.

The discipline is:

1. **The DUP rule is cheap**: it just wraps children in `DP` projections
   and keeps descending. This alone does NOT run the primitive twice; it
   just signals that the primitive may be duplicated.
2. **The primitive's own code owns correctness under duplication**: when
   a `SUP` reaches an arg, the primitive rewrites itself into a SUP of
   two continuations with cloned captured state.

This keeps the side-effect count matched to the duplication count the
consumer actually demands — not to a raw copy-by-structure.

## Port to TinyHVM `UOP_ASSIGN`

For TinyHVM we want the same discipline but with an even tighter rule:
a blit is expensive, so we don't want the generic `DUP ⊳ TOP`
commutation to create two `ASSIGN` nodes speculatively. Two local
rules, no eager eval:

### Rule 1: `DUP ⊳ UOP_ASSIGN` defers

In [src/interact/combinators.c](../src/interact/combinators.c), the
`DUP ⊳ TOP` branch special-cases `UOP_ASSIGN` and returns the DUP
unchanged (`return t`):

```c
// DUP ⊳ TOP: commute by duplicating the node and splitting children.
if (term_tag(val) == TAG_TOP) {
    u32 uop = term_ext(val);
    // UOP_ASSIGN is a side-effect — no commutation. Wait for the
    // ASSIGN to fire, then DUP ⊳ TEN handles the atomic copy.
    if (uop == UOP_ASSIGN) return t;
    ...
}
```

The reducer's trampoline has already descended into `heap[dup_loc]`
before calling `DUP ⊳ ASSIGN`. If the ASSIGN can fire (both args TEN),
it produces `dst` and the apply phase writes `dst` back into
`heap[dup_loc]`. Next time the DUP is visited, `DUP ⊳ TEN` copies the
atom into both projections — the DUP never clones the side-effect.

### Rule 2: `UOP_ASSIGN ⊳ SUP` redistributes

In [src/interact/tensor_ops.c](../src/interact/tensor_ops.c), inside the
`UOP_ASSIGN` block, add the SUP-arg cases before the "both args must be
TEN" check:

```c
// ASSIGN ⊳ SUP on dst: redistribute.
// ASSIGN(&L{d0,d1}, s) --> &L{ASSIGN(d0, DP0_L(s)), ASSIGN(d1, DP1_L(s))}
if (term_tag(dst_r) == TAG_SUP) {
    u32 lab = term_ext(dst_r);
    u64 sup_loc = term_val(dst_r);
    Term d0 = heap_read(ctx, sup_loc + 0);
    Term d1 = heap_read(ctx, sup_loc + 1);
    u64 sdup = heap_alloc(ctx, 1);
    heap_set(ctx, sdup, src_t);
    Term s0 = term_new(TAG_DP0, lab, sdup);
    Term s1 = term_new(TAG_DP1, lab, sdup);
    ctx->itrs++;
    RETURN_REDUCED(thvm_sup(ctx, lab,
        thvm_assign(ctx, d0, s0),
        thvm_assign(ctx, d1, s1)));
}
// Symmetric: ASSIGN(d, &L{s0,s1}) --> &L{ASSIGN(DP0_L(d), s0), ASSIGN(DP1_L(d), s1)}
if (term_tag(src_t) == TAG_SUP) { /* symmetric */ }
```

This is the exact structural mirror of
`prim_fn_log_go_0`'s `case SUP`. When an upstream DUP pushes a SUP
into ASSIGN's `dst` (or `src`), ASSIGN spawns the two expected blits
itself, with independent cloned counterparts on the other arg.

## Why both rules are needed together

Rule 2 alone is a **no-op** on today's recursive-SGD test: SUPs never
reach ASSIGN's args, because generic `DUP ⊳ TOP` commutation clones
`ASSIGN` into two DP-wrapped copies before a SUP can flow in. Landing
Rule 2 has no observable effect until Rule 1 is active.

Rule 1 alone **deadlocks** at n ≥ 2 even with the FUSE-propagation fix
already in place. The reducer descends through a DP whose val is an
ASSIGN and then recursively reduces that ASSIGN's args, but something
in the GRAD backward chain never gives the reducer a shape that lets
forward progress resume — the program eventually just stops emitting
diagnostics and burns CPU. This also holds with Rule 2 paired: in the
test trace, no SUP ever reaches the ASSIGN's args, so Rule 2 never
fires.

### Status

- Rule 2 (`ASSIGN ⊳ SUP` on either arg) is **landed** in
  [src/interact/tensor_ops.c](../src/interact/tensor_ops.c). It is
  harmless in its inactive path, and it is the correct mirror of
  `log_go_0`'s `case SUP` for when the upstream Rule-1 discipline
  does put a SUP on an ASSIGN arg.
- Rule 1 (`DUP ⊳ UOP_ASSIGN = t`) is **not landed** — it deadlocks
  the test at n ≥ 2.

### Diagnosed deadlock (SEQ + Rule 1 version)

With `SEQ(assign_w, SEQ(assign_b, rec))` in the test AND the FUSE-pass-SEQ
rule (tensor_ops.c — don't absorb SEQ into a parallel KERNEL shell), n=1
completes correctly. n ≥ 2 exposes a stale-DP issue that is **independent
of Rule 1**.

Reproducer: `THVM_TRAIN_STEPS=2`, keep the default `DUP ⊳ TOP`
commutation, result is `SEQ@3184` with `seq.a = ASSIGN@3234`, dst =
`ERA` (not `TEN`).

Root cause, traced via `THVM_WATCH_LOC=...` and `THVM_WATCH_DUP=...`:

1. Step 2's `w_binder` is bound by APP-LAM to an `ALO@2309` whose
   `book_term` is the book's `DP0` for step 1's `w_next` (`DUP5`).
2. `thvm_alo_force` memoises the realized dynamic `DP0(38, 2372)` back
   into `heap[2309+0]`. This memoised DP token is a *linear* IC
   resource.
3. Step 2's body has multiple dynamic DUP cells that ultimately chase
   through the same `VAR@2317 → ALO@2309 → DP0(38, 2372)` path. Each
   apply-phase writeback stores the same `DP0(38, 2372)` in a
   *different* heap slot (`heap[2702]`, `heap[3021]`, `heap[3249]`, …).
4. `port_slot[2372, 0]` can only track ONE slot at a time. Each write
   re-registers it to the most-recent slot.
5. Step 1's `DUP5` at `2372` eventually fires with `val = TEN(3)` and
   writes the `TEN` to whichever slot `port_slot[2372, 0]` currently
   points at (observed: `heap[2702]`).
6. The other slots (`heap[3249]`, ...) keep their stale `DP0(38, 2372)`
   forever. When the reducer later visits them, it descends through
   the DP into `heap[2372] = ERA` (DUP5 already cleared itself), the
   `DUP ⊳ ERA` rule fires, and the slot becomes `ERA`.
7. Step 2's ASSIGN inherits that `ERA` as its `dst`, SEQ can't advance
   past a stuck `TOP/ASSIGN` with `dst=ERA`, eval returns the SEQ.

This is the same ALO-memoisation vs. DP-linearity problem identified in
[plans/recursive_grad_loop_fix.md](plans/recursive_grad_loop_fix.md) —
now isolated to a specific `(dup, slot1, slot2)` tuple. The only reason
it didn't break the partial pre-SEQ state is that the default
`DUP ⊳ TOP` commutation cloned the offending `ASSIGN` into two copies
with independent DP chains, so "one copy's stale DP" didn't block the
whole computation.

### Diagnosed deadlock (original Rule-1-only version)

With Rule 1 active, a single `DUP ⊳ ASSIGN` pair at
`(dup_loc=1356, assign_loc=1020)` repeats forever. The pair has:

- `ASSIGN.dst = TEN/3` (the `w` tensor, already resolved)
- `ASSIGN.src = SUB@1363` (the update expression, lazy compute)

The reducer's behavior step by step:

1. `DP0(dup_loc=1356)` enter → descends into `heap[1356] = ASSIGN@1020`.
2. `ASSIGN` is a direct uop → descends into `arg0 = TEN`. WNF, back to apply.
3. Apply-phase of ASSIGN frame stores dst ok, then inspects `arg1 = SUB`.
4. `SUB` is `TAG_TOP` but **not** a "direct uop" (SUB has no interact
   rule that yields `TEN` on its own). The reducer treats it as WNF
   without descending into its children.
5. ASSIGN's apply phase reconstructs the `TOP/ASSIGN` term and returns
   to the DP0 frame.
6. DP0 calls `DUP ⊳ ASSIGN` (our Rule 1) → `return t` (defer).
7. Loop — nothing changed.

Wrapping `SUB` in `FUSE` (either inside ASSIGN's interact or in the
reducer's apply phase) doesn't break the cycle: `FUSE` becomes the
arg1, the reducer descends into `FUSE(SUB)`, `SUB`'s children are DPs
of step 1's DUP chain, `reduce_fuse_payload_top_ready` reports
"not ready", FUSE doesn't fire. Those DP children point back at the
same DUP chain that Rule 1 is blocking — so the "local" fix still
has a global dependency on forward progress that Rule 1 forbids.

### Why the default `DUP ⊳ TOP` commutation is actually correct here

In `combinators.c`, the generic `DUP ⊳ TOP` rule already shares atomic
children (`TEN` / `NUM` / `ERA` / `ANY` / `CTR`) **by reference** and
only DP-wraps non-atomic children. For `ASSIGN(TEN, SUB)` that means
the two commuted ASSIGNs share the same `dst=TEN(3)` and each get a
`DP{0,1}`-wrapped projection of the compute tree in `src`:

```
DUP ⊳ ASSIGN(TEN(3), SUB(...))
  -->
  ASSIGN₀ = ASSIGN(TEN(3), DP0_L(SUB'))
  ASSIGN₁ = ASSIGN(TEN(3), DP1_L(SUB'))
```

Both fire on the same buffer with the same computed value — wasteful
(the blit runs twice) but **correct**, and crucially each copy's
`src` has its own DP projection that can resolve independently
without cycling through the dup we came from. The "not commute"
intuition is right for purely unique side effects, but for
idempotent self-writes through a shared `TEN` dst, commutation
produces the right IC semantics without extra rules.

### When Rule 1 becomes viable

Rule 1 is only safe once `ASSIGN`'s `src` can reach `TEN` without
depending on the same `DUP` that Rule 1 blocks. Adding `SEQ(ASSIGN,
rec)` on the test side (landed) makes n=1 work correctly — it forces
the ASSIGN to fire before the recursion reads. But n ≥ 2 still fails,
because at that depth the **ALO-memoisation + DP-linearity** bug (see
"Diagnosed deadlock (SEQ + Rule 1 version)" above) produces stale DP
tokens that cascade to `ERA`, poisoning step ≥ 2's ASSIGN `dst`.

The remaining fix is no longer about interaction rules — it's about
ALO force semantics:

1. **Don't memoise a linear DP token in `heap[alo_loc+0]`.** When
   `thvm_alo_force` realises a book `DP0`/`DP1` into a dynamic
   `DP0(lab, dyn_dup)`, it must NOT leave that token in the ALO's
   cell for subsequent readers to grab. Each read should either
   (a) see `ERA` (ALO was consumed once) and fail loudly, or
   (b) re-realise with a fresh dynamic DUP wrapping a shared atomic
   backing, so each consumer gets its own DP chain.
2. **Alternatively**, bookkeeping at the ALO level: track "this ALO
   wraps a linear term, it has been forced, the realized term is
   owned by consumer X" — and have subsequent consumers route through
   a DUP-split instead of sharing the memoised token.

This is an ALO-subsystem change, not a new interaction rule. It's
orthogonal to Rule 1 / Rule 2 and should land first. Once ALO stops
minting duplicate linear tokens, Rule 1 + SEQ should advance all
three test cases to full base-case `CTR` with mutated buffers.

## What this pattern is NOT

- **Not a short-circuit.** Both rules are ordinary local interaction
  rules, pattern-matching on already-computed arguments. There is no
  `thvm_eval(val)` call, no re-entry into the reducer from inside an
  interaction.
- **Not a masking.** We're not hiding an `ERA` (that's a separate rule
  — `ASSIGN ⊳ ERA`), nor are we quietly dropping a side effect.
- **Not a sweep.** `FUSE` propagates reachable compute from the root
  once the separate fix in `reduce_fuse_payload_ready` lets it walk
  through `APP`/`LAM`/`REF`/`ALO`/`VAR` to reach the real structure.

## See also

- [HVM4 `log` primitive](../HVM4/clang/prim/fn/log.c)
- [HVM4 `log_go_0` SUP case](../HVM4/clang/prim/fn/log_go_0.c)
- [HVM4 `wnf_dup_nod`](../HVM4/clang/wnf/dup_nod.c)
- [TinyHVM recursive GRAD loop plan](plans/recursive_grad_loop_fix.md)
