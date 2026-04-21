# LAM-Body Fusion: Typed Binders + Parametric Kernels

## Status: Proposal

Goal: make the body of `λx. <tensor-expression>` a first-class fusable kernel —
compiled, cached, and dispatchable — **before** any argument is supplied. Today
LAM is an opaque WHNF atom; its body is frozen until APP-LAM substitutes `x`.

## Motivation

Three things line up:

1. **JIT / function re-use.** A training step is the same LAM applied to new
   batches/weights. If we could compile the body once as a kernel parameterised
   over its binders, we'd skip per-step fuser work entirely.
2. **HVM-style function compilation.** HVM4 now compiles IC functions with
   SUPs to machine code (labeled_sup_types.md §SupGen evolution). For TinyHVM,
   the tensor analogue is "compile a LAM body to a KernelEntry with formal
   parameter leaves."
3. **`symbolic_shape_backward_fusion.md` ceiling.** Dispatch-count work shows
   the remaining gap is structural, not per-kernel. Parametric kernels let us
   express "the whole step is one re-usable kernel DAG" instead of rebuilding
   it each iteration.

## Current Behavior (as of 2026-04-21)

Verified against source:

- **Reducer / scheduler treat LAM as an atom.** LAM is WHNF in
  `step_fuse_payload_ready` ([src/schedule/_.c:756](src/schedule/_.c#L756)), in
  the `thvm_step_predict_next_redex` switch ([src/schedule/_.c:917](src/schedule/_.c#L917)),
  and in the "no global cleanup" set ([src/schedule/_.c:512](src/schedule/_.c#L512)).
  Nothing recurses into the body.
- **Fusion happens by interaction rewriting, not by walking.** `UOP_FUSE` is a
  propagating agent ([src/interact/tensor_ops.c:340](src/interact/tensor_ops.c#L340)):
  it passes through atoms, distributes through CTR, wraps ASSIGN's src, and on
  a compute TAG_TOP calls `thvm_fuse_public_term` ([src/interact/_.c:215](src/interact/_.c#L215))
  which produces either a growing KERNEL (children still FUSE-wrapped) or a
  monolithic KERNEL (all children locally ready). No FUSE rule exists for
  LAM — default is pass-through ([src/interact/tensor_ops.c:431-432](src/interact/tensor_ops.c#L431-L432)).
- **`fuse_walk_inner` / `fuse_build_kernel` is the lowering pass**, not the
  fuser ([src/fuse/_.c:1017](src/fuse/_.c#L1017)). It's called from
  `thvm_kernel_register` ([src/interact/_.c:567](src/interact/_.c#L567)) once
  a KERNEL DAG has been built by FUSE, flattening it into a `KernelEntry` for
  codegen. It walks a *clean* DAG of compute TOPs + TEN/NUM leaves — not raw
  structure with unreduced VARs.
- **VAR is unbound until beta.** `thvm_lam` writes `term_set_sub(var)` at the
  binder slot ([src/inet/_.c:5](src/inet/_.c#L5)); APP-LAM is the only site
  that overwrites it ([src/interact/combinators.c:53-54](src/interact/combinators.c#L53-L54)).
- **No shape on VAR slot.** `term_view(VAR)` reads `st_get(var_loc)`
  ([src/ctx/init.c:182-187](src/ctx/init.c#L182-L187)); nothing writes there
  today, so `thvm_track_top_shape` skips every UOP whose input is the VAR
  (all shape-composing branches gate on `va != NULL`).
- **KernelEntry has no PARAM leaf kind.** `KernelLeafKind` is
  `{NONE, TENSOR, NUM}` ([src/tinyhvm.h:720-724](src/tinyhvm.h#L720-L724)).
  Leaves are concrete `leaf_ids[i] = tid` or immediates.

Net: LAM bodies with UOPs are fully dormant because **no FUSE rule propagates
into a LAM**, and even if one did, VARs have no shape and the lowering pass
has no way to emit a formal-parameter leaf.

## Design

Three pieces, split along the existing fuser-vs-lowering boundary:

1. **Typed binders** — shapes on VARs (affects term construction + ST table).
2. **New FUSE interaction rules** — FUSE ⊳ LAM and FUSE ⊳ APP(LAM,·) so the
   existing rewrite-driven fuser can enter a LAM body or absorb an
   APP(LAM, arg) pre-beta. This is the main architectural change.
3. **Parametric kernels** — a new `KERNEL_LEAF_PARAM` leaf kind, read only by
   the lowering pass (`fuse_walk_inner`) when it sees a typed unbound VAR.

### 1. Typed binders

```c
// inet/_.c — new constructor; existing thvm_lam forwards with no type
Term thvm_lam_typed(TinyHVM *ctx, Term *var_out, View var_view, u32 var_dtype,
                    Term body);
```

Semantics:
- Same heap layout as `thvm_lam` (2-cell: sub-VAR, body).
- Additionally: `st_set(var_loc, &var_view)` so `term_view(VAR)` returns
  `&var_view` until APP-LAM substitutes the slot.
- A parallel table (or re-use the ST table with a dtype side-channel) stores
  the VAR's dtype, because `ShapeTracker` doesn't carry one.

Optionally: `thvm_lam(...)` becomes `thvm_lam_typed` with an "unknown" view
(`rank = 0`). Unknown-shape LAMs fall back to the current dormant behavior —
no regressions.

**`thvm_track_top_shape` then works unchanged.** It already reads
`term_view(a)` (including VAR) and `st_get_tracker(var_loc)` via the VAR path
— so typed binders light up the whole downstream ST composition for free.

### 2. New FUSE interaction rules

#### 2a. FUSE ⊳ APP(LAM, arg) — pre-beta fusion (start here)

Add a case to `UOP_FUSE` in [tensor_ops.c:340+](src/interact/tensor_ops.c#L340)
that pattern-matches `FUSE(APP(LAM<x,body>, arg))` and, instead of forcing
beta-then-fuse, substitutes `x ← arg` directly and wraps the result:

```c
// New case in UOP_FUSE handler, before "Default: pass through"
if (ptag == TAG_APP) {
    u64 aloc = term_val(payload);
    Term fun = heap_read(ctx, aloc + 0);
    Term arg = heap_read(ctx, aloc + 1);
    if (term_tag(fun) == TAG_LAM) {
        u64 lam_loc = term_val(fun);
        Term body = heap_read(ctx, lam_loc + 1);
        // substitute x ← arg (overwrite VAR slot, same as APP-LAM body)
        heap_set(ctx, lam_loc + 0, arg);
        // re-wrap the substituted body in FUSE and re-reduce
        u64 fl = heap_alloc(ctx, 1);
        heap_set(ctx, fl, body);
        RETURN_REDUCED(term_new(TAG_TOP, UOP_FUSE, fl));
    }
}
```

This is one new rule, ~10 lines. Effect: if a LAM is immediately applied, FUSE
sees the APP, binds the VAR, and then the existing FUSE-TOP rule handles the
body exactly as it would after ordinary beta. Advantage vs. letting regular
APP-LAM fire first: the rewrite happens inside a FUSE context so children stay
FUSE-wrapped as they enter the body — no "unFUSE then re-FUSE" churn.

Does NOT require typed binders or PARAM leaves. This alone might be enough
for the JIT / training-step case where every LAM is immediately applied.

#### 2b. FUSE ⊳ LAM — enter an unapplied body (requires §1 typed binders)

Only useful when the LAM is NOT about to be applied (e.g. pre-compiling a
function). The rule propagates FUSE under the binder:

```c
if (ptag == TAG_LAM) {
    // Only enter if the VAR has a typed binding (shape available).
    u64 lam_loc = term_val(payload);
    Term var_slot = heap_read(ctx, lam_loc + 0);
    if (term_is_sub(var_slot) && st_get(lam_loc) != NULL) {
        Term body = heap_read(ctx, lam_loc + 1);
        u64 fl = heap_alloc(ctx, 1);
        heap_set(ctx, fl, body);
        heap_set(ctx, lam_loc + 1, term_new(TAG_TOP, UOP_FUSE, fl));
        RETURN_REDUCED(payload);  // LAM<x, FUSE(body)>
    }
    // Untyped binder → pass through (current behavior)
    RETURN_REDUCED(payload);
}
```

Now the body's TOPs reduce under FUSE; unbound typed VARs become leaves via §3.

### 3. KERNEL_LEAF_PARAM (only needed for §2b)

```c
typedef enum {
    KERNEL_LEAF_NONE   = 0,
    KERNEL_LEAF_TENSOR = 1,
    KERNEL_LEAF_NUM    = 2,
    KERNEL_LEAF_PARAM  = 3,   // formal kernel input: index into app-time bindings
} KernelLeafKind;
```

`leaf_ids[i]` for a PARAM leaf holds the **parameter index** (0-based,
de-Bruijn-ish — the nearest enclosing LAM is index 0).

In the lowering pass `fuse_walk_inner` ([src/fuse/_.c:415](src/fuse/_.c#L415)):

```c
if (term_tag(t) == TAG_VAR) {
    Term sub = heap_read(ctx, term_val(t));
    if (!term_is_sub(sub))
        return fuse_walk_inner(ctx, sub, ...);      // existing path
    // NEW: unbound VAR with a typed binder → PARAM leaf
    const View *vv = st_get(term_val(t));
    if (vv) return fuse_append_leaf_param(...);
    return -1;                                      // untyped → still bail
}
```

Codegen stays shape-agnostic: PARAM leaves carry `leaf_views[i]` + `leaf_sts[i]`
exactly like TENSOR leaves, so Metal MSL / CPU interpreter paths need no
changes.

### 4. APP-LAM for compiled bodies

When APP-LAM fires and the LAM's body is `KERNEL(..., params=[x0,x1,...])`
(produced by §2b's path), don't heap-substitute the VAR — rewrite to a
KERNEL with PARAMs bound:

```c
// combinators.c — APP-LAM, body = KERNEL
if (term_tag(body) == TAG_TOP && term_ext(body) == UOP_KERNEL) {
    // Build a PARAM binding: param_idx → arg_tid.
    // Option A (specialise-on-apply): clone the KernelEntry with PARAM
    //   leaves rewritten to TENSOR leaves carrying the app-time tid.
    // Option B (late-bind): keep PARAM leaves in the cached entry;
    //   dispatch takes a {param_idx → buf_id} map.
}
```

Specialise-on-apply is easier to land first; late-bind is the cache-efficient
end state (matches the "one JIT cache entry per body" goal).

### 5. Multi-arg LAMs (curried)

`λx. λy. body` nests typed binders; PARAM leaves carry de-Bruijn indices. The
FUSE ⊳ LAM rule (§2b) just fires twice, once per binder level. APP-LAM binds
one level at a time (as it does now).

## What this does NOT cover

- **Closures that capture tensors from the enclosing scope.** Those show up as
  ordinary TENSOR leaves (the capture is a concrete tid) — still works, but
  the cached kernel's reuse is limited to runs with the same captured tids.
  If we want true closures, add `KERNEL_LEAF_CAPTURE` (upvar-ish) as a future
  extension.
- **Reduction inside LAM bodies prior to APP-LAM.** Scheduler changes here
  are out of scope: we only need fusion to *build* the kernel; reduction of
  the KERNEL itself is driven by APP-LAM as described.
- **GRAD-through-LAM.** GRAD already walks provenance on materialised tensors
  — a LAM-body kernel, once applied, materialises like any other kernel.
  Compiling GRAD into a parametric kernel (dL/dparam as another LAM body) is
  a follow-up.

## Interactions with existing plans

| Plan | Relationship |
|------|--------------|
| `symbolic_shape_backward_fusion.md` | Same goal (bigger kernels, fewer dispatches) but via typed LAMs instead of backward-only ST propagation. Compatible. |
| `lazy_graph_compiler.md` / `unified_codegen.md` | Direct prerequisite: the "one JIT codegen function" is what the compiled LAM body targets. PARAM leaves slot into the unified codegen with no new leaf dispatch. |
| `labeled_sup_types.md` §ICC BRI/ANN (Phase 4) | BRI uses the same binder/VAR machinery. Typed binders generalise — add a dtype/shape to the VAR slot and BRI picks it up identically. |
| `kernel_dag_redesign.md` | LAM-body kernels are just KERNEL nodes living under a LAM; the DAG rules are unchanged. |

## Milestones

The pre-beta fusion rule (§2a) can ship on its own and may already close a
large fraction of the gap for the JIT case. Typed binders (§1, §2b, §3, §4)
only matter when the LAM must be compiled before being applied.

1. **M1 — FUSE ⊳ APP(LAM,·) interaction rule** (§2a). Single rule in
   [tensor_ops.c:340](src/interact/tensor_ops.c#L340). No new term tags, no
   KernelEntry changes. **Exit:** `FUSE(APP(LAM<x, x*x>, T))` reduces to a
   KERNEL whose compute is `MUL(T, T)` in one step, without a separate
   beta-then-fuse round.
2. **M2 — Typed binders** (§1). `thvm_lam_typed`, VAR-slot ST population,
   dtype side-table. `thvm_track_top_shape` starts composing STs through
   unbound typed VARs. **Exit:** a unit test reads `st_get(top_loc)` for a
   TOP inside a LAM body and gets the right composed view.
3. **M3 — FUSE ⊳ LAM propagation** (§2b). New interaction rule that enters
   typed LAMs. **Exit:** `FUSE(LAM<x_typed, x*x>)` reduces to
   `LAM<x_typed, KERNEL(MUL(x,x))>` with `x` still a typed unbound VAR inside
   the growing kernel.
4. **M4 — KERNEL_LEAF_PARAM** (§3). Extend `KernelLeafKind`, teach
   `fuse_walk_inner` to emit PARAM leaves when it hits a typed unbound VAR,
   update signature hashing. **Exit:** M3's kernel lowers to a `KernelEntry`
   with `n_leaves=1`, `leaf_kinds[0]=PARAM`, `leaf_views[0]=var_view`.
5. **M5 — APP-LAM specialise-on-apply** (§4). APP-LAM onto a KERNEL body
   rewrites PARAM leaves to TENSOR leaves at bind time. **Exit:** applying a
   compiled LAM twice with different inputs produces correct outputs without
   re-running the fuser.
6. **M6 — Late-bind dispatch.** Dispatch accepts `{param_idx → buf_id}`
   without cloning the entry. **Exit:** one `KernelEntry` in the cache serves
   N apps with different arg tids.
7. **M7 — JIT integration.** Training step captures to a single LAM; each
   iteration is one APP-LAM dispatch. **Exit:** measured dispatch count per
   step drops from ~375 to ~(1 + reduce count) on a small MLP.

## Risks / open questions

- **Dtype side-table.** ShapeTracker doesn't carry a dtype; adding one to
  `st_set_tracker` touches every ST write. Cheapest: a parallel `st_dtypes[]`
  table keyed identically.
- **Shape-polymorphic LAMs.** If a LAM is called with different input shapes
  (different batch sizes), the cached kernel is invalid. Either re-use the
  existing shape-keyed JIT cache (one entry per concrete input shape) or
  introduce symbolic shapes on binders. Start with shape-concrete.
- **Buffer aliasing under late-bind.** The memory planner
  (`memory_planner.md`) assigns buffer reuse based on a specific dispatch
  order; late-binding different tids to the same KernelEntry must not
  confuse the planner. Likely needs a per-app plan patch at bind time.
- **DUP across LAMs.** A LAM whose body DUPs the parameter
  (`λx. x + x`) goes through existing DUP-VAR machinery. PARAM leaves see
  the DUP as a boundary (same as TENSOR leaves today) — verify the
  atom-share DUP⊳TOP projection still works when the atom is a PARAM.
