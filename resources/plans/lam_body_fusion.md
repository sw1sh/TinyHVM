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
- **VAR is unbound until beta.** `thvm_lam` writes `term_set_sub(var)` at the
  binder slot ([src/inet/_.c:5](src/inet/_.c#L5)); APP-LAM is the only site
  that overwrites it ([src/interact/combinators.c:53-54](src/interact/combinators.c#L53-L54)).
- **Fuser bails on unbound VAR.** `fuse_walk_inner` dereferences substituted
  VARs, but on unbound VAR returns `-1` ([src/fuse/_.c:414-422](src/fuse/_.c#L414-L422)).
- **No shape on VAR slot.** `term_view(VAR)` reads `st_get(var_loc)`
  ([src/ctx/init.c:182-187](src/ctx/init.c#L182-L187)); nothing writes there
  today, so `thvm_track_top_shape` skips every UOP whose input is the VAR
  (all shape-composing branches gate on `va != NULL`).
- **KernelEntry has no PARAM leaf kind.** `KernelLeafKind` is
  `{NONE, TENSOR, NUM}` ([src/tinyhvm.h:720-724](src/tinyhvm.h#L720-L724)).
  Leaves are concrete `leaf_ids[i] = tid` or immediates.

Net: LAM bodies with UOPs are fully dormant. There is no representation for
"the i-th kernel input is a formal parameter of shape S and dtype D".

## Design

Two new pieces: **typed binders** (shapes on VARs) and **parametric kernels**
(a new leaf kind).

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

### 2. KERNEL_LEAF_PARAM

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

`fuse_walk_inner` gets one new case:

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

### 3. Kernel construction under a LAM

`fuse_build_kernel` stays shape-agnostic: it sees PARAM leaves with concrete
views exactly like it sees TENSOR leaves. The codegen (Metal MSL / CPU
interpreter) already consumes `leaf_views[i]` + `leaf_sts[i]`, so PARAM leaves
compile identically.

Two placements for the resulting KERNEL term:

- **Lazy placement (preferred).** A new `scheduler pass` (or `sched_one`
  extension) walks into LAM bodies once, builds the kernel, and stashes it
  **inside** the LAM as a KERNEL term whose formal leaves point back at the
  enclosing VAR(s). The LAM now reads `LAM<VAR_typed, KERNEL(..., params=[x])>`.
- **Eager placement.** Add a `thvm_compile_lam(ctx, lam_term)` entry point
  callable from user code (and from JIT capture). Same resulting shape, just
  user-driven.

Start with eager — it's a simpler invariant and unblocks JIT re-use.

### 4. APP-LAM for compiled bodies

When APP-LAM fires and `body = KERNEL(..., params=[x0, x1, ...])`, do **not**
heap-substitute the VAR. Instead:

```c
// combinators.c — APP-LAM, body = KERNEL
if (term_tag(body) == TAG_TOP && term_ext(body) == UOP_KERNEL) {
    // Collect formal params (VARs of enclosing LAMs).
    // Build a PARAM binding list: param_idx -> arg_tid.
    // Rewrite to a new KERNEL term with PARAM leaves resolved to TENSOR leaves.
    // (Or: carry a binding side-table on the KernelEntry; dispatch reads it.)
}
```

Two implementation choices:

- **Specialise-on-apply.** Clone the KernelEntry with PARAMs rewritten to
  TENSOR leaves carrying the app-time tid. One cache hit per shape class,
  zero re-fusion.
- **Late-bind.** Keep PARAM leaves in the cached entry; dispatch takes an
  `{param_idx -> buf_id}` map. One KernelEntry per body, many call sites.

Late-bind matches the "one JIT cache entry, reusable across steps" goal
(`unified_codegen.md`). Specialise-on-apply is easier to build first.

### 5. Multi-arg LAMs (curried)

`λx. λy. body` nests typed binders; PARAM leaves carry de-Bruijn indices.
Nothing new in the fuser — it just walks into multiple LAMs and assigns
params sequentially. APP-LAM binds one level at a time (as it does now).

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

1. **M1 — Typed binders.** `thvm_lam_typed`, VAR-slot ST population, dtype
   side-table. `thvm_track_top_shape` starts assigning STs to UOPs inside LAM
   bodies. No codegen changes yet. **Exit criterion:** a unit test reads
   `st_get(top_loc)` for a TOP inside a LAM body and gets the right view.
2. **M2 — PARAM leaf kind.** Extend `KernelLeafKind`, teach
   `fuse_append_leaf_*` to emit PARAM leaves, update kernel signature hashing
   (so PARAM-leaf kernels cache under their own key). **Exit:** `fuse_walk_inner`
   on a LAM body returns a leaf set containing PARAM leaves without `-1`.
3. **M3 — Eager compile.** `thvm_compile_lam(ctx, lam)` builds a KernelEntry
   and stores it as the LAM body. **Exit:** a two-op body (e.g. `x*x + x`)
   compiles to one fused KernelEntry whose n_leaves=1 and leaf_kinds[0]=PARAM.
4. **M4 — APP-LAM specialise-on-apply.** APP-LAM onto a KERNEL body rewrites
   PARAM leaves to TENSOR leaves at bind time. **Exit:** applying the compiled
   LAM twice with different inputs produces correct outputs, no re-fusion.
5. **M5 — Late-bind dispatch.** Dispatch accepts `{param_idx -> buf_id}`
   without cloning the entry. **Exit:** one KernelEntry in the cache serves N
   apps with different arg tids.
6. **M6 — JIT integration.** Training step captures to a single LAM; each
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
