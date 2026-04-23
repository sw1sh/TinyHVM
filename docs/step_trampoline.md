# Step Trampoline, Walker, and Tracing (C ↔ WL)

This doc maps the three layers that let you single-step TinyHVM
reductions and render each step as a graph:

1. **wnf trampoline** (C) — fires interactions with a per-fire hook.
2. **Heap walker** (WL / `src/debug/graph.c`) — renders the subgraph
   reachable from a root.
3. **Step-graph session** (C) — drives a GRAD root through a two-phase
   reduction and snapshots the walker output on every hook-visible
   change.

A matching WL-side API — `TStep`, `TStepTrace`, `TINetGraph` — wraps
all three so Wolfram code can step through the same reductions.

## Reducer primitive

```c
Term thvm_reduce(TinyHVM *ctx, Term term);                      // reduce to WHNF
Term thvm_reduce_budget(TinyHVM *ctx, Term term, u32 budget);   // budgeted (0 = unlimited)
```

`thvm_reduce_budget` is the worker loop in [`src/wnf/_.c`](../src/wnf/_.c).
It maintains an eliminator stack and fires interactions through
`thvm_interact` ([`src/interact/_.c`](../src/interact/_.c)) until either
the term is WNF or `budget` is non-zero and has been consumed.
`thvm_reduce` is the fixed point (`budget = 0`).

After each interaction the wnf trampoline calls the optional per-step
hook — used by the step-graph session to emit a `.dot` per rule.

## Interaction primitive

```c
static Term thvm_interact(TinyHVM *ctx, Term t);   // src/interact/_.c
```

Dispatches on `term_tag(t)` into the appropriate rule file
(combinators, tensor ops, GRAD).  Before applying a rule, the wnf
layer records the active rule name, cursor location, principal-edge
locations, and (for GRAD rules) the firing GRAD cell — all readable
through the `thvm_wnf_last_*` accessors.

## Walker

Given a root term, walk the heap following arity-based child slots
and build a `{Nodes, Edges}` set:

- **C side**: `thvm_heap_dot_root` in [`src/debug/graph.c`](../src/debug/graph.c)
  renders to a `.dot` file.  `heap_dot_root_only = 1` restricts to
  nodes reachable from the passed root (skipping unrelated heap
  residue); `include_all = 1` also walks ancillary roots like
  `step_root_slot`.
- **WL side**: `heapWalk` in [`wl/Kernel/Heap.wl`](../wl/Kernel/Heap.wl)
  returns `<|"Nodes" -> <|key -> record|>, "Edges" -> {...}, "Root" -> key|>`.

Both walkers use the same keying convention:

- **Compound terms** (TOP, APP, LAM, …) are keyed by their args-base
  `val` (`h<val>`).  Two compound terms sharing the same heap args
  address are the *same* node.
- **Atoms** (TEN, NUM, ERA, REF, VAR, ANY) are keyed by content.
- **KERNEL** terms skip their metadata slots (root_uop NUM, ANY
  placeholder) when computing children — those are surfaced in the
  node label instead of as graph edges.

## Step-graph session (C, per-interaction tracing)

[`thvm_trace_step_graph_session`](../src/schedule/_.c) in
`src/schedule/_.c` runs a two-phase reduction when `THVM_STEP_GRAPH=1`:

```c
// Phase 1: GRAD commutes — FUSE kernelisation deferred.
g_thvm_defer_fuse_kernelize = 1;
traced = thvm_reduce(ctx, traced);
traced = thvm_normalize(ctx, traced);   // deep-WHNF every reachable arg slot
g_thvm_defer_fuse_kernelize = 0;

// Phase 2: explicit FUSE fixed-point — kernelisation fires.
if (!getenv("THVM_STEP_GRAPH_NO_FUSE"))
    traced = thvm_eval_fuse_fixed_point(ctx, traced, ...);
```

The per-interaction hook (`wnf_step_session_hook`) writes one `.dot`
per wnf rule fire, dedupes by walker signature, and renames each
pending frame with the rule that just fired (e.g.
`step_001_GRAD-MUL.dot`, `step_005_FUSE-SUM.dot`).  On each callback
it:

1. Mirrors the current root term into `step_root_slot` so the dumper
   picks up root-shaped TOPs like GRAD.
2. Reads `thvm_wnf_last_rule()`, `thvm_wnf_last_interact_pair()`,
   and the principal-edge slot of the cell that's about to fire
   next.
3. Rewrites the previous pending `.dot` in place to highlight the
   about-to-fire redex edge (edits the file's edge attributes), then
   renames it with the firing rule's name.
4. Writes a new pending `.dot` reflecting the post-interaction state.

When the session ends, the final pending frame is overwritten with
the truly-final reduced state and renamed `step_%03u_final.dot`.
PNGs are rendered via `dot -Tpng -Gdpi=150` unless
`THVM_STEP_GRAPH_NO_PNG=1`.

For GRAD roots specifically, the session wraps the root in dual
`FUSE_f` (forward) + `FUSE_b` (backward) heap cells before driving
reduction — see [`step_graph_ic_goal.md`](step_graph_ic_goal.md) for
the topology and the canonical 10-step trace.

## Step-graph API (WL, visible-change stepping)

C-level per-interaction is finer than we want from WL: many
interactions don't alter the walker-visible tree at all (they shuffle
heap internals).  The WL primitive skips over those:

```wolfram
TStep[t, maxAttempts:100]       (* → {nextState, interactionsFired} *)
TStepTrace[t, maxSteps:50]      (* → list of snapshot records *)
```

Backed by C:

```c
EXTERN_C int thvmStepToNextVisible(WolframLibraryData libData,
                                   mint argc, MArgument *args, MArgument res);
```

([`wl/CSource/tinyhvmlink.m:1632`](../wl/CSource/tinyhvmlink.m)),
which loops `thvm_reduce_budget(ctx, t, 1)` until a tree-hash of the
walker-visible structure changes, the reducer stalls, or
`maxAttempts` is exceeded.

`TStepTrace` calls `TStep` in a loop and **snapshots the walker
output at each step** (`<|"Term", "Walk"|>`).  Without the snapshot,
later reductions that mutate heap would invalidate the earlier
captured term-ids — they point to term slots whose contents are no
longer what they were at capture time.  With the snapshot in hand,
`TINetGraph[snapshot]` renders the exact graph that existed at that
step, regardless of subsequent reductions.

`TStepTrace` stops on:

- fixed point (`TStep` fires zero interactions),
- dispatch reached (the result materialized to a `TAG_TEN` — that
  step is *not* included, because the user asked for public-graph
  reduction without materialization), or
- `maxSteps` exceeded.

## Rendering

```wolfram
TINetGraph[t_TTensor]                 (* walks live heap, then renders *)
TINetGraph[snapshot_?AssociationQ]    (* renders from pre-captured walk *)
TINetGraph[walk_?AssociationQ]        (* renders raw walker output *)
```

All three produce a WL `Graph` with manual `VertexCoordinates`
(top-down BFS layers from the free-out anchor down to leaves),
`graph.c`-style palette (`#cce5ff` TOP, `#ccffcc` KERNEL, `#e0e0e0`
TEN), port labels (`a`, `b`, `in`, …), `@slot` tail labels on
tensor-sourced edges, and a red edge/node highlight for the active
pair when a `TermId` option is passed.

## C ↔ WL correspondence summary

| C | WL | What it does |
|---|----|-------------|
| `thvm_reduce_budget(t, n)` | *(internal)* | Fire ≤ n interactions; return result. |
| `thvm_reduce(t)` | *(drives `TGet`, `TEval`)* | Fire until WHNF. |
| `thvm_eval(t)` | `TGet[t]` | Full FUSE + dispatch + passes. |
| `thvmStepToNextVisible(t, m)` | `TStep[t, m]` | Fire until walker-visible change. |
| *session loop*                | `TStepTrace[t]` | Per-visible-step trace + snapshots. |
| `thvm_trace_step_graph_session` | — | C-side two-phase GRAD session → `.dot` files. |
| `thvm_heap_dot_root` | `TINetGraph` / `heapWalk` | Walker + renderer. |

## Env vars (C-side tracing)

See [`docs/env.md`](env.md) for the canonical list.  The step-graph
session honors:

- `THVM_STEP_GRAPH=1` — enables the per-interaction dump path inside
  `thvm_eval` for GRAD roots.
- `THVM_STEP_GRAPH_DIR=<path>` — directory for `step_*.dot` /
  `step_*.png` (defaults to the step-graph's working directory).
- `THVM_STEP_GRAPH_NO_PNG=1` — write only `.dot`, skip rasterization.
- `THVM_STEP_GRAPH_MAX=<n>` — cap number of step snapshots (default
  `512`).
- `THVM_STEP_GRAPH_NO_FUSE=1` — skip phase-2 FUSE kernelisation;
  stop after GRAD commutes.
- `THVM_GRAPH_STOP_AFTER_SWEEP=1` — return the post-sweep topology
  without a final `thvm_normalize`; useful when inspecting the exact
  FUSE-sweep state.
- `THVM_INTERACT_TRACE=1` — per-interaction rule-name trace to
  stderr (independent of step-graph dumping).

## Invariants worth knowing

- `step_root_slot` is a dedicated heap cell mirroring the current root
  term.  The dumper reads it so root-shaped TOPs (GRAD, FUSE) are
  included in the walk even when they aren't held in any other heap
  cell.  The session keeps it current after each hook fire.
- A GRAD root is wrapped in `FUSE_f` (forward) + `FUSE_b` (backward)
  before driving.  Compute ops without GRAD are WNF in phase-1
  without a FUSE wrapper, so reducing an unwrapped `ADD+MUL` fires
  zero interactions.
- The reducer is re-entrant (`reduce_depth` thread-local) but the
  step-graph globals (`g_step_session_*`) are single-session — don't
  run two tracers in parallel.
- KERNEL-KERNEL absorption is **inlined** inside
  `thvm_public_kernel_absorb_child` when a new KERNEL is constructed
  (see `src/interact/_.c`).  "Two kernels merging" is therefore a
  single step with a chained rule label, not two separate frames.
