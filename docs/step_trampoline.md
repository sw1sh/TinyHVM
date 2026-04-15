# Step Trampoline, Walker, and Tracing (C ↔ WL)

This doc maps the three layers that let you single-step TinyHVM reductions
and render each step as a graph:

1. **Reducer trampoline** (C) — fires interactions.
2. **Heap walker** (WL / dump.c) — renders the tree reachable from a root.
3. **Step-graph tracer** (C) — loops the reducer at budget-1 granularity and
   snapshots the walker output on every *visible* change.

A matching WL-side API — `TStep`, `TStepTrace`, `TDotGraph` — wraps all three
so Wolfram code can step through the same reductions dump.c traces from C.

## Reducer primitive

```c
Term thvm_reduce_steps(TinyHVM *ctx, Term t, u32 max_steps);
Term thvm_reduce(TinyHVM *ctx, Term t);   // == thvm_reduce_steps(ctx, t, 0)
```

`thvm_reduce_steps` is the worker loop. It maintains an eliminator stack
([src/reduce/_.c:338](src/reduce/_.c#L338)) and fires interactions through
`thvm_interact` until either (a) the term is WNF, or (b) `max_steps` is
non-zero and has been consumed. `max_steps = 0` means *unlimited*, so
`thvm_reduce` is literally its fixed point — no extra wrapper function.

Each fired interaction passes through the `TRACE_STEP(before, result)` macro
([src/reduce/_.c:361](src/reduce/_.c#L361)), which updates
`ctx->steps_taken`, the interaction trace buffer, and — when step-graph
tracing is active — the per-interaction dump hook.

## Interaction primitive

```c
static Term thvm_interact(TinyHVM *ctx, Term t);   // src/interact/_.c
```

Dispatches on `term_tag(t)` into the appropriate rule file (combinators,
tensor ops, GRAD). Entry point calls the pre-interaction step-graph hook so
the dumper can read `before`-redex metadata (GRAD y, ERA payload, TOP
era/add-zero args) from *live heap* before the interaction mutates it.

## Walker

Given a root term, walk the heap following arity-based child slots and build
a set of `{Nodes, Edges}`:

- **C side**: [src/debug/graph.c `thvm_heap_dot_root`](src/debug/graph.c)
  renders to a `.dot` file. `heap_dot_root_only = 1` restricts to nodes
  reachable from the passed root (skipping unrelated heap residue).
- **WL side**: [wl/Kernel/Heap.wl `heapWalk`](wl/Kernel/Heap.wl) returns
  `<|"Nodes" -> <|key -> record|>, "Edges" -> {...}, "Root" -> key|>`.

Both walkers use the same keying convention:

- **Compound terms** (TOP, APP, LAM, …) are keyed by their args-base `val`
  (`h<val>`). Two compound terms sharing the same heap args address are
  the *same* node.
- **Atoms** (TEN, NUM, ERA, REF, VAR, ANY) are keyed by content.
- **KERNEL** terms skip their metadata slots (root_uop NUM, ANY placeholder)
  when computing children — those are surfaced in the node label instead of
  as graph edges.

## Step-graph session (C, per-interaction tracing)

[`thvm_trace_step_graph_session`](src/schedule/_.c) runs a budget-1 tracer
loop when `THVM_STEP_GRAPH=1`:

```c
for each step:
    1. predicted = thvm_phase1_predict_next_redex(ctx, t)
    2. capture_step_before_meta(ctx, predicted)        // pre-fire metadata
    3. r = thvm_reduce_steps(ctx, t, 1)                // fires one interaction
    4. t = r;  heap_set(phase1_root_slot, t)           // update root mirror
    5. source_slot = phase1_graph_source_slot(ctx, phase1_root_slot, t, before)
    6. thvm_step_graph_after_interaction(ctx, source_slot, before, t)
```

After-interaction writes `step_%03u_<shown>.dot`, deduping by structural
signature. The budget forces the reducer to unwind to the outer root after
each fired interaction, which is what the dumper's highlighter expects.
When the interaction produces a `TAG_TEN` root, the file is named
`step_%03u_state_final.dot` and `finalize()` skips writing a redundant
terminal.

The heap-sweep and predictor-fallback paths (firing interactions on
heap-resident agents that aren't reached from the root) run inside the same
loop and exist to capture administrative reductions the reducer's natural
traversal doesn't touch.

## Step-graph API (WL, visible-change stepping)

C-level budget-1 is finer than we want from WL: many interactions don't
alter the walker-visible tree at all (they shuffle heap internals). The WL
primitive skips over those:

```wolfram
TStep[t, maxAttempts:100]       (* → {nextState, interactionsFired} *)
TStepTrace[t, maxSteps:50]      (* → list of snapshot records *)
```

Backed by C:

```c
EXTERN_C int thvmStepToNextVisible(WolframLibraryData libData,
                                   mint argc, MArgument *args, MArgument res);
```

which loops `thvm_reduce_steps(ctx, t, 1)` until `thvm_walker_sig(ctx, t)`
— a tree-hash of the walker-visible structure — changes, or the reducer
stalls, or `maxAttempts` is exceeded.

`TStepTrace` calls `TStep` in a loop and **snapshots the walker output at
each step** (`<|"Term", "Walk"|>`). Without the snapshot, later reductions
that mutate heap would invalidate the earlier captured term-ids — they point
to term slots whose contents are no longer what they were at capture time.
With the snapshot in hand, `TDotGraph[snapshot]` renders the exact graph
that existed at that step, regardless of subsequent reductions.

`TStepTrace` stops on any of:

- fixed point (`TStep` fires zero interactions)
- dispatch reached (`tag == TAG_TEN` — the result has materialized to a
  tensor; the step that caused this is *not* included, because the user
  asked for pure public-graph reduction, no materialization)
- `maxSteps` exceeded

## Rendering

```wolfram
TDotGraph[t_TTerm]                    (* walks live heap, then renders *)
TDotGraph[snapshot_?AssociationQ]     (* renders from pre-captured walk *)
TDotGraph[walk_?AssociationQ]         (* renders raw walker output *)
```

All three produce a WL `Graph` with manual `VertexCoordinates` (top-down
BFS layers from the free-out anchor down to leaves), dump.c-style palette
(`#cce5ff` TOP, `#ccffcc` KERNEL, `#e0e0e0` TEN), port labels (`a`, `b`,
`in`, …), `@slot` tail labels on tensor-sourced edges, and a red edge/node
highlight for the active pair when `TermId` was passed through to allow
`thvm_next_interaction` lookup.

## C ↔ WL correspondence summary

| C | WL | What it does |
|---|----|-------------|
| `thvm_reduce_steps(t, n)` | `TReduceSteps[t, n]` | Fire ≤ n interactions; return result + count. |
| `thvm_reduce(t)` | *(built into TGet + global eval)* | Fire until WNF. |
| `thvm_eval(t)` | `TGet[t]` forces it via to-host path | Full FUSE + dispatch + passes. |
| `thvmStepToNextVisible(t, m)` | `TStep[t, m]` | Fire until walker-visible change. |
| *structural_nf session loop* | `TStepTrace[t]` | Per-visible-step trace + snapshots. |
| `thvm_heap_dot_root` | `TDotGraph` / `heapWalk` | Walker + renderer. |
| `thvm_step_graph_after_interaction` | — | Dump `.dot` for one C-side trace step. |

## Env vars (C-side tracing)

- `THVM_STEP_GRAPH=1` — enables the per-interaction dump path inside
  `thvm_eval`.
- `THVM_STEP_GRAPH_DIR=<path>` — directory for `step_*.dot` / `step_*.png`
  (defaults to current working directory).
- `THVM_STEP_GRAPH_NO_PNG=1` — write only `.dot`, skip rasterization.
- `THVM_STEP_GRAPH_MAX=<n>` — cap number of step snapshots (default 512).

## Invariants worth knowing

- `phase1_root_slot` is a dedicated heap cell mirroring the current root
  term. The dumper reads it; the tracer keeps it current via
  `heap_set(phase1_root_slot, t)` after each fired interaction.
- The tracer always wraps the input term in `UOP_FUSE` before driving it.
  Compute ops (ADD/MUL/SUM/etc.) are WNF in phase-1 without a FUSE wrapper,
  so an unwrapped ADD+MUL reduces to zero interactions.
- The reducer is re-entrant (`reduce_depth` thread-local) but the step-graph
  globals (`step_graph_active`, `phase1_root_slot`, `step_graph_n`, …) are
  single-session — don't run two tracers in parallel.
- KERNEL-KERNEL absorption is *inlined* inside `thvm_public_kernel_absorb_child`
  when a new KERNEL is constructed (see [src/interact/_.c:154](src/interact/_.c#L154)).
  This means the "two kernels merging" is not a separate TRACE_STEP — it's
  part of the construction of the outer kernel. Traces will show this as a
  single step with a chained label (`MUL+ADD`).
