# Step Graph Visualization — Handoff

## Status

The reduce loop (`test_loop_assign_simple`) works correctly for n=0..5.
The step graph checker PASSES for n=1 and n=2. Phase 1 step graphs are
generated and phase 2 (FUSE+dispatch) runs after to produce correct results.

## What the user wants fixed

The user has repeatedly asked for these specific things and been frustrated
that they keep getting partially addressed or papered over. **Read each one
carefully and fix the root cause, not the symptom.**

### 1. Definition body should show the LAM structure

Current: `REF → def#0 LAM → (body IFZ)` where `def#0` is a synthetic proxy
node because the outermost LAM isn't a heap cell.

Problem: The definition is `LAM(counter, LAM(w, IFZ(...)))`. The graph should
show the actual LAM→LAM→IFZ chain as real nodes, not skip through LAMs.

Root cause: `ctx->defs[0]` stores a TAG_LAM term but the LAM term itself is
NOT stored as `ctx->heap[something] = TAG_LAM`. The heap has the LAM's DATA
(var slot at heap[1], body at heap[2]) but not the LAM TERM. The main render
loop at `src/debug/dump.c:1290` scans `ctx->heap[h]` and only renders what
it finds there. It finds VAR SUB at heap[1], not TAG_LAM.

Fix approach: During the def-body seeding (dump.c line 620-625), explicitly
write the LAM terms from `ctx->defs[]` into the traversal so they appear as
real nodes. Or: add a special LAM rendering pass that emits LAM nodes from
`ctx->defs[]` with proper var/body edges, integrated into the main graph.

### 2. VAR substitution should be visible

When APP⊳LAM fires, `heap_set(lam_loc, arg)` immediately writes the arg
value to the VAR slot. After this, any node that reads the VAR gets the
substituted value.

Current graph behavior: VARs that appear as children of TAG_TOP/TAG_APP are
rendered directly from the heap. After substitution, the child slot still
contains the original VAR term (which resolves through the heap). The graph
shows `VAR @34` without indicating what it resolves to.

The user suggested: dashed edges from substituted VARs to their resolved
values (like REF→def links). This was attempted but created disconnected
`→tN` nodes. The standalone VAR rendering was then removed entirely.

What's needed: When a child is TAG_VAR and the VAR is substituted (not SUB),
the graph should show BOTH the VAR node AND a dashed "=" edge to the resolved
value. The RESOLVE_VAR macro exists but was reverted from inline use. It
should be used to draw supplementary dashed edges, not to replace the child.

### 3. Phase 1 stops at SEQ — the graph doesn't show full reduction

Phase 1 (structural_nf) stops at SEQ(ASSIGN(MUL...), APP(REF...)) because
ASSIGN and MUL are WNF compute ops. The step graph only visualizes phase 1.

The user asked "without consumer graph should be erased" — meaning the final
graph should show the fully reduced result (TEN), not a stuck SEQ.

Current fix: `thvm_eval` step graph mode now falls through to phase 2 via
`goto phase2` (schedule/_.c:1372). Results are correct. But phase 2
interactions (FUSE absorption, KERNEL dispatch, ASSIGN fire) are NOT
visualized step by step.

What's needed: Either extend the step graph to also visualize phase 2
interactions (FUSE, KERNEL, ASSIGN firing), or at minimum show a "phase 2
result" final graph showing the TEN outcome.

### 4. n=0 base case has unresolved ERA agents

For n=0, IFZ fires the base case and calls `thvm_spawn_detached_era` on the
succ_lam. This creates detached ERA agents that the structural_nf loop can't
fully resolve (they cycle). The step graph gets stuck iterating ERA cleanup.

Current fix: structural_nf capped at 500 iterations in step graph mode.
The final graph still shows ERA nodes connected to live computation.

Root cause: `thvm_spawn_detached_era` creates ERA terms that target nodes
in the discarded succ_lam branch. These ERA-on-LAM interactions should erase
the branch, but the structural_nf's fire_one approach can't resolve them in
the right order.

### 5. One-armed DUPs (n=0 specific)

`dup41` shows only dp1 output, no dp0. The dp0 consumer is in the discarded
succ_lam branch (erased by ERA but not fully cleaned up in phase 1).

### 6. Highlight prediction mismatches

The `predict_next_redex` function (schedule/_.c:391) doesn't perfectly match
the reducer's execution order for ERA vs DP interactions. The global heap scan
(find_next_actual line 580) finds interactions in heap-position order, which
may differ from the reducer's trampoline order.

Current tolerance: checker allows DP/ERA interleaving.

## Key files

- `src/debug/dump.c` — DOT graph renderer. Main render loop starts ~line 840.
  TAG_TOP nodes at ~940, TAG_APP at ~1067, TAG_VAR at ~1269, TAG_REF at ~1259.
  Def-body seeding at ~620. NODE_HL_ATTRS macro for node highlighting.

- `src/debug/graph.c` — Step graph coordination. `thvm_step_graph_eval_begin`
  at ~587, `after_interaction` at ~729, `finalize` at ~811.
  `highlight_from_before` at ~488. `find_next_interaction` at ~547.

- `src/schedule/_.c` — `predict_next_redex` at ~391. `structural_nf` at ~676.
  `thvm_eval` at ~1356. Phase 2 FUSE at ~1406.

- `scripts/check_step_graphs.py` — Graph invariant checker. Rules at ~248+.

- `test/test_loop_assign_simple.m` — The test program. Has assertions.

## Test commands

```bash
# Verify reduce correctness
for n in 0 1 2 3 4; do
  THVM_TRAIN_STEPS=$n bin/test_loop_assign_simple 2>/dev/null | grep final_w
done

# Generate step graphs and check
bash scripts/test_phase1_step_graphs.sh test/test_loop_assign_simple.m

# Generate step graphs manually
THVM_TRAIN_STEPS=1 THVM_STEP_GRAPH=1 THVM_STEP_GRAPH_NO_PNG=1 \
  bin/test_loop_assign_simple
# Then inspect thvm_steps/step_*.dot

# Build
cc -O0 -g -o bin/test_loop_assign_simple test/test_loop_assign_simple.m \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  -framework Accelerate
```

## Current checker results

- **n=1, n=2: PASS** (13 steps each, all highlights correct)
- **n=0: FAIL** (4 violations — DUP dp1 missing in 3 steps, ERA connected in final)
