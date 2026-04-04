# Plan: Refactor Reduce / Schedule / Realize

> **Rule**: Do NOT change this plan's architecture. If stuck on implementation, ASK the user instead of pivoting to a different approach. The three-phase reduce → schedule → reduce architecture is the target. No "single-pass" fallbacks, no flag hacks, no reverting to the old eager path.

## Architecture

```
thvm_eval(program):
  1. thvm_reduce(program)     ← Pure IC. Compute TAG_TOPs stay as TAG_TOPs.
  2. thvm_schedule(t)         ← Pure rewrite. TAG_TOPs → UOP_FUSING on heap.
  3. thvm_reduce(t)           ← UOP_FUSING fires → dispatch → TAG_TEN → ASSIGN fires.
```

### Invariants
- **TAG_TEN** = materialized tensor with real GPU buffer. Always.
- **TAG_TOP** = lazy op. Compute ops (ADD, SUM, MM, RESHAPE, ...) are constructors = WNF.
- **UOP_FUSING** = scheduled kernel node. Interaction: codegen → dispatch → TAG_TEN.
- **thvm_reduce** never dispatches GPU kernels. Only UOP_FUSING interaction dispatches.
- **thvm_schedule** never dispatches. Pure heap rewrite.

### First reduce: pure IC
The trampoline walks from root. APP/GRAD are eliminators — they fire:
- APP: standard β-reduction
- GRAD: creates backward TAG_TOPs (gradient formulas) on the heap

Compute TAG_TOPs have no eliminator demanding them → WNF. ASSIGN sees TAG_TOP src → not TAG_TEN → not active → unreduced.

Result: heap has TAG_TOP DAG (fwd + bwd) connected to unreduced ASSIGNs.

### GRAD constraint
GRAD currently walks TensorMeta `src_ids` for provenance. With pure reduce, compute TAG_TOPs never create TensorMeta. GRAD must walk TAG_TOPs on the heap instead:
- Shape: `st_get(heap_loc)` — already populated at TAG_TOP creation
- Provenance: `heap_read(ctx, term_val(y))` = arg0, `heap_read(ctx, term_val(y)+1)` = arg1
- Op type: `term_ext(y)` = UOP code
- `grad_prescan`: BFS through heap TAG_TOPs instead of TensorMeta src_ids

This is the **critical refactor** — GRAD must work without TensorMeta.

### Scheduler: pure rewrite
Walk TAG_TOP DAG on heap (from ASSIGN srcs backward). Group into fused kernels.

Fusion rule: **max 1 reduce per kernel, absorb all ew+view ops in the chain.**
- Each reduce (SUM/RMAX) + its ew/view input chain = 1 kernel
- Standalone ew chains at fusion barriers (multi-consumer) = 1 kernel
- View ops (RESHAPE, PERMUTE, EXPAND) are transparent — composed onto leaf views

For each kernel group: allocate heap for UOP_FUSING spec, write `TAG_TOP(UOP_FUSING, spec_loc)` replacing the root TAG_TOP of that group on the heap.

Plan memory: lifetime analysis on kernel schedule, greedy interval coloring for buffer reuse.

### Second reduce: fire UOP_FUSING
Trampoline walks from root again. ASSIGN demands src → walks into UOP_FUSING → interaction:
1. Read KernelSpec from heap
2. ENSURE leaf TAG_TENs (weights/inputs — already materialized)
3. Codegen → Metal dispatch → fill planned buffer
4. Create TensorMeta, return TAG_TEN

TAG_TEN propagates to ASSIGN → ASSIGN fires buf_copy → weights updated.

Inner UOP_FUSINGs fire first (leaves are TAG_TEN), outer fire after.

### JIT
```
Step 0: reduce → schedule(rewrite) → reduce(UOP_FUSING→dispatch, JIT captures)
Step 1+: JIT replays (same firing order → same dispatch sequence)
```

---

## Implementation Steps

### Step 0: Decompose UOP_MM to primitives ✓ DONE
`thvm_mm(ctx, a, b)` = RESHAPE+EXPAND+MUL+SUM+RESHAPE. All callers updated.
94.4% accuracy, 368 dispatches — identical to baseline. MPS path now dead code.

### Step 1: Port GRAD to heap-based provenance (tinygrad style)
**Files**: `src/interact/grad.c`, `src/grad/_.c`

Tinygrad's `compute_gradient` is a pure graph walk — no TensorMeta:
- Toposort UOp graph via `node.src` references
- Pattern-match op → gradient formula
- Accumulate in `dict[UOp, UOp]` keyed by node reference
- Shapes from `node.shape`, provenance from `node.src`
- `targets` set replaces `requires_grad`
- Dict accumulation replaces `grad_refs`/`grad_cache`

Port to TinyHVM:
- `y` is TAG_TOP (not TAG_TEN). `term_ext(y)` = UOP, `heap_read(loc)` = args
- Shapes from `st_get(term_val(y))` (already populated at construction)
- `requires_grad` → `targets` set (param TAG_TENs passed to thvm_grad_multi)
- `grad_refs`/`grad_cache` → hash table `Term → Term` (keyed by y Term value)
- GRAD references inputs as Terms directly (TAG_TOP or TAG_TEN), not tensor IDs

Key changes to GRAD handler:
- Line 56: `if (term_tag(y) == TAG_TEN)` → also handle TAG_TOP
- Line 80: `ctx->tensors[y_id]` → read from heap + st_get
- Line 107-108: `my->creator_op, my->src_ids` → `term_ext(y), heap_read`
- Line 112-118: input refs → just use heap_read terms directly
- Line 137: `requires_grad` → check if input Term is in targets set
- Line 89-103: grad_refs/cache → hash table lookup

**Status**: TAG_TOP GRAD path implemented. 368→71 dispatches. Correctness TBD (6% accuracy — gradient formulas likely have shape issues in sum_to_shape or missing cases). Architecture verified: GRAD walks TAG_TOPs on heap, no TensorMeta needed for provenance.

### Step 3: Scheduler fusion walk
**File**: `src/schedule/_.c`

Walk TAG_TOPs on heap from ASSIGN srcs. Build kernel groups.
Reuse `fuse_walk_inner` pattern for view composition.

Fusion rule: max 1 reduce per kernel. Everything between two kernel boundaries fuses.
Kernel boundaries: reduces (SUM/RMAX), multi-consumer nodes.

Output: array of KernelEntry with FusedOps, leaf refs, ReduceSpec.

**Test**: count kernels, compare with tinygrad (~20 target).

**Note**: Matmul is decomposed to EXPAND+MUL+SUM at graph construction (no UOP_MM).
Codegen recognizes the pattern and emits efficient code (GROUP_FOR_REDUCE, or MPS).

### Step 4: Memory planner
Port JIT planner (jit.m:128-315) to work on KernelEntry[].

### Step 5: Heap rewrite with UOP_FUSING
Write kernel specs to heap. Replace TAG_TOP subgraphs with `TAG_TOP(UOP_FUSING, spec_loc)`.

### Step 6: UOP_FUSING interaction handler
**File**: `src/interact/tensor_ops.c`

When UOP_FUSING fires: read spec → codegen → dispatch → create TensorMeta → return TAG_TEN.

**Test**: second thvm_reduce fires kernels, ASSIGNs fire, correct gradients.

### Step 7: Wire up thvm_eval, test, cleanup

---

## Current State

### After first reduce (verified via sched_dump_heap)
```
HEAP[619]: TEN=271 ERA=24 APP=7 TOP=274 other=42
  TOPs: NEG=2 EXP=2 LOG=2 RELU=8 SQRT=4 ADD=22 MUL=49 DIV=6 MAX=2
        SUB=8 SUM=22 RMAX=6 RESHAPE=68 PERMUTE=18 EXPAND=32 SHRINK=18
        ASSIGN=4 GRAD=1
```
- 274 TAG_TOPs on the heap: entire fwd+bwd compute graph (lazy)
- 271 TAG_TENs: weights, inputs, scalars (materialized)
- 1 GRAD: top-level gradient node (lazy — scheduler dispatches forward first)
- 4 ASSIGNs: BN running stat updates (gradient ASSIGNs created by GRAD later)
- `t tag=0` = TAG_APP (root is consumed APP — stale, don't re-reduce)

### What's done
- **Step 0** ✓ MM decomposed to EXPAND+MUL+SUM
- **Trampoline** ✓ Compute TAG_TOPs + GRAD are WNF (lazy). reduce/_.c:97-113
- **TAG_TOP GRAD handler** ✓ interact/grad.c:60. Handles TAG_TOP y.
- **Heap dump tool** ✓ sched_dump_heap in schedule/_.c
- **Verified**: 20 dispatches (no adam) when scheduler dispatches correctly

### What's NOT done
- The scheduler (Step 2-6) — the core missing piece

## Immediate Next Steps

The scheduler needs to:

### Step 2: Walk TAG_TOP graph, build kernel groups
- Scan the 274 TAG_TOPs on the heap
- Group into fused kernels (reuse `fuse_walk_inner` for TAG_TOP trees)
- Identify kernel boundaries: SUM/RMAX (reduces), multi-consumer nodes
- Output: array of KernelEntry with FusedOps, leaf TAG_TEN refs, ReduceSpec

### Step 3: Memory planner
- Compute lifetimes across kernel entries
- Greedy interval coloring for buffer reuse

### Step 4: Write UOP_FUSING specs + build new reducible graph
- For each kernel: allocate heap for UOP_FUSING node with kernel spec
- **Build a NEW root term** that chains:
  `APP(UOP_FUSING_0, APP(UOP_FUSING_1, ... APP(GRAD_term, ASSIGN_chain)))`
  The second reduce walks this chain — each UOP_FUSING fires (dispatch),
  GRAD fires (creates backward TAG_TOPs which get their own UOP_FUSING pass),
  ASSIGNs fire (buf_copy).
- Key: there is NO single root in an inet. The scheduler creates reducible
  entries for every output tensor. `thvm_reduce` can start from any of them.

### Step 5: UOP_FUSING interaction handler
- When UOP_FUSING fires: read spec → codegen → dispatch → TAG_TEN

### Step 6: Second reduce
- Reduce the new graph built by scheduler
- UOP_FUSINGs fire → forward materializes → GRAD fires → backward materializes → ASSIGNs fire

### Key design question for Step 4
The scheduler should produce a NEW reducible graph on the heap. Not re-reduce
the stale `t`. The new graph chains UOP_FUSING → GRAD → ASSIGN nodes.
Each UOP_FUSING has its kernel spec. GRAD references forward UOP_FUSING outputs.
ASSIGNs reference backward UOP_FUSING outputs.

The second `thvm_reduce` processes this new graph. No dispatch_mode flag needed —
UOP_FUSING is in the non-WNF set (line 103), so the trampoline fires it normally.

## Key Insight from Exploration
Scheduler DAG analysis showed:
- 2 kernel roots (ASSIGN outputs) containing 304 ops and 30 reduces
- After splitting at reduce boundaries: ~30 fused kernels + 14 adam = ~44 total
- MM decomposed to primitives: MM reduces fuse with post-MM ew → fewer kernels
- 222MB memory from scheduling alone (massive reduction from 2.1GB)

Tinygrad reference: MM is decomposed to EXPAND+MUL+REDUCE_AXIS. Post-MM ops (bias, relu) fuse into the same kernel. Each kernel has max 1 REDUCE_AXIS. Full_shape determines iteration domain; axis_types distinguish GLOBAL vs REDUCE dims.
