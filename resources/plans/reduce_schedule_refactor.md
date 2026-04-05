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

## Current State (2026-04-05)

### What works
- **Pure IC first reduce** ✓ no_fuse=1 → ALL compute+view ops WNF
- **Graph walk** ✓ Unified forward scan discovers and dispatches 25 kernels at 12GB budget
- **fuse_or_reduce** ✓ Absorbs ew chains into reduces, DP0 look-through, min_ops relaxed
- **fuse_walk_inner** ✓ Walks VIEW(ew_TAG_TOP) for EXPAND/PERMUTE/RESHAPE
- **View resolution** ✓ dispatch_mode + defer_all fires views without compute side-effects
- **GRAD** ✓ Fires on TAG_TOP args, creates backward TAG_TOPs
- **Buffer steal** ✓ dispatch_counter + buf_last_use in codegen.m

### Why hybrid dispatch hit a wall
The current approach dispatches directly from the scheduler via fuse_or_reduce.
This MIXES planning with execution:
- View resolution (thvm_reduce) triggers contiguify → large GPU allocations
- ENSURE inside fuse_or_reduce materializes deferred chains recursively
- Flag gymnastics (no_fuse/dispatch_mode/defer_all) create fragile interactions
- Forward chain stalls: ew→view→ew interleaving needs many passes, but the
  contiguify allocations accumulate without mid-step freeing

With 24GB budget: 205 dispatches work (91 fwd + 114 bwd), but view_expand
rank mismatch in backward. With 12GB: 25 dispatches, then stalls (view chains
need more passes that don't make progress due to ordering).

### Key lesson
**Cannot dispatch during scheduling.** The graph walk and kernel grouping work
correctly. The memory management and dispatch sequencing do not. This confirms
the plan's architecture: separate schedule (pure rewrite) from dispatch.

### BLOCKER: need to separate plan from dispatch
Two options (same as before, now informed by experience):

**Option A: Two-pass scheduler (simpler)**
1. Pass 1: walk heap with fuse_or_reduce in "dry run" mode → collect KernelEntry[]
   (output size, leaf IDs, reduce spec). NO dispatch, NO allocation.
2. Memory planner: greedy interval coloring on KernelEntry[] lifetimes
3. Pass 2: dispatch each KernelEntry with pre-assigned buffer from plan

**Option B: UOP_FUSING (per original architecture)**
1. Walk heap → group into fused kernels → write UOP_FUSING specs to heap
2. Memory plan on UOP_FUSING specs
3. Second thvm_reduce fires UOP_FUSING interactions → dispatch with planned buffers

Option A reuses the current fuse_or_reduce infrastructure (proven to work).
Option B is cleaner but requires new UOP_FUSING interaction handler.

## Immediate Next Step: Memory Planner

The scheduler currently dispatches directly via fuse_or_reduce. Each dispatch
allocates a new buffer. The memory planner must run BEFORE dispatch to assign
buffer slots with reuse.

### Option A: Pre-dispatch memory planning (simplest, within current architecture)
1. **Collect dispatch list**: before dispatching, do a "dry run" that collects
   KernelEntry[] (same as current fuse_or_reduce walk, but don't dispatch)
2. **Lifetime analysis**: kernel i's output is consumed by kernel j → lifetime [i, j]
3. **Greedy interval coloring**: assign buffer slots, reuse dead buffers
4. **Dispatch with planned buffers**: fuse_or_reduce modified to use pre-allocated buffers

### Option B: UOP_FUSING rewrite (per original plan)
1. Collect kernel groups (same dry run)
2. Plan memory
3. Write UOP_FUSING specs to heap (pure rewrite, no dispatch)
4. Second thvm_reduce fires UOP_FUSING interactions → dispatch with planned buffers

Option B is cleaner (matches three-phase architecture) but more work. Option A
is a pragmatic intermediate that unblocks training.

### Key: tinygrad's memory planner
Port from JIT planner (`jit.m:128-315`). The planner:
- Tracks buffer lifetimes across kernel schedule
- Greedy interval coloring: for each new buffer, reuse the smallest dead buffer
  that fits, or allocate new
- Result: 58 unique buffers → 1 reused buffer pool
- 2.1GB → 0.82GB at BS=512
