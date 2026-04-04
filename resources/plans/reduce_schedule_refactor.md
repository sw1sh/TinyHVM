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

### What's done
- **Step 0** ✓ MM decomposed to EXPAND+MUL+SUM (thvm_mm). All callers updated.
- **Trampoline** ✓ Compute TAG_TOPs treated as WNF (lazy). Code in reduce/_.c:99.
  GRAD/ASSIGN/IFZ/etc still reduce args normally.
- **Trampoline GRAD fix** ✓ Line 154: accepts TAG_TOP arg0 for GRAD (so GRAD fires
  when y is a lazy TAG_TOP).
- **TAG_TOP GRAD handler** ✓ In interact/grad.c:60. Handles all compute ops when
  y is TAG_TOP. Reads UOP from term_ext, args from heap_read, shapes from st_get.
  Uses BG/UG macros for binary/unary gradient formulas.
- **dispatch_mode flag** ✓ In tinyhvm.h:612, reduce/_.c:100. When set, trampoline
  processes compute TAG_TOPs normally (not lazy). Used by scheduler to dispatch.

### What's NOT working (blockers for Step 2-6)
- **Gradient ADD is WNF**: GRAD handler's BG macro returns `thvm_op(UOP_ADD, grad_a, grad_b)`.
  ADD is a compute TAG_TOP → WNF → trampoline never enters it → grad_b never fires.
  The `dispatch_mode=1` set by GRAD_RETURN contaminates forward TAG_TOPs too.
  Need a way for gradient expressions to force evaluation without affecting forward laziness.
- **Second reduce on stale t**: After first reduce, `t` points to consumed heap.
  Fix: scheduler finds ASSIGNs on heap and reduces each individually (not re-reducing t).
- **ASSIGN finding**: Only 4 BN assigns found via heap walk. Gradient ASSIGNs created
  by GRAD inside consumed APP chains not reachable from t. Heap scan needed.

### Experimental code in repo (not reverted)
- `src/reduce/_.c`: lazy TAG_TOP check (line 99), GRAD TAG_TOP arg0 fix (line 154)
- `src/interact/grad.c`: TAG_TOP GRAD handler (line 60), dispatch_mode in GRAD_RETURN (line 16)
- `src/interact/tensor_ops.c`: defer_all check (line 269), ASSIGN guard (line 14)
- `src/tinyhvm.h`: defer_all, dispatch_mode fields
- `src/schedule/_.c`: reverted to passthrough
- `test/test_beautiful_mnist.m`: uses thvm_reduce (not thvm_eval)

### Verified numbers (from earlier experiments)
- With lazy forward + scheduler dispatch: **20 dispatches** (no adam), **34 total**
- Memory: **631MB** (vs 2121MB baseline)
- Tinygrad: 73 kernels for same model
- Default path: 94.4% correct, 368 dispatches (unchanged)

## Immediate Next Steps

1. **Fix gradient ADD WNF problem** (Step 1 blocker):
   Gradient expressions created by GRAD must reduce, but they're compute TAG_TOPs (WNF).
   Options: (a) use a non-WNF combinator for gradient combining, (b) have GRAD reduce
   gradient expressions internally before returning, (c) structural tagging to distinguish
   gradient TAG_TOPs from forward TAG_TOPs. **Ask user which approach.**

2. **Fix ASSIGN finding** (Step 2 blocker):
   Gradient ASSIGNs are inside consumed APP chains. Need heap scan or track them during GRAD.

3. **Fix second reduce target** (Step 3):
   Don't re-reduce stale `t`. Reduce each found ASSIGN individually with dispatch_mode=1.

## Key Insight from Exploration
Scheduler DAG analysis showed:
- 2 kernel roots (ASSIGN outputs) containing 304 ops and 30 reduces
- After splitting at reduce boundaries: ~30 fused kernels + 14 adam = ~44 total
- MM decomposed to primitives: MM reduces fuse with post-MM ew → fewer kernels
- 222MB memory from scheduling alone (massive reduction from 2.1GB)

Tinygrad reference: MM is decomposed to EXPAND+MUL+REDUCE_AXIS. Post-MM ops (bias, relu) fuse into the same kernel. Each kernel has max 1 REDUCE_AXIS. Full_shape determines iteration domain; axis_types distinguish GLOBAL vs REDUCE dims.
