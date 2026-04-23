# Evaluation Pipeline

TinyHVM now has two architectural layers during `thvm_eval`:

1. A **local IC phase** that reduces the program and coarse-grains lazy tensor
   structure into visible `KERNEL` nodes.
2. A **global compiler phase** that can rewrite the settled coarse graph and
   optionally install executable triggers before the final strict reduce.

This matches the split we want from tinygrad-like systems:

- local graph rewriting chooses and grows kernel regions
- global passes reason about the settled kernel DAG
- lowering and dispatch stay downstream runtime details

For the debug and runtime knobs that control these stages, see `docs/env.md`.

## Live Pipeline

```text
thvm_eval(ctx, t):
  local phase
    1. structural reduction         -> lazy tensor frontier
    2. local coarse-graining        -> growing KERNEL -> settled KERNEL

  global phase
    3. named compiler passes        -> rewrite settled coarse IC / install EXEC
    4. second reduce                -> fire EXEC triggers if any
```

In code this is the `thvm_eval_reduce_fused()` + `thvm_run_global_passes()` +
optional second `thvm_reduce()` sequence in `src/schedule/_.c`.

## Local Phase

The local phase is still entirely IC-native.

### Phase 1: Structural Reduction

`thvm_reduce()` performs ordinary interaction-calculus reduction until the
program reaches a lazy tensor frontier. Tensor compute remains represented as
`TAG_TOP` nodes such as `MUL`, `ADD`, `SUM`, `RESHAPE`, and `PAD`.

This phase handles the structural rewrites:

- `APP/LAM`
- `IFZ`
- `GRAD`
- `MAT/CTR`
- `DUP` / `ERA`
- strict sequencing such as `SEQ`

### Phase 2: Local Coarse-Graining

The reduced root is wrapped in `FUSE(...)`, and the reducer runs again.

`FUSE` is only the **propagating pressure**. It does not dispatch kernels by
itself. Instead it turns fuseable compute into visible structural `KERNEL`
nodes:

```text
FUSE(MUL(a,b)) -> KERNEL(FUSE(a), FUSE(b), NUM(MUL))
```

There are now two meaningful public `KERNEL` states:

- **Growing `KERNEL`**: local coarse region still absorbing children.
- **Settled `KERNEL`**: monolithic public region whose payload is the full
  compute subgraph that will lower as one kernel bundle.

The key invariant is:

- child `KERNEL`s can settle locally under a parent
- a settled child is treated as absorbable WNF by the parent kernel
- only the topmost settled kernel is allowed to register/lower/dispatch

That is what keeps the public step graph incremental while the settled replay
still lowers only one final monolithic kernel.

## Global Phase

Once local coarse-graining has produced the settled coarse graph, TinyHVM runs
the global compiler-pass layer.

### Phase 3: Named Compiler Passes

`thvm_run_global_passes()` walks the settled coarse IC and applies any passes
registered via `thvm_register_pass()`.

This is where future work belongs for:

- explicit kernel DAG rewrites
- dependency analysis
- `EXEC` trigger installation
- planner-style global transformations

Today the default runtime often has `registered_passes=0`, but the phase is
real and already dumped by `THVM_GRAPH`.

### Phase 4: Second Reduce

If global passes install `UOP_EXEC` triggers, TinyHVM runs one more strict
reduce to fire them.

That keeps the layering clean:

- local phase chooses and settles public `KERNEL` structure
- global phase rewrites the settled graph
- second reduce executes whatever global passes made explicit

## From Public KERNEL to Backend Program

The runtime lowering path is:

```text
public KERNEL
  -> KernelEntry          (fused ops + leaves + views + reduce spec)
  -> LOP DAG              (private lowering IR in lower/_.c)
  -> UOpKernel / KOP list (linear backend program)
  -> backend dispatch
```

Each layer has a different job:

- `KERNEL`: public coarse IR visible in the heap and graph dumps
- `KernelEntry`: runtime description of one fused kernel region
- `LOP`: normalization-friendly private lowering DAG
- `KOP`: emitted linear kernel VM for CPU/Metal backends

## KOPs

`KOP`s live in `src/tinyhvm.h` as the emitted `UOpKernel` program.

| KOP | Meaning |
|-----|---------|
| `KOP_NOOP` | Reserved empty instruction |
| `KOP_GID` | Read one grid axis (`x/y/z`) |
| `KOP_CONST_F` | Float literal |
| `KOP_CONST_U` | Unsigned integer literal |
| `KOP_RANGE` | Begin a reduction loop with a fixed trip count |
| `KOP_ENDRANGE` | Close the range opened by `KOP_RANGE` |
| `KOP_LOAD` | Read one kernel input buffer at a computed index |
| `KOP_STORE` | Write one output buffer slot |
| `KOP_ALU` | Scalar math op parameterized by a `UOP_*` opcode |
| `KOP_ACC_INIT` | Initialize a reduction accumulator |
| `KOP_ACC` | Combine a value into an accumulator (`SUM`/`RMAX`) |
| `KOP_IDX` | Affine index arithmetic (`a * scale + b`) |
| `KOP_MOD` | Integer modulo |
| `KOP_DIV` | Integer division |
| `KOP_MASK` | Conditional mask (`cond ? value : 0`) |

The important point is that `KOP`s are not the user-visible IR. They are the
final backend program that CPU and Metal execute.

## LOPs

`LOP`s live in `src/lower/_.c` as the private lowering DAG used before linear
emission.

| LOP | Meaning |
|-----|---------|
| `LSINK` | Root/sink node for the lowered program |
| `LGID` | Grid-axis source in the lowering graph |
| `LCONST_U` | Unsigned integer literal |
| `LCONST_F` | Float literal |
| `LRANGE` | Reduction loop start |
| `LENDRANGE` | Reduction loop end |
| `LLOAD` | Input-buffer load |
| `LSTORE` | Output-buffer store |
| `LALU` | Scalar arithmetic node |
| `LACC_INIT` | Reduction accumulator initialization |
| `LACC` | Reduction accumulator update |
| `LINDEX` | Affine index node |
| `LMOD` | Integer modulo node |
| `LDIV` | Integer division node |
| `LMASK` | Conditional masking node |

`LOP` exists so TinyHVM can normalize, fold constants, simplify indexes, and
only then emit the final linear `KOP` stream.

## Observability

TinyHVM exposes the two layers with different debug modes:

### Local Phase Dumps

Use `THVM_STEP_GRAPH=1`.  By default the two-phase session dumps
both the GRAD-commute steps and the FUSE-kernelisation steps; pass
`THVM_STEP_GRAPH_NO_FUSE=1` to stop after phase 1.

This shows:

1. structural local rewrites (GRAD, APP/LAM, IFZ, MAT/CTR, DUP)
2. `FUSE` propagation
3. growing public `KERNEL` nodes
4. the final settled monolithic `KERNEL`
5. only then downstream forcing such as `ASSIGN`

### Global Phase Dumps

Use `THVM_GRAPH=1`.

This emits:

- `thvm_0_pre_reduce.dot`
- `thvm_1_post_reduce.dot`
- `thvm_2_post_dispatch.dot`
- `thvm_3_post_passes.dot`
- `thvm_4_post_exec.dot`
- `thvm_global_passes.txt`

Those dumps are the right place to inspect the whole coarse pipeline rather
than the local interaction-by-interaction trace. In debug-only runs, the dumped
root can still be coarse IR; the loop harness validates the final buffer state
separately from the displayed coarse root.

## Simple Loop Example

Run the end-to-end local+global harness with:

```bash
bash scripts/test_loop_local_global_eval.sh
```

The wrapper reads `THVM_GLOBAL_TRAIN_STEPS` (default `1`) for the
number of loop iterations in the global-pass dump; internally it
passes that value as `THVM_TRAIN_STEPS` to the binary.  A value of `1`
keeps the coarse graph artifact and the observable buffer update
aligned in one minimal loop iteration.

That script:

1. runs the local step-graph regression on the archived
   `test/archive/test_loop_assign_simple.m`
2. checks the fused local trace contract
3. checks kernel redispatch on repeated loop iterations
4. runs a global `THVM_GRAPH` dump for the same loop example

Artifacts land under:

- local: `graphs/loop_assign_simple/n*_steps*`
- global: `graphs/loop_assign_simple/global_eval/`

## What Is Still Missing for Tinygrad-Style Architecture

TinyHVM already has the right lazy graph shape, explicit public kernel
boundaries, and ShapeTracker-style movement ops. The remaining gaps are mostly
global compiler/runtime architecture, not more local rewrite tricks.

### 1. Real global pass content

The pass hook exists, but the default runtime still does little with it.
To cover more of tinygrad's architecture, this layer needs actual kernel-DAG
rewrites rather than just the empty pass dump.

### 2. Dependency DAG around side effects

`ASSIGN` makes ordering observable. A tinygrad-like global planner still needs
explicit read/write dependency analysis over the settled kernel DAG.

See `docs/dependencies.md`.

### 3. Memory planning and slot reuse

Tinygrad-style execution relies on a planner that separates logical values from
physical buffers. TinyHVM still needs a proper buffer-lifetime planner and
reuse policy over the kernel DAG.

See `docs/memory.md`.

### 4. Schedule search / rangeify / backend tuning

Tinygrad does more work in the scheduler when choosing the final executable
kernel shape. TinyHVM currently lowers one settled public kernel directly,
without a richer search/tuning layer over alternative schedules.

### 5. Broader global lowering contracts

The current `LOP -> KOP` path covers the scalar/index/load/store/reduce core.
If TinyHVM grows more aggressive backend optimization, that will likely show up
as richer global passes first, and only secondarily as new lowering opcodes.

## Key Files

- `src/schedule/_.c` — `thvm_eval`, global passes, graph dump orchestration
- `src/wnf/_.c` — wnf reducer trampoline (`thvm_reduce`, `thvm_reduce_budget`)
- `src/reduce/_.c` — kernel-readiness predicates used by FUSE / UOP_KERNEL
- `src/parallel/normalize.c` — `thvm_normalize`, root-reachable WHNF walker
- `src/interact/_.c` — interaction dispatcher and public-kernel helpers
- `src/interact/tensor_ops.c` — `FUSE`, `KERNEL`, `EXEC`, and dispatch behavior
- `src/lower/_.c` — private lowering IR (`LOP`) and `UOpKernel` emission
- `src/tinyhvm.h` — `KOP`, `UOpKernel`, and `KernelEntry` definitions
