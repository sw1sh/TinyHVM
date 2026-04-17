# Environment Variables

TinyHVM uses a mix of runtime, graphing, backend, and test-harness environment
variables.

Unless noted otherwise:

- presence enables a flag
- absence leaves the feature off
- path variables use the string value directly
- integer variables are parsed with `strtoul` / `atoi`

Some helpers explicitly treat `"0"` as false; those are called out below.

## Evaluation And Test Harness

| Variable | Type / default | Effect |
|---|---|---|
| `THVM_TEST_BACKEND` | string, test-specific | Selects backend in test binaries that honor it, typically `cpu` or `metal`. |
| `THVM_TRAIN_STEPS` | integer | Overrides the loop/training iteration count in tests that expose a step count. |
| `THVM_ALLOW_NON_TEN_RESULT` | bool, `"0"` disables | Test-only escape hatch in `test_loop_assign_simple.m` to allow a non-tensor final result without treating it as failure. |
| `THVM_EVAL_MIXED_DISPATCH` | bool, `"0"` disables | Re-enables the legacy eval path where the phase-2 `FUSE` pass may dispatch kernels immediately instead of stopping at a clean post-local coarse graph. |
| `THVM_NO_LOWER` | bool | Skips lowering in the scheduler’s dispatch path. Useful when you want to inspect the coarse graph without invoking the lowerer. |
| `THVM_PROFILE` | bool | Enables built-in runtime profiling counters and phase timing. Metal profile summaries use this. |

## Coarse Graph And Step Graph Dumps

| Variable | Type / default | Effect |
|---|---|---|
| `THVM_GRAPH` | bool | Emits coarse pipeline graphs for `thvm_eval`, such as `thvm_0_pre_reduce.dot` through `thvm_4_post_exec.dot`. |
| `THVM_GRAPH_DIR` | path, default `graphs` | Output directory for coarse graph dumps. |
| `THVM_GRAPH_TENSOR_VALUES` | bool | Includes literal tensor values in coarse graph node labels. |
| `THVM_STEP_GRAPH` | bool | Enables per-interaction step tracing and emits `step_*.dot` / `step_*.png`. |
| `THVM_STEP_GRAPH_DIR` | path, default `graphs` | Output directory for step-graph dumps. |
| `THVM_STEP_GRAPH_MAX` | integer, default `512` | Maximum number of step snapshots written in one step-graph session. |
| `THVM_STEP_GRAPH_DIAG` | bool | Prints step-graph chooser/debug diagnostics to stderr. |
| `THVM_STEP_GRAPH_NO_PNG` | bool | Writes `.dot` files only; skips `dot -Tpng` rendering for step graphs. Lower-graph PNG generation also respects this during step-trace runs. |
| `THVM_STEP_GRAPH_TENSOR_VALUES` | bool | Includes literal tensor values in step-graph node labels. |
| `THVM_HEAP_DOT_RAW_ONLY` | bool, `"0"` disables | Forces the heap renderer into raw-only mode and disables semantic + step overlays. |
| `THVM_HEAP_DOT_SEMANTIC` | bool, default `1`, `"0"` disables | Controls semantic overlay rendering for heap/graph dumps when raw-only mode is off. |
| `THVM_HEAP_DOT_STEP` | bool, default `1`, `"0"` disables | Controls step-specific overlay rendering when raw-only mode is off. |

## Lowering Graph Dumps

| Variable | Type / default | Effect |
|---|---|---|
| `THVM_LOWER_GRAPH` | bool, `"0"` disables | Emits lowered kernel graphs / manifests during lowering. |
| `THVM_LOWER_GRAPH_DIR` | path | Output directory for lowering dumps. If unset, lowering defaults to `graphs/lower`, or to a step-graph-adjacent directory during step tracing. |
| `THVM_LOWER_NO_PNG` | bool, `"0"` disables | Skips PNG rendering for lowering graph dumps. |

## Diagnostics And Tracing

| Variable | Type / default | Effect |
|---|---|---|
| `THVM_SCHED_DIAG` | bool | Scheduler, dispatch, and kernel-readiness diagnostics. This is the main coarse scheduling debug flag. |
| `THVM_SCHED_DIAG2` | bool | Extra fusion-walk diagnostics on top of `THVM_SCHED_DIAG`. |
| `THVM_LOOP_DIAG` | bool | Focused diagnostics for recursive loops, `ALO`, `GRAD`, keep-bundle handling, and related net rewrites. |
| `THVM_LOWER_DIAG` | bool | Prints lowering summaries, including whether a kernel was lowered or left alone. |
| `THVM_LOWER_TRACE` | bool | Per-kernel lowering trace output. |
| `THVM_KERN_DIAG` | bool | Additional kernel diagnostics in fusion/lowering paths, especially for reduction kernels. |
| `THVM_DIAG` | bool | General fusion/materialization diagnostics, for example when fusion hits op/leaf limits. |
| `THVM_TRACE` | bool | Verbose fusion/materialization traversal trace. |
| `THVM_DUMP_CODEGEN` | bool | Dumps generated Metal shader source / codegen output to stderr. |
| `THVM_DEBUG` | integer, default `0` | Metal debug/profiling knob. `>=2` enables per-kernel timing output in the profiled dispatch wrapper. |

## Metal Backend

| Variable | Type / default | Effect |
|---|---|---|
| `THVM_METALLIB` | path | Overrides the path to `shaders.metallib` during Metal backend initialization. |
| `THVM_NO_UOP` | bool | Intended to disable the UOp IR Metal codegen path. At the moment the relevant branch in `codegen.m` is hard-disabled, so this flag is effectively dormant. |
| `UNIFIED` | bool | Metal JIT allocator debug mode: allocate one unified heap buffer and alias sub-buffers via offsets. |
| `NO_PLAN` | bool | Metal JIT allocator debug mode: bypass the plan-based allocator and give each transient slot its own buffer. |

## Notes

- `THVM_STEP_GRAPH`, `THVM_GRAPH`, and `THVM_LOWER_GRAPH` are independent:
  they dump different layers of the pipeline.
- `THVM_EVAL_MIXED_DISPATCH=1` is a legacy debugging mode, not the intended
  default staged pipeline.
- Most shell scripts in `scripts/` only compose the variables above; they do
  not introduce a separate second set of env vars.
