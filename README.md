# TinyHVM

Lazy **interaction-net** runtime for tensor programs. Computation is graph rewriting in the style of [HVM4](https://github.com/HigherOrderCO/HVM4); tensor ops and scheduling borrow operational ideas from [tinygrad](https://github.com/tinygrad/tinygrad).

The engine is a **single C translation unit** (`src/tinyhvm.c`), **no third-party runtime deps** beyond libc and the chosen backend (**Accelerate** on Apple, **Metal** on macOS, **CPU + pthread** on Linux in `build.sh`). Optional pieces: **Wolfram Language** (`./build.sh --paclet`) and **Python** (`./build.sh --python`).

---

## What you get

- **64-bit packed terms** (`TAG` / `EXT` / `VAL`) for the core calculus plus tensor nodes (`TAG_TEN`, `TAG_TOP`, …).
- **Lazy tensor UOps** (matmul, elementwise, reduce, reshape/permute/expand, conv2d, pooling, fused kernels, in-place assign, …) kept as `TAG_TOP` until **`thvm_eval`** runs schedule + dispatch (or you hit a path that realizes them inside `thvm_reduce`).
- **Pluggable backends**: CPU (Accelerate) and **Metal** with JIT-style kernel generation under `src/backend/metal/`.
- **Autograd without a tape**: backward is expressed as **interaction rules** on the same net (`thvm_grad`, multi-arg variants, “keep” bundles for training loops).
- **Fusion** and pattern materialization under `src/fuse/`.
- **Tests and benches** as small Objective-C drivers (`test/*.m`) that include `tinyhvm.c` directly.

Agent-oriented build and file map: [`AGENTS.md`](AGENTS.md).

---

## Build

**Test binaries** (macOS + Xcode CLT for Metal):

```bash
./build.sh              # all test_*.m → bin/
make test_metal         # build + run test_term on GPU
make test_train_metal   # build + run test_train on GPU
```

**CPU-only** quick loop:

```bash
make test
```

**README code examples** (forward + autograd bundle; CPU then Metal):

```bash
make readme-verify
```

**Wolfram paclet** (macOS, Wolfram Language 13+):

```bash
./build.sh --paclet
```

Then in Wolfram:

```wolfram
PacletDirectoryLoad["/absolute/path/to/TinyHVM"];
Get["TinyHVM`"]
TInit["metal"]
```

**Everything**:

```bash
./build.sh --all
```

Profiling: set `THVM_PROFILE` in the environment to enable step-level UOp timing (see `ThvmProfile` in `src/tinyhvm.h`).

---

## C API sketch

Tests are plain C inside `.m` files; they `#include "../src/tinyhvm.c"` plus backends.

### Higher-level ops today

- **`Tensor` + `Linear`** ([`src/nn/tensor_api.c`](src/nn/tensor_api.c), declarations in [`src/tinyhvm.h`](src/tinyhvm.h)) — concise C wrappers over lazy terms: `tensor_matmul` / `tensor_add` / `tensor_mul` / `tensor_relu` / `tensor_sum_axes`, plus `linear_forward`.
- **`thvm_tg_*`** ([`src/nn/tg_tensor.c`](src/nn/tg_tensor.c)) remains the lower-level tinygrad-shaped bridge (`Tensor.linear`/`dot` semantics).
- **`Layer` + `thvm_sequential`** ([`src/tinyhvm.h`](src/tinyhvm.h), [`src/nn/sequential.c`](src/nn/sequential.c)) stays available for CNN-style stacks (conv, batchnorm, maxpool, flatten, linear, custom `LayerFn`).
- **Python surface** keeps the same concise names (`tinyhvm.Tensor`, `tinyhvm.nn.Linear`) in [`py/tinyhvm/tensor.py`](py/tinyhvm/tensor.py) and [`py/tinyhvm/nn/__init__.py`](py/tinyhvm/nn/__init__.py).

Lazy tensor compute stays **inet-normal** until you **`thvm_eval`** the term you want realized (scheduler + Metal/CPU dispatch inside the TU). Then **`thvm_to_host`** reads a concrete buffer.

**Forward: `relu(linear(x, w, b))`** — same numerics as the old `relu(x@w+b)` README check (`make readme-verify`):

```c
TinyHVM *ctx = thvm_init("metal");  // or "cpu"

f32 x_d[] = {1,2,3,4,5,6}; u32 xs[] = {2,3};
f32 w_d[] = {0.1f,-0.2f, 0.3f,0.4f, -0.5f,0.6f}; u32 ws[] = {3,2};
f32 b_d[] = {-0.1f, 0.2f}; u32 bs[] = {1,2};

Tensor x = tensor_from_f32(ctx, x_d, shape_of(xs, 2));
Tensor w = tensor_from_f32(ctx, w_d, shape_of(ws, 2));
Tensor b = tensor_from_f32(ctx, b_d, shape_of(bs, 2));
Linear lin = {.weight = w, .bias = b, .has_bias = 1};

Tensor y = linear_forward(&lin, x);
Tensor result = tensor_realize(tensor_relu(y));
f32 *out = tensor_to_host_f32(result);
/* out ≈ [0, 2.6, 0, 5.0] */
thvm_free(ctx);
```

**Autograd: `sum(relu(x * w))` w.r.t. `w`** (same as `test/test_tiny_single_param_keep.m`; `make readme-verify`):

```c
TinyHVM *ctx = thvm_init("metal");

f32 xd[] = {1,2,3, 4,5,6};
f32 wd[] = {0.1f, 0.2f, 0.3f};
Tensor x = tensor_from_f32(ctx, xd, (Shape){.dims={2,3}, .rank=2});
Tensor w = tensor_from_f32(ctx, wd, (Shape){.dims={3}, .rank=1});
thvm_set_requires_grad(ctx, w.term);

Tensor h = tensor_relu(tensor_mul(x, w));
Tensor loss = tensor_sum_axes(h, (u32[]){0, 1}, 2);
Tensor bundle = tensor_realize(tensor_backward_keep(loss, w));

f32 *dw = tensor_to_host_f32(tensor_bundle_get(bundle, 0));
/* dw ≈ [5, 7, 9] */

thvm_free(ctx);
```

Inet-style `thvm_grad` / `thvm_reduce` patterns for tiny scalars still appear in `test/test_term.m` (e.g. `test_grad_x2`); full lazy backward graphs usually use **`thvm_grad_keep`** (or multi-arg variants) plus **`thvm_eval`** before readback.

Public declarations live in [`src/tinyhvm.h`](src/tinyhvm.h). `thvm_eval` is in [`src/schedule/_.c`](src/schedule/_.c); tests such as [`test/test_tiny.m`](test/test_tiny.m) call it the same way as here.

---

## Graph pictures (recent runs)

The repo keeps **Graphviz** exports from debugging runs (`THVM_STEP_GRAPH` / `THVM_GRAPH`, see `src/debug/graph.c` and `thvm_eval` in `src/schedule/_.c`). Two useful views:

1. **Step graphs** — PNGs under `graphs/<fixture>/` while the reducer walks a training-shaped graph (examples below: `tiny_linear_bias_keep`).
2. **Pipeline snapshots** — `thvm_0` … `thvm_3` under `graphs/…` for **pre-reduce → post-reduce → post-schedule → post-dispatch** (Metal fixture `tiny_linear_bias_keep_metal`).

### How step filenames relate to highlights

Each step image is written after a reduction step, and the basename now matches the red-highlighted
interaction that is actually visible in that image:
`step_NNN_NAME1_hX_NAME2_hY.png`, for example `step_000_REF_h35_LAM_h39.png` or
`step_009_ERA_h55_TEN_h38.png`. `NAME1/hX` is the highlighted node/rule focus and `NAME2/hY` is
the highlighted peer on that same red edge.

The DOT metadata stores that same visible interaction in `PREV_INTERACTION`. Once the following
step is emitted, the tracer rewrites `NEXT_INTERACTION` to the following step's basename so adjacent
steps can still be checked for consistency.

Final heap snapshots still end as `step_NNN_state_final.png`.

For loop fixtures the repository now emits two parallel step sets:

1. `n*_steps/` traces structural unfolding without firing FUSE.
2. `n*_steps_fuse_vals/` traces `FUSE(program)` propagation with tensor values so KERNEL updates stay visible.

### Reduce → schedule → dispatch (phase 2, Metal)

Lazy graph before reduction:

![Pre-reduce — phase2 tiny_linear_bias_keep_metal](graphs/tiny_linear_bias_keep_metal/thvm_0_pre_reduce.png)

After scheduling (work grouped for backends):

![Post-schedule — phase2 tiny_linear_bias_keep_metal](graphs/tiny_linear_bias_keep_metal/thvm_2_post_sched.png)

After dispatch (concrete tensor / kernel wiring):

![Post-dispatch — phase2 tiny_linear_bias_keep_metal](graphs/tiny_linear_bias_keep_metal/thvm_3_post_dispatch.png)

More graphs live under [`graphs/`](graphs/) (other fixtures: `tiny_single_param_keep`, non-`_metal` CPU runs, etc.).

---

## Where to read next

| Topic | Location |
|--------|-----------|
| Term layout, UOp codes, profiling | `src/tinyhvm.h` |
| Interaction rules | `src/interact/_.c`, `src/interact/grad.c` |
| Reducer | `src/reduce/_.c` |
| Scheduler | `src/schedule/_.c` |
| Gradients | `src/grad/_.c` |
| Metal JIT | `src/backend/metal/codegen.m` |
| MNIST-style CNN smoke test | `test/test_cnn_small.m` |
| Design notes | `resources/`, `Notebooks/` |

---

## License / attribution

TinyHVM is an independent experiment informed by the HVM interaction-calculus lineage and by tinygrad-style minimal runtimes. See upstream projects for their licenses.
